import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import (
    Crc32CheckOperation,
    Crc32PacketizeOperation,
    _crc32_bits,
    _u32_to_bits,
)
from noema_lab.ops.metrics.bits import BitErrorRateOperation


def _operation_context(root, step_id, params, inputs):
    step_dir = root / step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    return OperationContext(
        recipe_name="packet_contract_v4_test",
        step_id=step_id,
        params=params,
        inputs=inputs,
        run_dir=root,
        step_dir=step_dir,
    )


def _bits_artifact(root, name, rows):
    bits = np.concatenate(rows).astype(np.uint8, copy=False)
    item_counts = [int(row.size) for row in rows]
    metadata = {
        "bit_count": int(bits.size),
        "payload_bit_count": int(bits.size),
        "source_item_count": len(item_counts),
        "source_item_payload_bit_counts": item_counts,
    }
    path = root / (name + ".npz")
    np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
    return artifact("channel.payload_bits.numpy", path, metadata)


def _load_bits_and_metadata(packetized_result):
    with np.load(packetized_result.outputs["bits"].path, allow_pickle=False) as payload:
        return payload["bits"].copy(), json.loads(str(payload["metadata_json"]))


def _corrupted_artifact(root, name, bits, metadata):
    path = root / (name + ".npz")
    np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
    return artifact("channel.payload_bits.numpy", path, metadata)


class PacketContractV4Tests(unittest.TestCase):
    def _packetize(self, root, rows, packet_payload_bits=64):
        source = _bits_artifact(root, "source", rows)
        return Crc32PacketizeOperation().run(
            _operation_context(
                root,
                "packetize",
                {"packet_payload_bits": packet_payload_bits},
                {"bits": source},
            )
        )

    def _check(self, root, name, packet_artifact, failure_policy):
        return Crc32CheckOperation().run(
            _operation_context(
                root,
                name,
                {"on_decode_failure": failure_policy},
                {"bits": packet_artifact},
            )
        )

    def test_all_packet_headers_destroyed_returns_full_zero_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = (np.arange(130, dtype=np.uint8) % 2).astype(np.uint8)
            second = np.ones((75,), dtype=np.uint8)
            packetized = self._packetize(root, [first, second])
            bits, metadata = _load_bits_and_metadata(packetized)

            packet_total_bits = int(metadata["packet_total_bits"])
            self.assertEqual(
                [
                    row["packet_bit_offset"]
                    for row in metadata["packet_contract"][
                        "ordered_packet_layout"
                    ]
                ],
                [
                    packet_index * packet_total_bits
                    for packet_index in range(int(metadata["packet_count"]))
                ],
            )
            for packet_index in range(int(metadata["packet_count"])):
                # One detected error in every compact header makes every
                # in-band header unusable while leaving framing intact.
                bits[packet_index * packet_total_bits] ^= 1
            corrupted = _corrupted_artifact(
                root, "all_headers_destroyed", bits, metadata
            )

            checked = self._check(root, "check_all_headers", corrupted, "gray_image")
            with np.load(checked.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            self.assertEqual(int(recovered.size), int(first.size + second.size))
            np.testing.assert_array_equal(recovered, np.zeros_like(recovered))
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_payload_bit_counts"],
                [int(first.size), int(second.size)],
            )
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_outage"], [1, 1]
            )
            self.assertEqual(
                checked.metrics["channel.crc_failed_packet_count"],
                int(metadata["packet_count"]),
            )
            self.assertEqual(
                checked.metrics["channel.unrecoverable_header_packet_count"],
                int(metadata["packet_count"]),
            )

    def test_one_destroyed_header_zeros_only_its_full_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = (np.arange(96, dtype=np.uint8) % 2).astype(np.uint8)
            second = np.ones((130,), dtype=np.uint8)
            packetized = self._packetize(root, [first, second])
            bits, metadata = _load_bits_and_metadata(packetized)
            second_first_packet = int(metadata["packet_source_item_counts"][0])
            second_packet_start = (
                second_first_packet * int(metadata["packet_total_bits"])
            )
            bits[second_packet_start] ^= 1
            received_payload_bit = (
                second_packet_start
                + int(metadata["packet_header_bits"])
                + int(metadata["packet_header_crc_bits"])
                + 3
            )
            bits[received_payload_bit] ^= 1
            corrupted = _corrupted_artifact(
                root, "one_header_destroyed", bits, metadata
            )

            checked = self._check(root, "check_gray_item", corrupted, "erasure")
            with np.load(checked.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            self.assertEqual(int(recovered.size), int(first.size + second.size))
            np.testing.assert_array_equal(recovered[: first.size], first)
            np.testing.assert_array_equal(
                recovered[first.size :], np.zeros_like(second)
            )
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_outage"], [0, 1]
            )
            self.assertEqual(
                checked.outputs["bits"].metadata["failed_item_bit_policy"],
                "zero_entire_item",
            )
            self.assertEqual(
                checked.outputs["bits"].metadata[
                    "source_item_packet_fail_counts"
                ],
                [0, 1],
            )

    def test_report_outage_is_best_effort_but_exact_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = np.ones((96,), dtype=np.uint8)
            second = np.ones((130,), dtype=np.uint8)
            packetized = self._packetize(root, [first, second])
            bits, metadata = _load_bits_and_metadata(packetized)
            second_first_packet = int(metadata["packet_source_item_counts"][0])
            second_packet_start = (
                second_first_packet * int(metadata["packet_total_bits"])
            )
            bits[second_packet_start] ^= 1
            received_payload_bit = (
                second_packet_start
                + int(metadata["packet_header_bits"])
                + int(metadata["packet_header_crc_bits"])
                + 3
            )
            bits[received_payload_bit] ^= 1
            corrupted = _corrupted_artifact(
                root, "best_effort_header_destroyed", bits, metadata
            )

            checked = self._check(
                root, "check_report_outage", corrupted, "report_outage"
            )
            with np.load(checked.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            expected_second = second.copy()
            expected_second[3] ^= 1
            expected = np.concatenate([first, expected_second])
            self.assertEqual(int(recovered.size), int(expected.size))
            np.testing.assert_array_equal(recovered, expected)
            self.assertEqual(
                checked.outputs["bits"].metadata["source_item_outage"], [0, 1]
            )
            self.assertEqual(
                checked.outputs["bits"].metadata["failed_item_bit_policy"],
                "best_effort_received_payload",
            )

    def test_packet_contract_hash_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packetized = self._packetize(
                root, [np.ones((65,), dtype=np.uint8)]
            )
            bits, metadata = _load_bits_and_metadata(packetized)
            metadata["packet_contract"]["source_item_payload_bit_counts"][0] = 64
            tampered = _corrupted_artifact(root, "tampered_contract", bits, metadata)
            with self.assertRaisesRegex(OperationError, "contract hash"):
                self._check(root, "check_tampered_contract", tampered, "gray_image")

    def test_recoverable_header_that_contradicts_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = np.ones((65,), dtype=np.uint8)
            packetized = self._packetize(root, [source])
            bits, metadata = _load_bits_and_metadata(packetized)
            header_bits = int(metadata["packet_header_bits"])
            header_crc_bits = int(metadata["packet_header_crc_bits"])
            packet_payload_bits = int(metadata["packet_payload_bits"])
            packet_total_bits = int(metadata["packet_total_bits"])

            # Change the first packet's declared source index, then recompute
            # both CRCs. The in-band header is recoverable and self-consistent,
            # but must still lose to the immutable ordered contract.
            bits[31] ^= 1
            header = bits[:header_bits]
            header_crc = _u32_to_bits(_crc32_bits(header, int(header.size)))
            bits[header_bits : header_bits + header_crc_bits] = header_crc
            protected_end = header_bits + header_crc_bits + packet_payload_bits
            protected = bits[:protected_end]
            bits[protected_end:packet_total_bits] = _u32_to_bits(
                _crc32_bits(protected, int(protected.size))
            )
            corrupted = _corrupted_artifact(
                root, "contract_mismatched_header", bits, metadata
            )

            checked = self._check(
                root, "check_contract_mismatched_header", corrupted, "erasure"
            )
            with np.load(checked.outputs["bits"].path, allow_pickle=False) as payload:
                recovered = payload["bits"]
            np.testing.assert_array_equal(recovered, np.zeros_like(source))
            self.assertEqual(
                checked.metrics[
                    "channel.header_contract_mismatch_packet_count"
                ],
                1,
            )
            self.assertEqual(
                checked.metrics["channel.unrecoverable_header_packet_count"], 0
            )

    def test_ber_rejects_unequal_bit_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _bits_artifact(
                root, "reference", [np.zeros((17,), dtype=np.uint8)]
            )
            candidate = _bits_artifact(
                root, "candidate", [np.zeros((16,), dtype=np.uint8)]
            )
            with self.assertRaisesRegex(OperationError, "equal bit counts"):
                BitErrorRateOperation().run(
                    _operation_context(
                        root,
                        "ber",
                        {},
                        {"reference": reference, "candidate": candidate},
                    )
                )


if __name__ == "__main__":
    unittest.main()
