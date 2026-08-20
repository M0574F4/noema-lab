from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops import build_registry
from noema_lab.ops.channel.nr_ldpc import _profile_sha256
from noema_lab.ops.resource_reliability import NrLdpcOfdmDeliveryMetricsOperation


class NrLdpcOfdmDeliveryMetricsTests(unittest.TestCase):
    def _inputs(
        self,
        root: Path,
        *,
        statuses: list[bool] | None = None,
        reference_bits: np.ndarray | None = None,
        decoded_bits: np.ndarray | None = None,
        blocks: list[dict] | None = None,
        power: np.ndarray | None = None,
        tx_symbols: np.ndarray | None = None,
        metadata_overrides: dict[str, dict] | None = None,
        report_overrides: dict | None = None,
        include_rx: bool = True,
    ) -> dict[str, Artifact]:
        root.mkdir(parents=True, exist_ok=True)
        statuses = list(statuses if statuses is not None else [True, False])
        reference_bits = np.asarray(
            reference_bits
            if reference_bits is not None
            else [1, 0, 1, 0, 0, 0, 0, 0],
            dtype=np.uint8,
        )
        decoded_bits = np.asarray(
            decoded_bits
            if decoded_bits is not None
            else [1, 0, 1, 0, 0, 0, 0, 0],
            dtype=np.uint8,
        )
        blocks = deepcopy(
            blocks
            if blocks is not None
            else [
                {
                    "source_item_index": 0,
                    "valid_payload_bit_count": 4,
                    "encoder_input_bit_count": 4,
                    "tail_zero_padding_bit_count": 0,
                    "coded_offset": 0,
                    "num_coded_bits": 8,
                    "target_coderate": 0.5,
                    "num_bits_per_symbol": 2,
                    "num_layers": 1,
                    "rate_matched_codeword_lengths": [8],
                },
                {
                    "source_item_index": 0,
                    "valid_payload_bit_count": 4,
                    "encoder_input_bit_count": 4,
                    "tail_zero_padding_bit_count": 0,
                    "coded_offset": 8,
                    "num_coded_bits": 8,
                    "target_coderate": 0.5,
                    "num_bits_per_symbol": 2,
                    "num_layers": 1,
                    "rate_matched_codeword_lengths": [8],
                },
            ]
        )
        valid_payload_count = sum(
            int(block["valid_payload_bit_count"]) for block in blocks
        )
        coded_bit_count = sum(int(block["num_coded_bits"]) for block in blocks)
        backend_versions = {"sionna": "test", "torch": "test"}
        decoder_iterations = 20
        profile_sha256 = _profile_sha256(
            blocks, decoder_iterations, backend_versions
        )
        binding = {
            "nr_transport_blocks": blocks,
            "nr_transport_block_count": len(blocks),
            "nr_profile_sha256": profile_sha256,
            "nr_decoder_num_bp_iter": decoder_iterations,
            "nr_backend_versions": backend_versions,
            "channel_code_input_bit_count": valid_payload_count,
            "channel_code_output_bit_count": coded_bit_count,
        }
        common = {
            **binding,
            "noise_variance": 0.5,
            "average_power_budget": 1.0,
            "channel_state_seed": 71,
        }
        tx_metadata = {
            **common,
            "bit_count": coded_bit_count,
            "coded_bit_count": coded_bit_count,
            "padded_bit_count": coded_bit_count,
            "modulation": "qpsk",
            "bits_per_symbol": 2,
            "symbol_count": coded_bit_count // 2,
            "transport_mode": "fixed_modulation",
            "power_allocator_selected_power": 1.0,
        }
        decoded_metadata = {
            **common,
            "bit_count": int(decoded_bits.size),
            "nr_transport_block_crc_status": statuses,
        }
        actual_metadata = {
            **common,
            "channel_application_state": True,
            "transmitter_visible": False,
            "csi_role": "actual_current_channel_state",
            "ofdm_resource_element_capacity": 12,
            "snapshot_count": 3,
        }
        allocation_metadata = {
            **common,
            "target_power": 1.0,
            "total_power": 4.0,
            "snapshot_count": 3,
            "subcarrier_count": 4,
            "policy": "water_filling_on_observed_csi",
        }
        rx_metadata = {
            **tx_metadata,
            "payload_symbol_count": coded_bit_count // 2,
            "channel_use_count": 12,
            "grid_padding_symbol_count": 12 - coded_bit_count // 2,
        }
        metadata = {
            "tx": tx_metadata,
            "decoded": decoded_metadata,
            "actual": actual_metadata,
            "allocation": allocation_metadata,
            "rx": rx_metadata,
        }
        for name, overrides in dict(metadata_overrides or {}).items():
            metadata[name].update(deepcopy(overrides))

        if power is None:
            power = np.ones((3, 4), dtype=np.float32)
        power = np.asarray(power, dtype=np.float32)
        if tx_symbols is None:
            tx_symbols = np.full(
                (coded_bit_count // 2,),
                (1.0 + 1.0j) / np.sqrt(2.0),
                dtype=np.complex64,
            )
        tx_symbols = np.asarray(tx_symbols, dtype=np.complex64)
        h_freq = np.ones((1, 3, 4), dtype=np.complex64)

        reference_path = root / "reference.npz"
        decoded_path = root / "decoded.npz"
        tx_path = root / "tx_symbols.npz"
        actual_path = root / "actual_state.npz"
        allocation_path = root / "allocation.npz"
        report_path = root / "decoder_report.json"
        np.savez_compressed(
            reference_path,
            bits=reference_bits,
            metadata_json=json.dumps({"bit_count": int(reference_bits.size)}),
        )
        np.savez_compressed(
            decoded_path,
            bits=decoded_bits,
            metadata_json=json.dumps(metadata["decoded"], sort_keys=True),
        )
        np.savez_compressed(
            tx_path,
            symbols=tx_symbols,
            metadata_json=json.dumps(metadata["tx"], sort_keys=True),
        )
        np.savez_compressed(
            actual_path,
            h_freq=h_freq,
            gains=np.ones((3, 4), dtype=np.float32),
            metadata_json=json.dumps(metadata["actual"], sort_keys=True),
        )
        np.savez_compressed(
            allocation_path,
            power=power,
            metadata_json=json.dumps(metadata["allocation"], sort_keys=True),
        )
        failure_indices = [
            index for index, passed in enumerate(statuses) if not passed
        ]
        report = {
            "schema_version": 1,
            "profile_sha256": profile_sha256,
            "transport_block_count": len(blocks),
            "transport_block_crc_status": statuses,
            "failed_transport_block_indices": failure_indices,
            "transport_block_crc_failure_count": len(failure_indices),
            "transport_block_error_rate": (
                float(len(failure_indices)) / float(len(blocks))
            ),
            "failure_policy": "zero_fill",
            "num_bp_iter": decoder_iterations,
            "decoded_bit_count": int(decoded_bits.size),
        }
        report.update(dict(report_overrides or {}))
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inputs = {
            "reference_payload": Artifact(
                "channel.payload_bits.numpy", reference_path, {"bit_count": int(reference_bits.size)}
            ),
            "decoded_payload": Artifact(
                "channel.payload_bits.numpy", decoded_path, metadata["decoded"]
            ),
            "decoder_report": Artifact("metrics.report", report_path, report),
            "tx_symbols": Artifact(
                "channel.symbols.complex_numpy", tx_path, metadata["tx"]
            ),
            "actual_state": Artifact(
                "channel.ofdm_channel_state.numpy", actual_path, metadata["actual"]
            ),
            "allocation": Artifact(
                "channel.power_allocation.numpy",
                allocation_path,
                metadata["allocation"],
            ),
        }
        if include_rx:
            rx_path = root / "rx_symbols.npz"
            np.savez_compressed(
                rx_path,
                symbols=tx_symbols,
                metadata_json=json.dumps(metadata["rx"], sort_keys=True),
            )
            inputs["rx_symbols"] = Artifact(
                "channel.rx_symbols.complex_numpy", rx_path, metadata["rx"]
            )
        return inputs

    def _run(self, root: Path, inputs: dict[str, Artifact]):
        return NrLdpcOfdmDeliveryMetricsOperation().run(
            OperationContext(
                recipe_name="nr_ldpc_ofdm_delivery_test",
                step_id="delivery",
                params={},
                inputs=inputs,
                run_dir=root,
                step_dir=root / "delivery",
            )
        )

    def test_crc_failed_all_zero_block_delivers_zero_bits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(Path(tmp), self._inputs(Path(tmp)))

        self.assertEqual(result.metadata["metric_profile"], "noema.nr_ldpc_ofdm_delivery.v1")
        self.assertEqual(len(result.metadata["rows"]), 2)
        failed = result.metadata["rows"][1]
        self.assertEqual(failed["encoder_input_bit_count"], 4)
        self.assertEqual(failed["tail_zero_padding_bit_count"], 0)
        self.assertEqual(failed["target_coderate"], 0.5)
        self.assertEqual(failed["rate_matched_codeword_lengths"], [8])
        self.assertEqual(failed["payload_mismatch_bit_count"], 0)
        self.assertFalse(failed["crc_passed"])
        self.assertEqual(failed["delivered_payload_bit_count"], 0)
        self.assertEqual(
            result.metrics["channel.nr_ldpc.delivered_payload_bit_count"], 4
        )
        self.assertEqual(
            result.metrics["channel.nr_ldpc.transport_block_error_rate"], 0.5
        )

    def test_primary_denominator_excludes_simulator_grid_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(Path(tmp), self._inputs(Path(tmp)))

        self.assertEqual(
            result.metrics["channel.occupied_qpsk_data_resource_element_count"], 8
        )
        self.assertEqual(
            result.metrics["channel.simulator_executed_channel_use_count"], 12
        )
        self.assertEqual(
            result.metrics["channel.simulator_grid_padding_channel_use_count"], 4
        )
        self.assertEqual(
            result.metrics[
                "channel.nr_ldpc_ofdm.all_attempt_goodput_bits_per_occupied_data_resource_element"
            ],
            0.5,
        )
        self.assertEqual(result.metrics["task.score"], 0.5)

    def test_profile_offset_and_crc_pass_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "crc").mkdir()
            with self.assertRaisesRegex(
                OperationError, "passed CRC.*mismatches"
            ):
                self._run(
                    root / "crc",
                    self._inputs(
                        root / "crc",
                        statuses=[True, True],
                        decoded_bits=np.asarray(
                            [1, 0, 1, 0, 1, 0, 0, 0], dtype=np.uint8
                        ),
                    ),
                )

            (root / "profile").mkdir()
            with self.assertRaisesRegex(OperationError, "profile"):
                self._run(
                    root / "profile",
                    self._inputs(
                        root / "profile",
                        metadata_overrides={
                            "decoded": {"nr_profile_sha256": "0" * 64}
                        },
                    ),
                )

            bad_blocks = deepcopy(
                self._inputs(root / "seed")["tx_symbols"].metadata[
                    "nr_transport_blocks"
                ]
            )
            bad_blocks[1]["coded_offset"] = 7
            (root / "offset").mkdir()
            with self.assertRaisesRegex(OperationError, "coded offsets"):
                self._run(
                    root / "offset",
                    self._inputs(root / "offset", blocks=bad_blocks),
                )

    def test_power_noise_and_symbol_binding_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "budget").mkdir()
            bad_power = np.ones((3, 4), dtype=np.float32)
            bad_power[1, 0] = 1.25
            with self.assertRaisesRegex(OperationError, "sum-power budget"):
                self._run(
                    root / "budget",
                    self._inputs(root / "budget", power=bad_power),
                )

            (root / "symbols").mkdir()
            bad_symbols = np.full(
                (8,), 2.0 + 0.0j, dtype=np.complex64
            )
            with self.assertRaisesRegex(OperationError, "power map"):
                self._run(
                    root / "symbols",
                    self._inputs(root / "symbols", tx_symbols=bad_symbols),
                )

            (root / "noise").mkdir()
            with self.assertRaisesRegex(OperationError, "noise_variance"):
                self._run(
                    root / "noise",
                    self._inputs(
                        root / "noise",
                        metadata_overrides={
                            "allocation": {"noise_variance": 0.25}
                        },
                    ),
                )

    def test_operation_is_registered(self) -> None:
        operation = build_registry().get("metrics.nr_ldpc_ofdm_delivery")
        self.assertIsInstance(operation, NrLdpcOfdmDeliveryMetricsOperation)


if __name__ == "__main__":
    unittest.main()
