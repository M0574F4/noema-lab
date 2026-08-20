from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import (
    Crc32CheckOperation,
    Crc32PacketizeOperation,
    DigitalDemodulateOperation,
    DigitalModulateOperation,
    OfdmChannelStateOperation,
    RepetitionChannelDecoderOperation,
    RepetitionChannelEncoderOperation,
    WirelessChannelOperation,
)
from noema_lab.ops.channel.tensor import (
    BitsToLatentsOperation,
    LatentsToBitsOperation,
)
from noema_lab.ops.phase_tracking import QpskPilotModulateOperation


def _context(
    root: Path,
    step_id: str,
    inputs: dict[str, Artifact],
    params: dict | None = None,
) -> OperationContext:
    return OperationContext(
        recipe_name="capture_record_layout",
        step_id=step_id,
        params=dict(params or {}),
        inputs=inputs,
        run_dir=root,
        step_dir=root / step_id,
    )


def _bit_artifact(
    root: Path,
    bits: np.ndarray,
    metadata: dict,
    name: str = "bits",
) -> Artifact:
    path = root / ("%s.npz" % name)
    np.savez_compressed(
        path,
        bits=np.asarray(bits, dtype=np.uint8),
        metadata_json=json.dumps(metadata),
    )
    return Artifact("channel.payload_bits.numpy", path, dict(metadata))


class CaptureRecordLayoutPropagationTests(unittest.TestCase):
    def test_repetition_modem_channel_round_trip_rewrites_each_record_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _bit_artifact(
                root,
                np.asarray([0, 1, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
                {
                    "bit_count": 10,
                    "capture_record_count": 2,
                    "capture_record_shape": [5],
                },
            )
            encoded = RepetitionChannelEncoderOperation().run(
                _context(
                    root,
                    "encode",
                    {"bits": source},
                    {"factor": 3, "data_plane_backend": "python_numpy"},
                )
            ).outputs["coded_bits"]
            self.assertEqual(
                (
                    encoded.metadata["capture_record_count"],
                    encoded.metadata["capture_record_shape"],
                ),
                (2, [15]),
            )

            modulated = DigitalModulateOperation().run(
                _context(
                    root,
                    "modulate",
                    {"bits": encoded},
                    {"modulation": "qpsk", "data_plane_backend": "python_numpy"},
                )
            ).outputs["symbols"]
            self.assertEqual(
                (
                    modulated.metadata["capture_record_count"],
                    modulated.metadata["capture_record_shape"],
                ),
                (2, [8]),
            )

            received = WirelessChannelOperation().run(
                _context(
                    root,
                    "channel",
                    {"symbols": modulated},
                    {
                        "channel": "awgn",
                        "noise_mode": "fixed_variance",
                        "noise_variance": 1e-12,
                        "wireless_backend": "numpy",
                        "data_plane_backend": "python_numpy",
                        "seed": 7,
                    },
                )
            ).outputs["rx_symbols"]
            self.assertEqual(received.metadata["capture_record_shape"], [8])

            demodulated = DigitalDemodulateOperation().run(
                _context(
                    root,
                    "demodulate",
                    {"rx_symbols": received},
                    {
                        "modulation": "qpsk",
                        "data_plane_backend": "python_numpy",
                    },
                )
            ).outputs["bits"]
            self.assertEqual(
                (
                    demodulated.metadata["capture_record_count"],
                    demodulated.metadata["capture_record_shape"],
                ),
                (2, [15]),
            )

            decoded = RepetitionChannelDecoderOperation().run(
                _context(
                    root,
                    "decode",
                    {"coded_bits": demodulated},
                    {"factor": 3, "data_plane_backend": "python_numpy"},
                )
            ).outputs["bits"]
            self.assertEqual(
                (
                    decoded.metadata["capture_record_count"],
                    decoded.metadata["capture_record_shape"],
                ),
                (2, [5]),
            )
            with np.load(decoded.path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["bits"], np.load(source.path)["bits"])

    def test_crc_packetization_keeps_batch_records_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _bit_artifact(
                root,
                np.arange(34, dtype=np.uint8) % 2,
                {
                    "bit_count": 34,
                    "capture_record_count": 2,
                    "capture_record_shape": [17],
                },
            )
            framed = Crc32PacketizeOperation().run(
                _context(
                    root,
                    "packetize",
                    {"bits": source},
                    {
                        "packet_payload_bits": 8,
                        "data_plane_backend": "python_numpy",
                    },
                )
            ).outputs["bits"]
            self.assertEqual(framed.metadata["source_item_payload_bit_counts"], [17, 17])
            self.assertEqual(framed.metadata["capture_record_count"], 2)
            self.assertEqual(
                framed.metadata["capture_record_shape"],
                [framed.metadata["source_item_framed_bit_counts"][0]],
            )

            recovered = Crc32CheckOperation().run(
                _context(
                    root,
                    "check",
                    {"bits": framed},
                    {"data_plane_backend": "python_numpy"},
                )
            ).outputs["bits"]
            self.assertEqual(recovered.metadata["capture_record_count"], 2)
            self.assertEqual(recovered.metadata["capture_record_shape"], [17])
            with np.load(recovered.path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["bits"], np.load(source.path)["bits"])

    def test_cardinality_transform_rejects_stale_or_conflicting_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = _bit_artifact(
                root,
                np.zeros((10,), dtype=np.uint8),
                {
                    "capture_record_count": 2,
                    "capture_record_shape": [4],
                },
                "stale",
            )
            with self.assertRaisesRegex(OperationError, "declares 8 elements"):
                RepetitionChannelEncoderOperation().run(
                    _context(root, "encode_stale", {"bits": stale}, {"factor": 3})
                )

            conflicting = _bit_artifact(
                root,
                np.zeros((10,), dtype=np.uint8),
                {
                    "capture_record_count": 2,
                    "capture_record_shape": [5],
                    "source_item_payload_bit_counts": [4, 6],
                },
                "conflicting",
            )
            with self.assertRaisesRegex(OperationError, "conflicts"):
                Crc32PacketizeOperation().run(
                    _context(
                        root,
                        "packetize_conflict",
                        {"bits": conflicting},
                        {"packet_payload_bits": 8},
                    )
                )

    def test_phase_modulator_rejects_packet_layout_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _bit_artifact(
                root,
                np.zeros((16,), dtype=np.uint8),
                {
                    "example_count": 2,
                    "bit_count_per_example": 8,
                    "capture_record_count": 1,
                    "capture_record_shape": [16],
                },
            )
            with self.assertRaisesRegex(OperationError, "conflicts"):
                QpskPilotModulateOperation().run(
                    _context(root, "pilot_modulate", {"bits": source})
                )

    def test_channel_state_replaces_source_layout_with_snapshot_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symbols_path = root / "symbols.npz"
            metadata = {
                "capture_record_count": 1,
                "capture_record_shape": [128],
                "array": "symbols",
            }
            np.savez_compressed(
                symbols_path,
                symbols=np.ones((128,), dtype=np.complex64),
                metadata_json=json.dumps(metadata),
            )
            source = Artifact(
                "channel.symbols.complex_numpy",
                symbols_path,
                metadata,
            )

            def fake_state(block_count, params, seed):
                return {
                    "h_freq": np.ones(
                        (block_count, 2, 8), dtype=np.complex64
                    ),
                    "tdl_model": "A",
                    "subcarrier_spacing_khz": 15.0,
                    "carrier_frequency_ghz": 3.5,
                    "delay_spread_ns": 100.0,
                    "mobility_kmh": 3.0,
                }

            with patch(
                "noema_lab.ops.channel.digital._generate_sionna_ofdm_channel_state",
                side_effect=fake_state,
            ):
                state = OfdmChannelStateOperation().run(
                    _context(
                        root,
                        "channel_state",
                        {"symbols": source},
                        {"ofdm_fft_size": 8, "num_ofdm_symbols": 2},
                    )
                ).outputs["state"]
            self.assertNotIn("capture_record_count", state.metadata)
            self.assertNotIn("capture_record_shape", state.metadata)
            self.assertEqual(state.metadata["capture_record_axis"], 0)
            self.assertEqual(state.metadata["array"], "gains")
            with np.load(state.path, allow_pickle=False) as payload:
                self.assertEqual(payload["gains"].shape[1], 8)

    def test_latent_bitpack_converts_axis_records_to_exact_flat_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latents_path = root / "latents.npz"
            latents = np.asarray(
                [[0.25, -0.5, 1.0], [2.0, 4.0, -8.0]],
                dtype=np.float32,
            )
            metadata = {
                "array": "latents",
                "capture_record_axis": 0,
            }
            np.savez_compressed(
                latents_path,
                latents=latents,
                metadata_json=json.dumps(metadata),
            )
            source = Artifact(
                "semantic.latents.numpy",
                latents_path,
                metadata,
            )
            packed = LatentsToBitsOperation().run(
                _context(
                    root,
                    "pack_latents",
                    {"latents": source},
                    {"dtype": "float32", "data_plane_backend": "python_numpy"},
                )
            ).outputs["bits"]
            self.assertEqual(packed.metadata["capture_record_count"], 2)
            self.assertEqual(packed.metadata["capture_record_shape"], [96])
            self.assertNotIn("capture_record_axis", packed.metadata)

            unpacked = BitsToLatentsOperation().run(
                _context(
                    root,
                    "unpack_latents",
                    {"bits": packed},
                    {"data_plane_backend": "python_numpy"},
                )
            ).outputs["latents"]
            self.assertEqual(unpacked.metadata["capture_record_count"], 2)
            self.assertEqual(unpacked.metadata["capture_record_shape"], [3])
            with np.load(unpacked.path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["latents"], latents)


if __name__ == "__main__":
    unittest.main()
