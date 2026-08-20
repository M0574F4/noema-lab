from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from noema_lab.core import dataplane
from noema_lab.core.artifacts import Artifact
from noema_lab.core.benchmarks import load_benchmark_pack, validate_benchmark_pack
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops import build_registry
from noema_lab.ops.channel.digital import (
    DigitalModulateOperation,
    IdentityModulateOperation,
    NeuralReceiverAdapterOperation,
    WirelessChannelOperation,
    _apply_sionna_awgn,
    _apply_wireless_channel,
    _sionna_available,
)
from noema_lab.ops.metrics.bits import BitErrorRateOperation, BlockErrorRateOperation


ROOT = Path(__file__).resolve().parents[1]


def _context(
    root: Path,
    step_id: str,
    operation_input: dict[str, Artifact],
    params: dict | None = None,
) -> OperationContext:
    return OperationContext(
        recipe_name="digital_adversarial_contract",
        step_id=step_id,
        params=dict(params or {}),
        inputs=operation_input,
        run_dir=root,
        step_dir=root / step_id,
    )


def _bits_artifact(
    root: Path,
    name: str,
    bits: np.ndarray,
    metadata: dict | None = None,
) -> Artifact:
    path = root / ("%s.npz" % name)
    payload_metadata = dict(metadata or {})
    np.savez_compressed(
        path,
        bits=bits,
        metadata_json=json.dumps(payload_metadata),
    )
    return Artifact("channel.payload_bits.numpy", path, payload_metadata)


class DigitalAdversarialContractTests(unittest.TestCase):
    def test_identity_modulation_image_rate_path_uses_its_actual_bit_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _bits_artifact(
                root,
                "identity_bits",
                np.asarray([1, 0, 1], dtype=np.uint8),
                {
                    "source_item_payload_bit_counts": [1, 2],
                    "original_shapes": [[1, 2, 2, 3], [1, 2, 4, 3]],
                },
            )
            result = IdentityModulateOperation().run(
                _context(root, "identity_modulate", {"bits": source})
            )

        self.assertEqual(result.metrics["rate.coded_bpp"], 0.25)
        self.assertEqual(result.metrics["rate.padded_bpp"], 0.25)
        self.assertEqual(
            result.outputs["symbols"].metadata["source_item_symbol_counts"],
            [1, 2],
        )
        self.assertEqual(
            result.outputs["symbols"].metadata[
                "source_item_channel_uses_per_pixel"
            ],
            [0.25, 0.25],
        )

    def test_each_source_item_is_padded_and_accounted_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _bits_artifact(
                root,
                "mixed_lengths",
                np.asarray([1, 0, 1], dtype=np.uint8),
                {"source_item_payload_bit_counts": [1, 2]},
            )
            expectations = {
                "qpsk": ([1, 1], [2, 2]),
                "qam16": ([1, 1], [4, 4]),
            }
            for modulation, (symbol_counts, padded_counts) in expectations.items():
                result = DigitalModulateOperation().run(
                    _context(
                        root,
                        "modulate_%s" % modulation,
                        {"bits": source},
                        {
                            "modulation": modulation,
                            "data_plane_backend": "python_numpy",
                        },
                    )
                )
                metadata = result.outputs["symbols"].metadata
                self.assertEqual(metadata["source_item_symbol_counts"], symbol_counts)
                self.assertEqual(
                    metadata["source_item_modulator_input_bit_counts"], [1, 2]
                )
                self.assertEqual(
                    metadata["source_item_padded_bit_counts"], padded_counts
                )

    def test_nonbinary_bits_are_rejected_before_backend_dispatch(self):
        malformed = np.asarray([2], dtype=np.uint8)
        backends = ["python_numpy"]
        if dataplane.native_available():
            backends.append("cpp_native")
        for backend in backends:
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                    dataplane.bpsk_modulate(malformed, backend)
                with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                    dataplane.bit_error_count(
                        np.asarray([0], dtype=np.uint8), malformed, backend
                    )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _bits_artifact(
                root, "reference", np.asarray([0], dtype=np.uint8)
            )
            candidate = _bits_artifact(root, "candidate", malformed)
            with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                BitErrorRateOperation().run(
                    _context(
                        root,
                        "malformed_ber",
                        {"reference": reference, "candidate": candidate},
                    )
                )

    def test_empty_ber_and_bler_are_undefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = _bits_artifact(root, "empty", np.asarray([], dtype=np.uint8))
            inputs = {"reference": empty, "candidate": empty}
            with self.assertRaisesRegex(OperationError, "undefined"):
                BitErrorRateOperation().run(
                    _context(root, "empty_ber", inputs)
                )
            with self.assertRaisesRegex(OperationError, "undefined"):
                BlockErrorRateOperation().run(
                    _context(root, "empty_bler", inputs)
                )

    def test_linear_npz_receiver_rejects_nonfinite_checkpoint_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            checkpoint_path = root / "receiver.npz"
            metadata = {"modulation": "qpsk", "bit_count": 2}
            np.savez_compressed(
                symbols_path,
                symbols=np.asarray([1.0 + 1.0j], dtype=np.complex64),
                metadata_json=json.dumps(metadata),
            )
            np.savez_compressed(
                checkpoint_path,
                weight=np.asarray([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32),
                bias=np.zeros((2,), dtype=np.float32),
            )
            source = Artifact(
                "channel.rx_symbols.complex_numpy", symbols_path, metadata
            )
            with self.assertRaisesRegex(OperationError, "finite values"):
                NeuralReceiverAdapterOperation().run(
                    _context(
                        root,
                        "linear_receiver",
                        {"rx_symbols": source},
                        {
                            "mode": "linear_npz",
                            "modulation": "qpsk",
                            "checkpoint_path": str(checkpoint_path),
                        },
                    )
                )

    def test_kodak_rate_distortion_packs_bind_native_codec_bits(self):
        pack_paths = [
            ROOT
            / "benchmarks"
            / "benchmark_v1"
            / "kodak_image_reconstruction_v1.yaml",
            ROOT / "benchmarks" / "image_reconstruction_kodak_smoke_v1.yaml",
            ROOT / "benchmarks" / "image_reconstruction_rd_v1.yaml",
        ]
        registry = build_registry()
        for path in pack_paths:
            with self.subTest(pack=path.name):
                pack = load_benchmark_pack(path)
                metric_ids = {metric["id"] for metric in pack.metrics}
                self.assertIn("codec.native_bit_count", metric_ids)
                self.assertNotIn("codec.bit_count", metric_ids)
                validation = validate_benchmark_pack(pack, registry, ROOT)
                self.assertEqual(validation["recipe_count"], len(pack.recipes))

    def test_auto_wireless_backend_is_numpy_and_never_probes_sionna(self):
        symbols = np.asarray([1.0 + 0.0j], dtype=np.complex64)
        failure = OperationError("synthetic incompatible Sionna runtime")
        with (
            patch(
                "noema_lab.ops.channel.digital._sionna_available",
                return_value=True,
            ),
            patch(
                "noema_lab.ops.channel.digital._apply_sionna_or_raise",
                side_effect=failure,
            ) as sionna_call,
        ):
            _rx, report = _apply_wireless_channel(
                symbols,
                "awgn",
                10.0,
                np.random.RandomState(7),
                {},
                "python_numpy",
                "auto",
                7,
            )
            self.assertEqual(report["requested_wireless_backend"], "auto")
            self.assertEqual(report["backend"], "python_numpy")
            self.assertNotIn("backend_fallback", report)
            self.assertNotIn("sionna_fallback_reason", report)
            sionna_call.assert_not_called()
            with self.assertRaisesRegex(
                OperationError, "synthetic incompatible Sionna runtime"
            ):
                _apply_wireless_channel(
                    symbols,
                    "awgn",
                    10.0,
                    np.random.RandomState(7),
                    {},
                    "python_numpy",
                    "sionna",
                    7,
                )
            self.assertEqual(sionna_call.call_count, 1)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                symbols_path = root / "symbols.npz"
                np.savez_compressed(symbols_path, symbols=symbols)
                source = Artifact(
                    "channel.symbols.complex_numpy", symbols_path, {}
                )
                result = WirelessChannelOperation().run(
                    _context(
                        root,
                        "wireless_auto",
                        {"symbols": source},
                        {
                            "channel": "awgn",
                            "wireless_backend": "auto",
                            "noise_mode": "snr_at_unit_power",
                            "snr_db": 10.0,
                            "seed": 7,
                        },
                    )
                )
                self.assertEqual(result.metadata["wireless_backend"], "numpy")
                self.assertEqual(
                    result.metadata["data_plane_backend"], "python_numpy"
                )
                self.assertEqual(
                    result.metadata["requested_wireless_backend"], "auto"
                )
                self.assertNotIn("backend_fallback", result.metadata)
                self.assertNotIn("sionna_fallback_reason", result.metadata)
                self.assertEqual(sionna_call.call_count, 1)

    @unittest.skipUnless(
        _sionna_available(),
        "Sionna 2 and PyTorch are optional",
    )
    def test_installed_sionna_awgn_path_is_seeded_and_pytorch_native(self):
        symbols = np.asarray([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex64)
        first, first_report = _apply_sionna_awgn(symbols, 10.0, 19)
        second, second_report = _apply_sionna_awgn(symbols, 10.0, 19)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_report["backend"], "sionna")
        self.assertEqual(
            first_report["backend_detail"], "sionna.awgn.pytorch"
        )
        self.assertEqual(first_report["data_plane_backend"], "torch")
        self.assertEqual(first_report, second_report)


if __name__ == "__main__":
    unittest.main()
