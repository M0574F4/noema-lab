from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "demo_trainings" / "neural_receiver_supervised_qpsk"
STANDALONE_MODULE_NAMES = (
    "datamodule",
    "evaluate",
    "frontend",
    "losses",
    "model",
    "structured_input",
    "train",
)
saved_standalone_modules = {
    name: sys.modules.pop(name)
    for name in STANDALONE_MODULE_NAMES
    if name in sys.modules
}
sys.path.insert(0, str(SCAFFOLD))
try:
    import evaluate as receiver_evaluate
    import train as receiver_train
    from datamodule import (
        held_out_split_integrity_report,
        load_capture_dataset,
        split_integrity_report,
    )
    from frontend import receiver_iq_forward_matrix
    from model import build_receiver, receiver_candidates
finally:
    sys.path.pop(0)
    for name in STANDALONE_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.modules.update(saved_standalone_modules)


class NeuralReceiverDemoScaffoldTests(unittest.TestCase):
    def test_demo_frontend_calibration_matches_runtime_transform(self):
        from noema_lab.ops.channel.digital import _receiver_iq_forward_matrix

        expected = _receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        actual = receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_supervised_affine_initialization_recovers_the_calibration_lines(self):
        matrix = receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        offset = np.asarray([0.18, -0.12], dtype=np.float64)
        bits = np.tile(
            np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8),
            (8, 1),
        )
        ideal = (1.0 - 2.0 * bits.astype(np.float64)) / np.sqrt(2.0)
        impaired = ideal @ matrix.T + offset.reshape(1, 2)
        dataset = type(
            "Dataset",
            (),
            {
                "features": impaired.astype(np.float32),
                "target_bits": bits,
            },
        )()
        candidate = receiver_candidates(
            {
                "candidates": [
                    {
                        "id": "affine_iq_calibrator",
                        "architecture": "affine",
                    }
                ]
            }
        )[0]
        model = build_receiver(candidate)
        receiver_train._initialize_affine_from_labels(model, dataset)

        torch = __import__("torch")
        with torch.no_grad():
            logits = model(torch.from_numpy(dataset.features)).numpy()
        np.testing.assert_array_equal(logits < 0.0, bits.astype(bool))
        np.testing.assert_allclose(
            model.network.weight.detach().numpy(),
            np.sqrt(2.0) * np.linalg.inv(matrix),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_affine_logistic_fit_converges_on_independent_noisy_symbols(self):
        rng = np.random.default_rng(314159)
        matrix = receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        offset = np.asarray([0.18, -0.12], dtype=np.float64)

        def dataset(sample_count):
            bits = rng.integers(0, 2, size=(sample_count, 2), dtype=np.uint8)
            ideal = (1.0 - 2.0 * bits.astype(np.float64)) / np.sqrt(2.0)
            impaired = ideal @ matrix.T + offset.reshape(1, 2)
            impaired += rng.normal(0.0, 0.18, size=impaired.shape)
            return type(
                "Dataset",
                (),
                {
                    "features": impaired.astype(np.float32),
                    "target_bits": bits,
                },
            )()

        candidate = receiver_candidates(
            {
                "candidates": [
                    {
                        "id": "affine_iq_calibrator",
                        "architecture": "affine",
                    }
                ]
            }
        )[0]
        torch = __import__("torch")
        model = build_receiver(candidate)
        row = receiver_train._fit_affine_calibrator(
            model,
            dataset(2048),
            dataset(1024),
            candidate=candidate,
            seed=23,
            batch_size=256,
            workers=0,
            device=torch.device("cpu"),
            max_iterations=50,
            history_size=20,
            weight_decay=0.0,
        )
        self.assertEqual(row["fit"], "supervised_affine_logistic_lbfgs")
        self.assertGreater(row["optimizer_iterations"], 0)
        self.assertLess(row["validation_ber"], 0.02)

    def test_legacy_candidate_default_still_supports_custom_nonlinear_searches(self):
        candidates = receiver_candidates({"hidden_dim": 8})
        self.assertEqual(
            [candidate.architecture for candidate in candidates],
            ["affine", "symbol_mlp"],
        )
        for candidate in candidates:
            output = build_receiver(candidate)(
                __import__("torch").zeros((5, 2), dtype=__import__("torch").float32)
            )
            self.assertEqual(tuple(output.shape), (5, 2))

    def test_loader_preserves_packet_snr_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _write_capture(
                Path(tmp),
                split="test",
                snr_db=[-2.0, 6.0],
                rx=np.asarray(
                    [
                        [1 + 1j, -1 + 1j, 1 - 1j],
                        [-1 - 1j, 1 + 1j, -1 + 1j],
                    ],
                    dtype=np.complex64,
                ),
                bits=np.asarray(
                    [[0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0]],
                    dtype=np.uint8,
                ),
            )
            dataset = load_capture_dataset(
                [str(capture)],
                feature_tap="rx",
                target_tap="bits",
                expected_split="test",
            )
            np.testing.assert_array_equal(
                dataset.snr_db,
                np.asarray([-2.0, -2.0, -2.0, 6.0, 6.0, 6.0]),
            )
            self.assertEqual(len(dataset.packet_sha256), 2)
            self.assertEqual(len(set(dataset.packet_sha256)), 2)

    def test_boundary_agreement_uses_a_dense_plane_not_test_bit_noise(self):
        class IdentitySession:
            def run(self, output_names, inputs):
                self.output_names = output_names
                return [np.asarray(inputs["rx_symbols_ri"], dtype=np.float32)]

        agreement = receiver_evaluate._decision_boundary_agreement(
            IdentitySession(),
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            grid_size=65,
        )
        self.assertEqual(agreement["grid_size"], 65)
        self.assertEqual(agreement["bit_decision_agreement_rate"], 1.0)
        self.assertEqual(agreement["symbol_region_agreement_rate"], 1.0)

    def test_split_integrity_rejects_repeated_and_cross_split_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rx = np.asarray([[1 + 1j, -1 - 1j]], dtype=np.complex64)
            bits = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
            train = load_capture_dataset(
                [str(_write_capture(root / "train", "train", [-2.0], rx, bits))],
                feature_tap="rx",
                target_tap="bits",
                expected_split="train",
            )
            validation = load_capture_dataset(
                [
                    str(
                        _write_capture(
                            root / "validation",
                            "validation",
                            [-2.0],
                            rx.copy(),
                            bits.copy(),
                        )
                    )
                ],
                feature_tap="rx",
                target_tap="bits",
                expected_split="validation",
            )
            with self.assertRaisesRegex(ValueError, "train and validation share 1 packet"):
                split_integrity_report(
                    {"train": train, "validation": validation}
                )

    def test_held_out_check_and_per_snr_comparison_are_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = load_capture_dataset(
                [
                    str(
                        _write_capture(
                            root / "train",
                            "train",
                            [-2.0],
                            np.asarray([[1 + 1j]], dtype=np.complex64),
                            np.asarray([[0, 0]], dtype=np.uint8),
                        )
                    )
                ],
                feature_tap="rx",
                target_tap="bits",
                expected_split="train",
            )
            validation = load_capture_dataset(
                [
                    str(
                        _write_capture(
                            root / "validation",
                            "validation",
                            [2.0],
                            np.asarray([[-1 - 1j]], dtype=np.complex64),
                            np.asarray([[1, 1]], dtype=np.uint8),
                        )
                    )
                ],
                feature_tap="rx",
                target_tap="bits",
                expected_split="validation",
            )
            evidence = split_integrity_report(
                {"train": train, "validation": validation}
            )
            test = load_capture_dataset(
                [
                    str(
                        _write_capture(
                            root / "test",
                            "test",
                            [6.0],
                            np.asarray([[1 - 1j]], dtype=np.complex64),
                            np.asarray([[0, 1]], dtype=np.uint8),
                        )
                    )
                ],
                feature_tap="rx",
                target_tap="bits",
                expected_split="test",
            )
            held_out = held_out_split_integrity_report(test, evidence)
            self.assertEqual(held_out["status"], "passed")
            self.assertEqual(
                held_out["pairwise_overlap_packet_counts"],
                {"train__test": 0, "validation__test": 0},
            )

        targets = np.asarray([[False, False], [True, True]], dtype=bool)
        uncompensated = np.asarray([[False, False], [True, True]], dtype=bool)
        calibrated = np.asarray([[False, False], [True, True]], dtype=bool)
        learned = np.asarray([[False, True], [True, True]], dtype=bool)
        rows = receiver_evaluate._per_snr_metrics(
            snr_db=np.asarray([-2.0, 6.0]),
            learned_decisions=learned,
            uncompensated_decisions=uncompensated,
            calibrated_decisions=calibrated,
            targets=targets,
        )
        self.assertEqual([row["snr_db"] for row in rows], [-2.0, 6.0])
        self.assertEqual(rows[0]["learned_receiver"]["bit_errors"], 1)
        self.assertEqual(rows[0]["uncompensated_qpsk"]["bit_errors"], 0)
        self.assertEqual(rows[0]["calibrated_iq_oracle"]["bit_errors"], 0)
        self.assertGreater(
            rows[0]["theoretical_qpsk_ber"],
            rows[1]["theoretical_qpsk_ber"],
        )


def _write_capture(
    path: Path,
    split: str,
    snr_db: list[float],
    rx: np.ndarray,
    bits: np.ndarray,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "shards").mkdir(exist_ok=True)
    np.savez(path / "shards" / "shard_0000.npz", rx=rx, bits=bits)
    schema = {
        "kind": "noema.capture_dataset",
        "split": split,
        "captured_samples": int(rx.shape[0]),
        "tap_schemas": {"rx": {}, "bits": {}},
        "runs": [
            {
                "captured_samples": 1,
                "sweep": {"wireless_channel.snr_db": value},
            }
            for value in snr_db
        ],
        "shards": [
            {
                "path": "shards/shard_0000.npz",
                "sample_start": 0,
            }
        ],
    }
    (path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
