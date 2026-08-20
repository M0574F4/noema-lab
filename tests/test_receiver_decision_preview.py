import unittest

import numpy as np

from noema_lab.core.operations import OperationError
from noema_lab.ops.channel.digital import (
    _apply_receiver_iq_impairment,
    _compensate_receiver_iq_impairment,
    _qpsk_calibrated_iq_decision_preview,
    _qpsk_decision_preview_from_logits,
    _qpsk_decision_probe,
    _qpsk_preview_with_observed_constellation,
    _qpsk_reference_decision_preview,
    _receiver_iq_forward_matrix,
)
from noema_lab.training.sionna_blocks import ReceiverIqImpairmentBlock


class ReceiverDecisionPreviewTests(unittest.TestCase):
    def test_reference_preview_has_four_canonical_qpsk_regions(self):
        preview = _qpsk_reference_decision_preview(
            receiver_mode="analytical_qpsk",
            receiver_label="Analytical QPSK demapper",
        )

        self.assertEqual(preview["schema_version"], 1)
        self.assertEqual(preview["kind"], "memoryless_qpsk_iq_decision_regions")
        self.assertEqual(preview["conditioning"], {"kind": "none", "memoryless": True})
        grid = preview["grid"]
        self.assertEqual((grid["width"], grid["height"]), (64, 64))
        self.assertEqual(len(grid["class_rows"]), 64)
        self.assertEqual(set("".join(grid["class_rows"])), {"0", "1", "2", "3"})
        self.assertEqual(
            {point["bits"] for point in preview["constellation"]},
            {"00", "01", "10", "11"},
        )

    def test_logit_preview_records_model_identity_and_shifted_boundaries(self):
        _axis, features = _qpsk_decision_probe()
        logits = features.copy()
        logits[:, 0] += 0.4
        preview = _qpsk_decision_preview_from_logits(
            logits,
            receiver_mode="learned_artifact",
            receiver_label="Learned neural receiver",
            model_sha256="a" * 64,
        )

        self.assertEqual(preview["model_sha256"], "a" * 64)
        rows = preview["grid"]["class_rows"]
        reference = _qpsk_reference_decision_preview(
            receiver_mode="reference_qpsk",
            receiver_label="Reference",
        )["grid"]["class_rows"]
        self.assertNotEqual(rows, reference)

    def test_logit_preview_rejects_a_non_memoryless_output_shape(self):
        with self.assertRaisesRegex(OperationError, "must have shape"):
            _qpsk_decision_preview_from_logits(
                np.zeros((63, 2), dtype=np.float32),
                receiver_mode="learned_artifact",
                receiver_label="Invalid",
            )

    def test_fixed_iq_impairment_is_invertible_and_moves_optimal_boundaries(self):
        matrix = _receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        offset = np.asarray([0.18, -0.12], dtype=np.float64)
        symbols = np.asarray(
            [1.0 + 1.0j, -1.0 + 0.5j, 0.25 - 0.75j],
            dtype=np.complex64,
        )
        impaired = _apply_receiver_iq_impairment(symbols, matrix, offset)
        metadata = {
            "receiver_iq_impairment": {
                "forward_matrix": matrix.tolist(),
                "dc_offset": offset.tolist(),
            }
        }
        recovered = _compensate_receiver_iq_impairment(impaired, metadata)
        np.testing.assert_allclose(recovered, symbols, atol=2e-6, rtol=2e-6)

        oracle = _qpsk_calibrated_iq_decision_preview(
            metadata,
            receiver_mode="oracle_frontend_calibrated",
            receiver_label="Calibrated I/Q oracle",
        )
        ordinary = _qpsk_reference_decision_preview(
            receiver_mode="reference_qpsk",
            receiver_label="Uncompensated QPSK",
        )
        self.assertNotEqual(
            oracle["grid"]["class_rows"],
            ordinary["grid"]["class_rows"],
        )

        observed_oracle = _qpsk_preview_with_observed_constellation(
            oracle,
            metadata,
        )
        grid = observed_oracle["grid"]
        for point in observed_oracle["constellation"]:
            column = round(
                (point["i"] - grid["i_min"])
                / (grid["i_max"] - grid["i_min"])
                * (grid["width"] - 1)
            )
            row = round(
                (point["q"] - grid["q_min"])
                / (grid["q_max"] - grid["q_min"])
                * (grid["height"] - 1)
            )
            self.assertEqual(
                int(grid["class_rows"][row][column]),
                point["class_id"],
                point["bits"],
            )

    def test_torch_frontend_materialization_matches_numpy_runtime(self):
        torch = __import__("torch")
        symbols = np.asarray(
            [1.0 + 1.0j, -0.5 + 0.25j, 0.1 - 1.2j],
            dtype=np.complex64,
        )
        matrix = _receiver_iq_forward_matrix(
            gain_imbalance_db=5.0,
            quadrature_error_deg=12.0,
            phase_offset_deg=20.0,
        )
        expected = _apply_receiver_iq_impairment(
            symbols,
            matrix,
            np.asarray([0.18, -0.12], dtype=np.float64),
        )
        block = ReceiverIqImpairmentBlock()
        actual = block(torch.as_tensor(symbols)).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
