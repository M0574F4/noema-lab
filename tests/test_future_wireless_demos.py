from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.operations import OperationContext
from noema_lab.core.execution_profiles import execution_profile_catalog
from noema_lab.ops import build_registry
from noema_lab.ops.future_wireless import (
    IsacOfdmAllocatorOperation,
    IsacOfdmMetricsOperation,
    IsacOfdmScenarioOperation,
    LeoNtnTrackingAdapterOperation,
    LeoNtnTrackingMetricsOperation,
    LeoNtnTrackingScenarioOperation,
    NearFieldEstimatorAdapterOperation,
    NearFieldMetricsOperation,
    NearFieldXlMimoScenarioOperation,
)


class FutureWirelessDemoTests(unittest.TestCase):
    def _run(self, operation, root: Path, step: str, *, params=None, inputs=None):
        return operation.run(
            OperationContext(
                recipe_name="future_wireless_test",
                step_id=step,
                params=dict(params or {}),
                inputs=dict(inputs or {}),
                run_dir=root,
                step_dir=root / step,
            )
        )

    def test_three_execution_profiles_are_registered(self) -> None:
        catalog = execution_profile_catalog()
        for profile_id in (
            "isac_ofdm_allocation",
            "near_field_range_angle_focusing",
            "leo_ntn_tracking",
        ):
            self.assertIsNotNone(catalog.get(profile_id, 1))

    def test_isac_references_preserve_the_power_simplex_and_emit_joint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._run(
                IsacOfdmScenarioOperation(), root, "data",
                params={"example_count": 8, "subcarriers": 8, "snr_db": 8, "sensing_weight": 0.45, "seed": 19},
            )
            for mode in ("equal_power", "communications_water_filling", "scalarized_reference"):
                decision = self._run(
                    IsacOfdmAllocatorOperation(), root, "allocator_" + mode,
                    params={"mode": mode}, inputs={"problem": source.outputs["problem"]},
                )
                with np.load(decision.outputs["decision"].path, allow_pickle=False) as payload:
                    power = np.asarray(payload["power"])
                self.assertEqual(power.shape, (8, 8))
                self.assertTrue(np.all(power >= 0.0))
                np.testing.assert_allclose(np.sum(power, axis=1), 1.0, atol=1e-6)
                report = self._run(
                    IsacOfdmMetricsOperation(), root, "metrics_" + mode,
                    inputs={"problem": source.outputs["problem"], "decision": decision.outputs["decision"]},
                )
                self.assertGreater(report.metrics["isac.scalarized_utility"], 0.0)
                self.assertGreater(report.metrics["isac.communication_rate_bps_hz"], 0.0)

    def test_near_field_references_use_separate_truth_and_score_focusing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._run(
                NearFieldXlMimoScenarioOperation(), root, "data",
                params={"example_count": 6, "antennas": 16, "snr_db": 18, "seed": 29},
            )
            self.assertNotEqual(source.outputs["problem"].path, source.outputs["truth"].path)
            with np.load(source.outputs["problem"].path, allow_pickle=False) as payload:
                self.assertNotIn("truth", payload.files)
            for mode in ("far_field_steering", "polar_codebook", "oracle_focus"):
                estimate = self._run(
                    NearFieldEstimatorAdapterOperation(), root, "estimator_" + mode,
                    params={"mode": mode},
                    inputs={"problem": source.outputs["problem"], "truth": source.outputs["truth"]},
                )
                report = self._run(
                    NearFieldMetricsOperation(), root, "metrics_" + mode,
                    inputs={"problem": source.outputs["problem"], "truth": source.outputs["truth"], "estimate": estimate.outputs["estimate"]},
                )
                self.assertGreaterEqual(report.metrics["near_field.normalized_focusing_gain"], 0.0)
                self.assertLessEqual(report.metrics["near_field.normalized_focusing_gain"], 1.000001)
            self.assertAlmostEqual(report.metrics["near_field.range_rmse_m"], 0.0, places=6)
            self.assertAlmostEqual(report.metrics["near_field.angle_rmse_deg"], 0.0, places=6)

    def test_leo_ntn_references_are_causal_except_explicit_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._run(
                LeoNtnTrackingScenarioOperation(), root, "data",
                params={"example_count": 12, "history_length": 6, "beam_count": 9, "snr_db": 14, "seed": 31},
            )
            with np.load(source.outputs["problem"].path, allow_pickle=False) as payload:
                self.assertNotIn("truth", payload.files)
            scores = {}
            for mode in ("hold_last", "linear_extrapolation", "oracle_future"):
                decision = self._run(
                    LeoNtnTrackingAdapterOperation(), root, "tracker_" + mode,
                    params={"mode": mode},
                    inputs={"problem": source.outputs["problem"], "truth": source.outputs["truth"]},
                )
                report = self._run(
                    LeoNtnTrackingMetricsOperation(), root, "metrics_" + mode,
                    inputs={"truth": source.outputs["truth"], "decision": decision.outputs["decision"]},
                )
                scores[mode] = report.metrics
            self.assertEqual(scores["oracle_future"]["ntn.doppler_mae_hz"], 0.0)
            self.assertEqual(scores["oracle_future"]["ntn.beam_handover_accuracy"], 1.0)
            self.assertGreaterEqual(scores["linear_extrapolation"]["ntn.beam_handover_accuracy"], 0.0)

    def test_registry_exposes_portable_artifact_capabilities(self) -> None:
        registry = build_registry()
        expected = {
            "model.isac_ofdm_allocator_adapter": "isac_allocator",
            "model.near_field_estimator_adapter": "near_field_estimator",
            "model.leo_ntn_tracking_adapter": "ntn_tracker",
        }
        for operation_id, entrypoint in expected.items():
            description = registry.get(operation_id).describe()
            self.assertTrue(description["training_capabilities"]["portable_replacement"])
            self.assertEqual(description["trained_artifact_abi"]["entrypoint_id"], entrypoint)


if __name__ == "__main__":
    unittest.main()
