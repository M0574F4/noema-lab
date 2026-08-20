from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops import build_registry
from noema_lab.ops.ai_phy import (
    AiPhyChannelRealizationSourceOperation,
    AiPhyPilotPatternSourceOperation,
    AoaEstimationMetricsOperation,
    AoaEstimatorAdapterOperation,
    AoaSceneSourceOperation,
    BeamformingAdapterOperation,
    BeamformingMetricsOperation,
    BeamformingScenarioSourceOperation,
    ChannelEstimationMetricsOperation,
    ChannelEstimatorAdapterOperation,
    LocalizationAdapterOperation,
    LocalizationGeometrySourceOperation,
    LocalizationMetricsOperation,
    LsChannelEstimatorOperation,
    MusicAoaEstimatorOperation,
    PilotObservationOperation,
    RangeObservationOperation,
    TrilaterationLocalizationOperation,
    UlaArrayObservationOperation,
    _add_awgn,
    _linear_mmse_channel_estimate,
)
from noema_lab.training.differentiability import sionna_available


class ExplicitAiPhyPipelineTests(unittest.TestCase):
    def _run(self, operation, root: Path, step_id: str, params=None, inputs=None):
        return operation.run(
            OperationContext(
                recipe_name="explicit_ai_phy_test",
                step_id=step_id,
                params=dict(params or {}),
                inputs=dict(inputs or {}),
                run_dir=root,
                step_dir=root / step_id,
            )
        )

    def test_direct_ai_phy_auto_wireless_backend_is_deterministic_numpy(self):
        values = np.ones((4,), dtype=np.complex64)
        with patch(
            "noema_lab.ops.ai_phy._sionna_available",
            return_value=True,
        ) as availability:
            output, backend = _add_awgn(
                values,
                0.1,
                np.random.RandomState(17),
                "auto",
            )
        availability.assert_not_called()
        self.assertEqual(backend, "numpy")
        self.assertEqual(output.shape, values.shape)
        self.assertTrue(np.all(np.isfinite(output)))

    @unittest.skipUnless(
        sionna_available(),
        "Sionna 2.x/PyTorch is optional",
    )
    def test_explicit_ai_phy_sionna_backend_is_seeded_and_strict(self):
        values = np.ones((4,), dtype=np.complex64)
        first, first_backend = _add_awgn(
            values,
            0.1,
            np.random.RandomState(23),
            "sionna",
        )
        second, second_backend = _add_awgn(
            values,
            0.1,
            np.random.RandomState(23),
            "sionna",
        )
        self.assertEqual(first_backend, "sionna")
        self.assertEqual(second_backend, "sionna")
        np.testing.assert_array_equal(first, second)

    def test_comb_pilots_reject_transmitter_offset_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                OperationError,
                r"tx_antennas <= min\(pilot_spacing, subcarriers\)",
            ):
                self._run(
                    AiPhyPilotPatternSourceOperation(),
                    root,
                    "pilots",
                    {
                        "scenario": "comb",
                        "tx_antennas": 5,
                        "subcarriers": 16,
                        "pilot_spacing": 4,
                    },
                )

    def test_frequency_lmmse_does_not_regress_ls_at_15_db_flat_oracle(self):
        rng = np.random.RandomState(501)
        count = 8192
        truth = (
            rng.normal(size=(count, 1, 1, 1))
            + 1j * rng.normal(size=(count, 1, 1, 1))
        ) / np.sqrt(2.0)
        noise_variance = 10.0 ** (-15.0 / 10.0)
        noise = np.sqrt(noise_variance / 2.0) * (
            rng.normal(size=truth.shape) + 1j * rng.normal(size=truth.shape)
        )
        observations = (truth + noise).astype(np.complex64)
        lmmse = _linear_mmse_channel_estimate(
            {
                "observations": observations,
                "pilots": np.ones((1, 1), dtype=np.complex64),
                "pilot_mask": np.ones((1, 1), dtype=np.uint8),
            },
            noise_variance=noise_variance,
            channel_variance=1.0,
            channel_tap_count=1,
        )
        self.assertLess(
            float(np.mean(np.abs(lmmse - truth) ** 2)),
            float(np.mean(np.abs(observations - truth) ** 2)),
        )

    def test_flat_siso_ls_divides_known_qpsk_pilots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            channel = self._run(
                AiPhyChannelRealizationSourceOperation(),
                root,
                "channel",
                {"scenario": "flat_siso", "example_count": 64, "subcarriers": 8, "seed": 11},
            )
            pilots = self._run(
                AiPhyPilotPatternSourceOperation(),
                root,
                "pilots",
                {"scenario": "unit", "tx_antennas": 1, "subcarriers": 8, "seed": 7},
            )
            observation = self._run(
                PilotObservationOperation(),
                root,
                "observation",
                {"snr_db": 80.0, "wireless_backend": "numpy", "seed": 13},
                {"channel": channel.outputs["channel"], "pilots": pilots.outputs["pilots"]},
            )
            with np.load(observation.outputs["observation"].path) as model_input, np.load(
                observation.outputs["truth"].path
            ) as evaluator_truth:
                self.assertNotIn("h_true", model_input.files)
                self.assertNotIn("observations", evaluator_truth.files)
            estimate = self._run(
                LsChannelEstimatorOperation(),
                root,
                "estimator",
                inputs={"problem": observation.outputs["observation"]},
            )
            report = self._run(
                ChannelEstimationMetricsOperation(),
                root,
                "evaluation",
                inputs={"problem": observation.outputs["truth"], "estimate": estimate.outputs["estimate"]},
            )
            with np.load(channel.outputs["channel"].path) as payload:
                h_true = payload["h_true"]
            with np.load(pilots.outputs["pilots"].path) as payload:
                pilot_symbols = payload["pilots"]
            with np.load(estimate.outputs["estimate"].path) as payload:
                h_hat = payload["h_hat"]
            self.assertEqual(h_true.shape, (64, 1, 1, 8))
            self.assertEqual(h_hat.shape, h_true.shape)
            self.assertTrue(np.any(np.abs(pilot_symbols - 1.0) > 0.5))
            self.assertLess(report.metrics["channel_estimation.nmse"], 1e-6)

    def test_comb_pilot_mimo_ofdm_preserves_tx_axis_and_interpolates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            channel = self._run(
                AiPhyChannelRealizationSourceOperation(),
                root,
                "channel",
                {
                    "scenario": "mimo_ofdm",
                    "example_count": 32,
                    "rx_antennas": 2,
                    "tx_antennas": 2,
                    "subcarriers": 32,
                    "channel_tap_count": 3,
                    "seed": 19,
                },
            )
            pilots = self._run(
                AiPhyPilotPatternSourceOperation(),
                root,
                "pilots",
                {"scenario": "comb", "tx_antennas": 2, "subcarriers": 32, "pilot_spacing": 4, "seed": 23},
            )
            observation = self._run(
                PilotObservationOperation(),
                root,
                "observation",
                {"snr_db": 35.0, "wireless_backend": "numpy", "seed": 29},
                {"channel": channel.outputs["channel"], "pilots": pilots.outputs["pilots"]},
            )
            estimate = self._run(
                LsChannelEstimatorOperation(),
                root,
                "estimator",
                inputs={"problem": observation.outputs["observation"]},
            )
            report = self._run(
                ChannelEstimationMetricsOperation(),
                root,
                "evaluation",
                inputs={"problem": observation.outputs["truth"], "estimate": estimate.outputs["estimate"]},
            )
            with np.load(estimate.outputs["estimate"].path) as payload:
                h_hat = payload["h_hat"]
            self.assertEqual(h_hat.shape, (32, 2, 2, 32))
            self.assertTrue(np.all(np.isfinite(h_hat)))
            self.assertLess(report.metrics["mimo.channel_estimation.nmse"], 0.35)

    def test_lmmse_adapter_improves_full_pilot_low_snr_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            channel = self._run(
                AiPhyChannelRealizationSourceOperation(),
                root,
                "channel",
                {"scenario": "flat_siso", "example_count": 1024, "seed": 31},
            )
            pilots = self._run(
                AiPhyPilotPatternSourceOperation(), root, "pilots", {"scenario": "unit", "seed": 37}
            )
            observation = self._run(
                PilotObservationOperation(),
                root,
                "observation",
                {"snr_db": 0.0, "wireless_backend": "numpy", "seed": 41},
                {"channel": channel.outputs["channel"], "pilots": pilots.outputs["pilots"]},
            )
            ls = self._run(
                LsChannelEstimatorOperation(), root, "ls", inputs={"problem": observation.outputs["observation"]}
            )
            adapter = self._run(
                ChannelEstimatorAdapterOperation(), root, "adapter", inputs={"problem": observation.outputs["observation"]}
            )
            adapter_ls = self._run(
                ChannelEstimatorAdapterOperation(),
                root,
                "adapter_ls",
                {"mode": "least_squares"},
                {"problem": observation.outputs["observation"]},
            )
            ls_report = self._run(
                ChannelEstimationMetricsOperation(),
                root,
                "ls_metrics",
                inputs={"problem": observation.outputs["truth"], "estimate": ls.outputs["estimate"]},
            )
            adapter_report = self._run(
                ChannelEstimationMetricsOperation(),
                root,
                "adapter_metrics",
                inputs={"problem": observation.outputs["truth"], "estimate": adapter.outputs["estimate"]},
            )
            self.assertLess(
                adapter_report.metrics["channel_estimation.nmse"], ls_report.metrics["channel_estimation.nmse"]
            )
            self.assertAlmostEqual(adapter.outputs["estimate"].metadata["lmmse_shrinkage"], 0.5)
            with np.load(ls.outputs["estimate"].path) as baseline_payload, np.load(
                adapter_ls.outputs["estimate"].path
            ) as adapter_payload:
                self.assertTrue(np.allclose(baseline_payload["h_hat"], adapter_payload["h_hat"]))
            self.assertEqual(adapter_ls.outputs["estimate"].metadata["adapter_mode"], "least_squares")

    def test_channel_metrics_reject_shape_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            channel = self._run(
                AiPhyChannelRealizationSourceOperation(),
                root,
                "channel",
                {"scenario": "mimo_ofdm", "example_count": 2, "tx_antennas": 2, "subcarriers": 8},
            )
            pilots = self._run(
                AiPhyPilotPatternSourceOperation(),
                root,
                "pilots",
                {"scenario": "unit", "tx_antennas": 2, "subcarriers": 8},
            )
            observation = self._run(
                PilotObservationOperation(),
                root,
                "observation",
                {"wireless_backend": "numpy"},
                {"channel": channel.outputs["channel"], "pilots": pilots.outputs["pilots"]},
            )
            bad_path = root / "bad_estimate.npz"
            with np.load(channel.outputs["channel"].path) as payload:
                bad = payload["h_true"][..., :-1]
            np.savez_compressed(bad_path, h_hat=bad, metadata_json=json.dumps({}))
            with self.assertRaisesRegex(OperationError, "exactly match"):
                self._run(
                    ChannelEstimationMetricsOperation(),
                    root,
                    "metrics",
                    inputs={
                        "problem": observation.outputs["truth"],
                        "estimate": Artifact("ai_phy.channel_estimate.numpy", bad_path, {}),
                    },
                )

    def test_range_observation_noise_and_localization_are_snr_causal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = self._run(
                LocalizationGeometrySourceOperation(),
                root,
                "scene",
                {"example_count": 128, "area_m": 20.0, "anchor_count": 4, "seed": 43},
            )
            low = self._run(
                RangeObservationOperation(),
                root,
                "low_observation",
                {"snr_db": 5.0, "range_noise_floor_m": 0.02, "seed": 47},
                {"scene": scene.outputs["scene"]},
            )
            high = self._run(
                RangeObservationOperation(),
                root,
                "high_observation",
                {"snr_db": 30.0, "range_noise_floor_m": 0.02, "seed": 47},
                {"scene": scene.outputs["scene"]},
            )
            with np.load(high.outputs["observation"].path) as model_input, np.load(
                high.outputs["truth"].path
            ) as evaluator_truth:
                self.assertNotIn("positions", model_input.files)
                self.assertNotIn("true_ranges", model_input.files)
                self.assertNotIn("ranges", evaluator_truth.files)
            low_estimate = self._run(
                TrilaterationLocalizationOperation(),
                root,
                "low_estimator",
                inputs={"problem": low.outputs["observation"]},
            )
            high_estimate = self._run(
                TrilaterationLocalizationOperation(),
                root,
                "high_estimator",
                inputs={"problem": high.outputs["observation"]},
            )
            adapter_trilateration = self._run(
                LocalizationAdapterOperation(),
                root,
                "adapter_trilateration",
                {"mode": "trilateration"},
                {"problem": high.outputs["observation"]},
            )
            low_report = self._run(
                LocalizationMetricsOperation(),
                root,
                "low_metrics",
                inputs={"problem": low.outputs["truth"], "estimate": low_estimate.outputs["estimate"]},
            )
            high_report = self._run(
                LocalizationMetricsOperation(),
                root,
                "high_metrics",
                inputs={"problem": high.outputs["truth"], "estimate": high_estimate.outputs["estimate"]},
            )
            self.assertLess(high.metadata["range_noise_m"], low.metadata["range_noise_m"])
            self.assertLess(high_report.metrics["localization.rmse_m"], low_report.metrics["localization.rmse_m"])
            with np.load(high_estimate.outputs["estimate"].path) as baseline_payload, np.load(
                adapter_trilateration.outputs["estimate"].path
            ) as adapter_payload:
                self.assertTrue(np.allclose(baseline_payload["positions"], adapter_payload["positions"]))
            self.assertNotIn("localization_preview", high_estimate.outputs["estimate"].metadata)
            preview = high_report.outputs["report"].metadata["metadata"]["localization_preview"]
            self.assertEqual(len(preview["anchors"]), 4)
            self.assertEqual(len(preview["true_positions"]), 16)
            self.assertEqual(len(preview["estimated_positions"]), 16)
            self.assertEqual(preview["shown_example_count"], 16)
            self.assertEqual(preview["total_example_count"], 128)
            json.dumps(preview)

    def test_music_and_trainable_adapter_estimate_half_wavelength_ula_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = self._run(
                AoaSceneSourceOperation(),
                root,
                "scene",
                {"example_count": 12, "angle_min_deg": -55.0, "angle_max_deg": 55.0, "seed": 53},
            )
            problem = self._run(
                UlaArrayObservationOperation(),
                root,
                "array_observation",
                {
                    "antenna_count": 8,
                    "snapshot_count": 96,
                    "snr_db": 20.0,
                    "element_spacing_wavelengths": 0.5,
                    "seed": 59,
                },
                {"scene": scene.outputs["scene"]},
            )
            with np.load(problem.outputs["observation"].path) as model_input, np.load(
                problem.outputs["truth"].path
            ) as evaluator_truth:
                self.assertNotIn("angles_deg", model_input.files)
                self.assertNotIn("snapshots", evaluator_truth.files)
            music = self._run(
                MusicAoaEstimatorOperation(),
                root,
                "music",
                {"grid_size": 441},
                {"problem": problem.outputs["observation"]},
            )
            adapter = self._run(
                AoaEstimatorAdapterOperation(),
                root,
                "adapter",
                {"grid_step_deg": 0.25},
                {"problem": problem.outputs["observation"]},
            )
            adapter_music = self._run(
                AoaEstimatorAdapterOperation(),
                root,
                "adapter_music",
                {"mode": "music", "grid_step_deg": 0.25},
                {"problem": problem.outputs["observation"]},
            )
            music_report = self._run(
                AoaEstimationMetricsOperation(),
                root,
                "music_metrics",
                inputs={"problem": problem.outputs["truth"], "estimate": music.outputs["estimate"]},
            )
            adapter_report = self._run(
                AoaEstimationMetricsOperation(),
                root,
                "adapter_metrics",
                inputs={"problem": problem.outputs["truth"], "estimate": adapter.outputs["estimate"]},
            )
            self.assertLess(music_report.metrics["aoa.rmse_deg"], 0.75)
            self.assertLess(adapter_report.metrics["aoa.rmse_deg"], 1.0)
            adapter_music_report = self._run(
                AoaEstimationMetricsOperation(),
                root,
                "adapter_music_metrics",
                inputs={"problem": problem.outputs["truth"], "estimate": adapter_music.outputs["estimate"]},
            )
            self.assertLess(adapter_music_report.metrics["aoa.rmse_deg"], 0.75)
            self.assertEqual(adapter_music.outputs["estimate"].metadata["estimator"], "music")
            self.assertTrue(AoaEstimatorAdapterOperation.differentiability["trainable_params"])
            with np.load(adapter.outputs["estimate"].path) as payload:
                self.assertEqual(payload["angles_deg"].shape, (12,))
            self.assertNotIn("aoa_preview", music.outputs["estimate"].metadata)
            preview = music_report.outputs["report"].metadata["metadata"]["aoa_preview"]
            self.assertEqual(len(preview["true_angles_deg"]), 12)
            self.assertEqual(len(preview["estimated_angles_deg"]), 12)
            self.assertEqual(preview["antenna_count"], 8)
            self.assertEqual(preview["element_spacing_wavelengths"], 0.5)
            json.dumps(preview)

    def test_beamforming_adapter_exposes_classical_method_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = self._run(
                BeamformingScenarioSourceOperation(),
                root,
                "beamforming_problem",
                {"example_count": 12, "tx_antennas": 8, "seed": 61},
            )
            mrt = self._run(
                BeamformingAdapterOperation(),
                root,
                "adapter_mrt",
                {"mode": "mrt"},
                {"problem": problem.outputs["problem"]},
            )
            codebook = self._run(
                BeamformingAdapterOperation(),
                root,
                "adapter_codebook",
                {"mode": "codebook_sweep_reference"},
                {"problem": problem.outputs["problem"]},
            )
            with np.load(mrt.outputs["decision"].path) as mrt_payload, np.load(
                codebook.outputs["decision"].path
            ) as codebook_payload:
                self.assertEqual(mrt_payload["weights"].shape, (12, 8))
                self.assertEqual(codebook_payload["weights"].shape, (12, 8))
                self.assertFalse(np.allclose(mrt_payload["weights"], codebook_payload["weights"]))
            self.assertEqual(mrt.outputs["decision"].metadata["beamformer"], "mrt")
            self.assertEqual(
                codebook.outputs["decision"].metadata["beamformer"],
                "codebook_sweep_reference",
            )

    def test_beamforming_metrics_reject_forged_or_malformed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = self._run(
                BeamformingScenarioSourceOperation(),
                root,
                "beamforming_problem",
                {"example_count": 4, "tx_antennas": 3, "seed": 17},
            )
            valid = self._run(
                BeamformingAdapterOperation(),
                root,
                "beamforming_valid",
                {"mode": "mrt"},
                {"problem": problem.outputs["problem"]},
            )
            report = self._run(
                BeamformingMetricsOperation(),
                root,
                "beamforming_metrics_valid",
                inputs={
                    "problem": problem.outputs["problem"],
                    "decision": valid.outputs["decision"],
                },
            )
            self.assertLessEqual(report.metrics["beamforming.normalized_gain"], 1.0)
            self.assertLessEqual(report.metrics["task.score"], 1.0)

            with np.load(valid.outputs["decision"].path) as payload:
                weights = payload["weights"]
            for name, forged in (
                ("scaled", weights * 10.0),
                ("wrong_shape", weights[:, :-1]),
                ("nonfinite", np.full_like(weights, np.nan)),
            ):
                path = root / f"{name}.npz"
                np.savez_compressed(
                    path,
                    weights=forged,
                    metadata_json=json.dumps({}),
                )
                decision = Artifact(
                    kind="ai_phy.beamforming_decision.numpy",
                    path=path,
                    metadata={},
                )
                with self.assertRaises(OperationError):
                    self._run(
                        BeamformingMetricsOperation(),
                        root,
                        f"beamforming_metrics_{name}",
                        inputs={
                            "problem": problem.outputs["problem"],
                            "decision": decision,
                        },
                    )

    def test_synthetic_spatial_sources_support_user_selected_fixed_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            localization = self._run(
                LocalizationGeometrySourceOperation(),
                root,
                "fixed_localization",
                {
                    "example_count": 5,
                    "area_m": 30.0,
                    "position_mode": "fixed",
                    "target_x_m": 7.5,
                    "target_y_m": 22.0,
                },
            )
            with np.load(localization.outputs["scene"].path) as payload:
                positions = payload["positions"]
            self.assertTrue(np.allclose(positions, [[7.5, 22.0]] * 5))
            self.assertEqual(localization.metadata["position_mode"], "fixed")

            aoa = self._run(
                AoaSceneSourceOperation(),
                root,
                "fixed_aoa",
                {
                    "example_count": 7,
                    "angle_mode": "fixed",
                    "angle_min_deg": -60.0,
                    "angle_max_deg": 60.0,
                    "source_angle_deg": -23.5,
                },
            )
            with np.load(aoa.outputs["scene"].path) as payload:
                angles = payload["angles_deg"]
            self.assertTrue(np.allclose(angles, [-23.5] * 7))
            self.assertEqual(aoa.metadata["angle_mode"], "fixed")

            with self.assertRaisesRegex(OperationError, "must lie inside"):
                self._run(
                    LocalizationGeometrySourceOperation(),
                    root,
                    "invalid_fixed_localization",
                    {"area_m": 10.0, "position_mode": "fixed", "target_x_m": 12.0, "target_y_m": 5.0},
                )
            with self.assertRaisesRegex(OperationError, "search range"):
                self._run(
                    AoaSceneSourceOperation(),
                    root,
                    "invalid_fixed_aoa",
                    {
                        "angle_mode": "fixed",
                        "angle_min_deg": -30.0,
                        "angle_max_deg": 30.0,
                        "source_angle_deg": 45.0,
                    },
                )

    def test_explicit_operation_ids_are_registered(self):
        registry = build_registry()
        for operation_id in (
            "source.ai_phy_channel_realization",
            "source.ai_phy_pilot_pattern",
            "wireless.pilot_observation",
            "source.localization_geometry",
            "wireless.range_observation",
            "source.aoa_scene",
            "wireless.ula_array_observation",
            "model.music_aoa_estimator",
            "model.aoa_estimator_adapter",
            "metrics.aoa_estimation",
        ):
            self.assertIsNotNone(registry.get(operation_id))


if __name__ == "__main__":
    unittest.main()
