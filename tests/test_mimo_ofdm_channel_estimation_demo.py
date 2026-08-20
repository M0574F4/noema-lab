from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.core.training_plans import apply_training_plan, load_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.channel_estimation_export import (
    CHANNEL_ESTIMATION_LOSS,
    build_channel_estimation_export_plan,
    write_channel_estimation_starter,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "mimo_ofdm_adapter_channel_estimation.yaml"
TRAINING_PLAN = (
    ROOT
    / "demo_trainings"
    / "mimo_ofdm_channel_estimation_cnn"
    / "training_plan.yaml"
)
SCAFFOLD = ROOT / "demo_trainings" / "mimo_ofdm_channel_estimation_cnn"
UI_SOURCE = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class MimoOfdmChannelEstimationDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_template_is_realistic_and_conformant(self) -> None:
        recipe = load_recipe(RECIPE)
        validate_recipe_against_registry(recipe, self.registry)
        self.assertEqual(recipe.name, "mimo_ofdm_mixed_tdl_channel_estimation")
        self.assertEqual(
            recipe.metadata["matrix"]["dimensions"]["channel.snr_db"],
            [-5, 0, 5, 10, 15, 20],
        )
        self.assertEqual(
            recipe.metadata["matrix"]["dimensions"]["channel.tdl_profile"],
            ["A", "C", "E"],
        )
        channel = next(step for step in recipe.steps if step.id == "data")
        self.assertEqual(channel.params["wireless_backend"], "sionna")
        self.assertEqual(channel.params["tdl_model"], "C")
        self.assertEqual(
            (channel.params["rx_antennas"], channel.params["tx_antennas"]),
            (2, 2),
        )

    def test_channel_estimator_has_portable_runtime_abi(self) -> None:
        operation = self.registry.get("model.channel_estimator_adapter").describe()
        abi = operation["trained_artifact_abi"]
        self.assertEqual(abi["entrypoint_id"], "channel_estimator")
        self.assertEqual(abi["required_operation_inputs"], ["problem"])
        self.assertEqual(
            list(abi["inputs"]),
            [
                "pilot_ls_ri",
                "pilot_mask",
                "ls_estimate_ri",
                "noise_variance",
            ],
        )
        self.assertEqual(list(abi["outputs"]), ["h_hat_ri"])

    def test_numpy_smoke_emits_training_tensors_and_system_metrics(self) -> None:
        payload = load_recipe(RECIPE).to_dict()
        payload["metadata"].pop("matrix", None)
        for step in payload["steps"]:
            if step["id"] == "data":
                step["params"]["example_count"] = 4
                step["params"]["wireless_backend"] = "numpy"
            elif step["id"] == "pilot_observation":
                step["params"]["wireless_backend"] = "numpy"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(
                self.registry,
                LocalStore(Path(tmp) / ".noema"),
            ).run(recipe_from_dict(payload))
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(summary["status"], "completed")
        observation = next(
            step for step in summary["steps"] if step["id"] == "pilot_observation"
        )
        output_names = set(observation["outputs"])
        self.assertTrue(
            {
                "observation",
                "pilot_ls",
                "pilot_mask",
                "ls_estimate",
                "noise_variance",
                "truth",
            }.issubset(output_names)
        )
        evaluation = next(
            step for step in summary["steps"] if step["id"] == "evaluation"
        )
        self.assertIn(
            "mimo.channel_estimation.zf_spectral_efficiency_bps_hz",
            evaluation["metrics"],
        )
        self.assertIn(
            "mimo.channel_estimation.zf_rate_retention",
            evaluation["metrics"],
        )

    def test_reference_training_plan_and_starter_export(self) -> None:
        recipe = apply_training_plan(
            load_recipe(RECIPE),
            load_training_plan(TRAINING_PLAN),
        )
        plan = build_channel_estimation_export_plan(
            recipe,
            self.registry,
            optimizable_steps=["estimator"],
            loss=CHANNEL_ESTIMATION_LOSS,
            framework="torch",
        )
        self.assertEqual(plan.capture_plan.total_samples, 18432)
        references = {tap.reference for tap in plan.capture_plan.taps}
        self.assertIn("pilot_observation.pilot_ls", references)
        self.assertIn("pilot_observation.pilot_mask", references)
        self.assertIn("pilot_observation.ls_estimate", references)
        self.assertIn("pilot_observation.noise_variance", references)
        self.assertIn("pilot_observation.truth", references)
        with tempfile.TemporaryDirectory() as tmp:
            payload = write_channel_estimation_starter(
                plan,
                Path(tmp) / "reference_training",
            )
            files = set(payload["files"])
        self.assertIn("train.py", files)
        self.assertIn("evaluate.py", files)
        self.assertIn("build_benchmark.py", files)

    def test_returned_artifact_and_benchmark_use_the_declared_runtime_identity(self) -> None:
        trainer = (SCAFFOLD / "train.py").read_text(encoding="utf-8")
        model = (SCAFFOLD / "model.py").read_text(encoding="utf-8")
        builder = (SCAFFOLD / "build_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('"id": "estimator"', trainer)
        self.assertIn('"role": "mimo_ofdm_channel_estimator"', trainer)
        self.assertIn('"component": "estimator"', trainer)
        self.assertIn('"pilot_mask"', model)
        self.assertIn('self.register_buffer("idft_cos"', model)
        self.assertIn("NoiseConditionedResidualBlock", model)
        self.assertIn("DelayResidualBlock", model)
        self.assertIn(
            "gain * features + frequency_residual + delay_residual",
            model,
        )
        self.assertIn('"artifact_package_sha256": artifact_package_sha256', builder)
        self.assertIn("inspect_trained_artifact(", builder)

    def test_default_post_training_campaign_is_compact_and_profile_paired(self) -> None:
        module = runpy.run_path(str(SCAFFOLD / "build_benchmark.py"))
        evidence = {"path": "evidence.json", "sha256": "a" * 64}
        pack = module["_pack"](
            recipe_path="benchmark_recipe.yaml",
            artifact_path="trained_artifact.yaml",
            artifact_package_sha256="b" * 64,
            snr_values=[-5, 0, 5, 10, 15, 20],
            seeds=[91001, 92001, 93001],
            tdl_profiles=["A", "C", "E"],
            history=evidence,
            evaluation=evidence,
            artifact=evidence,
        )
        self.assertEqual(len(pack["recipes"]), 54)
        self.assertEqual(
            pack["metadata"]["paired_held_out_scenario_units"],
            [
                {"tdl_profile": "A", "seed": 91001},
                {"tdl_profile": "C", "seed": 92001},
                {"tdl_profile": "E", "seed": 93001},
            ],
        )
        self.assertEqual(
            set(pack["baselines"]),
            {"least_squares", "fixed_prior_lmmse"},
        )

    def test_ui_has_explicit_learned_artifact_control_and_channel_preview(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"model.channel_estimator_adapter": Object.freeze({', source)
        self.assertIn("Set Reference method to Learned artifact", source)
        self.assertIn("function channelEstimationHeatmapFigure(entry)", source)
        self.assertIn("preview.linkLabels", source)


if __name__ == "__main__":
    unittest.main()
