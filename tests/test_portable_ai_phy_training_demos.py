from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.core.training_plans import apply_training_plan, load_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.exporter import available_exporters


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    (
        "range-localization",
        "recipes/localization_adapter_baseline.yaml",
        "demo_trainings/localization_supervised_mlp/training_plan.yaml",
        "localizer",
        "position.mse",
        "range_observation.observation",
        "range_observation.truth",
        "localization_estimator",
    ),
    (
        "aoa-estimation",
        "recipes/aoa_adapter_ula_baseline.yaml",
        "demo_trainings/aoa_estimation_covariance_mlp/training_plan.yaml",
        "estimator",
        "angle.mse",
        "array_observation.observation",
        "array_observation.truth",
        "aoa_estimator",
    ),
    (
        "beam-selection",
        "recipes/beamforming_adapter_baseline.yaml",
        "demo_trainings/beam_selection_supervised_mlp/training_plan.yaml",
        "beamformer",
        "beam.codebook_cross_entropy",
        "data.problem",
        "",
        "beam_policy",
    ),
)


class PortableAiPhyTrainingDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_exporters_write_complete_reference_projects(self) -> None:
        exporters = {exporter.id: exporter for exporter in available_exporters()}
        for demo, recipe_path, plan_path, step, loss, feature, target, entrypoint in CASES:
            with self.subTest(demo=demo), tempfile.TemporaryDirectory() as temporary:
                recipe = apply_training_plan(
                    load_recipe(ROOT / recipe_path),
                    load_training_plan(ROOT / plan_path),
                )
                exporter = exporters[demo]
                self.assertTrue(exporter.supports(recipe, self.registry))
                from noema_lab.training.exporter import ExportOptions

                plan = exporter.build_plan(
                    recipe,
                    self.registry,
                    ExportOptions(
                        optimizable_steps=[step],
                        loss=loss,
                        framework="torch",
                        exporter=demo,
                        project_root=ROOT,
                        include_starter=True,
                    ),
                )
                self.assertEqual(plan.feature_reference, feature)
                self.assertEqual(plan.target_reference, target)
                destination = Path(temporary) / "reference_training"
                payload = exporter.write_export(plan, destination)
                for relative in (
                    "README.md",
                    "template.yaml",
                    "task.py",
                    "datamodule.py",
                    "train.py",
                    "evaluate.py",
                    "build_benchmark.py",
                    "requirements.txt",
                    "train_config.yaml",
                    "project_manifest.yaml",
                ):
                    self.assertTrue((destination / relative).is_file(), relative)
                config = yaml.safe_load(
                    (destination / "train_config.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(config["training"]["artifact_entrypoint"], entrypoint)
                self.assertFalse(config["data"]["test_split_exposed_to_training"])
                self.assertIn("trained_artifact.yaml", str(payload["trained_artifacts"]))

    def test_operation_owned_abis_match_the_three_returned_entrypoints(self) -> None:
        expected = {
            "model.localization_adapter": ("localization_estimator", {"anchors", "ranges"}, {"positions"}),
            "model.aoa_estimator_adapter": ("aoa_estimator", {"snapshots_ri"}, {"angles_deg"}),
            "model.beamforming_adapter": ("beam_policy", {"channels_ri"}, {"weights_ri"}),
        }
        for operation_id, (entrypoint, inputs, outputs) in expected.items():
            description = self.registry.get(operation_id).describe()
            abi = description["trained_artifact_abi"]
            self.assertEqual(abi["entrypoint_id"], entrypoint)
            self.assertEqual(set(abi["inputs"]), inputs)
            self.assertEqual(set(abi["outputs"]), outputs)
            self.assertIn("learned_artifact", description["params_schema"]["properties"]["mode"]["enum"])

    def test_aoa_scalar_truth_capture_keeps_one_label_per_snapshot_record(self) -> None:
        payload = load_recipe(ROOT / "recipes/aoa_adapter_ula_baseline.yaml").to_dict()
        next(step for step in payload["steps"] if step["id"] == "data")["params"]["example_count"] = 4
        payload["dataset_capture"] = {
            "split": "train",
            "samples": 4,
            "max_runs": 1,
            "shard_size": 4,
            "seed_mode": "fixed_seed",
            "taps": [
                {"id": "snapshots", "from": "array_observation.observation"},
                {"id": "angles_deg", "from": "array_observation.truth"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capture"
            result = run_dataset_capture_recipe(
                recipe_from_dict(payload),
                self.registry,
                LocalStore(root / ".noema"),
                output,
            )
            self.assertEqual(result["captured_samples"], 4)
            with np.load(output / "shards" / "shard_0000.npz", allow_pickle=False) as shard:
                self.assertEqual(shard["snapshots"].shape, (4, 8, 64))
                self.assertEqual(shard["angles_deg"].shape, (4,))
            schema = json.loads((output / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["tap_schemas"]["angles_deg"]["record_shape"], [])

    def test_public_readiness_pages_link_all_three_complete_workflows(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count("✅ [Train + compare]"), 12)
        self.assertNotIn("train/replace adapter not yet available", readme)
        for name in (
            "learned_range_localization_demo",
            "learned_aoa_estimation_demo",
            "learned_beam_selection_demo",
        ):
            self.assertTrue((ROOT / "docs" / "tutorials" / (name + ".md")).is_file())
            self.assertIn(name + ".html", readme)


if __name__ == "__main__":
    unittest.main()
