from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from noema_lab.cli.main import build_parser, main
from noema_lab.core.recipes import load_recipe
from noema_lab.core.reproducibility import recipe_fingerprint
from noema_lab.core.training_plans import (
    TrainingPlan,
    apply_training_plan,
    extract_legacy_training_plan,
    load_training_plan,
    neutral_recipe,
    scenario_recipe_fingerprint,
    training_plan_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingPlanTests(unittest.TestCase):
    def test_checked_in_demo_plans_do_not_change_scenario_identity(self):
        cases = (
            (
                "recipes/resource_equal_power_baseline.yaml",
                "demo_trainings/resource_allocation_unsupervised_shannon/training_plan.yaml",
            ),
            (
                "recipes/neural_receiver_qpsk_iq_calibration.yaml",
                "demo_trainings/neural_receiver_supervised_qpsk/training_plan.yaml",
            ),
            (
                "recipes/modulation_recognition_awgn.yaml",
                "demo_trainings/modulation_recognition_supervised_cnn/training_plan.yaml",
            ),
            (
                "recipes/csi_feedback_truncated_angular_delay.yaml",
                "demo_trainings/csi_feedback_autoencoder/training_plan.yaml",
            ),
            (
                "recipes/deepjscc_kodak_awgn_train.yaml",
                "demo_trainings/deepjscc_image_reconstruction/training_plan.yaml",
            ),
            (
                "recipes/localization_adapter_baseline.yaml",
                "demo_trainings/localization_supervised_mlp/training_plan.yaml",
            ),
            (
                "recipes/aoa_adapter_ula_baseline.yaml",
                "demo_trainings/aoa_estimation_covariance_mlp/training_plan.yaml",
            ),
            (
                "recipes/beamforming_adapter_baseline.yaml",
                "demo_trainings/beam_selection_supervised_mlp/training_plan.yaml",
            ),
            (
                "recipes/isac_ofdm_joint_allocation.yaml",
                "demo_trainings/isac_joint_allocation_deepsets/training_plan.yaml",
            ),
            (
                "recipes/near_field_xl_mimo_focusing.yaml",
                "demo_trainings/near_field_range_angle_mlp/training_plan.yaml",
            ),
            (
                "recipes/leo_ntn_doppler_beam_tracking.yaml",
                "demo_trainings/leo_ntn_tracking_mlp/training_plan.yaml",
            ),
        )
        for recipe_name, plan_name in cases:
            with self.subTest(recipe=recipe_name):
                recipe = load_recipe(ROOT / recipe_name)
                plan = load_training_plan(ROOT / plan_name)
                self.assertFalse(recipe.dataset_capture)
                materialized = apply_training_plan(recipe, plan)
                self.assertEqual(
                    scenario_recipe_fingerprint(materialized),
                    scenario_recipe_fingerprint(recipe),
                )
                self.assertTrue(plan.selected_steps)

    def test_plan_is_separate_from_neutral_recipe(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        self.assertFalse(recipe.dataset_capture)
        self.assertNotIn("training_performed", recipe.metadata)

        plan = TrainingPlan(
            selected_steps=("tx_power",),
            loss_steps=("evaluation",),
            dataset_capture={
                "taps": [{"id": "channel_gains", "from": "channel_state.state"}],
                "split_plan": {"total_samples": 30},
            },
            objective="resource.negative_shannon_spectral_efficiency",
        )
        materialized = apply_training_plan(recipe, plan)
        self.assertEqual(materialized.dataset_capture["split_plan"]["total_samples"], 30)
        self.assertEqual(plan.to_dict()["loss_steps"], ["evaluation"])
        self.assertFalse(recipe.dataset_capture)
        self.assertEqual(recipe_fingerprint(neutral_recipe(materialized)), recipe_fingerprint(recipe))
        self.assertEqual(scenario_recipe_fingerprint(materialized), recipe_fingerprint(recipe))
        self.assertNotEqual(training_plan_fingerprint(plan), scenario_recipe_fingerprint(materialized))
        self.assertEqual(len(training_plan_fingerprint(plan)), 64)

    def test_cli_export_allows_targets_to_come_from_training_plan(self):
        args = build_parser().parse_args(
            [
                "differentiable",
                "export",
                "scenario.yaml",
                "--training-plan",
                "training.yaml",
                "--out",
                "bundle",
            ]
        )
        self.assertEqual(args.optimizable, "")
        self.assertEqual(args.route_loss, "")
        self.assertEqual(args.framework, "")

    def test_cli_keeps_recipe_loss_steps_separate_from_external_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "training_plan.yaml"
            plan_path.write_text(
                yaml.safe_dump(
                    TrainingPlan(
                        selected_steps=("sender", "receiver"),
                        loss_steps=("evaluation",),
                        objective="researcher.hybrid_objective/v2",
                    ).to_dict(),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "differentiable",
                        "inspect",
                        str(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"),
                        "--training-plan",
                        str(plan_path),
                        "--json",
                    ]
                )
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["selected_loss_steps"], ["evaluation"])
        self.assertEqual(report["selected_replacement_steps"], ["sender", "receiver"])

    def test_cli_neutral_export_does_not_inject_a_demo_objective(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "noema_lab.cli.main.export_differentiable_scenario",
            return_value={"kind": "noema.training_interface_bundle@1"},
        ) as export:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "differentiable",
                        "export",
                        str(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"),
                        "--replacement",
                        "sender,receiver",
                        "--out",
                        str(Path(temporary) / "contract"),
                        "--json",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(export.call_args.kwargs["loss"], "")
        self.assertFalse(export.call_args.kwargs["include_starter"])

    def test_cli_preserves_an_explicit_external_training_plan_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "training_plan.yaml"
            plan_path.write_text(
                yaml.safe_dump(
                    TrainingPlan(
                        selected_steps=("sender", "receiver"),
                        objective="researcher.external_objective/v1",
                    ).to_dict(),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "noema_lab.cli.main.export_differentiable_scenario",
                return_value={"kind": "noema.training_interface_bundle@1"},
            ) as export:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(
                        [
                            "differentiable",
                            "export",
                            str(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"),
                            "--training-plan",
                            str(plan_path),
                            "--out",
                            str(Path(temporary) / "contract"),
                            "--json",
                        ]
                    )
        self.assertEqual(code, 0)
        self.assertEqual(export.call_args.kwargs["loss"], "")
        self.assertEqual(
            export.call_args.kwargs["training_objective"],
            "researcher.external_objective/v1",
        )

    def test_legacy_fields_are_extracted_without_loss(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        legacy = apply_training_plan(
            recipe,
            TrainingPlan(dataset_capture={"samples": 12, "taps": [{"id": "x", "from": "data.bits"}]}),
        )
        legacy.metadata["training_performed"] = False
        migrated, plan = extract_legacy_training_plan(legacy)
        self.assertFalse(migrated.dataset_capture)
        self.assertNotIn("training_performed", migrated.metadata)
        self.assertEqual(plan.dataset_capture["samples"], 12)


if __name__ == "__main__":
    unittest.main()
