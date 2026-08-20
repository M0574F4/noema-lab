from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from demo_trainings import prepare_example
from noema_lab.core.capture import (
    DatasetCaptureError,
    validate_dataset_capture_contract,
)
from noema_lab.core.recipes import load_recipe
from noema_lab.core.reproducibility import (
    derive_seed,
    master_seed_from_recipe,
    seed_namespace_from_recipe,
)
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.ops import build_registry
from noema_lab.training.delayed_csi_resource_allocation_export import (
    DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
    DelayedCsiResourceAllocationExportError,
    build_delayed_csi_resource_allocation_export_plan,
)
from noema_lab.training.exporter import (
    ExportOptions,
    export_differentiable_scenario,
    select_differentiable_exporter,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "resource_delayed_csi_finite_blocklength.yaml"


class DelayedCsiResourceAllocationExportTests(unittest.TestCase):
    def _plan(self, recipe=None):
        return build_delayed_csi_resource_allocation_export_plan(
            recipe or load_recipe(RECIPE),
            build_registry(),
            optimizable_steps=["tx_power"],
            loss=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
            framework="torch",
        )

    def test_plan_separates_runtime_csi_from_training_only_current_state(self):
        plan = self._plan()

        self.assertEqual(
            plan.delayed_csi_reference,
            "csi_observation.transmitter_csi",
        )
        self.assertEqual(
            plan.current_csi_reference,
            "csi_observation.actual_state",
        )
        self.assertEqual(plan.delayed_csi_tap, "csi_history")
        self.assertEqual(plan.current_csi_tap, "current_csi")
        self.assertEqual(plan.csi_history_length, 4)
        self.assertEqual(
            plan.allocator_step.op,
            "model.causal_csi_power_allocator",
        )
        self.assertEqual(plan.blocklength_channel_uses, 128)
        self.assertEqual(plan.target_rate_bps_hz, 2.0)
        self.assertEqual(
            plan.benchmark_power_budgets,
            (0.4, 0.6, 0.8, 1.0, 1.4),
        )
        self.assertEqual(plan.capture_plan.total_samples, 3072)
        roles = {tap.reference: tap.role for tap in plan.capture_plan.taps}
        self.assertEqual(
            roles[plan.delayed_csi_reference],
            "runtime_input_causal_complex_csi_history",
        )
        self.assertEqual(
            roles[plan.current_csi_reference],
            "training_only_auxiliary_current_channel_outcome",
        )
        self.assertNotIn(
            "tx_power.allocation",
            {tap.reference for tap in plan.capture_plan.taps},
        )

    def test_exporter_is_explicitly_selectable(self):
        recipe = load_recipe(RECIPE)
        for requested in ("delayed-csi-resource-allocation", "auto"):
            exporter = select_differentiable_exporter(
                recipe,
                build_registry(),
                ExportOptions(
                    optimizable_steps=["tx_power"],
                    loss=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
                    framework="torch",
                    exporter=requested,
                ),
            )
            self.assertEqual(exporter.id, "delayed-csi-resource-allocation")

    def test_demo_rejects_source_allocator_output_as_a_label(self):
        recipe = copy.deepcopy(load_recipe(RECIPE))
        recipe.dataset_capture["taps"].append(
            {"id": "allocation_label", "from": "tx_power.allocation"}
        )
        with self.assertRaisesRegex(
            DelayedCsiResourceAllocationExportError,
            "label-free",
        ):
            self._plan(recipe)

    def test_demo_requires_current_state_on_the_propagation_channel(self):
        recipe = copy.deepcopy(load_recipe(RECIPE))
        wireless = next(
            step for step in recipe.steps if step.id == "wireless_channel"
        )
        wireless.inputs["channel_state"] = "channel_state.state"
        with self.assertRaisesRegex(
            DelayedCsiResourceAllocationExportError,
            "propagation channel",
        ):
            self._plan(recipe)

    def test_capture_matrix_mode_is_validated(self):
        recipe = copy.deepcopy(load_recipe(RECIPE))
        recipe.dataset_capture["matrix_mode"] = "sometimes"
        with self.assertRaisesRegex(
            DatasetCaptureError,
            "matrix_mode must be one of inherit or exclude",
        ):
            validate_dataset_capture_contract(recipe, build_registry())

    def test_starter_contract_keeps_current_csi_out_of_runtime_abi(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            result = export_differentiable_scenario(
                load_recipe(RECIPE),
                build_registry(),
                optimizable_steps=["tx_power"],
                loss=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
                framework="torch",
                out_dir=bundle,
                project_root=ROOT,
                exporter="delayed-csi-resource-allocation",
                include_starter=True,
            )
            self.assertEqual(
                result["starter_exporter"],
                "delayed-csi-resource-allocation",
            )
            training_project = bundle / "reference_training"
            config = load_strict_yaml_or_json(
                training_project / "train_config.yaml"
            )
            contract = load_strict_yaml_or_json(
                training_project / "reliability_allocation_contract.yaml"
            )
            self.assertEqual(
                config["model"]["runtime_inputs"],
                [
                    "csi_history",
                    "noise_variance",
                    "average_power_budget",
                ],
            )
            self.assertEqual(config["model"]["history_length"], 4)
            self.assertFalse(
                config["objective"]["uses_current_csi_at_runtime"]
            )
            self.assertTrue(
                config["objective"]["uses_current_csi_for_training_loss"]
            )
            self.assertFalse(
                config["objective"]["uses_allocation_labels"]
            )
            self.assertEqual(
                config["training"]["average_power_budget_range"],
                [0.4, 1.4],
            )
            self.assertEqual(
                config["training"]["validation_average_power_budgets"],
                [0.4, 0.6, 0.8, 1.0, 1.4],
            )
            self.assertEqual(
                config["evaluation"]["average_power_budgets"],
                [0.4, 0.6, 0.8, 1.0, 1.4],
            )
            self.assertNotIn(
                "current_csi",
                set(contract["runtime_abi"]["inputs"]),
            )
            self.assertEqual(
                contract["runtime_abi"]["inputs"]["csi_history"]["shape"],
                ["batch", 4, 128, 2],
            )
            self.assertIn(
                "current_csi",
                contract["runtime_abi"]["excludes"],
            )
            self.assertEqual(
                config["evaluation"][
                    "current_csi_shannon_diagnostic_role"
                ],
                "diagnostic_only_not_finite_blocklength_optimum",
            )

    def test_prepare_example_adds_current_csi_only_to_generated_training_plan(self):
        source = copy.deepcopy(load_recipe(RECIPE))
        source.dataset_capture["taps"] = [
            {
                "id": "csi_history",
                "from": "csi_observation.transmitter_csi",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            export_differentiable_scenario(
                source,
                build_registry(),
                optimizable_steps=["tx_power"],
                loss="",
                training_objective=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
                framework="torch",
                out_dir=bundle,
                project_root=ROOT,
                include_starter=False,
            )
            frozen_source = load_strict_yaml_or_json(
                bundle / "noema_recipe.yaml"
            )
            self.assertIn("matrix", frozen_source["metadata"])
            derived_split_seeds = {}
            expected_counts = {
                "train": 2048,
                "validation": 512,
                "test": 512,
            }
            for split, count in expected_counts.items():
                capture_recipe = load_recipe(
                    bundle / ("capture_%s_recipe.yaml" % split)
                )
                self.assertNotIn("matrix", capture_recipe.metadata)
                self.assertEqual(
                    capture_recipe.dataset_capture["matrix_mode"],
                    "exclude",
                )
                self.assertEqual(
                    capture_recipe.dataset_capture["seed_mode"],
                    "increment_run_seed",
                )
                self.assertEqual(
                    capture_recipe.dataset_capture["max_runs"],
                    count,
                )
                for step_id in ("channel_state", "csi_observation"):
                    step = next(
                        step
                        for step in capture_recipe.steps
                        if step.id == step_id
                    )
                    self.assertNotIn("seed", step.params)
                master_seed = master_seed_from_recipe(capture_recipe)
                self.assertIsNotNone(master_seed)
                namespace = "%s|dataset_capture_split=%s" % (
                    seed_namespace_from_recipe(capture_recipe),
                    split,
                )
                derived_split_seeds[split] = (
                    derive_seed(
                        int(master_seed),
                        namespace,
                        "channel_state",
                        "ofdm_channel_state",
                    ),
                    derive_seed(
                        int(master_seed),
                        namespace,
                        "csi_observation",
                        "ofdm_delayed_csi",
                    ),
                )
            self.assertEqual(len(set(derived_split_seeds.values())), 3)

            recipe, training_plan = prepare_example._load_planned_recipe(bundle)
            enriched_recipe, enriched_plan, additions = (
                prepare_example._ensure_delayed_csi_allocation_demo_capture_plan(
                    bundle,
                    recipe,
                    training_plan,
                    registry=build_registry(),
                    project_root=ROOT,
                )
            )

            self.assertEqual(
                additions,
                ("its aligned training-only current-CSI outcome",),
            )
            self.assertEqual(enriched_recipe.name, source.name)
            self.assertIn(
                "csi_observation.actual_state",
                {
                    str(item.get("from") or "")
                    for item in enriched_plan.dataset_capture["taps"]
                },
            )
            self.assertNotIn("dataset_capture", frozen_source)
            data_contract = load_strict_yaml_or_json(bundle / "data_contract.yaml")
            self.assertIn(
                "csi_observation.actual_state",
                {
                    str(item.get("reference") or "")
                    for item in data_contract["signals"]
                },
            )


if __name__ == "__main__":
    unittest.main()
