from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.training_plans import TrainingPlan, apply_training_plan
from noema_lab.ops import build_registry
from noema_lab.training.capture_plan import allocate_split_counts
from noema_lab.training.generic_capture_export import (
    GenericCaptureDataContractError,
    write_generic_capture_data_contract,
)
from noema_lab.training.exporter import (
    DifferentiableExportError,
    export_differentiable_scenario,
    inspect_training_capture,
)


ROOT = Path(__file__).resolve().parents[1]


def recipe_with_capture(path: Path, capture: dict):
    return apply_training_plan(load_recipe(path), TrainingPlan(dataset_capture=capture))


class TrainingCapturePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_largest_remainder_split_is_exact_and_deterministic(self):
        self.assertEqual(
            allocate_split_counts(
                10, {"train": 50, "validation": 25, "test": 25}
            ),
            {"train": 5, "validation": 3, "test": 2},
        )

    def test_catalog_templates_never_disable_optional_signals_by_training_method(self):
        catalog = yaml.safe_load(
            (ROOT / "src" / "noema_lab" / "recipe_templates.yaml").read_text(
                encoding="utf-8"
            )
        )
        inspected = 0
        for template in catalog["templates"]:
            recipe = load_recipe(ROOT / template["recipe_path"])
            for step in recipe.steps:
                operation = self.registry.get(step.op).describe()
                capabilities = dict(operation.get("training_capabilities") or {})
                if not capabilities.get("portable_replacement"):
                    continue
                inspected += 1
                payload = inspect_training_capture(
                    recipe,
                    self.registry,
                    optimizable_steps=[step.id],
                    project_root=ROOT,
                )
                disabled = [
                    item["from"]
                    for item in payload.get("candidates") or []
                    if not item.get("selectable")
                ]
                self.assertEqual(
                    disabled,
                    [],
                    "%s.%s disabled signals based on inferred training intent"
                    % (template["id"], step.id),
                )
        self.assertGreater(inspected, 0)

    def test_resource_inspection_exposes_generic_candidates_and_abi_input(self):
        recipe = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")
        payload = inspect_training_capture(
            recipe,
            self.registry,
            optimizable_steps=["tx_power"],
            project_root=ROOT,
        )
        self.assertTrue(payload["ready"], payload["issue"])
        self.assertEqual(payload["mode"], "captured_tensors")
        self.assertEqual(payload["sample_unit"], "recipe records")
        self.assertEqual(payload["total_samples"], 1000)
        self.assertEqual(
            payload["split_plan"]["counts"],
            {"train": 667, "validation": 167, "test": 166},
        )
        self.assertEqual(
            payload["required_taps"],
            [
                {
                    "id": "channel_state_state",
                    "from": "channel_state.state",
                    "role": "replacement_input:tx_power.channel_state",
                }
            ],
        )
        references = {item["from"] for item in payload["candidates"]}
        self.assertIn("channel_state.state", references)
        self.assertIn("data.bits", references)
        candidates = {item["from"]: item for item in payload["candidates"]}
        self.assertTrue(candidates["tx_power.allocation"]["selectable"])
        self.assertTrue(candidates["tx_power.symbols"]["selectable"])
        self.assertTrue(candidates["demodulator.bits"]["selectable"])
        self.assertTrue(candidates["channel_decoder.bits"]["selectable"])
        self.assertEqual(
            candidates["tx_power.allocation"]["relationship"],
            "current_replacement_output",
        )
        self.assertEqual(
            candidates["channel_decoder.bits"]["relationship"],
            "current_pipeline_dependent",
        )
        self.assertTrue(candidates["data.bits"]["selectable"])

    def test_export_freezes_user_selected_optional_signals_and_split_plan(self):
        recipe = recipe_with_capture(
            ROOT / "recipes" / "resource_water_filling_baseline.yaml",
            {
                "samples": 10,
                "split_plan": {
                    "total_samples": 10,
                    "percentages": {
                        "train": 50,
                        "validation": 25,
                        "test": 25,
                    },
                },
                "taps": [
                    {"id": "channel_gains", "from": "channel_state.state"},
                    {"id": "source_bits", "from": "data.bits"},
                ],
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "contract"
            payload = export_differentiable_scenario(
                recipe,
                self.registry,
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=out,
                project_root=Path(temporary),
            )
            self.assertEqual(
                [job["requested_samples"] for job in payload["capture_jobs"]],
                [5, 3, 2],
            )
            for split in ("train", "validation", "test"):
                capture = yaml.safe_load(
                    (out / ("capture_%s_recipe.yaml" % split)).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    capture["dataset_capture"]["taps"],
                    [
                        {"id": "channel_gains", "from": "channel_state.state"},
                        {"id": "source_bits", "from": "data.bits"},
                    ],
                )
                self.assertEqual(
                    capture["dataset_capture"]["max_runs"],
                    capture["dataset_capture"]["samples"],
                )
                self.assertEqual(
                    capture["execution_profile"],
                    {
                        "id": "custom",
                        "version": 1,
                        "based_on": {"id": "layered_digital", "version": 1},
                    },
                )
                validate_recipe_against_registry(
                    recipe_from_dict(capture), self.registry
                )
            data_contract = yaml.safe_load(
                (out / "data_contract.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                data_contract["capture_plan"]["split_plan"]["counts"],
                {"train": 5, "validation": 3, "test": 2},
            )

    def test_required_signal_cannot_be_deselected(self):
        recipe = recipe_with_capture(
            ROOT / "recipes" / "resource_water_filling_baseline.yaml",
            {"taps": [{"id": "source_bits", "from": "data.bits"}]},
        )
        inspection = inspect_training_capture(
            recipe, self.registry, optimizable_steps=["tx_power"]
        )
        self.assertFalse(inspection["ready"])
        self.assertIn("channel_state.state", inspection["issue"])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                DifferentiableExportError, "omit required training-contract signal"
            ):
                export_differentiable_scenario(
                    recipe,
                    self.registry,
                    optimizable_steps=["tx_power"],
                    loss="researcher.defined",
                    framework="torch",
                    out_dir=Path(temporary) / "contract",
                )

    def test_current_pipeline_signal_is_selectable_for_researcher_defined_use(self):
        recipe = recipe_with_capture(
            ROOT / "recipes" / "resource_water_filling_baseline.yaml",
            {
                "taps": [
                    {"id": "channel_gains", "from": "channel_state.state"},
                    {"id": "powered_symbols", "from": "tx_power.symbols"},
                ]
            },
        )
        inspection = inspect_training_capture(
            recipe, self.registry, optimizable_steps=["tx_power"]
        )
        self.assertTrue(inspection["ready"], inspection["issue"])
        candidate = next(
            item
            for item in inspection["candidates"]
            if item["from"] == "tx_power.symbols"
        )
        self.assertTrue(candidate["selectable"])
        self.assertEqual(candidate["relationship"], "current_replacement_output")
        with tempfile.TemporaryDirectory() as temporary:
            exported = export_differentiable_scenario(
                recipe,
                self.registry,
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=Path(temporary) / "contract",
            )
            signals = exported["data_contract"]["signals"]
            self.assertIn("tx_power.symbols", {item["reference"] for item in signals})

    def test_generic_capture_can_accumulate_multiple_seeded_recipe_runs(self):
        recipe = recipe_with_capture(
            ROOT / "recipes" / "resource_water_filling_baseline.yaml",
            {
                "taps": [{"id": "channel_gains", "from": "channel_state.state"}],
                "split_plan": {
                    "total_samples": 900,
                    "percentages": {"train": 60, "validation": 20, "test": 20},
                },
            },
        )
        inspection = inspect_training_capture(
            recipe, self.registry, optimizable_steps=["tx_power"]
        )
        self.assertTrue(inspection["ready"], inspection["issue"])
        self.assertEqual(
            inspection["split_plan"]["counts"],
            {"train": 540, "validation": 180, "test": 180},
        )
        with tempfile.TemporaryDirectory() as temporary:
            exported = export_differentiable_scenario(
                recipe,
                self.registry,
                optimizable_steps=["tx_power"],
                loss="researcher.defined",
                framework="torch",
                out_dir=Path(temporary) / "contract",
            )
            self.assertEqual(
                [job["requested_samples"] for job in exported["capture_jobs"]],
                [540, 180, 180],
            )

    def test_neural_receiver_requires_only_its_runtime_abi_input(self):
        recipe = load_recipe(
            ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml"
        )
        payload = inspect_training_capture(
            recipe,
            self.registry,
            optimizable_steps=["demodulator"],
            project_root=ROOT,
        )
        self.assertTrue(payload["ready"], payload["issue"])
        self.assertEqual(payload["sample_unit"], "recipe records")
        self.assertEqual(
            [item["from"] for item in payload["required_taps"]],
            ["wireless_channel.rx_symbols"],
        )
        candidates = {item["from"]: item for item in payload["candidates"]}
        self.assertTrue(candidates["tx_bit_boundary.bits"]["selectable"])
        self.assertFalse(candidates["tx_bit_boundary.bits"]["required"])

    def test_generic_export_rejects_sweep_target_pruned_from_capture_graph(self):
        recipe = recipe_with_capture(
            ROOT / "recipes" / "neural_receiver_qpsk_awgn_adapter.yaml",
            {
                "taps": [
                    {
                        "id": "rx_symbols",
                        "from": "wireless_channel.rx_symbols",
                    }
                ],
                "sweep": {"coded_bler.block_size": [128, 256]},
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "contract"
            with self.assertRaisesRegex(
                GenericCaptureDataContractError,
                r"selected-tap ancestor graph.*coded_bler\.block_size",
            ):
                write_generic_capture_data_contract(
                    recipe,
                    self.registry,
                    selected_step_ids=["demodulator"],
                    out_dir=out,
                    project_root=Path(temporary),
                )
            self.assertFalse(out.exists())

    def test_deepjscc_inspection_does_not_choose_a_data_partition(self):
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        payload = inspect_training_capture(
            recipe,
            self.registry,
            optimizable_steps=["sender", "receiver"],
            project_root=ROOT,
        )
        self.assertEqual(payload["mode"], "not_required")
        self.assertFalse(payload["capture_runner_required"])
        self.assertNotIn("split_plan", payload)
        self.assertIn("optional", payload["explanation"])

    def test_capture_inspection_does_not_validate_an_unselected_data_method(self):
        source = load_recipe(
            ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"
        ).to_dict()
        source_step = next(step for step in source["steps"] if step["id"] == "data")
        source_step["params"]["image_ids"] = "missing_image"
        recipe = recipe_from_dict(source)
        payload = inspect_training_capture(
            recipe,
            self.registry,
            optimizable_steps=["sender", "receiver"],
            project_root=ROOT,
        )
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["mode"], "not_required")
        self.assertEqual(payload["issue"], "")


if __name__ == "__main__":
    unittest.main()
