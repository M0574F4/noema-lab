from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.execution_profiles import (
    AOA_ARRAY_ESTIMATION_PROFILE_ID,
    BEAMFORMING_LINK_EVALUATION_PROFILE_ID,
    CSI_FEEDBACK_DOWNLINK_PROFILE_ID,
    CUSTOM_EXECUTION_PROFILE_ID,
    JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
    LAYERED_DIGITAL_PROFILE_ID,
    LEO_NTN_TRACKING_PROFILE_ID,
    ISAC_OFDM_ALLOCATION_PROFILE_ID,
    MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID,
    PILOT_CHANNEL_ESTIMATION_PROFILE_ID,
    NEAR_FIELD_RANGE_ANGLE_PROFILE_ID,
    RANGE_LOCALIZATION_PROFILE_ID,
    TASK_EVALUATION_PROFILE_ID,
    TASK_INFERENCE_PROFILE_ID,
    execution_profile_catalog,
    inspect_execution_profile,
)
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.planner import RecipePlanningError, validate_recipe_against_registry
from noema_lab.core.recipes import RecipeValidationError, load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


class ExecutionProfileTests(unittest.TestCase):
    def test_missing_declaration_normalizes_to_explicit_custom_ref(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unclassified",
                "steps": [{"id": "data", "op": "source.synthetic_images"}],
            }
        )

        self.assertEqual(recipe.execution_profile.id, CUSTOM_EXECUTION_PROFILE_ID)
        self.assertEqual(
            recipe.to_dict()["execution_profile"],
            {"id": CUSTOM_EXECUTION_PROFILE_ID, "version": 1},
        )
        self.assertEqual(inspect_execution_profile(recipe).status, "custom")

    def test_custom_ref_preserves_its_standard_origin(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "edited_layered_recipe",
                "execution_profile": {
                    "id": "custom",
                    "version": 1,
                    "based_on": {"id": "layered_digital", "version": 1},
                },
                "steps": [{"id": "data", "op": "source.synthetic_images"}],
            }
        )

        self.assertEqual(recipe.execution_profile.based_on.id, LAYERED_DIGITAL_PROFILE_ID)
        self.assertEqual(
            recipe.to_dict()["execution_profile"]["based_on"],
            {"id": LAYERED_DIGITAL_PROFILE_ID, "version": 1},
        )

    def test_unknown_or_malformed_declarations_are_rejected(self):
        base = {
            "schema_version": 1,
            "name": "bad_profile",
            "steps": [{"id": "data", "op": "source.synthetic_images"}],
        }
        for declaration in (
            "layered_digital",
            {"id": "missing", "version": 1},
            {"id": "layered_digital", "version": 0},
            {"id": "layered_digital", "version": 2},
            {"id": "layered_digital", "version": 1, "based_on": {"id": "layered_digital"}},
            {"id": "custom", "version": 1, "based_on": {"id": "missing", "version": 1}},
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaises(RecipeValidationError):
                    recipe_from_dict({**base, "execution_profile": declaration})

    def test_catalog_exposes_standard_profiles_and_fused_jscc_stages(self):
        payload = execution_profile_catalog().to_dict()
        profiles = {profile["id"]: profile for profile in payload["profiles"]}

        self.assertEqual(
            set(profiles),
            {
                LAYERED_DIGITAL_PROFILE_ID,
                JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
                CSI_FEEDBACK_DOWNLINK_PROFILE_ID,
                PILOT_CHANNEL_ESTIMATION_PROFILE_ID,
                MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID,
                BEAMFORMING_LINK_EVALUATION_PROFILE_ID,
                RANGE_LOCALIZATION_PROFILE_ID,
                AOA_ARRAY_ESTIMATION_PROFILE_ID,
                ISAC_OFDM_ALLOCATION_PROFILE_ID,
                NEAR_FIELD_RANGE_ANGLE_PROFILE_ID,
                LEO_NTN_TRACKING_PROFILE_ID,
                TASK_INFERENCE_PROFILE_ID,
                TASK_EVALUATION_PROFILE_ID,
            },
        )
        self.assertEqual(
            profiles[LAYERED_DIGITAL_PROFILE_ID]["required_ordered_spine"],
            [
                "payload_bit_boundary",
                "channel_encoder",
                "tx_bit_boundary",
                "wireless_channel",
                "rx_bit_boundary",
                "channel_bit_count_match",
                "channel_decoder",
            ],
        )
        fused = {
            stage["role"]: stage["fused_into"]
            for stage in profiles[JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID]["stages"]
            if stage["presence"] == "fused"
        }
        self.assertEqual(fused["channel_encoder"], "sender")
        self.assertEqual(fused["demodulator"], "receiver")
        self.assertEqual(
            profiles[CSI_FEEDBACK_DOWNLINK_PROFILE_ID]["required_ordered_spine"],
            [
                "channel_state",
                "feedback_encoder",
                "feedback_link",
                "feedback_decoder",
                "precoder",
                "evaluation",
            ],
        )
        self.assertEqual(
            profiles[TASK_EVALUATION_PROFILE_ID]["required_ordered_spine"],
            ["source", "evaluation"],
        )
        self.assertEqual(
            profiles[PILOT_CHANNEL_ESTIMATION_PROFILE_ID]["required_ordered_spine"],
            [
                "channel_state",
                "pilot_pattern",
                "pilot_observation",
                "channel_estimator",
                "evaluation",
            ],
        )
        self.assertEqual(
            profiles[MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID]["required_ordered_spine"],
            [
                "mimo_ofdm_channel",
                "pilot_grid",
                "pilot_observation",
                "channel_estimator",
                "evaluation",
            ],
        )
        mimo_stages = {
            stage["role"]: stage
            for stage in profiles[MIMO_OFDM_CHANNEL_ESTIMATION_PROFILE_ID]["stages"]
        }
        self.assertEqual(
            mimo_stages["mimo_ofdm_channel"]["required_params"],
            {"scenario": "mimo_ofdm"},
        )
        self.assertEqual(
            mimo_stages["pilot_grid"]["required_params"],
            {"scenario": "comb"},
        )
        self.assertEqual(
            profiles[BEAMFORMING_LINK_EVALUATION_PROFILE_ID]["required_ordered_spine"],
            ["link_scenario", "beamformer", "evaluation"],
        )
        self.assertEqual(
            profiles[RANGE_LOCALIZATION_PROFILE_ID]["required_ordered_spine"],
            ["geometry", "range_observation", "localizer", "evaluation"],
        )
        self.assertEqual(
            profiles[AOA_ARRAY_ESTIMATION_PROFILE_ID]["required_ordered_spine"],
            ["angular_scene", "array_observation", "angle_estimator", "evaluation"],
        )
        self.assertEqual(
            profiles[TASK_INFERENCE_PROFILE_ID]["required_ordered_spine"],
            ["source", "inference", "evaluation"],
        )

    def test_task_evaluation_recipe_conforms(self):
        recipe = load_recipe(ROOT / "recipes" / "task_classification_smoke.yaml")

        inspection = inspect_execution_profile(recipe)

        self.assertEqual(inspection.status, "conformant")
        self.assertEqual(
            dict(inspection.stage_bindings),
            {"source": "data", "evaluation": "evaluation"},
        )

    def test_domain_profiles_bind_their_semantic_spines(self):
        cases = {
            "channel_estimation_ls_awgn.yaml": {
                "channel_state": "data",
                "pilot_pattern": "pilots",
                "pilot_observation": "pilot_observation",
                "channel_estimator": "estimator",
                "evaluation": "evaluation",
            },
            "mimo_ofdm_ls_channel_estimation.yaml": {
                "mimo_ofdm_channel": "data",
                "pilot_grid": "pilots",
                "pilot_observation": "pilot_observation",
                "channel_estimator": "estimator",
                "evaluation": "evaluation",
            },
            "beamforming_mrt_baseline.yaml": {
                "link_scenario": "data",
                "beamformer": "beamformer",
                "evaluation": "evaluation",
            },
            "localization_trilateration_baseline.yaml": {
                "geometry": "data",
                "range_observation": "range_observation",
                "localizer": "localizer",
                "evaluation": "evaluation",
            },
            "aoa_music_ula_baseline.yaml": {
                "angular_scene": "data",
                "array_observation": "array_observation",
                "angle_estimator": "estimator",
                "evaluation": "evaluation",
            },
            "vqa_pretrained_transformers_smoke.yaml": {
                "source": "data",
                "inference": "receiver",
                "evaluation": "evaluation",
            },
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                inspection = inspect_execution_profile(
                    load_recipe(ROOT / "recipes" / filename)
                )
                self.assertEqual(inspection.status, "conformant", inspection.issues)
                self.assertEqual(dict(inspection.stage_bindings), expected)

    def test_domain_profile_rejects_an_incompatible_stage_operation(self):
        payload = load_recipe(
            ROOT / "recipes" / "localization_trilateration_baseline.yaml"
        ).to_dict()
        next(step for step in payload["steps"] if step["id"] == "data")["op"] = (
            "source.aoa_scene"
        )
        inspection = inspect_execution_profile(recipe_from_dict(payload))

        self.assertEqual(inspection.status, "invalid")
        self.assertIn(
            "required_step_op_invalid",
            {issue.code for issue in inspection.issues},
        )

    def test_mimo_profile_rejects_a_flat_siso_scenario(self):
        payload = load_recipe(
            ROOT / "recipes" / "mimo_ofdm_ls_channel_estimation.yaml"
        ).to_dict()
        next(step for step in payload["steps"] if step["id"] == "data")["params"][
            "scenario"
        ] = "flat_siso"
        inspection = inspect_execution_profile(recipe_from_dict(payload))

        self.assertEqual(inspection.status, "invalid")
        self.assertIn(
            "required_step_param_invalid",
            {issue.code for issue in inspection.issues},
        )

    def test_existing_layered_recipe_conforms_when_declared(self):
        payload = load_recipe(ROOT / "recipes" / "compressai_kodak_default.yaml").to_dict()
        payload["execution_profile"] = {"id": LAYERED_DIGITAL_PROFILE_ID, "version": 1}
        recipe = recipe_from_dict(payload)

        inspection = inspect_execution_profile(recipe)
        self.assertEqual(inspection.status, "conformant")
        self.assertEqual(
            dict(inspection.stage_bindings)["wireless_channel"],
            "wireless_channel",
        )

        report = lint_recipe_invariants(recipe, build_registry())
        self.assertEqual(report["execution_profile"]["status"], "conformant")
        self.assertFalse(
            [issue for issue in report["issues"] if issue["code"].startswith("execution_profile_")]
        )

    def test_existing_jscc_recipe_conforms_when_declared(self):
        payload = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict()
        payload["execution_profile"] = {
            "id": JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
            "version": 1,
        }
        recipe = recipe_from_dict(payload)

        inspection = inspect_execution_profile(recipe)
        self.assertEqual(inspection.status, "conformant")
        self.assertEqual(dict(inspection.stage_bindings)["tx_power"], "tx_power")

        report = lint_recipe_invariants(recipe, build_registry())
        self.assertEqual(report["execution_profile"]["status"], "conformant")

    def test_declared_standard_profile_reports_missing_required_stage(self):
        payload = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict()
        payload["execution_profile"] = {
            "id": JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
            "version": 1,
        }
        payload["steps"] = [
            step for step in payload["steps"] if step["id"] != "channel_symbol_count_match"
        ]
        recipe = recipe_from_dict(payload)

        inspection = inspect_execution_profile(recipe)
        self.assertEqual(inspection.status, "invalid")
        self.assertIn(
            "required_step_missing",
            {issue.code for issue in inspection.issues},
        )
        report = lint_recipe_invariants(recipe, build_registry())
        self.assertEqual(report["execution_profile"]["status"], "invalid")
        self.assertIn(
            "required_step_missing",
            {issue["code"] for issue in report["issues"]},
        )

    def test_declared_standard_profile_blocks_planning_and_execution(self):
        payload = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict()
        payload["steps"] = [
            step for step in payload["steps"] if step["id"] != "channel_symbol_count_match"
        ]
        recipe = recipe_from_dict(payload)
        registry = build_registry()

        with self.assertRaisesRegex(RecipePlanningError, "required_step_missing"):
            validate_recipe_against_registry(recipe, registry)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaisesRegex(RecipePlanningError, "required_step_missing"):
                LocalExecutor(registry, LocalStore(workspace)).run(recipe)
            self.assertFalse((workspace / "runs").exists())

    def test_custom_profile_based_on_standard_remains_plannable(self):
        payload = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict()
        payload["execution_profile"] = {
            "id": CUSTOM_EXECUTION_PROFILE_ID,
            "version": 1,
            "based_on": {"id": JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID, "version": 1},
        }
        payload["steps"] = [
            step for step in payload["steps"] if step["id"] != "channel_symbol_count_match"
        ]
        recipe = recipe_from_dict(payload)

        validate_recipe_against_registry(recipe, build_registry())
        self.assertEqual(inspect_execution_profile(recipe).status, "custom")

    def test_joint_profile_rejects_separately_exposed_digital_stages(self):
        payload = load_recipe(ROOT / "recipes" / "compressai_kodak_default.yaml").to_dict()
        payload["execution_profile"] = {
            "id": JOINT_SOURCE_CHANNEL_SYMBOLS_PROFILE_ID,
            "version": 1,
        }
        recipe = recipe_from_dict(payload)

        inspection = inspect_execution_profile(recipe)
        self.assertEqual(inspection.status, "invalid")
        self.assertIn(
            "execution_profile_fused_stage_exposed",
            {issue.code for issue in inspection.issues},
        )


if __name__ == "__main__":
    unittest.main()
