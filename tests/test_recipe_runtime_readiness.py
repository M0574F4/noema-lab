from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from noema_lab.core.operations import OperationError
from noema_lab.core.params import validate_params
from noema_lab.core.recipe_templates import inspect_recipe_template_catalog
from noema_lab.core.recipes import load_recipe
from noema_lab.core.runtime_readiness import inspect_recipe_run_readiness
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


class RecipeRuntimeReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_export_only_training_interface_is_not_reported_as_benchmark_ready(self):
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")

        readiness = inspect_recipe_run_readiness(recipe, self.registry)

        self.assertFalse(readiness["runnable"])
        self.assertEqual(readiness["role"], "training_starter")
        self.assertTrue(any("export-only training interface" in issue for issue in readiness["issues"]))

    def test_contract_smoke_and_explicit_aoa_have_distinct_roles(self):
        smoke = inspect_recipe_run_readiness(
            load_recipe(ROOT / "recipes" / "task_classification_smoke.yaml"),
            self.registry,
        )
        aoa = inspect_recipe_run_readiness(
            load_recipe(ROOT / "recipes" / "aoa_music_ula_baseline.yaml"),
            self.registry,
        )

        self.assertTrue(smoke["runnable"])
        self.assertEqual(smoke["role"], "contract_smoke")
        self.assertEqual(aoa, {"runnable": True, "role": "runnable_template", "issues": []})

    def test_packaged_template_inspection_includes_run_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            inspection = inspect_recipe_template_catalog(Path(tmp), self.registry).to_dict()

        aoa = next(
            row for row in inspection["templates"]
            if row["id"] == "ai_phy.aoa_estimation.adapter"
        )
        self.assertTrue(aoa["run_readiness"]["runnable"])
        self.assertEqual(aoa["run_readiness"]["issues"], [])

    def test_promoted_topology_starters_have_explicit_setup_readiness(self):
        missing_wireless = {
            "available": False,
            "extra": "wireless",
            "missing": ["sionna"],
            "reason": "Install optional dependencies with `uv sync --extra wireless`",
        }
        missing_textgen = {
            "available": False,
            "extra": "textgen",
            "missing": ["transformers"],
            "reason": "Install optional dependencies with `uv sync --extra textgen`",
        }
        with (
            mock.patch.object(
                self.registry.get("wireless.miso_ofdm_csi"),
                "runtime_availability",
                return_value=missing_wireless,
            ),
            mock.patch(
                "noema_lab.ops.channel.digital._sionna_availability",
                return_value=missing_wireless,
            ),
            mock.patch(
                "noema_lab.ops.models.text_codec._transformers_availability",
                return_value=missing_textgen,
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            rows = {
                row["id"]: row
                for row in inspect_recipe_template_catalog(
                    Path(tmp), self.registry
                ).to_dict()["templates"]
            }

        core_ready = (
            "semantic_comm.text_semantic_similarity.semantic_state_kb",
            "ai_phy.automatic_modulation_recognition.awgn",
        )
        for template_id in core_ready:
            with self.subTest(template_id=template_id):
                self.assertEqual(
                    rows[template_id]["run_readiness"],
                    {"runnable": True, "role": "runnable_template", "issues": []},
                )

        setup_required = {
            "ai_phy.csi_compression_feedback.truncated_angular_delay": "wireless",
            "ai_phy.resource_allocation.equal_power": "wireless",
            "ai_phy.resource_allocation.delayed_csi_finite_blocklength": "wireless",
            "semantic_comm.image_reconstruction.protected_digital": "wireless",
            "ai_phy.neural_receiver_demapping.adapter": "wireless",
            "ai_phy.neural_receiver_demapping.phase_tracking": "wireless",
            "semantic_comm.text_semantic_similarity.bart_joint_symbols": "textgen",
        }
        for template_id, extra in setup_required.items():
            with self.subTest(template_id=template_id):
                readiness = rows[template_id]["run_readiness"]
                self.assertFalse(readiness["runnable"])
                self.assertEqual(readiness["role"], "runnable_template")
                self.assertTrue(any(extra in issue for issue in readiness["issues"]))

    def test_parameter_selected_masked_lm_dependency_is_reported(self):
        recipe = load_recipe(ROOT / "recipes" / "text_semantic_utf8_mask_repair.yaml")

        with mock.patch("noema_lab.ops.foundation.importlib.util.find_spec", return_value=None):
            readiness = inspect_recipe_run_readiness(recipe, self.registry)

        self.assertFalse(readiness["runnable"])
        self.assertTrue(any("textgen" in issue for issue in readiness["issues"]))

    def test_clip_retrieval_stays_unavailable_when_only_data_extra_is_present(self):
        recipe = load_recipe(ROOT / "recipes" / "clip_retrieval_smoke.yaml")
        data_available = {
            "available": True,
            "extra": "retrieval-data",
            "missing": [],
        }

        with (
            mock.patch(
                "noema_lab.ops.source.retrieval_flickr8k._flickr8k_availability",
                return_value=data_available,
            ),
            mock.patch(
                "noema_lab.ops.foundation.importlib.util.find_spec",
                side_effect=lambda module: object() if module == "torch" else None,
            ),
        ):
            readiness = inspect_recipe_run_readiness(recipe, self.registry)

        self.assertFalse(readiness["runnable"])
        self.assertEqual(len(readiness["issues"]), 2)
        self.assertTrue(any(issue.startswith("image_encoder:") for issue in readiness["issues"]))
        self.assertTrue(any(issue.startswith("text_encoder:") for issue in readiness["issues"]))
        self.assertTrue(all("foundation" in issue for issue in readiness["issues"]))
        self.assertTrue(all("transformers" in issue for issue in readiness["issues"]))

    def test_image_generation_reports_diffusion_and_clip_foundation_dependencies(self):
        recipe = load_recipe(ROOT / "recipes" / "diffusion_flickr8k_generation.yaml")
        data_available = {
            "available": True,
            "extra": "retrieval-data",
            "missing": [],
        }

        with (
            mock.patch(
                "noema_lab.ops.source.retrieval_flickr8k._flickr8k_availability",
                return_value=data_available,
            ),
            mock.patch(
                "noema_lab.ops.foundation.importlib.util.find_spec",
                side_effect=lambda module: object() if module == "torch" else None,
            ),
        ):
            readiness = inspect_recipe_run_readiness(recipe, self.registry)

        self.assertFalse(readiness["runnable"])
        self.assertEqual(len(readiness["issues"]), 4)
        receiver_issue = next(
            issue for issue in readiness["issues"] if issue.startswith("receiver:")
        )
        self.assertIn("transformers", receiver_issue)
        self.assertIn("diffusers", receiver_issue)
        for step_id in ("text_encoder", "source_image_encoder", "generated_image_encoder"):
            self.assertTrue(
                any(issue.startswith(step_id + ":") for issue in readiness["issues"])
            )

    def test_diffusion_safety_checker_can_only_be_disabled_explicitly(self):
        operation = self.registry.get("foundation.diffusion_state_to_image")
        self.assertFalse(
            operation.params_schema["properties"]["disable_safety_checker"]["default"]
        )

        for recipe_path in (
            ROOT / "recipes" / "diffusion_flickr8k_generation.yaml",
            ROOT
            / "src"
            / "noema_lab"
            / "recipe_starters"
            / "diffusion_flickr8k_generation.yaml",
        ):
            recipe = load_recipe(recipe_path)
            receiver = next(
                step
                for step in recipe.steps
                if step.op == "foundation.diffusion_state_to_image"
            )
            self.assertIs(receiver.params["disable_safety_checker"], False)

    def test_foundation_availability_preserves_local_and_installed_backends(self):
        clip_text = self.registry.get("foundation.clip_text_embed")
        clip_image = self.registry.get("foundation.clip_image_embed")
        diffusion = self.registry.get("foundation.diffusion_state_to_image")

        with mock.patch(
            "noema_lab.ops.foundation.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertTrue(clip_text.runtime_availability({"backend": "local_hash"})["available"])
            self.assertTrue(
                clip_image.runtime_availability({"backend": "local_color_histogram"})["available"]
            )
            self.assertFalse(
                clip_text.runtime_availability({"backend": "transformers_clip"})["available"]
            )
            self.assertFalse(diffusion.runtime_availability({"backend": "diffusers"})["available"])

        with mock.patch(
            "noema_lab.ops.foundation.importlib.util.find_spec",
            return_value=object(),
        ):
            self.assertTrue(
                clip_text.runtime_availability({"backend": "transformers_clip"})["available"]
            )
            self.assertTrue(
                clip_image.runtime_availability({"backend": "transformers_clip"})["available"]
            )
            self.assertTrue(diffusion.runtime_availability({"backend": "diffusers"})["available"])

    def test_clip_text_backend_schema_only_advertises_runnable_branches(self):
        clip_text = self.registry.get("foundation.clip_text_embed")
        backend_schema = clip_text.params_schema["properties"]["backend"]

        self.assertEqual(
            backend_schema["enum"],
            ["local_semantic", "local_hash", "transformers_clip"],
        )
        for backend in backend_schema["enum"]:
            validate_params(
                clip_text.id,
                {"backend": backend},
                clip_text.params_schema,
            )
        with mock.patch(
            "noema_lab.ops.foundation.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertTrue(
                clip_text.runtime_availability({"backend": "local_semantic"})["available"]
            )
            self.assertTrue(
                clip_text.runtime_availability({"backend": "local_hash"})["available"]
            )
            self.assertFalse(
                clip_text.runtime_availability({"backend": "transformers_clip"})["available"]
            )
        with self.assertRaisesRegex(OperationError, "must be one of"):
            validate_params(
                clip_text.id,
                {"backend": "sentence_transformers"},
                clip_text.params_schema,
            )


if __name__ == "__main__":
    unittest.main()
