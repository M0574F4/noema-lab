from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from noema_lab.core.recipe_templates import (
    RecipeTemplateCatalogError,
    RecipeTemplateInstantiationError,
    inspect_recipe_template_catalog,
    instantiate_recipe_template,
    load_recipe_template_catalog,
    recipe_template_catalog_from_dict,
)
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


def _template_entry(**overrides):
    entry = {
        "id": "test.classification.default",
        "label": "Classification",
        "task_id": "classification",
        "editor": "task",
        "recipe_path": "recipes/task_classification_smoke.yaml",
        "order": 10,
        "default": True,
        "status": "supported",
        "execution_profile": {"id": "custom", "version": 1},
    }
    entry.update(overrides)
    return entry


def _catalog_payload(*entries):
    return {"schema_version": 1, "templates": list(entries)}


class RecipeTemplateCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_built_in_catalog_has_one_ordered_default_for_each_task(self) -> None:
        catalog = load_recipe_template_catalog()
        expected_template_tasks = [
            "image_reconstruction",
            "image_reconstruction",
            "image_reconstruction",
            "image_reconstruction",
            "text_semantic_similarity",
            "text_semantic_similarity",
            "text_semantic_similarity",
            "visual_question_answering",
            "object_detection",
            "segmentation",
            "image_text_retrieval",
            "image_generation",
            "pilot_channel_estimation",
            "mimo_ofdm_channel_estimation",
            "csi_compression_feedback",
            "beamforming_precoding",
            "wireless_localization",
            "aoa_estimation",
            "resource_allocation",
            "resource_allocation",
            "neural_receiver_demapping",
            "neural_receiver_demapping",
            "automatic_modulation_recognition",
        ]

        self.assertEqual(
            [item.task_id for item in catalog.ordered_templates()],
            expected_template_tasks,
        )
        self.assertEqual(len(catalog.templates), len(expected_template_tasks))
        for task_id in dict.fromkeys(expected_template_tasks):
            default = catalog.default_for_task(task_id)
            self.assertIsNotNone(default)
            self.assertTrue(default.default)
            self.assertEqual(default.status, "supported")
            self.assertTrue(default.label)
            self.assertTrue(default.editor)

        serialized = catalog.to_dict()
        self.assertEqual(serialized["schema_version"], 1)
        self.assertEqual(len(serialized["templates"]), 23)
        self.assertEqual(
            serialized["templates"][0]["execution_profile"],
            {"id": "layered_digital", "version": 1},
        )
        self.assertEqual(
            serialized["templates"][0]["starter_resource"],
            "recipe_starters/compressai_kodak_default.yaml",
        )
        self.assertEqual(
            serialized["templates"][0]["editor_bindings"]["image_ids"],
            {
                "step_id": "data",
                "op": "source.image_dataset",
                "param": "image_ids",
            },
        )

        wireless_defaults = {
            "pilot_channel_estimation": (
                "ai_phy.pilot_channel_estimation.adapter",
                "pilot_channel_estimation",
            ),
            "mimo_ofdm_channel_estimation": (
                "ai_phy.mimo_ofdm_channel_estimation.adapter",
                "mimo_ofdm_channel_estimation",
            ),
            "beamforming_precoding": (
                "ai_phy.beamforming_precoding.adapter",
                "beamforming_link_evaluation",
            ),
            "wireless_localization": (
                "ai_phy.wireless_localization.adapter",
                "range_localization",
            ),
            "aoa_estimation": (
                "ai_phy.aoa_estimation.adapter",
                "aoa_array_estimation",
            ),
        }
        for task_id, (default_id, profile_id) in wireless_defaults.items():
            with self.subTest(task_id=task_id):
                templates = catalog.templates_for_task(task_id)
                self.assertEqual(len(templates), 1)
                self.assertEqual(catalog.default_for_task(task_id).id, default_id)
                self.assertTrue(all(item.editor == "graph" for item in templates))
                self.assertTrue(
                    all(
                        item.execution_profile.to_dict()
                        == {"id": profile_id, "version": 1}
                        for item in templates
                    )
                )

        for task_id in (
            "visual_question_answering",
            "object_detection",
            "segmentation",
            "image_text_retrieval",
        ):
            with self.subTest(task_id=task_id):
                self.assertEqual(
                    catalog.default_for_task(task_id).execution_profile.to_dict(),
                    {"id": "task_inference", "version": 1},
                )
        promoted = {
            "semantic_comm.image_reconstruction.protected_digital": (
                "image_reconstruction",
                "Image reconstruction — protected digital link",
                "layered_digital",
                False,
                "supported",
            ),
            "semantic_comm.image_reconstruction.deepjscc_awgn": (
                "image_reconstruction",
                "Image reconstruction — DeepJSCC over AWGN",
                "joint_source_channel_symbols",
                False,
                "experimental",
            ),
            "semantic_comm.image_reconstruction.deepjscc_slow_rayleigh": (
                "image_reconstruction",
                "Image reconstruction — blind DeepJSCC over slow fading",
                "joint_source_channel_symbols",
                False,
                "experimental",
            ),
            "semantic_comm.text_semantic_similarity.semantic_state_kb": (
                "text_semantic_similarity",
                "Text transport — knowledge-assisted semantic packets",
                "layered_digital",
                False,
                "supported",
            ),
            "semantic_comm.text_semantic_similarity.bart_joint_symbols": (
                "text_semantic_similarity",
                "Text transport — continuous-symbol JSCC",
                "joint_source_channel_symbols",
                False,
                "experimental",
            ),
            "ai_phy.csi_compression_feedback.truncated_angular_delay": (
                "csi_compression_feedback",
                "Limited-feedback MISO-OFDM CSI",
                "csi_feedback_downlink",
                True,
                "supported",
            ),
            "ai_phy.resource_allocation.equal_power": (
                "resource_allocation",
                "OFDM subcarrier resource allocation",
                "layered_digital",
                True,
                "supported",
            ),
            "ai_phy.resource_allocation.delayed_csi_finite_blocklength": (
                "resource_allocation",
                "Reliability-aware OFDM allocation with delayed CSI",
                "layered_digital",
                False,
                "experimental",
            ),
            "ai_phy.neural_receiver_demapping.adapter": (
                "neural_receiver_demapping",
                "QPSK receiver calibration under I/Q imbalance",
                "layered_digital",
                True,
                "supported",
            ),
            "ai_phy.neural_receiver_demapping.phase_tracking": (
                "neural_receiver_demapping",
                "Pilot-aided QPSK carrier tracking",
                "layered_digital",
                False,
                "experimental",
            ),
            "ai_phy.automatic_modulation_recognition.awgn": (
                "automatic_modulation_recognition",
                "Automatic modulation recognition with blind carrier impairments",
                "task_inference",
                True,
                "supported",
            ),
        }
        for template_id, (task_id, label, profile_id, default, status) in promoted.items():
            with self.subTest(template_id=template_id):
                template = catalog.template(template_id)
                self.assertIsNotNone(template)
                self.assertEqual(template.task_id, task_id)
                self.assertEqual(template.label, label)
                self.assertEqual(template.editor, "graph")
                self.assertEqual(template.default, default)
                self.assertEqual(template.status, status)
                self.assertEqual(
                    template.execution_profile.to_dict(),
                    {"id": profile_id, "version": 1},
                )

    def test_built_in_templates_strict_compile_plan_and_match_catalog(self) -> None:
        inspection = inspect_recipe_template_catalog(ROOT, self.registry)

        self.assertEqual(inspection.status, "valid")
        self.assertEqual(len(inspection.rows), 23)
        for row in inspection.rows:
            self.assertTrue(row.available, row.errors)
            self.assertTrue(row.valid, row.errors)
            self.assertEqual(row.source_kind, "project_override")
            self.assertEqual(row.resolved_task_id, row.template.task_id)
            self.assertEqual(
                row.recipe_execution_profile,
                row.template.execution_profile.to_dict(),
            )

        payload = inspection.to_dict()
        self.assertEqual(payload["status"], "valid")
        self.assertTrue(all(item["available"] for item in payload["templates"]))
        self.assertTrue(
            all(item["validation"]["status"] == "valid" for item in payload["templates"])
        )

    def test_missing_project_recipes_resolve_to_packaged_starters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inspection = inspect_recipe_template_catalog(Path(tmp), self.registry)

        self.assertEqual(inspection.status, "valid")
        self.assertEqual(len(inspection.rows), 23)
        self.assertTrue(all(row.available for row in inspection.rows))
        self.assertTrue(all(row.valid for row in inspection.rows))
        self.assertTrue(
            all(row.source_kind == "packaged_builtin" for row in inspection.rows)
        )

    def test_custom_template_without_override_or_starter_is_unavailable(self) -> None:
        catalog = recipe_template_catalog_from_dict(
            _catalog_payload(_template_entry())
        )
        with tempfile.TemporaryDirectory() as tmp:
            inspection = inspect_recipe_template_catalog(
                Path(tmp), self.registry, catalog=catalog
            )

        row = inspection.rows[0]
        self.assertFalse(row.available)
        self.assertEqual(row.validation_status, "unavailable")
        self.assertIn("no packaged starter", row.errors[0])

    def test_loader_rejects_schema_unknown_fields_and_unsafe_paths(self) -> None:
        cases = [
            ({"schema_version": 2, "templates": [_template_entry()]}, "schema_version"),
            (
                {
                    "schema_version": 1,
                    "templates": [_template_entry()],
                    "unexpected": True,
                },
                "unknown field",
            ),
            (
                _catalog_payload(_template_entry(unexpected=True)),
                "unknown field",
            ),
            (
                _catalog_payload(_template_entry(recipe_path="../outside.yaml")),
                "safe project-relative path",
            ),
            (
                _catalog_payload(_template_entry(recipe_path="recipes\\outside.yaml")),
                "forward slashes",
            ),
            (
                _catalog_payload(_template_entry(task_id="not_a_catalog_task")),
                "unknown research task",
            ),
            (
                _catalog_payload(_template_entry(editor="typo")),
                "editor must be one of",
            ),
            (
                _catalog_payload(
                    _template_entry(starter_resource="../outside.yaml")
                ),
                "safe project-relative path",
            ),
            (
                _catalog_payload(
                    _template_entry(starter_resource="other/template.yaml")
                ),
                "recipe_starters/",
            ),
            (
                _catalog_payload(
                    _template_entry(
                        editor_bindings={
                            "sample_ids": {
                                "step_id": "data",
                                "op": "source.task_labels_smoke",
                                "param": "sample_ids",
                                "unexpected": True,
                            }
                        }
                    )
                ),
                "unknown field",
            ),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RecipeTemplateCatalogError, expected):
                    recipe_template_catalog_from_dict(payload)

    def test_editor_bindings_are_validated_against_the_selected_source(self) -> None:
        cases = [
            (
                {
                    "sample_ids": {
                        "step_id": "missing",
                        "op": "source.task_labels_smoke",
                        "param": "sample_ids",
                    }
                },
                "missing step",
            ),
            (
                {
                    "sample_ids": {
                        "step_id": "data",
                        "op": "source.image_dataset",
                        "param": "sample_ids",
                    }
                },
                "requires step `data`",
            ),
            (
                {
                    "sample_ids": {
                        "step_id": "data",
                        "op": "source.task_labels_smoke",
                        "param": "typo",
                    }
                },
                "unknown parameter",
            ),
        ]
        for bindings, expected in cases:
            with self.subTest(expected=expected):
                catalog = recipe_template_catalog_from_dict(
                    _catalog_payload(_template_entry(editor_bindings=bindings))
                )
                row = inspect_recipe_template_catalog(
                    ROOT,
                    self.registry,
                    catalog=catalog,
                ).rows[0]
                self.assertEqual(row.validation_status, "invalid")
                self.assertIn(expected, row.errors[0])

    def test_loader_requires_exactly_one_supported_default_per_task(self) -> None:
        no_default = _catalog_payload(_template_entry(default=False))
        with self.assertRaisesRegex(RecipeTemplateCatalogError, "exactly one default"):
            recipe_template_catalog_from_dict(no_default)

        second = _template_entry(
            id="semantic_comm.classification.alternate",
            recipe_path="recipes/task_classification_alternate.yaml",
            order=20,
        )
        with self.assertRaisesRegex(RecipeTemplateCatalogError, "exactly one default"):
            recipe_template_catalog_from_dict(
                _catalog_payload(_template_entry(), second)
            )

        unsupported_default = _catalog_payload(_template_entry(status="experimental"))
        with self.assertRaisesRegex(RecipeTemplateCatalogError, "status `supported`"):
            recipe_template_catalog_from_dict(unsupported_default)

    def test_inspection_reports_task_and_execution_profile_disagreement(self) -> None:
        source = ROOT / "recipes" / "task_classification_smoke.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            recipe_dir = project_root / "recipes"
            recipe_dir.mkdir()
            (recipe_dir / source.name).write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            task_mismatch = recipe_template_catalog_from_dict(
                _catalog_payload(
                    _template_entry(
                        id="semantic_comm.image_reconstruction.default",
                        task_id="image_reconstruction",
                    )
                )
            )
            task_row = inspect_recipe_template_catalog(
                project_root,
                self.registry,
                catalog=task_mismatch,
            ).rows[0]
            self.assertTrue(task_row.available)
            self.assertEqual(task_row.validation_status, "invalid")
            self.assertIn("resolves to `classification`", task_row.errors[0])

            profile_payload = _catalog_payload(_template_entry())
            profile_payload = copy.deepcopy(profile_payload)
            profile_payload["templates"][0]["execution_profile"] = {
                "id": "layered_digital",
                "version": 1,
            }
            profile_mismatch = recipe_template_catalog_from_dict(profile_payload)
            profile_row = inspect_recipe_template_catalog(
                project_root,
                self.registry,
                catalog=profile_mismatch,
            ).rows[0]
            self.assertTrue(profile_row.available)
            self.assertEqual(profile_row.validation_status, "invalid")
            self.assertIn("declares execution profile", profile_row.errors[0])

    def test_instantiation_returns_standalone_recipe_with_canonical_provenance(self) -> None:
        template_id = "ai_phy.pilot_channel_estimation.adapter"
        with tempfile.TemporaryDirectory() as tmp:
            result = instantiate_recipe_template(
                template_id,
                Path(tmp),
                self.registry,
            )

        provenance = result.provenance.to_dict()
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(provenance["kind"], "noema.recipe_template_provenance")
        self.assertEqual(provenance["template_id"], template_id)
        self.assertEqual(provenance["catalog_schema_version"], 1)
        self.assertRegex(provenance["template_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(provenance["source_kind"], "packaged_builtin")
        self.assertEqual(
            provenance["source_reference"],
            "noema_lab:recipe_starters/channel_estimation_adapter_awgn.yaml",
        )
        self.assertRegex(provenance["source_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(provenance["overrides_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(result.recipe.metadata["template_provenance"], provenance)
        self.assertFalse(
            any(str(key).startswith("ui_") for key in result.recipe.metadata)
        )
        self.assertNotIn("opened_from", result.recipe.metadata)
        self.assertNotIn("recipe_template_id", result.recipe.metadata)
        self.assertEqual(result.to_dict()["recipe"], result.recipe.to_dict())

    def test_project_override_has_precedence_and_is_read_fresh_per_call(self) -> None:
        source = ROOT / "recipes" / "channel_estimation_adapter_awgn.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            target = project_root / "recipes" / source.name
            target.parent.mkdir()
            original = source.read_text(encoding="utf-8")
            target.write_text(original, encoding="utf-8")

            first = instantiate_recipe_template(
                "ai_phy.pilot_channel_estimation.adapter",
                project_root,
                self.registry,
            )
            self.assertEqual(first.provenance.source_kind, "project_override")
            self.assertEqual(
                first.provenance.source_digest,
                "sha256:%s" % hashlib.sha256(original.encode("utf-8")).hexdigest(),
            )

            changed = original.replace(
                "name: siso_pilot_channel_estimation",
                "name: channel_estimation_override",
            )
            target.write_text(changed, encoding="utf-8")
            second = instantiate_recipe_template(
                "ai_phy.pilot_channel_estimation.adapter",
                project_root,
                self.registry,
            )
            self.assertEqual(second.recipe.name, "channel_estimation_override")
            self.assertNotEqual(
                first.provenance.source_digest,
                second.provenance.source_digest,
            )

            target.write_text("not: a recipe\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RecipeTemplateInstantiationError,
                "project_override|recipes/channel_estimation_adapter_awgn.yaml|invalid",
            ):
                instantiate_recipe_template(
                    "ai_phy.pilot_channel_estimation.adapter",
                    project_root,
                    self.registry,
                )
            row = next(
                item
                for item in inspect_recipe_template_catalog(
                    project_root, self.registry
                ).rows
                if item.template.id == "ai_phy.pilot_channel_estimation.adapter"
            )
            self.assertEqual(row.source_kind, "project_override")
            self.assertEqual(row.validation_status, "invalid")

    def test_typed_overrides_are_validated_applied_and_hashed(self) -> None:
        template_id = "ai_phy.pilot_channel_estimation.adapter"
        first_overrides = {
            "name": "pilot_estimation_ls",
            "description": "LS reference.",
            "step_params": {"estimator": {"mode": "least_squares"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = instantiate_recipe_template(
                template_id,
                Path(tmp),
                self.registry,
                overrides=first_overrides,
            )
            same = instantiate_recipe_template(
                template_id,
                Path(tmp),
                self.registry,
                overrides={
                    "step_params": {"estimator": {"mode": "least_squares"}},
                    "description": "LS reference.",
                    "name": "pilot_estimation_ls",
                },
            )
            changed = instantiate_recipe_template(
                template_id,
                Path(tmp),
                self.registry,
                overrides={"step_params": {"estimator": {"mode": "linear_mmse_reference"}}},
            )

        self.assertEqual(first.recipe.name, "pilot_estimation_ls")
        self.assertEqual(first.recipe.description, "LS reference.")
        estimator = next(step for step in first.recipe.steps if step.id == "estimator")
        self.assertEqual(estimator.params["mode"], "least_squares")
        self.assertEqual(
            first.provenance.overrides_digest,
            same.provenance.overrides_digest,
        )
        self.assertNotEqual(
            first.provenance.overrides_digest,
            changed.provenance.overrides_digest,
        )

    def test_instantiation_rejects_unknown_and_invalid_overrides(self) -> None:
        template_id = "ai_phy.pilot_channel_estimation.adapter"
        cases = [
            ({"typo": True}, "unknown field"),
            ({"step_params": {"missing": {"mode": "least_squares"}}}, "unknown step"),
            ({"step_params": {"estimator": {"typo": "x"}}}, "unknown parameter"),
            ({"step_params": {"estimator": {"mode": ["least_squares"]}}}, "params are invalid"),
            ({"step_params": {"estimator": {"mode": float("nan")}}}, "finite JSON"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for overrides, expected in cases:
                with self.subTest(expected=expected):
                    with self.assertRaisesRegex(
                        RecipeTemplateInstantiationError,
                        expected,
                    ):
                        instantiate_recipe_template(
                            template_id,
                            Path(tmp),
                            self.registry,
                            overrides=overrides,
                        )

    def test_instantiation_rejects_unknown_template_id(self) -> None:
        with self.assertRaisesRegex(
            RecipeTemplateInstantiationError,
            "Unknown recipe template id",
        ):
            instantiate_recipe_template(
                "semantic_comm.missing.default",
                ROOT,
                self.registry,
            )


if __name__ == "__main__":
    unittest.main()
