from __future__ import annotations

import unittest

from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.research_catalog import load_research_catalog
from noema_lab.core.research import (
    ResearchSpecError,
    research_specs_from_recipe,
    validate_recipe_research_metadata,
)


def _recipe(metadata: dict, metric_op: str):
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "research_task_contract_test",
            "metadata": metadata,
            "steps": [{"id": "evaluation", "op": metric_op}],
        }
    )


class ResearchTaskResolutionTests(unittest.TestCase):
    def test_semantic_state_task_uses_precise_supported_assertion_metric(self) -> None:
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "semantic_state_metric_contract",
                "steps": [
                    {"id": "quality", "op": "metrics.text_semantic_similarity"},
                    {"id": "faithfulness", "op": "metrics.semantic_state_faithfulness"},
                ],
            }
        )

        metrics = research_specs_from_recipe(recipe)["task"]["metrics"]

        self.assertIn("faithfulness.unsupported_assertion_rate", metrics)
        self.assertNotIn("faithfulness.hallucination_rate", metrics)

    def test_catalog_exposes_ordered_research_areas_for_every_task(self) -> None:
        catalog = load_research_catalog()
        payload = catalog.to_dict()

        self.assertEqual(
            [area["id"] for area in payload["research_areas"]],
            [
                "physical_layer_resource_optimization",
                "channel_estimation_feedback",
                "antennas_positioning_sensing",
                "semantic_goal_oriented_communication",
            ],
        )
        self.assertTrue(catalog.tasks)
        self.assertTrue(
            all(task.area_id in catalog.research_areas for task in catalog.tasks.values())
        )
        self.assertEqual(
            catalog.task("neural_receiver_demapping").area_id,
            "physical_layer_resource_optimization",
        )
        self.assertEqual(
            catalog.task("automatic_modulation_recognition").area_id,
            "physical_layer_resource_optimization",
        )
        self.assertEqual(
            catalog.task("mimo_ofdm_channel_estimation").area_id,
            "channel_estimation_feedback",
        )
        self.assertEqual(
            catalog.task("aoa_estimation").area_id,
            "antennas_positioning_sensing",
        )
        self.assertEqual(
            catalog.task("visual_question_answering").area_id,
            "semantic_goal_oriented_communication",
        )

    def test_canonical_catalog_task_can_be_declared_by_id_only(self) -> None:
        recipe = _recipe(
            {"research": {"task": {"id": "classification"}}},
            "metrics.classification",
        )

        specs = research_specs_from_recipe(recipe)

        self.assertEqual(specs["task"]["id"], "classification")
        self.assertEqual(specs["task"]["kind"], "task_success")
        self.assertEqual(specs["task"]["modality"], "generic")
        self.assertIn("task.accuracy", specs["task"]["metrics"])

    def test_partial_explicit_research_keeps_graph_inferred_task(self) -> None:
        recipe = _recipe(
            {
                "research": {
                    "dataset": {
                        "id": "task_smoke",
                        "modality": "generic",
                        "version": "task-smoke-v1",
                    }
                }
            },
            "metrics.classification",
        )

        specs = research_specs_from_recipe(recipe)

        self.assertEqual(specs["task"]["id"], "classification")
        self.assertEqual(specs["source"], "recipe.metadata.research+inferred_from_recipe")

    def test_legacy_task_id_resolves_through_catalog(self) -> None:
        recipe = _recipe({"task_id": "object_detection"}, "metrics.detection")

        specs = research_specs_from_recipe(recipe)

        self.assertEqual(specs["task"]["id"], "object_detection")
        self.assertEqual(specs["task"]["kind"], "task_success")
        self.assertIn("metadata.task_id", specs["source"])

    def test_explicit_and_legacy_task_ids_must_agree(self) -> None:
        recipe = _recipe(
            {
                "task_id": "classification",
                "research": {
                    "task": {
                        "id": "object_detection",
                        "kind": "task_success",
                        "modality": "image",
                    }
                },
            },
            "metrics.detection",
        )

        with self.assertRaisesRegex(ResearchSpecError, "task declarations disagree"):
            research_specs_from_recipe(recipe)

    def test_declared_task_must_match_specific_metric_graph(self) -> None:
        recipe = _recipe(
            {
                "research": {
                    "task": {
                        "id": "classification",
                        "kind": "task_success",
                        "modality": "generic",
                    }
                }
            },
            "metrics.image_reconstruction",
        )

        with self.assertRaisesRegex(ResearchSpecError, "metric graph implies `image_reconstruction`"):
            research_specs_from_recipe(recipe)

    def test_catalog_contract_errors_block_recipe_validation(self) -> None:
        recipe = _recipe(
            {
                "research": {
                    "task": {
                        "id": "image_reconstruction",
                        "kind": "task_success",
                        "modality": "image",
                    }
                }
            },
            "metrics.image_reconstruction",
        )

        with self.assertRaisesRegex(ResearchSpecError, "expected `reconstruction`"):
            validate_recipe_research_metadata(recipe)


if __name__ == "__main__":
    unittest.main()
