from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.matrix import (
    RecipeMatrixError,
    canonicalize_recipe_matrix,
    materialize_recipe_matrix_selection,
    matrix_variant_id,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationRegistry,
    OperationResult,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.variants import (
    RecipeVariantPlanningError,
    plan_recipe_variants,
    prepare_single_run_recipe,
)


class _ValueOperation(Operation):
    id = "test.variant_value"
    name = "Variant value"
    params_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "integer", "minimum": 0, "maximum": 1},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class _ConditionalOperation(Operation):
    id = "test.conditional_value"
    name = "Conditional value"
    params_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "default": "fixed", "enum": ["fixed", "reference"]},
            "variance": {
                "type": "number",
                "x-noema-effective-when": {"mode": "fixed"},
            },
            "reference_db": {
                "type": "number",
                "x-noema-effective-when": {"mode": "reference"},
            },
        },
        "additionalProperties": False,
    }

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


def _registry() -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(_ValueOperation())
    registry.register(_ConditionalOperation())
    return registry


def _payload(*, name: str = "variant_recipe", values=None, metadata=None):
    recipe_metadata = copy.deepcopy(metadata or {})
    if values is not None:
        recipe_metadata["matrix"] = {
            "dimensions": {"value": copy.deepcopy(values)},
            "step_params": {
                "value": {"value": {"matrix": "value"}},
            },
        }
    return {
        "schema_version": 1,
        "name": name,
        "execution_profile": {"id": "custom", "version": 1},
        "metadata": recipe_metadata,
        "steps": [
            {
                "id": "value",
                "op": "test.variant_value",
                "params": {"value": 0},
                "inputs": {},
            }
        ],
    }


class TypedMatrixIdentityTests(unittest.TestCase):
    def test_matrix_variant_cannot_bind_an_inactive_parameter(self):
        payload = {
            "schema_version": 1,
            "name": "inactive_matrix_parameter",
            "execution_profile": {"id": "custom", "version": 1},
            "metadata": {
                "matrix": {
                    "dimensions": {"reference_db": [0.0, 10.0]},
                    "step_params": {
                        "conditional": {"reference_db": {"matrix": "reference_db"}},
                    },
                },
            },
            "steps": [
                {
                    "id": "conditional",
                    "op": "test.conditional_value",
                    "params": {"mode": "fixed", "variance": 0.2},
                    "inputs": {},
                },
            ],
        }

        with self.assertRaisesRegex(RecipeVariantPlanningError, "reference_db.*inactive"):
            plan_recipe_variants(payload, _registry())

    def test_duplicate_typed_dimension_values_are_rejected(self):
        recipe = recipe_from_dict(
            _payload(
                values=[
                    {"label": "same", "nested": [True, 1]},
                    {"nested": [True, 1], "label": "same"},
                ]
            )
        )
        with self.assertRaisesRegex(
            RecipeMatrixError,
            "duplicate typed value at indices 0 and 1",
        ):
            canonicalize_recipe_matrix(recipe)

    def test_typed_equality_does_not_alias_boolean_integer_float_or_string(self):
        recipe = recipe_from_dict(_payload(values=[True, 1, 1.0, "1"]))
        self.assertEqual(canonicalize_recipe_matrix(recipe).variant_count, 4)
        self.assertEqual(
            matrix_variant_id({"b": [1, {"x": True}], "a": 1}),
            matrix_variant_id({"a": 1, "b": [1, {"x": True}]}),
        )
        self.assertNotEqual(
            matrix_variant_id({"value": True}),
            matrix_variant_id({"value": 1}),
        )
        self.assertNotEqual(
            matrix_variant_id({"value": 1}),
            matrix_variant_id({"value": 1.0}),
        )


class RecipeVariantPlanTests(unittest.TestCase):
    def test_plan_returns_canonical_metadata_ordered_variants_and_stable_ids(self):
        first = plan_recipe_variants(
            _payload(name="readable", values=[0, 1]),
            _registry(),
        )
        renamed = plan_recipe_variants(
            _payload(name="renamed", values=[0, 1]),
            _registry(),
        )

        self.assertEqual(first.expanded_count, 2)
        self.assertEqual(first.canonical_matrix.source, "matrix")
        self.assertEqual(
            [variant.matrix_selection for variant in first.variants],
            [{"value": 0}, {"value": 1}],
        )
        self.assertEqual(
            [variant.matrix_variant_id for variant in first.variants],
            [variant.matrix_variant_id for variant in renamed.variants],
        )
        for index, variant in enumerate(first.variants):
            self.assertEqual(variant.matrix_index, index)
            self.assertRegex(variant.matrix_variant_id, r"^mxv1-[0-9a-f]{64}$")
            self.assertEqual(
                variant.recipe.metadata["matrix_variant_id"],
                variant.matrix_variant_id,
            )
            self.assertEqual(
                variant.recipe.name,
                "readable__value_%d" % index,
            )
            self.assertNotIn("matrix", variant.recipe.metadata)

        envelope = first.to_expansion_dict()
        self.assertEqual(envelope["expanded_count"], 2)
        self.assertEqual(envelope["canonical_matrix"]["variant_count"], 2)
        self.assertEqual(len(envelope["recipes"]), 2)

    def test_plan_registry_validates_every_variant_with_single_run_parity(self):
        authored = _payload(values=[1, 2])
        with self.assertRaisesRegex(
            RecipeVariantPlanningError,
            r"matrix variant 1 .*value.*must be <= 1",
        ):
            plan_recipe_variants(authored, _registry())

        recipe = recipe_from_dict(authored)
        invalid_concrete = materialize_recipe_matrix_selection(
            recipe,
            {"value": 2},
        )
        with self.assertRaisesRegex(
            RecipeVariantPlanningError,
            r"single-run recipe.*value.*must be <= 1",
        ):
            prepare_single_run_recipe(invalid_concrete, _registry())

    def test_v1_legacy_sweep_is_planned_through_the_same_service(self):
        payload = _payload(metadata={"sweeps": {"value.value": "0,1"}})
        plan = plan_recipe_variants(payload, _registry())
        self.assertEqual(plan.canonical_matrix.source, "sweeps")
        self.assertEqual(
            [variant.matrix_selection for variant in plan.variants],
            [{"value.value": 0}, {"value.value": 1}],
        )

    def test_matrix_disabled_plan_preserves_expansion_compatibility(self):
        plan = plan_recipe_variants(_payload(), _registry())
        self.assertEqual(plan.expanded_count, 0)
        self.assertEqual(plan.variants, ())
        self.assertFalse(plan.canonical_matrix.enabled)


class SingleRunPreparationTests(unittest.TestCase):
    def test_unmaterialized_matrix_is_rejected_and_concrete_recipe_is_accepted(self):
        authored = _payload(values=[0, 1])
        with self.assertRaisesRegex(
            RecipeVariantPlanningError,
            "materialize metadata.matrix first",
        ):
            prepare_single_run_recipe(authored, _registry())

        concrete = plan_recipe_variants(authored, _registry()).variants[0].recipe
        prepared = prepare_single_run_recipe(concrete, _registry())
        self.assertEqual(
            prepared.metadata["matrix_variant_id"],
            concrete.metadata["matrix_variant_id"],
        )

    def test_legacy_concrete_selection_is_upgraded_but_mismatched_id_is_rejected(self):
        payload = _payload(metadata={"matrix_selection": {"value": 1}})
        prepared = prepare_single_run_recipe(payload, _registry())
        self.assertEqual(
            prepared.metadata["matrix_variant_id"],
            matrix_variant_id({"value": 1}),
        )

        payload["metadata"]["matrix_variant_id"] = "mxv1-" + "0" * 64
        with self.assertRaisesRegex(
            RecipeVariantPlanningError,
            "does not match metadata.matrix_selection",
        ):
            prepare_single_run_recipe(payload, _registry())


if __name__ == "__main__":
    unittest.main()
