from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from noema_lab.core.matrix import (
    MAX_MATRIX_VARIANTS,
    CanonicalRecipeMatrix,
    RecipeMatrixError,
    canonicalize_recipe_matrix,
    expand_recipe_matrix,
    matrix_variant_id,
)
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeValidationError, compile_recipe

JsonDict = dict[str, Any]


class RecipeVariantPlanningError(RecipeMatrixError):
    """Raised when an authored recipe or one of its variants cannot run."""


@dataclass(frozen=True)
class PlannedRecipeVariant:
    """One ordered, strictly validated concrete point in a recipe matrix."""

    matrix_variant_id: str
    matrix_index: int
    matrix_selection: JsonDict
    recipe: Recipe

    def to_dict(self) -> JsonDict:
        return {
            "matrix_variant_id": self.matrix_variant_id,
            "matrix_index": self.matrix_index,
            "matrix_selection": copy.deepcopy(self.matrix_selection),
            "recipe": self.recipe.to_dict(),
        }


@dataclass(frozen=True)
class RecipeVariantPlan:
    """Canonical matrix metadata and its validated concrete variants."""

    recipe: Recipe
    canonical_matrix: CanonicalRecipeMatrix
    variants: Tuple[PlannedRecipeVariant, ...]

    @property
    def expanded_count(self) -> int:
        return len(self.variants)

    def to_dict(self) -> JsonDict:
        return {
            "recipe": self.recipe.name,
            "expanded_count": self.expanded_count,
            "canonical_matrix": self.canonical_matrix.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
        }

    def to_expansion_dict(self) -> JsonDict:
        """Return the v1 expansion envelope while retaining canonical metadata."""

        return {
            "recipe": self.recipe.name,
            "expanded_count": self.expanded_count,
            "canonical_matrix": self.canonical_matrix.to_dict(),
            "recipes": [variant.recipe.to_dict() for variant in self.variants],
        }


def plan_recipe_variants(
    recipe_payload: Recipe | Mapping[str, Any],
    registry: OperationRegistry,
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> RecipeVariantPlan:
    """Strictly compile and registry-validate a matrix and every concrete point.

    Matrix-disabled recipes intentionally return no variants, preserving the
    v1 expansion contract. Use :func:`prepare_single_run_recipe` for the
    ordinary one-recipe execution path.
    """

    recipe = _strict_compiled_recipe(recipe_payload, registry, label="base recipe")
    canonical_matrix = canonicalize_recipe_matrix(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )
    if not canonical_matrix.enabled:
        return RecipeVariantPlan(recipe, canonical_matrix, ())

    expansion = expand_recipe_matrix(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )
    variants = []
    for expected_index, concrete_payload in enumerate(expansion["recipes"]):
        metadata = dict(concrete_payload.get("metadata") or {})
        selection = metadata.get("matrix_selection")
        if not isinstance(selection, Mapping):
            raise RecipeVariantPlanningError(
                "matrix variant %d is missing metadata.matrix_selection"
                % expected_index
            )
        expected_id = matrix_variant_id(selection)
        if metadata.get("matrix_variant_id") != expected_id:
            raise RecipeVariantPlanningError(
                "matrix variant %d has inconsistent metadata.matrix_variant_id"
                % expected_index
            )
        if metadata.get("matrix_index") != expected_index:
            raise RecipeVariantPlanningError(
                "matrix variant %d has inconsistent metadata.matrix_index"
                % expected_index
            )
        concrete = _strict_compiled_recipe(
            concrete_payload,
            registry,
            label="matrix variant %d (%s)" % (expected_index, expected_id),
        )
        variants.append(
            PlannedRecipeVariant(
                matrix_variant_id=expected_id,
                matrix_index=expected_index,
                matrix_selection=copy.deepcopy(dict(selection)),
                recipe=concrete,
            )
        )
    return RecipeVariantPlan(recipe, canonical_matrix, tuple(variants))


def prepare_single_run_recipe(
    recipe_payload: Recipe | Mapping[str, Any],
    registry: OperationRegistry,
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> Recipe:
    """Return one validated concrete recipe or reject an unresolved matrix.

    Concrete v1 recipes that already carry ``matrix_selection`` but predate
    ``matrix_variant_id`` remain accepted; the returned normalized copy is
    upgraded with the stable ID. A supplied new ID must match the selection.
    """

    recipe = _strict_compiled_recipe(
        recipe_payload,
        registry,
        label="single-run recipe",
    )
    return prepare_compiled_single_run_recipe(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )


def prepare_compiled_single_run_recipe(
    recipe: Recipe,
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> Recipe:
    """Gate an already compiled recipe without repeating registry resolution.

    This is the executor-facing half of :func:`prepare_single_run_recipe`.
    Callers must have compiled strictly and must still create an execution plan;
    the split keeps preflight to one compiler pass and one planner pass.
    """

    canonical_matrix = canonicalize_recipe_matrix(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )
    if canonical_matrix.enabled:
        raise RecipeVariantPlanningError(
            "single-run execution requires a concrete recipe; "
            "materialize metadata.matrix first"
        )

    metadata = recipe.metadata
    if "matrix_selection" not in metadata:
        if "matrix_variant_id" in metadata:
            raise RecipeVariantPlanningError(
                "metadata.matrix_variant_id requires metadata.matrix_selection"
            )
        return recipe

    selection = metadata.get("matrix_selection")
    if not isinstance(selection, Mapping):
        raise RecipeVariantPlanningError(
            "metadata.matrix_selection must be a mapping"
        )
    expected_id = matrix_variant_id(selection)
    supplied_id = metadata.get("matrix_variant_id")
    if supplied_id is not None and supplied_id != expected_id:
        raise RecipeVariantPlanningError(
            "metadata.matrix_variant_id does not match metadata.matrix_selection"
        )
    metadata["matrix_variant_id"] = expected_id
    return recipe


def _strict_compiled_recipe(
    recipe_payload: Recipe | Mapping[str, Any],
    registry: OperationRegistry,
    *,
    label: str,
) -> Recipe:
    try:
        compilation = compile_recipe(
            recipe_payload,
            mode="strict",
            registry=registry,
        )
        recipe = compilation.require_recipe()
        effective_recipe = compilation.require_recipe(effective=True)
        validate_recipe_against_registry(effective_recipe, registry)
    except (RecipeValidationError, ValueError) as exc:
        raise RecipeVariantPlanningError("%s is invalid: %s" % (label, exc)) from exc
    return recipe
