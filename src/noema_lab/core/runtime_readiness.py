from __future__ import annotations

from typing import Any, Dict, List, Mapping

from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import compile_recipe


JsonDict = Dict[str, Any]


def inspect_recipe_run_readiness(recipe: Any, registry: Any) -> JsonDict:
    """Inspect whether a valid recipe is ready for ordinary benchmark execution.

    Compilation answers whether a graph is structurally valid.  This inspection
    adds the distinct runtime question without downloading data or model weights.
    Parameter-aware availability determines whether the selected/default
    runtime is usable; an optional but selected dependency still blocks
    execution. Export-only training interfaces are also not benchmark-ready.
    """

    issues: List[str] = []
    role = "runnable_template"
    strict_codes = {
        "unknown_recipe_field",
        "unknown_recipe_step_field",
        "implicit_recipe_step_id",
    }
    for diagnostic in list(getattr(recipe, "diagnostics", ()) or ()):
        if getattr(diagnostic, "code", None) in strict_codes:
            issues.append(
                "strict recipe validation: %s"
                % getattr(diagnostic, "message", "recipe is not strict-compatible")
            )
    try:
        strict_compilation = compile_recipe(recipe.to_dict(), mode="strict")
    except Exception as exc:
        issues.append("strict recipe validation could not run: %s" % exc)
    else:
        for diagnostic in strict_compilation.errors:
            issues.append("strict recipe validation: %s" % diagnostic.message)
    try:
        validate_recipe_against_registry(recipe, registry)
    except Exception as exc:
        issues.append(str(exc))
    for step in list(getattr(recipe, "steps", ()) or ()):
        params = dict(getattr(step, "params", {}) or {})
        runtime = str(params.get("runtime") or "").strip().lower()
        if runtime == "training_interface":
            role = "training_starter"
            issues.append(
                "%s is an export-only training interface; export a Workbench contract and bind the returned artifact before benchmark execution"
                % step.id
            )
        try:
            operation = registry.get(step.op)
            description = operation.describe()
        except Exception as exc:
            issues.append("%s: %s" % (step.id, exc))
            continue
        if str(description.get("status") or "implemented") != "implemented":
            issues.append(
                "%s: operation %s has status %s"
                % (step.id, step.op, description.get("status"))
            )
        try:
            availability = operation.runtime_availability(params)
        except Exception as exc:
            issues.append("%s: could not inspect runtime availability: %s" % (step.id, exc))
            continue
        if not isinstance(availability, Mapping):
            continue
        requested_backend = str(
            params.get("wireless_backend") or params.get("backend") or "auto"
        ).strip().lower()
        unavailable = availability.get("available") is False
        if availability.get("sionna") is False and requested_backend == "sionna":
            unavailable = True
        if unavailable:
            issues.append(
                "%s: %s"
                % (
                    step.id,
                    availability.get("reason")
                    or "required runtime dependencies are unavailable",
                )
            )
    metadata = dict(getattr(recipe, "metadata", {}) or {})
    research_stage = str(metadata.get("research_stage") or "").strip().lower()
    recipe_name = str(getattr(recipe, "name", "") or "").lower()
    if role != "training_starter" and (
        "smoke" in recipe_name
        and any(token in recipe_name for token in ("classification", "artifact"))
    ):
        role = "contract_smoke"
    elif role != "training_starter" and research_stage in {"upper_bound", "oracle"}:
        role = research_stage
    return {
        "runnable": not issues,
        "role": role,
        "issues": list(dict.fromkeys(issues)),
    }
