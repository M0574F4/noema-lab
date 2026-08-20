from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from noema_lab.core.executor import MAX_PARALLEL_WORKERS
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.recipes import RecipeValidationError


JsonDict = Dict[str, Any]

DEFAULT_EXECUTION_CONTROLS: JsonDict = {
    "strict_lint": False,
    "backend": None,
    "implementation": None,
    "parallel_workers": 1,
    "use_plan_cache": True,
}


def execution_controls_contract() -> JsonDict:
    """Return the execution-control contract shared by CLI, HTTP, and UI."""

    return {
        "strict_lint": {"type": "boolean", "default": False},
        "backend": {
            "type": ["string", "null"],
            "default": None,
            "description": "Require one materialization backend for every step.",
        },
        "implementation": {
            "type": ["string", "null"],
            "default": None,
            "description": "Require one materialization implementation for every step.",
        },
        "parallel_workers": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_PARALLEL_WORKERS,
            "default": 1,
        },
        "use_plan_cache": {"type": "boolean", "default": True},
    }


def normalize_execution_controls(raw: Optional[Mapping[str, Any]] = None) -> JsonDict:
    """Validate and normalize the complete execution policy."""

    if raw is None:
        return dict(DEFAULT_EXECUTION_CONTROLS)
    if not isinstance(raw, Mapping):
        raise ValueError("execution must be a JSON object")
    unknown = sorted(
        str(key) for key in raw if key not in DEFAULT_EXECUTION_CONTROLS
    )
    if unknown:
        raise ValueError(
            "execution contains unsupported field%s: %s"
            % ("s" if len(unknown) != 1 else "", ", ".join(unknown))
        )

    strict_lint = raw.get(
        "strict_lint", DEFAULT_EXECUTION_CONTROLS["strict_lint"]
    )
    if not isinstance(strict_lint, bool):
        raise ValueError("execution.strict_lint must be a boolean")

    backend = _optional_nonempty_text(raw.get("backend"), "execution.backend")
    implementation = _optional_nonempty_text(
        raw.get("implementation"),
        "execution.implementation",
    )

    workers = raw.get(
        "parallel_workers",
        DEFAULT_EXECUTION_CONTROLS["parallel_workers"],
    )
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError("execution.parallel_workers must be an integer")
    if workers < 1 or workers > MAX_PARALLEL_WORKERS:
        raise ValueError(
            "execution.parallel_workers must be between 1 and %d"
            % MAX_PARALLEL_WORKERS
        )

    use_plan_cache = raw.get(
        "use_plan_cache",
        DEFAULT_EXECUTION_CONTROLS["use_plan_cache"],
    )
    if not isinstance(use_plan_cache, bool):
        raise ValueError("execution.use_plan_cache must be a boolean")

    return {
        "strict_lint": strict_lint,
        "backend": backend,
        "implementation": implementation,
        "parallel_workers": workers,
        "use_plan_cache": use_plan_cache,
    }


def executor_options(execution: Mapping[str, Any]) -> JsonDict:
    """Project normalized controls onto ``LocalExecutor.run`` keyword arguments."""

    normalized = normalize_execution_controls(execution)
    return {
        "backend": normalized["backend"],
        "implementation": normalized["implementation"],
        "parallel_workers": normalized["parallel_workers"],
        "use_plan_cache": normalized["use_plan_cache"],
    }


def require_strict_lint(
    recipe: Any,
    registry: Any,
    *,
    context: str = "",
) -> JsonDict:
    """Fail with the same strict-lint semantics on every execution surface."""

    report = lint_recipe_invariants(recipe, registry, strict=True)
    if report.get("status") == "passed":
        return report
    errors = [
        issue
        for issue in report.get("issues") or []
        if isinstance(issue, Mapping) and issue.get("severity") == "error"
    ]
    details = "; ".join(
        "%s%s: %s"
        % (
            issue.get("code") or "strict_lint",
            " [%s]" % issue["step_id"] if issue.get("step_id") else "",
            issue.get("message") or "strict lint failed",
        )
        for issue in errors
    )
    prefix = "strict lint failed"
    if context:
        prefix += " for %s" % context
    raise RecipeValidationError("%s: %s" % (prefix, details or "unknown error"))


def _optional_nonempty_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("%s must be a string or null" % field)
    normalized = value.strip()
    if not normalized:
        raise ValueError("%s must not be empty; use null for automatic selection" % field)
    return normalized
