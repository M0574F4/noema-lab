from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from noema_lab.core.operations import MATERIALIZATION_RUNNERS, OperationRegistry
from noema_lab.core.recipes import Recipe
from noema_lab.core.reproducibility import canonical_json_sha256

JsonDict = Dict[str, Any]

DIFFERENTIABLE_GRADIENTS = {"full", "surrogate"}
# Runner support is a capability summary, not a concrete backend plan.  Exact
# backend selection (including export-time Torch substitution) is enforced by
# the planner/exporter; branch selectors such as ``runtime`` remain binding
# here so an export-only interface cannot be advertised as benchmark-ready.
_RUNNER_SUPPORT_BACKEND_SELECTORS = {
    "data_plane_backend",
    "wireless_backend",
}

RUNNER_CONTRACTS: Mapping[str, JsonDict] = {
    "benchmark_run": {
        "id": "benchmark_run",
        "label": "Benchmark Run",
        "path": "artifact/evidence path",
        "description": "Executes recipes as reproducible artifact/result bundles for comparison.",
        "requires_gradient": False,
        "requires_exportable": False,
    },
    "dataset_capture": {
        "id": "dataset_capture",
        "label": "Dataset Capture",
        "path": "artifact dataset path",
        "description": "Executes recipes to produce supervised tap datasets for external training.",
        "requires_gradient": False,
        "requires_exportable": False,
    },
    "differentiable_export": {
        "id": "differentiable_export",
        "label": "Differentiable Export",
        "path": "differentiable module path",
        "description": "Materializes differentiable subgraphs for PyTorch/Sionna modules and optional reference trainers.",
        "requires_gradient": True,
        "requires_exportable": True,
    },
}


def runner_contracts() -> JsonDict:
    return {runner: dict(contract) for runner, contract in RUNNER_CONTRACTS.items()}


def operation_runner_supports(operation_description: Mapping[str, Any]) -> JsonDict:
    return {
        runner: operation_runner_support(operation_description, runner)
        for runner in sorted(MATERIALIZATION_RUNNERS)
    }


def operation_runner_support(operation_description: Mapping[str, Any], runner: str) -> JsonDict:
    runner_id = str(runner or "").strip().lower()
    if runner_id not in RUNNER_CONTRACTS:
        raise ValueError("Unknown runner contract: %s" % runner)
    op_id = str(operation_description.get("id") or "operation")
    op_name = str(operation_description.get("name") or op_id)
    contract = RUNNER_CONTRACTS[runner_id]
    materializations = [
        dict(item)
        for item in operation_description.get("materializations") or []
        if str(item.get("runner") or "").strip().lower() == runner_id
    ]
    implemented = [item for item in materializations if str(item.get("status") or "implemented") == "implemented"]
    differentiability = dict(operation_description.get("differentiability") or {})
    gradient = str(differentiability.get("gradient") or "none")
    exportable = bool(differentiability.get("exportable", False))

    supported = bool(implemented)
    reason = ""
    if not materializations:
        reason = "No %s materialization is declared for `%s`." % (contract["label"], op_id)
    elif not implemented:
        reason = "No implemented %s materialization is declared for `%s`." % (contract["label"], op_id)

    if runner_id == "differentiable_export":
        if gradient not in DIFFERENTIABLE_GRADIENTS:
            supported = False
            detail = str(differentiability.get("reason") or "gradient is `%s`" % gradient).strip()
            reason = "%s is a gradient break for the differentiable exporter: %s" % (op_name, detail)
        elif not exportable:
            supported = False
            detail = str(differentiability.get("reason") or "operation is not exportable").strip()
            reason = "%s is not exportable to the differentiable module: %s" % (op_name, detail)

    if supported and not reason:
        reason = "%s can use `%s` through %s." % (
            contract["label"],
            op_id,
            ", ".join(sorted({str(item.get("backend")) for item in implemented})),
        )

    return {
        "runner": runner_id,
        "label": contract["label"],
        "path": contract["path"],
        "applicable": True,
        "status": "supported" if supported else "unsupported",
        "supported": bool(supported),
        "reason": reason,
        "materializations": materializations,
        "implemented_materializations": implemented,
    }


def recipe_runner_support(recipe: Recipe, registry: OperationRegistry) -> JsonDict:
    step_support: List[JsonDict] = []
    summary: JsonDict = {
        runner: {
            "applicable": True,
            "status": "supported",
            "supported": True,
            "relevant_steps": [],
            "unsupported_steps": [],
            "not_applicable_steps": [],
        }
        for runner in sorted(MATERIALIZATION_RUNNERS)
    }
    for step in recipe.steps:
        operation = registry.get(step.op).describe()
        supports = operation_runner_supports(operation)
        step_payload = {
            "id": step.id,
            "op": step.op,
            "name": operation.get("name", step.op),
            "runner_support": supports,
        }
        for runner, generic_support in list(supports.items()):
            if runner == "differentiable_export" and not _step_relevant_to_differentiable_export(operation):
                supports[runner] = {
                    **generic_support,
                    "applicable": False,
                    "status": "not_applicable",
                    "supported": None,
                    "reason": (
                        "%s is outside the differentiable-export subgraph."
                        % operation.get("name", step.op)
                    ),
                }
                summary[runner]["not_applicable_steps"].append(
                    {
                        "step_id": step.id,
                        "op": step.op,
                        "reason": supports[runner]["reason"],
                    }
                )
                continue
            support = _parameter_aware_step_support(
                step,
                operation,
                runner,
                generic_support,
            )
            supports[runner] = support
            summary[runner]["relevant_steps"].append(
                {"step_id": step.id, "op": step.op}
            )
            if support.get("supported"):
                continue
            summary[runner]["supported"] = False
            summary[runner]["unsupported_steps"].append(
                {
                    "step_id": step.id,
                    "op": step.op,
                    "reason": support.get("reason") or "runner materialization is unavailable",
                }
            )
        step_support.append(step_payload)
    for runner, payload in summary.items():
        payload["relevant_count"] = len(payload["relevant_steps"])
        payload["unsupported_count"] = len(payload["unsupported_steps"])
        payload["not_applicable_count"] = len(payload["not_applicable_steps"])
        if payload["relevant_count"] == 0:
            payload["applicable"] = False
            payload["status"] = "not_applicable"
            payload["supported"] = False
        elif payload["unsupported_count"]:
            payload["status"] = "unsupported"
        else:
            payload["status"] = "supported"
    return {
        "schema_version": 1,
        "recipe": recipe.name,
        "contracts": runner_contracts(),
        "summary": summary,
        "steps": step_support,
    }


def _parameter_aware_step_support(
    step: Any,
    operation_description: Mapping[str, Any],
    runner: str,
    generic_support: Mapping[str, Any],
) -> JsonDict:
    """Restrict generic operation support to the recipe's selected branch."""

    payload = dict(generic_support)
    if not payload.get("supported"):
        return payload
    candidates = [
        dict(item)
        for item in payload.get("implemented_materializations") or []
        if isinstance(item, Mapping)
    ]
    matching = [
        item
        for item in candidates
        if _step_matches_parameter_bindings(
            step,
            operation_description,
            item.get("parameter_bindings"),
        )
    ]
    if matching:
        payload["selected_materializations"] = matching
        return payload
    if not any(item.get("parameter_bindings") for item in candidates):
        return payload

    binding_names = sorted(
        {
            str(name)
            for item in candidates
            for name in dict(item.get("parameter_bindings") or {})
            if str(name) not in _RUNNER_SUPPORT_BACKEND_SELECTORS
        }
    )
    selected = {
        name: _effective_step_parameter(step, operation_description, name)
        for name in binding_names
    }
    other_runners = sorted(
        {
            str(item.get("runner"))
            for item in operation_description.get("materializations") or []
            if (
                isinstance(item, Mapping)
                and str(item.get("status") or "implemented") == "implemented"
                and str(item.get("runner") or "") != runner
                and item.get("parameter_bindings")
                and _step_matches_parameter_bindings(
                    step,
                    operation_description,
                    item.get("parameter_bindings"),
                )
            )
        }
    )
    runner_label = str(payload.get("label") or runner)
    reason = (
        "Selected params %s have no implemented %s materialization for `%s`."
        % (selected, runner_label, operation_description.get("id") or step.op)
    )
    if other_runners:
        reason += " The selected branch is implemented for runner=%s." % ", ".join(
            other_runners
        )
    payload.update(
        {
            "status": "unsupported",
            "supported": False,
            "reason": reason,
            "selected_params": selected,
            "selected_materializations": [],
        }
    )
    return payload


def _step_matches_parameter_bindings(
    step: Any,
    operation_description: Mapping[str, Any],
    raw_bindings: Any,
) -> bool:
    bindings = dict(raw_bindings) if isinstance(raw_bindings, Mapping) else {}
    return all(
        canonical_json_sha256(
            {"value": _effective_step_parameter(step, operation_description, name)}
        )
        == canonical_json_sha256({"value": expected})
        for name, expected in bindings.items()
        if str(name) not in _RUNNER_SUPPORT_BACKEND_SELECTORS
    )


def _effective_step_parameter(
    step: Any,
    operation_description: Mapping[str, Any],
    name: str,
) -> Any:
    params = dict(getattr(step, "params", {}) or {})
    if name in params:
        return params[name]
    schema = operation_description.get("params_schema")
    properties = (
        schema.get("properties")
        if isinstance(schema, Mapping)
        and isinstance(schema.get("properties"), Mapping)
        else {}
    )
    property_schema = properties.get(name)
    if isinstance(property_schema, Mapping) and "default" in property_schema:
        return property_schema.get("default")
    return None


def _step_relevant_to_differentiable_export(operation_description: Mapping[str, Any]) -> bool:
    differentiability = dict(operation_description.get("differentiability") or {})
    training_capabilities = dict(operation_description.get("training_capabilities") or {})
    gradient = str(differentiability.get("gradient") or "none")
    return (
        bool(training_capabilities.get("built_in_fine_tuning", False))
        or bool(training_capabilities.get("portable_replacement", False))
        or bool(differentiability.get("exportable", False))
        or gradient in DIFFERENTIABLE_GRADIENTS
    )


def runner_summary_line(support: Mapping[str, Any]) -> str:
    summary = support.get("summary") if isinstance(support, Mapping) else None
    if not isinstance(summary, Mapping):
        return "runner support: unknown"
    parts = []
    for runner in ("benchmark_run", "dataset_capture", "differentiable_export"):
        payload = summary.get(runner) or {}
        status = str(payload.get("status") or "")
        if status == "not_applicable":
            label = "n/a"
        else:
            label = "yes" if payload.get("supported") else "no"
        parts.append("%s=%s" % (runner, label))
    return "runner support: %s" % ", ".join(parts)


def unsupported_steps_for_runner(support: Mapping[str, Any], runner: str) -> List[JsonDict]:
    summary = support.get("summary") if isinstance(support, Mapping) else None
    if not isinstance(summary, Mapping):
        return []
    payload = summary.get(str(runner)) or {}
    return [dict(item) for item in payload.get("unsupported_steps") or []]


def first_runner_blocker(support: Mapping[str, Any], runners: Iterable[str]) -> JsonDict:
    for runner in runners:
        blockers = unsupported_steps_for_runner(support, runner)
        if blockers:
            return blockers[0]
    return {}
