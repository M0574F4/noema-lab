from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.runner_contracts import (
    operation_runner_supports,
    recipe_runner_support,
    runner_contracts,
    runner_summary_line,
    unsupported_steps_for_runner,
)

JsonDict = Dict[str, Any]

_DIFFERENTIABLE_GRADIENTS = {"full", "surrogate"}
_LOSS_STEP_PREFIXES = ("metrics.",)
_LOSS_STEP_IDS = {"evaluation", "faithfulness", "loss"}


def inspect_training_feasibility(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Optional[Sequence[str] | str] = None,
    loss: Optional[Sequence[str] | str] = None,
) -> JsonDict:
    validate_recipe_against_registry(recipe, registry)
    runner_support = recipe_runner_support(recipe, registry)
    steps = [_step_report(step, registry) for step in recipe.steps]
    step_reports = {item["id"]: item for item in steps}
    step_by_id = {step.id: step for step in recipe.steps}
    consumers = _consumer_graph(recipe.steps)
    predecessors = _predecessor_graph(recipe.steps)
    explicit_loss_ids = _loss_step_ids(recipe.steps)
    loss_step_candidates = (
        set(explicit_loss_ids)
        if explicit_loss_ids
        else _terminal_step_ids(recipe.steps, consumers)
    )
    loss_ids = _resolve_loss_ids(recipe.steps, consumers, loss)
    selected_loss_ids = set(loss_ids)

    auto_optimizable_ids = [
        item["id"]
        for item in steps
        if bool(item["replacement_ready"]) and item["id"] in _ancestors_of_any(loss_ids, predecessors)
    ]
    requested_optimizable_ids = _normalize_step_selection(optimizable_steps)
    unknown_trainable = [step_id for step_id in requested_optimizable_ids if step_id not in step_by_id]
    if unknown_trainable:
        raise ValueError("Replacement step id is not in the recipe: %s" % ", ".join(unknown_trainable))
    optimizable_ids = requested_optimizable_ids or auto_optimizable_ids
    selected_replacements = set(optimizable_ids)
    replacement_candidates = [item["id"] for item in steps if bool(item["replacement_ready"])]
    fine_tunable = [item["id"] for item in steps if bool(item["fine_tuning_supported"])]
    portable_artifact_return = [
        item["id"]
        for item in steps
        if bool(item["replacement_ready"])
    ]
    interface_only_trainable = [
        item["id"]
        for item in steps
        if bool(item["fine_tuning_supported"]) and not bool(item["replacement_ready"])
    ]
    frozen_differentiable = [
        item["id"]
        for item in steps
        if item["gradient"] in _DIFFERENTIABLE_GRADIENTS and item["id"] not in selected_replacements
    ]
    exportable_differentiable = [
        item["id"]
        for item in steps
        if item["gradient"] in _DIFFERENTIABLE_GRADIENTS and bool(item["exportable"])
    ]
    frozen_exportable = [
        item["id"]
        for item in steps
        if item["gradient"] in _DIFFERENTIABLE_GRADIENTS
        and bool(item["exportable"])
        and item["id"] not in selected_replacements
    ]

    path_info = _training_paths(optimizable_ids, loss_ids, consumers, predecessors, step_reports)
    route_step_ids = {
        str(step_id)
        for path in path_info
        for step_id in list(path.get("downstream_route_steps") or [])
    }
    selected_downstream_support = [
        item["id"]
        for item in steps
        if item["id"] in route_step_ids
        and item["id"] not in selected_replacements
        and item["id"] not in selected_loss_ids
    ]
    clean_paths = [item for item in path_info if item["clean"]]
    gradient_breaks = _gradient_breaks(path_info)
    capture_possible = _capture_possible(recipe.steps, step_reports, explicit_loss_ids, loss_ids)
    path_issues = _path_issues(path_info, step_reports)
    if optimizable_ids and len(clean_paths) == len(path_info) and clean_paths:
        status = "full_gradient_possible"
        recommended_mode = "differentiable_export"
        gradient_path = _path_with_loss(clean_paths[0]["path"], set(loss_ids))
    elif clean_paths:
        status = "partial_gradient_possible"
        recommended_mode = "receiver_only"
        gradient_path = _path_with_loss(clean_paths[-1]["path"], set(loss_ids))
    elif capture_possible:
        status = "dataset_capture_only"
        recommended_mode = "dataset_capture"
        gradient_path = []
    else:
        status = "not_trainable"
        recommended_mode = "benchmark_run"
        gradient_path = []

    export_blockers = _export_blockers(path_info, step_reports)
    if status == "full_gradient_possible" and export_blockers:
        recommended_mode = "dataset_capture"
    suggested_capture_taps = _suggest_capture_taps(recipe.steps, registry)
    mode_explanation = _mode_explanation(status, recommended_mode, runner_support, gradient_breaks, export_blockers)

    return {
        "schema_version": 1,
        "recipe": recipe.name,
        "status": status,
        "recommended_mode": recommended_mode,
        "mode_explanation": mode_explanation,
        "runner_contracts": runner_contracts(),
        "runner_support": runner_support,
        "steps": steps,
        "loss_steps": sorted(loss_ids),
        "loss_step_candidates": sorted(loss_step_candidates),
        "selected_loss_steps": sorted(selected_loss_ids),
        "selected_replacement_steps": list(optimizable_ids),
        "selected_optimizable_steps": list(optimizable_ids),
        "gradient_path": gradient_path,
        "paths": path_info,
        "replacement_candidate_blocks": replacement_candidates,
        # Compatibility alias for v1 clients. These are replacement targets,
        # not necessarily operations with differentiable built-in parameters.
        "optimizable_candidate_blocks": replacement_candidates,
        "fine_tunable_blocks": fine_tunable,
        "portable_artifact_return_blocks": portable_artifact_return,
        "fine_tunable_without_artifact_return_blocks": interface_only_trainable,
        "interface_only_trainable_blocks": interface_only_trainable,
        "frozen_differentiable_blocks": frozen_differentiable,
        "exportable_differentiable_blocks": exportable_differentiable,
        "frozen_exportable_blocks": frozen_exportable,
        "selected_downstream_support_blocks": selected_downstream_support,
        "gradient_breaks": gradient_breaks,
        "export_blockers": export_blockers,
        "path_issues": path_issues,
        "suggested_capture_taps": suggested_capture_taps,
        "dataset_capture": {
            "possible": capture_possible,
            "reason": _capture_reason(recipe.steps, step_reports, explicit_loss_ids, loss_ids) if capture_possible else "",
        },
    }


def format_training_inspection_human(report: Mapping[str, Any]) -> str:
    lines = [
        "Recipe: %s" % report.get("recipe", ""),
        "Status: %s" % report.get("status", ""),
        "Recommendation: %s" % report.get("recommended_mode", ""),
    ]
    explanation = str(report.get("mode_explanation") or "").strip()
    if explanation:
        lines.append("Why: %s" % explanation)
    runner_support = report.get("runner_support") or {}
    lines.append(runner_summary_line(runner_support))
    gradient_path = list(report.get("gradient_path") or [])
    lines.append(
        "Gradient path: %s"
        % (" -> ".join(str(item) for item in gradient_path) if gradient_path else "none")
    )
    frozen = list(report.get("frozen_differentiable_blocks") or [])
    lines.append(
        "Frozen differentiable blocks: %s"
        % (", ".join(str(item) for item in frozen) if frozen else "none")
    )
    exportable = list(report.get("exportable_differentiable_blocks") or [])
    lines.append(
        "Exportable differentiable blocks: %s"
        % (", ".join(str(item) for item in exportable) if exportable else "none")
    )
    trainable = list(report.get("replacement_candidate_blocks") or report.get("optimizable_candidate_blocks") or [])
    lines.append(
        "Portable replacement blocks: %s"
        % (", ".join(str(item) for item in trainable) if trainable else "none")
    )
    fine_tunable = list(report.get("fine_tunable_blocks") or [])
    lines.append(
        "Built-in fine-tunable blocks: %s"
        % (", ".join(str(item) for item in fine_tunable) if fine_tunable else "none")
    )
    portable_return = list(report.get("portable_artifact_return_blocks") or [])
    lines.append(
        "Portable artifact-return blocks: %s"
        % (", ".join(str(item) for item in portable_return) if portable_return else "none")
    )
    interface_only = list(report.get("interface_only_trainable_blocks") or [])
    if interface_only:
        lines.append(
            "Fine-tunable blocks without portable return: %s" % ", ".join(str(item) for item in interface_only)
        )
    selected_optimizable = list(
        report.get("selected_replacement_steps")
        or report.get("selected_optimizable_steps")
        or []
    )
    if selected_optimizable:
        lines.append("Selected replacement blocks: %s" % ", ".join(str(item) for item in selected_optimizable))
    selected_loss = list(report.get("selected_loss_steps") or [])
    if selected_loss:
        lines.append("Selected loss steps: %s" % ", ".join(str(item) for item in selected_loss))
    paths = list(report.get("paths") or [])
    if paths:
        lines.append("Paths:")
        for item in paths:
            optimizable_id = item.get("replacement_step_id") or item.get("optimizable_step_id", "")
            path = item.get("replacement_to_loss_path") or item.get("optimizable_to_loss_path") or item.get("path") or []
            path_text = " -> ".join(str(part) for part in path) if path else "none"
            lines.append("  %s -> loss (representative): %s" % (optimizable_id, path_text))
            downstream_route = list(item.get("downstream_route_steps") or [])
            if downstream_route:
                lines.append(
                    "    complete downstream DAG: %s"
                    % ", ".join(str(part) for part in downstream_route)
                )
    breaks = list(report.get("gradient_breaks") or [])
    if breaks:
        lines.append("Gradient breaks:")
        for item in breaks:
            reason = item.get("reason") or item.get("gradient") or "gradient is not available"
            lines.append("  %s: %s" % (item.get("step_id", ""), reason))
    else:
        lines.append("Gradient breaks: none")
    blockers = list(report.get("export_blockers") or [])
    if blockers:
        lines.append("Export blockers:")
        for item in blockers:
            lines.append("  %s: %s" % (item.get("step_id", ""), item.get("reason", "not exportable")))
    issues = list(report.get("path_issues") or [])
    if issues:
        lines.append("Path issues:")
        for item in issues:
            lines.append("  %s: %s" % (item.get("step_id", ""), item.get("reason", "not on selected path")))
    taps = list(report.get("suggested_capture_taps") or [])
    if taps:
        lines.append("Suggested capture taps:")
        for item in taps:
            lines.append("  %s <- %s" % (item.get("id", ""), item.get("from", "")))
    return "\n".join(lines)


def _step_report(step: RecipeStep, registry: OperationRegistry) -> JsonDict:
    operation = registry.get(step.op).describe()
    differentiability = operation.get("differentiability") or {}
    training_capabilities = operation.get("training_capabilities") or {}
    runner_support = operation_runner_supports(operation)
    return {
        "id": step.id,
        "op": step.op,
        "framework": str(differentiability.get("framework", "numpy")),
        "gradient": str(differentiability.get("gradient", "none")),
        "trainable_params": bool(differentiability.get("trainable_params", False)),
        "fine_tuning_supported": bool(training_capabilities.get("built_in_fine_tuning", False)),
        "artifact_return_ready": bool(operation.get("trained_artifact_abi")),
        "replacement_ready": bool(training_capabilities.get("portable_replacement", False)),
        "exportable": bool(differentiability.get("exportable", False)),
        "reason": str(differentiability.get("reason", "")),
        "runner_support": runner_support,
    }


def _consumer_graph(steps: Iterable[RecipeStep]) -> Dict[str, Set[str]]:
    consumers: Dict[str, Set[str]] = defaultdict(set)
    for step in steps:
        consumers.setdefault(step.id, set())
        for reference in step.inputs.values():
            producer, _ = reference.split(".", 1)
            consumers[producer].add(step.id)
    return consumers


def _predecessor_graph(steps: Iterable[RecipeStep]) -> Dict[str, Set[str]]:
    predecessors: Dict[str, Set[str]] = defaultdict(set)
    for step in steps:
        predecessors.setdefault(step.id, set())
        for reference in step.inputs.values():
            producer, _ = reference.split(".", 1)
            predecessors[step.id].add(producer)
    return predecessors


def _loss_step_ids(steps: Iterable[RecipeStep]) -> Set[str]:
    return {
        step.id
        for step in steps
        if step.op.startswith(_LOSS_STEP_PREFIXES) or step.id in _LOSS_STEP_IDS
    }


def _terminal_step_ids(steps: Iterable[RecipeStep], consumers: Mapping[str, Set[str]]) -> Set[str]:
    return {step.id for step in steps if not consumers.get(step.id)}


def _normalize_step_selection(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    selected: List[str] = []
    seen: Set[str] = set()
    for item in raw_items:
        step_id = str(item).strip()
        if step_id and step_id not in seen:
            selected.append(step_id)
            seen.add(step_id)
    return selected


def _resolve_loss_ids(
    steps: Iterable[RecipeStep],
    consumers: Mapping[str, Set[str]],
    loss: Optional[Sequence[str] | str],
) -> Set[str]:
    step_list = list(steps)
    step_ids = {step.id for step in step_list}
    requested = _normalize_step_selection(loss)
    if requested:
        unknown = [step_id for step_id in requested if step_id not in step_ids]
        if unknown:
            raise ValueError("Loss step id is not in the recipe: %s" % ", ".join(unknown))
        return set(requested)
    loss_ids = _loss_step_ids(step_list)
    if loss_ids:
        return set(loss_ids)
    return _terminal_step_ids(step_list, consumers)


def _ancestors_of_any(step_ids: Iterable[str], predecessors: Mapping[str, Set[str]]) -> Set[str]:
    visited: Set[str] = set()
    queue: deque[str] = deque(step_ids)
    while queue:
        step_id = queue.popleft()
        if step_id in visited:
            continue
        visited.add(step_id)
        queue.extend(predecessors.get(step_id, set()) - visited)
    return visited


def _training_paths(
    optimizable_ids: Iterable[str],
    loss_ids: Iterable[str],
    consumers: Mapping[str, Set[str]],
    predecessors: Mapping[str, Set[str]],
    step_reports: Mapping[str, JsonDict],
) -> List[JsonDict]:
    loss_set = set(loss_ids)
    replacement_ids = list(optimizable_ids)
    replacement_set = set(replacement_ids)
    paths: List[JsonDict] = []
    for optimizable_id in replacement_ids:
        path = _shortest_path_to_loss(optimizable_id, loss_set, consumers)
        downstream_route = _downstream_route_to_losses(
            optimizable_id,
            loss_set,
            consumers,
            predecessors,
            ordered_ids=step_reports,
        )
        source_path = _shortest_source_path_to_step(optimizable_id, predecessors)
        # Selected targets become researcher-supplied placeholders. Their
        # original implementations are not executed, so their own gradient and
        # export metadata must never block replacement training.
        route_exclusions = loss_set | replacement_set
        path_breaks = (
            _path_gradient_breaks(
                downstream_route,
                step_reports,
                excluded_step_ids=route_exclusions,
            )
            if downstream_route
            else []
        )
        upstream_non_differentiable = _path_gradient_breaks(
            source_path,
            step_reports,
            excluded_step_ids={optimizable_id, source_path[0]} if source_path else {optimizable_id},
        )
        downstream_blockers = (
            _path_export_blockers(
                downstream_route,
                step_reports,
                excluded_step_ids=route_exclusions,
            )
            if downstream_route
            else []
        )
        upstream_non_exportable = _path_export_blockers(
            source_path,
            step_reports,
            excluded_step_ids={optimizable_id, source_path[0]} if source_path else {optimizable_id},
        )
        through_channel = (
            downstream_route
            if downstream_route and _path_has_channel(downstream_route, step_reports)
            else []
        )
        reachable_loss_steps = [
            step_id for step_id in downstream_route if step_id in loss_set
        ]
        step_report = step_reports.get(optimizable_id, {})
        issues: List[JsonDict] = []
        if not bool(step_report.get("replacement_ready")):
            issues.append(
                {
                    "step_id": optimizable_id,
                    "reason": "Selected block does not declare a complete trained-artifact replacement ABI.",
                }
            )
        if not path:
            issues.append(
                {
                    "step_id": optimizable_id,
                    "reason": "Selected replacement block is not on a path to the selected loss step.",
                }
            )
        paths.append(
            {
                "replacement_step_id": optimizable_id,
                "optimizable_step_id": optimizable_id,
                "path": path,
                "source_to_replacement_path": source_path,
                "source_to_optimizable_path": source_path,
                "replacement_to_loss_path": path,
                "optimizable_to_loss_path": path,
                # Complete induced DAG between this replacement and every
                # reachable selected loss. The singular path fields above are
                # retained only as compact display/compatibility values.
                "downstream_route_steps": downstream_route,
                "reachable_loss_steps": reachable_loss_steps,
                "replacement_channel_loss_path": through_channel,
                "optimizable_channel_loss_path": through_channel,
                "through_channel": bool(through_channel),
                "clean": bool(path) and not path_breaks and not issues,
                "reachable": bool(path),
                "breaks": path_breaks,
                "gradient_breaks": path_breaks,
                "upstream_gradient_required": False,
                "upstream_non_differentiable_steps": upstream_non_differentiable,
                "upstream_non_exportable_steps": upstream_non_exportable,
                # Compatibility keys are intentionally empty: upstream
                # producers supply captured/forward inputs and never block a
                # replacement model's backward route.
                "source_gradient_breaks": [],
                "export_blockers": downstream_blockers,
                "source_export_blockers": [],
                "path_issues": issues,
            }
        )
    return paths


def _downstream_route_to_losses(
    start: str,
    loss_ids: Set[str],
    consumers: Mapping[str, Set[str]],
    predecessors: Mapping[str, Set[str]],
    *,
    ordered_ids: Iterable[str],
) -> List[str]:
    """Return every node on any route from ``start`` to a selected loss."""

    forward = _reachable_steps(start, consumers)
    reverse = _ancestors_of_any(loss_ids, predecessors)
    route = forward.intersection(reverse)
    return [step_id for step_id in ordered_ids if step_id in route]


def _reachable_steps(
    start: str,
    adjacency: Mapping[str, Set[str]],
) -> Set[str]:
    visited: Set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        step_id = queue.popleft()
        if step_id in visited:
            continue
        visited.add(step_id)
        queue.extend(sorted(adjacency.get(step_id, set()) - visited))
    return visited


def _shortest_path_to_loss(
    start: str,
    loss_ids: Set[str],
    consumers: Mapping[str, Set[str]],
) -> List[str]:
    queue: deque[List[str]] = deque([[start]])
    visited: Set[str] = set()
    while queue:
        path = queue.popleft()
        step_id = path[-1]
        if step_id in visited:
            continue
        visited.add(step_id)
        if step_id in loss_ids:
            return path
        for child in sorted(consumers.get(step_id, set())):
            if child not in visited:
                queue.append([*path, child])
    return []


def _shortest_source_path_to_step(
    target: str,
    predecessors: Mapping[str, Set[str]],
) -> List[str]:
    queue: deque[List[str]] = deque([[target]])
    visited: Set[str] = set()
    while queue:
        reverse_path = queue.popleft()
        step_id = reverse_path[-1]
        if step_id in visited:
            continue
        visited.add(step_id)
        parents = sorted(predecessors.get(step_id, set()))
        if not parents:
            return list(reversed(reverse_path))
        for parent in parents:
            if parent not in visited:
                queue.append([*reverse_path, parent])
    return [target]


def _path_gradient_breaks(
    path: Iterable[str],
    step_reports: Mapping[str, JsonDict],
    *,
    excluded_step_ids: Set[str],
) -> List[JsonDict]:
    return [
        _break_report(step_reports[step_id])
        for step_id in path
        if step_id not in excluded_step_ids
        and step_reports[step_id]["gradient"] not in _DIFFERENTIABLE_GRADIENTS
    ]


def _path_export_blockers(
    path: Iterable[str],
    step_reports: Mapping[str, JsonDict],
    *,
    excluded_step_ids: Set[str],
) -> List[JsonDict]:
    blockers: List[JsonDict] = []
    for step_id in path:
        if step_id in excluded_step_ids:
            continue
        report = step_reports[step_id]
        if report["gradient"] in _DIFFERENTIABLE_GRADIENTS and not bool(report["exportable"]):
            blockers.append(
                {
                    "step_id": step_id,
                    "op": report["op"],
                    "reason": report.get("reason") or "Differentiable block is not exportable.",
                }
            )
            continue
        if report["gradient"] in _DIFFERENTIABLE_GRADIENTS and bool(report["exportable"]):
            train_support = dict((report.get("runner_support") or {}).get("differentiable_export") or {})
            if not train_support.get("supported", False):
                blockers.append(
                    {
                        "step_id": step_id,
                        "op": report["op"],
                        "reason": train_support.get("reason")
                        or "Differentiable block has no differentiable-export materialization.",
                    }
                )
    return blockers


def _path_has_channel(path: Iterable[str], step_reports: Mapping[str, JsonDict]) -> bool:
    return any(_is_channel_like_step(step_reports[step_id]) for step_id in path if step_id in step_reports)


def _is_channel_like_step(step_report: Mapping[str, Any]) -> bool:
    op = str(step_report.get("op") or "")
    step_id = str(step_report.get("id") or "")
    return (
        op.startswith(("wireless.", "channel.", "modulation.", "demodulation."))
        or "channel" in op
        or "demod" in op
        or "modulat" in op
        or "channel" in step_id
        or "demod" in step_id
        or "modulat" in step_id
    )


def _break_report(step_report: Mapping[str, Any]) -> JsonDict:
    reason = str(step_report.get("reason") or "").strip()
    if not reason:
        reason = "Operation gradient is %s." % step_report.get("gradient", "none")
    return {
        "step_id": step_report["id"],
        "op": step_report["op"],
        "gradient": step_report["gradient"],
        "reason": reason,
    }


def _gradient_breaks(paths: Iterable[Mapping[str, Any]]) -> List[JsonDict]:
    by_id: Dict[str, JsonDict] = {}
    for path in paths:
        for item in path.get("breaks") or []:
            by_id.setdefault(str(item["step_id"]), dict(item))
    return [by_id[key] for key in sorted(by_id)]


def _export_blockers(paths: Iterable[Mapping[str, Any]], step_reports: Mapping[str, JsonDict]) -> List[JsonDict]:
    blockers: Dict[str, JsonDict] = {}
    for path in paths:
        for item in path.get("export_blockers") or []:
            blockers.setdefault(str(item["step_id"]), dict(item))
    return [blockers[key] for key in sorted(blockers)]


def _mode_explanation(
    status: str,
    recommended_mode: str,
    runner_support: Mapping[str, Any],
    gradient_breaks: Sequence[Mapping[str, Any]],
    export_blockers: Sequence[Mapping[str, Any]],
) -> str:
    benchmark_summary = dict((runner_support.get("summary") or {}).get("benchmark_run") or {})
    can_benchmark = bool(benchmark_summary.get("supported"))
    if can_benchmark and recommended_mode == "differentiable_export":
        return (
            "This recipe can benchmark and can replace the selected block(s): every unchanged "
            "downstream block on the selected loss route has differentiable-export support."
        )
    if can_benchmark and gradient_breaks:
        first = dict(gradient_breaks[0])
        return "This recipe can benchmark, but the replacement-to-loss route has a gradient break at `%s`: %s" % (
            first.get("step_id", "a step"),
            first.get("reason", "gradient is unavailable"),
        )
    if can_benchmark and export_blockers:
        first = dict(export_blockers[0])
        return "This recipe can benchmark, but cannot export the selected differentiable path because `%s` is unsupported by the differentiable exporter: %s" % (
            first.get("step_id", "a step"),
            first.get("reason", "differentiable-export materialization is unavailable"),
        )
    if can_benchmark and status == "dataset_capture_only":
        return "This recipe can benchmark and capture supervised data, but no clean replacement-to-loss gradient route was found."
    if can_benchmark:
        return "This recipe can benchmark, but no portable replacement target or downstream differentiable route was found."
    blocker = (unsupported_steps_for_runner(runner_support, "benchmark_run") or [{}])[0]
    return "This recipe is not benchmark-ready because `%s` cannot be materialized by the benchmark runner: %s" % (
        blocker.get("step_id", "a step"),
        blocker.get("reason", "benchmark materialization is unavailable"),
    )


def _path_issues(paths: Iterable[Mapping[str, Any]], step_reports: Mapping[str, JsonDict]) -> List[JsonDict]:
    issues: List[JsonDict] = []
    for path in paths:
        issues.extend(dict(item) for item in path.get("path_issues") or [])
    return issues


def _path_with_loss(path: List[str], loss_ids: Set[str]) -> List[str]:
    if path and path[-1] in loss_ids:
        return [*path[:-1], "loss"]
    return path


def _suggest_capture_taps(steps: Iterable[RecipeStep], registry: OperationRegistry) -> List[JsonDict]:
    candidates: List[JsonDict] = []
    for step in steps:
        output_kinds = dict(registry.get(step.op).output_kinds)
        for output_name, kind in output_kinds.items():
            reference = "%s.%s" % (step.id, output_name)
            for suggestion in _tap_suggestions_for_kind(str(kind), reference):
                candidates.append(suggestion)
    selected: Dict[str, JsonDict] = {}
    for candidate in candidates:
        tap_id = str(candidate["id"])
        existing = selected.get(tap_id)
        if existing is None or int(candidate.get("priority", 100)) < int(existing.get("priority", 100)):
            selected[tap_id] = candidate
    order = [
        "rx_symbols",
        "llr",
        "target_bits",
        "payload_bits",
        "reference_images",
        "reference_texts",
        "target_labels",
        "semantic_state",
    ]
    ordered: List[JsonDict] = []
    for tap_id in order:
        if tap_id in selected:
            ordered.append(_public_tap_suggestion(selected.pop(tap_id)))
    ordered.extend(_public_tap_suggestion(selected[key]) for key in sorted(selected))
    return ordered[:8]


def _public_tap_suggestion(candidate: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in candidate.items() if key != "priority"}


def _tap_suggestions_for_kind(kind: str, reference: str) -> List[JsonDict]:
    suggestions: List[JsonDict] = []
    if kind == "channel.rx_symbols.complex_numpy":
        suggestions.append(
            {
                "id": "rx_symbols",
                "from": reference,
                "kind": kind,
                "priority": 10,
                "reason": "Receiver training feature: noisy received channel symbols.",
            }
        )
    elif kind == "channel.llr.numpy":
        suggestions.append(
            {
                "id": "llr",
                "from": reference,
                "kind": kind,
                "priority": 10,
                "reason": "Receiver training feature: soft demapper log-likelihood ratios.",
            }
        )
    elif kind == "channel.coded_bits.numpy":
        suggestions.append(
            {
                "id": "target_bits",
                "from": reference,
                "kind": kind,
                "priority": 10,
                "reason": "Receiver training target: protected coded bits before the noisy channel.",
            }
        )
    elif kind == "channel.payload_bits.numpy":
        suggestions.append(
            {
                "id": "payload_bits",
                "from": reference,
                "kind": kind,
                "priority": 20,
                "reason": "Payload target or source-code tap before channel protection.",
            }
        )
    elif kind == "channel.bits.numpy":
        suggestions.append(
            {
                "id": "target_bits",
                "from": reference,
                "kind": kind,
                "priority": 40,
                "reason": "Generic bit target for receiver or offline supervised training.",
            }
        )
    elif kind == "image.batch.numpy":
        suggestions.append(
            {
                "id": "reference_images",
                "from": reference,
                "kind": kind,
                "priority": 30,
                "reason": "Reference image targets for reconstruction or task-supervised training.",
            }
        )
    elif kind == "text.batch.json":
        suggestions.append(
            {
                "id": "reference_texts",
                "from": reference,
                "kind": kind,
                "priority": 30,
                "reason": "Reference text targets for semantic text training.",
            }
        )
    elif kind == "task.labels.json":
        suggestions.append(
            {
                "id": "target_labels",
                "from": reference,
                "kind": kind,
                "priority": 20,
                "reason": "Task labels for classifier or task-head training.",
            }
        )
    elif kind == "semantic.state.json":
        suggestions.append(
            {
                "id": "semantic_state",
                "from": reference,
                "kind": kind,
                "priority": 25,
                "reason": "Semantic artifact target or intermediate state.",
            }
        )
    return suggestions


def _capture_possible(
    steps: Iterable[RecipeStep],
    step_reports: Mapping[str, JsonDict],
    explicit_loss_ids: Set[str],
    loss_ids: Set[str],
) -> bool:
    if explicit_loss_ids:
        return len(step_reports) >= 2
    if loss_ids and len(step_reports) >= 2:
        return True
    output_like_steps = [
        step
        for step in steps
        if any(token in step.op for token in ("channel", "receiver", "decode", "demod", "payload"))
    ]
    return bool(output_like_steps)


def _capture_reason(
    steps: Iterable[RecipeStep],
    step_reports: Mapping[str, JsonDict],
    explicit_loss_ids: Set[str],
    loss_ids: Set[str],
) -> str:
    if explicit_loss_ids:
        return "Recipe has evaluation/loss steps, so Noema can capture supervised input/target pairs at declared taps."
    if loss_ids and len(step_reports) >= 2:
        return "Recipe has terminal artifacts that can be captured as offline training targets."
    if _capture_possible(steps, step_reports, explicit_loss_ids, loss_ids):
        return "Recipe has channel or receiver-side artifacts that can be captured for offline training."
    return ""
