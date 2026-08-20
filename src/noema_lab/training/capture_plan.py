from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import Recipe


JsonDict = Dict[str, Any]


class TrainingCapturePlanError(ValueError):
    """Raised when a recipe's persisted training-capture plan is invalid."""


@dataclass(frozen=True)
class CaptureTap:
    id: str
    reference: str
    role: str = "additional_signal"
    required: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "from": self.reference,
            "role": self.role,
            "required": bool(self.required),
        }


@dataclass(frozen=True)
class CaptureSplit:
    id: str
    percentage: float
    samples: int
    seed_offset: int
    training_use: str

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "percentage": self.percentage,
            "requested_samples": self.samples,
            "seed_offset": self.seed_offset,
            "training_use": self.training_use,
        }


@dataclass(frozen=True)
class TrainingCapturePlan:
    mode: str
    sample_unit: str
    total_samples: int
    taps: Tuple[CaptureTap, ...]
    splits: Tuple[CaptureSplit, ...]
    source: str
    candidates: Tuple[JsonDict, ...]

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "sample_unit": self.sample_unit,
            "total_samples": self.total_samples,
            "source": self.source,
            "taps": [tap.to_dict() for tap in self.taps],
            "split_plan": {
                "allocation": "deterministic_largest_remainder",
                "percentages": {
                    split.id: split.percentage for split in self.splits
                },
                "counts": {split.id: split.samples for split in self.splits},
                "splits": [split.to_dict() for split in self.splits],
            },
        }


_SPLIT_IDS = ("train", "validation", "test")
_DEFAULT_PERCENTAGES = {
    "train": 66.6666666667,
    "validation": 16.66666666665,
    "test": 16.66666666665,
}
_SEED_OFFSETS = {"train": 1000, "validation": 2000, "test": 3000}


def capture_tap_candidates(recipe: Recipe, registry: OperationRegistry) -> list[JsonDict]:
    """List recipe outputs that the NPZ capture runner can materialize."""

    candidates: list[JsonDict] = []
    for step in recipe.steps:
        operation = registry.get(step.op).describe()
        output_kinds = dict(operation.get("output_kinds") or {})
        for output_name, kind_value in output_kinds.items():
            kind = str(kind_value or "")
            # The capture runner currently consumes array artifacts.  Keep the
            # candidate list honest instead of offering metric reports or other
            # operation-defined non-array files that will fail at execution.
            if not _array_kind(kind):
                continue
            reference = "%s.%s" % (step.id, output_name)
            candidates.append(
                {
                    "id": _safe_tap_id(reference),
                    "from": reference,
                    "kind": kind,
                    "step_id": step.id,
                    "step_operation": step.op,
                    "output": str(output_name),
                    "capturable": True,
                }
            )
    return candidates


def replacement_boundary_capture_requirements(
    recipe: Recipe,
    registry: OperationRegistry,
    selected_step_ids: Sequence[str],
) -> JsonDict:
    """Derive capturable inputs entering externally implemented replacement slots.

    Connections between two jointly selected slots stay inside the researcher
    model and are intentionally not captured. For a portable replacement, the
    operation-owned artifact ABI decides which operation inputs actually enter
    the returned model. Pass-through inputs owned by the surrounding operation
    wrapper are not silently promoted to model-training inputs.
    """

    selected = {str(item).strip() for item in selected_step_ids if str(item).strip()}
    step_by_id = {step.id: step for step in recipe.steps}
    unknown = sorted(selected - set(step_by_id))
    if unknown:
        raise TrainingCapturePlanError(
            "Replacement step id is not in the recipe: %s" % ", ".join(unknown)
        )
    candidates = capture_tap_candidates(recipe, registry)
    candidate_by_reference = {str(item["from"]): item for item in candidates}
    required: list[JsonDict] = []
    unsupported: list[JsonDict] = []
    seen = set()
    for step in recipe.steps:
        if step.id not in selected:
            continue
        operation = registry.get(step.op).describe()
        artifact_abi = dict(operation.get("trained_artifact_abi") or {})
        abi_input_names = artifact_abi.get("required_operation_inputs")
        if isinstance(abi_input_names, list):
            boundary_inputs = [
                (str(input_name), step.inputs.get(str(input_name)))
                for input_name in abi_input_names
            ]
        else:
            # Compatibility fallback for older non-portable contracts. New
            # portable slots always declare required_operation_inputs.
            boundary_inputs = list(step.inputs.items())
        for input_name, raw_reference in boundary_inputs:
            reference = str(raw_reference or "").strip()
            producer_id = reference.split(".", 1)[0]
            if (
                not reference
                or producer_id in selected
                or _inside_selected_group(recipe, selected, step.id, producer_id)
                or reference in seen
            ):
                continue
            seen.add(reference)
            candidate = candidate_by_reference.get(reference)
            role = "replacement_input:%s.%s" % (step.id, input_name)
            if candidate is None:
                unsupported.append(
                    {
                        "step_id": step.id,
                        "input": str(input_name),
                        "from": reference,
                        "reason": "The generic NPZ capture runner supports array-valued recipe outputs only.",
                    }
                )
                continue
            required.append(
                {
                    "id": str(candidate.get("id") or _safe_tap_id(reference)),
                    "from": reference,
                    "role": role,
                }
            )
    return {
        "required_taps": required,
        "unsupported_inputs": unsupported,
    }


def _inside_selected_group(
    recipe: Recipe,
    selected_step_ids: set[str],
    consumer_id: str,
    producer_id: str,
) -> bool:
    """Return true when an input lies on a route between selected slots."""

    consumers: Dict[str, set[str]] = {step.id: set() for step in recipe.steps}
    for step in recipe.steps:
        for reference in step.inputs.values():
            upstream = str(reference).split(".", 1)[0]
            if upstream:
                consumers.setdefault(upstream, set()).add(step.id)
    for source_id in selected_step_ids:
        if source_id == consumer_id:
            continue
        pending = [source_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current == producer_id:
                return True
            if current in visited or current == consumer_id:
                continue
            visited.add(current)
            pending.extend(consumers.get(current, set()) - visited)
    return False


def resolve_training_capture_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    required_taps: Sequence[Mapping[str, Any]],
    sample_unit: str,
    suggested_total_samples: Optional[int] = None,
    suggested_percentages: Optional[Mapping[str, Any]] = None,
) -> TrainingCapturePlan:
    """Resolve and validate the capture plan persisted in ``recipe.dataset_capture``.

    ``dataset_capture.samples`` remains accepted as the recipe-level total for
    existing recipes.  New UI code should persist the clearer
    ``dataset_capture.split_plan.total_samples`` and percentages/counts.
    """

    config = dict(recipe.dataset_capture or {})
    split_config = config.get("split_plan") or {}
    if not isinstance(split_config, Mapping):
        raise TrainingCapturePlanError("dataset_capture.split_plan must be a mapping")
    split_config = dict(split_config)

    candidates = capture_tap_candidates(recipe, registry)
    candidate_by_reference = {str(item["from"]): item for item in candidates}
    normalized_required = _normalize_required_taps(required_taps)
    configured_taps = config.get("taps")
    source_parts = []
    if configured_taps is None:
        selected = [
            {"id": item["id"], "from": item["from"]}
            for item in normalized_required
        ]
        source_parts.append("suggested_required_taps")
    else:
        if not isinstance(configured_taps, list):
            raise TrainingCapturePlanError("dataset_capture.taps must be a list")
        selected = list(configured_taps)
        source_parts.append("recipe_taps")

    required_by_reference = {
        str(item["from"]): item for item in normalized_required
    }
    taps: list[CaptureTap] = []
    seen_ids = set()
    seen_references = set()
    for index, value in enumerate(selected):
        if not isinstance(value, Mapping):
            raise TrainingCapturePlanError(
                "dataset_capture.taps[%d] must be a mapping" % index
            )
        reference = str(value.get("from") or "").strip()
        tap_id = str(value.get("id") or _safe_tap_id(reference)).strip()
        if not reference:
            raise TrainingCapturePlanError(
                "dataset_capture.taps[%d] requires from=<step_id>.<output>" % index
            )
        if reference not in candidate_by_reference:
            raise TrainingCapturePlanError(
                "Captured signal %s is not a capturable array output in this recipe"
                % reference
            )
        if not tap_id:
            raise TrainingCapturePlanError(
                "dataset_capture.taps[%d] requires a non-empty id" % index
            )
        if tap_id in seen_ids:
            raise TrainingCapturePlanError("Duplicate dataset-capture tap id: %s" % tap_id)
        if reference in seen_references:
            raise TrainingCapturePlanError(
                "Captured signal is selected more than once: %s" % reference
            )
        seen_ids.add(tap_id)
        seen_references.add(reference)
        required = required_by_reference.get(reference)
        taps.append(
            CaptureTap(
                id=tap_id,
                reference=reference,
                role=str((required or {}).get("role") or "additional_signal"),
                required=required is not None,
            )
        )

    missing = [
        item for item in normalized_required if str(item["from"]) not in seen_references
    ]
    if missing:
        details = ", ".join(
            "%s (%s)" % (item["from"], item.get("role") or "required")
            for item in missing
        )
        raise TrainingCapturePlanError(
            "The selected captured signals omit required training-contract signal(s): %s"
            % details
        )

    total_value = split_config.get("total_samples", config.get("samples"))
    if total_value is None:
        if suggested_total_samples is None:
            raise TrainingCapturePlanError(
                "Set dataset_capture.split_plan.total_samples before exporting a capture-backed training contract"
            )
        total_value = suggested_total_samples
        source_parts.append("suggested_total")
    else:
        source_parts.append(
            "recipe_split_total"
            if split_config.get("total_samples") is not None
            else "recipe_samples_total"
        )
    total_samples = _positive_integer(total_value, "dataset_capture.split_plan.total_samples")

    counts_value = split_config.get("counts")
    percentages_value = split_config.get("percentages")
    if counts_value is not None:
        counts = _normalize_counts(counts_value)
        if sum(counts.values()) != total_samples:
            raise TrainingCapturePlanError(
                "dataset_capture.split_plan.counts must sum to total_samples=%d; got %d"
                % (total_samples, sum(counts.values()))
            )
        percentages = {
            split: (100.0 * counts[split] / float(total_samples))
            for split in _SPLIT_IDS
        }
        if percentages_value is not None:
            supplied = _normalize_percentages(percentages_value)
            supplied_counts = allocate_split_counts(total_samples, supplied)
            if supplied_counts != counts:
                raise TrainingCapturePlanError(
                    "dataset_capture.split_plan percentages and counts describe different splits"
                )
        source_parts.append("recipe_split_counts")
    else:
        if percentages_value is None:
            percentages_value = suggested_percentages or _DEFAULT_PERCENTAGES
            source_parts.append("suggested_percentages")
        else:
            source_parts.append("recipe_split_percentages")
        percentages = _normalize_percentages(percentages_value)
        counts = allocate_split_counts(total_samples, percentages)

    if any(counts[split] <= 0 for split in _SPLIT_IDS):
        raise TrainingCapturePlanError(
            "Training, validation, and held-out test splits must each contain at least one %s"
            % str(sample_unit or "sample")
        )

    splits = tuple(
        CaptureSplit(
            id=split,
            percentage=round(float(percentages[split]), 10),
            samples=int(counts[split]),
            seed_offset=_SEED_OFFSETS[split],
            training_use="held_out_evaluation" if split == "test" else split,
        )
        for split in _SPLIT_IDS
    )
    return TrainingCapturePlan(
        mode="captured_tensors",
        sample_unit=str(sample_unit or "samples"),
        total_samples=total_samples,
        taps=tuple(taps),
        splits=splits,
        source="+".join(source_parts),
        candidates=tuple(candidates),
    )


def capture_plan_inspection(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    required_taps: Sequence[Mapping[str, Any]],
    sample_unit: str,
    suggested_total_samples: Optional[int] = None,
    suggested_percentages: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    required = _normalize_required_taps(required_taps)
    selected_rows = list((recipe.dataset_capture or {}).get("taps") or [])
    selected_references = {
        str(item.get("from") or "")
        for item in selected_rows
        if isinstance(item, Mapping)
    }
    required_by_reference = {str(item["from"]): item for item in required}
    candidates = []
    for item in capture_tap_candidates(recipe, registry):
        row = dict(item)
        required_item = required_by_reference.get(str(row["from"]))
        row.update(
            {
                "selected": str(row["from"]) in selected_references,
                "required": required_item is not None,
                "role": str((required_item or {}).get("role") or "additional_signal"),
                "selectable": True,
                "reason": "",
            }
        )
        candidates.append(row)
    try:
        plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=required,
            sample_unit=sample_unit,
            suggested_total_samples=suggested_total_samples,
            suggested_percentages=suggested_percentages,
        )
    except TrainingCapturePlanError as exc:
        return {
            "mode": "captured_tensors",
            "ready": False,
            "issue": str(exc),
            "sample_unit": sample_unit,
            "candidates": candidates,
            "required_taps": required,
            "selected_taps": selected_rows,
            "suggested_total_samples": suggested_total_samples,
        }
    payload = plan.to_dict()
    resolved_references = {tap.reference for tap in plan.taps}
    for candidate in candidates:
        candidate["selected"] = str(candidate["from"]) in resolved_references
    payload.update(
        {
            "ready": True,
            "issue": "",
            "candidates": candidates,
            "required_taps": required,
            "selected_taps": [
                {"id": tap.id, "from": tap.reference}
                for tap in plan.taps
            ],
            "suggested_total_samples": suggested_total_samples,
        }
    )
    return payload


def allocate_split_counts(
    total_samples: int, percentages: Mapping[str, Any]
) -> Dict[str, int]:
    """Allocate an exact total with deterministic largest-remainder rounding."""

    total = _positive_integer(total_samples, "total_samples")
    normalized = _normalize_percentages(percentages)
    quotas = {
        split: total * normalized[split] / 100.0 for split in _SPLIT_IDS
    }
    counts = {split: int(math.floor(quotas[split])) for split in _SPLIT_IDS}
    remaining = total - sum(counts.values())
    order = sorted(
        _SPLIT_IDS,
        key=lambda split: (-(quotas[split] - counts[split]), _SPLIT_IDS.index(split)),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _normalize_required_taps(values: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    result = []
    seen = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TrainingCapturePlanError("required_taps[%d] must be a mapping" % index)
        reference = str(value.get("from") or "").strip()
        if not reference or "." not in reference:
            raise TrainingCapturePlanError(
                "required_taps[%d] requires from=<step_id>.<output>" % index
            )
        if reference in seen:
            continue
        seen.add(reference)
        result.append(
            {
                "id": str(value.get("id") or _safe_tap_id(reference)),
                "from": reference,
                "role": str(value.get("role") or "required_signal"),
            }
        )
    return result


def _normalize_percentages(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise TrainingCapturePlanError(
            "dataset_capture.split_plan.percentages must be a mapping"
        )
    unknown = set(str(key) for key in value) - set(_SPLIT_IDS)
    missing = set(_SPLIT_IDS) - set(str(key) for key in value)
    if unknown or missing:
        raise TrainingCapturePlanError(
            "split percentages require exactly train, validation, and test"
        )
    result = {}
    for split in _SPLIT_IDS:
        try:
            number = float(value[split])
        except (TypeError, ValueError) as exc:
            raise TrainingCapturePlanError(
                "split percentage %s must be numeric" % split
            ) from exc
        if not math.isfinite(number) or number <= 0.0:
            raise TrainingCapturePlanError(
                "split percentage %s must be finite and greater than zero" % split
            )
        result[split] = number
    total = sum(result.values())
    if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise TrainingCapturePlanError(
            "train, validation, and test percentages must sum to 100; got %.10g"
            % total
        )
    return result


def _normalize_counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        raise TrainingCapturePlanError(
            "dataset_capture.split_plan.counts must be a mapping"
        )
    if set(str(key) for key in value) != set(_SPLIT_IDS):
        raise TrainingCapturePlanError(
            "split counts require exactly train, validation, and test"
        )
    return {
        split: _positive_integer(value[split], "split count %s" % split)
        for split in _SPLIT_IDS
    }


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TrainingCapturePlanError("%s must be a positive integer" % label)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingCapturePlanError("%s must be a positive integer" % label) from exc
    try:
        exact = float(value)
    except (TypeError, ValueError):
        exact = float(number)
    if number <= 0 or not math.isfinite(exact) or exact != float(number):
        raise TrainingCapturePlanError("%s must be a positive integer" % label)
    return number


def _array_kind(kind: str) -> bool:
    text = str(kind or "").strip().lower()
    return text.endswith("numpy")


def _safe_tap_id(reference: str) -> str:
    text = str(reference or "signal")
    safe = "".join(character if character.isalnum() else "_" for character in text)
    return safe.strip("_") or "signal"
