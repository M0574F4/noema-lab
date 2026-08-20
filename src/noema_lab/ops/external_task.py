from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.structured_input import decode_strict_json
from noema_lab.ops.models.external import _load_callable

JsonDict = Dict[str, Any]


_TASK_ADAPTER_SCHEMA = {
    "path": {"type": "string", "default": ""},
    "module": {"type": "string", "default": ""},
    "callable": {"type": "string", "default": ""},
    "call_style": {
        "type": "string",
        "default": "dict",
        "enum": ["dict", "params", "none"],
    },
}


class ExternalClassificationDatasetOperation(Operation):
    id = "source.external_classification_dataset"
    name = "External classification dataset source"
    output_kinds = {
        "reference": "task.labels.json",
        "candidate": "task.predictions.json",
    }
    params_schema = object_schema(_TASK_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        result = _call_task_external(ctx.params, {"kind": "classification_dataset"})
        reference, candidate, metadata = _classification_payload(result)
        dataset = str(metadata.get("dataset") or ctx.params.get("dataset") or "external_classification")
        reference_payload = {
            "schema_version": 1,
            "kind": "task.labels",
            "dataset": dataset,
            "examples": reference,
        }
        candidate_payload = {
            "schema_version": 1,
            "kind": "task.predictions",
            "dataset": dataset,
            "examples": candidate,
        }
        metadata.update(
            {
                "dataset": dataset,
                "example_count": len(reference),
                "external_adapter_contract": "source.external_classification_dataset",
            }
        )
        reference_path = ctx.output_path("reference", ".json")
        candidate_path = ctx.output_path("candidate", ".json")
        reference_path.write_text(json.dumps(reference_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={
                "reference": artifact("task.labels.json", reference_path, metadata),
                "candidate": artifact("task.predictions.json", candidate_path, metadata),
            },
            metrics={"task.dataset.example_count": len(reference)},
            metadata=metadata,
        )


class ExternalClassificationMetricOperation(Operation):
    id = "metrics.external_classification"
    name = "External classification metric"
    input_kinds = {
        "reference": ["task.labels.json"],
        "candidate": ["task.predictions.json", "task.labels.json"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(_TASK_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        reference = _load_examples(ctx.require_input("reference").path)
        candidate = _load_examples(ctx.require_input("candidate").path)
        result = _call_task_external(
            ctx.params,
            {
                "kind": "classification_metric",
                "reference": reference,
                "candidate": candidate,
            },
        )
        report = _metric_payload(result)
        report.setdefault("schema_version", 1)
        report.setdefault("metric_family", "external_classification")
        report.setdefault("num_examples", len(report.get("per_example") or []))
        report.setdefault("per_example", [])
        report.setdefault("metadata", {})
        report["metadata"].update({"external_adapter_contract": "metrics.external_classification"})
        metrics = dict(report.get("metrics") or {})
        report_path = ctx.output_path("report", ".json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", report_path, report)},
            metrics=metrics,
            metadata=dict(report.get("metadata") or {}),
        )


def _call_task_external(params: JsonDict, payload: JsonDict) -> Any:
    function = _load_callable(params)
    adapter_params = {
        key: value
        for key, value in params.items()
        if key not in {"path", "module", "callable", "call_style", "runner"}
    }
    call_style = str(params.get("call_style") or "dict")
    if call_style == "none":
        return function()
    if call_style == "params":
        return function(adapter_params)
    request = {"params": adapter_params}
    request.update(payload)
    return function(request)


def _classification_payload(result: Any) -> Tuple[List[JsonDict], List[JsonDict], JsonDict]:
    if not isinstance(result, Mapping):
        raise OperationError("External classification dataset callable must return a mapping")
    metadata = dict(result.get("metadata") or {})
    examples = result.get("examples")
    if isinstance(examples, list):
        reference = []
        candidate = []
        for index, item in enumerate(examples):
            if not isinstance(item, Mapping):
                continue
            example_id = str(item.get("id") or "example_%03d" % (index + 1))
            reference.append({"id": example_id, "label": str(item.get("label") or item.get("answer") or "")})
            candidate.append({"id": example_id, "prediction": str(item.get("prediction") or item.get("label") or "")})
        return reference, candidate, metadata
    reference = _examples_from_value(result.get("reference") or result.get("labels"), "label")
    candidate = _examples_from_value(result.get("candidate") or result.get("predictions"), "prediction")
    if not reference or not candidate:
        raise OperationError("External classification dataset must return examples or reference/candidate lists")
    return reference, candidate, metadata


def _metric_payload(result: Any) -> JsonDict:
    if not isinstance(result, Mapping):
        raise OperationError("External classification metric callable must return a mapping")
    if "metrics" in result:
        payload = dict(result)
        payload["metrics"] = dict(payload.get("metrics") or {})
        if "rows" in payload and "per_example" not in payload:
            payload["per_example"] = payload.pop("rows")
        return payload
    metrics = {
        str(key): value
        for key, value in result.items()
        if isinstance(value, (int, float, bool))
    }
    if not metrics:
        raise OperationError("External classification metric did not return numeric metrics")
    return {"metrics": metrics, "per_example": []}


def _examples_from_value(value: Any, field_name: str) -> List[JsonDict]:
    if not isinstance(value, list):
        return []
    rows = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            example_id = str(item.get("id") or "example_%03d" % (index + 1))
            rows.append({"id": example_id, field_name: str(item.get(field_name) or item.get("label") or item.get("prediction") or "")})
        else:
            rows.append({"id": "example_%03d" % (index + 1), field_name: str(item)})
    return rows


def _load_examples(path) -> List[JsonDict]:
    data = decode_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("examples"), list):
        raise OperationError("Expected task examples JSON artifact: %s" % path)
    examples: List[JsonDict] = []
    for index, item in enumerate(data["examples"]):
        if not isinstance(item, Mapping):
            raise OperationError(
                "Task examples item %d must be an object in %s" % (index, path)
            )
        examples.append(dict(item))
    return examples
