from __future__ import annotations

import json
from typing import Any, Dict, List

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


TASK_SMOKE_EXAMPLES = [
    {"id": "signal_clear", "label": "clear", "prediction": "clear"},
    {"id": "signal_faded", "label": "faded", "prediction": "faded"},
    {"id": "signal_blocked", "label": "blocked", "prediction": "clear"},
    {"id": "signal_noisy", "label": "noisy", "prediction": "noisy"},
]


class TaskLabelsSmokeOperation(Operation):
    id = "source.task_labels_smoke"
    name = "Task-oriented label/prediction smoke source"
    output_kinds = {
        "reference": "task.labels.json",
        "candidate": "task.predictions.json",
    }
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "task_smoke",
                "enum": ["task_smoke"],
            },
            "sample_ids": {"type": "string", "default": ""},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset") or "task_smoke")
        if dataset != "task_smoke":
            raise RuntimeError("Unsupported task smoke dataset: %s" % dataset)
        sample_ids = _parse_sample_ids(ctx.params.get("sample_ids"))
        examples = [dict(item) for item in TASK_SMOKE_EXAMPLES if not sample_ids or item["id"] in sample_ids]
        if not examples:
            raise RuntimeError("No task smoke examples selected")
        reference_examples = [{"id": item["id"], "label": item["label"]} for item in examples]
        candidate_examples = [{"id": item["id"], "prediction": item["prediction"]} for item in examples]
        reference_payload = {
            "schema_version": 1,
            "kind": "task.labels",
            "dataset": dataset,
            "examples": reference_examples,
        }
        candidate_payload = {
            "schema_version": 1,
            "kind": "task.predictions",
            "dataset": dataset,
            "examples": candidate_examples,
        }
        metadata = {
            "dataset": dataset,
            "sample_ids": [item["id"] for item in examples],
            "example_count": len(examples),
        }
        reference_path = ctx.output_path("reference", ".json")
        candidate_path = ctx.output_path("candidate", ".json")
        reference_path.write_text(json.dumps(reference_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={
                "reference": artifact("task.labels.json", reference_path, metadata),
                "candidate": artifact("task.predictions.json", candidate_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples)},
            metadata=metadata,
        )


def _parse_sample_ids(value: Any) -> List[str]:
    if value is None:
        return []
    ids = [item.strip() for item in str(value).split(",") if item.strip()]
    valid = {item["id"] for item in TASK_SMOKE_EXAMPLES}
    unknown = [item for item in ids if item not in valid]
    if unknown:
        raise RuntimeError("Unknown task smoke sample id: %s" % ", ".join(unknown))
    return ids
