from __future__ import annotations

import json
from typing import Any, Dict, List

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]

TEXT_SMOKE_EXAMPLES = [
    {
        "id": "weather_report",
        "text": "A researcher transmits a short weather report over a noisy semantic channel.",
    },
    {
        "id": "robot_instruction",
        "text": "The robot should inspect the red valve and report whether it is open.",
    },
    {
        "id": "medical_note",
        "text": "The patient describes mild chest discomfort after climbing the stairs.",
    },
    {
        "id": "traffic_alert",
        "text": "A traffic alert says the northbound lane is closed near the bridge.",
    },
    {
        "id": "sensor_summary",
        "text": "Temperature stayed stable while humidity increased during the experiment.",
    },
]
DEFAULT_TEXT_SAMPLE_IDS = ",".join(example["id"] for example in TEXT_SMOKE_EXAMPLES)


class TextDatasetOperation(Operation):
    id = "source.text_dataset"
    name = "Text semantic dataset batch"
    output_kinds = {"texts": "text.batch.json"}
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "semantic_text_smoke",
                "enum": ["semantic_text_smoke"],
            },
            "sample_ids": {"type": "string", "default": DEFAULT_TEXT_SAMPLE_IDS},
            "repeat_count": {"type": "integer", "default": 1, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset") or "semantic_text_smoke")
        if dataset != "semantic_text_smoke":
            raise RuntimeError("Unsupported text dataset: %s" % dataset)
        sample_ids = _parse_sample_ids(str(ctx.params.get("sample_ids") or DEFAULT_TEXT_SAMPLE_IDS))
        repeat_count = max(1, int(ctx.params.get("repeat_count") or 1))
        examples_by_id = {example["id"]: example for example in TEXT_SMOKE_EXAMPLES}
        base_examples = [dict(examples_by_id[sample_id]) for sample_id in sample_ids]
        examples: List[JsonDict] = []
        total = max(len(base_examples) * repeat_count, 1)
        completed = 0
        for repeat_index in range(repeat_count):
            for example in base_examples:
                item = dict(example)
                if repeat_count > 1:
                    item["id"] = "%s_r%d" % (item["id"], repeat_index + 1)
                    item["base_id"] = example["id"]
                    item["repeat_index"] = repeat_index
                examples.append(item)
                completed += 1
                _report_text_data_progress(ctx, "Loaded %s" % example["id"], completed, total)
        payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": dataset,
            "split": "smoke",
            "examples": examples,
        }
        metadata = {
            "dataset": dataset,
            "split": "smoke",
            "sample_ids": sample_ids,
            "texts_preview": [{"id": example["id"], "text": example["text"]} for example in examples],
            "repeat_count": repeat_count,
            "base_text_count": len(base_examples),
            "text_count": len(examples),
            "source": "built_in_text_dataset",
        }
        path = ctx.output_path("texts", ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"texts": artifact("text.batch.json", path, metadata)},
            metadata=metadata,
            metrics={"text.dataset.sample_count": len(examples)},
        )


def _parse_sample_ids(value: str) -> List[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise RuntimeError("sample_ids must name at least one text sample")
    valid = {example["id"] for example in TEXT_SMOKE_EXAMPLES}
    for sample_id in ids:
        if sample_id not in valid:
            raise RuntimeError("Unknown text sample id: %s" % sample_id)
    return ids


def _report_text_data_progress(ctx: OperationContext, message: str, completed: int, total: int) -> None:
    total = max(int(total), 1)
    completed = max(0, min(int(completed), total))
    ctx.report_progress(
        message,
        phase="data",
        status="running",
        completed=completed,
        total=total,
        percent=float(completed) / float(total) * 100.0,
        unit="texts",
        op=ctx.step_id,
    )
