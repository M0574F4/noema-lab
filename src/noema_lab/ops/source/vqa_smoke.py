from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


VQA_SMOKE_EXAMPLES: List[JsonDict] = [
    {
        "id": "roof_antenna",
        "question": "What is mounted on the roof?",
        "answer": "antenna",
        "detections": [{"label": "antenna", "bbox": [9, 4, 14, 12], "score": 1.0}],
    },
    {
        "id": "roadside_vehicle",
        "question": "What is near the roadside sensor?",
        "answer": "vehicle",
        "detections": [{"label": "vehicle", "bbox": [2, 9, 9, 14], "score": 1.0}],
    },
]


class VqaSmokeOperation(Operation):
    id = "source.vqa_smoke"
    name = "COCO/VQA-style smoke source"
    output_kinds = {
        "images": "image.batch.numpy",
        "questions": "vqa.questions.json",
        "answers": "vqa.answers.json",
        "detections": "vision.detections.json",
    }
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "coco_vqa_smoke",
                "enum": ["coco_vqa_smoke"],
            },
            "sample_ids": {"type": "string", "default": ""},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset") or "coco_vqa_smoke")
        if dataset != "coco_vqa_smoke":
            raise RuntimeError("Unsupported VQA smoke dataset: %s" % dataset)
        sample_ids = _parse_sample_ids(ctx.params.get("sample_ids"))
        examples = [dict(item) for item in VQA_SMOKE_EXAMPLES if not sample_ids or item["id"] in sample_ids]
        if not examples:
            raise RuntimeError("No VQA smoke examples selected")

        images = np.stack([_image_for_example(item["id"]) for item in examples], axis=0).astype(np.uint8, copy=False)
        ids = [str(item["id"]) for item in examples]
        metadata = {
            "dataset": dataset,
            "split": "smoke",
            "sample_ids": ids,
            "image_ids": ids,
            "example_count": len(examples),
            "image_shape": [int(value) for value in images.shape[1:]],
            "questions_preview": [
                {
                    "id": item["id"],
                    "image_id": item["id"],
                    "question": item["question"],
                }
                for item in examples
            ],
            "answers_preview": [{"id": item["id"], "answer": item["answer"]} for item in examples],
        }

        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))

        questions_payload = {
            "schema_version": 1,
            "kind": "vqa.questions",
            "dataset": dataset,
            "examples": [
                {
                    "id": item["id"],
                    "image_id": item["id"],
                    "question": item["question"],
                }
                for item in examples
            ],
        }
        answers_payload = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "dataset": dataset,
            "examples": [{"id": item["id"], "answer": item["answer"]} for item in examples],
        }
        detections_payload = {
            "schema_version": 1,
            "kind": "vision.detections",
            "dataset": dataset,
            "examples": [{"id": item["id"], "detections": item["detections"]} for item in examples],
        }
        questions_path = _write_json(ctx, "questions", questions_payload)
        answers_path = _write_json(ctx, "answers", answers_payload)
        detections_path = _write_json(ctx, "detections", detections_payload)
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "questions": artifact("vqa.questions.json", questions_path, metadata),
                "answers": artifact("vqa.answers.json", answers_path, metadata),
                "detections": artifact("vision.detections.json", detections_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples)},
            metadata=metadata,
        )


def _image_for_example(example_id: str) -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :] = np.array([28, 32, 40], dtype=np.uint8)
    image[10:16, 0:16] = np.array([72, 75, 78], dtype=np.uint8)
    image[3:10, 4:14] = np.array([126, 94, 58], dtype=np.uint8)
    if example_id == "roof_antenna":
        image[4:12, 10:13] = np.array([230, 230, 236], dtype=np.uint8)
        image[3:5, 8:15] = np.array([190, 210, 245], dtype=np.uint8)
    elif example_id == "roadside_vehicle":
        image[9:14, 2:9] = np.array([32, 126, 210], dtype=np.uint8)
        image[13:15, 3:5] = np.array([20, 20, 22], dtype=np.uint8)
        image[13:15, 7:9] = np.array([20, 20, 22], dtype=np.uint8)
        image[4:14, 12:14] = np.array([232, 190, 42], dtype=np.uint8)
    return image


def _write_json(ctx: OperationContext, name: str, payload: JsonDict):
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_sample_ids(value: Any) -> List[str]:
    if value is None:
        return []
    ids = [item.strip() for item in str(value).split(",") if item.strip()]
    valid = {item["id"] for item in VQA_SMOKE_EXAMPLES}
    unknown = [item for item in ids if item not in valid]
    if unknown:
        raise RuntimeError("Unknown VQA smoke sample id: %s" % ", ".join(unknown))
    return ids
