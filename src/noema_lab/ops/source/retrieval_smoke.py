from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


RETRIEVAL_SMOKE_EXAMPLES: List[JsonDict] = [
    {"id": "red_square", "text": "a red square on a dark background", "color": [224, 42, 48], "shape": "square"},
    {"id": "green_circle", "text": "a green circle on a dark background", "color": [44, 178, 82], "shape": "circle"},
    {"id": "blue_triangle", "text": "a blue triangle on a dark background", "color": [56, 118, 224], "shape": "triangle"},
    {"id": "yellow_bar", "text": "a yellow horizontal bar on a dark background", "color": [232, 196, 48], "shape": "bar"},
]


class RetrievalSmokeOperation(Operation):
    id = "source.retrieval_smoke"
    name = "Image-text retrieval smoke source"
    output_kinds = {
        "images": "image.batch.numpy",
        "texts": "text.batch.json",
        "targets": "retrieval.targets.json",
    }
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "retrieval_smoke",
                "enum": ["retrieval_smoke"],
            },
            "sample_ids": {"type": "string", "default": ""},
            "image_size": {"type": "integer", "default": 224, "minimum": 32},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset") or "retrieval_smoke")
        if dataset != "retrieval_smoke":
            raise RuntimeError("Unsupported retrieval smoke dataset: %s" % dataset)
        sample_ids = _parse_sample_ids(ctx.params.get("sample_ids"))
        examples = [dict(item) for item in RETRIEVAL_SMOKE_EXAMPLES if not sample_ids or item["id"] in sample_ids]
        if not examples:
            raise RuntimeError("No retrieval smoke examples selected")
        image_size = max(32, int(ctx.params.get("image_size") or 224))
        images = np.stack([_image_for_example(item, image_size) for item in examples], axis=0).astype(np.uint8, copy=False)
        ids = [str(item["id"]) for item in examples]
        text_examples = [{"id": item["id"], "text": item["text"]} for item in examples]
        target_examples = [{"id": item["id"], "target_id": item["id"]} for item in examples]
        metadata = {
            "dataset": dataset,
            "split": "smoke",
            "sample_ids": ids,
            "example_count": len(examples),
            "image_shape": [int(value) for value in images.shape[1:]],
            "texts_preview": text_examples,
            "targets_preview": target_examples,
        }

        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        text_payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": dataset,
            "examples": text_examples,
        }
        target_payload = {
            "schema_version": 1,
            "kind": "retrieval.targets",
            "dataset": dataset,
            "examples": target_examples,
        }
        text_path = _write_json(ctx, "texts", text_payload)
        target_path = _write_json(ctx, "targets", target_payload)
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "texts": artifact("text.batch.json", text_path, metadata),
                "targets": artifact("retrieval.targets.json", target_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples)},
            metadata=metadata,
        )


def _image_for_example(example: JsonDict, image_size: int) -> np.ndarray:
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[:, :] = np.array([24, 28, 34], dtype=np.uint8)
    color = np.asarray(example["color"], dtype=np.uint8)
    y, x = np.ogrid[:image_size, :image_size]
    center = image_size / 2.0
    radius = image_size * 0.26
    shape = str(example["shape"])
    if shape == "circle":
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    elif shape == "triangle":
        top = image_size * 0.23
        left = image_size * 0.25
        right = image_size * 0.75
        bottom = image_size * 0.76
        rel_y = np.clip((y - top) / max(bottom - top, 1.0), 0.0, 1.0)
        x_left = center - (center - left) * rel_y
        x_right = center + (right - center) * rel_y
        mask = (y >= top) & (y <= bottom) & (x >= x_left) & (x <= x_right)
    elif shape == "bar":
        mask = (y >= image_size * 0.42) & (y <= image_size * 0.58) & (x >= image_size * 0.18) & (x <= image_size * 0.82)
    else:
        mask = (y >= image_size * 0.25) & (y <= image_size * 0.75) & (x >= image_size * 0.25) & (x <= image_size * 0.75)
    image[mask] = color
    return image


def _write_json(ctx: OperationContext, name: str, payload: JsonDict):
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_sample_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]
