from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.structured_input import decode_strict_yaml_or_json

JsonDict = Dict[str, Any]


class VqaManifestOperation(Operation):
    id = "source.vqa_manifest"
    name = "Local VQA manifest source"
    output_kinds = {
        "images": "image.batch.numpy",
        "questions": "vqa.questions.json",
        "answers": "vqa.answers.json",
    }
    params_schema = object_schema(
        {
            "manifest_path": {
                "type": "string",
                "default": "",
                "description": "JSON/JSONL file with id, image, question, and answer fields.",
            },
            "image_root": {
                "type": "string",
                "default": "",
                "description": "Optional root directory for relative image paths.",
            },
            "limit": {"type": "integer", "default": 8, "minimum": 1},
            "image_size": {"type": "integer", "default": 224, "minimum": 16},
        },
        required=["manifest_path"],
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        manifest_raw = str(ctx.params.get("manifest_path") or "").strip()
        if not manifest_raw:
            raise OperationError("VQA manifest_path is required. Choose built-in smoke examples or provide a JSON/JSONL manifest path.")
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_file():
            raise OperationError("VQA manifest_path does not exist: %s" % manifest_path)
        image_root = Path(str(ctx.params.get("image_root") or manifest_path.parent))
        limit = max(1, int(ctx.params.get("limit") or 8))
        image_size = max(16, int(ctx.params.get("image_size") or 224))
        examples = _load_manifest_examples(manifest_path)[:limit]
        if not examples:
            raise OperationError("VQA manifest has no examples")
        images = np.stack([_load_image(image_root, item, image_size) for item in examples], axis=0).astype(np.uint8, copy=False)
        questions = {
            "schema_version": 1,
            "kind": "vqa.questions",
            "dataset": manifest_path.stem,
            "examples": [
                {
                    "id": item["id"],
                    "image_id": item["image_id"],
                    "question": item["question"],
                }
                for item in examples
            ],
        }
        answers = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "dataset": manifest_path.stem,
            "examples": [{"id": item["id"], "answer": item["answer"]} for item in examples],
        }
        metadata = {
            "dataset": manifest_path.stem,
            "manifest_path": str(manifest_path),
            "image_root": str(image_root),
            "example_count": len(examples),
            "sample_ids": [item["id"] for item in examples],
            "image_ids": [item["image_id"] for item in examples],
            "questions_preview": questions["examples"],
            "answers_preview": answers["examples"],
            "image_shape": [int(value) for value in images.shape[1:]],
        }
        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        question_path = _write_json(ctx, "questions", questions)
        answer_path = _write_json(ctx, "answers", answers)
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "questions": artifact("vqa.questions.json", question_path, metadata),
                "answers": artifact("vqa.answers.json", answer_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples)},
            metadata=metadata,
        )


def _load_manifest_examples(path: Path) -> List[JsonDict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        if path.suffix.lower() == ".jsonl":
            rows = [
                decode_strict_yaml_or_json(line, input_format="json")
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            payload = decode_strict_yaml_or_json(text, input_format="json")
            rows = payload.get("examples") if isinstance(payload, dict) else payload
    except ValueError as exc:
        raise OperationError("VQA manifest contains invalid JSON: %s" % exc) from exc
    if not isinstance(rows, list):
        raise OperationError("VQA manifest must be a JSON list or an object with examples")
    examples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OperationError("VQA manifest row %d must be an object" % index)
        image = row.get("image") or row.get("image_path") or row.get("file_name")
        question = row.get("question")
        answer = row.get("answer")
        if not image or not question or answer is None:
            raise OperationError("VQA manifest row %d requires image, question, and answer" % index)
        examples.append(
            {
                "id": str(row.get("id") or row.get("question_id") or index),
                "image_id": str(row.get("image_id") or image),
                "image": str(image),
                "question": str(question),
                "answer": str(answer),
            }
        )
    return examples


def _load_image(root: Path, row: JsonDict, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to load VQA manifest images") from exc
    path = Path(row["image"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise OperationError("VQA manifest image does not exist: %s" % path)
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(rgb, dtype=np.uint8)


def _write_json(ctx: OperationContext, name: str, payload: JsonDict):
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
