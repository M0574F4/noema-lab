from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema

JsonDict = Dict[str, Any]

HF_REPO = "soumyasj/vqa-dataset-small"
HF_REVISION = "5ae2386321b830c3266f7a7bc212045841b2c69d"
HF_BASE_URL = "https://huggingface.co/datasets/%s/resolve/%s" % (HF_REPO, HF_REVISION)
SPLIT_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
SPLIT_SHA256 = {
    "train": "a0b851829c19e3d8997a2e47c6082c0731b2cf6256668fb39d7419a149a37911",
    "validation": "b9c6eda235f578b37e95eb7fd2bf5a3bf893b47662b85a47374db8503aae2d04",
    "test": "30435521495d2e6ca6ac04ea2c6f581bb0a9e04a8a490f60cfd8c4838baae4e6",
}
SPLIT_SIZE_BYTES = {
    "train": 2_831_236,
    "validation": 963_193,
    "test": 879_005,
}
DOWNLOAD_TIMEOUT_S = 120


class VqaSmallHfOperation(Operation):
    id = "source.vqa_small_hf"
    name = "VQA small sample source"
    output_kinds = {
        "images": "image.batch.numpy",
        "questions": "vqa.questions.json",
        "answers": "vqa.answers.json",
    }
    params_schema = object_schema(
        {
            "dataset_dir": {"type": "string", "default": ".noema/datasets/vqa_small"},
            "split": {"type": "string", "default": "validation", "enum": sorted(SPLIT_FILES)},
            "limit": {"type": "integer", "default": 8, "minimum": 1},
            "image_size": {"type": "integer", "default": 224, "minimum": 16},
            "download": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _vqa_small_availability()
        payload["dataset_info"] = {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "download_size": "about 4.7 MB",
            "splits": {"train": 60, "validation": 20, "test": 20},
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _ensure_parquet_reader()
        dataset_dir = Path(str(ctx.params.get("dataset_dir") or ".noema/datasets/vqa_small"))
        split = str(ctx.params.get("split") or "validation")
        if split not in SPLIT_FILES:
            raise OperationError("Unsupported VQA small split: %s" % split)
        limit = max(1, int(ctx.params.get("limit") or 8))
        image_size = max(16, int(ctx.params.get("image_size") or 224))
        should_download = bool(ctx.params.get("download", True))
        parquet_path = _ensure_split_file(dataset_dir, split, should_download)
        examples = _read_examples(parquet_path, split, limit, image_size)
        if not examples:
            raise OperationError("VQA small dataset split %s has no examples" % split)

        images = np.stack([item["image_array"] for item in examples], axis=0).astype(np.uint8, copy=False)
        question_examples = [
            {
                "id": item["id"],
                "image_id": item["image_id"],
                "question": item["question"],
            }
            for item in examples
        ]
        answer_examples = [{"id": item["id"], "answer": item["answer"]} for item in examples]
        metadata = {
            "dataset": "vqa_small",
            "source_repo": HF_REPO,
            "source_revision": HF_REVISION,
            "source_sha256": SPLIT_SHA256[split],
            "split": split,
            "example_count": len(examples),
            "sample_ids": [item["id"] for item in examples],
            "image_ids": [item["image_id"] for item in examples],
            "questions_preview": question_examples,
            "answers_preview": answer_examples,
            "image_shape": [int(value) for value in images.shape[1:]],
            "cached_parquet": str(parquet_path),
        }

        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        questions_payload = {
            "schema_version": 1,
            "kind": "vqa.questions",
            "dataset": "vqa_small",
            "split": split,
            "examples": question_examples,
        }
        answers_payload = {
            "schema_version": 1,
            "kind": "vqa.answers",
            "dataset": "vqa_small",
            "split": split,
            "examples": answer_examples,
        }
        question_path = _write_json(ctx, "questions", questions_payload)
        answer_path = _write_json(ctx, "answers", answers_payload)
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "questions": artifact("vqa.questions.json", question_path, metadata),
                "answers": artifact("vqa.answers.json", answer_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples)},
            metadata=metadata,
        )


def _ensure_split_file(dataset_dir: Path, split: str, should_download: bool) -> Path:
    relative = SPLIT_FILES[split]
    target = dataset_dir / relative
    if target.is_symlink():
        raise OperationError("VQA small cached split must not be a symbolic link: %s" % target)
    if target.is_file():
        _require_sha256(target, SPLIT_SHA256[split], "VQA small cached split")
        return target
    if not should_download:
        raise OperationError("VQA small split is not cached: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _download_file(
        "%s/%s" % (HF_BASE_URL, relative),
        target,
        SPLIT_SHA256[split],
        SPLIT_SIZE_BYTES[split],
    )
    return target


def _download_file(
    url: str, target: Path, expected_sha256: str, expected_size: int
) -> None:
    try:
        download_verified_https(
            url,
            target,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            max_bytes=expected_size,
            timeout_s=DOWNLOAD_TIMEOUT_S,
            opener=urllib.request.urlopen,
        )
    except Exception as exc:
        raise OperationError("Could not download VQA small dataset file %s: %s" % (url, exc)) from exc


def _require_sha256(path: Path, expected_sha256: str, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OperationError("Could not hash %s %s: %s" % (label, path, exc)) from exc
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise OperationError(
            "%s SHA-256 mismatch for %s: expected %s, got %s"
            % (label, path, expected_sha256, actual)
        )
    return actual


def _read_examples(path: Path, split: str, limit: int, image_size: int) -> List[JsonDict]:
    import pandas as pd

    frame = pd.read_parquet(path)
    rows = frame.head(limit).to_dict(orient="records")
    examples: List[JsonDict] = []
    for index, row in enumerate(rows):
        image_id = row.get("image_id", index)
        question_id = row.get("question_id", index)
        image_bytes = row.get("image")
        question = row.get("question")
        answer = row.get("gt_answer")
        if image_bytes is None or not question or answer is None:
            raise OperationError("VQA small row %d is missing image, question, or gt_answer" % index)
        examples.append(
            {
                "id": "%s_%s_%s" % (split, image_id, question_id),
                "image_id": str(image_id),
                "question": str(question),
                "answer": str(answer),
                "image_array": _decode_image_bytes(image_bytes, image_size),
            }
        )
    return examples


def _decode_image_bytes(value: Any, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to load VQA small images") from exc
    if isinstance(value, memoryview):
        data = value.tobytes()
    elif isinstance(value, bytearray):
        data = bytes(value)
    elif isinstance(value, bytes):
        data = value
    else:
        raise OperationError("VQA small image column has unsupported type: %s" % type(value).__name__)
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(rgb, dtype=np.uint8)


def _write_json(ctx: OperationContext, name: str, payload: JsonDict) -> Path:
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ensure_parquet_reader() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[vqa-data]"` in an '
            "installed environment, or `uv sync --extra vqa-data` in a source "
            "checkout, to use the downloadable VQA small dataset"
        )


def _vqa_small_availability() -> JsonDict:
    missing = []
    if importlib.util.find_spec("pandas") is None:
        missing.append("pandas")
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        missing.append("pyarrow")
    if missing:
        return {
            "available": False,
            "extra": "vqa-data",
            "missing": missing,
            "reason": (
                'Install with `python -m pip install "noema-lab[vqa-data]"` in an '
                "installed environment, or `uv sync --extra vqa-data` in a source "
                "checkout, to use the downloadable VQA small dataset"
            ),
        }
    return {"available": True, "extra": "vqa-data", "missing": []}
