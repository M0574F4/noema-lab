from __future__ import annotations

import hashlib
import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema

JsonDict = Dict[str, Any]

HF_REPO = "jxie/flickr8k"
HF_REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
HF_BASE_URL = "https://huggingface.co/datasets/%s/resolve/%s" % (HF_REPO, HF_REVISION)
SPLIT_FILES = {
    "validation": ["data/validation-00000-of-00001-7025a2b596f14b7b.parquet"],
    "test": ["data/test-00000-of-00001-42a2661d12c73e48.parquet"],
}
FILE_SHA256 = {
    "data/validation-00000-of-00001-7025a2b596f14b7b.parquet": (
        "2470e49d96fb37e19c8a1d29056623c84ed41c3760a12747e120695aa0c44ad4"
    ),
    "data/test-00000-of-00001-42a2661d12c73e48.parquet": (
        "f3bc4c3548ec1d9bb3c445c871cfcd3aca6003e3d4786e404fc9356b5215410d"
    ),
}
FILE_SIZE_BYTES = {
    "data/validation-00000-of-00001-7025a2b596f14b7b.parquet": 137_856_237,
    "data/test-00000-of-00001-42a2661d12c73e48.parquet": 136_773_352,
}
DOWNLOAD_TIMEOUT_S = 180


class Flickr8kRetrievalOperation(Operation):
    id = "source.flickr8k_retrieval"
    name = "Flickr8k image-text retrieval source"
    output_kinds = {
        "images": "image.batch.numpy",
        "texts": "text.batch.json",
        "targets": "retrieval.targets.json",
    }
    params_schema = object_schema(
        {
            "dataset_dir": {"type": "string", "default": ".noema/datasets/flickr8k"},
            "split": {"type": "string", "default": "validation", "enum": sorted(SPLIT_FILES)},
            "limit": {"type": "integer", "default": 16, "minimum": 1},
            "image_size": {"type": "integer", "default": 224, "minimum": 32},
            "caption_index": {"type": "integer", "default": 0, "minimum": 0, "maximum": 4},
            "download": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _flickr8k_availability()
        payload["dataset_info"] = {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "splits": {"validation": 1000, "test": 1000},
            "download_size": "about 138 MB per validation/test split",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _ensure_parquet_reader()
        dataset_dir = Path(str(ctx.params.get("dataset_dir") or ".noema/datasets/flickr8k"))
        split = str(ctx.params.get("split") or "validation")
        if split not in SPLIT_FILES:
            raise OperationError("Unsupported Flickr8k retrieval split: %s" % split)
        limit = max(1, int(ctx.params.get("limit") or 16))
        image_size = max(32, int(ctx.params.get("image_size") or 224))
        caption_index = min(4, max(0, int(ctx.params.get("caption_index") or 0)))
        should_download = bool(ctx.params.get("download", True))
        paths = _ensure_split_files(dataset_dir, split, should_download)
        examples = _read_examples(paths, split, limit, image_size, caption_index)
        if not examples:
            raise OperationError("Flickr8k split %s has no examples" % split)

        images = np.stack([item["image_array"] for item in examples], axis=0).astype(np.uint8, copy=False)
        text_examples = [{"id": item["id"], "text": item["text"]} for item in examples]
        target_examples = [{"id": item["id"], "target_id": item["id"]} for item in examples]
        metadata = {
            "dataset": "flickr8k",
            "source_repo": HF_REPO,
            "source_revision": HF_REVISION,
            "source_sha256": [FILE_SHA256[str(path.relative_to(dataset_dir))] for path in paths],
            "split": split,
            "example_count": len(examples),
            "sample_ids": [item["id"] for item in examples],
            "image_shape": [int(value) for value in images.shape[1:]],
            "caption_index": caption_index,
            "texts_preview": text_examples[:10],
            "targets_preview": target_examples[:10],
            "cached_parquet": [str(path) for path in paths],
        }

        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        text_payload = {
            "schema_version": 1,
            "kind": "text.batch",
            "dataset": "flickr8k",
            "split": split,
            "examples": text_examples,
        }
        target_payload = {
            "schema_version": 1,
            "kind": "retrieval.targets",
            "dataset": "flickr8k",
            "split": split,
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


def _ensure_split_files(dataset_dir: Path, split: str, should_download: bool) -> List[Path]:
    paths = []
    for relative in SPLIT_FILES[split]:
        target = dataset_dir / relative
        if target.is_symlink():
            raise OperationError("Flickr8k cached split must not be a symbolic link: %s" % target)
        if target.is_file():
            _require_sha256(target, FILE_SHA256[relative], "Flickr8k cached split")
        else:
            if not should_download:
                raise OperationError("Flickr8k split file is not cached: %s" % target)
            target.parent.mkdir(parents=True, exist_ok=True)
            _download_file(
                "%s/%s" % (HF_BASE_URL, relative),
                target,
                FILE_SHA256[relative],
                FILE_SIZE_BYTES[relative],
            )
        paths.append(target)
    return paths


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
        raise OperationError("Could not download Flickr8k file %s: %s" % (url, exc)) from exc


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


def _read_examples(paths: List[Path], split: str, limit: int, image_size: int, caption_index: int) -> List[JsonDict]:
    import pandas as pd

    rows: List[JsonDict] = []
    remaining = limit
    for path in paths:
        if remaining <= 0:
            break
        frame = pd.read_parquet(path)
        for row in frame.head(remaining).to_dict(orient="records"):
            rows.append(row)
        remaining = limit - len(rows)
    examples = []
    caption_key = "caption_%d" % caption_index
    for index, row in enumerate(rows[:limit]):
        caption = row.get(caption_key)
        if not caption:
            caption = next((row.get("caption_%d" % item) for item in range(5) if row.get("caption_%d" % item)), "")
        if not caption:
            raise OperationError("Flickr8k row %d has no caption" % index)
        examples.append(
            {
                "id": "%s_%06d" % (split, index),
                "text": str(caption),
                "image_array": _decode_image(row.get("image"), image_size),
            }
        )
    return examples


def _decode_image(value: Any, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to load Flickr8k images") from exc
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        import io

        with Image.open(io.BytesIO(value)) as image:
            return np.asarray(image.convert("RGB").resize((image_size, image_size)), dtype=np.uint8)
    if hasattr(value, "convert"):
        return np.asarray(value.convert("RGB").resize((image_size, image_size)), dtype=np.uint8)
    raise OperationError("Flickr8k image column has unsupported type: %s" % type(value).__name__)


def _write_json(ctx: OperationContext, name: str, payload: JsonDict) -> Path:
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ensure_parquet_reader() -> None:
    if importlib.util.find_spec("pandas") is None or (
        importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None
    ):
        raise OperationError(
            'Install with `python -m pip install "noema-lab[retrieval-data]"` in '
            "an installed environment, or `uv sync --extra retrieval-data` in a "
            "source checkout, to use Flickr8k retrieval data"
        )


def _flickr8k_availability() -> JsonDict:
    missing = []
    if importlib.util.find_spec("pandas") is None:
        missing.append("pandas")
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        missing.append("pyarrow")
    if missing:
        return {
            "available": False,
            "extra": "retrieval-data",
            "missing": missing,
            "reason": (
                'Install with `python -m pip install "noema-lab[retrieval-data]"` '
                "in an installed environment, or `uv sync --extra retrieval-data` "
                "in a source checkout, to use Flickr8k retrieval data"
            ),
        }
    return {"available": True, "extra": "retrieval-data", "missing": []}
