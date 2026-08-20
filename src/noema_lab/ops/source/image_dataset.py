from __future__ import annotations

import json
import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.ops.source.kodak import (
    KODAK_DATASET_PROVENANCE,
    KODAK_FILENAMES,
    download_kodak_dataset,
    kodak_file_record,
)

JsonDict = Dict[str, Any]
DEFAULT_KODAK_IMAGE_IDS = ",".join(Path(name).stem for name in KODAK_FILENAMES)


class ImageDatasetOperation(Operation):
    id = "source.image_dataset"
    name = "Real image dataset batch"
    output_kinds = {"images": "image.batch.numpy"}
    output_metadata_guarantees = {
        "images": ["original_shapes", "original_shape", "shape"]
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    params_schema = object_schema(
        {
            "dataset": {
                "type": "string",
                "default": "kodak",
                "description": "Dataset identifier. Kodak works without a manifest; every other dataset requires a hash-pinned manifest_path.",
            },
            "dataset_dir": {"type": "string", "default": ".noema/datasets/kodak"},
            "manifest_path": {
                "type": "string",
                "default": "",
                "description": "JSON/YAML noema.image_dataset_manifest. Sample paths are confined beneath its declared root.",
            },
            "manifest_sha256": {
                "type": "string",
                "default": "",
                "description": "Expected lowercase SHA-256 of manifest_path; required whenever a manifest is used.",
            },
            "split": {
                "type": "string",
                "default": "",
                "description": "Manifest split used when image_ids is empty.",
            },
            "image_ids": {
                "type": "string",
                "default": DEFAULT_KODAK_IMAGE_IDS,
            },
            "resize_shorter_side": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Optional aspect-preserving resize applied before the center crop; 0 keeps native resolution.",
            },
            "crop_size": {"type": "integer", "default": 0, "minimum": 0},
            "repeat_count": {"type": "integer", "default": 1, "minimum": 1},
            "training_validation_count": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": (
                    "Optional external-training export control. When positive, the "
                    "last N explicitly selected or manifest-split images form the "
                    "validation partition; ordinary benchmark execution is unchanged."
                ),
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset = str(ctx.params.get("dataset", "kodak"))
        dataset_dir = Path(str(ctx.params.get("dataset_dir", ".noema/datasets/kodak")))
        manifest_path = str(ctx.params.get("manifest_path") or "").strip()
        expected_manifest_sha256 = str(
            ctx.params.get("manifest_sha256") or ""
        ).strip().lower()
        split = str(ctx.params.get("split") or "").strip()
        raw_image_ids = str(ctx.params.get("image_ids", DEFAULT_KODAK_IMAGE_IDS))
        resize_shorter_side = int(ctx.params.get("resize_shorter_side", 0))
        crop_size = int(ctx.params.get("crop_size", 0))
        repeat_count = max(1, int(ctx.params.get("repeat_count", 1)))
        manifest: Optional[JsonDict] = None
        manifest_sha256: Optional[str] = None
        if manifest_path:
            manifest_candidate = Path(manifest_path).expanduser()
            if manifest_candidate.is_symlink() or not manifest_candidate.is_file():
                raise RuntimeError(
                    "Image dataset manifest is missing or unsafe: %s"
                    % manifest_candidate
                )
            resolved_manifest_path = manifest_candidate.resolve()
            expected_manifest_sha256 = _require_sha256(
                expected_manifest_sha256, "manifest_sha256"
            )
            actual_manifest_sha256 = file_sha256(resolved_manifest_path)
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise RuntimeError(
                    "Image dataset manifest SHA-256 mismatch: %s"
                    % resolved_manifest_path
                )
            manifest = _load_image_manifest(resolved_manifest_path, expected_dataset=dataset)
            if file_sha256(resolved_manifest_path) != expected_manifest_sha256:
                raise RuntimeError("Image dataset manifest changed while loading")
            manifest_sha256 = actual_manifest_sha256
            # The schema default is Kodak's full list for backwards compatibility.
            # It is not an implicit selection for a non-Kodak manifest.
            if raw_image_ids == DEFAULT_KODAK_IMAGE_IDS and dataset != "kodak":
                raw_image_ids = ""
            image_ids = _manifest_selection(manifest, raw_image_ids, split)
            records, paths = _manifest_records_and_paths(
                manifest,
                resolved_manifest_path,
                image_ids,
            )
        elif dataset == "kodak":
            image_ids = _parse_kodak_image_ids(raw_image_ids)
            records = []
            paths = [_kodak_path(dataset_dir, image_id) for image_id in image_ids]
        else:
            raise RuntimeError(
                "Dataset %s requires a hash-pinned manifest_path" % dataset
            )
        total_work = max(len(image_ids) + 3, 1)
        _report_data_progress(ctx, "Checking %s dataset" % dataset, 0, total_work)
        if manifest is None:
            download_kodak_dataset(dataset_dir)
        _report_data_progress(ctx, "%s dataset ready" % dataset, 1, total_work)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Real dataset files are missing. Missing: %s"
                % ", ".join(missing[:3])
            )
        images = []
        source_items: List[JsonDict] = []
        for index, path in enumerate(paths):
            if manifest is None:
                identity_transform = {"name": "identity"}
                record = {
                    "sample_id": image_ids[index],
                    "source_id": "kodak:%s" % image_ids[index],
                    "group_id": "kodak:%s" % image_ids[index],
                    "source_sha256": file_sha256(path),
                    "sha256": file_sha256(path),
                    "ancestry_ids": ["kodak:%s" % image_ids[index]],
                    "transform": identity_transform,
                    "transform_fingerprint_sha256": canonical_json_sha256(
                        identity_transform
                    ),
                }
            else:
                record = records[index]
                actual_sha256 = file_sha256(path)
                if actual_sha256 != record["sha256"]:
                    raise RuntimeError(
                        "Manifest SHA-256 mismatch for sample %s" % image_ids[index]
                    )
            decoded = _read_image_rgb(path)
            if manifest is not None and file_sha256(path) != record["sha256"]:
                raise RuntimeError(
                    "Manifest sample changed while decoding: %s" % image_ids[index]
                )
            image = _center_crop(
                _resize_shorter_side(decoded, resize_shorter_side), crop_size
            )
            images.append(image)
            transform = {
                "manifest_transform": record.get("transform") or {"name": "identity"},
                "resize_shorter_side": resize_shorter_side,
                "crop_size": crop_size,
                "decoder": "Pillow RGB canonicalization",
            }
            source_items.append(
                {
                    "sample_id": image_ids[index],
                    "source_id": str(record["source_id"]),
                    "group_id": str(record["group_id"]),
                    "ancestry_ids": list(record["ancestry_ids"]),
                    "source_sha256": str(record["source_sha256"]),
                    "file_sha256": str(record["sha256"]),
                    "manifest_transform_fingerprint_sha256": str(
                        record["transform_fingerprint_sha256"]
                    ),
                    "applied_transform": transform,
                    "applied_transform_fingerprint_sha256": canonical_json_sha256(
                        transform
                    ),
                    "post_transform_sha256": _image_identity_sha256(image),
                    "post_transform_shape": list(image.shape),
                    "post_transform_dtype": str(image.dtype),
                }
            )
            _report_data_progress(
                ctx,
                "Loaded %s" % path.name,
                2 + index,
                total_work,
            )
        base_shapes = [[1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])] for image in images]
        base_batch = _stack_images_with_padding(images)
        batch = np.concatenate([base_batch] * repeat_count, axis=0) if repeat_count > 1 else base_batch
        original_shapes = base_shapes * repeat_count
        materialized_items: List[JsonDict] = []
        for repeat_index in range(repeat_count):
            for base_index, source_item in enumerate(source_items):
                item = dict(source_item)
                item.update(
                    {
                        "order_index": len(materialized_items),
                        "base_order_index": base_index,
                        "repeat_index": repeat_index,
                        "item_id": "%s#repeat-%d"
                        % (source_item["sample_id"], repeat_index),
                    }
                )
                materialized_items.append(item)
        ordered_post_transform = [
            {
                "item_id": item["item_id"],
                "sample_id": item["sample_id"],
                "post_transform_sha256": item["post_transform_sha256"],
                "shape": item["post_transform_shape"],
                "dtype": item["post_transform_dtype"],
            }
            for item in materialized_items
        ]
        source_operation_contract = {
            "dataset": dataset,
            "manifest_sha256": manifest_sha256,
            "split": split or None,
            "image_ids": list(image_ids),
            "resize_shorter_side": resize_shorter_side,
            "crop_size": crop_size,
            "repeat_count": repeat_count,
        }
        if manifest is None:
            publication_ready = bool(KODAK_DATASET_PROVENANCE["publication_ready"])
            publication_blocker = KODAK_DATASET_PROVENANCE["publication_blocker"]
            dataset_license = dict(KODAK_DATASET_PROVENANCE["license"])
            dataset_manifest_version = KODAK_DATASET_PROVENANCE["manifest_version"]
            dataset_source_page = KODAK_DATASET_PROVENANCE["source_page"]
            dataset_file_records = [
                kodak_file_record(path.name, path) for path in paths
            ]
        else:
            publication_ready = bool(manifest.get("publication_ready", False))
            publication_blocker = str(manifest.get("publication_blocker") or "")
            dataset_license = dict(manifest.get("license") or {})
            dataset_manifest_version = manifest.get("version")
            dataset_source_page = manifest.get("source_page")
            dataset_file_records = [dict(record) for record in records]
        metadata = {
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "split": split or None,
            "image_ids": image_ids,
            "files": [str(path) for path in paths],
            "dataset_manifest_path": manifest_path or None,
            "dataset_manifest_sha256": manifest_sha256,
            "dataset_manifest_expected_sha256": expected_manifest_sha256 or None,
            "dataset_manifest_version": dataset_manifest_version,
            "dataset_source_page": dataset_source_page,
            "dataset_file_records": dataset_file_records,
            "dataset_license": dataset_license,
            "dataset_publication_ready": publication_ready,
            "dataset_publication_blocker": publication_blocker,
            "resize_shorter_side": resize_shorter_side,
            "crop_size": crop_size,
            "repeat_count": repeat_count,
            "base_image_count": int(base_batch.shape[0]),
            "shape": list(batch.shape),
            "storage_shape": list(batch.shape),
            "original_shape": list(batch.shape),
            "original_shapes": original_shapes,
            "padded_to_common_shape": any(tuple(shape[1:]) != tuple(batch.shape[1:]) for shape in original_shapes),
            "dtype": str(batch.dtype),
            "source": "real_dataset",
            "source_operation_contract": source_operation_contract,
            "source_operation_contract_sha256": canonical_json_sha256(
                source_operation_contract
            ),
            "source_items": materialized_items,
            "source_item_ids": [item["item_id"] for item in materialized_items],
            "source_ids": [item["source_id"] for item in materialized_items],
            "source_group_ids": [item["group_id"] for item in materialized_items],
            "ordered_post_transform_items": ordered_post_transform,
            "ordered_post_transform_sha256": canonical_json_sha256(
                ordered_post_transform
            ),
            "batch_tensor_sha256": _image_identity_sha256(batch),
            "source_item_count": len(materialized_items),
        }
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=batch, metadata_json=json.dumps(metadata))
        _report_data_progress(ctx, "Prepared image batch", total_work, total_work)
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, metadata)},
            metadata={
                "dataset": dataset,
                "base_image_count": int(base_batch.shape[0]),
                "repeat_count": repeat_count,
                "image_count": int(batch.shape[0]),
                "image_shape": list(batch.shape[1:]),
                "original_shapes": original_shapes,
            },
        )


def _report_data_progress(ctx: OperationContext, message: str, completed: int, total: int) -> None:
    total = max(int(total), 1)
    completed = max(0, min(int(completed), total))
    ctx.report_progress(
        message,
        phase="data",
        status="running",
        completed=completed,
        total=total,
        percent=float(completed) / float(total) * 100.0,
        unit="files",
        op=ctx.step_id,
    )


def _parse_kodak_image_ids(value: str) -> List[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise RuntimeError("image_ids must name at least one image")
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "Kodak image_ids must be unique; use repeat_count for intentional repetition"
        )
    valid = {Path(name).stem for name in KODAK_FILENAMES}
    for image_id in ids:
        if image_id not in valid:
            raise RuntimeError("Unknown Kodak image id: %s" % image_id)
    return ids


def _load_image_manifest(path: Path, *, expected_dataset: str) -> JsonDict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Image dataset manifest is missing or unsafe: %s" % path)
    try:
        payload = load_strict_yaml_or_json(path)
    except Exception as exc:
        raise RuntimeError("Cannot parse image dataset manifest %s: %s" % (path, exc)) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Image dataset manifest must be a mapping: %s" % path)
    manifest = dict(payload)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "noema.image_dataset_manifest":
        raise RuntimeError(
            "Image dataset manifest must use schema_version=1 and kind=noema.image_dataset_manifest"
        )
    dataset_id = str(manifest.get("id") or "").strip()
    if not dataset_id or dataset_id != expected_dataset:
        raise RuntimeError(
            "Image dataset manifest id %r does not match dataset %r"
            % (dataset_id, expected_dataset)
        )
    if manifest.get("version") in (None, ""):
        raise RuntimeError("Image dataset manifest requires version")
    if not str(manifest.get("root") or "").strip():
        raise RuntimeError("Image dataset manifest requires a root")
    if not isinstance(manifest.get("splits"), Mapping) or not manifest.get("splits"):
        raise RuntimeError("Image dataset manifest requires non-empty splits")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Image dataset manifest requires a non-empty samples list")
    normalized_samples: List[JsonDict] = []
    seen: set[str] = set()
    for index, raw in enumerate(samples):
        if not isinstance(raw, Mapping):
            raise RuntimeError("Image dataset manifest sample %d must be a mapping" % index)
        row = dict(raw)
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        relative_path = str(row.get("path") or "").strip()
        digest = _require_sha256(row.get("sha256"), "sample %s sha256" % (sample_id or index))
        source_id = str(row.get("source_id") or "").strip()
        group_id = str(row.get("group_id") or "").strip()
        source_sha256 = _require_sha256(
            row.get("source_sha256"), "sample %s source_sha256" % (sample_id or index)
        )
        ancestry = row.get("ancestry_ids")
        if not isinstance(ancestry, list) or not ancestry or not all(
            isinstance(item, str) and item.strip() for item in ancestry
        ):
            raise RuntimeError(
                "Image dataset manifest sample %s requires non-empty ancestry_ids"
                % (sample_id or index)
            )
        transform = row.get("transform") or {"name": "identity"}
        if not isinstance(transform, Mapping) or not str(transform.get("name") or "").strip():
            raise RuntimeError(
                "Image dataset manifest sample %s requires a named transform"
                % (sample_id or index)
            )
        transform_sha = row.get("transform_fingerprint_sha256")
        computed_transform_sha = canonical_json_sha256(dict(transform))
        declared_transform_sha = _require_sha256(
            transform_sha, "sample %s transform fingerprint" % (sample_id or index)
        )
        if declared_transform_sha != computed_transform_sha:
            raise RuntimeError(
                "Image dataset manifest sample %s transform fingerprint mismatch"
                % (sample_id or index)
            )
        if str(transform.get("name") or "").strip().lower() == "identity" and source_sha256 != digest:
            raise RuntimeError(
                "Image dataset manifest identity sample %s must use its file SHA-256 as source_sha256"
                % (sample_id or index)
            )
        if (
            not sample_id
            or sample_id.lower() in seen
            or not relative_path
            or not source_id
            or not group_id
        ):
            raise RuntimeError(
                "Image dataset manifest sample %d has missing or duplicate identity fields"
                % index
            )
        seen.add(sample_id.lower())
        normalized_samples.append(
            {
                **row,
                "sample_id": sample_id,
                "path": relative_path,
                "sha256": digest,
                "source_id": source_id,
                "group_id": group_id,
                "source_sha256": source_sha256,
                "ancestry_ids": [str(item).strip() for item in ancestry],
                "transform": dict(transform),
                "transform_fingerprint_sha256": computed_transform_sha,
            }
        )
    manifest["samples"] = normalized_samples
    return manifest


def _manifest_selection(manifest: Mapping[str, Any], raw_ids: str, split: str) -> List[str]:
    requested = [item.strip() for item in raw_ids.split(",") if item.strip()]
    if requested:
        if len({item.lower() for item in requested}) != len(requested):
            raise RuntimeError("Manifest image_ids must be unique; use repeat_count for repetition")
        return requested
    if not split:
        raise RuntimeError("Manifest dataset requires image_ids or split")
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, Mapping) or not isinstance(raw_splits.get(split), list):
        raise RuntimeError("Image dataset manifest has no split %s" % split)
    selected = [str(item).strip() for item in raw_splits[split] if str(item).strip()]
    if not selected:
        raise RuntimeError("Image dataset manifest split %s is empty" % split)
    if len({item.lower() for item in selected}) != len(selected):
        raise RuntimeError("Image dataset manifest split %s contains duplicate IDs" % split)
    return selected


def _manifest_records_and_paths(
    manifest: Mapping[str, Any], manifest_path: Path, image_ids: List[str]
) -> Tuple[List[JsonDict], List[Path]]:
    by_id = {
        str(row["sample_id"]).lower(): dict(row)
        for row in manifest.get("samples") or []
        if isinstance(row, Mapping)
    }
    root_value = str(manifest.get("root") or ".")
    root_relative = Path(root_value)
    if root_relative.is_absolute() or ".." in root_relative.parts:
        raise RuntimeError("Image dataset manifest root must be manifest-relative")
    root_candidate = manifest_path.parent / root_relative
    if root_candidate.is_symlink():
        raise RuntimeError("Image dataset manifest root may not be a symlink")
    root = root_candidate.resolve()
    records: List[JsonDict] = []
    paths: List[Path] = []
    for sample_id in image_ids:
        record = by_id.get(sample_id.lower())
        if record is None:
            raise RuntimeError("Unknown manifest image id: %s" % sample_id)
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Manifest sample path must be root-relative: %s" % sample_id)
        unresolved_candidate = root / relative
        if unresolved_candidate.is_symlink():
            raise RuntimeError("Manifest sample may not be a symlink: %s" % sample_id)
        candidate = unresolved_candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Manifest sample escapes its dataset root: %s" % sample_id) from exc
        records.append(record)
        paths.append(candidate)
    return records, paths


def _require_sha256(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("%s must be a lowercase SHA-256" % field)
    return digest


def _image_identity_sha256(image: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(canonical_json_sha256(
        {"shape": list(image.shape), "dtype": str(image.dtype)}
    ).encode("ascii"))
    hasher.update(np.ascontiguousarray(image).tobytes(order="C"))
    return hasher.hexdigest()


def _kodak_path(dataset_dir: Path, image_id: str) -> Path:
    return dataset_dir / ("%s.png" % image_id)


def _center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return image
    height, width, _channels = image.shape
    if crop_size > height or crop_size > width:
        raise RuntimeError(
            "Requested crop_size=%d exceeds the transformed image dimensions %dx%d; "
            "increase resize_shorter_side or request a smaller crop"
            % (crop_size, height, width)
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size, :]


def _resize_shorter_side(image: np.ndarray, shorter_side: int) -> np.ndarray:
    if shorter_side <= 0:
        return image
    height, width, _channels = image.shape
    if min(height, width) == shorter_side:
        return image
    if height <= width:
        target_height = shorter_side
        target_width = max(1, int(shorter_side * width / height))
    else:
        target_width = shorter_side
        target_height = max(1, int(shorter_side * height / width))
    resized = Image.fromarray(image, mode="RGB").resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def _stack_images_with_padding(images: List[np.ndarray]) -> np.ndarray:
    if not images:
        raise RuntimeError("image_ids must name at least one image")
    max_height = max(int(image.shape[0]) for image in images)
    max_width = max(int(image.shape[1]) for image in images)
    channels = int(images[0].shape[2])
    batch = np.zeros((len(images), max_height, max_width, channels), dtype=np.uint8)
    for index, image in enumerate(images):
        height, width, image_channels = image.shape
        if int(image_channels) != channels:
            raise RuntimeError("Selected images must have the same number of channels")
        batch[index, : int(height), : int(width), :] = image.astype(np.uint8, copy=False)
    return batch


def _read_image_rgb(path: Path) -> np.ndarray:
    """Decode an inventoried image into the operation's canonical RGB tensor."""

    if path.suffix.lower() == ".png":
        return _read_png_rgb(path)
    try:
        with Image.open(path) as opened:
            opened.load()
            return np.asarray(opened.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError("Cannot decode image file %s: %s" % (path, exc)) from exc


def _read_png_rgb(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Not a PNG file: %s" % path)
    offset = 8
    width: Optional[int] = None
    height: Optional[int] = None
    bit_depth: Optional[int] = None
    color_type: Optional[int] = None
    interlace: Optional[int] = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise RuntimeError("PNG is missing IHDR: %s" % path)
    if bit_depth != 8:
        raise RuntimeError("Only 8-bit PNG images are supported: %s" % path)
    if interlace not in (0, None):
        raise RuntimeError("Interlaced PNG images are not supported: %s" % path)
    channels = _png_channels(color_type)
    raw = zlib.decompress(bytes(compressed))
    row_bytes = width * channels
    expected = (row_bytes + 1) * height
    if len(raw) != expected:
        raise RuntimeError("PNG data length mismatch for %s" % path)
    rows = _unfilter_png_rows(raw, width=width, height=height, channels=channels)
    image = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape((height, width, channels))
    if color_type == 0:
        return np.repeat(image, 3, axis=2)
    if color_type == 2:
        return image
    if color_type == 6:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(np.rint(rgb * alpha + 255.0 * (1.0 - alpha)), 0, 255).astype(np.uint8)
    raise RuntimeError("Unsupported PNG color type %s for %s" % (color_type, path))


def _png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 6:
        return 4
    raise RuntimeError("Unsupported PNG color type: %s" % color_type)


def _unfilter_png_rows(raw: bytes, width: int, height: int, channels: int) -> List[bytes]:
    row_bytes = width * channels
    rows: List[bytes] = []
    previous = bytearray(row_bytes)
    offset = 0
    for _row_index in range(height):
        filter_type = raw[offset]
        offset += 1
        current = bytearray(raw[offset : offset + row_bytes])
        offset += row_bytes
        for index in range(row_bytes):
            left = current[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                value = current[index]
            elif filter_type == 1:
                value = current[index] + left
            elif filter_type == 2:
                value = current[index] + up
            elif filter_type == 3:
                value = current[index] + ((left + up) // 2)
            elif filter_type == 4:
                value = current[index] + _paeth(left, up, up_left)
            else:
                raise RuntimeError("Unsupported PNG filter type: %s" % filter_type)
            current[index] = value & 0xFF
        rows.append(bytes(current))
        previous = current
    return rows


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left
