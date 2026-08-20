from __future__ import annotations

import hashlib
import json
import math
import stat
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema

JsonDict = Dict[str, Any]

COCO128_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
COCO8_SEG_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip"
COCO128_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
COCO8_SEG_SHA256 = "82c651abd01d556c77769ea834ebff1e77f76a291463e987fb69b63a87e8eb80"
DATASET_ARCHIVE_SIZE_BYTES = {
    COCO128_URL: 6_983_030,
    COCO8_SEG_URL: 449_818,
}
DOWNLOAD_TIMEOUT_S = 180
MAX_ARCHIVE_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2048
MAX_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0

COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class Coco128DetectionOperation(Operation):
    id = "source.coco128_detection"
    name = "COCO128 object-detection source"
    output_kinds = {
        "images": "image.batch.numpy",
        "detections": "vision.detections.json",
    }
    params_schema = object_schema(
        {
            "dataset_dir": {"type": "string", "default": ".noema/datasets/coco128"},
            "split": {"type": "string", "default": "train2017", "enum": ["train2017"]},
            "limit": {"type": "integer", "default": 8, "minimum": 1},
            "image_size": {"type": "integer", "default": 320, "minimum": 32},
            "download": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["dataset_info"] = {
            "dataset": "coco128",
            "source": "Ultralytics COCO128, a 128-image COCO subset",
            "url": COCO128_URL,
            "sha256": COCO128_SHA256,
            "download_size": "about 6.8 MB",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset_dir = Path(str(ctx.params.get("dataset_dir") or ".noema/datasets/coco128"))
        split = str(ctx.params.get("split") or "train2017")
        if split != "train2017":
            raise OperationError("COCO128 detection currently supports split train2017")
        limit = max(1, int(ctx.params.get("limit") or 8))
        image_size = max(32, int(ctx.params.get("image_size") or 320))
        root = _ensure_dataset(
            dataset_dir,
            "coco128",
            COCO128_URL,
            COCO128_SHA256,
            bool(ctx.params.get("download", True)),
        )
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        examples = _load_detection_examples(image_dir, label_dir, limit, image_size)
        if not examples:
            raise OperationError("COCO128 detection source found no examples in %s" % image_dir)

        images = np.stack([item["image"] for item in examples], axis=0).astype(np.uint8, copy=False)
        detections = [{"id": item["id"], "detections": item["detections"]} for item in examples]
        metadata = {
            "dataset": "coco128",
            "split": split,
            "source_url": COCO128_URL,
            "source_sha256": COCO128_SHA256,
            "example_count": len(examples),
            "sample_ids": [item["id"] for item in examples],
            "image_ids": [item["id"] for item in examples],
            "image_shape": [int(value) for value in images.shape[1:]],
            "class_names": COCO80_NAMES,
            "detections_preview": detections[:10],
            "label_validation": "strict_yolo_detection_v1",
            "empty_label_count": sum(
                1 for item in examples if int(item["annotation_count"]) == 0
            ),
            "label_annotation_count": sum(
                int(item["annotation_count"]) for item in examples
            ),
        }
        image_path = ctx.output_path("images", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        detections_path = _write_json(
            ctx,
            "detections",
            {
                "schema_version": 1,
                "kind": "vision.detections",
                "dataset": "coco128",
                "split": split,
                "label_validation": metadata["label_validation"],
                "empty_label_count": metadata["empty_label_count"],
                "label_annotation_count": metadata[
                    "label_annotation_count"
                ],
                "examples": detections,
            },
        )
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "detections": artifact("vision.detections.json", detections_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples), "detection.reference_box_count": sum(len(item["detections"]) for item in examples)},
            metadata=metadata,
        )


class Coco8SegmentationOperation(Operation):
    id = "source.coco8_segmentation"
    name = "COCO8 segmentation source"
    output_kinds = {
        "images": "image.batch.numpy",
        "segmentation": "vision.segmentation_mask.numpy",
    }
    params_schema = object_schema(
        {
            "dataset_dir": {"type": "string", "default": ".noema/datasets/coco8-seg"},
            "split": {"type": "string", "default": "val", "enum": ["train", "val"]},
            "limit": {"type": "integer", "default": 4, "minimum": 1},
            "image_size": {"type": "integer", "default": 320, "minimum": 32},
            "download": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["dataset_info"] = {
            "dataset": "coco8-seg",
            "source": "Ultralytics COCO8-seg, an 8-image COCO segmentation subset",
            "url": COCO8_SEG_URL,
            "sha256": COCO8_SEG_SHA256,
            "download_size": "about 439 KB",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        dataset_dir = Path(str(ctx.params.get("dataset_dir") or ".noema/datasets/coco8-seg"))
        split = str(ctx.params.get("split") or "val")
        if split not in ("train", "val"):
            raise OperationError("COCO8 segmentation supports split train or val")
        limit = max(1, int(ctx.params.get("limit") or 4))
        image_size = max(32, int(ctx.params.get("image_size") or 320))
        root = _ensure_dataset(
            dataset_dir,
            "coco8-seg",
            COCO8_SEG_URL,
            COCO8_SEG_SHA256,
            bool(ctx.params.get("download", True)),
        )
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        examples = _load_segmentation_examples(image_dir, label_dir, limit, image_size)
        if not examples:
            raise OperationError("COCO8 segmentation source found no examples in %s" % image_dir)

        images = np.stack([item["image"] for item in examples], axis=0).astype(np.uint8, copy=False)
        masks = np.stack([item["mask"] for item in examples], axis=0).astype(np.uint16, copy=False)
        metadata = {
            "dataset": "coco8-seg",
            "split": split,
            "source_url": COCO8_SEG_URL,
            "source_sha256": COCO8_SEG_SHA256,
            "example_count": len(examples),
            "sample_ids": [item["id"] for item in examples],
            "image_ids": [item["id"] for item in examples],
            "image_shape": [int(value) for value in images.shape[1:]],
            "mask_shape": [int(value) for value in masks.shape[1:]],
            "class_names": COCO80_NAMES,
            "classes_present": sorted({int(value) for value in np.unique(masks) if int(value) > 0}),
            "label_validation": "strict_yolo_segmentation_v1",
            "empty_label_count": sum(
                1 for item in examples if int(item["annotation_count"]) == 0
            ),
            "label_annotation_count": sum(
                int(item["annotation_count"]) for item in examples
            ),
        }
        image_path = ctx.output_path("images", ".npz")
        mask_path = ctx.output_path("segmentation", ".npz")
        np.savez_compressed(image_path, images=images, metadata_json=json.dumps(metadata))
        np.savez_compressed(mask_path, masks=masks, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", image_path, metadata),
                "segmentation": artifact("vision.segmentation_mask.numpy", mask_path, metadata),
            },
            metrics={"task.dataset.example_count": len(examples), "segmentation.reference_pixel_count": int(masks.size)},
            metadata=metadata,
        )


def _ensure_dataset(
    dataset_dir: Path,
    root_name: str,
    url: str,
    expected_sha256: str,
    should_download: bool,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    root = dataset_dir / root_name
    archive = dataset_dir / ("%s.zip" % root_name)
    if root.is_symlink():
        raise OperationError("Cached dataset root must not be a symbolic link: %s" % root)
    if archive.is_symlink():
        raise OperationError("Cached dataset archive must not be a symbolic link: %s" % archive)
    if not archive.is_file():
        if not should_download:
            if root.is_dir():
                raise OperationError(
                    "Cannot verify cached dataset %s because its pinned source archive is missing: %s"
                    % (root, archive)
                )
            raise OperationError("Dataset is not cached: %s" % root)
        _download_file(url, archive, expected_sha256)
    else:
        _require_sha256(archive, expected_sha256, "cached dataset archive")
    if root.is_dir():
        _verify_extracted_zip(archive, dataset_dir, root_name)
        return root
    staging = dataset_dir / (".%s.extracting" % root_name)
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise OperationError("Unsafe stale dataset extraction path: %s" % staging)
        shutil.rmtree(staging)
    try:
        _safe_extract_zip(archive, staging)
        extracted = staging / root_name
        if not extracted.is_dir():
            raise OperationError(
                "Downloaded archive did not contain expected dataset root: %s"
                % root_name
            )
        _verify_extracted_zip(archive, staging, root_name)
        extracted.replace(root)
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
    return root


def _download_file(url: str, target: Path, expected_sha256: str) -> None:
    try:
        download_verified_https(
            url,
            target,
            expected_sha256=expected_sha256,
            expected_size=DATASET_ARCHIVE_SIZE_BYTES.get(url),
            max_bytes=MAX_ARCHIVE_DOWNLOAD_BYTES,
            timeout_s=DOWNLOAD_TIMEOUT_S,
            opener=urllib.request.urlopen,
        )
    except Exception as exc:
        raise OperationError("Could not download %s: %s" % (url, exc)) from exc


def _safe_extract_zip(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = _validated_zip_members(zf, target_dir)
            for member in members:
                relative = PurePosixPath(member.filename)
                destination = target_dir.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(member, "r") as source, destination.open("wb") as output:
                    while True:
                        chunk = source.read(min(1024 * 1024, MAX_MEMBER_UNCOMPRESSED_BYTES + 1 - written))
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_MEMBER_UNCOMPRESSED_BYTES or written > member.file_size:
                            raise OperationError(
                                "Dataset archive member exceeded its declared safe size: %s"
                                % member.filename
                            )
                        output.write(chunk)
                if written != member.file_size:
                    raise OperationError(
                        "Dataset archive member size mismatch for %s: expected %d, got %d"
                        % (member.filename, member.file_size, written)
                    )
    except zipfile.BadZipFile as exc:
        raise OperationError("Invalid dataset ZIP archive %s: %s" % (archive, exc)) from exc


def _validated_zip_members(zf: zipfile.ZipFile, target_dir: Path) -> List[zipfile.ZipInfo]:
    members = zf.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise OperationError(
            "Dataset archive has too many members: %d exceeds %d"
            % (len(members), MAX_ARCHIVE_MEMBERS)
        )
    target_root = target_dir.resolve()
    total_size = 0
    seen: Set[str] = set()
    for member in members:
        name = member.filename
        relative = PurePosixPath(name)
        normalized = relative.as_posix()
        folded = normalized.casefold()
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or (relative.parts and ":" in relative.parts[0])
        ):
            raise OperationError("Unsafe path in dataset archive: %s" % name)
        if folded in seen:
            raise OperationError("Duplicate path in dataset archive: %s" % name)
        seen.add(folded)
        member_path = target_dir.joinpath(*relative.parts).resolve()
        if target_root != member_path and target_root not in member_path.parents:
            raise OperationError("Unsafe path in dataset archive: %s" % name)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise OperationError("Symbolic links are not allowed in dataset archives: %s" % name)
        file_type = stat.S_IFMT(mode)
        if file_type and not member.is_dir() and file_type != stat.S_IFREG:
            raise OperationError("Special files are not allowed in dataset archives: %s" % name)
        if member.flag_bits & 0x1:
            raise OperationError("Encrypted dataset archive members are not supported: %s" % name)
        if member.file_size < 0 or member.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise OperationError(
                "Dataset archive member is too large: %s (%d bytes)"
                % (name, member.file_size)
            )
        total_size += member.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise OperationError(
                "Dataset archive uncompressed size exceeds %d bytes"
                % MAX_TOTAL_UNCOMPRESSED_BYTES
            )
        if member.file_size:
            ratio = float(member.file_size) / float(max(member.compress_size, 1))
            if ratio > MAX_COMPRESSION_RATIO:
                raise OperationError(
                    "Dataset archive member compression ratio is unsafe: %s (%.1f:1)"
                    % (name, ratio)
                )
    return members


def _verify_extracted_zip(archive: Path, target_dir: Path, root_name: str) -> None:
    expected_files: Set[str] = set()
    try:
        with zipfile.ZipFile(archive) as zf:
            members = _validated_zip_members(zf, target_dir)
            for member in members:
                relative = PurePosixPath(member.filename)
                if not relative.parts or relative.parts[0] != root_name:
                    raise OperationError(
                        "Dataset archive contains a path outside expected root %s: %s"
                        % (root_name, member.filename)
                    )
                destination = target_dir.joinpath(*relative.parts)
                if member.is_dir():
                    if not destination.is_dir() or destination.is_symlink():
                        raise OperationError("Cached dataset directory is missing or unsafe: %s" % destination)
                    continue
                expected_files.add(relative.as_posix())
                if not destination.is_file() or destination.is_symlink():
                    raise OperationError("Cached dataset file is missing or unsafe: %s" % destination)
                if destination.stat().st_size != member.file_size:
                    raise OperationError("Cached dataset file size mismatch: %s" % destination)
                with zf.open(member, "r") as expected, destination.open("rb") as actual:
                    while True:
                        expected_chunk = expected.read(1024 * 1024)
                        actual_chunk = actual.read(1024 * 1024)
                        if expected_chunk != actual_chunk:
                            raise OperationError("Cached dataset file content mismatch: %s" % destination)
                        if not expected_chunk:
                            break
    except zipfile.BadZipFile as exc:
        raise OperationError("Invalid dataset ZIP archive %s: %s" % (archive, exc)) from exc
    root = target_dir / root_name
    actual_files: Set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise OperationError("Cached dataset contains a symbolic link: %s" % path)
        if path.is_file():
            actual_files.add(path.relative_to(target_dir).as_posix())
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise OperationError("Cached dataset contains unexpected file: %s" % unexpected[0])


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


def _load_detection_examples(image_dir: Path, label_dir: Path, limit: int, image_size: int) -> List[JsonDict]:
    examples: List[JsonDict] = []
    for image_path in _image_files(image_dir)[:limit]:
        label_path = label_dir / (image_path.stem + ".txt")
        image = _load_image(image_path, image_size)
        height, width = int(image.shape[0]), int(image.shape[1])
        detections = _read_yolo_boxes(label_path, width, height)
        examples.append(
            {
                "id": image_path.stem,
                "image": image,
                "detections": detections,
                "annotation_count": len(detections),
            }
        )
    return examples


def _load_segmentation_examples(image_dir: Path, label_dir: Path, limit: int, image_size: int) -> List[JsonDict]:
    examples: List[JsonDict] = []
    for image_path in _image_files(image_dir)[:limit]:
        label_path = label_dir / (image_path.stem + ".txt")
        image = _load_image(image_path, image_size)
        height, width = int(image.shape[0]), int(image.shape[1])
        mask = _read_yolo_segmentation_mask(label_path, width, height)
        examples.append(
            {
                "id": image_path.stem,
                "image": image,
                "mask": mask,
                "annotation_count": _nonempty_label_line_count(label_path),
            }
        )
    return examples


def _image_files(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise OperationError("Image directory does not exist: %s" % image_dir)
    files = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(image_dir.glob(suffix))
    return sorted(files)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to load COCO image files") from exc
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(rgb, dtype=np.uint8)


def _read_yolo_boxes(path: Path, width: int, height: int) -> List[JsonDict]:
    if not path.is_file():
        raise OperationError("YOLO detection label file is missing: %s" % path)
    if width <= 0 or height <= 0:
        raise OperationError("YOLO detection raster dimensions must be positive")
    detections = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = _strict_float_parts(path, line_number, line)
        if len(parts) != 5:
            raise OperationError(
                "Malformed YOLO detection label %s:%d: expected exactly 5 fields, got %d"
                % (path, line_number, len(parts))
            )
        class_id = _strict_class_id(path, line_number, parts[0])
        cx, cy, box_w, box_h = parts[1:5]
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise OperationError(
                "Malformed YOLO detection label %s:%d: center coordinates must be in [0, 1]"
                % (path, line_number)
            )
        if not (0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0):
            raise OperationError(
                "Malformed YOLO detection label %s:%d: box width/height must be in (0, 1]"
                % (path, line_number)
            )
        x1 = max(0.0, (cx - box_w / 2.0) * width)
        y1 = max(0.0, (cy - box_h / 2.0) * height)
        x2 = min(float(width), (cx + box_w / 2.0) * width)
        y2 = min(float(height), (cy + box_h / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            raise OperationError(
                "Malformed YOLO detection label %s:%d: clipped box has no positive area"
                % (path, line_number)
            )
        detections.append(
            {
                "label": _class_name(class_id),
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2],
                "score": 1.0,
            }
        )
    return detections


def _read_yolo_segmentation_mask(path: Path, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise OperationError("Install Pillow to rasterize COCO segmentation labels") from exc
    output = np.zeros((height, width), dtype=np.uint16)
    if not path.is_file():
        raise OperationError("YOLO segmentation label file is missing: %s" % path)
    if width <= 0 or height <= 0:
        raise OperationError("YOLO segmentation raster dimensions must be positive")
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = _strict_float_parts(path, line_number, line)
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise OperationError(
                "Malformed YOLO segmentation label %s:%d: expected a class and at "
                "least three x/y coordinate pairs"
                % (path, line_number)
            )
        class_id = _strict_class_id(path, line_number, parts[0])
        coords = parts[1:]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise OperationError(
                "Malformed YOLO segmentation label %s:%d: polygon coordinates must "
                "be in [0, 1]"
                % (path, line_number)
            )
        points = _polygon_points(coords, width, height)
        if _polygon_area(points) <= 0.0:
            raise OperationError(
                "Malformed YOLO segmentation label %s:%d: polygon is degenerate"
                % (path, line_number)
            )
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon(points, outline=1, fill=1)
        raster = np.asarray(layer, dtype=bool)
        if not raster.any():
            raise OperationError(
                "Malformed YOLO segmentation label %s:%d: polygon rasterizes to no pixels"
                % (path, line_number)
            )
        output[raster] = np.uint16(class_id + 1)
    return output


def _polygon_points(coords: Sequence[float], width: int, height: int) -> List[Tuple[float, float]]:
    points = []
    usable = len(coords) - (len(coords) % 2)
    for index in range(0, usable, 2):
        points.append((float(coords[index]) * width, float(coords[index + 1]) * height))
    return points


def _strict_float_parts(path: Path, line_number: int, line: str) -> List[float]:
    output: List[float] = []
    for column, part in enumerate(line.strip().split(), start=1):
        try:
            value = float(part)
        except ValueError as exc:
            raise OperationError(
                "Malformed YOLO label %s:%d column %d: `%s` is not numeric"
                % (path, line_number, column, part)
            ) from exc
        if not math.isfinite(value):
            raise OperationError(
                "Malformed YOLO label %s:%d column %d: values must be finite"
                % (path, line_number, column)
            )
        output.append(value)
    return output


def _strict_class_id(path: Path, line_number: int, value: float) -> int:
    if not float(value).is_integer():
        raise OperationError(
            "Malformed YOLO label %s:%d: class id must be an integer"
            % (path, line_number)
        )
    class_id = int(value)
    if not 0 <= class_id < len(COCO80_NAMES):
        raise OperationError(
            "Malformed YOLO label %s:%d: class id %d is outside [0, %d]"
            % (path, line_number, class_id, len(COCO80_NAMES) - 1)
        )
    return class_id


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _nonempty_label_line_count(path: Path) -> int:
    if not path.is_file():
        raise OperationError("YOLO label file is missing: %s" % path)
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO80_NAMES):
        return COCO80_NAMES[class_id]
    return "class_%d" % class_id


def _write_json(ctx: OperationContext, name: str, payload: JsonDict) -> Path:
    path = ctx.output_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
