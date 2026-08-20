from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata
from noema_lab.ops.source.coco_yolo import COCO80_NAMES

JsonDict = Dict[str, Any]
YOLO_ASSET_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
YOLO_KNOWN_ASSETS = {"yolo11n.pt", "yolo11n-seg.pt", "yolov8n.pt", "yolov8n-seg.pt"}
YOLO_KNOWN_SHA256 = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11n-seg.pt": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "yolov8n-seg.pt": "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c",
}
YOLO_KNOWN_SIZE_BYTES = {
    "yolo11n.pt": 5_613_764,
    "yolo11n-seg.pt": 6_182_636,
    "yolov8n.pt": 6_549_796,
    "yolov8n-seg.pt": 7_071_756,
}
MAX_YOLO_DOWNLOAD_BYTES = 1024 * 1024 * 1024


class YoloDetectOperation(Operation):
    id = "foundation.yolo_detect"
    name = "YOLO object detector"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"detections": "vision.detections.json"}
    params_schema = object_schema(
        {
            "model_id": {"type": "string", "default": "yolo11n.pt"},
            "expected_model_sha256": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{64})$",
                "description": "Required for a remote model URL; known built-in asset names use release-pinned hashes.",
            },
            "device": {"type": "string", "default": "cpu"},
            "confidence": {"type": "number", "default": 0.25, "minimum": 0.0, "maximum": 1.0},
            "iou": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 1.0},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _ultralytics_availability()
        payload["model_info"] = {
            "default_model": "yolo11n.pt",
            "family": "Ultralytics YOLO pretrained object detection",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _prepare_ultralytics_env()
        YOLO = _require_yolo()
        images, image_metadata = _load_images(ctx.require_input("images").path)
        sample_ids = _sample_ids(image_metadata, int(images.shape[0]))
        model_id = str(ctx.params.get("model_id") or "yolo11n.pt")
        model_ref = _resolve_yolo_model_ref(
            model_id, ctx.params.get("expected_model_sha256")
        )
        model_sha256 = _sha256_file(Path(model_ref))
        device = str(ctx.params.get("device") or "cpu")
        confidence = float(ctx.params.get("confidence") if ctx.params.get("confidence") is not None else 0.25)
        iou = float(ctx.params.get("iou") if ctx.params.get("iou") is not None else 0.7)
        setup_start = time.perf_counter()
        model = YOLO(model_ref)
        setup_s = time.perf_counter() - setup_start
        measurements: List[JsonDict] = []
        append_measurement(measurements, None, "decoder.setup", setup_s, example_count=1, model_id=model_id)
        examples = []
        total = max(1, int(images.shape[0]))
        for index, image in enumerate(images):
            ctx.report_progress(
                "Detecting objects %d/%d examples" % (index, total),
                phase="yolo_detect",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="images",
            )
            infer_start = time.perf_counter()
            result = model.predict(
                source=np.asarray(image, dtype=np.uint8),
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            infer_s = time.perf_counter() - infer_start
            sample_id = sample_ids[index]
            append_measurement(measurements, index, "decoder.inference", infer_s, example_count=1, model_id=model_id, sample_id=sample_id)
            examples.append({"id": sample_id, "detections": _detections_from_result(result)})
        ctx.report_progress(
            "Detecting objects %d/%d examples" % (int(images.shape[0]), total),
            phase="yolo_detect",
            status="running",
            completed=int(images.shape[0]),
            total=total,
            percent=100.0,
            unit="images",
        )
        metadata = {
            "adapter_family": "yolo_detection",
            "model_id": model_id,
            "model_sha256": model_sha256,
            "runner": "pytorch",
            "device": device,
            "confidence": confidence,
            "iou": iou,
            "inference_confidence_threshold": confidence,
            "inference_iou_threshold": iou,
            "predictions_pre_filtered_by_model": True,
            "sample_ids": sample_ids,
            "detections_preview": examples[:10],
            "source_image_metadata": image_metadata,
            "codec_timing": codec_timing_metadata(
                "decoder",
                measurements,
                runner="pytorch",
                notes={"runtime_execution_language": "Python", "foundation_adapter": "YOLO object detection"},
            ),
        }
        path = ctx.output_path("detections", ".json")
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "vision.detections",
                    "model_id": model_id,
                    "model_sha256": model_sha256,
                    "inference_confidence_threshold": confidence,
                    "inference_iou_threshold": iou,
                    "predictions_pre_filtered_by_model": True,
                    "examples": examples,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return OperationResult(
            outputs={"detections": artifact("vision.detections.json", path, metadata)},
            metrics={"detection.predicted_box_count": sum(len(item["detections"]) for item in examples)},
            metadata=metadata,
        )


class YoloSegmentOperation(Operation):
    id = "foundation.yolo_segment"
    name = "YOLO instance segmenter"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"segmentation": "vision.segmentation_mask.numpy"}
    params_schema = object_schema(
        {
            "model_id": {"type": "string", "default": "yolo11n-seg.pt"},
            "expected_model_sha256": {
                "type": "string",
                "default": "",
                "pattern": "^(|[0-9a-fA-F]{64})$",
                "description": "Required for a remote model URL; known built-in asset names use release-pinned hashes.",
            },
            "device": {"type": "string", "default": "cpu"},
            "confidence": {"type": "number", "default": 0.25, "minimum": 0.0, "maximum": 1.0},
            "iou": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 1.0},
            "mask_threshold": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _ultralytics_availability()
        payload["model_info"] = {
            "default_model": "yolo11n-seg.pt",
            "family": "Ultralytics YOLO pretrained instance segmentation",
        }
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _prepare_ultralytics_env()
        YOLO = _require_yolo()
        images, image_metadata = _load_images(ctx.require_input("images").path)
        sample_ids = _sample_ids(image_metadata, int(images.shape[0]))
        model_id = str(ctx.params.get("model_id") or "yolo11n-seg.pt")
        model_ref = _resolve_yolo_model_ref(
            model_id, ctx.params.get("expected_model_sha256")
        )
        model_sha256 = _sha256_file(Path(model_ref))
        device = str(ctx.params.get("device") or "cpu")
        confidence = float(ctx.params.get("confidence") if ctx.params.get("confidence") is not None else 0.25)
        iou = float(ctx.params.get("iou") if ctx.params.get("iou") is not None else 0.7)
        mask_threshold = float(ctx.params.get("mask_threshold") if ctx.params.get("mask_threshold") is not None else 0.5)
        setup_start = time.perf_counter()
        model = YOLO(model_ref)
        setup_s = time.perf_counter() - setup_start
        measurements: List[JsonDict] = []
        append_measurement(measurements, None, "decoder.setup", setup_s, example_count=1, model_id=model_id)
        output_masks = []
        total = max(1, int(images.shape[0]))
        for index, image in enumerate(images):
            ctx.report_progress(
                "Segmenting images %d/%d examples" % (index, total),
                phase="yolo_segment",
                status="running",
                completed=index,
                total=total,
                percent=float(index) / float(total) * 100.0,
                unit="images",
            )
            infer_start = time.perf_counter()
            result = model.predict(
                source=np.asarray(image, dtype=np.uint8),
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            infer_s = time.perf_counter() - infer_start
            sample_id = sample_ids[index]
            append_measurement(measurements, index, "decoder.inference", infer_s, example_count=1, model_id=model_id, sample_id=sample_id)
            output_masks.append(_segmentation_mask_from_result(result, int(image.shape[1]), int(image.shape[0]), mask_threshold))
        ctx.report_progress(
            "Segmenting images %d/%d examples" % (int(images.shape[0]), total),
            phase="yolo_segment",
            status="running",
            completed=int(images.shape[0]),
            total=total,
            percent=100.0,
            unit="images",
        )
        masks = np.stack(output_masks, axis=0).astype(np.uint16, copy=False)
        metadata = {
            "adapter_family": "yolo_segmentation",
            "model_id": model_id,
            "model_sha256": model_sha256,
            "runner": "pytorch",
            "device": device,
            "confidence": confidence,
            "iou": iou,
            "mask_threshold": mask_threshold,
            "sample_ids": sample_ids,
            "mask_shape": [int(value) for value in masks.shape[1:]],
            "class_names": COCO80_NAMES,
            "source_image_metadata": image_metadata,
            "codec_timing": codec_timing_metadata(
                "decoder",
                measurements,
                runner="pytorch",
                notes={"runtime_execution_language": "Python", "foundation_adapter": "YOLO instance segmentation"},
            ),
        }
        path = ctx.output_path("segmentation", ".npz")
        np.savez_compressed(path, masks=masks, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"segmentation": artifact("vision.segmentation_mask.numpy", path, metadata)},
            metrics={"segmentation.predicted_pixel_count": int((masks > 0).sum())},
            metadata=metadata,
        )


def _require_yolo():
    if importlib.util.find_spec("ultralytics") is None:
        raise OperationError(
            'Install with `python -m pip install "noema-lab[vision]"` in an '
            "installed environment, or `uv sync --extra vision` in a source "
            "checkout, to use YOLO detection/segmentation adapters"
        )
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise OperationError("Could not import ultralytics YOLO: %s" % exc) from exc
    return YOLO


def _ultralytics_availability() -> JsonDict:
    if importlib.util.find_spec("ultralytics") is None:
        return {
            "available": False,
            "extra": "vision",
            "missing": ["ultralytics"],
            "reason": (
                'Install with `python -m pip install "noema-lab[vision]"` in an '
                "installed environment, or `uv sync --extra vision` in a source "
                "checkout, to use YOLO detection/segmentation adapters"
            ),
        }
    return {"available": True, "extra": "vision", "missing": []}


def _prepare_ultralytics_env() -> None:
    config_dir = Path(".noema") / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir.resolve()))


def _resolve_yolo_model_ref(model_id: str, expected_sha256: Any = "") -> str:
    raw = str(model_id or "").strip()
    if not raw:
        raise OperationError("YOLO model_id must not be empty")
    supplied_sha256 = _normalized_sha256(expected_sha256)
    path = Path(raw)
    if path.is_file():
        required_sha256 = supplied_sha256 or YOLO_KNOWN_SHA256.get(raw, "")
        if required_sha256:
            _require_sha256(path, required_sha256, "local YOLO model")
        return str(path)
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.netloc:
            raise OperationError("Remote YOLO model URLs must use HTTPS")
        if not supplied_sha256:
            raise OperationError(
                "Remote YOLO model URLs require expected_model_sha256"
            )
        filename = Path(urllib.parse.unquote(parsed.path)).name
        if not filename or filename in (".", ".."):
            raise OperationError("Remote YOLO model URL has no safe filename")
        target_dir = Path(".noema") / "checkpoints" / "yolo"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        if target.is_file():
            _require_sha256(target, supplied_sha256, "cached YOLO model")
        else:
            _download_yolo_asset(raw, target, supplied_sha256)
        return str(target)
    if path.parent != Path(".") or raw not in YOLO_KNOWN_ASSETS:
        raise OperationError(
            "YOLO model `%s` is not a local file. Use a pinned built-in asset name, "
            "an existing local path, or an HTTPS URL with expected_model_sha256."
            % raw
        )
    pinned_sha256 = supplied_sha256 or YOLO_KNOWN_SHA256[raw]
    target_dir = Path(".noema") / "checkpoints" / "yolo"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / raw
    if target.is_file():
        _require_sha256(target, pinned_sha256, "cached YOLO model")
        return str(target)
    _download_yolo_asset(
        "%s/%s" % (YOLO_ASSET_BASE_URL, raw), target, pinned_sha256
    )
    return str(target)


def _download_yolo_asset(url: str, target: Path, expected_sha256: str) -> None:
    try:
        expected_size = None
        if YOLO_KNOWN_SHA256.get(target.name) == expected_sha256:
            expected_size = YOLO_KNOWN_SIZE_BYTES[target.name]
        download_verified_https(
            url,
            target,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            max_bytes=MAX_YOLO_DOWNLOAD_BYTES,
            timeout_s=180,
            opener=urllib.request.urlopen,
        )
    except Exception as exc:
        raise OperationError("Could not download YOLO model %s: %s" % (url, exc)) from exc


def _normalized_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if not digest:
        return ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise OperationError("expected_model_sha256 must be a 64-character hexadecimal SHA-256")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OperationError("Could not hash YOLO model %s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _require_sha256(path: Path, expected_sha256: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise OperationError(
            "%s SHA-256 mismatch for %s: expected %s, got %s"
            % (label, path, expected_sha256, actual)
        )
    return actual


def _load_images(path: Path):
    with np.load(str(path), allow_pickle=False) as payload:
        images = np.asarray(payload["images"], dtype=np.uint8)
        metadata: JsonDict = {}
        if "metadata_json" in payload:
            metadata = decode_strict_json_object(
                str(payload["metadata_json"]),
                label="YOLO image artifact metadata_json",
            )
    return images, metadata


def _sample_ids(metadata: JsonDict, count: int) -> List[str]:
    raw = metadata.get("sample_ids") or metadata.get("ids")
    if isinstance(raw, list) and len(raw) >= count:
        return [str(item) for item in raw[:count]]
    return ["image_%03d" % (index + 1) for index in range(count)]


def _detections_from_result(result: Any) -> List[JsonDict]:
    if result.boxes is None:
        return []
    names = result.names if isinstance(getattr(result, "names", None), dict) else {}
    boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((len(boxes),), dtype=int)
    scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(boxes),), dtype=np.float32)
    detections = []
    for bbox, class_id, score in zip(boxes, classes, scores):
        detections.append(
            {
                "label": str(names.get(int(class_id), _class_name(int(class_id)))),
                "class_id": int(class_id),
                "bbox": [float(value) for value in bbox.tolist()],
                "score": float(score),
            }
        )
    return detections


def _segmentation_mask_from_result(result: Any, width: int, height: int, threshold: float) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint16)
    if result.masks is None or result.boxes is None:
        return output
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((len(masks),), dtype=int)
    scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(masks),), dtype=np.float32)
    order = np.argsort(scores)
    for index in order:
        mask = np.asarray(masks[int(index)], dtype=np.float32)
        if mask.shape != (height, width):
            mask = _resize_mask(mask, width, height)
        output[mask >= threshold] = np.uint16(int(classes[int(index)]) + 1)
    return output


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as exc:
        raise OperationError("Install Pillow to resize segmentation masks") from exc
    image = Image.fromarray(np.asarray(mask * 255.0, dtype=np.uint8))
    return np.asarray(image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO80_NAMES):
        return COCO80_NAMES[class_id]
    return "class_%d" % class_id
