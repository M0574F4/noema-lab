from __future__ import annotations

import base64
import io
import importlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema

JsonDict = Dict[str, Any]


class ImageReconstructionMetricsOperation(Operation):
    id = "metrics.image_reconstruction"
    name = "Image reconstruction metrics"
    input_kinds = {
        "reference": ["image.batch.numpy"],
        "reconstruction": ["image.batch.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "data_range": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": (
                    "Maximum minus minimum pixel value used by PSNR; 0 infers the canonical "
                    "range from a shared uint8 ([0,255]) or float32 ([0,1]) dtype."
                ),
            },
            "psnr_cap_db": {
                "type": "number",
                "default": 99.0,
                "minimum": 0.0,
                "description": (
                    "Requested finite reporting floor for exact reconstructions. The evaluator "
                    "raises it when necessary so an exact reconstruction always scores above "
                    "every finite PSNR representable for the evaluated image domain and shape."
                ),
            },
            "preview_count": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 4,
                "description": (
                    "Opt in to embedding up to four compact reference/reconstruction PNG "
                    "thumbnails in the metrics report for portable result documentation."
                ),
            },
            "preview_size": {
                "type": "integer",
                "default": 128,
                "minimum": 64,
                "maximum": 256,
                "description": "Maximum thumbnail width or height when preview_count is nonzero.",
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        reference, reference_metadata = _load_images(ctx.require_input("reference").path)
        reconstruction, reconstruction_metadata = _load_images(ctx.require_input("reconstruction").path)
        if reference.shape[0] != reconstruction.shape[0]:
            raise RuntimeError(
                "Image reconstruction count mismatch: reference has %d images, reconstruction has %d"
                % (reference.shape[0], reconstruction.shape[0])
            )
        count = int(reference.shape[0])
        if count <= 0:
            raise RuntimeError("Image reconstruction metrics require at least one image")
        reference_shapes = _image_shapes(reference, reference_metadata, "reference")
        reconstruction_shapes = _image_shapes(
            reconstruction, reconstruction_metadata, "reconstruction"
        )
        if reference_shapes != reconstruction_shapes:
            raise RuntimeError(
                "Image reconstruction shape mismatch: reference original shapes %s, reconstruction original shapes %s"
                % (reference_shapes, reconstruction_shapes)
            )
        sample_ids, pairing_mode = _paired_image_ids(
            reference_metadata, reconstruction_metadata, count
        )
        declared_data_range = float(ctx.params.get("data_range", 0.0))
        data_range, data_range_source = _resolve_data_range(
            reference, reconstruction, declared_data_range
        )
        requested_psnr_cap_db = float(ctx.params.get("psnr_cap_db", 99.0))
        if not math.isfinite(requested_psnr_cap_db) or requested_psnr_cap_db < 0.0:
            raise RuntimeError("psnr_cap_db must be finite and non-negative")
        minimum_safe_psnr_cap_db = _minimum_safe_exact_psnr_cap_db(
            reference.dtype, reference_shapes, data_range
        )
        psnr_cap_db = max(requested_psnr_cap_db, minimum_safe_psnr_cap_db)
        rows = []
        for index in range(count):
            height, width, channels = reference_shapes[index]
            ref = reference[index, :height, :width, :channels].astype(np.float64)
            rec = reconstruction[index, :height, :width, :channels].astype(np.float64)
            _validate_image_values(ref, data_range, "reference", index)
            _validate_image_values(rec, data_range, "reconstruction", index)
            diff = ref - rec
            mse = float(np.mean(np.square(diff)))
            mae = float(np.mean(np.abs(diff)))
            psnr, exact = _psnr(mse, data_range, psnr_cap_db)
            rows.append(
                {
                    "index": index,
                    "id": sample_ids[index],
                    "shape": [1, height, width, channels],
                    "mse": mse,
                    "mae": mae,
                    "psnr_db": psnr,
                    "perfect_reconstruction": exact,
                }
            )
        mse = float(np.mean([row["mse"] for row in rows]))
        mae = float(np.mean([row["mae"] for row in rows]))
        psnr = float(np.mean([row["psnr_db"] for row in rows]))
        perfect_reconstruction_fraction = float(
            np.mean([1.0 if row["perfect_reconstruction"] else 0.0 for row in rows])
        )
        ms_ssim_values = _try_ms_ssim(
            reference, reconstruction, reference_shapes, data_range
        )
        ms_ssim_eligible = [
            min(int(height), int(width)) >= 161
            for height, width, _channels in reference_shapes
        ]
        if ms_ssim_values is not None:
            if len(ms_ssim_values) != count:
                raise RuntimeError(
                    "MS-SSIM evaluator returned %d rows for %d images"
                    % (len(ms_ssim_values), count)
                )
            for row, value in zip(rows, ms_ssim_values):
                row["ms_ssim"] = value
        for index, row in enumerate(rows):
            row["ms_ssim_eligible"] = bool(ms_ssim_eligible[index])
            if not ms_ssim_eligible[index]:
                row["ms_ssim_status"] = "ineligible_min_dimension_below_161"
            elif ms_ssim_values is None:
                row["ms_ssim_status"] = "dependency_unavailable"
            elif ms_ssim_values[index] is None:
                row["ms_ssim_status"] = "eligible_not_evaluated"
            else:
                row["ms_ssim_status"] = "evaluated"
        valid_ms_ssim = [value for value in (ms_ssim_values or []) if value is not None]
        ms_ssim_value = float(np.mean(valid_ms_ssim)) if valid_ms_ssim else None
        ms_ssim_eligible_count = int(sum(ms_ssim_eligible))
        ms_ssim_evaluated_count = int(len(valid_ms_ssim))
        ms_ssim_coverage = {
            "total_image_count": int(count),
            "eligible_image_count": ms_ssim_eligible_count,
            "evaluated_image_count": ms_ssim_evaluated_count,
            "evaluated_fraction_of_total": float(ms_ssim_evaluated_count) / float(count),
            "evaluated_fraction_of_eligible": (
                float(ms_ssim_evaluated_count) / float(ms_ssim_eligible_count)
                if ms_ssim_eligible_count
                else 0.0
            ),
            "eligibility_rule": "minimum original image dimension >= 161 pixels",
            "aggregation_scope": "arithmetic mean over evaluated eligible images only",
            "dependency_available": ms_ssim_values is not None,
        }
        report = {
            "schema_version": 2,
            "metric_family": "image_reconstruction",
            "num_images": int(count),
            "reference_storage_shape": list(reference.shape),
            "reconstruction_storage_shape": list(reconstruction.shape),
            "original_shapes": [[1, *shape] for shape in reference_shapes],
            "sample_pairing": pairing_mode,
            "data_range": data_range,
            "data_range_source": data_range_source,
            "psnr_cap_db": psnr_cap_db,
            "requested_psnr_cap_db": requested_psnr_cap_db,
            "minimum_safe_psnr_cap_db": minimum_safe_psnr_cap_db,
            "aggregation": {
                "mse": "arithmetic mean of per-image pixel MSE",
                "mae": "arithmetic mean of per-image pixel MAE",
                "psnr_db": (
                    "arithmetic mean of per-image PSNR; exact reconstructions use the effective "
                    "psnr_cap_db, which is strictly above the finite PSNR ceiling for this image domain"
                ),
                "ms_ssim": "arithmetic mean over images large enough for MS-SSIM",
            },
            "mse": mse,
            "mae": mae,
            "psnr_db": psnr,
            "perfect_reconstruction_fraction": perfect_reconstruction_fraction,
            "ms_ssim_coverage": ms_ssim_coverage,
            "per_example": rows,
        }
        preview_count = min(count, int(ctx.params.get("preview_count", 0)))
        if preview_count:
            report["image_preview"] = _image_preview(
                reference,
                reconstruction,
                reference_shapes,
                sample_ids,
                data_range,
                preview_count,
                int(ctx.params.get("preview_size", 128)),
            )
        metrics = {
            "quality.mse": mse,
            "quality.mae": mae,
            "quality.psnr_db": psnr,
            "quality.perfect_reconstruction_fraction": perfect_reconstruction_fraction,
            "quality.ms_ssim.total_image_count": int(count),
            "quality.ms_ssim.eligible_image_count": ms_ssim_eligible_count,
            "quality.ms_ssim.evaluated_image_count": ms_ssim_evaluated_count,
            "quality.ms_ssim.evaluated_fraction_of_total": ms_ssim_coverage[
                "evaluated_fraction_of_total"
            ],
            "quality.ms_ssim.evaluated_fraction_of_eligible": ms_ssim_coverage[
                "evaluated_fraction_of_eligible"
            ],
        }
        if ms_ssim_value is not None:
            report["ms_ssim"] = ms_ssim_value
            metrics["quality.ms_ssim"] = ms_ssim_value
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics=metrics,
            metadata={
                "num_images": int(count),
                "original_shapes": report["original_shapes"],
                "sample_pairing": pairing_mode,
                "data_range": data_range,
                "data_range_source": data_range_source,
                "psnr_cap_db": psnr_cap_db,
                "requested_psnr_cap_db": requested_psnr_cap_db,
                "minimum_safe_psnr_cap_db": minimum_safe_psnr_cap_db,
                "ms_ssim_coverage": ms_ssim_coverage,
            },
        )


class ImageDeliveryStatusOperation(Operation):
    """Bind a continuous reconstruction to its attempted-item denominator."""

    id = "metrics.image_delivery_status"
    name = "Image delivery and denominator status"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "on_decode_failure": {
                "type": "string",
                "enum": ["report_outage"],
                "default": "report_outage",
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        images, metadata = _load_images(ctx.require_input("images").path)
        if images.ndim != 4 or int(images.shape[0]) < 1:
            raise RuntimeError(
                "Image delivery status requires a non-empty image batch"
            )
        if not np.all(np.isfinite(images)):
            raise RuntimeError(
                "Image delivery status rejects non-finite reconstructions"
            )
        count = int(images.shape[0])
        inherited = metadata.get("source_item_outage")
        if inherited is None:
            outages = [0] * count
            status_source = "finite_reconstruction_present"
        elif (
            isinstance(inherited, list)
            and len(inherited) == count
            and all(value in {0, 1, False, True} for value in inherited)
        ):
            outages = [int(bool(value)) for value in inherited]
            status_source = "inherited_source_item_outage"
        else:
            raise RuntimeError(
                "Image delivery status received malformed source_item_outage"
            )
        policy = str(ctx.params.get("on_decode_failure") or "report_outage")
        failed = int(sum(outages))
        report = {
            "schema_version": 1,
            "metric_family": "image_delivery_status",
            "attempted_source_item_count": count,
            "failed_source_item_count": failed,
            "source_item_outage": outages,
            "source_item_success_rate": float(count - failed) / float(count),
            "source_item_outage_rate": float(failed) / float(count),
            "on_decode_failure": policy,
            "denominator_policy": "all attempted source items",
            "status_source": status_source,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics={
                "channel.source_item_count": count,
                "channel.failed_source_item_count": failed,
                "channel.source_item_success_rate": report[
                    "source_item_success_rate"
                ],
                "channel.outage_rate": report["source_item_outage_rate"],
            },
            metadata=report,
        )


def _image_preview(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    shapes: List[List[int]],
    sample_ids: List[str],
    data_range: float,
    count: int,
    preview_size: int,
) -> JsonDict:
    try:
        Image = importlib.import_module("PIL.Image")
    except Exception as exc:  # pragma: no cover - Pillow is part of the image extra
        raise RuntimeError(
            "Image previews require Pillow; install the image dependencies or set preview_count=0"
        ) from exc
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    samples = []
    for index in range(count):
        height, width, channels = shapes[index]
        ref = _preview_uint8(
            reference[index, :height, :width, :channels],
            data_range,
        )
        rec = _preview_uint8(
            reconstruction[index, :height, :width, :channels],
            data_range,
        )
        target_scale = min(1.0, float(preview_size) / float(max(height, width)))
        target_size = (
            max(1, int(round(width * target_scale))),
            max(1, int(round(height * target_scale))),
        )
        samples.append(
            {
                "index": int(index),
                "id": str(sample_ids[index]),
                "reference": _png_data_uri(Image, ref, target_size, resampling),
                "reconstruction": _png_data_uri(
                    Image, rec, target_size, resampling
                ),
            }
        )
    return {
        "schema_version": 1,
        "encoding": "data:image/png;base64",
        "maximum_dimension": int(preview_size),
        "samples": samples,
    }


def _preview_uint8(image: np.ndarray, data_range: float) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    scaled = np.asarray(image, dtype=np.float64)
    if data_range <= 1.0 + 1.0e-9:
        scaled = scaled * 255.0
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _png_data_uri(Image, image: np.ndarray, target_size, resampling) -> str:
    preview = Image.fromarray(image)
    if preview.size != target_size:
        preview = preview.resize(target_size, resample=resampling)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _load_images(path):
    with np.load(str(path), allow_pickle=False) as payload:
        images = payload["images"]
        metadata = {}
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Image metric artifact metadata_json",
                )
            )
    if images.ndim != 4 or images.shape[-1] != 3:
        raise RuntimeError("Expected images with shape [N,H,W,3], got %s" % (images.shape,))
    if images.dtype not in {np.dtype("uint8"), np.dtype("float32")}:
        raise RuntimeError(
            "Expected canonical uint8 or float32 image pixels, got dtype %s" % images.dtype
        )
    return images, metadata


def _single_original_shape(metadata: JsonDict, index: int):
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, (list, tuple)) and index < len(shapes):
        value = shapes[index]
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [1, int(value[1]), int(value[2]), int(value[3])]
    original = metadata.get("original_shape") or metadata.get("shape")
    if isinstance(original, (list, tuple)) and len(original) == 4:
        return [1, int(original[1]), int(original[2]), int(original[3])]
    return None


def _image_shapes(images: np.ndarray, metadata: JsonDict, label: str) -> List[List[int]]:
    shapes = metadata.get("original_shapes")
    if shapes is not None:
        if not isinstance(shapes, (list, tuple)) or len(shapes) != images.shape[0]:
            raise RuntimeError(
                "%s original_shapes must contain exactly %d entries" % (label, images.shape[0])
            )
    output = []
    for index in range(int(images.shape[0])):
        if shapes is not None:
            raw_shape = shapes[index]
            if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
                raise RuntimeError(
                    "%s original_shapes[%d] must be [1,H,W,C]" % (label, index)
                )
            try:
                declared = [int(value) for value in raw_shape]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "%s original_shapes[%d] must contain integers" % (label, index)
                ) from exc
        else:
            declared = _single_original_shape(metadata, index)
        if declared is None:
            shape = [int(images.shape[1]), int(images.shape[2]), int(images.shape[3])]
        else:
            if declared[0] not in {1, int(images.shape[0])}:
                raise RuntimeError(
                    "%s original shape %d has invalid batch dimension %d"
                    % (label, index, declared[0])
                )
            shape = [int(declared[1]), int(declared[2]), int(declared[3])]
        if (
            shape[0] <= 0
            or shape[1] <= 0
            or shape[2] != 3
            or shape[0] > images.shape[1]
            or shape[1] > images.shape[2]
        ):
            raise RuntimeError(
                "%s original shape %d is invalid for storage shape %s: %s"
                % (label, index, list(images.shape), shape)
            )
        output.append(shape)
    return output


def _metadata_ids(metadata: Mapping[str, Any], count: int, label: str) -> Optional[List[str]]:
    raw = None
    for key in ("sample_ids", "image_ids", "ids"):
        if key in metadata:
            raw = metadata.get(key)
            break
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("%s sample IDs must be a list" % label)
    values = [str(value) for value in raw]
    repeat_count = int(metadata.get("repeat_count") or 1)
    if len(values) != count and repeat_count > 1 and len(values) * repeat_count == count:
        values = values * repeat_count
    if len(values) != count or any(not value for value in values):
        raise RuntimeError(
            "%s sample IDs must contain exactly %d non-empty entries" % (label, count)
        )
    return values


def _paired_image_ids(
    reference_metadata: Mapping[str, Any],
    reconstruction_metadata: Mapping[str, Any],
    count: int,
) -> Tuple[List[str], str]:
    reference_ids = _metadata_ids(reference_metadata, count, "reference")
    reconstruction_ids = _metadata_ids(reconstruction_metadata, count, "reconstruction")
    if (reference_ids is None) != (reconstruction_ids is None):
        raise RuntimeError(
            "Image sample identity mismatch: both inputs must declare IDs when either input does"
        )
    if reference_ids is not None:
        if reference_ids != reconstruction_ids:
            raise RuntimeError(
                "Image sample ID/order mismatch: reference IDs %s, reconstruction IDs %s"
                % (reference_ids, reconstruction_ids)
            )
        return reference_ids, "declared_id_and_order"
    return ["index_%06d" % index for index in range(count)], "exact_positional_no_ids"


def _validate_image_values(
    values: np.ndarray, data_range: float, label: str, index: int
) -> None:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("%s image %d contains empty or non-finite pixels" % (label, index))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum < 0.0 or maximum > data_range:
        raise RuntimeError(
            "%s image %d has pixel range [%s, %s], outside declared [0, %s]"
            % (label, index, minimum, maximum, data_range)
        )


def _resolve_data_range(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    declared: float,
) -> Tuple[float, str]:
    if not math.isfinite(declared) or declared < 0.0:
        raise RuntimeError("data_range must be finite and non-negative")
    if reference.dtype != reconstruction.dtype:
        raise RuntimeError(
            "Image pixel-domain mismatch: reference dtype %s, reconstruction dtype %s"
            % (reference.dtype, reconstruction.dtype)
        )
    if declared > 0.0:
        return declared, "explicit_parameter"
    if reference.dtype == np.dtype("uint8"):
        return 255.0, "canonical_uint8"
    if reference.dtype == np.dtype("float32"):
        return 1.0, "canonical_float32"
    raise RuntimeError("Cannot infer data_range for image dtype %s" % reference.dtype)


def _psnr(mse: float, data_range: float, cap_db: float) -> Tuple[float, bool]:
    if mse == 0.0:
        return cap_db, True
    return float(20.0 * np.log10(data_range / np.sqrt(mse))), False


def _minimum_safe_exact_psnr_cap_db(
    dtype: np.dtype,
    shapes: List[List[int]],
    data_range: float,
) -> float:
    """Return a finite exact-match sentinel above every attainable finite PSNR.

    Images are compared after conversion to float64, but their source domain remains
    uint8 or float32. The smallest non-zero source-domain step and the largest
    evaluated image determine a conservative upper bound for finite per-image PSNR.
    """

    if dtype == np.dtype("uint8"):
        minimum_step = 1.0
    elif dtype == np.dtype("float32"):
        minimum_step = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    else:  # Guarded by _load_images; keep this helper fail-closed when called directly.
        raise RuntimeError("Cannot derive a safe exact PSNR cap for dtype %s" % dtype)
    maximum_element_count = max(
        int(height) * int(width) * int(channels)
        for height, width, channels in shapes
    )
    finite_ceiling_db = (
        20.0 * math.log10(data_range)
        - 20.0 * math.log10(minimum_step)
        + 10.0 * math.log10(float(maximum_element_count))
    )
    return float(finite_ceiling_db) + max(1e-9, abs(float(finite_ceiling_db)) * 1e-12)


def _try_ms_ssim(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    shapes: List[List[int]],
    data_range: float,
) -> Optional[List[Optional[float]]]:
    torch = _import_optional_ms_ssim_dependency("torch")
    if torch is None:
        return None
    pytorch_msssim = _import_optional_ms_ssim_dependency("pytorch_msssim")
    if pytorch_msssim is None:
        return None
    ms_ssim = getattr(pytorch_msssim, "ms_ssim", None)
    if not callable(ms_ssim):
        raise RuntimeError(
            "Optional MS-SSIM dependency pytorch_msssim imported successfully "
            "but does not expose a callable ms_ssim"
        )

    values: List[Optional[float]] = []
    for index, (height, width, channels) in enumerate(shapes):
        ref = reference[index, :height, :width, :channels].astype(np.float32)
        rec = reconstruction[index, :height, :width, :channels].astype(np.float32)
        if min(ref.shape[0], ref.shape[1]) < 161:
            values.append(None)
            continue
        ref_t = torch.from_numpy(np.ascontiguousarray(ref.transpose(2, 0, 1))).unsqueeze(0) / data_range
        rec_t = torch.from_numpy(np.ascontiguousarray(rec.transpose(2, 0, 1))).unsqueeze(0) / data_range
        try:
            values.append(float(ms_ssim(rec_t.float(), ref_t.float(), data_range=1.0).item()))
        except Exception as exc:
            raise RuntimeError("MS-SSIM failed for image %d: %s" % (index, exc)) from exc
    return values


def _import_optional_ms_ssim_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        missing_name = exc.name or "an unidentified transitive module"
        raise RuntimeError(
            "Optional MS-SSIM dependency %s is installed but failed to import "
            "because %s is missing: %s"
            % (module_name, missing_name, exc)
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Optional MS-SSIM dependency %s is installed but failed to import: %s"
            % (module_name, exc)
        ) from exc
