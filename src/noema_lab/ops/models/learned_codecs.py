from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core import dataplane
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema
from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.structured_input import (
    decode_strict_json_object,
    decode_strict_yaml_or_json,
)
from noema_lab.ops.models.catalog import (
    compressai_model_config,
    compressai_model_ids,
    diffusers_load_kwargs,
    default_compressai_metric,
    default_compressai_model,
    default_compressai_quality,
    default_diffusers_model,
    load_model_catalog,
)
from noema_lab.ops.models.codec_wire import (
    PAYLOAD_FORMAT as COMPACT_CODEC_PAYLOAD_FORMAT,
    dumps as compact_codec_payload_dumps,
    loads as compact_codec_payload_loads,
)
from noema_lab.ops.models.onnx_evidence import onnxruntime_native_evidence
from noema_lab.ops.models.onnx_cpp_runtime import (
    CppOnnxSession,
    cpp_runtime_available,
    cpp_runtime_evidence,
)
from noema_lab.ops.models.safe_payload import (
    PAYLOAD_FORMAT as SAFE_PAYLOAD_FORMAT,
    dumps as safe_payload_dumps,
    loads as safe_payload_loads,
)
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata, timed_call

JsonDict = Dict[str, Any]

JPEG_ENTROPY_INFO: JsonDict = {
    "algorithm": "JPEG Huffman coding",
    "coder": "Huffman",
    "implementation": "Pillow JPEG backend",
    "language": "C/Python",
    "note": "Baseline JPEG comparisons normally use non-progressive JPEG without optimized Huffman-table search.",
}

COMPRESSAI_ENTROPY_INFO: JsonDict = {
    "algorithm": "range coding / ANS backend",
    "coder": "CompressAI entropy bottleneck + GaussianConditional",
    "implementation": "CompressAI entropy backend",
    "language": "C++ extension + Python/PyTorch",
    "note": "The neural transforms may run in PyTorch, AOT Inductor, ONNX Runtime, or OpenVINO; entropy coding remains the CompressAI bitstream backend.",
}

CODEC_WIRE_FORMATS = {
    "safe_json_base64": SAFE_PAYLOAD_FORMAT,
    "compact_binary_v1": COMPACT_CODEC_PAYLOAD_FORMAT,
}


def _entropy_metadata(info: JsonDict) -> JsonDict:
    return {
        "entropy_coder": info.get("coder", ""),
        "entropy_algorithm": info.get("algorithm", ""),
        "entropy_implementation": info.get("implementation", ""),
        "entropy_language": info.get("language", ""),
        "entropy_note": info.get("note", ""),
    }


def _compressai_payload_decode_error(stage: str, exc: Exception) -> str:
    return (
        f"{stage} failed because the received entropy-coded payload is not a valid CompressAI bitstream. "
        "This usually means channel bit errors reached the payload decoder, so even one flipped bit can corrupt "
        "the serialized entropy stream. Use a bit-perfect channel, higher SNR, stronger/protected channel coding, "
        "or set the payload decoder failure policy to zeros only when you want an explicit outage fallback. "
        f"Decoder detail: {exc}"
    )


def _independent_item_payload_bits(
    payloads: List[JsonDict],
    backend: str,
    payload_format: str = SAFE_PAYLOAD_FORMAT,
) -> Tuple[np.ndarray, List[int], List[int], str]:
    """Serialize independently decodable source-item payloads.

    Keeping each image in its own byte stream is essential for per-item packet
    ownership: corruption of one item's entropy payload must not invalidate the
    deserialization of every other image in the executor batch.
    """

    bit_rows: List[np.ndarray] = []
    byte_counts: List[int] = []
    bit_counts: List[int] = []
    selected_backend = "python_numpy"
    for payload in payloads:
        if payload_format == SAFE_PAYLOAD_FORMAT:
            raw = safe_payload_dumps(payload)
        elif payload_format == COMPACT_CODEC_PAYLOAD_FORMAT:
            raw = compact_codec_payload_dumps(payload)
        else:
            raise ValueError("Unsupported codec payload format: %s" % payload_format)
        row, selected_backend = _bytes_to_bits(raw, backend)
        bit_rows.append(row)
        byte_counts.append(len(raw))
        bit_counts.append(int(row.size))
    bits = (
        np.concatenate(bit_rows).astype(np.uint8, copy=False)
        if bit_rows
        else np.zeros((0,), dtype=np.uint8)
    )
    return bits, bit_counts, byte_counts, selected_backend


def _codec_payload_format(value: Any) -> str:
    key = str(value or "safe_json_base64")
    if key in CODEC_WIRE_FORMATS:
        return CODEC_WIRE_FORMATS[key]
    if key in CODEC_WIRE_FORMATS.values():
        return key
    raise ValueError("Unsupported codec wire format: %s" % key)


def _codec_payload_loads(raw: bytes, payload_format: Any) -> JsonDict:
    selected = _codec_payload_format(payload_format or SAFE_PAYLOAD_FORMAT)
    if selected == SAFE_PAYLOAD_FORMAT:
        return safe_payload_loads(raw)
    return compact_codec_payload_loads(raw)


def _binary_payload_byte_count(value: Any) -> int:
    """Count native codec bytes without counting JSON/base64/container bytes."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_binary_payload_byte_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_binary_payload_byte_count(item) for item in value.values())
    return 0


def _native_codec_accounting(
    native_item_byte_counts: List[int],
    serialized_byte_count: int,
    serialized_item_byte_counts: Optional[List[int]] = None,
    serialized_payload_format: str = SAFE_PAYLOAD_FORMAT,
) -> JsonDict:
    native_counts = [int(value) for value in native_item_byte_counts]
    if any(value < 0 for value in native_counts):
        raise RuntimeError("Native codec byte counts must be non-negative")
    native_total = int(sum(native_counts))
    serialized_total = int(serialized_byte_count)
    if serialized_total < native_total:
        raise RuntimeError(
            "Serialized codec payload cannot be smaller than its native bitstreams"
        )
    payload: JsonDict = {
        "native_codec_byte_count": native_total,
        "native_codec_bit_count": native_total * 8,
        "source_item_native_codec_byte_counts": native_counts,
        "source_item_native_codec_bit_counts": [value * 8 for value in native_counts],
        "serialized_payload_byte_count": serialized_total,
        "serialized_payload_bit_count": serialized_total * 8,
        "safe_serialization_wrapper_byte_count": serialized_total - native_total,
        "safe_serialization_wrapper_bit_count": (serialized_total - native_total) * 8,
        "wire_protocol_overhead_byte_count": serialized_total - native_total,
        "wire_protocol_overhead_bit_count": (serialized_total - native_total) * 8,
        "native_codec_rate_boundary": "codec_emitted_bytes_before_noema_serialization",
        "serialized_payload_rate_boundary": (
            "noema_safe_data_json_base64_bytes"
            if serialized_payload_format == SAFE_PAYLOAD_FORMAT
            else "noema_compact_codec_wire_bytes"
        ),
        "serialized_payload_format": serialized_payload_format,
    }
    if serialized_item_byte_counts is not None:
        wrapper_counts = [int(value) for value in serialized_item_byte_counts]
        if len(wrapper_counts) != len(native_counts):
            raise RuntimeError(
                "Serialized and native per-item codec counts have different lengths"
            )
        if any(
            serialized < native
            for serialized, native in zip(wrapper_counts, native_counts)
        ):
            raise RuntimeError(
                "A serialized source-item payload is smaller than its native codec bytes"
            )
        payload.update(
            {
                "source_item_serialized_payload_byte_counts": wrapper_counts,
                "source_item_serialized_payload_bit_counts": [
                    value * 8 for value in wrapper_counts
                ],
                "source_item_safe_serialization_wrapper_byte_counts": [
                    serialized - native
                    for serialized, native in zip(wrapper_counts, native_counts)
                ],
            }
        )
    return payload


def _native_codec_metrics(accounting: Mapping[str, Any], pixel_count: int) -> JsonDict:
    metrics: JsonDict = {
        "codec.native_bit_count": int(accounting["native_codec_bit_count"]),
        "codec.native_bytes": int(accounting["native_codec_byte_count"]),
        "codec.serialized_payload_bit_count": int(
            accounting["serialized_payload_bit_count"]
        ),
        "codec.serialized_payload_bytes": int(
            accounting["serialized_payload_byte_count"]
        ),
        "codec.safe_serialization_wrapper_bit_count": int(
            accounting["safe_serialization_wrapper_bit_count"]
        ),
        "codec.safe_serialization_wrapper_bytes": int(
            accounting["safe_serialization_wrapper_byte_count"]
        ),
        "codec.wire_protocol_overhead_bit_count": int(
            accounting["wire_protocol_overhead_bit_count"]
        ),
        "codec.wire_protocol_overhead_bytes": int(
            accounting["wire_protocol_overhead_byte_count"]
        ),
    }
    if pixel_count:
        metrics.update(
            {
                "rate.native_codec_bpp": float(
                    accounting["native_codec_bit_count"]
                )
                / float(pixel_count),
                "rate.serialized_payload_bpp": float(
                    accounting["serialized_payload_bit_count"]
                )
                / float(pixel_count),
            }
        )
    return metrics


def _independent_item_payload_bytes(
    bits: np.ndarray,
    metadata: JsonDict,
    backend: str,
) -> Optional[Tuple[List[bytes], str]]:
    raw_counts = metadata.get("source_item_payload_bit_counts")
    if not isinstance(raw_counts, list):
        return None
    counts = [int(item) for item in raw_counts]
    raw_outages = metadata.get("source_item_outage")
    outages = (
        [bool(item) for item in raw_outages]
        if isinstance(raw_outages, list) and len(raw_outages) == len(counts)
        else [False] * len(counts)
    )
    if any(
        item < 0 or item % 8 or (item == 0 and not outages[index])
        for index, item in enumerate(counts)
    ):
        raise RuntimeError(
            "Independent image payload lengths must be whole bytes; zero is valid only for an in-band-declared outage"
        )
    if sum(counts) != int(bits.size):
        raise RuntimeError(
            "Independent image payload lengths sum to %d bits, received %d"
            % (sum(counts), int(bits.size))
        )
    rows: List[bytes] = []
    offset = 0
    selected_backend = "python_numpy"
    for count in counts:
        row, selected_backend = _bits_to_bytes(
            bits[offset : offset + count], count // 8, backend
        )
        rows.append(row)
        offset += count
    return rows, selected_backend


def _source_item_ids(metadata: JsonDict, count: int) -> List[str]:
    raw = metadata.get("source_item_ids") or metadata.get("image_ids")
    values = [str(item) for item in raw] if isinstance(raw, list) else []
    if values and len(values) != count:
        repeat_count = max(1, int(metadata.get("repeat_count") or 1))
        if len(values) * repeat_count == count:
            values = values * repeat_count
    if len(values) != count:
        values = ["source_item_%06d" % index for index in range(count)]
    return values


def _source_pixel_count(metadata: JsonDict) -> int:
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, list):
        total = sum(
            int(shape[0] or 1) * int(shape[1]) * int(shape[2])
            for shape in shapes
            if isinstance(shape, (list, tuple)) and len(shape) >= 4
        )
        if total > 0:
            return int(total)
    shape = metadata.get("original_shape") or metadata.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        return int(shape[0] or 1) * int(shape[1]) * int(shape[2])
    return 0


def _source_item_outages(metadata: JsonDict, count: int) -> List[bool]:
    raw = metadata.get("source_item_outage")
    if not isinstance(raw, list):
        return [False] * count
    if len(raw) != count:
        raise RuntimeError(
            "source_item_outage has %d entries for %d source items" % (len(raw), count)
        )
    return [bool(item) for item in raw]


def _fallback_decode_image(metadata: JsonDict, index: int, policy: str) -> np.ndarray:
    shape = _single_original_shape(metadata, index)
    value = 128 if policy == "gray_image" else 0
    return np.full((int(shape[1]), int(shape[2]), int(shape[3])), value, dtype=np.uint8)


def _merged_availability(extra: str, modules: list, label: str, native: JsonDict | None = None) -> JsonDict:
    base = _optional_dependency_availability(extra, modules, label)
    missing = list(base.get("missing") or [])
    if native and native.get("available") is False:
        missing.extend(str(item) for item in native.get("missing") or [])
    if missing:
        reason = base.get("reason") if base.get("available") is False else ""
        native_reason = str((native or {}).get("reason") or "")
        return {
            "available": False,
            "extra": extra,
            "missing": sorted(set(missing)),
            "reason": " ".join(item for item in (reason, native_reason) if item).strip(),
        }
    return {"available": True, "extra": extra, "missing": []}


def _onnx_base_evidence_for_runtime(runtime: str, ort) -> JsonDict:
    if runtime == "onnxruntime_cpp":
        evidence = onnxruntime_native_evidence(ort)
        evidence.update(
            {
                "runtime": "onnxruntime_cpp",
                "runtime_engine": "onnxruntime_cxx",
                "runtime_execution_language": "C++",
                "runtime_api": "onnxruntime_c_api",
                "runtime_api_binding": "pybind11_native_extension",
                "native_inference_engine": True,
                "runtime_note": "Inference is executed through a Noema pybind11 adapter calling the ONNX Runtime C API directly.",
            }
        )
        return evidence
    return onnxruntime_native_evidence(ort)


class JpegEncodeOperation(Operation):
    id = "model.jpeg_encode"
    name = "JPEG image encoder to payload bits"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = JPEG_ENTROPY_INFO
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Classical JPEG quantization and entropy coding are non-differentiable in Noema benchmark execution.",
    }
    params_schema = object_schema(
        {
            "quality": {
                "type": "integer",
                "default": 75,
                "minimum": 1,
                "maximum": 95,
                "description": "JPEG quality factor. Higher values keep more detail and use more bits.",
            },
            "subsampling": {
                "type": "string",
                "default": "420",
                "enum": ["keep", "444", "422", "420"],
                "description": "Chroma subsampling. 444 preserves chroma best; 420 is the common higher-compression setting.",
            },
            "optimize": {
                "type": "boolean",
                "default": False,
                "description": "Advanced. Extra entropy-coding pass for smaller files at the same quantization; leave off for a plain baseline JPEG comparison unless you report optimized JPEG.",
            },
            "progressive": {
                "type": "boolean",
                "default": False,
                "description": "Advanced. Store the JPEG in progressive scan order for streaming/preview; leave off for the usual baseline sequential JPEG comparison unless explicitly studying progressive JPEG.",
            },
            "wire_format": {
                "type": "string",
                "default": "safe_json_base64",
                "enum": ["safe_json_base64", "compact_binary_v1"],
                "description": "Artifact-safe JSON/base64 remains the compatibility default; compact_binary_v1 is the bounded communication wire format for transmitted-resource studies.",
            },
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _pillow_availability()
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        Image = _require_pillow()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        quality = int(ctx.params.get("quality", 75))
        subsampling = str(ctx.params.get("subsampling", "420"))
        optimize = bool(ctx.params.get("optimize", False))
        progressive = bool(ctx.params.get("progressive", False))
        payload_format = _codec_payload_format(ctx.params.get("wire_format"))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        save_kwargs = {
            "format": "JPEG",
            "quality": max(1, min(quality, 95)),
            "optimize": optimize,
            "progressive": progressive,
        }
        subsampling_value = _jpeg_subsampling_value(subsampling)
        if subsampling_value is not None:
            save_kwargs["subsampling"] = subsampling_value

        entries = []
        timing_records = []
        _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
        for index in range(int(images.shape[0])):
            image = _image_for_index(images, input_metadata, index)
            total_start = time.perf_counter()
            buffer = io.BytesIO()
            timed_call(
                timing_records,
                index,
                "encoder.payload_encode",
                lambda image=image, buffer=buffer: Image.fromarray(image).save(buffer, **save_kwargs),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            entries.append(
                {
                    "data": buffer.getvalue(),
                    "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                }
            )
            _report_example_progress(ctx, index + 1, int(images.shape[0]), "encoding")

        item_payloads = [
            {
                "payload_version": 2,
                "codec": "jpeg",
                "entry": entry,
                "quality": save_kwargs["quality"],
                "subsampling": subsampling,
                "optimize": optimize,
                "progressive": progressive,
            }
            for entry in entries
        ]
        bits, item_bit_counts, item_byte_counts, byte_backend = (
            _independent_item_payload_bits(
                item_payloads, data_backend, payload_format
            )
        )
        byte_count = int(sum(item_byte_counts))
        native_accounting = _native_codec_accounting(
            [len(entry["data"]) for entry in entries],
            byte_count,
            item_byte_counts,
            payload_format,
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "jpeg",
                "quality": save_kwargs["quality"],
                "subsampling": subsampling,
                "optimize": optimize,
                "progressive": progressive,
                "byte_count": byte_count,
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_bit_count": int(bits.size),
                "payload_format": payload_format,
                "payload_framing": "independent_source_items_v1",
                "source_item_count": int(len(item_payloads)),
                "source_item_ids": _source_item_ids(input_metadata, len(item_payloads)),
                "source_item_payload_bit_counts": item_bit_counts,
                "source_item_payload_byte_counts": item_byte_counts,
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(JPEG_ENTROPY_INFO),
            }
        )
        if pixel_count:
            metadata["rate_payload_bpp"] = float(bits.size) / float(pixel_count)
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": byte_count,
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
                **(
                    {"rate.payload_bpp": float(bits.size) / float(pixel_count)}
                    if pixel_count
                    else {}
                ),
            },
            metadata={
                "codec": "jpeg",
                "quality": save_kwargs["quality"],
                "subsampling": subsampling,
                **_entropy_metadata(JPEG_ENTROPY_INFO),
                "codec_timing": codec_timing_metadata("encoder", timing_records, runner="local_python"),
            },
        )


class JpegCapacityOracleOperation(Operation):
    """Ideal separation reference with average-SNR-adaptive JPEG source rate.

    This operation implements the digital upper-bound protocol used by the
    original DeepJSCC comparison: reserve a fixed number of complex channel
    uses per source pixel and choose the highest JPEG quality whose native
    bitstream fits the complex-AWGN capacity at the selected average SNR. In
    slow-Rayleigh mode the same fixed source rate is used for every fading
    realization; delivery succeeds only when the instantaneous capacity
    supports that rate. An outage uses the configured fallback.
    """

    id = "channel.jpeg_capacity_oracle"
    name = "JPEG + ideal-capacity separation"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = JPEG_ENTROPY_INFO
    differentiability = {
        "framework": "none",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "This is a theoretical separation reference combining classical "
            "JPEG with an ideal capacity-achieving channel code."
        ),
    }
    params_schema = object_schema(
        {
            "channel_model": {
                "type": "string",
                "default": "awgn",
                "enum": ["awgn", "slow_rayleigh"],
                "description": (
                    "awgn uses the average-SNR capacity directly. "
                    "slow_rayleigh fixes the source rate from average SNR and "
                    "tests it against one unknown block-fading gain per image."
                ),
            },
            "snr_db": {"type": "number", "default": 12.0},
            "channel_uses_per_pixel": {
                "type": "number",
                "default": 0.5,
                "exclusiveMinimum": 0.0,
                "description": (
                    "Fixed complex channel uses available per spatial source pixel."
                ),
            },
            "minimum_quality": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 95,
            },
            "maximum_quality": {
                "type": "integer",
                "default": 95,
                "minimum": 1,
                "maximum": 95,
            },
            "subsampling": {
                "type": "string",
                "default": "420",
                "enum": ["keep", "444", "422", "420"],
            },
            "optimize": {"type": "boolean", "default": False},
            "progressive": {"type": "boolean", "default": False},
            "on_outage": {
                "type": "string",
                "default": "gray_image",
                "enum": ["gray_image", "zeros", "image_channel_mean"],
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional explicit channel seed. Use the same value on a "
                    "paired learned recipe to reproduce per-image fading gains."
                ),
            },
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _pillow_availability()
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        Image = _require_pillow()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        channel_model = str(ctx.params.get("channel_model") or "awgn")
        snr_db = float(ctx.params.get("snr_db", 12.0))
        uses_per_pixel = float(ctx.params.get("channel_uses_per_pixel", 0.5))
        minimum_quality = int(ctx.params.get("minimum_quality", 1))
        maximum_quality = int(ctx.params.get("maximum_quality", 95))
        if not math.isfinite(snr_db):
            raise ValueError("JPEG capacity oracle requires a finite snr_db")
        if not math.isfinite(uses_per_pixel) or uses_per_pixel <= 0.0:
            raise ValueError(
                "JPEG capacity oracle requires channel_uses_per_pixel > 0"
            )
        if minimum_quality < 1 or maximum_quality > 95:
            raise ValueError("JPEG quality search must remain within [1, 95]")
        if minimum_quality > maximum_quality:
            raise ValueError("minimum_quality cannot exceed maximum_quality")

        subsampling = str(ctx.params.get("subsampling", "420"))
        optimize = bool(ctx.params.get("optimize", False))
        progressive = bool(ctx.params.get("progressive", False))
        outage_policy = str(ctx.params.get("on_outage") or "gray_image")
        if channel_model not in {"awgn", "slow_rayleigh"}:
            raise ValueError("Unknown JPEG capacity channel_model: %s" % channel_model)
        if outage_policy not in {"gray_image", "zeros", "image_channel_mean"}:
            raise ValueError("Unknown JPEG capacity outage policy: %s" % outage_policy)
        subsampling_value = _jpeg_subsampling_value(subsampling)
        snr_linear = 10.0 ** (snr_db / 10.0)
        average_capacity_per_use = math.log2(1.0 + snr_linear)
        source_item_ids = _source_item_ids(input_metadata, int(images.shape[0]))
        channel_seed = ctx.seed("wireless_channel")

        decoded: List[np.ndarray] = []
        decoded_shapes: List[List[int]] = []
        selected_qualities: List[int] = []
        native_byte_counts: List[int] = []
        item_pixel_counts: List[int] = []
        item_channel_uses: List[int] = []
        item_capacity_bits: List[float] = []
        item_instantaneous_capacity_bits: List[float] = []
        item_channel_gain_real: List[float] = []
        item_channel_gain_imag: List[float] = []
        item_channel_gain_magnitude: List[float] = []
        item_channel_gain_power: List[float] = []
        item_success: List[bool] = []
        timing_records = []
        _report_example_progress(ctx, 0, int(images.shape[0]), "capacity matching")
        for index in range(int(images.shape[0])):
            image = _image_for_index(images, input_metadata, index)
            height, width, channels = [int(value) for value in image.shape]
            pixel_count = height * width
            channel_uses = int(math.floor(uses_per_pixel * pixel_count + 1.0e-12))
            if channel_uses < 1:
                raise ValueError(
                    "JPEG capacity oracle budget yields no channel uses for source item %d"
                    % index
                )
            capacity_bits = float(channel_uses) * average_capacity_per_use
            if channel_model == "slow_rayleigh":
                item_seed = derive_seed(
                    int(channel_seed),
                    "wireless.channel.source_item",
                    str(source_item_ids[index]),
                    "occurrence_0",
                )
                item_rng = np.random.RandomState(item_seed)
                # wireless.channel draws the complex noise vector before the
                # source-item gain. Consume the paired draws here so both
                # operations use exactly the same fading realization.
                item_rng.randn(channel_uses)
                item_rng.randn(channel_uses)
                fading = complex(
                    float(item_rng.randn()) / math.sqrt(2.0),
                    float(item_rng.randn()) / math.sqrt(2.0),
                )
            else:
                fading = complex(1.0, 0.0)
            gain_power = float(abs(fading) ** 2)
            instantaneous_capacity_per_use = math.log2(
                1.0 + gain_power * snr_linear
            )
            instantaneous_capacity_bits = (
                float(channel_uses) * instantaneous_capacity_per_use
            )
            total_start = time.perf_counter()
            selected_quality = 0
            selected_payload = b""
            for quality in range(maximum_quality, minimum_quality - 1, -1):
                buffer = io.BytesIO()
                save_kwargs: JsonDict = {
                    "format": "JPEG",
                    "quality": quality,
                    "optimize": optimize,
                    "progressive": progressive,
                }
                if subsampling_value is not None:
                    save_kwargs["subsampling"] = subsampling_value
                Image.fromarray(image).save(buffer, **save_kwargs)
                candidate = buffer.getvalue()
                if float(len(candidate) * 8) <= capacity_bits + 1.0e-9:
                    selected_quality = quality
                    selected_payload = candidate
                    break

            success = bool(
                selected_quality
                and float(len(selected_payload) * 8)
                <= instantaneous_capacity_bits + 1.0e-9
            )
            if success:
                reconstruction = np.asarray(
                    Image.open(io.BytesIO(selected_payload)).convert("RGB"),
                    dtype=np.uint8,
                )
            else:
                if outage_policy == "image_channel_mean":
                    channel_means = np.mean(
                        image.astype(np.float64), axis=(0, 1), keepdims=True
                    )
                    reconstruction = np.broadcast_to(
                        np.rint(channel_means).astype(np.uint8),
                        (height, width, channels),
                    ).copy()
                else:
                    fallback_value = (
                        128 if outage_policy == "gray_image" else 0
                    )
                    reconstruction = np.full(
                        (height, width, channels),
                        fallback_value,
                        dtype=np.uint8,
                    )
            append_measurement(
                timing_records,
                index,
                "capacity_oracle.total",
                time.perf_counter() - total_start,
                image_shape=[height, width, channels],
            )
            decoded.append(reconstruction)
            decoded_shapes.append([1, height, width, channels])
            selected_qualities.append(selected_quality)
            native_byte_counts.append(len(selected_payload))
            item_pixel_counts.append(pixel_count)
            item_channel_uses.append(channel_uses)
            item_capacity_bits.append(capacity_bits)
            item_instantaneous_capacity_bits.append(
                instantaneous_capacity_bits
            )
            item_channel_gain_real.append(float(fading.real))
            item_channel_gain_imag.append(float(fading.imag))
            item_channel_gain_magnitude.append(float(abs(fading)))
            item_channel_gain_power.append(gain_power)
            item_success.append(success)
            _report_example_progress(
                ctx, index + 1, int(images.shape[0]), "capacity matching"
            )

        output_images, stacked_shapes = _stack_image_list(decoded)
        if stacked_shapes:
            decoded_shapes = stacked_shapes
        total_pixels = int(sum(item_pixel_counts))
        total_channel_uses = int(sum(item_channel_uses))
        total_capacity_bits = float(sum(item_capacity_bits))
        total_native_bits = int(sum(native_byte_counts) * 8)
        source_item_uses_per_pixel = [
            float(use_count) / float(pixel_count)
            for use_count, pixel_count in zip(
                item_channel_uses, item_pixel_counts
            )
        ]
        success_rate = float(sum(item_success)) / float(len(item_success))
        quality_mean = float(np.mean(selected_qualities))
        output_metadata = dict(input_metadata)
        output_metadata.update(
            {
                "source": "jpeg_capacity_oracle",
                "shape": list(output_images.shape),
                "dtype": str(output_images.dtype),
                "original_shapes": decoded_shapes,
                "codec": "jpeg",
                "subsampling": subsampling,
                "optimize": optimize,
                "progressive": progressive,
                "channel": (
                    "flat_rayleigh"
                    if channel_model == "slow_rayleigh"
                    else "awgn"
                ),
                "channel_model": channel_model,
                "channel_mode": "capacity_oracle_%s" % channel_model,
                "protected_digital_baseline": "jpeg_capacity_oracle",
                "protected_digital_note": (
                    "JPEG quality is selected from the capacity at average SNR. "
                    "The slow-Rayleigh reference declares an outage when the "
                    "instantaneous block-fading capacity cannot support that "
                    "fixed rate. Successful packets are delivered perfectly."
                ),
                "snr_db": snr_db,
                "snr_linear": snr_linear,
                "capacity_bits_per_channel_use": average_capacity_per_use,
                "channel_use_count": total_channel_uses,
                "channel_uses_per_pixel": (
                    float(total_channel_uses) / float(total_pixels)
                ),
                "capacity_bits": total_capacity_bits,
                "capacity_bpp": total_capacity_bits / float(total_pixels),
                "native_codec_byte_count": int(sum(native_byte_counts)),
                "native_codec_bit_count": total_native_bits,
                "rate_native_codec_bpp": (
                    float(total_native_bits) / float(total_pixels)
                ),
                "source_item_count": len(item_success),
                "source_item_ids": source_item_ids,
                "source_item_pixel_counts": item_pixel_counts,
                "source_item_channel_use_counts": item_channel_uses,
                "source_item_channel_uses_per_pixel": (
                    source_item_uses_per_pixel
                ),
                "max_source_item_channel_uses_per_pixel": max(
                    source_item_uses_per_pixel
                ),
                "source_item_capacity_bits": item_capacity_bits,
                "source_item_instantaneous_capacity_bits": (
                    item_instantaneous_capacity_bits
                ),
                "source_item_channel_gain_real": item_channel_gain_real,
                "source_item_channel_gain_imag": item_channel_gain_imag,
                "source_item_channel_gain_magnitude": (
                    item_channel_gain_magnitude
                ),
                "source_item_channel_gain_power": item_channel_gain_power,
                "source_item_native_codec_byte_counts": native_byte_counts,
                "source_item_native_codec_bit_counts": [
                    value * 8 for value in native_byte_counts
                ],
                "source_item_selected_jpeg_quality": selected_qualities,
                "source_item_outage": [
                    0 if success else 1 for success in item_success
                ],
                "source_item_success_rate": success_rate,
                "source_item_outage_rate": 1.0 - success_rate,
                "on_outage": outage_policy,
                **_entropy_metadata(JPEG_ENTROPY_INFO),
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(
            path,
            images=output_images,
            metadata_json=json.dumps(output_metadata),
        )
        return OperationResult(
            outputs={
                "images": artifact("image.batch.numpy", path, output_metadata)
            },
            metrics={
                "channel.snr_db": snr_db,
                "channel.capacity_bits": total_capacity_bits,
                "channel.capacity_bpp": (
                    total_capacity_bits / float(total_pixels)
                ),
                **(
                    {
                        "channel.awgn.capacity_bits": total_capacity_bits,
                        "channel.awgn.capacity_bpp": (
                            total_capacity_bits / float(total_pixels)
                        ),
                    }
                    if channel_model == "awgn"
                    else {}
                ),
                "channel.channel_use_count": total_channel_uses,
                "channel.uses_per_pixel": (
                    float(total_channel_uses) / float(total_pixels)
                ),
                "channel.max_source_item_uses_per_pixel": max(
                    source_item_uses_per_pixel
                ),
                "channel.packet_success_rate": success_rate,
                "channel.source_item_success_rate": success_rate,
                "channel.outage_rate": 1.0 - success_rate,
                "channel.gain_magnitude.minimum": min(
                    item_channel_gain_magnitude
                ),
                "channel.gain_magnitude.maximum": max(
                    item_channel_gain_magnitude
                ),
                "channel.gain_power.minimum": min(item_channel_gain_power),
                "channel.gain_power.maximum": max(item_channel_gain_power),
                "channel.capacity_oracle": 1,
                "channel.protected_digital.theoretical_reference": 1,
                "codec.bit_count": total_native_bits,
                "codec.bytes": int(sum(native_byte_counts)),
                "codec.native_bit_count": total_native_bits,
                "codec.native_bytes": int(sum(native_byte_counts)),
                "codec.jpeg.selected_quality_mean": quality_mean,
                "codec.jpeg.selected_quality_min": min(selected_qualities),
                "codec.jpeg.selected_quality_max": max(selected_qualities),
                "rate.native_codec_bpp": (
                    float(total_native_bits) / float(total_pixels)
                ),
            },
            metadata={
                "codec": "jpeg",
                "snr_db": snr_db,
                "channel_model": channel_model,
                "channel_seed": int(channel_seed),
                "channel_use_count": total_channel_uses,
                "source_item_success_rate": success_rate,
                "selected_jpeg_quality_mean": quality_mean,
                "codec_timing": codec_timing_metadata(
                    "capacity_oracle", timing_records, runner="local_python"
                ),
            },
        )


class JpegDecodeOperation(Operation):
    id = "model.jpeg_decode"
    name = "JPEG payload bits decoder to image batch"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = JPEG_ENTROPY_INFO
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Classical JPEG parsing, dequantization, and entropy decoding are not exposed as a differentiable training block.",
    }
    params_schema = object_schema(
        {
            "on_error": {
                "type": "string",
                "default": "fail",
                "enum": ["fail", "zeros", "gray_image", "erasure", "report_outage"],
                "description": "Use non-fail policies only when intentionally studying protected-link outages or corrupted entropy-coded payloads.",
            },
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _pillow_availability()
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        Image = _require_pillow()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        decoded_item_success: List[bool] = []
        try:
            independent = _independent_item_payload_bytes(bits, metadata, data_backend)
            if independent is not None:
                raw_rows, byte_backend = independent
                outages = _source_item_outages(metadata, len(raw_rows))
                timing_records = []
                decoded = []
                on_error = str(ctx.params.get("on_error", "fail"))
                _report_example_progress(ctx, 0, len(raw_rows), "decoding")
                for index, raw in enumerate(raw_rows):
                    try:
                        if outages[index]:
                            raise RuntimeError("source item failed CRC protection")
                        codec_payload = _codec_payload_loads(
                            raw, metadata.get("payload_format")
                        )
                        entry = codec_payload["entry"]
                        total_start = time.perf_counter()
                        image = timed_call(
                            timing_records,
                            index,
                            "decoder.payload_decode",
                            lambda entry=entry: Image.open(io.BytesIO(entry["data"])).convert("RGB"),
                            image_shape=entry.get("original_shape"),
                        )
                        array = np.asarray(image, dtype=np.uint8)
                        decoded_item_success.append(True)
                        append_measurement(
                            timing_records,
                            index,
                            "decoder.total",
                            time.perf_counter() - total_start,
                            image_shape=[int(array.shape[0]), int(array.shape[1]), int(array.shape[2])],
                        )
                    except Exception as item_exc:
                        if on_error not in {"zeros", "gray_image", "erasure", "report_outage"}:
                            raise RuntimeError(
                                "JPEG source item %d decode failed: %s" % (index, item_exc)
                            ) from item_exc
                        array = _fallback_decode_image(metadata, index, on_error)
                        decoded_item_success.append(False)
                    decoded.append(array)
                    _report_example_progress(ctx, index + 1, len(raw_rows), "decoding")
                images, decoded_shapes = _stack_image_list(decoded)
            else:
                raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
                codec_payload = _codec_payload_loads(
                    raw, metadata.get("payload_format")
                )
                entries = codec_payload.get("entries") or []
                timing_records = []
                _report_example_progress(ctx, 0, len(entries) or 1, "decoding")
                decoded = []
                for index, entry in enumerate(entries):
                    total_start = time.perf_counter()
                    image = timed_call(
                        timing_records,
                        index,
                        "decoder.payload_decode",
                        lambda entry=entry: Image.open(io.BytesIO(entry["data"])).convert("RGB"),
                        image_shape=entry.get("original_shape"),
                    )
                    array = np.asarray(image, dtype=np.uint8)
                    decoded.append(array)
                    decoded_item_success.append(True)
                    append_measurement(
                        timing_records,
                        index,
                        "decoder.total",
                        time.perf_counter() - total_start,
                        image_shape=[int(array.shape[0]), int(array.shape[1]), int(array.shape[2])],
                    )
                    _report_example_progress(ctx, index + 1, len(entries), "decoding")
                if not decoded:
                    raise RuntimeError("JPEG payload did not contain any encoded images")
                images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            on_error = str(ctx.params.get("on_error", "fail"))
            if on_error not in {"zeros", "gray_image", "erasure", "report_outage"}:
                raise RuntimeError("JPEG decode failed, likely due to corrupted payload bits: %s" % exc)
            images = _fallback_decode_images(metadata, on_error)
            decoded_item_success = [False] * int(images.shape[0])
            timing_records = []
            _report_example_progress(ctx, int(images.shape[0]) if images.ndim == 4 else 1, int(images.shape[0]) if images.ndim == 4 else 1, "decoding")

        output_metadata = dict(metadata)
        if "decoded_shapes" in locals():
            output_metadata["original_shapes"] = decoded_shapes
        output_metadata.update({
            "source": "jpeg_decode",
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "source_item_decode_success": [bool(item) for item in decoded_item_success],
            "byte_stream_data_plane_backend": byte_backend,
        })
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "codec_timing": codec_timing_metadata("decoder", timing_records, runner="local_python"),
            },
        )


class CompressAiEncodeOperation(Operation):
    id = "model.compressai_encode"
    name = "CompressAI model-zoo image encoder to payload bits"
    input_kinds = {"images": ["image.batch.numpy"]}
    input_metadata_requirements = {
        "images": {
            "any_of": ["original_shapes", "original_shape", "shape"],
        }
    }
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = COMPRESSAI_ENTROPY_INFO

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        model_ids = compressai_model_ids(self.catalog)
        self.params_schema = object_schema(
            {
                "model": {
                    "type": "string",
                    "default": default_compressai_model(self.catalog),
                    "enum": model_ids,
                },
                "quality": {
                    "type": "integer",
                    "default": default_compressai_quality(self.catalog),
                    "minimum": 1,
                    "maximum": 8,
                },
                "metric": {
                    "type": "string",
                    "default": default_compressai_metric(self.catalog),
                    "enum": ["mse", "ms-ssim"],
                },
                "pretrained": {"type": "boolean", "default": True},
                "expected_model_state_sha256": {
                    "type": "string",
                    "default": "",
                    "pattern": "^[0-9a-f]{64}$|^$",
                    "description": (
                        "Optional publication pin for the fully materialized "
                        "CompressAI state after model.update()."
                    ),
                },
                "device": {"type": "string", "default": "cpu"},
                "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
                "data_plane_backend": dataplane.backend_schema("auto"),
                "vbr_scale_index": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 7,
                    "description": "Only used by CompressAI *_vbr models. Selects the variable-rate scale index s, where larger values usually mean higher rate and quality.",
                },
                "vbr_stage": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 2,
                    "description": "Only used by CompressAI *_vbr models. Stage 2 uses the variable-rate path; stage 1 behaves like the base model path.",
                },
                "wire_format": {
                    "type": "string",
                    "default": "safe_json_base64",
                    "enum": ["safe_json_base64", "compact_binary_v1"],
                    "description": "Artifact-safe JSON/base64 remains the compatibility default; compact_binary_v1 is the bounded communication wire format for transmitted-resource studies.",
                },
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["model_catalog"] = {"compressai": compressai_model_ids(self.catalog)}
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        timing_records = []
        _report_setup_progress(ctx, "Preparing CompressAI PyTorch sender")
        setup_start = time.perf_counter()
        model_name = str(ctx.params.get("model", default_compressai_model(self.catalog)))
        quality = int(ctx.params.get("quality", default_compressai_quality(self.catalog)))
        metric = str(ctx.params.get("metric", default_compressai_metric(self.catalog)))
        _validate_compressai_choice(self.catalog, model_name, quality, metric)
        model = _compressai_model(zoo, model_name, quality, metric, bool(ctx.params.get("pretrained", True)))
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        payload_format = _codec_payload_format(ctx.params.get("wire_format"))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        vbr_params = _compressai_vbr_params(model, ctx.params)
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        model_state_sha256 = _torch_module_state_sha256(torch, model)
        expected_model_state_sha256 = str(
            ctx.params.get("expected_model_state_sha256") or ""
        )
        if (
            expected_model_state_sha256
            and expected_model_state_sha256 != model_state_sha256
        ):
            raise RuntimeError(
                "CompressAI model-state SHA-256 does not match "
                "expected_model_state_sha256"
            )
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        pad_to_multiple = int(ctx.params.get("pad_to_multiple", 64))
        _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
        entries = []
        with torch.no_grad():
            for index in range(int(images.shape[0])):
                _report_example_active_progress(ctx, index, int(images.shape[0]), "encoding")
                image = _image_for_index(images, input_metadata, index)
                total_start = time.perf_counter()
                tensor, padded_shape = timed_call(
                    timing_records,
                    index,
                    "encoder.preprocess",
                    lambda image=image: _images_to_torch(torch, image[None, ...], device, pad_to_multiple),
                    image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                )
                image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
                with _compressai_timing_scope(
                    model,
                    timing_records,
                    index,
                    "encoder",
                    image_shape,
                    model_stage="encoder.model",
                ):
                    compressed = _compressai_compress(model, tensor, vbr_params)
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                entries.append(
                    {
                        "strings": compressed["strings"],
                        "shape": compressed["shape"],
                        "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                        "padded_shape": list(padded_shape),
                    }
                )
                _report_example_progress(ctx, index + 1, int(images.shape[0]), "encoding")
        item_payloads = [
            {
                "codec": "compressai",
                "entry": entry,
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "pretrained": bool(ctx.params.get("pretrained", True)),
                "model_state_sha256": model_state_sha256,
                "vbr": vbr_params,
                "payload_version": 3,
            }
            for entry in entries
        ]
        bits, item_bit_counts, item_byte_counts, byte_backend = (
            _independent_item_payload_bits(
                item_payloads, data_backend, payload_format
            )
        )
        byte_count = int(sum(item_byte_counts))
        native_accounting = _native_codec_accounting(
            [_binary_payload_byte_count(entry["strings"]) for entry in entries],
            byte_count,
            item_byte_counts,
            payload_format,
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "model_state_sha256": model_state_sha256,
                "vbr_scale_index": vbr_params.get("scale_index"),
                "vbr_stage": vbr_params.get("stage"),
                "byte_count": byte_count,
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_format": payload_format,
                "payload_bit_count": int(bits.size),
                "payload_framing": "independent_source_items_v1",
                "source_item_count": int(len(item_payloads)),
                "source_item_ids": _source_item_ids(input_metadata, len(item_payloads)),
                "source_item_payload_bit_counts": item_bit_counts,
                "source_item_payload_byte_counts": item_byte_counts,
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "entropy_stage": "compressai_native_compress",
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(COMPRESSAI_ENTROPY_INFO),
            }
        )
        if pixel_count:
            metadata["rate_payload_bpp"] = float(bits.size) / float(pixel_count)
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": byte_count,
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
                **(
                    {"rate.payload_bpp": float(bits.size) / float(pixel_count)}
                    if pixel_count
                    else {}
                ),
            },
            metadata={
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "model_state_sha256": model_state_sha256,
                "vbr_scale_index": vbr_params.get("scale_index"),
                "vbr_stage": vbr_params.get("stage"),
                **_entropy_metadata(COMPRESSAI_ENTROPY_INFO),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "encoder.model": "Combined CompressAI neural transforms observed inside model.compress.",
                        "encoder.payload_model_inference": "CompressAI probability-index preparation observed inside model.compress.",
                        "encoder.symbol_encode": "Entropy coder calls observed inside CompressAI model.compress.",
                    },
                ),
            },
        )


class CompressAiDecodeOperation(Operation):
    id = "model.compressai_decode"
    name = "CompressAI model-zoo payload bits decoder to image batch"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = COMPRESSAI_ENTROPY_INFO

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        self.params_schema = object_schema(
            {
                "device": {"type": "string", "default": "cpu"},
                "on_error": {
                    "type": "string",
                    "default": "fail",
                    "enum": ["fail", "zeros", "gray_image", "erasure", "report_outage"],
                },
                "data_plane_backend": dataplane.backend_schema("auto"),
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        timing_records = []
        decoded_item_success: List[bool] = []
        try:
            independent = _independent_item_payload_bytes(bits, metadata, data_backend)
            if independent is not None:
                raw_rows, byte_backend = independent
                outages = _source_item_outages(metadata, len(raw_rows))
                on_error = str(ctx.params.get("on_error", "fail"))
                decoded = []
                model = None
                model_signature = None
                vbr_params = None
                _report_example_progress(ctx, 0, len(raw_rows), "decoding")
                with torch.no_grad():
                    for index, raw in enumerate(raw_rows):
                        try:
                            if outages[index]:
                                raise RuntimeError("source item failed CRC protection")
                            codec_payload = _codec_payload_loads(
                                raw, metadata.get("payload_format")
                            )
                            signature = (
                                str(codec_payload["model"]),
                                int(codec_payload["quality"]),
                                str(codec_payload["metric"]),
                                bool(codec_payload["pretrained"]),
                                str(codec_payload.get("model_state_sha256") or ""),
                            )
                            if model is None:
                                setup_start = time.perf_counter()
                                model_signature = signature
                                model = _compressai_model(
                                    zoo,
                                    signature[0],
                                    signature[1],
                                    signature[2],
                                    signature[3],
                                )
                                device = str(ctx.params.get("device", "cpu"))
                                model = model.to(device).eval()
                                vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
                                if hasattr(model, "update"):
                                    _compressai_update_model(model, vbr_params)
                                actual_model_state_sha256 = (
                                    _torch_module_state_sha256(torch, model)
                                )
                                expected_model_state_sha256 = str(
                                    codec_payload.get("model_state_sha256") or ""
                                )
                                if (
                                    expected_model_state_sha256
                                    and expected_model_state_sha256
                                    != actual_model_state_sha256
                                ):
                                    raise RuntimeError(
                                        "CompressAI decoder model-state SHA-256 "
                                        "does not match the encoded payload"
                                    )
                                append_measurement(
                                    timing_records,
                                    None,
                                    "decoder.setup",
                                    time.perf_counter() - setup_start,
                                )
                            elif signature != model_signature:
                                raise RuntimeError("source items declare different CompressAI models")
                            entry = codec_payload["entry"]
                            total_start = time.perf_counter()
                            with _compressai_timing_scope(
                                model,
                                timing_records,
                                index,
                                "decoder",
                                entry.get("original_shape"),
                            ):
                                output = _compressai_decompress(
                                    model, entry["strings"], entry["shape"], vbr_params
                                )
                            image = _torch_to_images(
                                torch, output["x_hat"], tuple(entry["original_shape"])
                            )[0]
                            decoded_item_success.append(True)
                            append_measurement(
                                timing_records,
                                index,
                                "decoder.total",
                                time.perf_counter() - total_start,
                                image_shape=entry.get("original_shape"),
                            )
                        except Exception as item_exc:
                            if on_error not in {"zeros", "gray_image", "erasure", "report_outage"}:
                                raise RuntimeError(
                                    "CompressAI source item %d decode failed: %s"
                                    % (index, item_exc)
                                ) from item_exc
                            image = _fallback_decode_image(metadata, index, on_error)
                            decoded_item_success.append(False)
                        decoded.append(image)
                        _report_example_progress(ctx, index + 1, len(raw_rows), "decoding")
                images, decoded_shapes = _stack_image_list(decoded)
            else:
                raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
                codec_payload = _codec_payload_loads(
                    raw, metadata.get("payload_format")
                )
                setup_start = time.perf_counter()
                model_name = str(codec_payload["model"])
                quality = int(codec_payload["quality"])
                metric = str(codec_payload["metric"])
                model = _compressai_model(zoo, model_name, quality, metric, bool(codec_payload["pretrained"]))
                device = str(ctx.params.get("device", "cpu"))
                model = model.to(device).eval()
                vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
                if hasattr(model, "update"):
                    _compressai_update_model(model, vbr_params)
                actual_model_state_sha256 = _torch_module_state_sha256(torch, model)
                expected_model_state_sha256 = str(
                    codec_payload.get("model_state_sha256") or ""
                )
                if (
                    expected_model_state_sha256
                    and expected_model_state_sha256 != actual_model_state_sha256
                ):
                    raise RuntimeError(
                        "CompressAI decoder model-state SHA-256 does not match "
                        "the encoded payload"
                    )
                append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
                entries = codec_payload.get("entries")
                with torch.no_grad():
                    if entries:
                        _report_example_progress(ctx, 0, len(entries), "decoding")
                        decoded = []
                        for index, entry in enumerate(entries):
                            total_start = time.perf_counter()
                            with _compressai_timing_scope(model, timing_records, index, "decoder", entry.get("original_shape")):
                                output = _compressai_decompress(model, entry["strings"], entry["shape"], vbr_params)
                            decoded.append(_torch_to_images(torch, output["x_hat"], tuple(entry["original_shape"]))[0])
                            decoded_item_success.append(True)
                            append_measurement(
                                timing_records,
                                index,
                                "decoder.total",
                                time.perf_counter() - total_start,
                                image_shape=entry.get("original_shape"),
                            )
                            _report_example_progress(ctx, index + 1, len(entries), "decoding")
                        images, decoded_shapes = _stack_image_list(decoded)
                    else:
                        total = int(codec_payload.get("original_shape", [1])[0] or 1)
                        _report_example_progress(ctx, 0, total, "decoding")
                        total_start = time.perf_counter()
                        with _compressai_timing_scope(model, timing_records, 0, "decoder", codec_payload.get("original_shape")):
                            output = _compressai_decompress(model, codec_payload["strings"], codec_payload["shape"], vbr_params)
                        images = _torch_to_images(torch, output["x_hat"], tuple(codec_payload["original_shape"]))
                        decoded_item_success = [True] * int(images.shape[0])
                        append_measurement(
                            timing_records,
                            0,
                            "decoder.total",
                            time.perf_counter() - total_start,
                            image_shape=codec_payload.get("original_shape"),
                        )
                        _report_example_progress(ctx, total, total, "decoding")
        except Exception as exc:
            on_error = str(ctx.params.get("on_error", "fail"))
            if on_error not in {"zeros", "gray_image", "erasure", "report_outage"}:
                raise RuntimeError("CompressAI decode failed, likely due to corrupted payload bits: %s" % exc)
            images = _fallback_decode_images(metadata, on_error)
            decoded_item_success = [False] * int(images.shape[0])
        output_metadata = dict(metadata)
        if "decoded_shapes" in locals():
            output_metadata["original_shapes"] = decoded_shapes
        output_metadata.update({
            "source": "compressai_decode",
            "shape": list(images.shape),
            "source_item_decode_success": [bool(item) for item in decoded_item_success],
            "byte_stream_data_plane_backend": byte_backend,
        })
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "decoder.model": "Combined CompressAI neural transforms observed inside model.decompress.",
                        "decoder.payload_model_inference": "CompressAI probability-index preparation observed inside model.decompress.",
                        "decoder.symbol_decode": "Entropy decoder calls observed inside CompressAI model.decompress.",
                    },
                ),
            },
        )


class CompressAiAnalysisEncodeOperation(Operation):
    id = "model.compressai_analysis_encode"
    name = "CompressAI PyTorch analysis transform to latents"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "The analysis transform is a PyTorch module; Noema benchmark execution runs pretrained eval parameters.",
    }

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        model_ids = compressai_model_ids(self.catalog)
        self.params_schema = object_schema(
            {
                "model": {
                    "type": "string",
                    "default": default_compressai_model(self.catalog),
                    "enum": model_ids,
                },
                "quality": {
                    "type": "integer",
                    "default": default_compressai_quality(self.catalog),
                    "minimum": 1,
                    "maximum": 8,
                },
                "metric": {
                    "type": "string",
                    "default": default_compressai_metric(self.catalog),
                    "enum": ["mse", "ms-ssim"],
                },
                "pretrained": {"type": "boolean", "default": True},
                "device": {"type": "string", "default": "cpu"},
                "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
                "quantize_latents": {"type": "boolean", "default": False},
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["model_catalog"] = {"compressai": compressai_model_ids(self.catalog)}
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI PyTorch analysis transform",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        timing_records = []
        setup_start = time.perf_counter()
        model_name = str(ctx.params.get("model", default_compressai_model(self.catalog)))
        quality = int(ctx.params.get("quality", default_compressai_quality(self.catalog)))
        metric = str(ctx.params.get("metric", default_compressai_metric(self.catalog)))
        _validate_compressai_choice(self.catalog, model_name, quality, metric)
        model = _compressai_model(zoo, model_name, quality, metric, bool(ctx.params.get("pretrained", True)))
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        vbr_params = _compressai_vbr_params(model, ctx.params)
        if vbr_params.get("enabled"):
            raise RuntimeError("Split CompressAI analysis/payload coding is not implemented for VBR models yet.")
        if not hasattr(model, "g_a"):
            raise RuntimeError("CompressAI model class %s does not expose g_a for split analysis." % model.__class__.__name__)
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        total = int(images.shape[0])
        latents = []
        latent_shapes = []
        padded_shapes = []
        pad_to_multiple = int(ctx.params.get("pad_to_multiple", 64))
        quantize = bool(ctx.params.get("quantize_latents", False))
        _report_example_progress(ctx, 0, total, "encoding")
        with torch.no_grad():
            for index in range(total):
                image = _image_for_index(images, input_metadata, index)
                image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
                total_start = time.perf_counter()
                tensor, padded_shape = timed_call(
                    timing_records,
                    index,
                    "encoder.preprocess",
                    lambda image=image: _images_to_torch(torch, image[None, ...], device, pad_to_multiple),
                    image_shape=image_shape,
                )
                padded_shapes.append([int(item) for item in padded_shape])
                y = timed_call(
                    timing_records,
                    index,
                    "encoder.inference",
                    lambda tensor=tensor: model.g_a(tensor),
                    image_shape=image_shape,
                )
                if quantize:
                    y = torch.round(y)
                array = timed_call(
                    timing_records,
                    index,
                    "encoder.postprocess",
                    lambda y=y: _torch_to_numpy(torch, y).astype(np.float32, copy=False),
                    image_shape=image_shape,
                )
                latents.append(array)
                latent_shapes.append([int(item) for item in array.shape])
                append_measurement(timing_records, index, "encoder.total", time.perf_counter() - total_start, image_shape=image_shape)
                _report_example_progress(ctx, index + 1, total, "encoding")
        latent_batch, stacked_latent_shapes = _stack_nchw_list(latents)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "compressai_analysis_encode",
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "pretrained": bool(ctx.params.get("pretrained", True)),
                "vbr": vbr_params,
                "semantic_form": "latents",
                "shape": list(latent_batch.shape),
                "dtype": str(latent_batch.dtype),
                "original_shape": list(images.shape),
                "latent_shapes": stacked_latent_shapes or latent_shapes,
                "padded_shapes": padded_shapes,
                "quantized": quantize,
                "runtime": "pytorch",
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latent_batch, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, metadata)},
            metadata={
                "shape": list(latent_batch.shape),
                "runtime": "pytorch",
                "codec_timing": codec_timing_metadata("encoder", timing_records, runner="pytorch"),
            },
        )


class CompressAiSynthesisDecodeOperation(Operation):
    id = "model.compressai_synthesis_decode"
    name = "CompressAI PyTorch synthesis transform to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "The synthesis transform is a PyTorch module; Noema benchmark execution runs pretrained eval parameters.",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def __init__(self) -> None:
        self.catalog = load_model_catalog()

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI PyTorch synthesis transform",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        timing_records = []
        try:
            setup_start = time.perf_counter()
            model_name = str(metadata.get("model") or default_compressai_model(self.catalog))
            quality = int(metadata.get("quality") or default_compressai_quality(self.catalog))
            metric = str(metadata.get("metric") or default_compressai_metric(self.catalog))
            model = _compressai_model(zoo, model_name, quality, metric, bool(metadata.get("pretrained", True)))
            device = str(ctx.params.get("device", "cpu"))
            model = model.to(device).eval()
            vbr_params = metadata.get("vbr") or _compressai_vbr_params(model, {})
            if vbr_params.get("enabled"):
                raise RuntimeError("Split CompressAI synthesis/payload coding is not implemented for VBR models yet.")
            if not hasattr(model, "g_s"):
                raise RuntimeError("CompressAI model class %s does not expose g_s for split synthesis." % model.__class__.__name__)
            if hasattr(model, "update"):
                _compressai_update_model(model, vbr_params)
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            decoded = []
            total = int(latents.shape[0])
            _report_example_progress(ctx, 0, total, "decoding")
            with torch.no_grad():
                for index in range(total):
                    latent = _latent_for_index(latents, metadata, index)
                    image_shape = _single_original_shape(metadata, index)
                    total_start = time.perf_counter()
                    tensor = torch.from_numpy(latent.astype(np.float32, copy=False)).to(device)
                    output = timed_call(
                        timing_records,
                        index,
                        "decoder.inference",
                        lambda tensor=tensor: model.g_s(tensor),
                        latent_shape=[int(item) for item in latent.shape],
                    )
                    image = timed_call(
                        timing_records,
                        index,
                        "decoder.postprocess",
                        lambda output=output, image_shape=image_shape: _torch_to_images(torch, output, tuple(image_shape))[0],
                        image_shape=image_shape,
                    )
                    decoded.append(image)
                    append_measurement(timing_records, index, "decoder.total", time.perf_counter() - total_start, image_shape=image_shape)
                    _report_example_progress(ctx, index + 1, total, "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("CompressAI synthesis decode failed: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
            decoded_shapes = [[1, int(shape[1]), int(shape[2]), int(shape[3])]] if len(shape) == 4 else []
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "source": "compressai_synthesis_decode",
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "original_shapes": decoded_shapes,
                "runtime": "pytorch",
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "runtime": "pytorch",
                "codec_timing": codec_timing_metadata("decoder", timing_records, runner="pytorch"),
            },
        )


class CompressAiEntropyEncodeOperation(Operation):
    id = "model.compressai_entropy_encode"
    name = "CompressAI payload encode latents to bits"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = COMPRESSAI_ENTROPY_INFO
    differentiability = {
        "framework": "torch",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Entropy coding and bitstream serialization are discrete payload-coding steps.",
    }

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        model_ids = compressai_model_ids(self.catalog)
        self.params_schema = object_schema(
            {
                "model": {
                    "type": "string",
                    "default": default_compressai_model(self.catalog),
                    "enum": model_ids,
                },
                "quality": {
                    "type": "integer",
                    "default": default_compressai_quality(self.catalog),
                    "minimum": 1,
                    "maximum": 8,
                },
                "metric": {
                    "type": "string",
                    "default": default_compressai_metric(self.catalog),
                    "enum": ["mse", "ms-ssim"],
                },
                "pretrained": {"type": "boolean", "default": True},
                "device": {"type": "string", "default": "cpu"},
                "data_plane_backend": dataplane.backend_schema("auto"),
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        latents, input_metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        timing_records = []
        setup_start = time.perf_counter()
        model_name = str(input_metadata.get("model") or ctx.params.get("model") or default_compressai_model(self.catalog))
        quality = int(input_metadata.get("quality") or ctx.params.get("quality") or default_compressai_quality(self.catalog))
        metric = str(input_metadata.get("metric") or ctx.params.get("metric") or default_compressai_metric(self.catalog))
        _validate_compressai_choice(self.catalog, model_name, quality, metric)
        model = _compressai_model(zoo, model_name, quality, metric, bool(input_metadata.get("pretrained", ctx.params.get("pretrained", True))))
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("byte_stream_data_plane_backend", "auto")))
        vbr_params = _compressai_vbr_params(model, input_metadata)
        if vbr_params.get("enabled"):
            raise RuntimeError("Split CompressAI payload coding is not implemented for VBR models yet.")
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        total = int(latents.shape[0])
        entries = []
        tensor = torch.from_numpy(latents.astype(np.float32, copy=False)).to(device)
        _report_example_progress(ctx, 0, total, "payload encoding")
        with torch.no_grad():
            for index in range(total):
                total_start = time.perf_counter()
                image_shape = _single_original_shape(input_metadata, index)
                with _compressai_timing_scope(
                    model,
                    timing_records,
                    index,
                    "encoder",
                    image_shape,
                    model_stage="encoder.payload_model_inference",
                ):
                    compressed = _compressai_entropy_encode_latents(model, tensor[index : index + 1])
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                entries.append(
                    {
                        "strings": compressed["strings"],
                        "shape": compressed["shape"],
                        "original_shape": image_shape,
                        "padded_shape": _metadata_sequence_item(input_metadata, "padded_shapes", index),
                        "latent_shape": [1] + [int(item) for item in latents[index].shape],
                    }
                )
                _report_example_progress(ctx, index + 1, total, "payload encoding")

        codec_payload = {
            "entries": entries,
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(input_metadata.get("pretrained", ctx.params.get("pretrained", True))),
            "vbr": vbr_params,
            "original_shape": list(input_metadata.get("original_shape") or input_metadata.get("shape") or []),
            "payload_version": 3,
            "split_entropy": True,
        }
        raw = safe_payload_dumps(codec_payload)
        bits, byte_backend = _bytes_to_bits(raw, data_backend)
        native_accounting = _native_codec_accounting(
            [_binary_payload_byte_count(entry["strings"]) for entry in entries],
            len(raw),
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "byte_count": len(raw),
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_format": SAFE_PAYLOAD_FORMAT,
                "payload_bit_count": int(bits.size),
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "entropy_stage": "compressai_split_entropy_encode",
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(COMPRESSAI_ENTROPY_INFO),
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": len(raw),
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
            },
            metadata={
                "model": model_name,
                "quality": quality,
                "metric": metric,
                **_entropy_metadata(COMPRESSAI_ENTROPY_INFO),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="pytorch_entropy",
                    notes={
                        "encoder.payload_model_inference": "CompressAI probability/prior neural transforms used by the payload codec.",
                        "encoder.symbol_encode": "CompressAI entropy coder calls that convert quantized latents into transmitted symbols/bytes.",
                    },
                ),
            },
        )


class CompressAiEntropyDecodeOperation(Operation):
    id = "model.compressai_entropy_decode"
    name = "CompressAI payload decode bits to latents"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    entropy_info = COMPRESSAI_ENTROPY_INFO
    differentiability = {
        "framework": "torch",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Entropy decoding and bitstream parsing are discrete payload-coding steps.",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        timing_records = []
        try:
            raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
            codec_payload = safe_payload_loads(raw)
            setup_start = time.perf_counter()
            model_name = str(codec_payload["model"])
            quality = int(codec_payload["quality"])
            metric = str(codec_payload["metric"])
            model = _compressai_model(zoo, model_name, quality, metric, bool(codec_payload["pretrained"]))
            device = str(ctx.params.get("device", "cpu"))
            model = model.to(device).eval()
            vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
            if vbr_params.get("enabled"):
                raise RuntimeError("Split CompressAI payload decoding is not implemented for VBR models yet.")
            if hasattr(model, "update"):
                _compressai_update_model(model, vbr_params)
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            entries = codec_payload.get("entries") or []
            if not entries:
                raise RuntimeError("CompressAI split entropy payload did not contain entries")
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "payload decoding")
            with torch.no_grad():
                for index, entry in enumerate(entries):
                    total_start = time.perf_counter()
                    with _compressai_timing_scope(
                        model,
                        timing_records,
                        index,
                        "decoder",
                        entry.get("original_shape"),
                        model_stage="decoder.payload_model_inference",
                    ):
                        y_hat = _compressai_entropy_decode_latents(model, entry["strings"], entry["shape"])
                    decoded.append(y_hat.detach().cpu().numpy().astype(np.float32, copy=False))
                    append_measurement(
                        timing_records,
                        index,
                        "decoder.total",
                        time.perf_counter() - total_start,
                        image_shape=entry.get("original_shape"),
                    )
                    _report_example_progress(ctx, index + 1, len(entries), "payload decoding")
            latents, latent_shapes = _stack_nchw_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError(_compressai_payload_decode_error("CompressAI payload decode", exc)) from exc
            latent_shape = tuple(int(item) for item in metadata.get("shape", [1, 1, 1, 1]))
            latents = np.zeros(latent_shape, dtype=np.float32)

        output_metadata = dict(metadata)
        if "latent_shapes" in locals():
            output_metadata["latent_shapes"] = latent_shapes
        output_metadata.update(
            {
                "source": "compressai_entropy_decode",
                "semantic_form": "latents",
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "entropy_coded": True,
                "entropy_stage": "compressai_split_entropy_decode",
                "byte_stream_data_plane_backend": byte_backend,
                **_entropy_metadata(COMPRESSAI_ENTROPY_INFO),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={
                "shape": list(latents.shape),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch_entropy",
                    notes={
                        "decoder.symbol_decode": "CompressAI entropy decoder calls that recover quantized latent symbols from transmitted bytes.",
                        "decoder.payload_model_inference": "CompressAI probability/prior neural transforms used by the payload decoder.",
                    },
                ),
            },
        )


class CompressAiOnnxEntropyEncodeOperation(Operation):
    id = "model.compressai_onnx_entropy_encode"
    name = "CompressAI ONNX payload encode latents to bits"
    runtime_id = "onnxruntime"
    runtime_label = "ONNX Runtime"
    timing_runner = "onnxruntime+compressai_entropy"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **COMPRESSAI_ENTROPY_INFO,
        "implementation": "ONNX Runtime hyperprior transforms + CompressAI entropy backend",
        "language": "ONNX Runtime + C++ extension + Python/PyTorch",
    }
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "device": {"type": "string", "default": "cpu"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def __init__(self) -> None:
        self.catalog = load_model_catalog()

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["compressai", "torch", "onnxruntime"],
            "CompressAI ONNX payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        _, ort = _require_onnx_stack(require_onnx=False)
        latents, input_metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("byte_stream_data_plane_backend", "auto")))
        runtime_evidence = _onnx_base_evidence_for_runtime(self.runtime_id, ort)
        _report_setup_progress(ctx, "Preparing CompressAI ONNX payload encoder")
        setup_start = time.perf_counter()
        model_name = str(bundle.get("model") or input_metadata.get("model") or default_compressai_model(self.catalog))
        quality = int(bundle.get("quality") or input_metadata.get("quality") or default_compressai_quality(self.catalog))
        metric = str(bundle.get("metric") or input_metadata.get("metric") or default_compressai_metric(self.catalog))
        model = _compressai_model(zoo, model_name, quality, metric, bool(bundle.get("pretrained", input_metadata.get("pretrained", True))))
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        vbr_params = _compressai_vbr_params(model, input_metadata)
        if vbr_params.get("enabled"):
            raise RuntimeError("ONNX split payload coding is not implemented for CompressAI VBR models yet.")
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        hyper_analysis_session = None
        hyper_synthesis_session = None
        if not _is_compressai_factorized_model(model):
            hyper_analysis_session, hyper_synthesis_session = _onnx_hyper_sessions(bundle, ort, providers, self.runtime_id, ctx)
            runtime_evidence["hyper_analysis_evidence"] = _onnx_session_evidence_for_runtime(
                self.runtime_id,
                ort, hyper_analysis_session, bundle["hyper_analysis_path"], providers[0]
            )
            runtime_evidence["hyper_synthesis_evidence"] = _onnx_session_evidence_for_runtime(
                self.runtime_id,
                ort, hyper_synthesis_session, bundle["hyper_synthesis_path"], providers[0]
            )
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        total = int(latents.shape[0])
        entries = []
        _report_example_progress(ctx, 0, total, "payload encoding")
        with torch.no_grad():
            for index in range(total):
                _report_example_active_progress(ctx, index, total, "payload encoding")
                total_start = time.perf_counter()
                image_shape = _single_original_shape(input_metadata, index)
                latent = _latent_for_index(latents, input_metadata, index)
                y_tensor = torch.from_numpy(latent.astype(np.float32, copy=False)).to(device)
                compressed = _compressai_entropy_encode_latents_onnx(
                    torch,
                    model,
                    y_tensor,
                    latent,
                    hyper_analysis_session,
                    hyper_synthesis_session,
                    timing_records,
                    index,
                    image_shape,
                    device,
                )
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                entries.append(
                    {
                        "strings": compressed["strings"],
                        "shape": compressed["shape"],
                        "original_shape": image_shape,
                        "padded_shape": _metadata_sequence_item(input_metadata, "padded_shapes", index),
                        "latent_shape": [int(item) for item in latent.shape],
                    }
                )
                _report_example_progress(ctx, index + 1, total, "payload encoding")

        hyper_runtime = self.runtime_id if hyper_analysis_session is not None else "not_applicable"
        codec_payload = {
            "entries": entries,
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(bundle.get("pretrained", input_metadata.get("pretrained", True))),
            "vbr": vbr_params,
            "original_shape": list(input_metadata.get("original_shape") or input_metadata.get("shape") or []),
            "payload_version": 4,
            "split_entropy": True,
        }
        raw = safe_payload_dumps(codec_payload)
        bits, byte_backend = _bytes_to_bits(raw, data_backend)
        native_accounting = _native_codec_accounting(
            [_binary_payload_byte_count(entry["strings"]) for entry in entries],
            len(raw),
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "byte_count": len(raw),
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_format": SAFE_PAYLOAD_FORMAT,
                "payload_bit_count": int(bits.size),
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "entropy_stage": "compressai_onnx_split_entropy_encode",
                "hyper_runtime": hyper_runtime,
                "runtime_evidence": runtime_evidence,
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": len(raw),
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
            },
            metadata={
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "runtime": self.runtime_id,
                "hyper_runtime": hyper_runtime,
                "runtime_evidence": runtime_evidence,
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner=self.timing_runner,
                    notes={
                        **runtime_evidence,
                        "encoder.payload_model_inference": "CompressAI h_a/h_s hyperprior transforms run with ONNX Runtime during payload encoding.",
                        "encoder.symbol_encode": "CompressAI entropy coder calls produce the transmitted bitstream.",
                    },
                ),
            },
        )


class CompressAiOnnxEntropyDecodeOperation(Operation):
    id = "model.compressai_onnx_entropy_decode"
    name = "CompressAI ONNX payload decode bits to latents"
    runtime_id = "onnxruntime"
    runtime_label = "ONNX Runtime"
    timing_runner = "onnxruntime+compressai_entropy"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    entropy_info = CompressAiOnnxEntropyEncodeOperation.entropy_info
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["compressai", "torch", "onnxruntime"],
            "CompressAI ONNX payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        _, ort = _require_onnx_stack(require_onnx=False)
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        hyper_synthesis_session = None
        runtime_evidence = _onnx_base_evidence_for_runtime(self.runtime_id, ort)
        try:
            raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
            codec_payload = safe_payload_loads(raw)
            _report_setup_progress(ctx, "Preparing CompressAI ONNX payload decoder")
            setup_start = time.perf_counter()
            model_name = str(codec_payload["model"])
            quality = int(codec_payload["quality"])
            metric = str(codec_payload["metric"])
            model = _compressai_model(zoo, model_name, quality, metric, bool(codec_payload["pretrained"]))
            device = str(ctx.params.get("device", "cpu"))
            model = model.to(device).eval()
            vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
            if vbr_params.get("enabled"):
                raise RuntimeError("ONNX split payload decoding is not implemented for CompressAI VBR models yet.")
            if hasattr(model, "update"):
                _compressai_update_model(model, vbr_params)
            if not _is_compressai_factorized_model(model):
                _, hyper_synthesis_session = _onnx_hyper_sessions(bundle, ort, providers, self.runtime_id, ctx)
                runtime_evidence["hyper_synthesis_evidence"] = _onnx_session_evidence_for_runtime(
                    self.runtime_id,
                    ort, hyper_synthesis_session, bundle["hyper_synthesis_path"], providers[0]
                )
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            entries = codec_payload.get("entries") or []
            if not entries:
                raise RuntimeError("CompressAI ONNX split entropy payload did not contain entries")
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "payload decoding")
            with torch.no_grad():
                for index, entry in enumerate(entries):
                    _report_example_active_progress(ctx, index, len(entries), "payload decoding")
                    total_start = time.perf_counter()
                    y_hat = _compressai_entropy_decode_latents_onnx(
                        torch,
                        model,
                        entry["strings"],
                        entry["shape"],
                        hyper_synthesis_session,
                        timing_records,
                        index,
                        entry.get("original_shape"),
                        device,
                    )
                    decoded.append(y_hat.detach().cpu().numpy().astype(np.float32, copy=False))
                    append_measurement(
                        timing_records,
                        index,
                        "decoder.total",
                        time.perf_counter() - total_start,
                        image_shape=entry.get("original_shape"),
                    )
                    _report_example_progress(ctx, index + 1, len(entries), "payload decoding")
            latents, latent_shapes = _stack_nchw_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError(_compressai_payload_decode_error("CompressAI ONNX payload decode", exc)) from exc
            latent_shape = tuple(int(item) for item in metadata.get("shape", [1, 1, 1, 1]))
            latents = np.zeros(latent_shape, dtype=np.float32)
            runtime_evidence = {**_onnx_base_evidence_for_runtime(self.runtime_id, ort), "error": str(exc)}

        output_metadata = dict(metadata)
        if "latent_shapes" in locals():
            output_metadata["latent_shapes"] = latent_shapes
        output_metadata.update(
            {
                "source": "compressai_onnx_entropy_decode",
                "semantic_form": "latents",
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "entropy_coded": True,
                "entropy_stage": "compressai_onnx_split_entropy_decode",
                "hyper_runtime": self.runtime_id if hyper_synthesis_session is not None else "not_applicable",
                "runtime_evidence": runtime_evidence,
                "byte_stream_data_plane_backend": byte_backend,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={
                "shape": list(latents.shape),
                "runtime": self.runtime_id,
                "hyper_runtime": output_metadata["hyper_runtime"],
                "runtime_evidence": runtime_evidence,
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner=self.timing_runner,
                    notes={
                        **runtime_evidence,
                        "decoder.symbol_decode": "CompressAI entropy decoder calls recover quantized latent symbols from transmitted bytes.",
                        "decoder.payload_model_inference": "CompressAI h_s hyperprior transform runs with ONNX Runtime during payload decoding.",
                    },
                ),
            },
        )


class CompressAiOnnxExportOperation(Operation):
    id = "model.compressai_export_onnx"
    name = "Convert CompressAI PyTorch transforms to ONNX"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"model": "model.onnx.bundle"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        model_ids = compressai_model_ids(self.catalog)
        self.params_schema = object_schema(
            {
                "model": {
                    "type": "string",
                    "default": default_compressai_model(self.catalog),
                    "enum": model_ids,
                },
                "quality": {
                    "type": "integer",
                    "default": default_compressai_quality(self.catalog),
                    "minimum": 1,
                    "maximum": 8,
                },
                "metric": {
                    "type": "string",
                    "default": default_compressai_metric(self.catalog),
                    "enum": ["mse", "ms-ssim"],
                },
                "pretrained": {"type": "boolean", "default": True},
                "device": {"type": "string", "default": "cpu"},
                "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
                "export_height": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Advanced. 0 exports for the selected data shape after padding.",
                },
                "export_width": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Advanced. 0 exports for the selected data shape after padding.",
                },
                "opset": {"type": "integer", "default": 17, "minimum": 11},
                "vbr_scale_index": {"type": "integer", "default": 1, "minimum": 0, "maximum": 7},
                "vbr_stage": {"type": "integer", "default": 2, "minimum": 1, "maximum": 2},
                "validate_export": {"type": "boolean", "default": True},
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["compressai", "torch", "onnx", "onnxruntime"],
            "CompressAI ONNX Runtime",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        torch, zoo = _require_compressai()
        onnx, ort = _require_onnx_stack()
        images, image_metadata = _load_images(ctx.require_input("images").path)
        model_name = str(ctx.params.get("model", default_compressai_model(self.catalog)))
        quality = int(ctx.params.get("quality", default_compressai_quality(self.catalog)))
        metric = str(ctx.params.get("metric", default_compressai_metric(self.catalog)))
        _validate_compressai_choice(self.catalog, model_name, quality, metric)
        model = _compressai_model(zoo, model_name, quality, metric, bool(ctx.params.get("pretrained", True)))
        vbr_params = _compressai_vbr_params(model, ctx.params)
        if vbr_params.get("enabled"):
            raise RuntimeError(
                "ONNX export for CompressAI VBR models is not implemented yet. "
                "Use a fixed-quality CompressAI model or PyTorch runtime for this recipe."
            )
        if not hasattr(model, "g_a") or not hasattr(model, "g_s"):
            raise RuntimeError(
                "This CompressAI model does not expose g_a/g_s transforms, so Noema cannot split it into ONNX encoder/decoder graphs."
            )
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        data_height = int(images.shape[1])
        data_width = int(images.shape[2])
        requested_height = int(ctx.params.get("export_height", 0) or 0)
        requested_width = int(ctx.params.get("export_width", 0) or 0)
        height = max(data_height, requested_height) if requested_height > 0 else data_height
        width = max(data_width, requested_width) if requested_width > 0 else data_width
        multiple = int(ctx.params.get("pad_to_multiple", 64))
        height += (multiple - height % multiple) % multiple
        width += (multiple - width % multiple) % multiple
        opset = int(ctx.params.get("opset", 17))
        bundle_dir = ctx.step_dir / "onnx_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = bundle_dir / "analysis.onnx"
        synthesis_path = bundle_dir / "synthesis.onnx"
        has_hyper_transforms = hasattr(model, "h_a") and hasattr(model, "h_s")
        total_steps = 6 if has_hyper_transforms else 4
        ctx.report_progress(
            "Exporting analysis transform",
            phase="conversion",
            completed=1,
            total=total_steps,
            percent=100.0 / float(total_steps),
        )
        dummy = torch.zeros((1, 3, height, width), dtype=torch.float32, device=device)
        try:
            with torch.no_grad():
                latents = model.g_a(dummy)
            torch.onnx.export(
                model.g_a,
                dummy,
                str(analysis_path),
                input_names=["images"],
                output_names=["latents"],
                dynamic_axes={
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
                },
                opset_version=opset,
            )
        except Exception as exc:
            raise RuntimeError("CompressAI analysis transform could not be exported to ONNX: %s" % exc) from exc

        hyper_analysis_path = None
        hyper_synthesis_path = None
        if has_hyper_transforms:
            hyper_analysis_path = bundle_dir / "hyper_analysis.onnx"
            hyper_synthesis_path = bundle_dir / "hyper_synthesis.onnx"
            hyper_analysis = _compressai_hyper_analysis_wrapper(
                torch,
                model.h_a,
                use_abs=_is_compressai_scale_hyperprior_model(model),
            ).to(device).eval()
            ctx.report_progress(
                "Exporting hyper-analysis transform",
                phase="conversion",
                completed=2,
                total=total_steps,
                percent=200.0 / float(total_steps),
            )
            try:
                with torch.no_grad():
                    hyper_latents = hyper_analysis(latents)
                torch.onnx.export(
                    hyper_analysis,
                    latents,
                    str(hyper_analysis_path),
                    input_names=["latents"],
                    output_names=["hyper_latents"],
                    dynamic_axes={
                        "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
                        "hyper_latents": {0: "batch", 2: "hyper_height", 3: "hyper_width"},
                    },
                    opset_version=opset,
                )
            except Exception as exc:
                raise RuntimeError("CompressAI hyper-analysis transform could not be exported to ONNX: %s" % exc) from exc
            ctx.report_progress(
                "Exporting hyper-synthesis transform",
                phase="conversion",
                completed=3,
                total=total_steps,
                percent=300.0 / float(total_steps),
            )
            try:
                torch.onnx.export(
                    model.h_s,
                    hyper_latents,
                    str(hyper_synthesis_path),
                    input_names=["hyper_latents"],
                    output_names=["gaussian_params"],
                    dynamic_axes={
                        "hyper_latents": {0: "batch", 2: "hyper_height", 3: "hyper_width"},
                        "gaussian_params": {0: "batch", 2: "latent_height", 3: "latent_width"},
                    },
                    opset_version=opset,
                )
            except Exception as exc:
                raise RuntimeError("CompressAI hyper-synthesis transform could not be exported to ONNX: %s" % exc) from exc

        ctx.report_progress(
            "Exporting synthesis transform",
            phase="conversion",
            completed=4 if has_hyper_transforms else 2,
            total=total_steps,
            percent=(400.0 if has_hyper_transforms else 200.0) / float(total_steps),
        )
        try:
            torch.onnx.export(
                model.g_s,
                latents,
                str(synthesis_path),
                input_names=["latents"],
                output_names=["images"],
                dynamic_axes={
                    "latents": {0: "batch", 2: "latent_height", 3: "latent_width"},
                    "images": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=opset,
            )
        except Exception as exc:
            raise RuntimeError("CompressAI synthesis transform could not be exported to ONNX: %s" % exc) from exc
        ctx.report_progress(
            "Validating ONNX bundle",
            phase="conversion",
            completed=5 if has_hyper_transforms else 3,
            total=total_steps,
            percent=(500.0 if has_hyper_transforms else 300.0) / float(total_steps),
        )
        try:
            onnx.checker.check_model(str(analysis_path))
            onnx.checker.check_model(str(synthesis_path))
            if hyper_analysis_path is not None and hyper_synthesis_path is not None:
                onnx.checker.check_model(str(hyper_analysis_path))
                onnx.checker.check_model(str(hyper_synthesis_path))
            if bool(ctx.params.get("validate_export", True)):
                providers = _onnx_providers("CPUExecutionProvider", ort)
                ort.InferenceSession(str(analysis_path), providers=providers)
                ort.InferenceSession(str(synthesis_path), providers=providers)
                if hyper_analysis_path is not None and hyper_synthesis_path is not None:
                    ort.InferenceSession(str(hyper_analysis_path), providers=providers)
                    ort.InferenceSession(str(hyper_synthesis_path), providers=providers)
        except Exception as exc:
            raise RuntimeError("Exported ONNX bundle failed validation: %s" % exc) from exc
        metadata = {
            "codec": "compressai",
            "source_runtime": "pytorch",
            "runtime": "onnxruntime",
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(ctx.params.get("pretrained", True)),
            "opset": opset,
            "analysis_path": str(analysis_path),
            "synthesis_path": str(synthesis_path),
            "representation": "latents",
            "latent_dtype": "float32",
            "data_input_shape": list(images.shape),
            "data_image_ids": image_metadata.get("image_ids"),
            "export_input_shape": [1, 3, height, width],
            "export_shape_source": "data.images",
            "hyper_analysis_path": str(hyper_analysis_path) if hyper_analysis_path is not None else "",
            "hyper_synthesis_path": str(hyper_synthesis_path) if hyper_synthesis_path is not None else "",
            "hyper_transforms_exported": bool(has_hyper_transforms),
            "note": "This ONNX bundle exports CompressAI neural analysis/synthesis transforms. Use CompressAI entropy blocks to keep the transmitted payload as entropy-coded bitstream.",
        }
        artifact_sizes = _model_artifact_sizes(
            encoder_paths=[analysis_path, hyper_analysis_path, hyper_synthesis_path],
            decoder_paths=[synthesis_path, hyper_synthesis_path],
        )
        metadata.update(artifact_sizes)
        manifest_path = ctx.output_path("onnx_bundle", ".json")
        manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        ctx.report_progress("Prepared ONNX bundle", phase="conversion", completed=total_steps, total=total_steps, percent=100.0)
        return OperationResult(
            outputs={"model": artifact("model.onnx.bundle", manifest_path, metadata)},
            metrics={
                "model_conversion.wall_time_s": time.perf_counter() - start,
                "model_artifact.encoder_bytes": int(artifact_sizes["encoder_model_bytes"]),
                "model_artifact.decoder_bytes": int(artifact_sizes["decoder_model_bytes"]),
                "model_artifact.total_bytes": int(artifact_sizes["model_artifact_bytes"]),
            },
            metadata={"codec_timing": codec_timing_metadata("conversion", [], runner="pytorch_to_onnx", notes=metadata)},
        )


class CompressAiOnnxEncodeOperation(Operation):
    id = "model.compressai_onnx_encode"
    name = "CompressAI ONNX Runtime analysis transform to latents"
    runtime_id = "onnxruntime"
    runtime_label = "ONNX Runtime"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
            "quantize_latents": {"type": "boolean", "default": False},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["onnxruntime"],
            "ONNX Runtime",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        timing_records = []
        _report_setup_progress(ctx, "Creating CompressAI %s encoder session" % self.runtime_label)
        setup_start = time.perf_counter()
        session = _create_onnx_session_for_runtime(self.runtime_id, ort, bundle["analysis_path"], providers, ctx, "encoder")
        _assert_onnx_session_provider(session, providers, "encoder")
        runtime_evidence = _onnx_session_evidence_for_runtime(
            self.runtime_id,
            ort,
            session,
            bundle["analysis_path"],
            providers[0],
        )
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        total = int(images.shape[0])
        _report_example_progress(ctx, 0, total, "encoding")
        latents = []
        padded_shapes = []
        quantize = bool(ctx.params.get("quantize_latents", False))
        for index in range(int(images.shape[0])):
            _report_example_active_progress(ctx, index, total, "encoding")
            image = _image_for_index(images, input_metadata, index)
            total_start = time.perf_counter()
            tensor = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: _images_to_nchw_numpy(image[None, ...], int(ctx.params.get("pad_to_multiple", 64)), data_backend),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            padded_shapes.append([int(item) for item in tensor.shape])
            output = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda tensor=tensor: session.run(None, {"images": tensor})[0],
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            if quantize:
                output = np.round(output)
            latents.append(output.astype(np.float32, copy=False))
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            _report_example_progress(ctx, index + 1, total, "encoding")
        latent_batch, latent_shapes = _stack_nchw_list(latents)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "compressai_%s_encode" % self.runtime_id,
                "codec": "compressai",
                "model": bundle.get("model"),
                "quality": bundle.get("quality"),
                "metric": bundle.get("metric"),
                "semantic_form": "latents",
                "shape": list(latent_batch.shape),
                "dtype": str(latent_batch.dtype),
                "original_shape": list(images.shape),
                "latent_shapes": latent_shapes,
                "padded_shapes": padded_shapes,
                "quantized": quantize,
                "runtime": self.runtime_id,
                "provider": providers[0],
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "image_to_nchw"),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latent_batch, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, metadata)},
            metadata={
                "shape": list(latent_batch.shape),
                "runtime": self.runtime_id,
                "provider": providers[0],
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner=self.runtime_id,
                    notes=runtime_evidence,
                ),
            },
        )


class CompressAiOnnxDecodeOperation(Operation):
    id = "model.compressai_onnx_decode"
    name = "CompressAI ONNX Runtime quantized latent decoder to image batch"
    runtime_id = "onnxruntime"
    runtime_label = "ONNX Runtime"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["onnxruntime"],
            "ONNX Runtime",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("data_plane_backend", "auto")))
        timing_records = []
        _report_setup_progress(ctx, "Creating CompressAI %s decoder session" % self.runtime_label)
        setup_start = time.perf_counter()
        session = _create_onnx_session_for_runtime(self.runtime_id, ort, bundle["synthesis_path"], providers, ctx, "decoder")
        _assert_onnx_session_provider(session, providers, "decoder")
        runtime_evidence = _onnx_session_evidence_for_runtime(
            self.runtime_id,
            ort,
            session,
            bundle["synthesis_path"],
            providers[0],
        )
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        total = int(latents.shape[0])
        _report_example_progress(ctx, 0, total, "decoding")
        decoded = []
        try:
            for index in range(total):
                _report_example_active_progress(ctx, index, total, "decoding")
                latent = _latent_for_index(latents, metadata, index)[0]
                total_start = time.perf_counter()
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda latent=latent: session.run(None, {"latents": latent[None, ...].astype(np.float32, copy=False)})[0],
                    latent_shape=[int(item) for item in latent.shape],
                )
                image_shape = _single_original_shape(metadata, index)
                decoded.append(_nchw_numpy_to_images(output, image_shape, data_backend)[0])
                append_measurement(
                    timing_records,
                    index,
                    "decoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                _report_example_progress(ctx, index + 1, total, "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("CompressAI ONNX decode failed: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
        output_metadata = dict(metadata)
        if "decoded_shapes" in locals():
            output_metadata["original_shapes"] = decoded_shapes
        output_metadata.update(
            {
                "source": "compressai_%s_decode" % self.runtime_id,
                "shape": list(images.shape),
                "runtime": self.runtime_id,
                "provider": providers[0],
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "nchw_to_image"),
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "runtime": self.runtime_id,
                "provider": providers[0],
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner=self.runtime_id,
                    notes=runtime_evidence,
                ),
            },
        )


class CompressAiOnnxCppEncodeOperation(CompressAiOnnxEncodeOperation):
    id = "model.compressai_onnx_cpp_encode"
    name = "CompressAI ONNX Runtime C++ API analysis transform to latents"
    runtime_id = "onnxruntime_cpp"
    runtime_label = "ONNX Runtime C++ API"
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "library_path": {"type": "string", "default": ""},
            "intra_op_num_threads": {"type": "integer", "default": 0, "minimum": 0},
            "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
            "quantize_latents": {"type": "boolean", "default": False},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = Operation.describe(self)
        payload["availability"] = _merged_availability(
            "onnx",
            ["onnxruntime"],
            "CompressAI ONNX Runtime C++ API",
            cpp_runtime_available(),
        )
        return payload


class CompressAiOnnxCppDecodeOperation(CompressAiOnnxDecodeOperation):
    id = "model.compressai_onnx_cpp_decode"
    name = "CompressAI ONNX Runtime C++ API latent decoder to image batch"
    runtime_id = "onnxruntime_cpp"
    runtime_label = "ONNX Runtime C++ API"
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "library_path": {"type": "string", "default": ""},
            "intra_op_num_threads": {"type": "integer", "default": 0, "minimum": 0},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = Operation.describe(self)
        payload["availability"] = _merged_availability(
            "onnx",
            ["onnxruntime"],
            "CompressAI ONNX Runtime C++ API",
            cpp_runtime_available(),
        )
        return payload


class CompressAiOnnxCppEntropyEncodeOperation(CompressAiOnnxEntropyEncodeOperation):
    id = "model.compressai_onnx_cpp_entropy_encode"
    name = "CompressAI ONNX Runtime C++ API payload encode latents to bits"
    runtime_id = "onnxruntime_cpp"
    runtime_label = "ONNX Runtime C++ API"
    timing_runner = "onnxruntime_cpp+compressai_entropy"
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "library_path": {"type": "string", "default": ""},
            "intra_op_num_threads": {"type": "integer", "default": 0, "minimum": 0},
            "device": {"type": "string", "default": "cpu"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = Operation.describe(self)
        payload["availability"] = _merged_availability(
            "onnx",
            ["compressai", "torch", "onnxruntime"],
            "CompressAI ONNX Runtime C++ API payload coding",
            cpp_runtime_available(),
        )
        return payload


class CompressAiOnnxCppEntropyDecodeOperation(CompressAiOnnxEntropyDecodeOperation):
    id = "model.compressai_onnx_cpp_entropy_decode"
    name = "CompressAI ONNX Runtime C++ API payload decode bits to latents"
    runtime_id = "onnxruntime_cpp"
    runtime_label = "ONNX Runtime C++ API"
    timing_runner = "onnxruntime_cpp+compressai_entropy"
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "library_path": {"type": "string", "default": ""},
            "intra_op_num_threads": {"type": "integer", "default": 0, "minimum": 0},
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = Operation.describe(self)
        payload["availability"] = _merged_availability(
            "onnx",
            ["compressai", "torch", "onnxruntime"],
            "CompressAI ONNX Runtime C++ API payload coding",
            cpp_runtime_available(),
        )
        return payload


class CompressAiAotInductorExportOperation(Operation):
    id = "model.compressai_export_aoti"
    name = "Compile CompressAI PyTorch transforms with AOT Inductor"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"model": "model.aot_inductor.bundle"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        model_ids = compressai_model_ids(self.catalog)
        self.params_schema = object_schema(
            {
                "model": {
                    "type": "string",
                    "default": default_compressai_model(self.catalog),
                    "enum": model_ids,
                },
                "quality": {
                    "type": "integer",
                    "default": default_compressai_quality(self.catalog),
                    "minimum": 1,
                    "maximum": 8,
                },
                "metric": {
                    "type": "string",
                    "default": default_compressai_metric(self.catalog),
                    "enum": ["mse", "ms-ssim"],
                },
                "pretrained": {"type": "boolean", "default": True},
                "device": {"type": "string", "default": "cpu"},
                "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
                "export_height": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Advanced. 0 compiles for the selected data shape after padding.",
                },
                "export_width": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Advanced. 0 compiles for the selected data shape after padding.",
                },
                "vbr_scale_index": {"type": "integer", "default": 1, "minimum": 0, "maximum": 7},
                "vbr_stage": {"type": "integer", "default": 2, "minimum": 1, "maximum": 2},
                "dynamic_shapes": {
                    "type": "boolean",
                    "default": False,
                    "description": "Advanced. Keep off for reproducible selected-shape AOTI packages.",
                },
                "validate_export": {"type": "boolean", "default": True},
            }
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI AOT Inductor",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        torch, zoo = _require_compressai()
        images, image_metadata = _load_images(ctx.require_input("images").path)
        model_name = str(ctx.params.get("model", default_compressai_model(self.catalog)))
        quality = int(ctx.params.get("quality", default_compressai_quality(self.catalog)))
        metric = str(ctx.params.get("metric", default_compressai_metric(self.catalog)))
        _validate_compressai_choice(self.catalog, model_name, quality, metric)
        model = _compressai_model(zoo, model_name, quality, metric, bool(ctx.params.get("pretrained", True)))
        vbr_params = _compressai_vbr_params(model, ctx.params)
        if vbr_params.get("enabled"):
            raise RuntimeError(
                "AOT Inductor export for CompressAI VBR models is not implemented yet. "
                "Use a fixed-quality CompressAI model or PyTorch runtime for this recipe."
            )
        if not hasattr(model, "g_a") or not hasattr(model, "g_s"):
            raise RuntimeError(
                "This CompressAI model does not expose g_a/g_s transforms, so Noema cannot split it into AOT Inductor encoder/decoder graphs."
            )
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        requested_height = int(ctx.params.get("export_height", 0) or 0)
        requested_width = int(ctx.params.get("export_width", 0) or 0)
        multiple = int(ctx.params.get("pad_to_multiple", 64))
        bundle_dir = ctx.step_dir / "aoti_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        has_hyper_transforms = hasattr(model, "h_a") and hasattr(model, "h_s")
        allow_dynamic_shapes = bool(ctx.params.get("dynamic_shapes", False))
        sample_arrays = _aoti_export_sample_arrays(
            images,
            image_metadata,
            multiple,
            requested_height=requested_height,
            requested_width=requested_width,
        )
        if not sample_arrays:
            raise RuntimeError("AOT Inductor export could not find any input image shapes to compile")
        hyper_analysis_module = None
        if has_hyper_transforms:
            hyper_analysis_module = _compressai_hyper_analysis_wrapper(
                torch,
                model.h_a,
                use_abs=_is_compressai_scale_hyperprior_model(model),
            ).to(device).eval()
        steps_per_shape = 5 if has_hyper_transforms else 3
        total_steps = max(1, len(sample_arrays) * steps_per_shape + 1)
        completed_steps = 0

        def report(message: str) -> None:
            nonlocal completed_steps
            completed_steps += 1
            ctx.report_progress(
                message,
                phase="conversion",
                completed=completed_steps,
                total=total_steps,
                percent=float(completed_steps) / float(total_steps) * 100.0,
            )

        shape_packages: JsonDict = {}
        all_encoder_paths = []
        all_decoder_paths = []
        default_package = None
        for package_index, (shape_key, sample_array) in enumerate(sample_arrays.items()):
            suffix = "" if package_index == 0 else "_%s" % shape_key
            analysis_path = bundle_dir / ("analysis%s.pt2" % suffix)
            synthesis_path = bundle_dir / ("synthesis%s.pt2" % suffix)
            hyper_analysis_path = bundle_dir / ("hyper_analysis%s.pt2" % suffix) if has_hyper_transforms else None
            hyper_synthesis_path = bundle_dir / ("hyper_synthesis%s.pt2" % suffix) if has_hyper_transforms else None
            sample = torch.from_numpy(sample_array).to(device).contiguous()
            try:
                with torch.no_grad():
                    latents = model.g_a(sample)
                report("Compiling analysis transform for %s" % shape_key)
                analysis_shape_mode = _save_aoti_package(
                    torch,
                    model.g_a,
                    sample,
                    analysis_path,
                    "analysis",
                    allow_dynamic_shapes=allow_dynamic_shapes,
                )
            except Exception as exc:
                raise RuntimeError("CompressAI analysis transform could not be compiled with AOT Inductor: %s" % exc) from exc

            hyper_latents = None
            hyper_analysis_shape_mode = ""
            hyper_synthesis_shape_mode = ""
            if has_hyper_transforms and hyper_analysis_module is not None:
                try:
                    with torch.no_grad():
                        hyper_latents = hyper_analysis_module(latents)
                    report("Compiling hyper-analysis transform for %s" % shape_key)
                    hyper_analysis_shape_mode = _save_aoti_package(
                        torch,
                        hyper_analysis_module,
                        latents,
                        hyper_analysis_path,
                        "hyper-analysis",
                        allow_dynamic_shapes=allow_dynamic_shapes,
                    )
                except Exception as exc:
                    raise RuntimeError("CompressAI hyper-analysis transform could not be compiled with AOT Inductor: %s" % exc) from exc
                try:
                    report("Compiling hyper-synthesis transform for %s" % shape_key)
                    hyper_synthesis_shape_mode = _save_aoti_package(
                        torch,
                        model.h_s,
                        hyper_latents,
                        hyper_synthesis_path,
                        "hyper-synthesis",
                        allow_dynamic_shapes=allow_dynamic_shapes,
                    )
                except Exception as exc:
                    raise RuntimeError("CompressAI hyper-synthesis transform could not be compiled with AOT Inductor: %s" % exc) from exc

            try:
                report("Compiling synthesis transform for %s" % shape_key)
                synthesis_shape_mode = _save_aoti_package(
                    torch,
                    model.g_s,
                    latents,
                    synthesis_path,
                    "synthesis",
                    allow_dynamic_shapes=allow_dynamic_shapes,
                )
            except Exception as exc:
                raise RuntimeError("CompressAI synthesis transform could not be compiled with AOT Inductor: %s" % exc) from exc

            package = {
                "input_shape": [int(item) for item in sample.shape],
                "latent_shape": [int(item) for item in latents.shape],
                "hyper_shape": [int(item) for item in hyper_latents.shape] if hyper_latents is not None else [],
                "analysis_path": str(analysis_path),
                "synthesis_path": str(synthesis_path),
                "hyper_analysis_path": str(hyper_analysis_path) if hyper_analysis_path is not None else "",
                "hyper_synthesis_path": str(hyper_synthesis_path) if hyper_synthesis_path is not None else "",
                "analysis_shape_mode": analysis_shape_mode,
                "synthesis_shape_mode": synthesis_shape_mode,
                "hyper_analysis_shape_mode": hyper_analysis_shape_mode,
                "hyper_synthesis_shape_mode": hyper_synthesis_shape_mode,
            }
            if bool(ctx.params.get("validate_export", True)):
                try:
                    report("Validating AOT Inductor package for %s" % shape_key)
                    with torch.no_grad():
                        expected_latents = _torch_to_numpy(torch, latents)
                        traced_latents = _AotInductorSession(torch, analysis_path, device).run_first_output(sample_array)
                        _assert_runtime_output_close("AOT Inductor analysis", traced_latents, expected_latents)
                        expected_images = _torch_to_numpy(torch, model.g_s(latents))
                        traced_images = _AotInductorSession(torch, synthesis_path, device).run_first_output(expected_latents)
                        _assert_runtime_output_close("AOT Inductor synthesis", traced_images, expected_images)
                        if hyper_analysis_path is not None and hyper_synthesis_path is not None and hyper_latents is not None:
                            expected_hyper = _torch_to_numpy(torch, hyper_latents)
                            traced_hyper = _AotInductorSession(torch, hyper_analysis_path, device).run_first_output(expected_latents)
                            _assert_runtime_output_close("AOT Inductor hyper-analysis", traced_hyper, expected_hyper)
                            expected_gaussian_params = _torch_to_numpy(torch, model.h_s(hyper_latents))
                            traced_gaussian_params = _AotInductorSession(torch, hyper_synthesis_path, device).run_first_output(expected_hyper)
                            _assert_runtime_output_close(
                                "AOT Inductor hyper-synthesis",
                                traced_gaussian_params,
                                expected_gaussian_params,
                            )
                except Exception as exc:
                    raise RuntimeError("Compiled AOT Inductor bundle failed validation: %s" % exc) from exc
            shape_packages[shape_key] = package
            all_encoder_paths.extend([analysis_path, hyper_analysis_path, hyper_synthesis_path])
            all_decoder_paths.extend([synthesis_path, hyper_synthesis_path])
            if default_package is None:
                default_package = package

        metadata = {
            "codec": "compressai",
            "source_runtime": "pytorch",
            "runtime": "aot_inductor",
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(ctx.params.get("pretrained", True)),
            "analysis_path": str(default_package["analysis_path"]),
            "synthesis_path": str(default_package["synthesis_path"]),
            "representation": "latents",
            "latent_dtype": "float32",
            "data_input_shape": list(images.shape),
            "data_image_ids": image_metadata.get("image_ids"),
            "export_input_shape": list(default_package["input_shape"]),
            "available_input_shapes": [list(package["input_shape"]) for package in shape_packages.values()],
            "export_shape_source": "data.original_shapes",
            "dynamic_shapes_requested": allow_dynamic_shapes,
            "analysis_shape_mode": default_package["analysis_shape_mode"],
            "synthesis_shape_mode": default_package["synthesis_shape_mode"],
            "latent_shape": list(default_package["latent_shape"]),
            "hyper_shape": list(default_package["hyper_shape"]),
            "hyper_analysis_path": str(default_package["hyper_analysis_path"]),
            "hyper_synthesis_path": str(default_package["hyper_synthesis_path"]),
            "hyper_transforms_exported": bool(has_hyper_transforms),
            "hyper_analysis_shape_mode": default_package["hyper_analysis_shape_mode"],
            "hyper_synthesis_shape_mode": default_package["hyper_synthesis_shape_mode"],
            "shape_packages": shape_packages,
            "note": "This AOT Inductor bundle compiles CompressAI neural analysis/synthesis transforms. Use CompressAI entropy blocks to keep the transmitted payload as entropy-coded bitstream.",
        }
        artifact_sizes = _model_artifact_sizes(
            encoder_paths=all_encoder_paths,
            decoder_paths=all_decoder_paths,
        )
        metadata.update(artifact_sizes)
        manifest_path = ctx.output_path("aoti_bundle", ".json")
        manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        ctx.report_progress("Prepared AOT Inductor bundle", phase="conversion", completed=total_steps, total=total_steps, percent=100.0)
        return OperationResult(
            outputs={"model": artifact("model.aot_inductor.bundle", manifest_path, metadata)},
            metrics={
                "model_conversion.wall_time_s": time.perf_counter() - start,
                "model_artifact.encoder_bytes": int(artifact_sizes["encoder_model_bytes"]),
                "model_artifact.decoder_bytes": int(artifact_sizes["decoder_model_bytes"]),
                "model_artifact.total_bytes": int(artifact_sizes["model_artifact_bytes"]),
            },
            metadata={"codec_timing": codec_timing_metadata("conversion", [], runner="pytorch_to_aot_inductor", notes=metadata)},
        )


class CompressAiAotInductorEncodeOperation(Operation):
    id = "model.compressai_aoti_encode"
    name = "CompressAI AOT Inductor analysis transform to latents"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
            "quantize_latents": {"type": "boolean", "default": False},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("compressai", ["torch"], "AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, _zoo = _require_torch()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device", "cpu"))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        timing_records = []
        sessions: Dict[str, _AotInductorSession] = {}
        runtime_evidence: JsonDict = {
            "runtime": "aot_inductor",
            "shape_routed": True,
            "available_input_shapes": bundle.get("available_input_shapes") or [bundle.get("export_input_shape")],
        }
        total = int(images.shape[0])
        _report_example_progress(ctx, 0, total, "encoding")
        latents = []
        padded_shapes = []
        quantize = bool(ctx.params.get("quantize_latents", False))
        for index in range(total):
            _report_example_active_progress(ctx, index, total, "encoding")
            image = _image_for_index(images, input_metadata, index)
            total_start = time.perf_counter()
            tensor = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: torch.from_numpy(
                    _images_to_nchw_numpy(image[None, ...], int(ctx.params.get("pad_to_multiple", 64)), data_backend)
                ).to(device).contiguous(),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            package = _aoti_package_for_field_shape(bundle, "input_shape", [int(item) for item in tensor.shape])
            session_path = _aoti_package_path(package, "analysis_path")
            if session_path not in sessions:
                _report_setup_progress(ctx, "Loading AOT Inductor encoder package")
                setup_start = time.perf_counter()
                session = _aoti_session_from_cache(sessions, torch, session_path, device)
                append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
                runtime_evidence["last_loaded_analysis"] = _aoti_runtime_evidence(torch, session, session_path, device)
            else:
                session = sessions[session_path]
            padded_shapes.append([int(item) for item in tensor.shape])
            output_tensor = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda tensor=tensor: session.run_first_output_tensor(tensor),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            output = timed_call(
                timing_records,
                index,
                "encoder.postprocess",
                lambda output_tensor=output_tensor: session.tensor_to_numpy(output_tensor),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            if quantize:
                output = np.round(output)
            latents.append(output.astype(np.float32, copy=False))
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            _report_example_progress(ctx, index + 1, total, "encoding")
        latent_batch, latent_shapes = _stack_nchw_list(latents)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "compressai_aoti_encode",
                "codec": "compressai",
                "model": bundle.get("model"),
                "quality": bundle.get("quality"),
                "metric": bundle.get("metric"),
                "semantic_form": "latents",
                "shape": list(latent_batch.shape),
                "dtype": str(latent_batch.dtype),
                "original_shape": list(images.shape),
                "latent_shapes": latent_shapes,
                "padded_shapes": padded_shapes,
                "quantized": quantize,
                "runtime": "aot_inductor",
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "image_to_nchw"),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latent_batch, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, metadata)},
            metadata={
                "shape": list(latent_batch.shape),
                "runtime": "aot_inductor",
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata("encoder", timing_records, runner="aot_inductor", notes=runtime_evidence),
            },
        )


class CompressAiAotInductorEntropyEncodeOperation(Operation):
    id = "model.compressai_aoti_entropy_encode"
    name = "CompressAI AOT Inductor payload encode latents to bits"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **COMPRESSAI_ENTROPY_INFO,
        "implementation": "AOT Inductor hyperprior transforms + CompressAI entropy backend",
        "language": "AOT Inductor/PyTorch + C++ extension + Python",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def __init__(self) -> None:
        self.catalog = load_model_catalog()

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI AOT Inductor payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        latents, input_metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_aoti_bundle(ctx.require_input("model"))
        timing_records = []
        _report_setup_progress(ctx, "Compiling CompressAI OpenVINO encoder model")
        setup_start = time.perf_counter()
        model_name = str(bundle.get("model") or input_metadata.get("model") or default_compressai_model(self.catalog))
        quality = int(bundle.get("quality") or input_metadata.get("quality") or default_compressai_quality(self.catalog))
        metric = str(bundle.get("metric") or input_metadata.get("metric") or default_compressai_metric(self.catalog))
        model = _compressai_model(zoo, model_name, quality, metric, bool(bundle.get("pretrained", input_metadata.get("pretrained", True))))
        device = str(ctx.params.get("device", "cpu"))
        model = model.to(device).eval()
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("byte_stream_data_plane_backend", "auto")))
        vbr_params = _compressai_vbr_params(model, input_metadata)
        if vbr_params.get("enabled"):
            raise RuntimeError("AOT Inductor split payload coding is not implemented for CompressAI VBR models yet.")
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        hyper_analysis_session = None
        hyper_synthesis_session = None
        runtime_notes: JsonDict = {"runtime": "aot_inductor", "device": device}
        if not _is_compressai_factorized_model(model):
            hyper_analysis_session, hyper_synthesis_session = _aoti_hyper_sessions(bundle, torch, device)
            runtime_notes["shape_routed_hyperprior"] = True
            runtime_notes["available_latent_shapes"] = [
                package.get("latent_shape")
                for package in _aoti_shape_packages(bundle).values()
                if package.get("latent_shape")
            ]
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        total = int(latents.shape[0])
        entries = []
        _report_example_progress(ctx, 0, total, "payload encoding")
        with torch.no_grad():
            for index in range(total):
                total_start = time.perf_counter()
                image_shape = _single_original_shape(input_metadata, index)
                latent = _latent_for_index(latents, input_metadata, index)
                y_tensor = torch.from_numpy(latent.astype(np.float32, copy=False)).to(device)
                compressed = _compressai_entropy_encode_latents_onnx(
                    torch,
                    model,
                    y_tensor,
                    latent,
                    hyper_analysis_session,
                    hyper_synthesis_session,
                    timing_records,
                    index,
                    image_shape,
                    device,
                )
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                entries.append(
                    {
                        "strings": compressed["strings"],
                        "shape": compressed["shape"],
                        "original_shape": image_shape,
                        "padded_shape": _metadata_sequence_item(input_metadata, "padded_shapes", index),
                        "latent_shape": [int(item) for item in latent.shape],
                    }
                )
                _report_example_progress(ctx, index + 1, total, "payload encoding")

        hyper_runtime = "aot_inductor" if hyper_analysis_session is not None else "not_applicable"
        codec_payload = {
            "entries": entries,
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(bundle.get("pretrained", input_metadata.get("pretrained", True))),
            "vbr": vbr_params,
            "original_shape": list(input_metadata.get("original_shape") or input_metadata.get("shape") or []),
            "payload_version": 4,
            "split_entropy": True,
        }
        raw = safe_payload_dumps(codec_payload)
        bits, byte_backend = _bytes_to_bits(raw, data_backend)
        native_accounting = _native_codec_accounting(
            [_binary_payload_byte_count(entry["strings"]) for entry in entries],
            len(raw),
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "byte_count": len(raw),
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_format": SAFE_PAYLOAD_FORMAT,
                "payload_bit_count": int(bits.size),
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "entropy_stage": "compressai_aoti_split_entropy_encode",
                "hyper_runtime": hyper_runtime,
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": len(raw),
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
            },
            metadata={
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "runtime": "aot_inductor",
                "hyper_runtime": hyper_runtime,
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="aot_inductor+compressai_entropy",
                    notes={
                        **runtime_notes,
                        "encoder.payload_model_inference": "CompressAI h_a/h_s hyperprior transforms run with AOT Inductor during payload encoding.",
                        "encoder.symbol_encode": "CompressAI entropy coder calls produce the transmitted bitstream.",
                    },
                ),
            },
        )


class CompressAiAotInductorEntropyDecodeOperation(Operation):
    id = "model.compressai_aoti_entropy_decode"
    name = "CompressAI AOT Inductor payload decode bits to latents"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    entropy_info = CompressAiAotInductorEntropyEncodeOperation.entropy_info
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "compressai",
            ["compressai", "torch"],
            "CompressAI AOT Inductor payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_aoti_bundle(ctx.require_input("model"))
        timing_records = []
        hyper_synthesis_session = None
        device = str(ctx.params.get("device", "cpu"))
        runtime_notes: JsonDict = {"runtime": "aot_inductor", "device": device}
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        try:
            raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
            codec_payload = safe_payload_loads(raw)
            setup_start = time.perf_counter()
            model_name = str(codec_payload["model"])
            quality = int(codec_payload["quality"])
            metric = str(codec_payload["metric"])
            model = _compressai_model(zoo, model_name, quality, metric, bool(codec_payload["pretrained"]))
            model = model.to(device).eval()
            vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
            if vbr_params.get("enabled"):
                raise RuntimeError("AOT Inductor split payload decoding is not implemented for CompressAI VBR models yet.")
            if hasattr(model, "update"):
                _compressai_update_model(model, vbr_params)
            if not _is_compressai_factorized_model(model):
                _, hyper_synthesis_session = _aoti_hyper_sessions(bundle, torch, device)
                runtime_notes["shape_routed_hyperprior"] = True
                runtime_notes["available_hyper_shapes"] = [
                    package.get("hyper_shape")
                    for package in _aoti_shape_packages(bundle).values()
                    if package.get("hyper_shape")
                ]
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            entries = codec_payload.get("entries") or []
            if not entries:
                raise RuntimeError("CompressAI AOT Inductor split entropy payload did not contain entries")
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "payload decoding")
            with torch.no_grad():
                for index, entry in enumerate(entries):
                    total_start = time.perf_counter()
                    y_hat = _compressai_entropy_decode_latents_onnx(
                        torch,
                        model,
                        entry["strings"],
                        entry["shape"],
                        hyper_synthesis_session,
                        timing_records,
                        index,
                        entry.get("original_shape"),
                        device,
                    )
                    decoded.append(y_hat.detach().cpu().numpy().astype(np.float32, copy=False))
                    append_measurement(
                        timing_records,
                        index,
                        "decoder.total",
                        time.perf_counter() - total_start,
                        image_shape=entry.get("original_shape"),
                    )
                    _report_example_progress(ctx, index + 1, len(entries), "payload decoding")
            latents, latent_shapes = _stack_nchw_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError(_compressai_payload_decode_error("CompressAI AOT Inductor payload decode", exc)) from exc
            latent_shape = tuple(int(item) for item in metadata.get("shape", [1, 1, 1, 1]))
            latents = np.zeros(latent_shape, dtype=np.float32)

        output_metadata = dict(metadata)
        if "latent_shapes" in locals():
            output_metadata["latent_shapes"] = latent_shapes
        output_metadata.update(
            {
                "source": "compressai_aoti_entropy_decode",
                "semantic_form": "latents",
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "entropy_coded": True,
                "entropy_stage": "compressai_aoti_split_entropy_decode",
                "hyper_runtime": "aot_inductor" if hyper_synthesis_session is not None else "not_applicable",
                "byte_stream_data_plane_backend": byte_backend,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={
                "shape": list(latents.shape),
                "runtime": "aot_inductor",
                "hyper_runtime": output_metadata["hyper_runtime"],
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="aot_inductor+compressai_entropy",
                    notes={
                        **runtime_notes,
                        "decoder.symbol_decode": "CompressAI entropy decoder calls recover quantized latent symbols from transmitted bytes.",
                        "decoder.payload_model_inference": "CompressAI h_s hyperprior transform runs with AOT Inductor during payload decoding.",
                    },
                ),
            },
        )


class CompressAiAotInductorDecodeOperation(Operation):
    id = "model.compressai_aoti_decode"
    name = "CompressAI AOT Inductor synthesis transform to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("compressai", ["torch"], "AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, _zoo = _require_torch()
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device", "cpu"))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("data_plane_backend", "auto")))
        timing_records = []
        sessions: Dict[str, _AotInductorSession] = {}
        runtime_evidence: JsonDict = {
            "runtime": "aot_inductor",
            "shape_routed": True,
            "available_latent_shapes": [
                package.get("latent_shape")
                for package in _aoti_shape_packages(bundle).values()
                if package.get("latent_shape")
            ],
        }
        total = int(latents.shape[0])
        _report_example_progress(ctx, 0, total, "decoding")
        decoded = []
        try:
            for index in range(total):
                latent = _latent_for_index(latents, metadata, index)[0]
                total_start = time.perf_counter()
                tensor = timed_call(
                    timing_records,
                    index,
                    "decoder.preprocess",
                    lambda latent=latent: torch.from_numpy(
                        np.ascontiguousarray(latent[None, ...].astype(np.float32, copy=False))
                    ).to(device).contiguous(),
                    latent_shape=[int(item) for item in latent.shape],
                )
                package = _aoti_package_for_field_shape(bundle, "latent_shape", [int(item) for item in tensor.shape])
                session_path = _aoti_package_path(package, "synthesis_path")
                if session_path not in sessions:
                    setup_start = time.perf_counter()
                    session = _aoti_session_from_cache(sessions, torch, session_path, device)
                    append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
                    runtime_evidence["last_loaded_synthesis"] = _aoti_runtime_evidence(torch, session, session_path, device)
                else:
                    session = sessions[session_path]
                output_tensor = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda tensor=tensor: session.run_first_output_tensor(tensor),
                    latent_shape=[int(item) for item in latent.shape],
                )
                image_shape = _single_original_shape(metadata, index)
                decoded.append(
                    timed_call(
                        timing_records,
                        index,
                        "decoder.postprocess",
                        lambda output_tensor=output_tensor, image_shape=image_shape: _nchw_numpy_to_images(
                            session.tensor_to_numpy(output_tensor), image_shape, data_backend
                        )[0],
                        image_shape=image_shape,
                    )
                )
                append_measurement(
                    timing_records,
                    index,
                    "decoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                _report_example_progress(ctx, index + 1, total, "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("CompressAI AOT Inductor decode failed: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
        output_metadata = dict(metadata)
        if "decoded_shapes" in locals():
            output_metadata["original_shapes"] = decoded_shapes
        output_metadata.update(
            {
                "source": "compressai_aoti_decode",
                "shape": list(images.shape),
                "runtime": "aot_inductor",
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "nchw_to_image"),
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "runtime": "aot_inductor",
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata("decoder", timing_records, runner="aot_inductor", notes=runtime_evidence),
            },
        )


class CompressAiOpenVinoEncodeOperation(Operation):
    id = "model.compressai_openvino_encode"
    name = "CompressAI OpenVINO analysis transform to latents"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "pad_to_multiple": {"type": "integer", "default": 64, "minimum": 1},
            "quantize_latents": {"type": "boolean", "default": False},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "openvino",
            ["openvino"],
            "OpenVINO",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        timing_records = []
        setup_start = time.perf_counter()
        compiled = _openvino_compile_model(ov, bundle["analysis_path"], device)
        runtime_evidence = _openvino_runtime_evidence(ov, compiled, bundle["analysis_path"], device)
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        total = int(images.shape[0])
        _report_example_progress(ctx, 0, total, "encoding")
        latents = []
        padded_shapes = []
        quantize = bool(ctx.params.get("quantize_latents", False))
        for index in range(int(images.shape[0])):
            _report_example_active_progress(ctx, index, total, "encoding")
            image = _image_for_index(images, input_metadata, index)
            total_start = time.perf_counter()
            tensor = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: _images_to_nchw_numpy(image[None, ...], int(ctx.params.get("pad_to_multiple", 64)), data_backend),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            padded_shapes.append([int(item) for item in tensor.shape])
            output = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda tensor=tensor: _openvino_run_first_output(compiled, tensor),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            if quantize:
                output = np.round(output)
            latents.append(output.astype(np.float32, copy=False))
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            )
            _report_example_progress(ctx, index + 1, total, "encoding")
        latent_batch, latent_shapes = _stack_nchw_list(latents)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "compressai_openvino_encode",
                "codec": "compressai",
                "model": bundle.get("model"),
                "quality": bundle.get("quality"),
                "metric": bundle.get("metric"),
                "semantic_form": "latents",
                "shape": list(latent_batch.shape),
                "dtype": str(latent_batch.dtype),
                "original_shape": list(images.shape),
                "latent_shapes": latent_shapes,
                "padded_shapes": padded_shapes,
                "quantized": quantize,
                "runtime": "openvino",
                "device": device,
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "image_to_nchw"),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latent_batch, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, metadata)},
            metadata={
                "shape": list(latent_batch.shape),
                "runtime": "openvino",
                "device": device,
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="openvino",
                    notes=runtime_evidence,
                ),
            },
        )


class CompressAiOpenVinoEntropyEncodeOperation(Operation):
    id = "model.compressai_openvino_entropy_encode"
    name = "CompressAI OpenVINO payload encode latents to bits"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **COMPRESSAI_ENTROPY_INFO,
        "implementation": "OpenVINO hyperprior transforms + CompressAI entropy backend",
        "language": "OpenVINO + C++ extension + Python/PyTorch",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "torch_device": {"type": "string", "default": "cpu"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def __init__(self) -> None:
        self.catalog = load_model_catalog()

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "openvino",
            ["compressai", "torch", "openvino"],
            "CompressAI OpenVINO payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        ov = _require_openvino()
        latents, input_metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        setup_start = time.perf_counter()
        model_name = str(bundle.get("model") or input_metadata.get("model") or default_compressai_model(self.catalog))
        quality = int(bundle.get("quality") or input_metadata.get("quality") or default_compressai_quality(self.catalog))
        metric = str(bundle.get("metric") or input_metadata.get("metric") or default_compressai_metric(self.catalog))
        model = _compressai_model(zoo, model_name, quality, metric, bool(bundle.get("pretrained", input_metadata.get("pretrained", True))))
        torch_device = str(ctx.params.get("torch_device", "cpu"))
        model = model.to(torch_device).eval()
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("byte_stream_data_plane_backend", "auto")))
        vbr_params = _compressai_vbr_params(model, input_metadata)
        if vbr_params.get("enabled"):
            raise RuntimeError("OpenVINO split payload coding is not implemented for CompressAI VBR models yet.")
        if hasattr(model, "update"):
            _compressai_update_model(model, vbr_params)
        hyper_analysis_session = None
        hyper_synthesis_session = None
        runtime_notes: JsonDict = {"runtime": "openvino", "requested_device": device}
        if not _is_compressai_factorized_model(model):
            hyper_analysis_session, hyper_synthesis_session = _openvino_hyper_sessions(bundle, ov, device)
            runtime_notes["hyper_analysis_evidence"] = _openvino_runtime_evidence(
                ov, hyper_analysis_session, bundle["hyper_analysis_path"], device
            )
            runtime_notes["hyper_synthesis_evidence"] = _openvino_runtime_evidence(
                ov, hyper_synthesis_session, bundle["hyper_synthesis_path"], device
            )
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        total = int(latents.shape[0])
        entries = []
        _report_example_progress(ctx, 0, total, "payload encoding")
        with torch.no_grad():
            for index in range(total):
                total_start = time.perf_counter()
                image_shape = _single_original_shape(input_metadata, index)
                latent = _latent_for_index(latents, input_metadata, index)
                y_tensor = torch.from_numpy(latent.astype(np.float32, copy=False)).to(torch_device)
                compressed = _compressai_entropy_encode_latents_onnx(
                    torch,
                    model,
                    y_tensor,
                    latent,
                    hyper_analysis_session,
                    hyper_synthesis_session,
                    timing_records,
                    index,
                    image_shape,
                    torch_device,
                )
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                entries.append(
                    {
                        "strings": compressed["strings"],
                        "shape": compressed["shape"],
                        "original_shape": image_shape,
                        "padded_shape": _metadata_sequence_item(input_metadata, "padded_shapes", index),
                        "latent_shape": [int(item) for item in latent.shape],
                    }
                )
                _report_example_progress(ctx, index + 1, total, "payload encoding")

        hyper_runtime = "openvino" if hyper_analysis_session is not None else "not_applicable"
        codec_payload = {
            "entries": entries,
            "model": model_name,
            "quality": quality,
            "metric": metric,
            "pretrained": bool(bundle.get("pretrained", input_metadata.get("pretrained", True))),
            "vbr": vbr_params,
            "original_shape": list(input_metadata.get("original_shape") or input_metadata.get("shape") or []),
            "payload_version": 4,
            "split_entropy": True,
        }
        raw = safe_payload_dumps(codec_payload)
        bits, byte_backend = _bytes_to_bits(raw, data_backend)
        native_accounting = _native_codec_accounting(
            [_binary_payload_byte_count(entry["strings"]) for entry in entries],
            len(raw),
        )
        pixel_count = _source_pixel_count(input_metadata)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "compressai",
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "byte_count": len(raw),
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "payload_format": SAFE_PAYLOAD_FORMAT,
                "payload_bit_count": int(bits.size),
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": True,
                "entropy_stage": "compressai_openvino_split_entropy_encode",
                "hyper_runtime": hyper_runtime,
                "byte_stream_data_plane_backend": byte_backend,
                **native_accounting,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": len(raw),
                "channel.payload_bit_count": int(bits.size),
                **_native_codec_metrics(native_accounting, pixel_count),
            },
            metadata={
                "model": model_name,
                "quality": quality,
                "metric": metric,
                "runtime": "openvino",
                "hyper_runtime": hyper_runtime,
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="openvino+compressai_entropy",
                    notes={
                        **runtime_notes,
                        "encoder.payload_model_inference": "CompressAI h_a/h_s hyperprior transforms run with OpenVINO during payload encoding.",
                        "encoder.symbol_encode": "CompressAI entropy coder calls produce the transmitted bitstream.",
                    },
                ),
            },
        )


class CompressAiOpenVinoEntropyDecodeOperation(Operation):
    id = "model.compressai_openvino_entropy_decode"
    name = "CompressAI OpenVINO payload decode bits to latents"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    entropy_info = CompressAiOpenVinoEntropyEncodeOperation.entropy_info
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "torch_device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "openvino",
            ["compressai", "torch", "openvino"],
            "CompressAI OpenVINO payload coding",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, zoo = _require_compressai()
        ov = _require_openvino()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        hyper_synthesis_session = None
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
        byte_backend = "python_numpy"
        try:
            raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
            codec_payload = safe_payload_loads(raw)
            setup_start = time.perf_counter()
            model_name = str(codec_payload["model"])
            quality = int(codec_payload["quality"])
            metric = str(codec_payload["metric"])
            model = _compressai_model(zoo, model_name, quality, metric, bool(codec_payload["pretrained"]))
            torch_device = str(ctx.params.get("torch_device", "cpu"))
            model = model.to(torch_device).eval()
            vbr_params = codec_payload.get("vbr") or _compressai_vbr_params(model, {})
            if vbr_params.get("enabled"):
                raise RuntimeError("OpenVINO split payload decoding is not implemented for CompressAI VBR models yet.")
            if hasattr(model, "update"):
                _compressai_update_model(model, vbr_params)
            runtime_notes: JsonDict = {"runtime": "openvino", "requested_device": device}
            if not _is_compressai_factorized_model(model):
                _, hyper_synthesis_session = _openvino_hyper_sessions(bundle, ov, device)
                runtime_notes["hyper_synthesis_evidence"] = _openvino_runtime_evidence(
                    ov, hyper_synthesis_session, bundle["hyper_synthesis_path"], device
                )
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            entries = codec_payload.get("entries") or []
            if not entries:
                raise RuntimeError("CompressAI OpenVINO split entropy payload did not contain entries")
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "payload decoding")
            with torch.no_grad():
                for index, entry in enumerate(entries):
                    total_start = time.perf_counter()
                    y_hat = _compressai_entropy_decode_latents_onnx(
                        torch,
                        model,
                        entry["strings"],
                        entry["shape"],
                        hyper_synthesis_session,
                        timing_records,
                        index,
                        entry.get("original_shape"),
                        torch_device,
                    )
                    decoded.append(y_hat.detach().cpu().numpy().astype(np.float32, copy=False))
                    append_measurement(
                        timing_records,
                        index,
                        "decoder.total",
                        time.perf_counter() - total_start,
                        image_shape=entry.get("original_shape"),
                    )
                    _report_example_progress(ctx, index + 1, len(entries), "payload decoding")
            latents, latent_shapes = _stack_nchw_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError(_compressai_payload_decode_error("CompressAI OpenVINO payload decode", exc)) from exc
            latent_shape = tuple(int(item) for item in metadata.get("shape", [1, 1, 1, 1]))
            latents = np.zeros(latent_shape, dtype=np.float32)
            runtime_notes = {"runtime": "openvino", "requested_device": device}

        output_metadata = dict(metadata)
        if "latent_shapes" in locals():
            output_metadata["latent_shapes"] = latent_shapes
        output_metadata.update(
            {
                "source": "compressai_openvino_entropy_decode",
                "semantic_form": "latents",
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "entropy_coded": True,
                "entropy_stage": "compressai_openvino_split_entropy_decode",
                "hyper_runtime": "openvino" if hyper_synthesis_session is not None else "not_applicable",
                "byte_stream_data_plane_backend": byte_backend,
                **_entropy_metadata(self.entropy_info),
            }
        )
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={
                "shape": list(latents.shape),
                "runtime": "openvino",
                "hyper_runtime": output_metadata["hyper_runtime"],
                **_entropy_metadata(self.entropy_info),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="openvino+compressai_entropy",
                    notes={
                        **runtime_notes,
                        "decoder.symbol_decode": "CompressAI entropy decoder calls recover quantized latent symbols from transmitted bytes.",
                        "decoder.payload_model_inference": "CompressAI h_s hyperprior transform runs with OpenVINO during payload decoding.",
                    },
                ),
            },
        )


class CompressAiOpenVinoDecodeOperation(Operation):
    id = "model.compressai_openvino_decode"
    name = "CompressAI OpenVINO synthesis transform to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "openvino",
            ["openvino"],
            "OpenVINO",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        bundle = _load_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("data_plane_backend", "auto")))
        timing_records = []
        setup_start = time.perf_counter()
        compiled = _openvino_compile_model(ov, bundle["synthesis_path"], device)
        runtime_evidence = _openvino_runtime_evidence(ov, compiled, bundle["synthesis_path"], device)
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        total = int(latents.shape[0])
        _report_example_progress(ctx, 0, total, "decoding")
        decoded = []
        try:
            for index in range(total):
                latent = _latent_for_index(latents, metadata, index)[0]
                total_start = time.perf_counter()
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda latent=latent: _openvino_run_first_output(compiled, latent[None, ...].astype(np.float32, copy=False)),
                    latent_shape=[int(item) for item in latent.shape],
                )
                image_shape = _single_original_shape(metadata, index)
                decoded.append(_nchw_numpy_to_images(output, image_shape, data_backend)[0])
                append_measurement(
                    timing_records,
                    index,
                    "decoder.total",
                    time.perf_counter() - total_start,
                    image_shape=image_shape,
                )
                _report_example_progress(ctx, index + 1, total, "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("CompressAI OpenVINO decode failed: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
        output_metadata = dict(metadata)
        if "decoded_shapes" in locals():
            output_metadata["original_shapes"] = decoded_shapes
        output_metadata.update(
            {
                "source": "compressai_openvino_decode",
                "shape": list(images.shape),
                "runtime": "openvino",
                "device": device,
                "runtime_evidence": runtime_evidence,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": data_backend}, "nchw_to_image"),
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metadata={
                "shape": list(images.shape),
                "runtime": "openvino",
                "device": device,
                "runtime_evidence": runtime_evidence,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="openvino",
                    notes=runtime_evidence,
                ),
            },
        )


class DiffusersAutoencoderKlEncodeOperation(Operation):
    id = "model.diffusers_autoencoderkl_encode"
    name = "Diffusers AutoencoderKL image encoder to continuous latents"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        self.params_schema = object_schema(
            {
                "model_id": {
                    "type": "string",
                    "default": default_diffusers_model(self.catalog, "autoencoderkl", "stabilityai/sd-vae-ft-mse"),
                    "description": "Diffusers AutoencoderKL repository id used by both encoder and decoder.",
                },
                "revision": {
                    "type": "string",
                    "default": "",
                    "description": "Full 40-character model-repository commit; required for remote models and matched by the decoder.",
                },
                "device": {"type": "string", "default": "cpu"},
                "sample_mode": {
                    "type": "string",
                    "default": "mean",
                    "enum": ["mean", "sample"],
                    "description": "mean is deterministic; sample draws from the VAE latent distribution and can vary between runs.",
                },
                "scale_latents": {
                    "type": "boolean",
                    "default": True,
                    "description": "Applies the Stable Diffusion latent scale before transmission; decoder applies the inverse.",
                },
                "scaling_factor": {
                    "type": "number",
                    "default": 0.18215,
                    "description": "Latent scale factor shared by encoder and decoder. Keep this matched to the model config.",
                },
            },
            additional=True,
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "diffusers",
            ["diffusers", "torch"],
            "Diffusers",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, AutoencoderKL, _ = _require_diffusers()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        model_id = str(ctx.params.get("model_id") or "stabilityai/sd-vae-ft-mse")
        device = str(ctx.params.get("device", "cpu"))
        timing_records = []
        setup_start = time.perf_counter()
        load_kwargs = diffusers_load_kwargs(
            self.catalog, "autoencoderkl", model_id, ctx.params
        )
        model_revision = str(load_kwargs.get("revision") or "local_path")
        model = AutoencoderKL.from_pretrained(
            model_id,
            **load_kwargs,
        ).to(device).eval()
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
        chunks = []
        with torch.no_grad():
            for index in range(int(images.shape[0])):
                image = _image_for_index(images, input_metadata, index)
                total_start = time.perf_counter()
                image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
                tensor, _ = timed_call(
                    timing_records,
                    index,
                    "encoder.preprocess",
                    lambda image=image: _images_to_torch(torch, image[None, ...], device, 8, normalize=True),
                    image_shape=image_shape,
                )
                encoded = timed_call(
                    timing_records,
                    index,
                    "encoder.model",
                    lambda tensor=tensor: model.encode(tensor),
                    image_shape=image_shape,
                )
                latents = timed_call(
                    timing_records,
                    index,
                    "encoder.postprocess",
                    lambda encoded=encoded: _autoencoderkl_latents_from_encoded(
                        encoded,
                        model,
                        str(ctx.params.get("sample_mode", "mean")),
                        bool(ctx.params.get("scale_latents", True)),
                        float(ctx.params.get("scaling_factor", getattr(model.config, "scaling_factor", 0.18215))),
                    ),
                    image_shape=image_shape,
                )
                chunks.append(latents)
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                )
                _report_example_progress(ctx, index + 1, int(images.shape[0]), "encoding")
            latents = torch.cat(chunks, dim=0)
        return _write_latents(
            ctx,
            _torch_to_numpy(torch, latents),
            input_metadata,
            "diffusers_autoencoderkl",
            model_id,
            {**ctx.params, "model_revision": model_revision},
            codec_timing_metadata("encoder", timing_records, runner="pytorch"),
        )


class DiffusersAutoencoderKlDecodeOperation(Operation):
    id = "model.diffusers_autoencoderkl_decode"
    name = "Diffusers AutoencoderKL continuous latents decoder to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        self.params_schema = object_schema(
            {
                "model_id": {
                    "type": "string",
                    "default": default_diffusers_model(self.catalog, "autoencoderkl", "stabilityai/sd-vae-ft-mse"),
                    "description": "Matched to the encoder model id by the dashboard.",
                },
                "revision": {
                    "type": "string",
                    "default": "",
                    "description": "Full 40-character model-repository commit; required for remote models and matched to encoder evidence.",
                },
                "device": {"type": "string", "default": "cpu"},
                "scale_latents": {
                    "type": "boolean",
                    "default": True,
                    "description": "Inverse of the encoder latent scaling.",
                },
                "scaling_factor": {
                    "type": "number",
                    "default": 0.18215,
                    "description": "Matched to the encoder scaling factor by the dashboard.",
                },
            },
            additional=True,
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "diffusers",
            ["diffusers", "torch"],
            "Diffusers",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, AutoencoderKL, _ = _require_diffusers()
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        model_id = str(ctx.params.get("model_id") or metadata.get("model_id") or "stabilityai/sd-vae-ft-mse")
        device = str(ctx.params.get("device", "cpu"))
        timing_records = []
        setup_start = time.perf_counter()
        declared_revision = str(metadata.get("model_revision") or "")
        load_params = dict(ctx.params)
        if (
            not load_params.get("revision")
            and declared_revision
            and declared_revision != "local_path"
        ):
            load_params["revision"] = declared_revision
        load_kwargs = diffusers_load_kwargs(
            self.catalog, "autoencoderkl", model_id, load_params
        )
        model_revision = str(load_kwargs.get("revision") or "local_path")
        if declared_revision and declared_revision != model_revision:
            raise RuntimeError(
                "Diffusers decoder model revision `%s` does not match encoder revision `%s`"
                % (model_revision, declared_revision)
            )
        model = AutoencoderKL.from_pretrained(
            model_id,
            **load_kwargs,
        ).to(device).eval()
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        scaling_factor = float(ctx.params.get("scaling_factor", metadata.get("scaling_factor", 0.18215)))
        scale_latents = bool(ctx.params.get("scale_latents", metadata.get("scale_latents", True)))
        _report_example_progress(ctx, 0, int(latents.shape[0]), "decoding")
        chunks = []
        with torch.no_grad():
            for index in range(int(latents.shape[0])):
                total_start = time.perf_counter()
                tensor = timed_call(
                    timing_records,
                    index,
                    "decoder.preprocess",
                    lambda index=index: _diffusers_latent_to_torch(torch, latents[index : index + 1], device, scale_latents, scaling_factor),
                )
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda tensor=tensor: model.decode(tensor).sample,
                )
                chunks.append(
                    timed_call(
                        timing_records,
                        index,
                        "decoder.postprocess",
                        lambda output=output: _decoded_torch_to_images(torch, output),
                    )
                )
                append_measurement(timing_records, index, "decoder.total", time.perf_counter() - total_start)
                _report_example_progress(ctx, index + 1, int(latents.shape[0]), "decoding")
        images = np.concatenate(chunks, axis=0)
        return _write_images(
            ctx,
            images,
            metadata,
            "diffusers_autoencoderkl_decode",
            codec_timing_metadata("decoder", timing_records, runner="pytorch"),
        )


class DiffusersVqModelEncodeOperation(Operation):
    id = "model.diffusers_vqmodel_encode"
    name = "Diffusers VQModel image encoder to quantized semantic latents"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        self.params_schema = object_schema(
            {
                "model_id": {
                    "type": "string",
                    "default": default_diffusers_model(self.catalog, "vqmodel", "CompVis/ldm-celebahq-256"),
                },
                "revision": {
                    "type": "string",
                    "default": "",
                    "description": "Full 40-character model-repository commit; required for remote models and matched by the decoder.",
                },
                "device": {"type": "string", "default": "cpu"},
            },
            additional=True,
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "diffusers",
            ["diffusers", "torch"],
            "Diffusers",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, _, VQModel = _require_diffusers()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        model_id = str(ctx.params.get("model_id") or "CompVis/ldm-celebahq-256")
        device = str(ctx.params.get("device", "cpu"))
        timing_records = []
        setup_start = time.perf_counter()
        load_kwargs = diffusers_load_kwargs(
            self.catalog, "vqmodel", model_id, ctx.params
        )
        model_revision = str(load_kwargs.get("revision") or "local_path")
        model = VQModel.from_pretrained(
            model_id,
            **load_kwargs,
        ).to(device).eval()
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
        chunks = []
        with torch.no_grad():
            for index in range(int(images.shape[0])):
                image = _image_for_index(images, input_metadata, index)
                total_start = time.perf_counter()
                image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
                tensor, _ = timed_call(
                    timing_records,
                    index,
                    "encoder.preprocess",
                    lambda image=image: _images_to_torch(torch, image[None, ...], device, 8, normalize=True),
                    image_shape=image_shape,
                )
                encoded = timed_call(
                    timing_records,
                    index,
                    "encoder.model",
                    lambda tensor=tensor: model.encode(tensor),
                    image_shape=image_shape,
                )
                chunks.append(
                    timed_call(
                        timing_records,
                        index,
                        "encoder.postprocess",
                        lambda encoded=encoded: _diffusers_encoded_tensor(encoded),
                        image_shape=image_shape,
                    )
                )
                append_measurement(
                    timing_records,
                    index,
                    "encoder.total",
                    time.perf_counter() - total_start,
                    image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                )
                _report_example_progress(ctx, index + 1, int(images.shape[0]), "encoding")
            latents = torch.cat(chunks, dim=0)
        return _write_latents(
            ctx,
            _torch_to_numpy(torch, latents),
            input_metadata,
            "diffusers_vqmodel",
            model_id,
            {**ctx.params, "model_revision": model_revision},
            codec_timing_metadata("encoder", timing_records, runner="pytorch"),
        )


class DiffusersVqModelDecodeOperation(Operation):
    id = "model.diffusers_vqmodel_decode"
    name = "Diffusers VQModel quantized semantic latents decoder to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}

    def __init__(self) -> None:
        self.catalog = load_model_catalog()
        self.params_schema = object_schema(
            {
                "model_id": {
                    "type": "string",
                    "default": default_diffusers_model(self.catalog, "vqmodel", "CompVis/ldm-celebahq-256"),
                },
                "revision": {
                    "type": "string",
                    "default": "",
                    "description": "Full 40-character model-repository commit; required for remote models and matched to encoder evidence.",
                },
                "device": {"type": "string", "default": "cpu"},
            },
            additional=True,
        )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "diffusers",
            ["diffusers", "torch"],
            "Diffusers",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch, _, VQModel = _require_diffusers()
        latents, metadata = _load_latents(ctx.require_input("latents").path, ctx.require_input("latents").metadata)
        model_id = str(ctx.params.get("model_id") or metadata.get("model_id") or "CompVis/ldm-celebahq-256")
        device = str(ctx.params.get("device", "cpu"))
        timing_records = []
        setup_start = time.perf_counter()
        declared_revision = str(metadata.get("model_revision") or "")
        load_params = dict(ctx.params)
        if (
            not load_params.get("revision")
            and declared_revision
            and declared_revision != "local_path"
        ):
            load_params["revision"] = declared_revision
        load_kwargs = diffusers_load_kwargs(
            self.catalog, "vqmodel", model_id, load_params
        )
        model_revision = str(load_kwargs.get("revision") or "local_path")
        if declared_revision and declared_revision != model_revision:
            raise RuntimeError(
                "Diffusers decoder model revision `%s` does not match encoder revision `%s`"
                % (model_revision, declared_revision)
            )
        model = VQModel.from_pretrained(
            model_id,
            **load_kwargs,
        ).to(device).eval()
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        _report_example_progress(ctx, 0, int(latents.shape[0]), "decoding")
        chunks = []
        with torch.no_grad():
            for index in range(int(latents.shape[0])):
                total_start = time.perf_counter()
                tensor = timed_call(
                    timing_records,
                    index,
                    "decoder.preprocess",
                    lambda index=index: torch.from_numpy(latents[index : index + 1]).to(device),
                )
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda tensor=tensor: model.decode(tensor).sample,
                )
                chunks.append(
                    timed_call(
                        timing_records,
                        index,
                        "decoder.postprocess",
                        lambda output=output: _decoded_torch_to_images(torch, output),
                    )
                )
                append_measurement(timing_records, index, "decoder.total", time.perf_counter() - total_start)
                _report_example_progress(ctx, index + 1, int(latents.shape[0]), "decoding")
        images = np.concatenate(chunks, axis=0)
        return _write_images(
            ctx,
            images,
            metadata,
            "diffusers_vqmodel_decode",
            codec_timing_metadata("decoder", timing_records, runner="pytorch"),
        )


def _require_compressai():
    try:
        import torch
        _assert_torch_runtime(torch)
        import compressai.zoo as zoo
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[compressai]"` in an '
            "installed environment, or `uv sync --extra compressai` in a source "
            "checkout, to use CompressAI adapters. "
            "The current environment does not expose a usable PyTorch runtime."
        ) from exc
    return torch, zoo


def _require_torch():
    try:
        import torch
        _assert_torch_runtime(torch)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[compressai]"` in an '
            "installed environment, or `uv sync --extra compressai` in a source "
            "checkout, to use PyTorch runtime adapters. "
            "The current environment does not expose a usable PyTorch runtime."
        ) from exc
    return torch, None


def _require_diffusers():
    try:
        import torch
        _assert_torch_runtime(torch)
        from diffusers import AutoencoderKL, VQModel
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[diffusers]"` in an '
            "installed environment, or `uv sync --extra diffusers` in a source "
            "checkout, to use Diffusers adapters. "
            "The current environment does not expose a usable PyTorch runtime."
        ) from exc
    return torch, AutoencoderKL, VQModel


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `uv sync` to use JPEG codec adapters") from exc
    return Image


def _pillow_availability() -> JsonDict:
    if importlib.util.find_spec("PIL") is not None:
        return {"available": True, "extra": "", "missing": []}
    return {
        "available": False,
        "extra": "",
        "missing": ["PIL"],
        "reason": "Install project dependencies with `uv sync` to use JPEG codec adapters",
    }


def _jpeg_subsampling_value(value: str):
    return {
        "444": 0,
        "422": 1,
        "420": 2,
        "keep": None,
    }.get(str(value), 2)


def _optional_dependency_availability(extra: str, modules, label: str) -> JsonDict:
    missing = []
    broken = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
            continue
        if module == "torch":
            try:
                imported = importlib.import_module("torch")
                _assert_torch_runtime(imported)
            except Exception as exc:  # pragma: no cover - depends on optional dependency state
                broken.append("torch: %s" % exc)
    if not missing:
        if not broken:
            return {"available": True, "extra": extra, "missing": []}
    return {
        "available": False,
        "extra": extra,
        "missing": missing,
        "broken": broken,
        "reason": _optional_dependency_reason(extra, label, broken),
    }


def _assert_torch_runtime(torch_module) -> None:
    if not (hasattr(torch_module, "Tensor") and hasattr(torch_module, "nn") and hasattr(torch_module.nn, "Module")):
        raise AttributeError("PyTorch import resolved to a namespace/stub package, not a usable torch runtime")


def _optional_dependency_reason(extra: str, label: str, broken: List[str]) -> str:
    base = (
        'Install with `python -m pip install "noema-lab[%s]"` in an installed '
        "environment, or `uv sync --extra %s` in a source checkout, to use %s adapters"
        % (extra, extra, label)
    )
    if broken:
        return "%s. Broken optional dependency: %s" % (base, "; ".join(broken))
    return base


def _require_onnx_stack(require_onnx: bool = True):
    onnx = None
    if require_onnx:
        try:
            import onnx as onnx_module
        except ImportError as exc:
            raise RuntimeError(
                'Install with `python -m pip install "noema-lab[onnx]"` in an '
                "installed environment, or `uv sync --extra onnx` in a source "
                "checkout, to export models to ONNX"
            ) from exc
        onnx = onnx_module
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[onnx]"` in an '
            "installed environment, or `uv sync --extra onnx` in a source "
            "checkout, to run ONNX Runtime recipes"
        ) from exc
    return onnx, ort


def _require_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[openvino]"` in an '
            "installed environment, or `uv sync --extra openvino` in a source "
            "checkout, to run OpenVINO recipes"
        ) from exc
    return ov


def _onnx_providers(requested: str, ort) -> list:
    provider = str(requested or "CPUExecutionProvider")
    available = list(ort.get_available_providers())
    if provider not in available:
        raise RuntimeError(
            "ONNX Runtime provider %s is not available. Available providers: %s"
            % (provider, ", ".join(available) or "none")
        )
    return [provider]


def _openvino_device(requested: str) -> str:
    text = str(requested or "CPU").strip() or "CPU"
    return text.upper()


def _assert_onnx_session_provider(session, expected_providers: list, role: str) -> None:
    active = list(session.get_providers())
    if active != list(expected_providers):
        raise RuntimeError(
            "ONNX Runtime %s session provider mismatch. Requested %s but session uses %s. "
            "Refusing to run with an implicit provider fallback."
            % (role, expected_providers, active)
        )


def _openvino_compile_model(ov, model_path, device: str):
    path = Path(str(model_path))
    if not path.is_file():
        raise RuntimeError("OpenVINO model path does not exist: %s" % path)
    core = ov.Core()
    available = {str(item).upper() for item in core.available_devices}
    requested = _openvino_device(device)
    if requested not in available:
        raise RuntimeError(
            "OpenVINO device %s is not available. Available devices: %s"
            % (requested, ", ".join(sorted(available)) or "none")
        )
    return core.compile_model(str(path), requested)


def _onnx_runtime_evidence(ort, session, model_path, requested_provider: str) -> JsonDict:
    model = Path(str(model_path))
    return {
        "runtime": "onnxruntime",
        "onnxruntime_version": str(getattr(ort, "__version__", "")),
        **onnxruntime_native_evidence(ort),
        "session_class": "%s.%s" % (session.__class__.__module__, session.__class__.__name__),
        "requested_provider": requested_provider,
        "active_providers": list(session.get_providers()),
        "model_path": str(model),
        "model_sha256": file_sha256(model) if model.is_file() else "",
        "session_inputs": [_onnx_value_info(item) for item in session.get_inputs()],
        "session_outputs": [_onnx_value_info(item) for item in session.get_outputs()],
    }


def _create_onnx_session_for_runtime(
    runtime: str,
    ort,
    model_path,
    providers: list,
    ctx: OperationContext,
    role: str,
):
    del role
    provider = providers[0] if providers else "CPUExecutionProvider"
    if runtime == "onnxruntime_cpp":
        return CppOnnxSession(
            model_path,
            provider=provider,
            library_path=str(ctx.params.get("library_path") or "") or None,
            intra_op_num_threads=int(ctx.params.get("intra_op_num_threads") or 0),
        )
    return ort.InferenceSession(str(model_path), providers=providers)


def _onnx_session_evidence_for_runtime(
    runtime: str,
    ort,
    session,
    model_path,
    requested_provider: str,
) -> JsonDict:
    if runtime == "onnxruntime_cpp":
        return cpp_runtime_evidence(session, model_path, requested_provider)
    return _onnx_runtime_evidence(ort, session, model_path, requested_provider)


def _openvino_runtime_evidence(ov, compiled_model, model_path, requested_device: str) -> JsonDict:
    model = Path(str(model_path))
    inputs = [_openvino_value_info(item) for item in list(compiled_model.inputs)]
    outputs = [_openvino_value_info(item) for item in list(compiled_model.outputs)]
    return {
        "runtime": "openvino",
        "openvino_version": str(getattr(ov, "__version__", "")),
        "compiled_model_class": "%s.%s" % (compiled_model.__class__.__module__, compiled_model.__class__.__name__),
        "requested_device": _openvino_device(requested_device),
        "model_path": str(model),
        "model_sha256": file_sha256(model) if model.is_file() else "",
        "session_inputs": inputs,
        "session_outputs": outputs,
    }


def _onnx_value_info(value) -> JsonDict:
    return {
        "name": str(getattr(value, "name", "")),
        "type": str(getattr(value, "type", "")),
        "shape": [str(item) for item in (getattr(value, "shape", None) or [])],
    }


def _openvino_value_info(value) -> JsonDict:
    try:
        shape = list(value.partial_shape)
    except Exception:
        shape = []
    try:
        dtype = value.element_type
    except Exception:
        dtype = ""
    return {
        "name": str(getattr(value, "any_name", "")),
        "type": str(dtype),
        "shape": [str(item) for item in shape],
    }


class _AotInductorSession:
    def __init__(self, torch, model_path, device: str) -> None:
        _ensure_aoti_available(torch)
        path = Path(str(model_path))
        if not path.is_file():
            raise RuntimeError("AOT Inductor package path does not exist: %s" % path)
        self.torch = torch
        self.path = path
        self.device = str(device or "cpu")
        self.device_index = _aoti_device_index(self.device)
        self.module = _aoti_load_package_compat(torch, path, self.device_index)

    def run_first_output(self, array: np.ndarray) -> np.ndarray:
        tensor = self.array_to_tensor(array)
        output = self.run_first_output_tensor(tensor)
        return self.tensor_to_numpy(output)

    def array_to_tensor(self, array: np.ndarray):
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        return self.torch.from_numpy(contiguous).to(self.device).contiguous()

    def run_first_output_tensor(self, tensor):
        with self.torch.no_grad():
            output = self.module(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output

    def tensor_to_numpy(self, tensor) -> np.ndarray:
        return _torch_to_numpy(self.torch, tensor).astype(np.float32, copy=False)


class _AotInductorShapeRouter:
    def __init__(self, torch, bundle: JsonDict, device: str, shape_field: str, path_field: str) -> None:
        self.torch = torch
        self.bundle = bundle
        self.device = str(device or "cpu")
        self.shape_field = shape_field
        self.path_field = path_field
        self.cache: Dict[str, _AotInductorSession] = {}

    def run_first_output(self, array: np.ndarray) -> np.ndarray:
        package = _aoti_package_for_field_shape(self.bundle, self.shape_field, np.asarray(array).shape)
        path = _aoti_package_path(package, self.path_field)
        session = _aoti_session_from_cache(self.cache, self.torch, path, self.device)
        return session.run_first_output(array)


def _save_aoti_package(torch, module, example, path, label: str, allow_dynamic_shapes: bool = False) -> str:
    _ensure_aoti_available(torch)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dynamic_shapes = _aoti_dynamic_hw_shapes(torch, example) if allow_dynamic_shapes else None

    def compile_with_shapes(shape_spec):
        if target.exists():
            target.unlink()
        exported = torch.export.export(
            module.eval(),
            (example,),
            dynamic_shapes=shape_spec,
            strict=False,
        )
        torch._inductor.aoti_compile_and_package(exported, package_path=str(target))

    try:
        compile_with_shapes(dynamic_shapes)
        return "dynamic" if dynamic_shapes is not None else "static"
    except Exception as exc:
        if dynamic_shapes is not None:
            try:
                compile_with_shapes(None)
                return "static"
            except Exception as static_exc:
                raise RuntimeError(
                    "could not compile %s module with AOT Inductor: dynamic export failed: %s; static fallback failed: %s"
                    % (label, exc, static_exc)
                ) from static_exc
        raise RuntimeError("could not compile %s module with AOT Inductor: %s" % (label, exc)) from exc


def _aoti_export_sample_arrays(
    images: np.ndarray,
    metadata: JsonDict,
    multiple: int,
    requested_height: int = 0,
    requested_width: int = 0,
) -> JsonDict:
    samples: JsonDict = {}
    for index in range(int(images.shape[0])):
        image = _image_for_index(images, metadata, index)
        array = _images_to_nchw_numpy(image[None, ...], int(multiple))
        array = _pad_nchw_numpy_to_min_shape(array, requested_height, requested_width, int(multiple))
        key = _aoti_shape_key(array.shape)
        if key not in samples:
            samples[key] = array
    return samples


def _pad_nchw_numpy_to_min_shape(array: np.ndarray, requested_height: int, requested_width: int, multiple: int) -> np.ndarray:
    tensor = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    height = int(tensor.shape[2])
    width = int(tensor.shape[3])
    target_height = max(height, int(requested_height or 0))
    target_width = max(width, int(requested_width or 0))
    multiple = max(1, int(multiple))
    target_height += (multiple - target_height % multiple) % multiple
    target_width += (multiple - target_width % multiple) % multiple
    pad_h = max(0, target_height - height)
    pad_w = max(0, target_width - width)
    if pad_h or pad_w:
        tensor = np.pad(tensor, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return np.ascontiguousarray(tensor)


def _aoti_shape_key(shape) -> str:
    return "x".join(str(int(item)) for item in list(shape))


def _aoti_default_package(bundle: JsonDict) -> JsonDict:
    return {
        "input_shape": list(bundle.get("export_input_shape") or []),
        "latent_shape": list(bundle.get("latent_shape") or []),
        "hyper_shape": list(bundle.get("hyper_shape") or []),
        "analysis_path": str(bundle.get("analysis_path") or ""),
        "synthesis_path": str(bundle.get("synthesis_path") or ""),
        "hyper_analysis_path": str(bundle.get("hyper_analysis_path") or ""),
        "hyper_synthesis_path": str(bundle.get("hyper_synthesis_path") or ""),
    }


def _aoti_shape_packages(bundle: JsonDict) -> JsonDict:
    packages: JsonDict = {}
    raw = bundle.get("shape_packages")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                packages[str(key)] = dict(value)
    default_package = _aoti_default_package(bundle)
    if default_package.get("input_shape"):
        packages.setdefault(_aoti_shape_key(default_package["input_shape"]), default_package)
    return packages


def _aoti_package_for_field_shape(bundle: JsonDict, field: str, shape) -> JsonDict:
    key = _aoti_shape_key(shape)
    packages = _aoti_shape_packages(bundle)
    if field == "input_shape" and key in packages:
        return packages[key]
    for package in packages.values():
        value = package.get(field) or []
        if value and _aoti_shape_key(value) == key:
            return package
    raise RuntimeError(
        "AOT Inductor bundle does not include a package for %s=%s. "
        "Rerun codec_export so it compiles packages for the selected image shapes."
        % (field, key)
    )


def _aoti_package_path(package: JsonDict, field: str) -> str:
    path = str(package.get(field) or "")
    if not path:
        raise RuntimeError("AOT Inductor package is missing %s" % field)
    if not Path(path).is_file():
        raise RuntimeError("AOT Inductor package path for %s does not exist: %s" % (field, path))
    return path


def _aoti_session_from_cache(cache: Dict[str, _AotInductorSession], torch, path: str, device: str) -> _AotInductorSession:
    key = str(path)
    if key not in cache:
        cache[key] = _AotInductorSession(torch, key, device)
    return cache[key]


def _assert_runtime_output_close(label: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise RuntimeError(
            "%s output shape mismatch: got %s, expected %s"
            % (label, list(actual_array.shape), list(expected_array.shape))
        )
    if not np.isfinite(actual_array).all():
        raise RuntimeError("%s output contains non-finite values" % label)
    if np.allclose(actual_array, expected_array, atol=atol, rtol=rtol):
        return
    diff = np.abs(actual_array - expected_array)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    raise RuntimeError(
        "%s output does not match the PyTorch reference within atol=%g rtol=%g "
        "(max_abs=%g, mean_abs=%g)"
        % (label, atol, rtol, max_abs, mean_abs)
    )


def _ensure_aoti_available(torch) -> None:
    inductor = getattr(torch, "_inductor", None)
    if inductor is None or not callable(getattr(inductor, "aoti_compile_and_package", None)):
        raise RuntimeError("PyTorch AOT Inductor package export requires torch._inductor.aoti_compile_and_package")
    if not callable(getattr(inductor, "aoti_load_package", None)):
        raise RuntimeError("PyTorch AOT Inductor package runtime requires torch._inductor.aoti_load_package")


def _aoti_load_package_compat(torch, path, device_index: int):
    loader = torch._inductor.aoti_load_package
    try:
        return loader(str(path), device_index=int(device_index))
    except TypeError as exc:
        message = str(exc)
        if "device_index" not in message and "unexpected keyword" not in message:
            raise
        return loader(str(path))


def _aoti_runtime_evidence(torch, session: _AotInductorSession, model_path, device: str) -> JsonDict:
    model = Path(str(model_path))
    return {
        "runtime": "aot_inductor",
        "torch_version": str(getattr(torch, "__version__", "")),
        "module_class": "%s.%s" % (session.module.__class__.__module__, session.module.__class__.__name__),
        "device": str(device or "cpu"),
        "device_index": int(session.device_index),
        "model_path": str(model),
        "model_sha256": file_sha256(model) if model.is_file() else "",
        "artifact": "torch._inductor AOTI .pt2 package",
    }


def _aoti_dynamic_hw_shapes(torch, example):
    if not hasattr(example, "ndim") or int(example.ndim) != 4:
        return None
    height = int(example.shape[2])
    width = int(example.shape[3])
    min_height = max(1, height)
    min_width = max(1, width)
    max_height = max(4096, height)
    max_width = max(4096, width)
    return (
        {
            2: torch.export.Dim("height", min=min_height, max=max_height),
            3: torch.export.Dim("width", min=min_width, max=max_width),
        },
    )


def _aoti_device_index(device: str) -> int:
    text = str(device or "cpu").lower()
    if text == "cpu" or text.startswith("cpu:"):
        return -1
    if ":" in text:
        try:
            return int(text.rsplit(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _load_onnx_bundle(input_artifact) -> JsonDict:
    try:
        payload = decode_strict_yaml_or_json(
            Path(input_artifact.path).read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("bundle manifest root must be a JSON object")
    except Exception as exc:
        raise RuntimeError("Could not read ONNX bundle manifest at %s: %s" % (input_artifact.path, exc)) from exc
    payload = dict(payload)
    for key in ("analysis_path", "synthesis_path"):
        path = payload.get(key)
        if not path:
            raise RuntimeError("ONNX bundle manifest is missing %s" % key)
        if not Path(str(path)).is_file():
            raise RuntimeError("ONNX bundle manifest points to missing %s: %s" % (key, path))
    return payload


def _load_aoti_bundle(input_artifact) -> JsonDict:
    try:
        payload = decode_strict_yaml_or_json(
            Path(input_artifact.path).read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("bundle manifest root must be a JSON object")
    except Exception as exc:
        raise RuntimeError("Could not read AOT Inductor bundle manifest at %s: %s" % (input_artifact.path, exc)) from exc
    payload = dict(payload)
    for key in ("analysis_path", "synthesis_path"):
        path = payload.get(key)
        if not path:
            raise RuntimeError("AOT Inductor bundle manifest is missing %s" % key)
        if not Path(str(path)).is_file():
            raise RuntimeError("AOT Inductor bundle manifest points to missing %s: %s" % (key, path))
    return payload


def _model_artifact_sizes(encoder_paths, decoder_paths) -> JsonDict:
    encoder = _sum_existing_file_sizes(encoder_paths)
    decoder = _sum_existing_file_sizes(decoder_paths)
    total = _sum_existing_file_sizes(list(encoder_paths or []) + list(decoder_paths or []))
    return {
        "encoder_model_bytes": int(encoder),
        "decoder_model_bytes": int(decoder),
        "model_artifact_bytes": int(total),
    }


def _sum_existing_file_sizes(paths) -> int:
    seen = set()
    total = 0
    for value in paths or []:
        if not value:
            continue
        path = Path(str(value))
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        total += int(path.stat().st_size)
    return total


def _onnx_hyper_sessions(bundle: JsonDict, ort, providers: list, runtime: str = "onnxruntime", ctx: OperationContext | None = None):
    hyper_analysis_path = str(bundle.get("hyper_analysis_path") or "")
    hyper_synthesis_path = str(bundle.get("hyper_synthesis_path") or "")
    if not hyper_analysis_path or not hyper_synthesis_path:
        raise RuntimeError(
            "ONNX bundle does not include h_a/h_s exports. Rebuild the recipe so codec_export creates hyper_analysis.onnx and hyper_synthesis.onnx."
        )
    if not Path(hyper_analysis_path).is_file() or not Path(hyper_synthesis_path).is_file():
        raise RuntimeError("ONNX bundle h_a/h_s paths are missing on disk")
    if ctx is None:
        raise RuntimeError("ONNX runtime session creation requires an operation context")
    hyper_analysis = _create_onnx_session_for_runtime(runtime, ort, hyper_analysis_path, providers, ctx, "hyper-analysis")
    hyper_synthesis = _create_onnx_session_for_runtime(runtime, ort, hyper_synthesis_path, providers, ctx, "hyper-synthesis")
    _assert_onnx_session_provider(hyper_analysis, providers, "hyper-analysis")
    _assert_onnx_session_provider(hyper_synthesis, providers, "hyper-synthesis")
    return hyper_analysis, hyper_synthesis


def _openvino_hyper_sessions(bundle: JsonDict, ov, device: str):
    hyper_analysis_path = str(bundle.get("hyper_analysis_path") or "")
    hyper_synthesis_path = str(bundle.get("hyper_synthesis_path") or "")
    if not hyper_analysis_path or not hyper_synthesis_path:
        raise RuntimeError(
            "ONNX bundle does not include h_a/h_s exports. Rebuild the recipe so codec_export creates hyper_analysis.onnx and hyper_synthesis.onnx."
        )
    if not Path(hyper_analysis_path).is_file() or not Path(hyper_synthesis_path).is_file():
        raise RuntimeError("ONNX bundle h_a/h_s paths are missing on disk")
    return (
        _openvino_compile_model(ov, hyper_analysis_path, device),
        _openvino_compile_model(ov, hyper_synthesis_path, device),
    )


def _aoti_hyper_sessions(bundle: JsonDict, torch, device: str):
    packages = _aoti_shape_packages(bundle)
    if not packages:
        raise RuntimeError(
            "AOT Inductor bundle does not include h_a/h_s exports. Rebuild the recipe so codec_export creates hyper_analysis.pt2 and hyper_synthesis.pt2."
        )
    for package in packages.values():
        hyper_analysis_path = str(package.get("hyper_analysis_path") or "")
        hyper_synthesis_path = str(package.get("hyper_synthesis_path") or "")
        if not hyper_analysis_path or not hyper_synthesis_path:
            raise RuntimeError(
                "AOT Inductor bundle does not include h_a/h_s exports. Rebuild the recipe so codec_export creates hyper_analysis.pt2 and hyper_synthesis.pt2."
            )
        if not Path(hyper_analysis_path).is_file() or not Path(hyper_synthesis_path).is_file():
            raise RuntimeError("AOT Inductor bundle h_a/h_s paths are missing on disk")
    return (
        _AotInductorShapeRouter(torch, bundle, device, "latent_shape", "hyper_analysis_path"),
        _AotInductorShapeRouter(torch, bundle, device, "hyper_shape", "hyper_synthesis_path"),
    )


def _onnx_run_first_output(session, array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    input_array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return session.run(None, {input_name: input_array})[0].astype(np.float32, copy=False)


def _model_runtime_run_first_output(session, array: np.ndarray) -> np.ndarray:
    if hasattr(session, "run_first_output") and callable(session.run_first_output):
        return session.run_first_output(array)
    if hasattr(session, "get_inputs") and callable(session.get_inputs):
        return _onnx_run_first_output(session, array)
    return _openvino_run_first_output(session, array)


def _timed_model_runtime_run_first_output(
    timing_records,
    example_index: int,
    stage: str,
    session,
    array: np.ndarray,
    fields: JsonDict,
) -> np.ndarray:
    role = str(stage).split(".", 1)[0] or "encoder"
    if hasattr(session, "array_to_tensor") and callable(session.array_to_tensor):
        tensor = timed_call(
            timing_records,
            example_index,
            "%s.preprocess" % role,
            lambda: session.array_to_tensor(array),
            **fields,
        )
        output = timed_call(
            timing_records,
            example_index,
            stage,
            lambda: session.run_first_output_tensor(tensor),
            **fields,
        )
        return timed_call(
            timing_records,
            example_index,
            "%s.postprocess" % role,
            lambda: session.tensor_to_numpy(output),
            **fields,
        )
    return timed_call(
        timing_records,
        example_index,
        stage,
        lambda: _model_runtime_run_first_output(session, array),
        **fields,
    )


def _openvino_run_first_output(compiled_model, array: np.ndarray) -> np.ndarray:
    input_port = compiled_model.inputs[0]
    output_port = compiled_model.outputs[0]
    input_array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    result = compiled_model({input_port: input_array})
    return np.asarray(result[output_port]).astype(np.float32, copy=False)


def _compressai_model(zoo, model_name: str, quality: int, metric: str, pretrained: bool):
    if not hasattr(zoo, model_name):
        raise RuntimeError("CompressAI model is not available in this installed version: %s" % model_name)
    return getattr(zoo, model_name)(quality=quality, metric=metric, pretrained=pretrained)


def _torch_module_state_sha256(torch, model) -> str:
    """Hash a materialized Torch module state without serialization metadata."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": [int(value) for value in tensor.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _compressai_vbr_params(model, params: JsonDict) -> JsonDict:
    if not _is_compressai_vbr_model(model):
        return {"enabled": False}
    levels = int(getattr(model, "levels", len(getattr(model, "Gain", [])) or 8))
    scale_index = int(params.get("vbr_scale_index", 1))
    scale_index = max(0, min(scale_index, max(0, levels - 1)))
    stage = int(params.get("vbr_stage", 2))
    stage = 2 if stage != 1 else 1
    return {
        "enabled": True,
        "scale_index": scale_index,
        "stage": stage,
        "levels": levels,
    }


def _is_compressai_vbr_model(model) -> bool:
    return hasattr(model, "Gain") and hasattr(model, "levels")


def _compressai_update_model(model, vbr_params: JsonDict) -> None:
    if vbr_params.get("enabled"):
        model.update(force=True, scale=_compressai_vbr_scale(model, vbr_params))
    else:
        model.update(force=True)


@contextmanager
def _compressai_timing_scope(model, timing_records, example_index: int, role: str, image_shape, model_stage: str = ""):
    restorers = []
    fields = {"image_shape": image_shape}

    def patch_method(owner, name: str, stage: str) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                append_measurement(
                    timing_records,
                    example_index,
                    stage,
                    time.perf_counter() - start,
                    **fields,
                )

        setattr(owner, name, wrapped)
        restorers.append(lambda owner=owner, name=name, original=original: setattr(owner, name, original))

    def patch_forward(module, stage: str) -> None:
        original = getattr(module, "forward", None)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                append_measurement(
                    timing_records,
                    example_index,
                    stage,
                    time.perf_counter() - start,
                    **fields,
                )

        module.forward = wrapped
        restorers.append(lambda module=module, original=original: setattr(module, "forward", original))

    for name, module in getattr(model, "named_children", lambda: [])():
        if name in {"entropy_bottleneck", "gaussian_conditional"}:
            continue
        patch_forward(module, model_stage or "%s.model" % role)

    entropy_bottleneck = getattr(model, "entropy_bottleneck", None)
    gaussian_conditional = getattr(model, "gaussian_conditional", None)
    if role == "encoder":
        patch_method(entropy_bottleneck, "compress", "encoder.symbol_encode")
        patch_method(entropy_bottleneck, "decompress", "encoder.symbol_encode")
        patch_method(gaussian_conditional, "build_indexes", "encoder.payload_model_inference")
        patch_method(gaussian_conditional, "compress", "encoder.symbol_encode")
    else:
        patch_method(gaussian_conditional, "build_indexes", "decoder.payload_model_inference")
        patch_method(entropy_bottleneck, "decompress", "decoder.symbol_decode")
        patch_method(gaussian_conditional, "decompress", "decoder.symbol_decode")

    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def _compressai_compress(model, tensor, vbr_params: JsonDict):
    if not vbr_params.get("enabled"):
        return model.compress(tensor)
    return model.compress(
        tensor,
        stage=int(vbr_params.get("stage", 2)),
        s=int(vbr_params.get("scale_index", 1)),
    )


def _compressai_decompress(model, strings, shape, vbr_params: JsonDict):
    if not vbr_params.get("enabled"):
        return model.decompress(strings, shape)
    return model.decompress(
        strings,
        shape,
        stage=int(vbr_params.get("stage", 2)),
        s=int(vbr_params.get("scale_index", 1)),
    )


def _compressai_hyper_analysis_wrapper(torch, module, use_abs: bool):
    class HyperAnalysisWrapper(torch.nn.Module):
        def __init__(self, inner, apply_abs: bool) -> None:
            super().__init__()
            self.inner = inner
            self.apply_abs = apply_abs

        def forward(self, y):
            if self.apply_abs:
                y = y.abs()
            return self.inner(y)

    return HyperAnalysisWrapper(module, use_abs)


def _compressai_entropy_encode_latents(model, y):
    if _is_compressai_factorized_model(model):
        y_strings = model.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": [int(item) for item in y.size()[-2:]]}

    if not hasattr(model, "h_a") or not hasattr(model, "h_s") or not hasattr(model, "gaussian_conditional"):
        raise RuntimeError(
            "Split payload coding is not implemented for CompressAI model class %s"
            % model.__class__.__name__
        )

    if _is_compressai_scale_hyperprior_model(model):
        z = model.h_a(y.abs())
    else:
        z = model.h_a(y)
    z_strings = model.entropy_bottleneck.compress(z)
    z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    if _is_compressai_autoregressive_model(model):
        params = model.h_s(z_hat)
        return {
            "strings": [_compressai_autoregressive_compress(model, y, params, z_hat), z_strings],
            "shape": [int(item) for item in z.size()[-2:]],
        }

    gaussian_params = model.h_s(z_hat)
    if _is_compressai_mean_scale_model(model):
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)
        y_strings = model.gaussian_conditional.compress(y, indexes, means=means_hat)
    else:
        indexes = model.gaussian_conditional.build_indexes(gaussian_params)
        y_strings = model.gaussian_conditional.compress(y, indexes)
    return {"strings": [y_strings, z_strings], "shape": [int(item) for item in z.size()[-2:]]}


def _compressai_entropy_encode_latents_onnx(
    torch,
    model,
    y_torch,
    y_numpy: np.ndarray,
    hyper_analysis_session,
    hyper_synthesis_session,
    timing_records,
    example_index: int,
    image_shape,
    device: str,
):
    fields = {"image_shape": image_shape}
    if _is_compressai_factorized_model(model):
        y_strings = timed_call(
            timing_records,
            example_index,
            "encoder.symbol_encode",
            lambda: model.entropy_bottleneck.compress(y_torch),
            **fields,
        )
        return {"strings": [y_strings], "shape": [int(item) for item in y_torch.size()[-2:]]}

    z_numpy = _timed_model_runtime_run_first_output(
        timing_records,
        example_index,
        "encoder.payload_model_inference",
        hyper_analysis_session,
        y_numpy,
        fields,
    )
    z_torch = torch.from_numpy(z_numpy).to(device)
    z_strings = timed_call(
        timing_records,
        example_index,
        "encoder.symbol_encode",
        lambda: model.entropy_bottleneck.compress(z_torch),
        **fields,
    )
    z_hat = timed_call(
        timing_records,
        example_index,
        "encoder.symbol_encode",
        lambda: model.entropy_bottleneck.decompress(z_strings, z_torch.size()[-2:]),
        **fields,
    )
    gaussian_params_numpy = _timed_model_runtime_run_first_output(
        timing_records,
        example_index,
        "encoder.payload_model_inference",
        hyper_synthesis_session,
        _torch_to_numpy(torch, z_hat),
        fields,
    )
    gaussian_params = torch.from_numpy(gaussian_params_numpy).to(device)
    y_strings = timed_call(
        timing_records,
        example_index,
        "encoder.symbol_encode",
        lambda: _compressai_encode_y_strings(model, y_torch, gaussian_params, z_hat),
        **fields,
    )
    return {"strings": [y_strings, z_strings], "shape": [int(item) for item in z_torch.size()[-2:]]}


def _compressai_entropy_decode_latents(model, strings, shape):
    if _is_compressai_factorized_model(model):
        return model.entropy_bottleneck.decompress(strings[0], shape)

    if not hasattr(model, "h_s") or not hasattr(model, "gaussian_conditional"):
        raise RuntimeError(
            "Split payload decoding is not implemented for CompressAI model class %s"
            % model.__class__.__name__
        )

    z_hat = model.entropy_bottleneck.decompress(strings[1], shape)

    if _is_compressai_autoregressive_model(model):
        params = model.h_s(z_hat)
        return _compressai_autoregressive_decompress(model, strings[0], params, z_hat)

    gaussian_params = model.h_s(z_hat)
    if _is_compressai_mean_scale_model(model):
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)
        return model.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)

    indexes = model.gaussian_conditional.build_indexes(gaussian_params)
    return model.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)


def _compressai_entropy_decode_latents_onnx(
    torch,
    model,
    strings,
    shape,
    hyper_synthesis_session,
    timing_records,
    example_index: int,
    image_shape,
    device: str,
):
    fields = {"image_shape": image_shape}
    if _is_compressai_factorized_model(model):
        return timed_call(
            timing_records,
            example_index,
            "decoder.symbol_decode",
            lambda: model.entropy_bottleneck.decompress(strings[0], shape),
            **fields,
        )

    z_hat = timed_call(
        timing_records,
        example_index,
        "decoder.symbol_decode",
        lambda: model.entropy_bottleneck.decompress(strings[1], shape),
        **fields,
    )
    gaussian_params_numpy = _timed_model_runtime_run_first_output(
        timing_records,
        example_index,
        "decoder.payload_model_inference",
        hyper_synthesis_session,
        _torch_to_numpy(torch, z_hat),
        fields,
    )
    gaussian_params = torch.from_numpy(gaussian_params_numpy).to(device)
    return timed_call(
        timing_records,
        example_index,
        "decoder.symbol_decode",
        lambda: _compressai_decode_y_strings(model, strings[0], gaussian_params, z_hat),
        **fields,
    )


def _compressai_encode_y_strings(model, y, gaussian_params, z_hat):
    if _is_compressai_autoregressive_model(model):
        return _compressai_autoregressive_compress(model, y, gaussian_params, z_hat)
    if _is_compressai_mean_scale_model(model):
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)
        return model.gaussian_conditional.compress(y, indexes, means=means_hat)
    indexes = model.gaussian_conditional.build_indexes(gaussian_params)
    return model.gaussian_conditional.compress(y, indexes)


def _compressai_decode_y_strings(model, y_strings, gaussian_params, z_hat):
    if _is_compressai_autoregressive_model(model):
        return _compressai_autoregressive_decompress(model, y_strings, gaussian_params, z_hat)
    if _is_compressai_mean_scale_model(model):
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat)
        return model.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
    indexes = model.gaussian_conditional.build_indexes(gaussian_params)
    return model.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)


def _compressai_autoregressive_compress(model, y, params, z_hat):
    from torch.nn import functional as F

    scale = 4
    kernel_size = 5
    padding = (kernel_size - 1) // 2
    y_height = int(z_hat.size(2)) * scale
    y_width = int(z_hat.size(3)) * scale
    y_hat = F.pad(y, (padding, padding, padding, padding))
    y_strings = []
    for index in range(int(y.size(0))):
        y_strings.append(
            model._compress_ar(
                y_hat[index : index + 1],
                params[index : index + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
        )
    return y_strings


def _compressai_autoregressive_decompress(model, y_strings, params, z_hat):
    from torch.nn import functional as F

    scale = 4
    kernel_size = 5
    padding = (kernel_size - 1) // 2
    y_height = int(z_hat.size(2)) * scale
    y_width = int(z_hat.size(3)) * scale
    y_hat = z_hat.new_zeros((int(z_hat.size(0)), int(model.M), y_height + 2 * padding, y_width + 2 * padding))
    for index, y_string in enumerate(y_strings):
        model._decompress_ar(
            y_string,
            y_hat[index : index + 1],
            params[index : index + 1],
            y_height,
            y_width,
            kernel_size,
            padding,
        )
    return F.pad(y_hat, (-padding, -padding, -padding, -padding))


def _is_compressai_factorized_model(model) -> bool:
    return not hasattr(model, "h_a") and hasattr(model, "entropy_bottleneck")


def _is_compressai_scale_hyperprior_model(model) -> bool:
    return model.__class__.__name__ == "ScaleHyperprior"


def _is_compressai_mean_scale_model(model) -> bool:
    return model.__class__.__name__ == "MeanScaleHyperprior"


def _is_compressai_autoregressive_model(model) -> bool:
    return callable(getattr(model, "_compress_ar", None)) and callable(getattr(model, "_decompress_ar", None))


def _compressai_vbr_scale(model, vbr_params: JsonDict):
    index = int(vbr_params.get("scale_index", 1))
    gain = getattr(model, "Gain")
    index = max(0, min(index, len(gain) - 1))
    return gain[index].detach().abs()


def _validate_compressai_choice(catalog: JsonDict, model_name: str, quality: int, metric: str) -> None:
    config = compressai_model_config(catalog, model_name)
    if quality not in [int(item) for item in config.get("qualities", [])]:
        raise RuntimeError("Quality %s is not listed for CompressAI model %s" % (quality, model_name))
    if metric not in [str(item) for item in config.get("metrics", [])]:
        raise RuntimeError("Metric %s is not listed for CompressAI model %s" % (metric, model_name))


def _decode_artifact_metadata(value, artifact_label: str) -> JsonDict:
    try:
        raw = value
        if isinstance(value, np.ndarray):
            if int(value.size) != 1:
                raise ValueError("metadata is not scalar")
            raw = value.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = decode_strict_json_object(
            str(raw),
            label="%s metadata_json" % artifact_label,
        )
    except Exception as exc:
        raise RuntimeError(
            "%s metadata_json must contain one unambiguous UTF-8 JSON object: %s"
            % (artifact_label, exc)
        ) from exc
    return payload


def _load_images(path) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        images = payload["images"]
        metadata = {}
        if "metadata_json" in payload:
            metadata.update(
                _decode_artifact_metadata(payload["metadata_json"], "Image artifact")
            )
    if images.ndim != 4 or images.shape[-1] != 3:
        raise RuntimeError("Expected images with shape [N,H,W,3], got %s" % (images.shape,))
    return images.astype(np.uint8, copy=False), metadata


def _image_for_index(images: np.ndarray, metadata: JsonDict, index: int) -> np.ndarray:
    shape = _single_original_shape(metadata, index)
    if shape:
        _count, height, width, _channels = shape
        return images[index, :height, :width, :]
    return images[index]


def _latent_for_index(latents: np.ndarray, metadata: JsonDict, index: int) -> np.ndarray:
    shape = _metadata_sequence_item(metadata, "latent_shapes", index)
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        count, channels, height, width = [int(item) for item in shape]
        return latents[index : index + count, :channels, :height, :width]
    return latents[index : index + 1]


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                _decode_artifact_metadata(payload["metadata_json"], "Bitstream artifact")
            )
    canonical, boundary_metadata = validate_channel_bits(
        bits,
        label="learned codec bitstream",
        backend="python_numpy",
    )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _load_latents(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        latents = payload["latents"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                _decode_artifact_metadata(payload["metadata_json"], "Latent artifact")
            )
    return latents.astype(np.float32, copy=False), metadata


def _images_to_torch(torch, images: np.ndarray, device: str, multiple: int, normalize: bool = False):
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, tuple(tensor.shape)


def _images_to_nchw_numpy(images: np.ndarray, multiple: int, backend: str = "auto") -> np.ndarray:
    tensor, _selected_backend = dataplane.image_to_nchw(images, backend)
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        tensor = np.pad(tensor, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False))


def _torch_to_images(torch, tensor, original_shape) -> np.ndarray:
    count, height, width, _ = original_shape
    cropped = tensor[:, :, :height, :width].detach().cpu().clamp(0.0, 1.0)
    images = (cropped.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return images[:count]


def _nchw_numpy_to_images(tensor: np.ndarray, original_shape, backend: str = "auto") -> np.ndarray:
    count, height, width, _ = [int(item) for item in original_shape]
    cropped = np.asarray(tensor, dtype=np.float32)[:count, :, :height, :width]
    images, _selected_backend = dataplane.nchw_to_image(cropped, backend)
    return images[:count]


def _single_original_shape(metadata: JsonDict, index: int):
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, (list, tuple)) and index < len(shapes):
        value = shapes[index]
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [1, int(value[1]), int(value[2]), int(value[3])]
    original = metadata.get("original_shape") or metadata.get("shape")
    if isinstance(original, (list, tuple)) and len(original) == 4:
        return [1, int(original[1]), int(original[2]), int(original[3])]
    raise RuntimeError("Artifact is missing original image shape metadata")


def _fallback_decode_images(metadata: JsonDict, policy: str) -> np.ndarray:
    shape = _fallback_image_shape(metadata)
    if policy == "gray_image":
        return np.full(shape, 128, dtype=np.uint8)
    return np.zeros(shape, dtype=np.uint8)


def _fallback_image_shape(metadata: JsonDict) -> Tuple[int, int, int, int]:
    shape = metadata.get("original_shape") or metadata.get("storage_shape") or metadata.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        return tuple(int(item) for item in shape)
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, (list, tuple)) and shapes:
        count = len(shapes)
        height = max(int(item[1]) for item in shapes if isinstance(item, (list, tuple)) and len(item) == 4)
        width = max(int(item[2]) for item in shapes if isinstance(item, (list, tuple)) and len(item) == 4)
        channels = int(shapes[0][3]) if isinstance(shapes[0], (list, tuple)) and len(shapes[0]) == 4 else 3
        return (count, height, width, channels)
    return (1, 64, 64, 3)


def _stack_image_list(images) -> Tuple[np.ndarray, list]:
    rows = [np.asarray(image, dtype=np.uint8) for image in images]
    if not rows:
        raise RuntimeError("No decoded images were produced")
    max_height = max(int(image.shape[0]) for image in rows)
    max_width = max(int(image.shape[1]) for image in rows)
    channels = int(rows[0].shape[2])
    batch = np.zeros((len(rows), max_height, max_width, channels), dtype=np.uint8)
    shapes = []
    for index, image in enumerate(rows):
        height, width, image_channels = image.shape
        if int(image_channels) != channels:
            raise RuntimeError("Decoded images have inconsistent channel counts")
        batch[index, : int(height), : int(width), :] = image
        shapes.append([1, int(height), int(width), int(image_channels)])
    return batch, shapes


def _stack_nchw_list(chunks) -> Tuple[np.ndarray, list]:
    rows = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
    if not rows:
        raise RuntimeError("No latent tensors were produced")
    max_channels = max(int(item.shape[1]) for item in rows)
    max_height = max(int(item.shape[2]) for item in rows)
    max_width = max(int(item.shape[3]) for item in rows)
    total = sum(int(item.shape[0]) for item in rows)
    batch = np.zeros((total, max_channels, max_height, max_width), dtype=np.float32)
    shapes = []
    offset = 0
    for item in rows:
        count, channels, height, width = [int(value) for value in item.shape]
        batch[offset : offset + count, :channels, :height, :width] = item
        for _ in range(count):
            shapes.append([1, channels, height, width])
        offset += count
    return batch, shapes


def _metadata_sequence_item(metadata: JsonDict, key: str, index: int):
    values = metadata.get(key)
    if isinstance(values, (list, tuple)) and index < len(values):
        value = values[index]
        if isinstance(value, (list, tuple)):
            return [int(item) if isinstance(item, (int, np.integer)) else item for item in value]
        return value
    return None


def _decoded_torch_to_images(torch, tensor) -> np.ndarray:
    decoded = ((tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)
    return (decoded.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)


def _autoencoderkl_latents_from_encoded(encoded, model, sample_mode: str, scale_latents: bool, scaling_factor: float):
    latent_dist = encoded.latent_dist
    latents = latent_dist.sample() if str(sample_mode) == "sample" else latent_dist.mean
    factor = float(scaling_factor if scaling_factor is not None else getattr(model.config, "scaling_factor", 0.18215))
    return latents * factor if bool(scale_latents) else latents


def _diffusers_latent_to_torch(torch, latent: np.ndarray, device: str, scale_latents: bool, scaling_factor: float):
    tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)
    return tensor / float(scaling_factor) if bool(scale_latents) else tensor


def _torch_to_numpy(torch, tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def _diffusers_encoded_tensor(encoded):
    if hasattr(encoded, "latents"):
        return encoded.latents
    if hasattr(encoded, "sample"):
        return encoded.sample
    if isinstance(encoded, tuple) and encoded:
        return encoded[0]
    raise RuntimeError("Could not find latent tensor in Diffusers VQModel encode output")


def _write_latents(
    ctx: OperationContext,
    latents: np.ndarray,
    input_metadata: JsonDict,
    source: str,
    model_id: str,
    params: JsonDict,
    timing_metadata: JsonDict | None = None,
) -> OperationResult:
    metadata = dict(input_metadata)
    metadata.update(
        {
            "source": source,
            "model_id": model_id,
            "model_revision": str(
                params.get("model_revision")
                or params.get("revision")
                or "local_path"
            ),
            "shape": list(latents.shape),
            "dtype": str(latents.dtype),
            "semantic_form": "latents",
            "scale_latents": bool(params.get("scale_latents", False)),
            "scaling_factor": float(params.get("scaling_factor", 1.0)),
        }
    )
    path = ctx.output_path("latents", ".npz")
    np.savez_compressed(path, latents=latents.astype(np.float32, copy=False), metadata_json=json.dumps(metadata))
    return OperationResult(
        outputs={"latents": artifact("semantic.latents.numpy", path, metadata)},
        metadata={
            "shape": list(latents.shape),
            "model_id": model_id,
            "model_revision": metadata["model_revision"],
            **({"codec_timing": timing_metadata} if timing_metadata else {}),
        },
    )


def _write_images(
    ctx: OperationContext,
    images: np.ndarray,
    metadata: JsonDict,
    source: str,
    timing_metadata: JsonDict | None = None,
) -> OperationResult:
    output_metadata = dict(metadata)
    output_metadata.update({"source": source, "shape": list(images.shape), "dtype": str(images.dtype)})
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metadata={
            "shape": list(images.shape),
            **({"codec_timing": timing_metadata} if timing_metadata else {}),
        },
    )


def _report_example_progress(ctx: OperationContext, completed: int, total: int, phase: str) -> None:
    total = max(int(total), 1)
    completed = max(0, min(int(completed), total))
    ctx.report_progress(
        "%s %d/%d examples" % (phase.capitalize(), completed, total),
        phase=phase,
        status="running",
        completed=completed,
        total=total,
        percent=float(completed) / float(total) * 100.0,
        unit="examples",
        op=ctx.step_id,
    )


def _report_example_active_progress(ctx: OperationContext, index: int, total: int, phase: str) -> None:
    total = max(int(total), 1)
    index = max(0, min(int(index), total - 1))
    percent = max(float(index) / float(total) * 100.0, 1.0)
    ctx.report_progress(
        "%s example %d/%d" % (phase.capitalize(), index + 1, total),
        phase=phase,
        status="running",
        completed=index,
        total=total,
        percent=percent,
        unit="examples",
        op=ctx.step_id,
    )


def _report_setup_progress(ctx: OperationContext, message: str) -> None:
    ctx.report_progress(
        message,
        phase="setup",
        status="running",
        completed=0,
        total=1,
        percent=1.0,
        unit="setup",
        op=ctx.step_id,
    )


def _bytes_to_bits(payload: bytes, backend: str = "auto") -> Tuple[np.ndarray, str]:
    return dataplane.bytes_to_bits(payload, backend)


def _bits_to_bytes(bits: np.ndarray, byte_count: int, backend: str = "auto") -> Tuple[bytes, str]:
    return dataplane.bits_to_bytes(bits, byte_count, backend)
