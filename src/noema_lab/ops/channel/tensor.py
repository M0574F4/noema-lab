from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core.capture_layout import (
    CaptureRecordLayout,
    CaptureRecordLayoutError,
    explicit_capture_record_layout,
    remove_capture_record_metadata,
    set_capture_record_shape,
    set_uniform_capture_record_layout,
)
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core import dataplane
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema

JsonDict = Dict[str, Any]


def _capture_layout(
    metadata: JsonDict,
    array: np.ndarray,
    label: str,
) -> CaptureRecordLayout | None:
    try:
        layout = explicit_capture_record_layout(
            metadata,
            int(array.size),
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc
    if layout is not None:
        return layout
    if metadata.get("capture_record_axis") is not None:
        try:
            axis = int(metadata["capture_record_axis"])
        except (TypeError, ValueError) as exc:
            raise OperationError("%s has invalid capture_record_axis" % label) from exc
        if axis != 0 or array.ndim < 2 or int(array.shape[0]) <= 0:
            raise OperationError(
                "%s cannot preserve its declared capture_record_axis" % label
            )
        return CaptureRecordLayout(
            count=int(array.shape[0]),
            shape=tuple(int(item) for item in array.shape[1:]),
        )
    return None


class LatentsToBitsOperation(Operation):
    id = "channel.latents_to_bits"
    name = "Pack continuous semantic latents into payload bits"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(
        {
            "dtype": {
                "type": "string",
                "default": "float32",
                "enum": ["float16", "float32"],
            },
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("latents")
        latents, metadata = _load_latents(input_artifact.path, input_artifact.metadata)
        capture_layout = _capture_layout(
            metadata,
            latents,
            "Latent bit-packer input at %s" % ctx.step_id,
        )
        dtype = np.dtype(str(ctx.params.get("dtype", "float32")))
        tensor = np.ascontiguousarray(latents.astype(dtype, copy=False))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        if dtype == np.dtype("float32"):
            bits, selected_backend = dataplane.float32_to_bits(tensor, backend)
            payload_byte_count = int(tensor.size) * int(tensor.dtype.itemsize)
        else:
            if backend == "cpp_native":
                raise OperationError("data_plane_backend=cpp_native is only implemented for float32 latent payloads")
            payload = tensor.tobytes(order="C")
            bits = _bytes_to_bits(payload)
            payload_byte_count = len(payload)
            selected_backend = "python_numpy"
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "tensor_kind": "semantic.latents.numpy",
                "tensor_array": "latents",
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "byte_count": payload_byte_count,
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "source_bit_storage": "packed_bytes",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "packer": "latents_to_bits",
                "payload_coder": "raw_latent_bitpack",
                "payload_coder_type": "raw_tensor_serialization",
                "payload_coder_label": "Raw latent bit-packing",
                "payload_coder_lossless": True,
                "entropy_coded": False,
                "data_plane_backend": selected_backend,
            }
        )
        remove_capture_record_metadata(output_metadata)
        try:
            set_uniform_capture_record_layout(
                output_metadata,
                capture_layout,
                int(bits.size),
                label="Latent bit-packer output at %s" % ctx.step_id,
            )
        except CaptureRecordLayoutError as exc:
            raise OperationError(str(exc)) from exc
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.payload_bit_count": int(bits.size)},
            metadata={
                "byte_count": payload_byte_count,
                "bit_count": int(bits.size),
                "payload_coder": "raw_latent_bitpack",
                "payload_coder_type": "raw_tensor_serialization",
                "payload_coder_label": "Raw latent bit-packing",
                "payload_coder_lossless": True,
                "entropy_coded": False,
                "data_plane_backend": selected_backend,
            },
        )


class BitsToLatentsOperation(Operation):
    id = "channel.bits_to_latents"
    name = "Unpack payload bits into continuous semantic latents"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    params_schema = object_schema(
        {
            "sanitize": {"type": "boolean", "default": True},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        capture_layout = _capture_layout(
            metadata,
            bits,
            "Latent bit-unpacker input at %s" % ctx.step_id,
        )
        try:
            raw_byte_count = metadata["byte_count"]
            raw_dtype = metadata["tensor_dtype"]
            raw_shape = metadata["tensor_shape"]
        except KeyError as exc:
            raise OperationError(
                "Latent bit payload metadata must declare byte_count, "
                "tensor_dtype, and tensor_shape"
            ) from exc
        if isinstance(raw_byte_count, bool) or not isinstance(
            raw_byte_count, (int, np.integer)
        ):
            raise OperationError(
                "Latent bit payload byte_count must be an integer"
            )
        byte_count = int(raw_byte_count)
        try:
            dtype = np.dtype(str(raw_dtype))
        except (TypeError, ValueError) as exc:
            raise OperationError(
                "Latent bit payload tensor_dtype is invalid"
            ) from exc
        if not isinstance(raw_shape, list):
            raise OperationError(
                "Latent bit payload tensor_shape must be an integer array"
            )
        shape_values = []
        for index, value in enumerate(raw_shape):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise OperationError(
                    "Latent bit payload tensor_shape[%d] must be an integer"
                    % index
                )
            shape_values.append(int(value))
        shape = tuple(shape_values)
        if byte_count < 0:
            raise OperationError("Latent bit payload byte_count must be non-negative")
        if dtype not in {np.dtype("float16"), np.dtype("float32")}:
            raise OperationError(
                "Latent bit payload tensor_dtype must be float16 or float32"
            )
        if not shape or any(value <= 0 for value in shape):
            raise OperationError(
                "Latent bit payload tensor_shape must contain positive dimensions"
            )
        needed_bits = byte_count * 8
        expected_bytes = math.prod(shape) * int(dtype.itemsize)
        if byte_count != expected_bytes:
            raise OperationError(
                "Latent bit payload byte_count %d conflicts with tensor shape %s "
                "and dtype %s, which require %d bytes"
                % (byte_count, shape, dtype, expected_bytes)
            )
        if int(bits.size) != needed_bits:
            raise OperationError(
                "Latent bit payload requires exactly %d bits; got %d"
                % (needed_bits, int(bits.size))
            )
        bits = bits.astype(np.uint8, copy=False)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        if dtype == np.dtype("float32"):
            latents, selected_backend = dataplane.bits_to_float32(bits, shape, backend)
        else:
            if backend == "cpp_native":
                raise OperationError("data_plane_backend=cpp_native is only implemented for float32 latent payloads")
            payload = _bits_to_bytes(bits, byte_count)
            latents = np.frombuffer(payload, dtype=dtype).reshape(shape).astype(np.float32, copy=True)
            selected_backend = "python_numpy"
        if bool(ctx.params.get("sanitize", True)):
            latents = np.nan_to_num(latents, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "source_bit_count": int(bits.size),
                "unpacker": "bits_to_latents",
                "dtype": str(latents.dtype),
                "shape": list(latents.shape),
                "data_plane_backend": selected_backend,
            }
        )
        remove_capture_record_metadata(output_metadata)
        if capture_layout is not None:
            if not shape or int(shape[0]) != capture_layout.count:
                raise OperationError(
                    "Latent tensor shape conflicts with the explicit capture-record "
                    "count"
                )
            try:
                set_capture_record_shape(
                    output_metadata,
                    capture_layout,
                    shape[1:] or (1,),
                    int(latents.size),
                    label="Latent bit-unpacker output at %s" % ctx.step_id,
                )
            except CaptureRecordLayoutError as exc:
                raise OperationError(str(exc)) from exc
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={"shape": list(latents.shape), "data_plane_backend": selected_backend},
        )


def _load_latents(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        latents = payload["latents"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Tensor channel bit metadata_json",
                )
            )
    return latents.astype(np.float32, copy=False), metadata


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Tensor channel latent metadata_json",
                )
            )
    canonical, boundary_metadata = validate_channel_bits(
        bits,
        label="channel.tensor bits",
        backend="python_numpy",
    )
    if "byte_count" in metadata:
        raw_byte_count = metadata["byte_count"]
        if isinstance(raw_byte_count, bool) or not isinstance(
            raw_byte_count, (int, np.integer)
        ):
            raise OperationError(
                "Tensor channel metadata byte_count must be an integer"
            )
        declared_bytes = int(raw_byte_count)
        if declared_bytes < 0:
            raise OperationError(
                "Tensor channel metadata byte_count must be non-negative"
            )
        declared_bits = declared_bytes * 8
        if int(canonical.size) != declared_bits:
            raise OperationError(
                "Latent bit payload requires exactly %d bits; got %d"
                % (declared_bits, int(canonical.size))
            )
    for field in ("bit_count", "payload_bit_count"):
        if field in metadata:
            raw_declared = metadata[field]
            if isinstance(raw_declared, bool) or not isinstance(
                raw_declared, (int, np.integer)
            ):
                raise OperationError(
                    "Tensor channel metadata %s must be an integer" % field
                )
            declared = int(raw_declared)
            if declared != int(canonical.size):
                raise OperationError(
                    "Tensor channel metadata %s declares %d bits but the "
                    "artifact stores %d"
                    % (field, declared, int(canonical.size))
                )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _bytes_to_bits(payload: bytes) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(data).astype(np.uint8, copy=False)


def _bits_to_bytes(bits: np.ndarray, byte_count: int) -> bytes:
    packed = np.packbits(bits.astype(np.uint8, copy=False))
    return packed[:byte_count].tobytes()
