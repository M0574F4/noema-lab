from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.boundaries import validate_payload_bits
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.sample_identity import content_sha256_item_ids
from noema_lab.ops.models.deepjscc_checkpoint import (
    CHECKPOINT_FORMAT as DEEPJSCC_CHECKPOINT_FORMAT,
    decode_deepjscc_symbols,
    encode_deepjscc_images,
    load_deepjscc_reference_checkpoint,
    symbol_shape_from_metadata,
)
from noema_lab.ops.models.timing import codec_timing_metadata, timed_call

JsonDict = Dict[str, Any]


_ADAPTER_SCHEMA = {
    "path": {"type": "string", "default": ""},
    "module": {"type": "string", "default": ""},
    "callable": {"type": "string", "default": ""},
    "call_style": {
        "type": "string",
        "default": "array_params",
        "enum": ["array_params", "array", "dict"],
    },
}

_BIT_ADAPTER_SCHEMA = dict(_ADAPTER_SCHEMA)
_BIT_ADAPTER_SCHEMA.update(
    {
        "bit_storage": {
            "type": "string",
            "default": "unpacked_bits",
            "enum": ["unpacked_bits", "packed_bytes"],
            "description": "unpacked_bits means one array element is one channel bit. packed_bytes means the callable returns bytes that Noema expands with np.unpackbits before channel transmission.",
        },
        "bit_order": {
            "type": "string",
            "default": "big",
            "enum": ["big", "little"],
            "description": "Bit order used when converting packed bytes to and from the channel bit vector.",
        },
    }
)

_DEEPJSCC_ADAPTER_SCHEMA = dict(_ADAPTER_SCHEMA)
_DEEPJSCC_ADAPTER_SCHEMA.update(
    {
        "runtime": {
            "type": "string",
            "default": "training_interface",
            "enum": ["training_interface", "external_callable", "learned_checkpoint", "learned_artifact"],
            "description": "Declare an export-only interface, invoke a trusted callable, use the reference checkpoint runtime, or run a registered portable trained artifact.",
        },
        "checkpoint_path": {
            "type": "string",
            "default": "",
            "description": "Managed safe-NPZ DeepJSCC checkpoint used by runtime=learned_checkpoint.",
            "x-noema-ui": {
                "control": "trained_artifact",
                "label": "Learned checkpoint",
                "accept": ".npz,application/octet-stream",
                "visible_when": {"runtime": "learned_checkpoint"},
                "derived_params": [
                    "runtime",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "checkpoint_format",
                    "checkpoint_strict",
                    "checkpoint_max_bytes",
                    "symbol_channels",
                ],
            },
        },
        "checkpoint_sha256": {
            "type": "string",
            "default": "",
            "description": "Required lowercase SHA-256 of the frozen DeepJSCC checkpoint.",
            "x-noema-ui": {"hidden": True},
        },
        "checkpoint_format": {
            "type": "string",
            "default": DEEPJSCC_CHECKPOINT_FORMAT,
            "enum": [DEEPJSCC_CHECKPOINT_FORMAT],
            "x-noema-ui": {"hidden": True},
        },
        "checkpoint_strict": {
            "type": "boolean",
            "default": True,
            "x-noema-ui": {"hidden": True},
        },
        "checkpoint_max_bytes": {
            "type": "integer",
            "default": 67108864,
            "minimum": 1,
            "x-noema-ui": {"hidden": True},
        },
        "symbol_channels": {
            "type": "integer",
            "default": 16,
            "minimum": 1,
            "maximum": 1024,
            "description": "Complex channel count declared by the managed checkpoint.",
            "x-noema-ui": {"hidden": True},
        },
        "artifact_manifest_path": {
            "type": "string",
            "default": "",
            "description": "Registered trained-artifact manifest implementing this DeepJSCC slot.",
            "x-noema-ui": {
                "control": "trained_artifact",
                "label": "Trained artifact",
                "accept": ".zip,.noema-artifact,application/zip,application/octet-stream",
                "visible_when": {"runtime": "learned_artifact"},
                "derived_params": [
                    "runtime",
                    "artifact_manifest_path",
                    "artifact_entrypoint",
                    "artifact_package_sha256",
                ],
            },
        },
        "artifact_entrypoint": {
            "type": "string",
            "default": "",
            "description": "Entrypoint in the registered artifact manifest; defaults to encoder or decoder by slot role.",
            "x-noema-ui": {"hidden": True},
        },
        "artifact_package_sha256": {
            "type": "string",
            "default": "",
            "description": "Required package identity of the frozen trained artifact.",
            "x-noema-ui": {"hidden": True},
        },
    }
)


class ExternalIndicesEncodeOperation(Operation):
    id = "model.external_encode_indices"
    name = "External image encoder to semantic indices"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    params_schema = object_schema(_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        images, metadata = _load_array(ctx.require_input("images"), "images")
        total = _example_count(images)
        timing_records = []
        _report_example_progress(ctx, 0, total, "encoding")
        result = timed_call(
            timing_records,
            None,
            "encoder.external_call",
            lambda: _call_external(ctx.params, images, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "encoding")
        indices, output_metadata = _result_array(result, np.int64, "indices")
        output_metadata.update({"adapter": "external", "semantic_form": "indices"})
        output_metadata.setdefault("codebook_size", _infer_codebook_size(indices))
        path = ctx.output_path("indices", ".npz")
        np.savez_compressed(path, indices=indices, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"indices": artifact("semantic.indices.numpy", path, output_metadata)},
            metadata={
                "shape": list(indices.shape),
                "codebook_size": output_metadata["codebook_size"],
                "codec_timing": _external_codec_timing("encoder", timing_records, ctx.params),
            },
        )


class ExternalIndicesDecodeOperation(Operation):
    id = "model.external_decode_indices"
    name = "External semantic indices decoder to image batch"
    input_kinds = {"indices": ["semantic.indices.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        indices, metadata = _load_array(ctx.require_input("indices"), "indices")
        total = _example_count(indices)
        timing_records = []
        _report_example_progress(ctx, 0, total, "decoding")
        result = timed_call(
            timing_records,
            None,
            "decoder.external_call",
            lambda: _call_external(ctx.params, indices, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "decoding")
        return _write_images(ctx, result, "external_decode_indices", _external_codec_timing("decoder", timing_records, ctx.params))


class ExternalLatentsEncodeOperation(Operation):
    id = "model.external_encode_latents"
    name = "External image encoder to continuous semantic latents"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"latents": "semantic.latents.numpy"}
    params_schema = object_schema(_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        images, metadata = _load_array(ctx.require_input("images"), "images")
        total = _example_count(images)
        timing_records = []
        _report_example_progress(ctx, 0, total, "encoding")
        result = timed_call(
            timing_records,
            None,
            "encoder.external_call",
            lambda: _call_external(ctx.params, images, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "encoding")
        latents, output_metadata = _result_array(result, np.float32, "latents")
        output_metadata.update({"adapter": "external", "semantic_form": "latents"})
        path = ctx.output_path("latents", ".npz")
        np.savez_compressed(path, latents=latents, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"latents": artifact("semantic.latents.numpy", path, output_metadata)},
            metadata={
                "shape": list(latents.shape),
                "codec_timing": _external_codec_timing("encoder", timing_records, ctx.params),
            },
        )


class ExternalLatentsDecodeOperation(Operation):
    id = "model.external_decode_latents"
    name = "External continuous semantic latents decoder to image batch"
    input_kinds = {"latents": ["semantic.latents.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        latents, metadata = _load_array(ctx.require_input("latents"), "latents")
        total = _example_count(latents)
        timing_records = []
        _report_example_progress(ctx, 0, total, "decoding")
        result = timed_call(
            timing_records,
            None,
            "decoder.external_call",
            lambda: _call_external(ctx.params, latents, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "decoding")
        return _write_images(ctx, result, "external_decode_latents", _external_codec_timing("decoder", timing_records, ctx.params))


class ExternalBitsEncodeOperation(Operation):
    id = "model.external_encode_bits"
    name = "External image encoder to payload bits"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(_BIT_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        images, metadata = _load_array(ctx.require_input("images"), "images")
        total = _example_count(images)
        timing_records = []
        _report_example_progress(ctx, 0, total, "encoding")
        result = timed_call(
            timing_records,
            None,
            "encoder.external_call",
            lambda: _call_external(ctx.params, images, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "encoding")
        array, output_metadata = _result_array(result, np.uint8, "bits")
        storage = str(ctx.params.get("bit_storage", "unpacked_bits"))
        bit_order = str(ctx.params.get("bit_order", "big"))
        if storage == "packed_bytes":
            byte_payload = np.ascontiguousarray(array.reshape(-1).astype(np.uint8, copy=False))
            bit_count = int(output_metadata.get("bit_count") or output_metadata.get("valid_bit_count") or byte_payload.size * 8)
            bits = np.unpackbits(byte_payload, bitorder=bit_order)[:bit_count].astype(np.uint8, copy=False)
            output_metadata.update(
                {
                    "byte_count": int(byte_payload.size),
                    "source_storage_dtype": str(byte_payload.dtype),
                }
            )
        else:
            bits, _boundary_metadata = validate_payload_bits(
                array,
                label="external encoder output bits",
                backend="python_numpy",
            )
            bit_count = int(output_metadata.get("bit_count") or output_metadata.get("valid_bit_count") or bits.size)
            bits = bits[:bit_count]
        bits, boundary_metadata = validate_payload_bits(
            bits,
            label="external encoder canonical payload bits",
            backend="python_numpy",
        )
        output_metadata.update(boundary_metadata)
        output_metadata.update(
            {
                "adapter": "external",
                "bit_count": int(bits.size),
                "payload_bit_count": int(bits.size),
                "bit_role": "payload",
                "adapter_bit_storage": storage,
                "bit_storage": "unpacked_uint8",
                "bit_order": bit_order,
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.payload_bit_count": int(bits.size)},
            metadata={
                "bit_count": int(bits.size),
                "codec_timing": _external_codec_timing("encoder", timing_records, ctx.params),
            },
        )


class ExternalBitsDecodeOperation(Operation):
    id = "model.external_decode_bits"
    name = "External payload bits decoder to image batch"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    params_schema = object_schema(_BIT_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        bits, metadata = _load_array(ctx.require_input("bits"), "bits")
        bits, boundary_metadata = validate_payload_bits(
            bits,
            label="external decoder input bits",
            backend="python_numpy",
        )
        metadata.update(boundary_metadata)
        total = _example_count_from_metadata(metadata, bits)
        timing_records = []
        _report_example_progress(ctx, 0, total, "decoding")
        storage = str(ctx.params.get("bit_storage", "unpacked_bits"))
        bit_order = str(ctx.params.get("bit_order", metadata.get("bit_order", "big")))
        if storage == "packed_bytes":
            bit_count = int(metadata.get("bit_count") or bits.size)
            byte_count = int(metadata.get("byte_count") or ((bit_count + 7) // 8))
            adapter_input = np.packbits(bits[:bit_count].astype(np.uint8, copy=False), bitorder=bit_order)[:byte_count]
        else:
            adapter_input = bits.astype(np.uint8, copy=False)
        result = timed_call(
            timing_records,
            None,
            "decoder.external_call",
            lambda: _call_external(ctx.params, adapter_input, metadata),
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "decoding")
        return _write_images(ctx, result, "external_decode_bits", _external_codec_timing("decoder", timing_records, ctx.params))


class DeepJsccExternalEncodeOperation(Operation):
    id = "model.deepjscc_external_encode"
    name = "DeepJSCC encoder replacement / returned-artifact slot"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "This operation defines a portable encoder replacement boundary. When selected, its current implementation is omitted and the researcher-supplied module owns parameters and autograd behavior.",
    }
    backends = {
        "benchmark_run": ["external", "torch", "onnxruntime"],
        "dataset_capture": ["external", "torch", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "external",
            "implementation": "external_callable_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "external_callable"},
        },
        {
            "runner": "benchmark_run",
            "backend": "torch",
            "implementation": "safe_npz_reference_cnn_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_checkpoint"},
        },
        {
            "runner": "benchmark_run",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_artifact"},
        },
        {
            "runner": "dataset_capture",
            "backend": "external",
            "implementation": "external_callable_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "external_callable"},
        },
        {
            "runner": "dataset_capture",
            "backend": "torch",
            "implementation": "safe_npz_reference_cnn_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_checkpoint"},
        },
        {
            "runner": "dataset_capture",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_artifact"},
        },
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "external_model_encoder_slot",
            "status": "implemented",
            "parameter_bindings": {"runtime": "training_interface"},
            "notes": "The recipe supplies the typed encoder boundary; the researcher supplies the model architecture and training method.",
        },
    ]
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    trained_artifact_abi = {
        "component_id": "encoder",
        "component_role": "encoder",
        "entrypoint_id": "encoder",
        "required_operation_inputs": ["images"],
        "inputs": {
            "images": {"dtype": "float32", "shape": ["batch", 3, "height", "width"]},
        },
        "outputs": {
            "symbols_ri": {
                "dtype": "float32",
                "shape": ["batch", "real_imag_channel", "symbol_height", "symbol_width"],
            },
        },
        "binding_params": {
            "runtime": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "encoder",
        },
    }
    params_schema = object_schema(_DEEPJSCC_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        images, metadata = _load_array(ctx.require_input("images"), "images")
        total = _example_count(images)
        timing_records = []
        _report_example_progress(ctx, 0, total, "encoding")
        runtime = _deepjscc_runtime(ctx.params)
        if runtime == "learned_checkpoint":
            checkpoint = _load_deepjscc_checkpoint(ctx.params)
            result_fn = lambda: {
                "array": encode_deepjscc_images(checkpoint, images),
                "metadata": {
                    "checkpoint_path": str(checkpoint.path),
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_format": DEEPJSCC_CHECKPOINT_FORMAT,
                    "image_shape": list(images.shape),
                },
            }
            timing_label = "encoder.learned_checkpoint"
            adapter_name = "learned_checkpoint"
        elif runtime == "learned_artifact":
            checkpoint = None
            result_fn = lambda: _run_deepjscc_artifact_encoder(ctx.params, images)
            timing_label = "encoder.learned_artifact"
            adapter_name = "learned_artifact"
        else:
            checkpoint = None
            result_fn = lambda: _call_external(ctx.params, images, metadata)
            timing_label = "encoder.external_call"
            adapter_name = "external"
        result = timed_call(
            timing_records,
            None,
            timing_label,
            result_fn,
            example_count=total,
        )
        _report_example_progress(ctx, total, total, "encoding")
        symbols, output_metadata = _result_array(result, np.complex64, "symbols")
        # Keep exact source-image extents attached to the continuous symbol
        # stream. Downstream channel accounting must count complex channel uses
        # per real source pixel without interpreting float storage as payload
        # bits. ``image_shape`` remains the fallback for external artifacts that
        # do not receive per-image shape metadata.
        for source_metadata_key in (
            "original_shapes",
            "original_shape",
            "image_ids",
            "sample_ids",
            "source_item_ids",
        ):
            if source_metadata_key in metadata:
                output_metadata.setdefault(
                    source_metadata_key, metadata[source_metadata_key]
                )
        output_metadata.setdefault("image_shape", metadata.get("shape") or list(images.shape))
        symbol_shape = output_metadata.get("symbol_shape") or list(symbols.shape)
        try:
            symbol_shape = [int(value) for value in symbol_shape]
        except (TypeError, ValueError) as exc:
            raise OperationError("DeepJSCC encoder symbol_shape metadata must contain integers") from exc
        if (
            not symbol_shape
            or symbol_shape[0] != total
            or int(np.prod(symbol_shape, dtype=np.int64)) != int(symbols.size)
            or int(symbols.size) % total
        ):
            raise OperationError(
                "DeepJSCC encoder must preserve a rectangular per-source-item symbol boundary"
            )
        source_item_ids, source_item_id_source = _deepjscc_source_item_ids(
            metadata, images
        )
        source_item_symbol_count = int(symbols.size) // total
        symbols = symbols.reshape(-1).astype(np.complex64, copy=False)
        output_metadata.update(
            {
                "adapter": adapter_name,
                "semantic_form": "deepjscc_symbols",
                "symbol_count": int(symbols.size),
                "symbol_shape": symbol_shape,
                "source_item_count": total,
                "source_item_ids": source_item_ids,
                "source_item_id_source": source_item_id_source,
                "source_item_symbol_counts": [
                    source_item_symbol_count
                ]
                * total,
            }
        )
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics={"channel.symbol_count": int(symbols.size)},
            metadata={
                "symbol_count": int(symbols.size),
                "codec_timing": _deepjscc_codec_timing(
                    "encoder", timing_records, ctx.params, runtime
                ),
            },
        )


class DeepJsccExternalDecodeOperation(Operation):
    id = "model.deepjscc_external_decode"
    name = "DeepJSCC decoder replacement / returned-artifact slot"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "This operation defines a portable decoder replacement boundary. When selected, its current implementation is omitted and the researcher-supplied module owns parameters and autograd behavior.",
    }
    backends = {
        "benchmark_run": ["external", "torch", "onnxruntime"],
        "dataset_capture": ["external", "torch", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "external",
            "implementation": "external_callable_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "external_callable"},
        },
        {
            "runner": "benchmark_run",
            "backend": "torch",
            "implementation": "safe_npz_reference_cnn_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_checkpoint"},
        },
        {
            "runner": "benchmark_run",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_artifact"},
        },
        {
            "runner": "dataset_capture",
            "backend": "external",
            "implementation": "external_callable_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "external_callable"},
        },
        {
            "runner": "dataset_capture",
            "backend": "torch",
            "implementation": "safe_npz_reference_cnn_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_checkpoint"},
        },
        {
            "runner": "dataset_capture",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"runtime": "learned_artifact"},
        },
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "external_model_decoder_slot",
            "status": "implemented",
            "parameter_bindings": {"runtime": "training_interface"},
            "notes": "The recipe supplies the typed decoder boundary; the researcher supplies the model architecture and training method.",
        },
    ]
    input_kinds = {"symbols": ["channel.rx_symbols.complex_numpy", "channel.symbols.complex_numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    trained_artifact_abi = {
        "component_id": "decoder",
        "component_role": "decoder",
        "entrypoint_id": "decoder",
        "required_operation_inputs": ["symbols"],
        "inputs": {
            "symbols_ri": {
                "dtype": "float32",
                "shape": ["batch", "real_imag_channel", "symbol_height", "symbol_width"],
            },
        },
        "outputs": {
            "reconstruction": {"dtype": "float32", "shape": ["batch", 3, "height", "width"]},
        },
        "binding_params": {
            "runtime": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "decoder",
        },
    }
    params_schema = object_schema(_DEEPJSCC_ADAPTER_SCHEMA, additional=True)

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        array_name = "rx_symbols" if input_artifact.kind == "channel.rx_symbols.complex_numpy" else "symbols"
        symbols, metadata = _load_array(input_artifact, array_name)
        total = _example_count_from_metadata(metadata, symbols)
        timing_records = []
        _report_example_progress(ctx, 0, total, "decoding")
        runtime = _deepjscc_runtime(ctx.params)
        if runtime == "learned_checkpoint":
            checkpoint = _load_deepjscc_checkpoint(ctx.params)
            shape = symbol_shape_from_metadata(
                metadata,
                int(symbols.size),
                checkpoint.symbol_channels,
            )
            reshaped = symbols.astype(np.complex64, copy=False).reshape(shape)
            image_shape = metadata.get("image_shape")
            result_fn = lambda: {
                "array": decode_deepjscc_symbols(
                    checkpoint,
                    reshaped,
                    image_shape=image_shape,
                ),
                "metadata": {
                    "adapter": "learned_checkpoint",
                    "checkpoint_path": str(checkpoint.path),
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_format": DEEPJSCC_CHECKPOINT_FORMAT,
                    "source_symbol_shape": list(shape),
                },
            }
            timing_label = "decoder.learned_checkpoint"
            adapter_name = "learned_checkpoint"
        elif runtime == "learned_artifact":
            checkpoint = None
            result_fn = lambda: _run_deepjscc_artifact_decoder(
                ctx.params,
                symbols,
                metadata,
            )
            timing_label = "decoder.learned_artifact"
            adapter_name = "learned_artifact"
        else:
            checkpoint = None
            result_fn = lambda: _call_external(
                ctx.params,
                symbols.astype(np.complex64, copy=False),
                metadata,
            )
            timing_label = "decoder.external_call"
            adapter_name = "external"
        result = timed_call(
            timing_records,
            None,
            timing_label,
            result_fn,
            example_count=total,
        )
        result = _inherit_source_item_metadata(result, metadata)
        _report_example_progress(ctx, total, total, "decoding")
        return _write_images(
            ctx,
            result,
            "deepjscc_external_decode",
            _deepjscc_codec_timing("decoder", timing_records, ctx.params, runtime),
            adapter=adapter_name,
        )


def _deepjscc_source_item_ids(
    metadata: Mapping[str, Any], images: np.ndarray
) -> Tuple[list[str], str]:
    count = int(images.shape[0])
    for field in ("source_item_ids", "image_ids", "sample_ids"):
        raw = metadata.get(field)
        if isinstance(raw, list) and len(raw) == count:
            values = [str(item) for item in raw]
            if all(values):
                return values, field
    return content_sha256_item_ids(images), "content_sha256"


def _inherit_source_item_metadata(result: Any, metadata: Mapping[str, Any]) -> JsonDict:
    if isinstance(result, Mapping):
        payload = dict(result)
        output_metadata = dict(payload.get("metadata") or {})
        if "array" not in payload and "images" not in payload:
            return payload
    else:
        payload = {"array": result}
        output_metadata = {}
    for key in (
        "source_item_count",
        "source_item_ids",
        "source_item_id_source",
        "source_items",
        "image_ids",
        "sample_ids",
        "repeat_count",
        "source_item_outage",
        "source_item_success_rate",
        "source_item_outage_rate",
        "original_shapes",
        "original_shape",
        "image_shape",
    ):
        if key in metadata:
            output_metadata.setdefault(key, metadata[key])
    if "source_item_ids" in output_metadata:
        output_metadata.setdefault("image_ids", output_metadata["source_item_ids"])
    payload["metadata"] = output_metadata
    return payload


def _call_external(params: JsonDict, array: np.ndarray, metadata: JsonDict) -> Any:
    function = _load_callable(params)
    adapter_params = {
        key: value
        for key, value in params.items()
        if key not in {"path", "module", "callable", "call_style", "runner"}
    }
    call_style = str(params.get("call_style", "array_params"))
    if call_style == "array":
        return function(array)
    if call_style == "dict":
        return function({"array": array, "metadata": metadata, "params": adapter_params})
    return function(array, adapter_params)


def _external_codec_timing(role: str, timing_records, params: JsonDict) -> JsonDict:
    requested_runner = str(params.get("runner") or "").strip()
    notes: JsonDict = {
        "executor": "External adapters are invoked as local Python callables. ONNX/OpenVINO/TorchScript dispatch is not implemented in this adapter yet.",
        "timing_scope": "batch_call",
        "fairness_note": "External callable timing is one adapter call for the provided batch unless the user adapter emits its own per-example timings.",
    }
    if requested_runner and requested_runner != "local_python":
        notes["requested_runner"] = requested_runner
    metadata = codec_timing_metadata(role, timing_records, runner="local_python", notes=notes)
    metadata["scope"] = "batch_stage"
    return metadata


def _deepjscc_codec_timing(
    role: str,
    timing_records,
    params: JsonDict,
    runtime: str,
) -> JsonDict:
    if runtime == "external_callable":
        return _external_codec_timing(role, timing_records, params)
    if runtime == "learned_artifact":
        metadata = codec_timing_metadata(
            role,
            timing_records,
            runner="onnxruntime",
            notes={
                "executor": "Registered portable trained-artifact runtime.",
                "timing_scope": "batch_call",
                "fairness_note": "Timing covers one hash-verified artifact entrypoint invocation for the provided batch.",
            },
        )
        metadata["scope"] = "batch_stage"
        return metadata
    metadata = codec_timing_metadata(
        role,
        timing_records,
        runner="torch",
        notes={
            "executor": "Noema safe-NPZ reference-CNN checkpoint runtime using PyTorch CPU inference.",
            "timing_scope": "batch_call",
            "fairness_note": "Timing covers one frozen reference-CNN inference call for the provided batch.",
        },
    )
    metadata["scope"] = "batch_stage"
    return metadata


def _deepjscc_runtime(params: JsonDict) -> str:
    runtime = str(params.get("runtime") or "training_interface").strip().lower()
    if runtime not in {"training_interface", "external_callable", "learned_checkpoint", "learned_artifact"}:
        raise OperationError(
            "DeepJSCC runtime must be training_interface, external_callable, learned_checkpoint, or learned_artifact; got %r"
            % runtime
        )
    if runtime == "training_interface":
        raise OperationError(
            "DeepJSCC runtime=training_interface is an export-only typed interface. "
            "Export and train a DeepJSCC project, then select its returned trained artifact, "
            "or configure runtime=external_callable with a Python callable before running the recipe."
        )
    return runtime


def _run_deepjscc_artifact_encoder(params: JsonDict, images: np.ndarray) -> JsonDict:
    from noema_lab.core.trained_artifact_runtime import run_trained_artifact_entrypoint

    values = np.asarray(images)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
        raise OperationError("DeepJSCC artifact encoder expects uint8 images shaped [N,H,W,3]")
    manifest_path = str(params.get("artifact_manifest_path") or "").strip()
    entrypoint = str(params.get("artifact_entrypoint") or "encoder").strip()
    if not manifest_path:
        raise OperationError("runtime=learned_artifact requires artifact_manifest_path")
    normalized = np.ascontiguousarray(values.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    outputs = run_trained_artifact_entrypoint(
        Path(manifest_path),
        entrypoint,
        {"images": normalized},
        expected_package_sha256=str(params.get("artifact_package_sha256") or ""),
    )
    raw = outputs.get("symbols_ri")
    if raw is None:
        raise OperationError("DeepJSCC encoder artifact must return symbols_ri")
    encoded = np.asarray(raw, dtype=np.float32)
    if encoded.ndim != 4 or encoded.shape[0] != values.shape[0] or encoded.shape[1] < 2 or encoded.shape[1] % 2:
        raise OperationError("DeepJSCC symbols_ri must have shape [N,2C,H,W]")
    channels = int(encoded.shape[1] // 2)
    symbols = encoded[:, :channels] + 1j * encoded[:, channels:]
    if not np.all(np.isfinite(symbols.real)) or not np.all(np.isfinite(symbols.imag)):
        raise OperationError("DeepJSCC encoder artifact produced non-finite symbols")
    return {
        "array": np.asarray(symbols, dtype=np.complex64),
        "metadata": {
            "adapter": "learned_artifact",
            "artifact_manifest_path": manifest_path,
            "artifact_entrypoint": entrypoint,
            "artifact_package_sha256": str(params.get("artifact_package_sha256") or ""),
            "image_shape": list(values.shape),
            "symbol_shape": list(symbols.shape),
        },
    }


def _run_deepjscc_artifact_decoder(
    params: JsonDict,
    symbols: np.ndarray,
    metadata: JsonDict,
) -> JsonDict:
    from noema_lab.core.trained_artifact_runtime import run_trained_artifact_entrypoint

    manifest_path = str(params.get("artifact_manifest_path") or "").strip()
    entrypoint = str(params.get("artifact_entrypoint") or "decoder").strip()
    if not manifest_path:
        raise OperationError("runtime=learned_artifact requires artifact_manifest_path")
    raw_shape = metadata.get("symbol_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
        raise OperationError("DeepJSCC artifact decoder requires symbol_shape=[N,C,H,W] metadata")
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise OperationError("DeepJSCC symbol_shape metadata must contain integers") from exc
    if any(value < 1 for value in shape) or int(np.prod(shape, dtype=np.int64)) != int(np.asarray(symbols).size):
        raise OperationError("DeepJSCC symbol_shape does not match the received symbols")
    complex_values = np.asarray(symbols, dtype=np.complex64).reshape(shape)
    symbols_ri = np.ascontiguousarray(
        np.concatenate([complex_values.real, complex_values.imag], axis=1),
        dtype=np.float32,
    )
    outputs = run_trained_artifact_entrypoint(
        Path(manifest_path),
        entrypoint,
        {"symbols_ri": symbols_ri},
        expected_package_sha256=str(params.get("artifact_package_sha256") or ""),
    )
    raw = outputs.get("reconstruction")
    if raw is None:
        raise OperationError("DeepJSCC decoder artifact must return reconstruction")
    reconstruction = np.asarray(raw, dtype=np.float32)
    if reconstruction.ndim != 4 or reconstruction.shape[0] != shape[0] or reconstruction.shape[1] != 3:
        raise OperationError("DeepJSCC reconstruction must have shape [N,3,H,W]")
    reconstruction = reconstruction.transpose(0, 2, 3, 1)
    image_shape = metadata.get("image_shape")
    if isinstance(image_shape, (list, tuple)) and len(image_shape) == 4:
        requested = tuple(int(value) for value in image_shape)
        if requested[0] != reconstruction.shape[0] or requested[-1] != 3:
            raise OperationError("DeepJSCC image_shape metadata is incompatible with reconstruction")
        if reconstruction.shape[1] < requested[1] or reconstruction.shape[2] < requested[2]:
            raise OperationError("DeepJSCC reconstruction is smaller than image_shape metadata")
        reconstruction = reconstruction[:, : requested[1], : requested[2], :]
    if not np.all(np.isfinite(reconstruction)):
        raise OperationError("DeepJSCC decoder artifact produced non-finite images")
    images = np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0).astype(np.uint8)
    return {
        "array": images,
        "metadata": {
            "adapter": "learned_artifact",
            "artifact_manifest_path": manifest_path,
            "artifact_entrypoint": entrypoint,
            "artifact_package_sha256": str(params.get("artifact_package_sha256") or ""),
            "source_symbol_shape": list(shape),
        },
    }


def _load_deepjscc_checkpoint(params: JsonDict):
    checkpoint = load_deepjscc_reference_checkpoint(
        str(params.get("checkpoint_path") or ""),
        str(params.get("checkpoint_sha256") or ""),
        checkpoint_format=str(
            params.get("checkpoint_format") or DEEPJSCC_CHECKPOINT_FORMAT
        ),
        strict=bool(params.get("checkpoint_strict", True)),
        max_bytes=int(params.get("checkpoint_max_bytes") or 67108864),
    )
    if "symbol_channels" in params and params.get("symbol_channels") is not None:
        try:
            declared_channels = int(params["symbol_channels"])
        except (TypeError, ValueError) as exc:
            raise OperationError("DeepJSCC params.symbol_channels must be an integer") from exc
        if declared_channels != checkpoint.symbol_channels:
            raise OperationError(
                "DeepJSCC params.symbol_channels=%d does not match checkpoint metadata %d"
                % (declared_channels, checkpoint.symbol_channels)
            )
    return checkpoint


def _load_callable(params: JsonDict) -> Callable:
    callable_name = str(params.get("callable") or "")
    module_name = str(params.get("module") or "")
    path = str(params.get("path") or "")
    if ":" in callable_name and not module_name:
        module_name, callable_name = callable_name.split(":", 1)
    if not callable_name:
        raise RuntimeError("External model adapter requires params.callable")
    if path:
        path_obj = Path(path).expanduser().resolve()
        if path_obj.is_file():
            module = _load_module_from_file(path_obj)
        else:
            sys.path.insert(0, str(path_obj))
            module = importlib.import_module(module_name)
    else:
        module = importlib.import_module(module_name)
    target = module
    for part in callable_name.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise RuntimeError("External adapter target is not callable: %s" % callable_name)
    return target


def _load_module_from_file(path: Path):
    module_name = "noema_external_%s" % abs(hash(str(path)))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load external module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_array(input_artifact, array_name: str) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(input_artifact.path), allow_pickle=False) as payload:
        array = payload[array_name]
        metadata = dict(input_artifact.metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="External model artifact metadata_json",
                )
            )
    return array, metadata


def _result_array(result: Any, dtype, default_name: str) -> Tuple[np.ndarray, JsonDict]:
    metadata: JsonDict = {}
    array = result
    if isinstance(result, Mapping):
        metadata = dict(result.get("metadata") or {})
        array = result.get("array", result.get(default_name))
    if array is None:
        raise RuntimeError("External adapter did not return an array")
    if dtype == np.uint8 and isinstance(array, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(array), dtype=np.uint8), metadata
    raw = np.asarray(array)
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        if not np.issubdtype(raw.dtype, np.integer):
            raise RuntimeError(
                "External adapter %s output must use an integer dtype, got %s"
                % (default_name, raw.dtype)
            )
        limits = np.iinfo(target_dtype)
        if raw.size and (
            int(np.min(raw)) < int(limits.min)
            or int(np.max(raw)) > int(limits.max)
        ):
            raise RuntimeError(
                "External adapter %s output exceeds %s range"
                % (default_name, target_dtype)
            )
    canonical = raw.astype(target_dtype, copy=False)
    if np.issubdtype(target_dtype, np.floating):
        if not bool(np.all(np.isfinite(canonical))):
            raise RuntimeError(
                "External adapter %s output contains non-finite values"
                % default_name
            )
    elif np.issubdtype(target_dtype, np.complexfloating):
        if not bool(np.all(np.isfinite(canonical.real))) or not bool(
            np.all(np.isfinite(canonical.imag))
        ):
            raise RuntimeError(
                "External adapter %s output contains non-finite values"
                % default_name
            )
    return canonical, metadata


def _write_images(
    ctx: OperationContext,
    result: Any,
    source: str,
    timing_metadata: JsonDict | None = None,
    *,
    adapter: str = "external",
) -> OperationResult:
    images, output_metadata = _result_array(result, np.uint8, "images")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise RuntimeError("External decoder must return images shaped [N,H,W,3]")
    output_metadata.update({"adapter": adapter, "source": source, "shape": list(images.shape)})
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metadata={
            "shape": list(images.shape),
            **({"codec_timing": timing_metadata} if timing_metadata else {}),
        },
    )


def _infer_codebook_size(indices: np.ndarray) -> int:
    return max(int(indices.max()) + 1, 2) if indices.size else 2


def _example_count(array: np.ndarray) -> int:
    if array.ndim >= 4:
        return max(int(array.shape[0]), 1)
    return 1


def _example_count_from_metadata(metadata: JsonDict, array: np.ndarray) -> int:
    raw_source_item_count = metadata.get("source_item_count")
    if raw_source_item_count is not None:
        try:
            source_item_count = int(raw_source_item_count)
        except (TypeError, ValueError) as exc:
            raise OperationError("source_item_count must be an integer") from exc
        if source_item_count <= 0:
            raise OperationError("source_item_count must be greater than zero")
        return source_item_count
    for key in (
        "original_shape",
        "image_shape",
        "symbol_shape",
        "shape",
        "tensor_shape",
        "indices_shape",
    ):
        if key not in metadata:
            continue
        value = metadata.get(key)
        if not isinstance(value, (list, tuple)) or not value:
            raise OperationError("%s metadata must be a non-empty shape array" % key)
        try:
            count = int(value[0])
        except (TypeError, ValueError) as exc:
            raise OperationError("%s[0] must be an integer" % key) from exc
        if count <= 0:
            raise OperationError("%s[0] must be greater than zero" % key)
        return count
    return _example_count(array)


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
