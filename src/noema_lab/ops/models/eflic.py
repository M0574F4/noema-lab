from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
from urllib.request import urlopen

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core import dataplane
from noema_lab.core.downloads import download_verified_https
from noema_lab.core.operations import Operation, OperationContext, OperationResult, object_schema
from noema_lab.core.structured_input import (
    decode_strict_json_object,
    decode_strict_yaml_or_json,
)
from noema_lab.ops.models.onnx_evidence import onnxruntime_native_evidence
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata, timed_call

JsonDict = Dict[str, Any]

DEFAULT_REPO_PATH = ".noema/upstreams/EF-LIC"
DEFAULT_CHECKPOINT = ".noema/checkpoints/eflic/checkpoint.pth.tar"
DEFAULT_MODEL_URL = "https://raw.githubusercontent.com/SevenCTHU/EF-LIC/main/EF_LIC.py"
DEFAULT_CHECKPOINT_URL = "https://drive.google.com/file/d/1XrfmdUx0nFFBg9_ToVzz-A2jFZ5qiGR6/view?usp=sharing"
MAX_MODEL_SOURCE_DOWNLOAD_BYTES = 8 * 1024 * 1024
EFLIC_INDEX_OUTPUT_NAMES = ("z_inds", "y0_inds", "y1_inds", "y2_inds", "y3_inds")

EFLIC_CODING_INFO: JsonDict = {
    "algorithm": "fixed-length VQ/RVQ index packing",
    "coder": "none",
    "implementation": "EF-LIC official inference model + NumPy bit packing",
    "language": "Python/PyTorch + NumPy",
    "note": "EF-LIC removes entropy coding; transmitted bits are fixed-length packed VQ/RVQ indices selected by force_ind.",
}


def _coding_metadata() -> JsonDict:
    return {
        "entropy_coder": EFLIC_CODING_INFO["coder"],
        "entropy_algorithm": EFLIC_CODING_INFO["algorithm"],
        "entropy_implementation": EFLIC_CODING_INFO["implementation"],
        "entropy_language": EFLIC_CODING_INFO["language"],
        "entropy_note": EFLIC_CODING_INFO["note"],
    }


def _asset_schema() -> JsonDict:
    return {
        "repo_path": {
            "type": "string",
            "default": DEFAULT_REPO_PATH,
            "description": "Local path containing the official EF_LIC.py inference file.",
        },
        "checkpoint": {
            "type": "string",
            "default": DEFAULT_CHECKPOINT,
            "description": "Local path to the pretrained EF-LIC checkpoint.pth.tar file.",
        },
        "model_url": {
            "type": "string",
            "default": DEFAULT_MODEL_URL,
            "description": "Raw EF_LIC.py URL used when auto setup is enabled and repo_path is missing the file.",
        },
        "checkpoint_url": {
            "type": "string",
            "default": DEFAULT_CHECKPOINT_URL,
            "description": "Official checkpoint page. Browser download is often more reliable than automated Google Drive download.",
        },
        "auto_setup": {
            "type": "boolean",
            "default": False,
            "description": "Opt in to downloading EF_LIC.py. An expected_model_sha256 is mandatory when enabled.",
        },
        "expected_model_sha256": {
            "type": "string",
            "pattern": "^[0-9a-fA-F]{64}$",
            "description": "Required SHA-256 for EF_LIC.py when auto setup is enabled; also verifies an existing local file when supplied.",
        },
        "expected_checkpoint_sha256": {
            "type": "string",
            "pattern": "^[0-9a-fA-F]{64}$",
            "description": "Optional expected SHA-256 for the local EF-LIC checkpoint; release recipes must supply it.",
        },
        "force_ind": {
            "type": "integer",
            "default": 2,
            "minimum": 0,
            "maximum": 4,
            "description": "EF-LIC rate point. The official inference script evaluates force_ind values 0, 1, 2, 3, and 4.",
        },
        "device": {"type": "string", "default": "cpu"},
        "pad_to_multiple": {
            "type": "integer",
            "default": 64,
            "minimum": 1,
            "description": "EF-LIC pads inputs to multiples of 64 using replicate padding.",
        },
        "data_plane_backend": dataplane.backend_schema("auto"),
    }


class EfLicEncodeOperation(Operation):
    id = "model.eflic_encode"
    name = "EF-LIC encoder to fixed-length payload bits"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema(_asset_schema())

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        _prepare_eflic_assets(ctx.params)
        images, input_metadata = _load_images(ctx.require_input("images").path)
        runner = _EfLicRunner(ctx.params)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        force_ind = runner.force_ind
        timing_records = []
        setup_start = time.perf_counter()
        network = runner.model()
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        entries = []
        bit_chunks = []
        bit_offset = 0
        _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
        for index in range(int(images.shape[0])):
            image = _image_for_index(images, input_metadata, index)
            image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
            total_start = time.perf_counter()
            frame, padded_shape = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: runner.image_to_tensor(torch, image),
                image_shape=image_shape,
            )
            _sync_torch(torch, runner.device)
            inds = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda frame=frame: network.compress(frame, force_ind=force_ind),
                image_shape=image_shape,
            )
            _sync_torch(torch, runner.device)
            raw_bits, meta = timed_call(
                timing_records,
                index,
                "encoder.symbol_encode",
                lambda inds=inds: _pack_inds(network, inds, backend),
                image_shape=image_shape,
            )
            entries.append(
                {
                    "bit_offset": int(bit_offset),
                    "bit_count": int(raw_bits.size),
                    "index_meta": meta,
                    "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                    "padded_shape": list(padded_shape),
                    "force_ind": int(force_ind),
                }
            )
            bit_chunks.append(raw_bits)
            bit_offset += int(raw_bits.size)
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=image_shape,
            )
            _report_example_progress(ctx, index + 1, int(images.shape[0]), "encoding")

        bits = np.concatenate(bit_chunks).astype(np.uint8, copy=False) if bit_chunks else np.zeros((0,), dtype=np.uint8)
        model_artifact_bytes = _file_size_or_zero(runner.checkpoint)
        model_artifact_metrics = {
            "model_artifact.encoder_bytes": model_artifact_bytes,
            "model_artifact.decoder_bytes": model_artifact_bytes,
            "model_artifact.total_bytes": model_artifact_bytes,
        }
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "eflic",
                "model": "EF-LIC",
                "force_ind": int(force_ind),
                "checkpoint": runner.checkpoint,
                "checkpoint_sha256": runner.checkpoint_sha256,
                "repo_path": runner.repo_path,
                "model_source_sha256": runner.model_source_sha256,
                **model_artifact_metrics,
                "bit_count": int(bits.size),
                "byte_count": int((bits.size + 7) // 8),
                "bit_role": "payload",
                "payload_bit_count": int(bits.size),
                "payload_format": "eflic.fixed_length_indices",
                "source_bit_storage": "unpacked_uint8",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": False,
                "fixed_length_coded": True,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "indices_to_bits"),
                "entries": entries,
                "original_shape": list(images.shape),
                **_coding_metadata(),
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": int((bits.size + 7) // 8),
                "channel.payload_bit_count": int(bits.size),
                **model_artifact_metrics,
            },
            metadata={
                "model": "EF-LIC",
                "force_ind": int(force_ind),
                **model_artifact_metrics,
                **_coding_metadata(),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "encoder.model": "EF-LIC neural analysis, decorrelation, and VQ/RVQ index generation.",
                        "encoder.symbol_encode": "NumPy fixed-length packing of EF-LIC VQ/RVQ indices; no entropy coding is used.",
                    },
                ),
            },
        )


class EfLicDecodeOperation(Operation):
    id = "model.eflic_decode"
    name = "EF-LIC fixed-length payload bits decoder to image batch"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema(
        {
            **_asset_schema(),
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        params = dict(ctx.params)
        for key in ("repo_path", "checkpoint", "model_url", "checkpoint_url", "auto_setup", "expected_model_sha256", "expected_checkpoint_sha256", "device", "pad_to_multiple"):
            if key in metadata and (key not in params or not str(params.get(key) or "").strip()):
                params[key] = metadata[key]
        if "force_ind" not in params or params.get("force_ind") is None:
            params["force_ind"] = int(metadata.get("force_ind", 2))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("data_plane_backend", "auto")))
        entries = _validated_eflic_payload_entries(
            bits,
            metadata,
            "EF-LIC payload",
        )
        try:
            _prepare_eflic_assets(params)
            runner = _EfLicRunner(params)
            timing_records = []
            setup_start = time.perf_counter()
            network = runner.model()
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "decoding")
            for index, entry in enumerate(entries):
                total_start = time.perf_counter()
                offset = entry["bit_offset"]
                count = entry["bit_count"]
                chunk = bits[offset : offset + count]
                inds = timed_call(
                    timing_records,
                    index,
                    "decoder.symbol_decode",
                    lambda chunk=chunk, entry=entry: _unpack_inds(chunk, entry["index_meta"], runner.device, torch, backend),
                    image_shape=entry.get("original_shape"),
                )
                _sync_torch(torch, runner.device)
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda inds=inds, entry=entry: network.decompress(inds, force_ind=int(entry.get("force_ind", runner.force_ind))),
                    image_shape=entry.get("original_shape"),
                )
                _sync_torch(torch, runner.device)
                decoded.append(_eflic_tensor_to_image(output, entry["original_shape"]))
                append_measurement(
                    timing_records,
                    index,
                    "decoder.total",
                    time.perf_counter() - total_start,
                    image_shape=entry.get("original_shape"),
                )
                _report_example_progress(ctx, index + 1, len(entries), "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("EF-LIC decode failed, likely due to corrupted fixed-length payload bits: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
            timing_records = []
            decoded_shapes = [[1, int(shape[1]), int(shape[2]), int(shape[3])]] if len(shape) == 4 else []

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "source": "eflic_decode",
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "original_shapes": decoded_shapes,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "bits_to_indices"),
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        model_artifact_bytes = _file_size_or_zero(str(params.get("checkpoint") or metadata.get("checkpoint") or DEFAULT_CHECKPOINT))
        model_artifact_metrics = {
            "model_artifact.encoder_bytes": model_artifact_bytes,
            "model_artifact.decoder_bytes": model_artifact_bytes,
            "model_artifact.total_bytes": model_artifact_bytes,
        }
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metrics=model_artifact_metrics,
            metadata={
                "shape": list(images.shape),
                **model_artifact_metrics,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch",
                    notes={
                        "decoder.symbol_decode": "NumPy fixed-length unpacking of EF-LIC VQ/RVQ indices; no entropy decoding is used.",
                        "decoder.model": "EF-LIC neural synthesis from fixed-length VQ/RVQ indices.",
                    },
                ),
            },
        )


class EfLicEncodeIndicesOperation(Operation):
    id = "model.eflic_encode_indices"
    name = "EF-LIC encoder to VQ/RVQ indices"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema(_asset_schema())

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        _prepare_eflic_assets(ctx.params)
        images, input_metadata = _load_images(ctx.require_input("images").path)
        runner = _EfLicRunner(ctx.params)
        timing_records = []
        setup_start = time.perf_counter()
        network = runner.model()
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)

        arrays: JsonDict = {}
        entries = []
        total = int(images.shape[0])
        _report_example_progress(ctx, 0, total, "encoding")
        for index in range(total):
            image = _image_for_index(images, input_metadata, index)
            image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
            total_start = time.perf_counter()
            frame, padded_shape = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: runner.image_to_tensor(torch, image),
                image_shape=image_shape,
            )
            _sync_torch(torch, runner.device)
            inds = timed_call(
                timing_records,
                index,
                "encoder.inference",
                lambda frame=frame: network.compress(frame, force_ind=runner.force_ind),
                image_shape=image_shape,
            )
            _sync_torch(torch, runner.device)
            entry, entry_arrays = _eflic_torch_indices_to_entry(
                network,
                inds,
                index,
                [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                list(padded_shape),
                list(padded_shape),
                runner.force_ind,
            )
            arrays.update(entry_arrays)
            entries.append(entry)
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=image_shape,
            )
            _report_example_progress(ctx, index + 1, total, "encoding")

        model_artifact_bytes = _file_size_or_zero(runner.checkpoint)
        model_artifact_metrics = {
            "model_artifact.encoder_bytes": model_artifact_bytes,
            "model_artifact.decoder_bytes": model_artifact_bytes,
            "model_artifact.total_bytes": model_artifact_bytes,
        }
        metadata = dict(input_metadata)
        metadata.update(
            {
                "codec": "eflic",
                "model": "EF-LIC",
                "source": "eflic_encode_indices",
                "semantic_form": "indices",
                "representation": "eflic.vq_rvq_indices",
                "force_ind": int(runner.force_ind),
                "checkpoint": runner.checkpoint,
                "checkpoint_sha256": runner.checkpoint_sha256,
                "repo_path": runner.repo_path,
                "model_source_sha256": runner.model_source_sha256,
                **model_artifact_metrics,
                "entries": entries,
                "original_shape": list(images.shape),
                "dtype": "int64",
                "entropy_coded": False,
                "fixed_length_coded": False,
                **_coding_metadata(),
            }
        )
        path = ctx.output_path("indices", ".npz")
        np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)
        return OperationResult(
            outputs={"indices": artifact("semantic.indices.numpy", path, metadata)},
            metrics=model_artifact_metrics,
            metadata={
                "model": "EF-LIC",
                "force_ind": int(runner.force_ind),
                **model_artifact_metrics,
                **_coding_metadata(),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="pytorch",
                    notes={"encoder.inference": "EF-LIC neural analysis, decorrelation, and VQ/RVQ index generation."},
                ),
            },
        )


class EfLicIndicesToBitsOperation(Operation):
    id = "model.eflic_indices_to_bits"
    name = "EF-LIC payload encoder from VQ/RVQ indices to bits"
    input_kinds = {"indices": ["semantic.indices.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema({"data_plane_backend": dataplane.backend_schema("auto")})

    def run(self, ctx: OperationContext) -> OperationResult:
        arrays, input_metadata = _load_eflic_indices(ctx.require_input("indices").path, ctx.require_input("indices").metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        timing_records = []
        entries = []
        bit_chunks = []
        bit_offset = 0
        source_entries = input_metadata.get("entries") or []
        _report_example_progress(ctx, 0, len(source_entries), "payload encoding")
        for index, source_entry in enumerate(source_entries):
            image_shape = source_entry.get("original_shape")
            total_start = time.perf_counter()
            raw_bits, meta = timed_call(
                timing_records,
                index,
                "encoder.symbol_encode",
                lambda source_entry=source_entry: _pack_eflic_entry_arrays(arrays, source_entry, backend),
                image_shape=image_shape,
            )
            entry = {
                "bit_offset": int(bit_offset),
                "bit_count": int(raw_bits.size),
                "index_meta": meta,
                "original_shape": source_entry.get("original_shape"),
                "padded_shape": source_entry.get("padded_shape"),
                "model_input_shape": source_entry.get("model_input_shape") or source_entry.get("padded_shape"),
                "force_ind": int(source_entry.get("force_ind", input_metadata.get("force_ind", 2))),
            }
            entries.append(entry)
            bit_chunks.append(raw_bits)
            bit_offset += int(raw_bits.size)
            append_measurement(
                timing_records,
                index,
                "encoder.total",
                time.perf_counter() - total_start,
                image_shape=image_shape,
            )
            _report_example_progress(ctx, index + 1, len(source_entries), "payload encoding")

        bits = np.concatenate(bit_chunks).astype(np.uint8, copy=False) if bit_chunks else np.zeros((0,), dtype=np.uint8)
        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "eflic_indices_to_bits",
                "bit_count": int(bits.size),
                "byte_count": int((bits.size + 7) // 8),
                "bit_role": "payload",
                "payload_bit_count": int(bits.size),
                "payload_format": "eflic.fixed_length_indices",
                "source_bit_storage": "unpacked_uint8",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "entropy_coded": False,
                "fixed_length_coded": True,
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "indices_to_bits"),
                "entries": entries,
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
            metrics={
                "codec.bit_count": int(bits.size),
                "codec.bytes": int((bits.size + 7) // 8),
                "channel.payload_bit_count": int(bits.size),
            },
            metadata={
                **_coding_metadata(),
                "codec_timing": codec_timing_metadata(
                    "encoder",
                    timing_records,
                    runner="numpy_payload",
                    notes={"encoder.symbol_encode": "NumPy fixed-length packing of EF-LIC VQ/RVQ indices; no entropy coding is used."},
                ),
            },
        )


class EfLicBitsToIndicesOperation(Operation):
    id = "model.eflic_bits_to_indices"
    name = "EF-LIC payload decoder from bits to VQ/RVQ indices"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema({"data_plane_backend": dataplane.backend_schema("auto")})

    def run(self, ctx: OperationContext) -> OperationResult:
        bits, input_metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("data_plane_backend", "auto")))
        timing_records = []
        arrays: JsonDict = {}
        entries = []
        source_entries = _validated_eflic_payload_entries(
            bits,
            input_metadata,
            "EF-LIC payload",
        )
        _report_example_progress(ctx, 0, len(source_entries), "payload decoding")
        for index, source_entry in enumerate(source_entries):
            image_shape = source_entry.get("original_shape")
            total_start = time.perf_counter()
            offset = source_entry["bit_offset"]
            count = source_entry["bit_count"]
            chunk = bits[offset : offset + count]
            decoded_arrays = timed_call(
                timing_records,
                index,
                "decoder.symbol_decode",
                lambda chunk=chunk, source_entry=source_entry: _unpack_inds_numpy(chunk, source_entry["index_meta"], backend),
                image_shape=image_shape,
            )
            entry, entry_arrays = _eflic_numpy_indices_to_entry(
                decoded_arrays,
                source_entry.get("index_meta") or [],
                index,
                source_entry.get("original_shape"),
                source_entry.get("padded_shape"),
                source_entry.get("model_input_shape") or source_entry.get("padded_shape"),
                int(source_entry.get("force_ind", input_metadata.get("force_ind", 2))),
            )
            arrays.update(entry_arrays)
            entries.append(entry)
            append_measurement(
                timing_records,
                index,
                "decoder.total",
                time.perf_counter() - total_start,
                image_shape=image_shape,
            )
            _report_example_progress(ctx, index + 1, len(source_entries), "payload decoding")

        metadata = dict(input_metadata)
        metadata.update(
            {
                "source": "eflic_bits_to_indices",
                "semantic_form": "indices",
                "representation": "eflic.vq_rvq_indices",
                "entries": entries,
                "dtype": "int64",
                "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "bits_to_indices"),
            }
        )
        path = ctx.output_path("indices", ".npz")
        np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)
        return OperationResult(
            outputs={"indices": artifact("semantic.indices.numpy", path, metadata)},
            metadata={
                "shape": metadata.get("original_shape"),
                **_coding_metadata(),
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="numpy_payload",
                    notes={"decoder.symbol_decode": "NumPy fixed-length unpacking of EF-LIC VQ/RVQ indices; no entropy decoding is used."},
                ),
            },
        )


class EfLicDecodeIndicesOperation(Operation):
    id = "model.eflic_decode_indices"
    name = "EF-LIC decoder from VQ/RVQ indices to image batch"
    input_kinds = {"indices": ["semantic.indices.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = EFLIC_CODING_INFO
    params_schema = object_schema(
        {
            **_asset_schema(),
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        arrays, metadata = _load_eflic_indices(ctx.require_input("indices").path, ctx.require_input("indices").metadata)
        params = _eflic_params_from_metadata(ctx.params, metadata)
        try:
            _prepare_eflic_assets(params)
            runner = _EfLicRunner(params)
            timing_records = []
            setup_start = time.perf_counter()
            network = runner.model()
            append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
            entries = metadata.get("entries") or []
            if not entries:
                raise RuntimeError("EF-LIC indices metadata does not contain entries")
            decoded = []
            _report_example_progress(ctx, 0, len(entries), "decoding")
            for index, entry in enumerate(entries):
                total_start = time.perf_counter()
                inds = _eflic_entry_arrays_to_torch(arrays, entry, runner.device, torch)
                _sync_torch(torch, runner.device)
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.inference",
                    lambda inds=inds, entry=entry: network.decompress(inds, force_ind=int(entry.get("force_ind", runner.force_ind))),
                    image_shape=entry.get("original_shape"),
                )
                _sync_torch(torch, runner.device)
                decoded.append(_eflic_tensor_to_image(output, entry["original_shape"]))
                append_measurement(
                    timing_records,
                    index,
                    "decoder.total",
                    time.perf_counter() - total_start,
                    image_shape=entry.get("original_shape"),
                )
                _report_example_progress(ctx, index + 1, len(entries), "decoding")
            images, decoded_shapes = _stack_image_list(decoded)
        except Exception as exc:
            if str(ctx.params.get("on_error", "fail")) != "zeros":
                raise RuntimeError("EF-LIC index decode failed: %s" % exc) from exc
            shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
            images = np.zeros(shape, dtype=np.uint8)
            timing_records = []
            decoded_shapes = [[1, int(shape[1]), int(shape[2]), int(shape[3])]] if len(shape) == 4 else []

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "source": "eflic_decode_indices",
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "original_shapes": decoded_shapes,
            }
        )
        path = ctx.output_path("images", ".npz")
        np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
        model_artifact_bytes = _file_size_or_zero(str(params.get("checkpoint") or metadata.get("checkpoint") or DEFAULT_CHECKPOINT))
        model_artifact_metrics = {
            "model_artifact.encoder_bytes": model_artifact_bytes,
            "model_artifact.decoder_bytes": model_artifact_bytes,
            "model_artifact.total_bytes": model_artifact_bytes,
        }
        return OperationResult(
            outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
            metrics=model_artifact_metrics,
            metadata={
                "shape": list(images.shape),
                **model_artifact_metrics,
                "codec_timing": codec_timing_metadata(
                    "decoder",
                    timing_records,
                    runner="pytorch",
                    notes={"decoder.inference": "EF-LIC neural synthesis from fixed-length VQ/RVQ indices."},
                ),
            },
        )


class EfLicOnnxExportOperation(Operation):
    id = "model.eflic_export_onnx"
    name = "Export EF-LIC fixed-shape ONNX Runtime bundle"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"model": "model.onnx.bundle"}
    params_schema = object_schema(
        {
            **_asset_schema(),
            "opset": {"type": "integer", "default": 18, "minimum": 18},
            "validate_export": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "onnx",
            ["torch", "onnx", "onnxruntime"],
            "EF-LIC ONNX Runtime",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        torch = _require_torch()
        onnx, ort = _require_onnx_stack(require_onnx=True)
        _prepare_eflic_assets(ctx.params)
        images, image_metadata = _load_images(ctx.require_input("images").path)
        runner = _EfLicRunner(ctx.params)
        force_ind = runner.force_ind
        network = runner.model()
        sample_arrays = _eflic_export_sample_arrays(images, image_metadata, runner.pad_to_multiple)
        if not sample_arrays:
            raise RuntimeError("EF-LIC ONNX export could not find any input image shapes to export")
        bundle_dir = ctx.step_dir / "eflic_onnx_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shape_packages: JsonDict = {}
        all_encoder_paths = []
        all_decoder_paths = []
        default_package = None
        total_steps = max(1, len(sample_arrays) * 3 + 1)
        completed = 0

        def report(message: str) -> None:
            nonlocal completed
            completed += 1
            ctx.report_progress(
                message,
                phase="conversion",
                completed=completed,
                total=total_steps,
                percent=float(completed) / float(total_steps) * 100.0,
            )

        for package_index, (shape_key, sample_array) in enumerate(sample_arrays.items()):
            suffix = "" if package_index == 0 else "_%s" % shape_key
            encoder_path = bundle_dir / ("encoder%s.onnx" % suffix)
            decoder_path = bundle_dir / ("decoder%s.onnx" % suffix)
            sample_tensor = torch.from_numpy(sample_array).to(runner.device).contiguous()
            encoder = _EfLicEncodeWrapper(network, force_ind).to(runner.device).eval()
            decoder = _EfLicDecodeWrapper(network, force_ind).to(runner.device).eval()
            report("Exporting EF-LIC encoder for %s" % shape_key)
            try:
                with torch.no_grad():
                    index_tensors = encoder(sample_tensor)
                torch.onnx.export(
                    encoder,
                    sample_tensor,
                    str(encoder_path),
                    input_names=["images"],
                    output_names=list(EFLIC_INDEX_OUTPUT_NAMES),
                    opset_version=int(ctx.params.get("opset", 18)),
                )
            except Exception as exc:
                raise RuntimeError("EF-LIC encoder could not be exported to ONNX for %s: %s" % (shape_key, exc)) from exc
            report("Exporting EF-LIC decoder for %s" % shape_key)
            try:
                with torch.no_grad():
                    decoded = decoder(*index_tensors)
                torch.onnx.export(
                    decoder,
                    tuple(index_tensors),
                    str(decoder_path),
                    input_names=list(EFLIC_INDEX_OUTPUT_NAMES),
                    output_names=["images"],
                    opset_version=int(ctx.params.get("opset", 18)),
                )
            except Exception as exc:
                raise RuntimeError("EF-LIC decoder could not be exported to ONNX for %s: %s" % (shape_key, exc)) from exc
            if bool(ctx.params.get("validate_export", True)):
                report("Validating EF-LIC ONNX bundle for %s" % shape_key)
                try:
                    onnx.checker.check_model(str(encoder_path))
                    onnx.checker.check_model(str(decoder_path))
                    providers = _onnx_providers("CPUExecutionProvider", ort)
                    encoder_session = ort.InferenceSession(str(encoder_path), providers=providers)
                    decoder_session = ort.InferenceSession(str(decoder_path), providers=providers)
                    encoded = encoder_session.run(None, {"images": np.ascontiguousarray(sample_array)})
                    expected_indices = [_torch_to_numpy(tensor).astype(np.int64, copy=False) for tensor in index_tensors]
                    if not all(np.array_equal(actual, expected) for actual, expected in zip(encoded, expected_indices)):
                        raise RuntimeError("encoder index outputs differ from PyTorch reference")
                    decoded_onnx = decoder_session.run(
                        None,
                        {name: value.astype(np.int64, copy=False) for name, value in zip(EFLIC_INDEX_OUTPUT_NAMES, encoded)},
                    )[0]
                    _assert_close("EF-LIC ONNX decoder", decoded_onnx, _torch_to_numpy(decoded))
                except Exception as exc:
                    raise RuntimeError("EF-LIC ONNX bundle failed validation for %s: %s" % (shape_key, exc)) from exc
            else:
                report("Prepared EF-LIC ONNX bundle for %s" % shape_key)
            package = {
                "input_shape": [int(item) for item in sample_array.shape],
                "index_shapes": {
                    name: [int(item) for item in tensor.shape]
                    for name, tensor in zip(EFLIC_INDEX_OUTPUT_NAMES, index_tensors)
                },
                "analysis_path": str(encoder_path),
                "synthesis_path": str(decoder_path),
            }
            shape_packages[shape_key] = package
            all_encoder_paths.append(encoder_path)
            all_decoder_paths.append(decoder_path)
            if default_package is None:
                default_package = package

        artifact_sizes = _model_artifact_sizes(all_encoder_paths, all_decoder_paths)
        metadata = {
            "codec": "eflic",
            "model": "EF-LIC",
            "source_runtime": "pytorch",
            "runtime": "onnxruntime",
            "format": "onnx",
            "force_ind": int(force_ind),
            "repo_path": runner.repo_path,
            "checkpoint": runner.checkpoint,
            "model_source_sha256": runner.model_source_sha256,
            "checkpoint_sha256": runner.checkpoint_sha256,
            "pad_to_multiple": int(runner.pad_to_multiple),
            "opset": int(ctx.params.get("opset", 18)),
            "representation": "bits",
            "payload_format": "eflic.fixed_length_indices",
            "entropy_coded": False,
            "fixed_length_coded": True,
            "n_e": [int(item) for item in network.n_e],
            "analysis_path": str(default_package["analysis_path"]),
            "synthesis_path": str(default_package["synthesis_path"]),
            "export_input_shape": list(default_package["input_shape"]),
            "available_input_shapes": [list(package["input_shape"]) for package in shape_packages.values()],
            "shape_packages": shape_packages,
            "data_input_shape": list(images.shape),
            "data_image_ids": image_metadata.get("image_ids"),
            **artifact_sizes,
            **_coding_metadata(),
            "note": "EF-LIC ONNX bundle exports fixed-shape neural compress/decompress wrappers; NumPy fixed-length index packing remains the payload adapter.",
        }
        manifest_path = ctx.output_path("onnx_bundle", ".json")
        manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        ctx.report_progress("Prepared EF-LIC ONNX bundle", phase="conversion", completed=total_steps, total=total_steps, percent=100.0)
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


class EfLicAotInductorExportOperation(Operation):
    id = "model.eflic_export_aoti"
    name = "Compile EF-LIC fixed-shape wrappers with AOT Inductor"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"model": "model.aot_inductor.bundle"}
    params_schema = object_schema(
        {
            **_asset_schema(),
            "validate_export": {"type": "boolean", "default": True},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            "eflic",
            ["torch"],
            "EF-LIC AOT Inductor",
        )
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        start = time.perf_counter()
        torch = _require_torch()
        _ensure_aoti_available(torch)
        _prepare_eflic_assets(ctx.params)
        images, image_metadata = _load_images(ctx.require_input("images").path)
        runner = _EfLicRunner(ctx.params)
        force_ind = runner.force_ind
        network = runner.model()
        sample_arrays = _eflic_export_sample_arrays(images, image_metadata, runner.pad_to_multiple)
        if not sample_arrays:
            raise RuntimeError("EF-LIC AOT Inductor export could not find any input image shapes to compile")
        bundle_dir = ctx.step_dir / "eflic_aoti_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shape_packages: JsonDict = {}
        all_encoder_paths = []
        all_decoder_paths = []
        default_package = None
        total_steps = max(1, len(sample_arrays) * 3 + 1)
        completed = 0

        def report(message: str) -> None:
            nonlocal completed
            completed += 1
            ctx.report_progress(
                message,
                phase="conversion",
                completed=completed,
                total=total_steps,
                percent=float(completed) / float(total_steps) * 100.0,
            )

        for package_index, (shape_key, sample_array) in enumerate(sample_arrays.items()):
            suffix = "" if package_index == 0 else "_%s" % shape_key
            encoder_path = bundle_dir / ("encoder%s.pt2" % suffix)
            decoder_path = bundle_dir / ("decoder%s.pt2" % suffix)
            sample_tensor = torch.from_numpy(sample_array).to(runner.device).contiguous()
            encoder = _EfLicEncodeWrapper(network, force_ind).to(runner.device).eval()
            decoder = _EfLicDecodeWrapper(network, force_ind).to(runner.device).eval()
            with torch.no_grad():
                index_tensors = encoder(sample_tensor)
                decoded = decoder(*index_tensors)

            report("Compiling EF-LIC encoder for %s" % shape_key)
            try:
                _save_aoti_package(torch, encoder, (sample_tensor,), encoder_path, "EF-LIC encoder %s" % shape_key)
            except Exception as exc:
                raise RuntimeError("EF-LIC encoder could not be compiled with AOT Inductor for %s: %s" % (shape_key, exc)) from exc

            report("Compiling EF-LIC decoder for %s" % shape_key)
            try:
                _save_aoti_package(torch, decoder, tuple(index_tensors), decoder_path, "EF-LIC decoder %s" % shape_key)
            except Exception as exc:
                raise RuntimeError("EF-LIC decoder could not be compiled with AOT Inductor for %s: %s" % (shape_key, exc)) from exc

            if bool(ctx.params.get("validate_export", True)):
                report("Validating EF-LIC AOT Inductor bundle for %s" % shape_key)
                try:
                    encoder_session = _EfLicAotInductorSession(torch, encoder_path, runner.device)
                    decoder_session = _EfLicAotInductorSession(torch, decoder_path, runner.device)
                    encoded = encoder_session.run(sample_array)
                    expected_indices = [_torch_to_numpy(tensor).astype(np.int64, copy=False) for tensor in index_tensors]
                    if not all(np.array_equal(actual.astype(np.int64, copy=False), expected) for actual, expected in zip(encoded, expected_indices)):
                        raise RuntimeError("encoder index outputs differ from PyTorch reference")
                    decoded_aoti = decoder_session.run(encoded)[0]
                    _assert_close("EF-LIC AOT Inductor decoder", decoded_aoti, _torch_to_numpy(decoded))
                except Exception as exc:
                    raise RuntimeError("EF-LIC AOT Inductor bundle failed validation for %s: %s" % (shape_key, exc)) from exc
            else:
                report("Prepared EF-LIC AOT Inductor bundle for %s" % shape_key)

            package = {
                "input_shape": [int(item) for item in sample_array.shape],
                "index_shapes": {
                    name: [int(item) for item in tensor.shape]
                    for name, tensor in zip(EFLIC_INDEX_OUTPUT_NAMES, index_tensors)
                },
                "analysis_path": str(encoder_path),
                "synthesis_path": str(decoder_path),
            }
            shape_packages[shape_key] = package
            all_encoder_paths.append(encoder_path)
            all_decoder_paths.append(decoder_path)
            if default_package is None:
                default_package = package

        artifact_sizes = _model_artifact_sizes(all_encoder_paths, all_decoder_paths)
        metadata = {
            "codec": "eflic",
            "model": "EF-LIC",
            "source_runtime": "pytorch",
            "runtime": "aot_inductor",
            "format": "pt2_aoti",
            "force_ind": int(force_ind),
            "repo_path": runner.repo_path,
            "checkpoint": runner.checkpoint,
            "model_source_sha256": runner.model_source_sha256,
            "checkpoint_sha256": runner.checkpoint_sha256,
            "pad_to_multiple": int(runner.pad_to_multiple),
            "representation": "bits",
            "payload_format": "eflic.fixed_length_indices",
            "entropy_coded": False,
            "fixed_length_coded": True,
            "n_e": [int(item) for item in network.n_e],
            "analysis_path": str(default_package["analysis_path"]),
            "synthesis_path": str(default_package["synthesis_path"]),
            "export_input_shape": list(default_package["input_shape"]),
            "available_input_shapes": [list(package["input_shape"]) for package in shape_packages.values()],
            "shape_packages": shape_packages,
            "data_input_shape": list(images.shape),
            "data_image_ids": image_metadata.get("image_ids"),
            **artifact_sizes,
            **_coding_metadata(),
            "note": "EF-LIC AOT Inductor bundle compiles fixed-shape neural compress/decompress wrappers; NumPy fixed-length index packing remains the payload adapter.",
        }
        manifest_path = ctx.output_path("aoti_bundle", ".json")
        manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        ctx.report_progress("Prepared EF-LIC AOT Inductor bundle", phase="conversion", completed=total_steps, total=total_steps, percent=100.0)
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


class EfLicOnnxEncodeOperation(Operation):
    id = "model.eflic_onnx_encode"
    name = "EF-LIC ONNX Runtime encoder to fixed-length payload bits"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "ONNX Runtime EF-LIC compress wrapper + NumPy bit packing",
        "language": "ONNX Runtime + Python/NumPy",
    }
    params_schema = object_schema({"provider": {"type": "string", "default": "CPUExecutionProvider"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("onnx", ["onnxruntime"], "EF-LIC ONNX Runtime")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        _report_setup_progress(ctx, "Creating EF-LIC ONNX Runtime encoder sessions")
        setup_start = time.perf_counter()
        sessions = _eflic_onnx_sessions(bundle, ort, providers, "analysis_path")
        runtime_evidence = _eflic_onnx_bundle_evidence(bundle, ort, sessions, providers[0], "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: sessions[_eflic_shape_key(package["input_shape"])].run(None, {"images": array}),
            "eflic_onnx_encode",
            "onnxruntime",
            runtime_evidence,
        )


class EfLicAotInductorEncodeOperation(Operation):
    id = "model.eflic_aoti_encode"
    name = "EF-LIC AOT Inductor encoder to fixed-length payload bits"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "AOT Inductor EF-LIC compress wrapper + NumPy bit packing",
        "language": "PyTorch AOT Inductor + Python/NumPy",
    }
    params_schema = object_schema({"device": {"type": "string", "default": "cpu"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device") or bundle.get("device") or "cpu")
        timing_records = []
        _report_setup_progress(ctx, "Loading EF-LIC AOT Inductor encoder packages")
        setup_start = time.perf_counter()
        sessions = _eflic_aoti_sessions(bundle, torch, device, "analysis_path")
        runtime_evidence = _eflic_aoti_bundle_evidence(bundle, torch, sessions, device, "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: sessions[_eflic_shape_key(package["input_shape"])].run(array),
            "eflic_aoti_encode",
            "aot_inductor",
            runtime_evidence,
        )


class EfLicOpenVinoEncodeOperation(Operation):
    id = "model.eflic_openvino_encode"
    name = "EF-LIC OpenVINO encoder to fixed-length payload bits"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "OpenVINO EF-LIC compress wrapper + NumPy bit packing",
        "language": "OpenVINO + Python/NumPy",
    }
    params_schema = object_schema({"device": {"type": "string", "default": "CPU"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("openvino", ["openvino"], "EF-LIC OpenVINO")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        _report_setup_progress(ctx, "Compiling EF-LIC OpenVINO encoder models")
        setup_start = time.perf_counter()
        compiled = _eflic_openvino_sessions(bundle, ov, device, "analysis_path")
        runtime_evidence = _eflic_openvino_bundle_evidence(bundle, ov, compiled, device, "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: _openvino_run_named_outputs(compiled[_eflic_shape_key(package["input_shape"])], {"images": array}),
            "eflic_openvino_encode",
            "openvino",
            runtime_evidence,
        )


class EfLicAotInductorDecodeOperation(Operation):
    id = "model.eflic_aoti_decode"
    name = "EF-LIC AOT Inductor fixed-length payload bits decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "NumPy bit unpacking + AOT Inductor EF-LIC decompress wrapper",
        "language": "Python/NumPy + PyTorch AOT Inductor",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_eflic_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device") or bundle.get("device") or "cpu")
        timing_records = []
        _report_setup_progress(ctx, "Loading EF-LIC AOT Inductor decoder packages")
        setup_start = time.perf_counter()
        sessions = _eflic_aoti_sessions(bundle, torch, device, "synthesis_path")
        runtime_evidence = _eflic_aoti_bundle_evidence(bundle, torch, sessions, device, "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode(
            ctx,
            bits,
            metadata,
            bundle,
            timing_records,
            lambda package, arrays: sessions[_eflic_shape_key(package["input_shape"])].run(arrays)[0],
            "eflic_aoti_decode",
            "aot_inductor",
            runtime_evidence,
        )


class EfLicOnnxDecodeOperation(Operation):
    id = "model.eflic_onnx_decode"
    name = "EF-LIC ONNX Runtime fixed-length payload bits decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "NumPy bit unpacking + ONNX Runtime EF-LIC decompress wrapper",
        "language": "Python/NumPy + ONNX Runtime",
    }
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("onnx", ["onnxruntime"], "EF-LIC ONNX Runtime")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        _report_setup_progress(ctx, "Creating EF-LIC ONNX Runtime decoder sessions")
        setup_start = time.perf_counter()
        sessions = _eflic_onnx_sessions(bundle, ort, providers, "synthesis_path")
        runtime_evidence = _eflic_onnx_bundle_evidence(bundle, ort, sessions, providers[0], "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode(
            ctx,
            bits,
            metadata,
            bundle,
            timing_records,
            lambda package, arrays: sessions[_eflic_shape_key(package["input_shape"])].run(None, arrays)[0],
            "eflic_onnx_decode",
            "onnxruntime",
            runtime_evidence,
        )


class EfLicOpenVinoDecodeOperation(Operation):
    id = "model.eflic_openvino_decode"
    name = "EF-LIC OpenVINO fixed-length payload bits decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "NumPy bit unpacking + OpenVINO EF-LIC decompress wrapper",
        "language": "Python/NumPy + OpenVINO",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("openvino", ["openvino"], "EF-LIC OpenVINO")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        bits, metadata = _load_bits(ctx.require_input("bits").path, ctx.require_input("bits").metadata)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        _report_setup_progress(ctx, "Compiling EF-LIC OpenVINO decoder models")
        setup_start = time.perf_counter()
        compiled = _eflic_openvino_sessions(bundle, ov, device, "synthesis_path")
        runtime_evidence = _eflic_openvino_bundle_evidence(bundle, ov, compiled, device, "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode(
            ctx,
            bits,
            metadata,
            bundle,
            timing_records,
            lambda package, arrays: _openvino_run_named_outputs(compiled[_eflic_shape_key(package["input_shape"])], arrays)[0],
            "eflic_openvino_decode",
            "openvino",
            runtime_evidence,
        )


class EfLicOnnxEncodeIndicesOperation(Operation):
    id = "model.eflic_onnx_encode_indices"
    name = "EF-LIC ONNX Runtime encoder to VQ/RVQ indices"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "ONNX Runtime EF-LIC compress wrapper",
        "language": "ONNX Runtime",
    }
    params_schema = object_schema({"provider": {"type": "string", "default": "CPUExecutionProvider"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("onnx", ["onnxruntime"], "EF-LIC ONNX Runtime")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        _report_setup_progress(ctx, "Creating EF-LIC ONNX Runtime encoder sessions")
        setup_start = time.perf_counter()
        sessions = _eflic_onnx_sessions(bundle, ort, providers, "analysis_path")
        runtime_evidence = _eflic_onnx_bundle_evidence(bundle, ort, sessions, providers[0], "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode_indices(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: sessions[_eflic_shape_key(package["input_shape"])].run(None, {"images": array}),
            "eflic_onnx_encode_indices",
            "onnxruntime",
            runtime_evidence,
        )


class EfLicAotInductorEncodeIndicesOperation(Operation):
    id = "model.eflic_aoti_encode_indices"
    name = "EF-LIC AOT Inductor encoder to VQ/RVQ indices"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "AOT Inductor EF-LIC compress wrapper",
        "language": "PyTorch AOT Inductor",
    }
    params_schema = object_schema({"device": {"type": "string", "default": "cpu"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device") or bundle.get("device") or "cpu")
        timing_records = []
        _report_setup_progress(ctx, "Loading EF-LIC AOT Inductor encoder packages")
        setup_start = time.perf_counter()
        sessions = _eflic_aoti_sessions(bundle, torch, device, "analysis_path")
        runtime_evidence = _eflic_aoti_bundle_evidence(bundle, torch, sessions, device, "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode_indices(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: sessions[_eflic_shape_key(package["input_shape"])].run(array),
            "eflic_aoti_encode_indices",
            "aot_inductor",
            runtime_evidence,
        )


class EfLicOpenVinoEncodeIndicesOperation(Operation):
    id = "model.eflic_openvino_encode_indices"
    name = "EF-LIC OpenVINO encoder to VQ/RVQ indices"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "OpenVINO EF-LIC compress wrapper",
        "language": "OpenVINO",
    }
    params_schema = object_schema({"device": {"type": "string", "default": "CPU"}})

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("openvino", ["openvino"], "EF-LIC OpenVINO")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        images, input_metadata = _load_images(ctx.require_input("images").path)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        _report_setup_progress(ctx, "Compiling EF-LIC OpenVINO encoder models")
        setup_start = time.perf_counter()
        compiled = _eflic_openvino_sessions(bundle, ov, device, "analysis_path")
        runtime_evidence = _eflic_openvino_bundle_evidence(bundle, ov, compiled, device, "analysis_path")
        append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_encode_indices(
            ctx,
            images,
            input_metadata,
            bundle,
            timing_records,
            lambda package, array: _openvino_run_named_outputs(compiled[_eflic_shape_key(package["input_shape"])], {"images": array}),
            "eflic_openvino_encode_indices",
            "openvino",
            runtime_evidence,
        )


class EfLicAotInductorDecodeIndicesOperation(Operation):
    id = "model.eflic_aoti_decode_indices"
    name = "EF-LIC AOT Inductor decoder from VQ/RVQ indices"
    input_kinds = {"indices": ["semantic.indices.numpy"], "model": ["model.aot_inductor.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "AOT Inductor EF-LIC decompress wrapper",
        "language": "PyTorch AOT Inductor",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("eflic", ["torch"], "EF-LIC AOT Inductor")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        torch = _require_torch()
        arrays, metadata = _load_eflic_indices(ctx.require_input("indices").path, ctx.require_input("indices").metadata)
        bundle = _load_eflic_aoti_bundle(ctx.require_input("model"))
        device = str(ctx.params.get("device") or bundle.get("device") or "cpu")
        timing_records = []
        _report_setup_progress(ctx, "Loading EF-LIC AOT Inductor decoder packages")
        setup_start = time.perf_counter()
        sessions = _eflic_aoti_sessions(bundle, torch, device, "synthesis_path")
        runtime_evidence = _eflic_aoti_bundle_evidence(bundle, torch, sessions, device, "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode_indices(
            ctx,
            arrays,
            metadata,
            bundle,
            timing_records,
            lambda package, runtime_inputs: sessions[_eflic_shape_key(package["input_shape"])].run(runtime_inputs)[0],
            "eflic_aoti_decode_indices",
            "aot_inductor",
            runtime_evidence,
        )


class EfLicOnnxDecodeIndicesOperation(Operation):
    id = "model.eflic_onnx_decode_indices"
    name = "EF-LIC ONNX Runtime decoder from VQ/RVQ indices"
    input_kinds = {"indices": ["semantic.indices.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "ONNX Runtime EF-LIC decompress wrapper",
        "language": "ONNX Runtime",
    }
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("onnx", ["onnxruntime"], "EF-LIC ONNX Runtime")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        _, ort = _require_onnx_stack(require_onnx=False)
        arrays, metadata = _load_eflic_indices(ctx.require_input("indices").path, ctx.require_input("indices").metadata)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
        timing_records = []
        _report_setup_progress(ctx, "Creating EF-LIC ONNX Runtime decoder sessions")
        setup_start = time.perf_counter()
        sessions = _eflic_onnx_sessions(bundle, ort, providers, "synthesis_path")
        runtime_evidence = _eflic_onnx_bundle_evidence(bundle, ort, sessions, providers[0], "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode_indices(
            ctx,
            arrays,
            metadata,
            bundle,
            timing_records,
            lambda package, runtime_inputs: sessions[_eflic_shape_key(package["input_shape"])].run(None, runtime_inputs)[0],
            "eflic_onnx_decode_indices",
            "onnxruntime",
            runtime_evidence,
        )


class EfLicOpenVinoDecodeIndicesOperation(Operation):
    id = "model.eflic_openvino_decode_indices"
    name = "EF-LIC OpenVINO decoder from VQ/RVQ indices"
    input_kinds = {"indices": ["semantic.indices.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = {
        **EFLIC_CODING_INFO,
        "implementation": "OpenVINO EF-LIC decompress wrapper",
        "language": "OpenVINO",
    }
    params_schema = object_schema(
        {
            "device": {"type": "string", "default": "CPU"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability("openvino", ["openvino"], "EF-LIC OpenVINO")
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        ov = _require_openvino()
        arrays, metadata = _load_eflic_indices(ctx.require_input("indices").path, ctx.require_input("indices").metadata)
        bundle = _load_eflic_onnx_bundle(ctx.require_input("model"))
        device = _openvino_device(str(ctx.params.get("device", "CPU")))
        timing_records = []
        _report_setup_progress(ctx, "Compiling EF-LIC OpenVINO decoder models")
        setup_start = time.perf_counter()
        compiled = _eflic_openvino_sessions(bundle, ov, device, "synthesis_path")
        runtime_evidence = _eflic_openvino_bundle_evidence(bundle, ov, compiled, device, "synthesis_path")
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        return _run_eflic_runtime_decode_indices(
            ctx,
            arrays,
            metadata,
            bundle,
            timing_records,
            lambda package, runtime_inputs: _openvino_run_named_outputs(compiled[_eflic_shape_key(package["input_shape"])], runtime_inputs)[0],
            "eflic_openvino_decode_indices",
            "openvino",
            runtime_evidence,
        )


class _EfLicRunner:
    def __init__(self, params: JsonDict) -> None:
        self.params = dict(params)
        self.repo_path = str(Path(str(params.get("repo_path") or DEFAULT_REPO_PATH)).expanduser().resolve())
        self.checkpoint = str(Path(str(params.get("checkpoint") or DEFAULT_CHECKPOINT)).expanduser().resolve())
        self.model_source_sha256 = ""
        self.checkpoint_sha256 = ""
        self.device = str(params.get("device") or "cpu")
        self.force_ind = max(0, min(4, int(params.get("force_ind", 2))))
        self.pad_to_multiple = max(1, int(params.get("pad_to_multiple", 64)))
        self._model = None

    def model(self):
        if self._model is not None:
            return self._model
        torch = _require_torch()
        checkpoint_path = Path(self.checkpoint)
        model_path = Path(self.repo_path) / "EF_LIC.py"
        _validate_file(checkpoint_path, "EF-LIC checkpoint")
        _validate_file(model_path, "EF-LIC EF_LIC.py")
        self.model_source_sha256 = file_sha256(model_path)
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        _require_expected_sha256(
            model_path,
            self.params.get("expected_model_sha256"),
            label="EF-LIC model source",
        )
        _require_expected_sha256(
            checkpoint_path,
            self.params.get("expected_checkpoint_sha256"),
            label="EF-LIC checkpoint",
        )
        module = _load_eflic_module(Path(self.repo_path))
        network = module.model().to(self.device).eval()
        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        network.load_state_dict(state, strict=True)
        network.prepare_inference_(force_ind=self.force_ind)
        self._model = network
        return network

    def image_to_tensor(self, torch, image: np.ndarray):
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = tensor * 2.0 - 1.0
        height, width = int(tensor.shape[2]), int(tensor.shape[3])
        new_h = (height + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple
        new_w = (width + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple
        pad_h = new_h - height
        pad_w = new_w - width
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor, tuple(int(item) for item in tensor.shape)


class _EfLicEncodeWrapper:
    def __new__(cls, network, force_ind: int):
        torch = _require_torch()

        class Wrapper(torch.nn.Module):
            def __init__(self, model, selected_force_ind: int) -> None:
                super().__init__()
                self.model = model
                self.force_ind = int(selected_force_ind)

            def forward(self, images):
                payload = self.model.compress(images, force_ind=self.force_ind)
                return (
                    payload["z_inds"].to(dtype=torch.int64),
                    payload["y_inds"][0].to(dtype=torch.int64),
                    payload["y_inds"][1].to(dtype=torch.int64),
                    payload["y_inds"][2].to(dtype=torch.int64),
                    payload["y_inds"][3].to(dtype=torch.int64),
                )

        return Wrapper(network, force_ind)


class _EfLicDecodeWrapper:
    def __new__(cls, network, force_ind: int):
        torch = _require_torch()

        class Wrapper(torch.nn.Module):
            def __init__(self, model, selected_force_ind: int) -> None:
                super().__init__()
                self.model = model
                self.force_ind = int(selected_force_ind)

            def forward(self, z_inds, y0_inds, y1_inds, y2_inds, y3_inds):
                return self.model.decompress(
                    {
                        "z_inds": z_inds,
                        "y_inds": [y0_inds, y1_inds, y2_inds, y3_inds],
                        "force_ind": self.force_ind,
                    },
                    force_ind=self.force_ind,
                )

        return Wrapper(network, force_ind)


def _run_eflic_runtime_encode(
    ctx: OperationContext,
    images: np.ndarray,
    input_metadata: JsonDict,
    bundle: JsonDict,
    timing_records: list,
    run_model,
    source: str,
    runner_name: str,
    runtime_evidence: JsonDict,
) -> OperationResult:
    backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
    entries = []
    bit_chunks = []
    bit_offset = 0
    total = int(images.shape[0])
    _report_example_progress(ctx, 0, total, "encoding")
    for index in range(total):
        _report_example_active_progress(ctx, index, total, "encoding")
        image = _image_for_index(images, input_metadata, index)
        image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
        total_start = time.perf_counter()
        frame, padded_shape = timed_call(
            timing_records,
            index,
            "encoder.preprocess",
            lambda image=image: _image_to_nchw_numpy(image, int(bundle.get("pad_to_multiple") or 64)),
            image_shape=image_shape,
        )
        package = _eflic_package_for_input_shape(bundle, padded_shape)
        encoded = timed_call(
            timing_records,
            index,
            "encoder.model",
            lambda package=package, frame=frame: run_model(package, frame),
            image_shape=image_shape,
        )
        raw_bits, meta = timed_call(
            timing_records,
            index,
            "encoder.symbol_encode",
            lambda encoded=encoded, bundle=bundle: _pack_inds_arrays(bundle.get("n_e") or (), encoded[0], encoded[1:], backend),
            image_shape=image_shape,
        )
        entries.append(
            {
                "bit_offset": int(bit_offset),
                "bit_count": int(raw_bits.size),
                "index_meta": meta,
                "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                "padded_shape": list(padded_shape),
                "model_input_shape": list(package["input_shape"]),
                "force_ind": int(bundle.get("force_ind", 2)),
            }
        )
        bit_chunks.append(raw_bits)
        bit_offset += int(raw_bits.size)
        append_measurement(
            timing_records,
            index,
            "encoder.total",
            time.perf_counter() - total_start,
            image_shape=image_shape,
        )
        _report_example_progress(ctx, index + 1, total, "encoding")

    bits = np.concatenate(bit_chunks).astype(np.uint8, copy=False) if bit_chunks else np.zeros((0,), dtype=np.uint8)
    model_artifact_metrics = _bundle_model_artifact_metrics(bundle)
    metadata = dict(input_metadata)
    metadata.update(
        {
            "codec": "eflic",
            "model": "EF-LIC",
            "source": source,
            "runtime": runner_name,
            "force_ind": int(bundle.get("force_ind", 2)),
            "checkpoint": bundle.get("checkpoint", ""),
            "repo_path": bundle.get("repo_path", ""),
            **model_artifact_metrics,
            "bit_count": int(bits.size),
            "byte_count": int((bits.size + 7) // 8),
            "bit_role": "payload",
            "payload_bit_count": int(bits.size),
            "payload_format": "eflic.fixed_length_indices",
            "source_bit_storage": "unpacked_uint8",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "entropy_coded": False,
            "fixed_length_coded": True,
            "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "indices_to_bits"),
            "entries": entries,
            "original_shape": list(images.shape),
            "runtime_evidence": runtime_evidence,
            **_coding_metadata(),
        }
    )
    path = ctx.output_path("bits", ".npz")
    np.savez_compressed(path, bits=bits, metadata_json=json.dumps(metadata))
    return OperationResult(
        outputs={"bits": artifact("channel.payload_bits.numpy", path, metadata)},
        metrics={
            "codec.bit_count": int(bits.size),
            "codec.bytes": int((bits.size + 7) // 8),
            "channel.payload_bit_count": int(bits.size),
            **model_artifact_metrics,
        },
        metadata={
            "model": "EF-LIC",
            "force_ind": int(bundle.get("force_ind", 2)),
            "runtime": runner_name,
            "runtime_evidence": runtime_evidence,
            **model_artifact_metrics,
            **_coding_metadata(),
            "codec_timing": codec_timing_metadata(
                "encoder",
                timing_records,
                runner=runner_name,
                notes={
                    "encoder.model": "%s EF-LIC neural compress wrapper returning VQ/RVQ index tensors." % runner_name,
                    "encoder.symbol_encode": "NumPy fixed-length packing of EF-LIC VQ/RVQ indices; no entropy coding is used.",
                    **runtime_evidence,
                },
            ),
        },
    )


def _run_eflic_runtime_decode(
    ctx: OperationContext,
    bits: np.ndarray,
    metadata: JsonDict,
    bundle: JsonDict,
    timing_records: list,
    run_model,
    source: str,
    runner_name: str,
    runtime_evidence: JsonDict,
) -> OperationResult:
    backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("data_plane_backend", "auto")))
    entries = _validated_eflic_payload_entries(
        bits,
        metadata,
        "EF-LIC %s payload" % runner_name,
    )
    try:
        decoded = []
        _report_example_progress(ctx, 0, len(entries), "decoding")
        for index, entry in enumerate(entries):
            _report_example_active_progress(ctx, index, len(entries), "decoding")
            total_start = time.perf_counter()
            offset = entry["bit_offset"]
            count = entry["bit_count"]
            chunk = bits[offset : offset + count]
            package = _eflic_package_for_input_shape(bundle, entry.get("model_input_shape") or entry.get("padded_shape"))
            arrays = timed_call(
                timing_records,
                index,
                "decoder.symbol_decode",
                lambda chunk=chunk, entry=entry: _unpack_inds_numpy(chunk, entry["index_meta"], backend),
                image_shape=entry.get("original_shape"),
            )
            output = timed_call(
                timing_records,
                index,
                "decoder.model",
                lambda package=package, arrays=arrays: run_model(
                    package,
                    {name: array for name, array in zip(EFLIC_INDEX_OUTPUT_NAMES, arrays)},
                ),
                image_shape=entry.get("original_shape"),
            )
            decoded.append(_nchw_numpy_to_image(output, entry["original_shape"]))
            append_measurement(
                timing_records,
                index,
                "decoder.total",
                time.perf_counter() - total_start,
                image_shape=entry.get("original_shape"),
            )
            _report_example_progress(ctx, index + 1, len(entries), "decoding")
        images, decoded_shapes = _stack_image_list(decoded)
    except Exception as exc:
        if str(ctx.params.get("on_error", "fail")) != "zeros":
            raise RuntimeError("EF-LIC %s decode failed, likely due to corrupted fixed-length payload bits: %s" % (runner_name, exc)) from exc
        shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
        images = np.zeros(shape, dtype=np.uint8)
        decoded_shapes = [[1, int(shape[1]), int(shape[2]), int(shape[3])]] if len(shape) == 4 else []

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "source": source,
            "runtime": runner_name,
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "original_shapes": decoded_shapes,
            "runtime_evidence": runtime_evidence,
            "data_plane_backend": dataplane.selected_backend({"data_plane_backend": backend}, "bits_to_indices"),
        }
    )
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    model_artifact_metrics = _bundle_model_artifact_metrics(bundle)
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metrics=model_artifact_metrics,
        metadata={
            "shape": list(images.shape),
            "runtime": runner_name,
            "runtime_evidence": runtime_evidence,
            **model_artifact_metrics,
            "codec_timing": codec_timing_metadata(
                "decoder",
                timing_records,
                runner=runner_name,
                notes={
                    "decoder.symbol_decode": "NumPy fixed-length unpacking of EF-LIC VQ/RVQ indices; no entropy decoding is used.",
                    "decoder.model": "%s EF-LIC neural decompress wrapper from VQ/RVQ index tensors." % runner_name,
                    **runtime_evidence,
                },
            ),
        },
    )


def _run_eflic_runtime_encode_indices(
    ctx: OperationContext,
    images: np.ndarray,
    input_metadata: JsonDict,
    bundle: JsonDict,
    timing_records: list,
    run_model,
    source: str,
    runner_name: str,
    runtime_evidence: JsonDict,
) -> OperationResult:
    arrays: JsonDict = {}
    entries = []
    total = int(images.shape[0])
    _report_example_progress(ctx, 0, total, "encoding")
    for index in range(total):
        _report_example_active_progress(ctx, index, total, "encoding")
        image = _image_for_index(images, input_metadata, index)
        image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
        total_start = time.perf_counter()
        frame, padded_shape = timed_call(
            timing_records,
            index,
            "encoder.preprocess",
            lambda image=image: _image_to_nchw_numpy(image, int(bundle.get("pad_to_multiple") or 64)),
            image_shape=image_shape,
        )
        package = _eflic_package_for_input_shape(bundle, padded_shape)
        encoded = timed_call(
            timing_records,
            index,
            "encoder.inference",
            lambda package=package, frame=frame: run_model(package, frame),
            image_shape=image_shape,
        )
        entry, entry_arrays = _eflic_numpy_indices_to_entry(
            encoded,
            _eflic_index_meta_from_arrays(bundle.get("n_e"), encoded),
            index,
            [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            list(padded_shape),
            list(package["input_shape"]),
            int(bundle.get("force_ind", 2)),
            bundle.get("n_e"),
        )
        arrays.update(entry_arrays)
        entries.append(entry)
        append_measurement(
            timing_records,
            index,
            "encoder.total",
            time.perf_counter() - total_start,
            image_shape=image_shape,
        )
        _report_example_progress(ctx, index + 1, total, "encoding")

    model_artifact_metrics = _bundle_model_artifact_metrics(bundle)
    metadata = dict(input_metadata)
    metadata.update(
        {
            "codec": "eflic",
            "model": "EF-LIC",
            "source": source,
            "runtime": runner_name,
            "semantic_form": "indices",
            "representation": "eflic.vq_rvq_indices",
            "force_ind": int(bundle.get("force_ind", 2)),
            "checkpoint": bundle.get("checkpoint", ""),
            "repo_path": bundle.get("repo_path", ""),
            **model_artifact_metrics,
            "entries": entries,
            "original_shape": list(images.shape),
            "runtime_evidence": runtime_evidence,
            "dtype": "int64",
            "entropy_coded": False,
            "fixed_length_coded": False,
            **_coding_metadata(),
        }
    )
    path = ctx.output_path("indices", ".npz")
    np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)
    return OperationResult(
        outputs={"indices": artifact("semantic.indices.numpy", path, metadata)},
        metrics=model_artifact_metrics,
        metadata={
            "model": "EF-LIC",
            "force_ind": int(bundle.get("force_ind", 2)),
            "runtime": runner_name,
            "runtime_evidence": runtime_evidence,
            **model_artifact_metrics,
            **_coding_metadata(),
            "codec_timing": codec_timing_metadata(
                "encoder",
                timing_records,
                runner=runner_name,
                notes={
                    "encoder.inference": "%s EF-LIC neural compress wrapper returning VQ/RVQ index tensors." % runner_name,
                    **runtime_evidence,
                },
            ),
        },
    )


def _run_eflic_runtime_decode_indices(
    ctx: OperationContext,
    arrays: JsonDict,
    metadata: JsonDict,
    bundle: JsonDict,
    timing_records: list,
    run_model,
    source: str,
    runner_name: str,
    runtime_evidence: JsonDict,
) -> OperationResult:
    try:
        entries = metadata.get("entries") or []
        if not entries:
            raise RuntimeError("EF-LIC indices metadata does not contain entries")
        decoded = []
        _report_example_progress(ctx, 0, len(entries), "decoding")
        for index, entry in enumerate(entries):
            _report_example_active_progress(ctx, index, len(entries), "decoding")
            total_start = time.perf_counter()
            package = _eflic_package_for_input_shape(bundle, entry.get("model_input_shape") or entry.get("padded_shape"))
            output = timed_call(
                timing_records,
                index,
                "decoder.inference",
                lambda package=package, entry=entry: run_model(
                    package,
                    _eflic_entry_arrays_to_runtime_inputs(arrays, entry),
                ),
                image_shape=entry.get("original_shape"),
            )
            decoded.append(_nchw_numpy_to_image(output, entry["original_shape"]))
            append_measurement(
                timing_records,
                index,
                "decoder.total",
                time.perf_counter() - total_start,
                image_shape=entry.get("original_shape"),
            )
            _report_example_progress(ctx, index + 1, len(entries), "decoding")
        images, decoded_shapes = _stack_image_list(decoded)
    except Exception as exc:
        if str(ctx.params.get("on_error", "fail")) != "zeros":
            raise RuntimeError("EF-LIC %s index decode failed: %s" % (runner_name, exc)) from exc
        shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
        images = np.zeros(shape, dtype=np.uint8)
        decoded_shapes = [[1, int(shape[1]), int(shape[2]), int(shape[3])]] if len(shape) == 4 else []

    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "source": source,
            "runtime": runner_name,
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "original_shapes": decoded_shapes,
            "runtime_evidence": runtime_evidence,
        }
    )
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    model_artifact_metrics = _bundle_model_artifact_metrics(bundle)
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metrics=model_artifact_metrics,
        metadata={
            "shape": list(images.shape),
            "runtime": runner_name,
            "runtime_evidence": runtime_evidence,
            **model_artifact_metrics,
            "codec_timing": codec_timing_metadata(
                "decoder",
                timing_records,
                runner=runner_name,
                notes={
                    "decoder.inference": "%s EF-LIC neural decompress wrapper from VQ/RVQ index tensors." % runner_name,
                    **runtime_evidence,
                },
            ),
        },
    )


def _eflic_export_sample_arrays(images: np.ndarray, metadata: JsonDict, multiple: int) -> JsonDict:
    samples: JsonDict = {}
    for index in range(int(images.shape[0])):
        image = _image_for_index(images, metadata, index)
        array, _shape = _image_to_nchw_numpy(image, multiple)
        key = _eflic_shape_key(array.shape)
        if key not in samples:
            samples[key] = array
    return samples


def _image_to_nchw_numpy(image: np.ndarray, multiple: int) -> Tuple[np.ndarray, list]:
    tensor = image.astype(np.float32, copy=False) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    tensor = tensor * 2.0 - 1.0
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    multiple = max(1, int(multiple))
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    pad_h = new_h - height
    pad_w = new_w - width
    if pad_h or pad_w:
        tensor = np.pad(tensor, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False)), [int(item) for item in tensor.shape]


def _nchw_numpy_to_image(tensor: np.ndarray, original_shape) -> np.ndarray:
    _count, height, width, _channels = [int(item) for item in original_shape]
    array = np.asarray(tensor, dtype=np.float32)
    cropped = np.clip(array[:, :, :height, :width], -1.0, 1.0)
    image = ((np.transpose(cropped[0], (1, 2, 0)) + 1.0) * 0.5 * 255.0).round()
    return np.clip(image, 0, 255).astype(np.uint8)


def _pack_inds_arrays(n_e_values, z_inds, y_inds, backend: str = "auto") -> Tuple[np.ndarray, list]:
    n_e = tuple(int(item) for item in (n_e_values or (1024, 512, 256, 128, 1024)))
    bits_per_index = [(size - 1).bit_length() for size in n_e]
    y_arrays = list(y_inds)
    if len(y_arrays) != 4:
        raise RuntimeError("EF-LIC runtime encoder expected 4 y index tensors, got %d" % len(y_arrays))
    items = [(z_inds, bits_per_index[-1])] + [(y_arrays[index], bits_per_index[index]) for index in range(4)]
    meta = []
    chunks = []
    for array, width in items:
        np_array = np.asarray(array, dtype=np.int64)
        shape = [int(item) for item in np_array.shape]
        meta.append({"shape": shape, "bits_per_index": int(width)})
        chunks.append(_array_to_bits(np_array, width, backend))
    return np.concatenate(chunks).astype(np.uint8, copy=False), meta


def _eflic_index_meta_from_arrays(n_e_values, arrays) -> list:
    n_e = tuple(int(item) for item in (n_e_values or (1024, 512, 256, 128, 1024)))
    bits_per_index = [(size - 1).bit_length() for size in n_e]
    ordered = list(arrays)
    if len(ordered) != 5:
        raise RuntimeError("EF-LIC expected 5 index tensors, got %d" % len(ordered))
    widths = [bits_per_index[-1]] + [bits_per_index[index] for index in range(4)]
    return [
        {"shape": [int(item) for item in np.asarray(array).shape], "bits_per_index": int(width)}
        for array, width in zip(ordered, widths)
    ]


def _eflic_entry_array_key(example_index: int, name: str) -> str:
    return "entry_%d_%s" % (int(example_index), name)


def _eflic_torch_indices_to_entry(
    network,
    inds,
    example_index: int,
    original_shape,
    padded_shape,
    model_input_shape,
    force_ind: int,
) -> Tuple[JsonDict, JsonDict]:
    z_array = np.ascontiguousarray(_torch_to_numpy(inds["z_inds"]).astype(np.int64, copy=False))
    y_arrays = [
        np.ascontiguousarray(_torch_to_numpy(item).astype(np.int64, copy=False))
        for item in list(inds["y_inds"])
    ]
    return _eflic_numpy_indices_to_entry(
        [z_array] + y_arrays,
        _eflic_index_meta_from_arrays(getattr(network, "n_e", None), [z_array] + y_arrays),
        example_index,
        original_shape,
        padded_shape,
        model_input_shape,
        force_ind,
        getattr(network, "n_e", None),
    )


def _eflic_numpy_indices_to_entry(
    arrays,
    index_meta,
    example_index: int,
    original_shape,
    padded_shape,
    model_input_shape,
    force_ind: int,
    n_e_values=None,
) -> Tuple[JsonDict, JsonDict]:
    ordered = [np.ascontiguousarray(np.asarray(array).astype(np.int64, copy=False)) for array in list(arrays)]
    if len(ordered) != 5:
        raise RuntimeError("EF-LIC expected 5 index tensors, got %d" % len(ordered))
    names = list(EFLIC_INDEX_OUTPUT_NAMES)
    entry_arrays: JsonDict = {}
    key_map: JsonDict = {}
    for name, array in zip(names, ordered):
        key = _eflic_entry_array_key(example_index, name)
        key_map[name] = key
        entry_arrays[key] = array
    meta = list(index_meta or _eflic_index_meta_from_arrays(n_e_values, ordered))
    entry = {
        "index_arrays": key_map,
        "index_meta": meta,
        "original_shape": [int(item) for item in list(original_shape or [])],
        "padded_shape": [int(item) for item in list(padded_shape or [])],
        "model_input_shape": [int(item) for item in list(model_input_shape or padded_shape or [])],
        "force_ind": int(force_ind),
    }
    if n_e_values is not None:
        entry["n_e"] = [int(item) for item in list(n_e_values)]
    return entry, entry_arrays


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


def _load_eflic_indices(path, fallback_metadata: JsonDict) -> Tuple[JsonDict, JsonDict]:
    arrays: JsonDict = {}
    metadata = dict(fallback_metadata)
    with np.load(str(path), allow_pickle=False) as payload:
        if "metadata_json" in payload:
            metadata.update(
                _decode_artifact_metadata(
                    payload["metadata_json"], "EF-LIC indices artifact"
                )
            )
        for key in payload.files:
            if key == "metadata_json":
                continue
            arrays[key] = np.asarray(payload[key]).astype(np.int64, copy=False)
    if not arrays:
        raise RuntimeError("EF-LIC indices artifact contains no index arrays")
    return arrays, metadata


def _eflic_ordered_arrays_for_entry(arrays: JsonDict, entry: JsonDict) -> list:
    key_map = entry.get("index_arrays") or {}
    ordered = []
    for name in EFLIC_INDEX_OUTPUT_NAMES:
        key = str(key_map.get(name) or "")
        if not key or key not in arrays:
            raise RuntimeError("EF-LIC indices entry is missing array %s" % name)
        ordered.append(np.ascontiguousarray(np.asarray(arrays[key]).astype(np.int64, copy=False)))
    return ordered


def _pack_eflic_entry_arrays(arrays: JsonDict, entry: JsonDict, backend: str = "auto") -> Tuple[np.ndarray, list]:
    ordered = _eflic_ordered_arrays_for_entry(arrays, entry)
    meta = list(entry.get("index_meta") or _eflic_index_meta_from_arrays(entry.get("n_e"), ordered))
    chunks = []
    for array, item in zip(ordered, meta):
        chunks.append(_array_to_bits(array, int(item["bits_per_index"]), backend))
    return np.concatenate(chunks).astype(np.uint8, copy=False), meta


def _eflic_entry_arrays_to_torch(arrays: JsonDict, entry: JsonDict, device: str, torch):
    ordered = _eflic_ordered_arrays_for_entry(arrays, entry)
    tensors = [torch.from_numpy(array.astype(np.int64, copy=False)).to(device=device) for array in ordered]
    return {"z_inds": tensors[0], "y_inds": tensors[1:]}


def _eflic_entry_arrays_to_runtime_inputs(arrays: JsonDict, entry: JsonDict) -> JsonDict:
    ordered = _eflic_ordered_arrays_for_entry(arrays, entry)
    return {name: array for name, array in zip(EFLIC_INDEX_OUTPUT_NAMES, ordered)}


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError("%s must be an integer" % label)
    normalized = int(value)
    if normalized < 0:
        raise RuntimeError("%s must be non-negative" % label)
    return normalized


def _required_eflic_index_bits(meta: Any, label: str) -> int:
    if not isinstance(meta, list) or len(meta) != len(EFLIC_INDEX_OUTPUT_NAMES):
        raise RuntimeError(
            "%s must contain exactly %d index tensor descriptors"
            % (label, len(EFLIC_INDEX_OUTPUT_NAMES))
        )
    required = 0
    for index, raw_item in enumerate(meta):
        if not isinstance(raw_item, Mapping):
            raise RuntimeError("%s item %d must be an object" % (label, index))
        raw_shape = raw_item.get("shape")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise RuntimeError(
                "%s item %d shape must be a non-empty integer array"
                % (label, index)
            )
        element_count = 1
        for axis, raw_dimension in enumerate(raw_shape):
            dimension = _strict_nonnegative_int(
                raw_dimension,
                "%s item %d shape[%d]" % (label, index, axis),
            )
            if dimension == 0:
                raise RuntimeError(
                    "%s item %d shape dimensions must be positive"
                    % (label, index)
                )
            element_count *= dimension
        width = _strict_nonnegative_int(
            raw_item.get("bits_per_index"),
            "%s item %d bits_per_index" % (label, index),
        )
        if width == 0:
            raise RuntimeError(
                "%s item %d bits_per_index must be positive" % (label, index)
            )
        required += element_count * width
    return required


def _validated_eflic_payload_entries(
    bits: np.ndarray,
    metadata: Mapping[str, Any],
    label: str,
) -> list[JsonDict]:
    raw_entries = metadata.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RuntimeError("%s metadata must contain a non-empty entries array" % label)
    total_bits = int(bits.size)
    cursor = 0
    entries: list[JsonDict] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError(
                "%s entries item %d must be an object" % (label, index)
            )
        offset = _strict_nonnegative_int(
            raw_entry.get("bit_offset"),
            "%s entries item %d bit_offset" % (label, index),
        )
        count = _strict_nonnegative_int(
            raw_entry.get("bit_count"),
            "%s entries item %d bit_count" % (label, index),
        )
        if count == 0:
            raise RuntimeError(
                "%s entries item %d bit_count must be positive" % (label, index)
            )
        if offset != cursor:
            relationship = "overlaps a prior entry" if offset < cursor else "leaves a gap"
            raise RuntimeError(
                "%s entries item %d starts at bit %d, expected %d; it %s"
                % (label, index, offset, cursor, relationship)
            )
        end = offset + count
        if end > total_bits:
            raise RuntimeError(
                "%s entries item %d range [%d, %d) exceeds the %d-bit payload"
                % (label, index, offset, end, total_bits)
            )
        required = _required_eflic_index_bits(
            raw_entry.get("index_meta"),
            "%s entries item %d index_meta" % (label, index),
        )
        if count != required:
            raise RuntimeError(
                "%s entries item %d declares %d bits but index_meta requires %d"
                % (label, index, count, required)
            )
        entry = dict(raw_entry)
        entry["bit_offset"] = offset
        entry["bit_count"] = count
        entries.append(entry)
        cursor = end
    if cursor != total_bits:
        raise RuntimeError(
            "%s entries cover %d of %d payload bits; trailing bits are not allowed"
            % (label, cursor, total_bits)
        )
    return entries


def _unpack_inds_numpy(bits: np.ndarray, meta: list, backend: str = "auto") -> list:
    required_bits = _required_eflic_index_bits(meta, "EF-LIC index_meta")
    if int(bits.size) != required_bits:
        raise RuntimeError(
            "EF-LIC index payload requires exactly %d bits; got %d"
            % (required_bits, int(bits.size))
        )
    arrays = []
    pos = 0
    for item in meta:
        shape = [int(value) for value in item["shape"]]
        width = int(item["bits_per_index"])
        count = int(np.prod(shape))
        nbits = count * width
        chunk = bits[pos : pos + nbits]
        arrays.append(_bits_to_array(chunk, shape, width, backend))
        pos += nbits
    if len(arrays) != 5:
        raise RuntimeError("EF-LIC payload expected 5 index tensors, got %d" % len(arrays))
    return arrays


def _array_to_bits(array: np.ndarray, width: int, backend: str = "auto") -> np.ndarray:
    values = np.asarray(array).reshape(-1).astype(np.int64, copy=False)
    bits, _selected_backend = dataplane.indices_to_bits(values, int(width), backend)
    return bits.astype(np.uint8, copy=False)


def _bits_to_array(bits: np.ndarray, shape: list, width: int, backend: str = "auto") -> np.ndarray:
    decoded, _invalid_fraction, _selected_backend = dataplane.bits_to_indices(
        bits,
        int(width),
        tuple(int(item) for item in shape),
        1 << int(width),
        "mod",
        backend,
    )
    return decoded.astype(np.int64, copy=False)


def _eflic_shape_key(shape) -> str:
    return "x".join(str(int(item)) for item in list(shape))


def _eflic_shape_packages(bundle: JsonDict) -> JsonDict:
    packages: JsonDict = {}
    raw = bundle.get("shape_packages")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                packages[str(key)] = dict(value)
    if not packages:
        packages[_eflic_shape_key(bundle.get("export_input_shape") or [])] = {
            "input_shape": list(bundle.get("export_input_shape") or []),
            "analysis_path": str(bundle.get("analysis_path") or ""),
            "synthesis_path": str(bundle.get("synthesis_path") or ""),
        }
    return packages


def _eflic_package_for_input_shape(bundle: JsonDict, shape) -> JsonDict:
    key = _eflic_shape_key(shape)
    packages = _eflic_shape_packages(bundle)
    if key in packages:
        return packages[key]
    raise RuntimeError(
        "EF-LIC runtime bundle does not include a package for input shape %s. "
        "Rerun codec_export with the selected image set." % key
    )


def _load_eflic_onnx_bundle(input_artifact) -> JsonDict:
    try:
        payload = decode_strict_yaml_or_json(
            Path(input_artifact.path).read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("bundle manifest root must be a JSON object")
    except Exception as exc:
        raise RuntimeError("Could not read EF-LIC ONNX bundle manifest at %s: %s" % (input_artifact.path, exc)) from exc
    payload = dict(payload)
    if payload.get("codec") != "eflic":
        raise RuntimeError("Expected an EF-LIC ONNX bundle, got codec=%s" % payload.get("codec"))
    packages = _eflic_shape_packages(payload)
    for key, package in packages.items():
        for field in ("analysis_path", "synthesis_path"):
            path = str(package.get(field) or "")
            if not path or not Path(path).is_file():
                raise RuntimeError("EF-LIC ONNX bundle package %s points to missing %s: %s" % (key, field, path))
    return payload


def _load_eflic_aoti_bundle(input_artifact) -> JsonDict:
    try:
        payload = decode_strict_yaml_or_json(
            Path(input_artifact.path).read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(payload, Mapping):
            raise ValueError("bundle manifest root must be a JSON object")
    except Exception as exc:
        raise RuntimeError("Could not read EF-LIC AOT Inductor bundle manifest at %s: %s" % (input_artifact.path, exc)) from exc
    payload = dict(payload)
    if payload.get("codec") != "eflic":
        raise RuntimeError("Expected an EF-LIC AOT Inductor bundle, got codec=%s" % payload.get("codec"))
    if payload.get("runtime") != "aot_inductor":
        raise RuntimeError("Expected EF-LIC runtime=aot_inductor, got %s" % payload.get("runtime"))
    packages = _eflic_shape_packages(payload)
    for key, package in packages.items():
        for field in ("analysis_path", "synthesis_path"):
            path = str(package.get(field) or "")
            if not path or not Path(path).is_file():
                raise RuntimeError("EF-LIC AOT Inductor bundle package %s points to missing %s: %s" % (key, field, path))
    return payload


def _eflic_onnx_sessions(bundle: JsonDict, ort, providers: list, path_field: str) -> Dict[str, object]:
    sessions = {}
    for key, package in _eflic_shape_packages(bundle).items():
        session = ort.InferenceSession(str(package[path_field]), providers=providers)
        _assert_onnx_session_provider(session, providers, "%s %s" % (path_field, key))
        sessions[key] = session
    return sessions


def _eflic_openvino_sessions(bundle: JsonDict, ov, device: str, path_field: str) -> Dict[str, object]:
    return {
        key: _openvino_compile_model(ov, package[path_field], device)
        for key, package in _eflic_shape_packages(bundle).items()
    }


def _eflic_aoti_sessions(bundle: JsonDict, torch, device: str, path_field: str) -> Dict[str, object]:
    return {
        key: _EfLicAotInductorSession(torch, package[path_field], device)
        for key, package in _eflic_shape_packages(bundle).items()
    }


def _eflic_onnx_bundle_evidence(bundle: JsonDict, ort, sessions: Dict[str, object], provider: str, path_field: str) -> JsonDict:
    return {
        "runtime": "onnxruntime",
        "onnxruntime_version": str(getattr(ort, "__version__", "")),
        **onnxruntime_native_evidence(ort),
        "requested_provider": provider,
        "active_providers": {key: list(session.get_providers()) for key, session in sessions.items()},
        "shape_count": len(sessions),
        "model_paths": {key: str(_eflic_shape_packages(bundle)[key].get(path_field)) for key in sessions},
        "model_sha256": {
            key: file_sha256(Path(str(_eflic_shape_packages(bundle)[key].get(path_field))))
            for key in sessions
            if Path(str(_eflic_shape_packages(bundle)[key].get(path_field))).is_file()
        },
        "session_inputs": {
            key: [_onnx_value_info(item) for item in session.get_inputs()]
            for key, session in sessions.items()
        },
        "session_outputs": {
            key: [_onnx_value_info(item) for item in session.get_outputs()]
            for key, session in sessions.items()
        },
    }


def _eflic_aoti_bundle_evidence(bundle: JsonDict, torch, sessions: Dict[str, object], device: str, path_field: str) -> JsonDict:
    packages = _eflic_shape_packages(bundle)
    return {
        "runtime": "aot_inductor",
        "torch_version": str(getattr(torch, "__version__", "")),
        "device": str(device or "cpu"),
        "shape_count": len(sessions),
        "artifact": "torch._inductor AOTI .pt2 package",
        "model_paths": {key: str(packages[key].get(path_field)) for key in sessions},
        "model_sha256": {
            key: file_sha256(Path(str(packages[key].get(path_field))))
            for key in sessions
            if Path(str(packages[key].get(path_field))).is_file()
        },
        "session_inputs": {
            key: list(packages[key].get("input_shape") or [])
            if path_field == "analysis_path"
            else dict(packages[key].get("index_shapes") or {})
            for key in sessions
        },
        "session_outputs": {
            key: dict(packages[key].get("index_shapes") or {})
            if path_field == "analysis_path"
            else list(packages[key].get("input_shape") or [])
            for key in sessions
        },
    }


def _eflic_openvino_bundle_evidence(bundle: JsonDict, ov, sessions: Dict[str, object], device: str, path_field: str) -> JsonDict:
    return {
        "runtime": "openvino",
        "openvino_version": str(getattr(ov, "__version__", "")),
        "requested_device": _openvino_device(device),
        "shape_count": len(sessions),
        "model_paths": {key: str(_eflic_shape_packages(bundle)[key].get(path_field)) for key in sessions},
        "model_sha256": {
            key: file_sha256(Path(str(_eflic_shape_packages(bundle)[key].get(path_field))))
            for key in sessions
            if Path(str(_eflic_shape_packages(bundle)[key].get(path_field))).is_file()
        },
        "session_inputs": {
            key: [_openvino_value_info(item) for item in list(session.inputs)]
            for key, session in sessions.items()
        },
        "session_outputs": {
            key: [_openvino_value_info(item) for item in list(session.outputs)]
            for key, session in sessions.items()
        },
    }


def _bundle_model_artifact_metrics(bundle: JsonDict) -> JsonDict:
    encoder = int(bundle.get("encoder_model_bytes") or bundle.get("model_artifact.encoder_bytes") or 0)
    decoder = int(bundle.get("decoder_model_bytes") or bundle.get("model_artifact.decoder_bytes") or 0)
    total = int(bundle.get("model_artifact_bytes") or bundle.get("model_artifact.total_bytes") or encoder + decoder)
    return {
        "model_artifact.encoder_bytes": encoder,
        "model_artifact.decoder_bytes": decoder,
        "model_artifact.total_bytes": total,
    }


def _torch_to_numpy(torch_tensor) -> np.ndarray:
    return torch_tensor.detach().cpu().numpy()


def _assert_close(label: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise RuntimeError("%s shape mismatch: got %s, expected %s" % (label, actual_array.shape, expected_array.shape))
    if not np.allclose(actual_array, expected_array, atol=atol, rtol=rtol):
        diff = np.abs(actual_array - expected_array)
        raise RuntimeError("%s mismatch: max_abs=%g mean_abs=%g" % (label, float(diff.max()), float(diff.mean())))


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


def _require_onnx_stack(require_onnx: bool = True):
    onnx = None
    if require_onnx:
        try:
            import onnx as onnx_module
        except ImportError as exc:
            raise RuntimeError(
                'Install with `python -m pip install "noema-lab[onnx]"` in an '
                "installed environment, or `uv sync --extra onnx` in a source "
                "checkout, to export EF-LIC to ONNX"
            ) from exc
        onnx = onnx_module
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[onnx]"` in an '
            "installed environment, or `uv sync --extra onnx` in a source "
            "checkout, to run EF-LIC ONNX Runtime recipes"
        ) from exc
    return onnx, ort


def _require_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[openvino]"` in an '
            "installed environment, or `uv sync --extra openvino` in a source "
            "checkout, to run EF-LIC OpenVINO recipes"
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


def _assert_onnx_session_provider(session, expected_providers: list, role: str) -> None:
    active = list(session.get_providers())
    if active != list(expected_providers):
        raise RuntimeError(
            "ONNX Runtime %s session provider mismatch. Requested %s but session uses %s. "
            "Refusing to run with an implicit provider fallback."
            % (role, expected_providers, active)
        )


def _openvino_device(requested: str) -> str:
    text = str(requested or "CPU").strip() or "CPU"
    return text.upper()


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


def _openvino_run_named_outputs(compiled_model, arrays_by_name: JsonDict) -> list:
    feed = {}
    for port in list(compiled_model.inputs):
        name = str(getattr(port, "any_name", ""))
        if name not in arrays_by_name:
            raise RuntimeError("OpenVINO model input %s was not provided" % name)
        feed[port] = np.ascontiguousarray(np.asarray(arrays_by_name[name]))
    result = compiled_model(feed)
    return [np.asarray(result[port]) for port in list(compiled_model.outputs)]


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


class _EfLicAotInductorSession:
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

    def run(self, arrays) -> list:
        if isinstance(arrays, dict):
            ordered = [arrays[name] for name in EFLIC_INDEX_OUTPUT_NAMES]
        elif isinstance(arrays, (list, tuple)):
            ordered = list(arrays)
        else:
            ordered = [arrays]
        tensors = [self._array_to_tensor(array) for array in ordered]
        with self.torch.no_grad():
            outputs = self.module(*tensors)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return [_torch_to_numpy(output) for output in outputs]

    def _array_to_tensor(self, array):
        np_array = np.asarray(array)
        if np.issubdtype(np_array.dtype, np.integer):
            np_array = np.ascontiguousarray(np_array.astype(np.int64, copy=False))
        else:
            np_array = np.ascontiguousarray(np_array.astype(np.float32, copy=False))
        return self.torch.from_numpy(np_array).to(self.device).contiguous()


def _save_aoti_package(torch, module, example_args, path, label: str) -> str:
    _ensure_aoti_available(torch)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        exported = torch.export.export(
            module.eval(),
            tuple(example_args),
            strict=False,
        )
        torch._inductor.aoti_compile_and_package(exported, package_path=str(target))
    except Exception as exc:
        raise RuntimeError("could not compile %s module with AOT Inductor: %s" % (label, exc)) from exc
    return str(target)


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


def _prepare_eflic_assets(params: JsonDict) -> None:
    repo_path = Path(str(params.get("repo_path") or DEFAULT_REPO_PATH)).expanduser()
    model_path = repo_path / "EF_LIC.py"
    if model_path.exists():
        _validate_file(model_path, "EF-LIC EF_LIC.py")
        _require_expected_sha256(
            model_path,
            params.get("expected_model_sha256"),
            label="EF-LIC model source",
        )
        return
    if not bool(params.get("auto_setup", False)):
        raise RuntimeError(
            "EF-LIC repo_path is missing EF_LIC.py: %s. Automatic executable-code "
            "downloads are disabled; install a reviewed local copy or explicitly enable "
            "auto_setup with expected_model_sha256." % model_path
        )
    expected = _normalized_expected_sha256(
        params.get("expected_model_sha256"),
        label="EF-LIC model source",
        required=True,
    )
    repo_path.mkdir(parents=True, exist_ok=True)
    url = str(params.get("model_url") or DEFAULT_MODEL_URL)
    download_verified_https(
        url,
        model_path,
        expected_sha256=expected,
        max_bytes=MAX_MODEL_SOURCE_DOWNLOAD_BYTES,
        timeout_s=120,
        opener=urlopen,
    )
    _validate_file(model_path, "Downloaded EF-LIC EF_LIC.py")


def _normalized_expected_sha256(value: Any, *, label: str, required: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if not digest:
        if required:
            raise RuntimeError("%s requires an expected SHA-256 digest" % label)
        return ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("%s expected SHA-256 must be 64 hexadecimal characters" % label)
    return digest


def _require_expected_sha256(path: Path, expected: Any, *, label: str) -> str:
    digest = _normalized_expected_sha256(expected, label=label)
    actual = file_sha256(path)
    if digest and actual != digest:
        raise RuntimeError(
            "%s SHA-256 mismatch: expected %s, got %s" % (label, digest, actual)
        )
    return actual


def _load_eflic_module(repo_path: Path):
    model_path = repo_path / "EF_LIC.py"
    _validate_file(model_path, "EF-LIC EF_LIC.py")
    module_name = "_noema_eflic_%s" % abs(hash(str(model_path.resolve())))
    with _temporary_sys_path(str(repo_path.resolve())):
        spec = importlib.util.spec_from_file_location(module_name, str(model_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import EF-LIC model file at %s" % model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if not hasattr(module, "model"):
        raise RuntimeError("EF-LIC model file at %s does not define model()" % model_path)
    return module


@contextmanager
def _temporary_sys_path(path: str):
    old_path = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old_path


def _validate_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise RuntimeError("%s does not exist: %s" % (label, path))
    try:
        head = path.read_bytes()[:512].lstrip().lower()
    except OSError as exc:
        raise RuntimeError("Could not read %s at %s: %s" % (label, path, exc)) from exc
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:128]:
        raise RuntimeError("%s at %s is an HTML page, not the expected file" % (label, path))


def _file_size_or_zero(path: str) -> int:
    try:
        return int(Path(path).expanduser().resolve().stat().st_size)
    except OSError:
        return 0


def _pack_inds(network, inds, backend: str = "auto") -> Tuple[np.ndarray, list]:
    n_e = tuple(int(item) for item in network.n_e)
    bits_per_index = [(size - 1).bit_length() for size in n_e]
    items = [(inds["z_inds"], bits_per_index[-1])] + [
        (inds["y_inds"][index], bits_per_index[index]) for index in range(4)
    ]
    meta = []
    chunks = []
    for tensor, width in items:
        shape = [int(item) for item in tensor.shape]
        meta.append({"shape": shape, "bits_per_index": int(width)})
        chunks.append(_tensor_to_bits(tensor, width, backend))
    return np.concatenate(chunks).astype(np.uint8, copy=False), meta


def _unpack_inds(bits: np.ndarray, meta: list, device: str, torch, backend: str = "auto"):
    required_bits = _required_eflic_index_bits(meta, "EF-LIC index_meta")
    if int(bits.size) != required_bits:
        raise RuntimeError(
            "EF-LIC index payload requires exactly %d bits; got %d"
            % (required_bits, int(bits.size))
        )
    tensors = []
    pos = 0
    for item in meta:
        shape = [int(value) for value in item["shape"]]
        width = int(item["bits_per_index"])
        count = int(np.prod(shape))
        nbits = count * width
        chunk = bits[pos : pos + nbits]
        tensors.append(_bits_to_tensor(chunk, shape, width, device, torch, backend))
        pos += nbits
    if len(tensors) != 5:
        raise RuntimeError("EF-LIC payload expected 5 index tensors, got %d" % len(tensors))
    return {"z_inds": tensors[0], "y_inds": tensors[1:]}


def _tensor_to_bits(tensor, width: int, backend: str = "auto") -> np.ndarray:
    array = tensor.detach().reshape(-1).to("cpu", dtype=_require_torch().long).numpy().astype(np.int64, copy=False)
    bits, _selected_backend = dataplane.indices_to_bits(array, int(width), backend)
    return bits.astype(np.uint8, copy=False)


def _bits_to_tensor(bits: np.ndarray, shape: list, width: int, device: str, torch, backend: str = "auto"):
    decoded = _bits_to_array(bits, shape, width, backend)
    return torch.from_numpy(decoded).to(device=device).view(*shape)


def _eflic_tensor_to_image(tensor, original_shape) -> np.ndarray:
    _count, height, width, _channels = [int(item) for item in original_shape]
    cropped = tensor[:, :, :height, :width].detach().cpu().clamp(-1.0, 1.0)
    image = ((cropped[0].permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0).round()
    return np.clip(image, 0, 255).astype(np.uint8)


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
        label="EF-LIC bitstream",
        backend="python_numpy",
    )
    actual_bit_count = int(canonical.size)
    for field in ("bit_count", "payload_bit_count"):
        if field not in metadata:
            continue
        declared = _strict_nonnegative_int(
            metadata[field],
            "EF-LIC bitstream metadata %s" % field,
        )
        if declared != actual_bit_count:
            raise RuntimeError(
                "EF-LIC bitstream metadata %s declares %d bits but the "
                "artifact stores %d"
                % (field, declared, actual_bit_count)
            )
    if "byte_count" in metadata:
        declared_bytes = _strict_nonnegative_int(
            metadata["byte_count"],
            "EF-LIC bitstream metadata byte_count",
        )
        required_bytes = (actual_bit_count + 7) // 8
        if declared_bytes != required_bytes:
            raise RuntimeError(
                "EF-LIC bitstream metadata byte_count declares %d bytes but "
                "%d bits require %d"
                % (declared_bytes, actual_bit_count, required_bytes)
            )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _eflic_params_from_metadata(params: JsonDict, metadata: JsonDict) -> JsonDict:
    merged = dict(params)
    for key in ("repo_path", "checkpoint", "model_url", "checkpoint_url", "auto_setup", "expected_model_sha256", "expected_checkpoint_sha256", "device", "pad_to_multiple"):
        if key in metadata and (key not in merged or not str(merged.get(key) or "").strip()):
            merged[key] = metadata[key]
    if "force_ind" not in merged or merged.get("force_ind") is None:
        merged["force_ind"] = int(metadata.get("force_ind", 2))
    return merged


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


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[eflic]"` in an installed '
            "environment, or `uv sync --extra eflic` in a source checkout, to use EF-LIC"
        ) from exc
    return torch


def _sync_torch(torch, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _optional_dependency_availability(extra: str, modules: Iterable[str], label: str) -> JsonDict:
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if not missing:
        return {"available": True, "extra": extra, "missing": []}
    return {
        "available": False,
        "extra": extra,
        "missing": missing,
        "reason": (
            'Install with `python -m pip install "noema-lab[%s]"` in an installed '
            "environment, or `uv sync --extra %s` in a source checkout, to use %s adapters"
            % (extra, extra, label)
        ),
    }


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
