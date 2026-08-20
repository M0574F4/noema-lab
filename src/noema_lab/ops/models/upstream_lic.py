from __future__ import annotations

import importlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
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
from noema_lab.ops.models.safe_payload import (
    PAYLOAD_FORMAT as SAFE_PAYLOAD_FORMAT,
    dumps as safe_payload_dumps,
    loads as safe_payload_loads,
)
from noema_lab.ops.models.timing import append_measurement, codec_timing_metadata, timed_call

JsonDict = Dict[str, Any]
DEFAULT_SETUP_TIMEOUT_S = 900
MAX_CHECKPOINT_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


_ASSET_DEFAULTS = {
    "tcm": {
        "repo_path": ".noema/upstreams/LIC_TCM",
        "repo_clone_path": ".noema/upstreams/LIC_TCM",
        "repo_url": "https://github.com/jmliu206/LIC_TCM.git",
        "checkpoint": ".noema/checkpoints/tcm/tcm_N128_lambda0.05_mse.pth",
        "checkpoint_url": "https://drive.google.com/file/d/1TK-CPiD2QwtWJqZoT_OyCtnxdQ7UNP56/view?usp=share_link",
        "checkpoint_preset": "tcm_n128_l0.05_mse",
    },
    "hpcm": {
        "repo_path": ".noema/upstreams/LIC-HPCM",
        "repo_clone_path": ".noema/upstreams/LIC-HPCM",
        "repo_url": "https://github.com/lyq133/LIC-HPCM.git",
        "checkpoint": ".noema/checkpoints/hpcm/hpcm_base_lambda0.013_mse.pth",
        "checkpoint_url": "https://drive.google.com/file/d/1Snq7vkWQdApzCe-gK_V-WuRyMHQRL443/view?usp=drive_link",
        "checkpoint_preset": "hpcm_base_l0.013_mse",
    },
    "evc": {
        "repo_path": ".noema/upstreams/DCVC/DCVC-family/EVC",
        "repo_clone_path": ".noema/upstreams/DCVC",
        "repo_url": "https://github.com/microsoft/DCVC.git",
        "checkpoint": ".noema/checkpoints/evc/EVC_SS_MD.pth.tar",
        "checkpoint_url": "https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211231&authkey=ANrIn85RgtBH2wM",
        "checkpoint_preset": "evc_ss_md",
    },
}

_UPSTREAM_ENTROPY_INFO: Dict[str, JsonDict] = {
    "tcm": {
        "algorithm": "rANS",
        "coder": "CompressAI BufferedRansEncoder/RansDecoder",
        "implementation": "CompressAI ANS backend used by LIC_TCM",
        "language": "C++ extension + Python/PyTorch",
        "note": "TCM's upstream model imports compressai.ans.BufferedRansEncoder and RansDecoder for bitstream coding.",
    },
    "hpcm": {
        "algorithm": "unbounded range Asymmetric Numeral System",
        "coder": "unbounded rANS",
        "implementation": "LIC-HPCM unbounded_ans pybind11 extension",
        "language": "C++ extension + Python/PyTorch",
        "note": "HPCM uses its upstream unbounded_rans arithmetic coder for real bitstream writing.",
    },
    "evc": {
        "algorithm": "range Asymmetric Numeral System",
        "coder": "rANS",
        "implementation": "EVC MLCodec_rans pybind11 extension",
        "language": "C++ extension + Python/PyTorch",
        "note": "EVC's upstream EntropyCoder wraps MLCodec_rans.RansEncoder/RansDecoder.",
    },
}


def _entropy_metadata(codec: str) -> JsonDict:
    info = _UPSTREAM_ENTROPY_INFO.get(codec, {})
    return {
        "entropy_coder": info.get("coder", ""),
        "entropy_algorithm": info.get("algorithm", ""),
        "entropy_implementation": info.get("implementation", ""),
        "entropy_language": info.get("language", ""),
        "entropy_note": info.get("note", ""),
    }


_COMMON_SCHEMA = {
    "repo_path": {
        "type": "string",
        "default": "",
        "description": "Local path to the official upstream repository clone.",
    },
    "repo_url": {
        "type": "string",
        "default": "",
        "description": "Official upstream Git repository URL used when auto setup is enabled.",
    },
    "repo_clone_path": {
        "type": "string",
        "default": "",
        "description": "Local clone target. For EVC this is the DCVC root; repo_path points to DCVC-family/EVC inside it.",
    },
    "checkpoint": {
        "type": "string",
        "default": "",
        "description": "Local path to the pretrained checkpoint for this RD point.",
    },
    "checkpoint_url": {
        "type": "string",
        "default": "",
        "description": "Pretrained checkpoint URL used when auto setup is enabled.",
    },
    "checkpoint_preset": {
        "type": "string",
        "default": "",
        "description": "Dashboard preset identifier for the selected upstream checkpoint.",
    },
    "auto_setup": {
        "type": "boolean",
        "default": False,
        "description": "Opt in to fetching pinned upstream assets. repo_revision and expected_checkpoint_sha256 are mandatory when downloads are needed.",
    },
    "repo_revision": {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{40}$",
        "description": "Immutable 40-character Git commit required for automatic repository setup.",
    },
    "expected_checkpoint_sha256": {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{64}$",
        "description": "Expected checkpoint SHA-256 required for automatic checkpoint download.",
    },
    "setup_timeout_s": {
        "type": "integer",
        "default": DEFAULT_SETUP_TIMEOUT_S,
        "minimum": 30,
        "description": "Maximum time for each upstream auto-setup clone/download operation.",
    },
    "device": {"type": "string", "default": "cpu"},
    "data_plane_backend": dataplane.backend_schema("auto"),
}


def _asset_schema_defaults(codec: str) -> JsonDict:
    defaults = dict(_ASSET_DEFAULTS[codec])
    schema: JsonDict = {}
    for key, value in defaults.items():
        if key in _COMMON_SCHEMA:
            schema[key] = {**_COMMON_SCHEMA[key], "default": value}
    schema["auto_setup"] = {**_COMMON_SCHEMA["auto_setup"], "default": False}
    schema["setup_timeout_s"] = {**_COMMON_SCHEMA["setup_timeout_s"], "default": DEFAULT_SETUP_TIMEOUT_S}
    return schema


_DECODER_SCHEMA = {
    **_COMMON_SCHEMA,
    "on_error": {
        "type": "string",
        "default": "fail",
        "enum": ["fail", "zeros"],
    },
}


class _UpstreamLicOperation(Operation):
    availability_extra = "upstream-lic"
    availability_modules: Tuple[str, ...] = ("torch",)
    availability_label = "upstream LIC"

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _optional_dependency_availability(
            self.availability_extra,
            self.availability_modules,
            self.availability_label,
        )
        return payload


class TcmEncodeOperation(_UpstreamLicOperation):
    id = "model.tcm_encode"
    name = "TCM upstream checkpoint encoder to payload bits"
    availability_modules = ("torch", "compressai", "einops", "timm")
    availability_label = "TCM"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["tcm"]
    params_schema = object_schema(
        {
            **_COMMON_SCHEMA,
            **_asset_schema_defaults("tcm"),
            "N": {"type": "integer", "default": 128, "minimum": 1},
            "M": {"type": "integer", "default": 320, "minimum": 1},
            "pad_to_multiple": {"type": "integer", "default": 128, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _encode_upstream_bits(ctx, "tcm")


class TcmDecodeOperation(_UpstreamLicOperation):
    id = "model.tcm_decode"
    name = "TCM upstream payload bits decoder to image batch"
    availability_modules = TcmEncodeOperation.availability_modules
    availability_label = "TCM"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["tcm"]
    params_schema = object_schema({**_DECODER_SCHEMA, **_asset_schema_defaults("tcm")})

    def run(self, ctx: OperationContext) -> OperationResult:
        return _decode_upstream_bits(ctx, "tcm")


class HpcmEncodeOperation(_UpstreamLicOperation):
    id = "model.hpcm_encode"
    name = "HPCM upstream checkpoint encoder to payload bits"
    availability_modules = ("torch", "einops")
    availability_label = "HPCM"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["hpcm"]
    params_schema = object_schema(
        {
            **_COMMON_SCHEMA,
            **_asset_schema_defaults("hpcm"),
            "model_name": {
                "type": "string",
                "default": "HPCM_Base",
                "enum": ["HPCM_Base", "HPCM_Large", "HPCM_Base_PhiContext", "HPCM_1B"],
            },
            "scale_table_levels": {"type": "integer", "default": 60, "minimum": 1},
            "pad_to_multiple": {"type": "integer", "default": 256, "minimum": 1},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _encode_upstream_bits(ctx, "hpcm")


class HpcmDecodeOperation(_UpstreamLicOperation):
    id = "model.hpcm_decode"
    name = "HPCM upstream payload bits decoder to image batch"
    availability_modules = HpcmEncodeOperation.availability_modules
    availability_label = "HPCM"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["hpcm"]
    params_schema = object_schema({**_DECODER_SCHEMA, **_asset_schema_defaults("hpcm")})

    def run(self, ctx: OperationContext) -> OperationResult:
        return _decode_upstream_bits(ctx, "hpcm")


class EvcEncodeOperation(_UpstreamLicOperation):
    id = "model.evc_encode"
    name = "EVC upstream checkpoint encoder to payload bits"
    availability_modules = ("torch",)
    availability_label = "EVC"
    input_kinds = {"images": ["image.batch.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["evc"]
    params_schema = object_schema(
        {
            **_COMMON_SCHEMA,
            **_asset_schema_defaults("evc"),
            "model_name": {
                "type": "string",
                "default": "EVC_SS",
                "enum": [
                    "EVC_LL",
                    "EVC_ML",
                    "EVC_SL",
                    "EVC_LM",
                    "EVC_LS",
                    "EVC_MM",
                    "EVC_SS",
                    "Scale_EVC_SL",
                    "Scale_EVC_SS",
                ],
            },
            "rate_idx": {"type": "integer", "default": 0, "minimum": 0},
            "ec_thread": {"type": "boolean", "default": False},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _encode_upstream_bits(ctx, "evc")


class EvcDecodeOperation(_UpstreamLicOperation):
    id = "model.evc_decode"
    name = "EVC upstream payload bits decoder to image batch"
    availability_modules = EvcEncodeOperation.availability_modules
    availability_label = "EVC"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["evc"]
    params_schema = object_schema({**_DECODER_SCHEMA, **_asset_schema_defaults("evc")})

    def run(self, ctx: OperationContext) -> OperationResult:
        return _decode_upstream_bits(ctx, "evc")


class EvcOnnxExportOperation(_UpstreamLicOperation):
    id = "model.evc_export_onnx"
    name = "Convert EVC neural transforms to ONNX"
    availability_extra = "onnx"
    availability_modules = ("torch", "onnx", "onnxruntime")
    availability_label = "EVC ONNX Runtime"
    output_kinds = {"model": "model.onnx.bundle"}
    entropy_info = _UPSTREAM_ENTROPY_INFO["evc"]
    params_schema = object_schema(
        {
            **_COMMON_SCHEMA,
            **_asset_schema_defaults("evc"),
            "model_name": {
                "type": "string",
                "default": "EVC_SS",
                "enum": [
                    "EVC_LL",
                    "EVC_ML",
                    "EVC_SL",
                    "EVC_LM",
                    "EVC_LS",
                    "EVC_MM",
                    "EVC_SS",
                    "Scale_EVC_SL",
                    "Scale_EVC_SS",
                ],
            },
            "rate_idx": {"type": "integer", "default": 0, "minimum": 0},
            "ec_thread": {"type": "boolean", "default": False},
            "export_height": {"type": "integer", "default": 64, "minimum": 64},
            "export_width": {"type": "integer", "default": 64, "minimum": 64},
            "opset": {"type": "integer", "default": 18, "minimum": 18},
            "validate_export": {"type": "boolean", "default": True},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _export_evc_onnx(ctx)


class EvcOnnxEncodeOperation(_UpstreamLicOperation):
    id = "model.evc_onnx_encode"
    name = "EVC ONNX Runtime encoder to payload bits"
    availability_extra = "onnx"
    availability_modules = ("torch", "onnxruntime")
    availability_label = "EVC ONNX Runtime"
    input_kinds = {"images": ["image.batch.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    entropy_info = {
        **_UPSTREAM_ENTROPY_INFO["evc"],
        "implementation": "ONNX Runtime EVC neural transforms + EVC MLCodec_rans entropy backend",
        "language": "ONNX Runtime + C++ extension + Python/PyTorch",
    }
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "device": {"type": "string", "default": "cpu"},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _encode_evc_onnx_bits(ctx, self.entropy_info)


class EvcOnnxDecodeOperation(_UpstreamLicOperation):
    id = "model.evc_onnx_decode"
    name = "EVC ONNX Runtime payload bits decoder to image batch"
    availability_extra = "onnx"
    availability_modules = ("torch", "onnxruntime")
    availability_label = "EVC ONNX Runtime"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"], "model": ["model.onnx.bundle"]}
    output_kinds = {"images": "image.batch.numpy"}
    entropy_info = EvcOnnxEncodeOperation.entropy_info
    params_schema = object_schema(
        {
            "provider": {"type": "string", "default": "CPUExecutionProvider"},
            "device": {"type": "string", "default": "cpu"},
            "on_error": {"type": "string", "default": "fail", "enum": ["fail", "zeros"]},
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        return _decode_evc_onnx_bits(ctx, self.entropy_info)


def _encode_upstream_bits(ctx: OperationContext, codec: str) -> OperationResult:
    images, input_metadata = _load_images(ctx.require_input("images").path)
    data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
    timing_records = []
    _report_setup_progress(ctx, "Preparing %s sender model" % codec.upper())
    setup_start = time.perf_counter()
    _prepare_upstream_assets(ctx, codec, ctx.params)
    runner = _runner(codec, ctx.params)
    runner.model()
    append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
    _report_example_progress(ctx, 0, int(images.shape[0]), "encoding")
    entries = []
    for index in range(int(images.shape[0])):
        _report_example_active_progress(ctx, index, int(images.shape[0]), "encoding")
        image = _image_for_index(images, input_metadata, index)
        total_start = time.perf_counter()
        entries.append(
            timed_call(
                timing_records,
                index,
                "encoder.payload_encode",
                lambda image=image: runner.encode_one(image),
                image_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
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
    codec_payload = {
        "payload_version": 1,
        "codec": codec,
        "runner": runner.payload_metadata(),
        "entries": entries,
        "original_shape": list(images.shape),
    }
    raw = safe_payload_dumps(codec_payload)
    bits, byte_backend = _bytes_to_bits(raw, data_backend)
    metadata = dict(input_metadata)
    metadata.update(
        {
            "codec": codec,
            "upstream_repo": runner.repo_path,
            "checkpoint": runner.checkpoint,
            "byte_count": len(raw),
            "bit_count": int(bits.size),
            "bit_role": "payload",
            "payload_bit_count": int(bits.size),
            "payload_format": SAFE_PAYLOAD_FORMAT,
            "source_bit_storage": "packed_bytes",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "entropy_coded": True,
            "byte_stream_data_plane_backend": byte_backend,
            **_entropy_metadata(codec),
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
        },
        metadata={
            "codec": codec,
            "checkpoint": runner.checkpoint,
            "model": runner.model_label,
            **_entropy_metadata(codec),
            "codec_timing": codec_timing_metadata(
                "encoder",
                timing_records,
                runner="pytorch",
                notes={
                    "encoder.payload_encode": "%s upstream encode_one calls a combined bitstream API; it can include neural transforms plus entropy coding." % codec.upper(),
                },
            ),
        },
    )


def _decode_upstream_bits(ctx: OperationContext, codec: str) -> OperationResult:
    input_artifact = ctx.require_input("bits")
    bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
    data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
    byte_backend = "python_numpy"
    timing_records = []
    try:
        raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
        codec_payload = safe_payload_loads(raw)
        if codec_payload.get("codec") != codec:
            raise RuntimeError("Payload codec is %s, expected %s" % (codec_payload.get("codec"), codec))
        params = _payload_params(codec_payload.get("runner", {}), ctx.params)
        _report_setup_progress(ctx, "Preparing %s receiver model" % codec.upper())
        setup_start = time.perf_counter()
        _prepare_upstream_assets(ctx, codec, params)
        runner = _runner(codec, params)
        runner.model()
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        entries = codec_payload.get("entries") or []
        _report_example_progress(ctx, 0, len(entries) or 1, "decoding")
        decoded = []
        for index, entry in enumerate(entries):
            _report_example_active_progress(ctx, index, len(entries) or 1, "decoding")
            total_start = time.perf_counter()
            decoded.append(
                timed_call(
                    timing_records,
                    index,
                    "decoder.payload_decode",
                    lambda entry=entry: runner.decode_one(entry),
                    image_shape=entry.get("original_shape"),
                )
            )
            append_measurement(
                timing_records,
                index,
                "decoder.total",
                time.perf_counter() - total_start,
                image_shape=entry.get("original_shape"),
            )
            _report_example_progress(ctx, index + 1, len(entries), "decoding")
        if not decoded:
            raise RuntimeError("Payload did not contain encoded images")
        images, decoded_shapes = _stack_image_list(decoded)
    except Exception as exc:
        if str(ctx.params.get("on_error", "fail")) != "zeros":
            raise RuntimeError("%s decode failed, likely due to corrupted payload bits or missing upstream setup: %s" % (codec.upper(), exc))
        shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
        images = np.zeros(shape, dtype=np.uint8)
        _report_example_progress(ctx, int(shape[0]) if shape else 1, int(shape[0]) if shape else 1, "decoding")

    output_metadata = dict(metadata)
    if "decoded_shapes" in locals():
        output_metadata["original_shapes"] = decoded_shapes
    output_metadata.update({
        "source": "%s_decode" % codec,
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "byte_stream_data_plane_backend": byte_backend,
    })
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metadata={
            "shape": list(images.shape),
            **_entropy_metadata(codec),
            "codec_timing": codec_timing_metadata(
                "decoder",
                timing_records,
                runner="pytorch",
                notes={
                    "decoder.payload_decode": "%s upstream decode_one calls a combined bitstream API; it can include entropy decoding plus neural synthesis transforms." % codec.upper(),
                },
            ),
        },
    )


def _export_evc_onnx(ctx: OperationContext) -> OperationResult:
    start = time.perf_counter()
    torch = _require_torch()
    onnx, ort = _require_onnx_stack()
    _prepare_upstream_assets(ctx, "evc", ctx.params)
    runner = _EvcRunner(ctx.params)
    model = runner.model()
    q_scale = runner._q_scale()
    device = runner.device
    height = int(ctx.params.get("export_height", 64))
    width = int(ctx.params.get("export_width", 64))
    height += (64 - height % 64) % 64
    width += (64 - width % 64) % 64
    opset = int(ctx.params.get("opset", 17))
    bundle_dir = ctx.step_dir / "onnx_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = bundle_dir / "analysis.onnx"
    hyper_analysis_path = bundle_dir / "hyper_analysis.onnx"
    hyper_synthesis_path = bundle_dir / "hyper_synthesis.onnx"
    spatial_prior_path = bundle_dir / "spatial_prior.onnx"
    synthesis_path = bundle_dir / "synthesis.onnx"
    dummy = torch.zeros((1, 3, height, width), dtype=torch.float32, device=device)
    total_steps = 6
    ctx.report_progress("Exporting EVC analysis transform", phase="conversion", completed=1, total=total_steps, percent=100.0 / total_steps)
    try:
        analysis = _evc_analysis_wrapper(torch, model, q_scale).to(device).eval()
        with torch.no_grad():
            y = analysis(dummy)
        torch.onnx.export(
            analysis,
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
        raise RuntimeError("EVC analysis transform could not be exported to ONNX: %s" % exc) from exc

    ctx.report_progress("Exporting EVC hyper-analysis transform", phase="conversion", completed=2, total=total_steps, percent=200.0 / total_steps)
    try:
        with torch.no_grad():
            z = model.hyper_enc(y)
        torch.onnx.export(
            model.hyper_enc,
            y,
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
        raise RuntimeError("EVC hyper-analysis transform could not be exported to ONNX: %s" % exc) from exc

    ctx.report_progress("Exporting EVC hyper-synthesis/prior transform", phase="conversion", completed=3, total=total_steps, percent=300.0 / total_steps)
    try:
        z_hat = torch.round(z)
        hyper_synthesis = _evc_hyper_synthesis_wrapper(torch, model).to(device).eval()
        with torch.no_grad():
            q_step, scales, means = hyper_synthesis(z_hat)
        torch.onnx.export(
            hyper_synthesis,
            z_hat,
            str(hyper_synthesis_path),
            input_names=["hyper_latents"],
            output_names=["q_step", "scales", "means"],
            dynamic_axes={
                "hyper_latents": {0: "batch", 2: "hyper_height", 3: "hyper_width"},
                "q_step": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "scales": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "means": {0: "batch", 2: "latent_height", 3: "latent_width"},
            },
            opset_version=opset,
        )
    except Exception as exc:
        raise RuntimeError("EVC hyper-synthesis transform could not be exported to ONNX: %s" % exc) from exc

    ctx.report_progress("Exporting EVC spatial prior", phase="conversion", completed=4, total=total_steps, percent=400.0 / total_steps)
    try:
        spatial_channels = int(q_step.shape[1]) * 4
        spatial_dummy = torch.zeros((1, spatial_channels, int(q_step.shape[2]), int(q_step.shape[3])), dtype=torch.float32, device=device)
        torch.onnx.export(
            model.y_spatial_prior,
            spatial_dummy,
            str(spatial_prior_path),
            input_names=["prior_params"],
            output_names=["spatial_params"],
            dynamic_axes={
                "prior_params": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "spatial_params": {0: "batch", 2: "latent_height", 3: "latent_width"},
            },
            opset_version=opset,
        )
    except Exception as exc:
        raise RuntimeError("EVC spatial prior could not be exported to ONNX: %s" % exc) from exc

    ctx.report_progress("Exporting EVC synthesis transform", phase="conversion", completed=5, total=total_steps, percent=500.0 / total_steps)
    try:
        torch.onnx.export(
            model.dec,
            y,
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
        raise RuntimeError("EVC synthesis transform could not be exported to ONNX: %s" % exc) from exc

    try:
        for path in (analysis_path, hyper_analysis_path, hyper_synthesis_path, spatial_prior_path, synthesis_path):
            onnx.checker.check_model(str(path))
        if bool(ctx.params.get("validate_export", True)):
            providers = _onnx_providers("CPUExecutionProvider", ort)
            for path in (analysis_path, hyper_analysis_path, hyper_synthesis_path, spatial_prior_path, synthesis_path):
                session = ort.InferenceSession(str(path), providers=providers)
                _assert_onnx_session_provider(session, providers, path.name)
    except Exception as exc:
        raise RuntimeError("Exported EVC ONNX bundle failed validation: %s" % exc) from exc

    artifact_sizes = _model_artifact_sizes(
        encoder_paths=[analysis_path, hyper_analysis_path, hyper_synthesis_path, spatial_prior_path],
        decoder_paths=[hyper_synthesis_path, spatial_prior_path, synthesis_path],
    )
    metadata = {
        "codec": "evc",
        "source_runtime": "pytorch",
        "runtime": "onnxruntime",
        "model": runner.model_label,
        "model_name": runner.model_label,
        "checkpoint": runner.checkpoint,
        "repo_path": runner.repo_path,
        "rate_idx": int(ctx.params.get("rate_idx", 0)),
        "q_scale": float(q_scale.detach().cpu().reshape(-1)[0]),
        "ec_thread": bool(ctx.params.get("ec_thread", False)),
        "opset": opset,
        "analysis_path": str(analysis_path),
        "hyper_analysis_path": str(hyper_analysis_path),
        "hyper_synthesis_path": str(hyper_synthesis_path),
        "spatial_prior_path": str(spatial_prior_path),
        "synthesis_path": str(synthesis_path),
        "representation": "bits",
        "export_input_shape": [1, 3, height, width],
        "runner": runner.payload_metadata(),
        "note": "This ONNX bundle exports EVC neural transforms. The EVC MLCodec_rans entropy coder remains the bitstream adapter.",
        **artifact_sizes,
        **_entropy_metadata("evc"),
    }
    manifest_path = ctx.output_path("onnx_bundle", ".json")
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    ctx.report_progress("Prepared EVC ONNX bundle", phase="conversion", completed=total_steps, total=total_steps, percent=100.0)
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


def _encode_evc_onnx_bits(ctx: OperationContext, entropy_info: JsonDict) -> OperationResult:
    torch = _require_torch()
    _, ort = _require_onnx_stack(require_onnx=False)
    images, input_metadata = _load_images(ctx.require_input("images").path)
    bundle = _load_onnx_bundle(ctx.require_input("model"), ("analysis_path", "hyper_analysis_path", "hyper_synthesis_path", "spatial_prior_path"))
    providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
    data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", input_metadata.get("byte_stream_data_plane_backend", "auto")))
    runner_params = _payload_params(bundle.get("runner", {}), {"device": str(ctx.params.get("device") or bundle.get("runner", {}).get("device") or "cpu")})
    _report_setup_progress(ctx, "Preparing EVC ONNX sender assets")
    _prepare_upstream_assets(ctx, "evc", runner_params)
    runner = _EvcRunner(runner_params)
    _report_setup_progress(ctx, "Loading EVC sender model")
    model = runner.model()
    q_scale = runner._q_scale()
    timing_records = []
    _report_setup_progress(ctx, "Creating EVC ONNX Runtime encoder sessions")
    setup_start = time.perf_counter()
    sessions = _evc_onnx_sessions(bundle, ort, providers, include_synthesis=False)
    evidence = _evc_onnx_evidence(ort, sessions, providers[0])
    append_measurement(timing_records, None, "encoder.setup", time.perf_counter() - setup_start)
    entries = []
    total = int(images.shape[0])
    _report_example_progress(ctx, 0, total, "encoding")
    with _cuda_sync_guard(torch, runner.device), torch.no_grad():
        for index in range(int(images.shape[0])):
            _report_example_active_progress(ctx, index, total, "encoding")
            image = _image_for_index(images, input_metadata, index)
            total_start = time.perf_counter()
            image_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
            tensor = timed_call(
                timing_records,
                index,
                "encoder.preprocess",
                lambda image=image: _image_to_torch(torch, image, runner.device),
                image_shape=image_shape,
            )
            padded, padding = _pad_bottom_right(torch, tensor, 64)
            y_np = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda padded=padded: _onnx_run(sessions["analysis"], _torch_to_numpy(torch, padded))[0],
                image_shape=image_shape,
            )
            z_np = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda y_np=y_np: _onnx_run(sessions["hyper_analysis"], y_np)[0],
                image_shape=image_shape,
            )
            z_hat = torch.round(torch.from_numpy(z_np.astype(np.float32, copy=False))).to(runner.device)
            q_step_np, scales_np, means_np = timed_call(
                timing_records,
                index,
                "encoder.model",
                lambda z_hat=z_hat: _onnx_run(sessions["hyper_synthesis"], _torch_to_numpy(torch, z_hat)),
                image_shape=image_shape,
            )
            y_q_w_0, y_q_w_1, scales_w_0, scales_w_1 = _evc_encode_dual_prior_onnx(
                torch,
                model,
                sessions["spatial_prior"],
                y_np,
                q_step_np,
                scales_np,
                means_np,
                runner.device,
                timing_records,
                index,
                image_shape,
            )
            entropy_start = time.perf_counter()
            model.entropy_coder.reset()
            model.bit_estimator_z.encode(z_hat)
            model.gaussian_encoder.encode(y_q_w_0, scales_w_0)
            model.gaussian_encoder.encode(y_q_w_1, scales_w_1)
            model.entropy_coder.flush()
            bit_stream = model.entropy_coder.get_encoded_stream()
            append_measurement(timing_records, index, "encoder.symbol_encode", time.perf_counter() - entropy_start, image_shape=image_shape)
            entries.append(
                {
                    "bit_stream": bit_stream,
                    "height": int(padded.shape[2]),
                    "width": int(padded.shape[3]),
                    "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
                    "padding": padding,
                    "q_scale": float(q_scale.detach().cpu().reshape(-1)[0]),
                    "payload_bytes": len(bit_stream),
                }
            )
            append_measurement(timing_records, index, "encoder.total", time.perf_counter() - total_start, image_shape=image_shape)
            _report_example_progress(ctx, index + 1, total, "encoding")
    codec_payload = {
        "payload_version": 2,
        "codec": "evc",
        "runtime": "onnxruntime",
        "runner": runner.payload_metadata(),
        "entries": entries,
        "original_shape": list(images.shape),
    }
    raw = safe_payload_dumps(codec_payload)
    bits, byte_backend = _bytes_to_bits(raw, data_backend)
    metadata = dict(input_metadata)
    metadata.update(
        {
            "codec": "evc",
            "runtime": "onnxruntime",
            "provider": providers[0],
            "upstream_repo": runner.repo_path,
            "checkpoint": runner.checkpoint,
            "byte_count": len(raw),
            "bit_count": int(bits.size),
            "bit_role": "payload",
            "payload_bit_count": int(bits.size),
            "payload_format": SAFE_PAYLOAD_FORMAT,
            "source_bit_storage": "packed_bytes",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "entropy_coded": True,
            "runtime_evidence": evidence,
            "byte_stream_data_plane_backend": byte_backend,
            **_entropy_metadata("evc"),
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
        },
        metadata={
            "codec": "evc",
            "checkpoint": runner.checkpoint,
            "model": runner.model_label,
            "runtime": "onnxruntime",
            "provider": providers[0],
            "runtime_evidence": evidence,
            **_entropy_metadata("evc"),
            "codec_timing": codec_timing_metadata(
                "encoder",
                timing_records,
                runner="onnxruntime+evc_entropy",
                notes={
                    **evidence,
                    "encoder.model": "EVC analysis, hyper-analysis, hyper-synthesis, and spatial-prior neural transforms run with ONNX Runtime.",
                    "encoder.symbol_encode": "EVC MLCodec_rANS writes z/y bitstreams.",
                },
            ),
        },
    )


def _decode_evc_onnx_bits(ctx: OperationContext, entropy_info: JsonDict) -> OperationResult:
    torch = _require_torch()
    _, ort = _require_onnx_stack(require_onnx=False)
    input_artifact = ctx.require_input("bits")
    bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
    bundle = _load_onnx_bundle(ctx.require_input("model"), ("hyper_synthesis_path", "spatial_prior_path", "synthesis_path"))
    providers = _onnx_providers(str(ctx.params.get("provider", "CPUExecutionProvider")), ort)
    data_backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", metadata.get("byte_stream_data_plane_backend", "auto")))
    byte_backend = "python_numpy"
    timing_records = []
    try:
        raw, byte_backend = _bits_to_bytes(bits, int(metadata["byte_count"]), data_backend)
        codec_payload = safe_payload_loads(raw)
        if codec_payload.get("codec") != "evc":
            raise RuntimeError("Payload codec is %s, expected evc" % codec_payload.get("codec"))
        runner_params = _payload_params(codec_payload.get("runner", {}), {"device": str(ctx.params.get("device") or "cpu")})
        _report_setup_progress(ctx, "Preparing EVC ONNX receiver assets")
        setup_start = time.perf_counter()
        _prepare_upstream_assets(ctx, "evc", runner_params)
        runner = _EvcRunner(runner_params)
        _report_setup_progress(ctx, "Loading EVC receiver model and ONNX Runtime sessions")
        model = runner.model()
        sessions = _evc_onnx_sessions(bundle, ort, providers, include_analysis=False)
        evidence = _evc_onnx_evidence(ort, sessions, providers[0])
        append_measurement(timing_records, None, "decoder.setup", time.perf_counter() - setup_start)
        entries = codec_payload.get("entries") or []
        if not entries:
            raise RuntimeError("Payload did not contain encoded images")
        decoded = []
        _report_example_progress(ctx, 0, len(entries), "decoding")
        with _cuda_sync_guard(torch, runner.device), torch.no_grad():
            for index, entry in enumerate(entries):
                _report_example_active_progress(ctx, index, len(entries), "decoding")
                total_start = time.perf_counter()
                image_shape = entry.get("original_shape")
                model.entropy_coder.set_stream(entry["bit_stream"])
                z_size = _evc_z_size(runner, int(entry["height"]), int(entry["width"]))
                entropy_start = time.perf_counter()
                z_hat = model.bit_estimator_z.decode_stream(z_size, torch.float32, runner.device)
                append_measurement(timing_records, index, "decoder.symbol_decode", time.perf_counter() - entropy_start, image_shape=image_shape)
                q_step_np, scales_np, means_np = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda z_hat=z_hat: _onnx_run(sessions["hyper_synthesis"], _torch_to_numpy(torch, z_hat)),
                    image_shape=image_shape,
                )
                y_hat = _evc_decode_dual_prior_onnx(
                    torch,
                    model,
                    sessions["spatial_prior"],
                    q_step_np,
                    scales_np,
                    means_np,
                    runner.device,
                    timing_records,
                    index,
                    image_shape,
                )
                curr_q = torch.clamp_min(model.q_basic.to(runner.device).to(torch.float32), 0.5) * float(entry["q_scale"])
                y_hat = y_hat * curr_q
                output = timed_call(
                    timing_records,
                    index,
                    "decoder.model",
                    lambda y_hat=y_hat: _onnx_run(sessions["synthesis"], _torch_to_numpy(torch, y_hat))[0],
                    image_shape=image_shape,
                )
                cropped = _crop_bottom_right(torch.from_numpy(output).to(runner.device), tuple(entry["padding"]))
                decoded.append(_torch_to_image(torch, cropped, tuple(entry["original_shape"])))
                append_measurement(timing_records, index, "decoder.total", time.perf_counter() - total_start, image_shape=image_shape)
                _report_example_progress(ctx, index + 1, len(entries), "decoding")
        images, decoded_shapes = _stack_image_list(decoded)
    except Exception as exc:
        if str(ctx.params.get("on_error", "fail")) != "zeros":
            raise RuntimeError("EVC ONNX decode failed, likely due to corrupted payload bits or missing upstream setup: %s" % exc)
        shape = tuple(int(item) for item in metadata.get("original_shape", [1, 64, 64, 3]))
        images = np.zeros(shape, dtype=np.uint8)
        evidence = {"runtime": "onnxruntime", "error": _exception_text(exc)}
    output_metadata = dict(metadata)
    if "decoded_shapes" in locals():
        output_metadata["original_shapes"] = decoded_shapes
    output_metadata.update({
        "source": "evc_onnx_decode",
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "byte_stream_data_plane_backend": byte_backend,
    })
    path = ctx.output_path("images", ".npz")
    np.savez_compressed(path, images=images, metadata_json=json.dumps(output_metadata))
    return OperationResult(
        outputs={"images": artifact("image.batch.numpy", path, output_metadata)},
        metadata={
            "shape": list(images.shape),
            "runtime": "onnxruntime",
            "provider": providers[0],
            "runtime_evidence": evidence,
            **_entropy_metadata("evc"),
            "codec_timing": codec_timing_metadata(
                "decoder",
                timing_records,
                runner="onnxruntime+evc_entropy",
                notes={
                    **evidence,
                    "decoder.model": "EVC hyper-synthesis, spatial-prior, and synthesis neural transforms run with ONNX Runtime.",
                    "decoder.symbol_decode": "EVC MLCodec_rANS reads z/y bitstreams.",
                },
            ),
        },
    )


def _runner(codec: str, params: JsonDict):
    if codec == "tcm":
        return _TcmRunner(params)
    if codec == "hpcm":
        return _HpcmRunner(params)
    if codec == "evc":
        return _EvcRunner(params)
    raise RuntimeError("Unknown upstream LIC codec: %s" % codec)


def _prepare_upstream_assets(ctx: OperationContext, codec: str, params: JsonDict) -> None:
    auto_setup = _truthy(params.get("auto_setup", False))
    repo_path = _path_value(params.get("repo_path"))
    clone_path = _path_value(params.get("repo_clone_path")) or repo_path
    checkpoint = _path_value(params.get("checkpoint"))
    repo_revision = _normalized_git_revision(params.get("repo_revision"))
    checkpoint_sha256 = _normalized_sha256(
        params.get("expected_checkpoint_sha256"),
        label="%s checkpoint" % codec.upper(),
    )
    if repo_revision and clone_path and clone_path.exists():
        _require_repo_revision(clone_path, repo_revision)
    if checkpoint and checkpoint.exists():
        _validate_checkpoint_file(checkpoint)
        _require_checkpoint_sha256(checkpoint, checkpoint_sha256, codec)
    if not auto_setup:
        return
    timeout_s = max(30, int(params.get("setup_timeout_s") or DEFAULT_SETUP_TIMEOUT_S))
    repo_url = str(params.get("repo_url") or "").strip()
    if repo_path and not repo_path.exists() and repo_url:
        _require_https_url(repo_url, "upstream repository")
        repo_revision = _normalized_git_revision(
            params.get("repo_revision"),
            required=True,
        )
        ctx.report_progress(
            "Preparing %s source repository" % codec.upper(),
            phase="setup",
            status="running",
            completed=0,
            total=2,
            percent=0,
            unit="assets",
            op=ctx.step_id,
        )
        _clone_repo(repo_url, clone_path, timeout_s, repo_revision)
        if not repo_path.exists():
            raise RuntimeError(
                "%s setup cloned %s, but expected repo path is missing: %s"
                % (codec.upper(), clone_path, repo_path)
            )
    checkpoint_url = str(params.get("checkpoint_url") or "").strip()
    if checkpoint and checkpoint.exists():
        try:
            _validate_checkpoint_file(checkpoint)
        except RuntimeError:
            if not checkpoint_url:
                raise
            try:
                checkpoint.unlink()
            except OSError:
                raise
    if checkpoint and not checkpoint.exists() and checkpoint_url:
        _require_https_url(checkpoint_url, "%s checkpoint" % codec.upper())
        checkpoint_sha256 = _normalized_sha256(
            params.get("expected_checkpoint_sha256"),
            label="%s checkpoint" % codec.upper(),
            required=True,
        )
        ctx.report_progress(
            "Downloading %s pretrained checkpoint" % codec.upper(),
            phase="setup",
            status="running",
            completed=1,
            total=2,
            percent=50,
            unit="assets",
            op=ctx.step_id,
        )
        _download_checkpoint(
            checkpoint_url,
            checkpoint,
            timeout_s,
            expected_sha256=checkpoint_sha256,
        )
    ctx.report_progress(
        "Prepared %s upstream assets" % codec.upper(),
        phase="setup",
        status="running",
        completed=2,
        total=2,
        percent=100,
        unit="assets",
        op=ctx.step_id,
    )


def _clone_repo(url: str, target: Path, timeout_s: int, revision: str) -> None:
    target = target.expanduser().resolve()
    if target.exists():
        _require_repo_revision(target, revision)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".%s." % target.name,
            suffix=".clone-part",
            dir=str(target.parent),
        )
    )
    try:
        commands = (
            ["git", "init", str(temporary)],
            ["git", "-C", str(temporary), "remote", "add", "origin", url],
            ["git", "-C", str(temporary), "fetch", "--depth", "1", "origin", revision],
            ["git", "-C", str(temporary), "checkout", "--detach", "FETCH_HEAD"],
        )
        for command in commands:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        _require_repo_revision(temporary, revision)
        try:
            # Reserving the final pathname with mkdir is an atomic no-clobber
            # operation on every supported platform.  Only this clone owns the
            # empty directory after it succeeds.
            target.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise RuntimeError(
                "Upstream repository destination appeared during clone: %s" % target
            ) from exc
        try:
            for child in temporary.iterdir():
                child.rename(target / child.name)
            temporary.rmdir()
            temporary = None
        except Exception:
            # target was created atomically by this call, so this cleanup never
            # removes a pre-existing or competing process's directory.
            if target.exists() and not target.is_symlink():
                shutil.rmtree(target)
            raise
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out after %ss while fetching %s at %s" % (timeout_s, url, revision)) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError("Could not fetch %s at %s: %s" % (url, revision, detail or exc)) from exc
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _download_checkpoint(
    url: str,
    target: Path,
    timeout_s: int,
    *,
    expected_sha256: str,
) -> None:
    target = target.expanduser().resolve()
    if target.exists():
        _validate_checkpoint_file(target)
        _require_checkpoint_sha256(target, expected_sha256, "upstream")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError(
            "Refusing to replace a symbolic-link checkpoint destination: %s" % target
        )
    hostname = (urllib.parse.urlsplit(url).hostname or "").lower()
    if hostname in {"drive.google.com", "docs.google.com"}:
        url = _google_drive_direct_download_url(url)
    _download_file(
        url,
        target,
        timeout_s,
        expected_sha256=expected_sha256,
    )
    _validate_checkpoint_file(target)


def _download_file(
    url: str,
    target: Path,
    timeout_s: int,
    *,
    expected_sha256: str,
) -> None:
    try:
        download_verified_https(
            url,
            target,
            expected_sha256=expected_sha256,
            max_bytes=MAX_CHECKPOINT_DOWNLOAD_BYTES,
            timeout_s=timeout_s,
            opener=urlopen,
        )
    except Exception as exc:
        raise RuntimeError("Could not download checkpoint %s within %ss: %s" % (url, timeout_s, exc)) from exc


def _google_drive_direct_download_url(url: str) -> str:
    file_id = _google_drive_file_id(url)
    if not file_id:
        raise RuntimeError("Could not resolve Google Drive file id from URL: %s" % url)
    return "https://drive.usercontent.google.com/download?%s" % urllib.parse.urlencode(
        {"id": file_id, "export": "download", "confirm": "t"}
    )


def _google_drive_file_id(url: str) -> str:
    match = re.search(r"/d/([^/]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else ""


def _require_https_url(url: str, label: str) -> str:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise RuntimeError("%s URL must use HTTPS" % label)
    return str(url)


def _require_bounded_file_size(path: Path, max_bytes: int, label: str) -> int:
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise RuntimeError("Could not inspect %s at %s: %s" % (label, path, exc)) from exc
    if size <= 0 or size > int(max_bytes):
        raise RuntimeError(
            "%s size %d is outside the allowed 1..%d byte range"
            % (label, size, int(max_bytes))
        )
    return size


def _validate_checkpoint_file(path: Path) -> None:
    path = path.expanduser().resolve()
    _require_bounded_file_size(
        path, MAX_CHECKPOINT_DOWNLOAD_BYTES, "upstream checkpoint"
    )
    try:
        head = path.read_bytes()[:1024]
    except OSError as exc:
        raise RuntimeError("Could not read checkpoint %s: %s" % (path, exc)) from exc
    normalized = head.lstrip().lower()
    if b"<!doctype html" in normalized or b"<html" in normalized:
        raise RuntimeError(
            "Checkpoint download at %s is an HTML page, not a model checkpoint. "
            "The upstream link probably requires sign-in or has expired; provide a local checkpoint path or update the checkpoint URL."
            % path
        )


def _normalized_sha256(value: Any, *, label: str, required: bool = False) -> str:
    digest = str(value or "").strip().lower()
    if not digest:
        if required:
            raise RuntimeError("%s requires expected_checkpoint_sha256" % label)
        return ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("%s SHA-256 must be 64 hexadecimal characters" % label)
    return digest


def _require_checkpoint_sha256(path: Path, expected: str, codec: str) -> str:
    actual = file_sha256(path)
    if expected and actual != expected:
        raise RuntimeError(
            "%s checkpoint SHA-256 mismatch: expected %s, got %s"
            % (codec.upper(), expected, actual)
        )
    return actual


def _normalized_git_revision(value: Any, *, required: bool = False) -> str:
    revision = str(value or "").strip().lower()
    if not revision:
        if required:
            raise RuntimeError("Automatic upstream setup requires a pinned repo_revision")
        return ""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError("repo_revision must be a full 40-character hexadecimal commit")
    return revision


def _repo_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Could not resolve upstream repository revision at %s" % path) from exc
    return completed.stdout.strip().lower()


def _require_repo_revision(path: Path, expected: str) -> str:
    actual = _repo_head(path)
    if actual != expected:
        raise RuntimeError(
            "Upstream repository revision mismatch at %s: expected %s, got %s"
            % (path, expected, actual)
        )
    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if dirty.stdout.strip():
            raise RuntimeError(
                "Upstream repository has tracked changes at pinned revision %s: %s"
                % (actual, path)
            )
        untracked = subprocess.run(
            ["git", "-C", str(path), "ls-files", "--others", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not verify upstream repository worktree at %s" % path
        ) from exc
    executable_sources = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked.stdout.split(b"\0")
        if item and Path(item.decode("utf-8", errors="surrogateescape")).suffix.lower()
        in {".py", ".pyw"}
    )
    if executable_sources:
        raise RuntimeError(
            "Upstream repository contains untracked executable Python source: %s"
            % executable_sources[0]
        )
    return actual


def _exception_text(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _path_value(value: Any):
    text = str(value or "").strip()
    return Path(text).expanduser().resolve() if text else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


class _BaseRunner:
    repo_modules: Tuple[str, ...] = ()

    def __init__(self, params: JsonDict) -> None:
        self.params = dict(params)
        self.repo_path = _required_path(params, "repo_path", "upstream repository")
        self.checkpoint = _required_path(params, "checkpoint", "pretrained checkpoint")
        self.device = str(params.get("device") or "cpu")
        self._model = None
        self.model_label = str(params.get("model_name") or getattr(self, "model_label", self.__class__.__name__))

    def payload_metadata(self) -> JsonDict:
        keep = {
            "repo_path",
            "checkpoint",
            "device",
            "N",
            "M",
            "model_name",
            "scale_table_levels",
            "pad_to_multiple",
        "rate_idx",
        "ec_thread",
        "repo_url",
        "repo_clone_path",
        "checkpoint_url",
        "checkpoint_preset",
        "auto_setup",
        "repo_revision",
        "expected_checkpoint_sha256",
        }
        payload = {key: value for key, value in self.params.items() if key in keep}
        try:
            payload["repo_revision_resolved"] = _repo_head(Path(self.repo_path))
        except RuntimeError:
            payload["repo_revision_resolved"] = ""
        payload["checkpoint_sha256"] = file_sha256(Path(self.checkpoint))
        return payload

    @contextmanager
    def imports(self):
        with _isolated_repo_import(self.repo_path, self.repo_modules):
            yield


class _TcmRunner(_BaseRunner):
    repo_modules = ("models",)
    model_label = "TCM"

    def model(self):
        if self._model is not None:
            return self._model
        torch = _require_torch()
        with self.imports():
            try:
                from models import TCM
            except ImportError as exc:
                raise RuntimeError(
                    "TCM requires the official LIC_TCM repo. Install dependencies "
                    'with `python -m pip install "noema-lab[upstream-lic]"` in an '
                    "installed environment, or `uv sync --extra upstream-lic` in a "
                    "source checkout."
                ) from exc
            net = TCM(
                config=[2, 2, 2, 2, 2, 2],
                head_dim=[8, 16, 32, 32, 16, 8],
                drop_path_rate=0.0,
                N=int(self.params.get("N", 128)),
                M=int(self.params.get("M", 320)),
            ).to(self.device).eval()
            checkpoint = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=True,
            )
            state = checkpoint.get("state_dict", checkpoint)
            state = {str(key).replace("module.", ""): value for key, value in state.items()}
            net.load_state_dict(state)
            if hasattr(net, "update"):
                try:
                    net.update()
                except Exception as exc:
                    raise RuntimeError("TCM entropy model update failed; check CompressAI compatibility") from exc
            self._model = net
        return self._model

    def encode_one(self, image: np.ndarray) -> JsonDict:
        torch = _require_torch()
        model = self.model()
        tensor = _image_to_torch(torch, image, self.device)
        padded, padding = _pad_symmetric(torch, tensor, int(self.params.get("pad_to_multiple", 128)))
        with torch.no_grad():
            encoded = model.compress(padded)
        return {
            "strings": encoded["strings"],
            "shape": encoded["shape"],
            "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            "padding": padding,
            "payload_bytes": _nested_byte_count(encoded["strings"]),
        }

    def decode_one(self, entry: JsonDict) -> np.ndarray:
        torch = _require_torch()
        model = self.model()
        with torch.no_grad():
            decoded = model.decompress(entry["strings"], entry["shape"])["x_hat"]
        cropped = _crop_symmetric(torch, decoded, tuple(entry["padding"]))
        return _torch_to_image(torch, cropped, tuple(entry["original_shape"]))


class _HpcmRunner(_BaseRunner):
    repo_modules = ("src",)

    def model(self):
        if self._model is not None:
            return self._model
        torch = _require_torch()
        model_name = str(self.params.get("model_name") or "HPCM_Base")
        self.model_label = model_name
        with self.imports():
            try:
                module = importlib.import_module("src.models.%s" % model_name)
                net_cls = getattr(module, "HPCM")
            except Exception as exc:
                raise RuntimeError(
                    "HPCM requires the official LIC-HPCM repo. Install dependencies "
                    'with `python -m pip install "noema-lab[upstream-lic]"` in an '
                    "installed environment, or `uv sync --extra upstream-lic` in a "
                    "source checkout."
                ) from exc
            checkpoint = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=True,
            )
            state = checkpoint.get("state_dict", checkpoint)
            with _torch_cuda_allocation_guard(torch, self.device):
                model = net_cls()
            model = model.to(self.device).eval()
            load_result = torch.nn.Module.load_state_dict(model, state, strict=False)
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            missing_not_allowed = [key for key in missing if not key.startswith("adaptive_params_list.")]
            if missing_not_allowed or unexpected:
                raise RuntimeError(
                    "HPCM checkpoint did not match %s; missing=%s unexpected=%s"
                    % (model_name, missing_not_allowed, unexpected)
                )
            try:
                model.update(_scale_table(torch, 0.12, 64, int(self.params.get("scale_table_levels", 60))).to(self.device))
            except Exception as exc:
                raise RuntimeError("HPCM entropy coder update failed; build the LIC-HPCM unbounded rANS extension for this Python/PyTorch environment") from exc
            self._model = model
        return self._model

    def encode_one(self, image: np.ndarray) -> JsonDict:
        torch = _require_torch()
        model = self.model()
        tensor = _image_to_torch(torch, image, self.device)
        padded, padding = _pad_symmetric(torch, tensor, int(self.params.get("pad_to_multiple", 256)))
        with self.imports(), _cuda_sync_guard(torch, self.device), torch.no_grad():
            encoded = model.compress(padded)
        return {
            "strings": encoded["strings"],
            "shape": encoded["shape"],
            "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            "padding": padding,
            "payload_bytes": _nested_byte_count(encoded["strings"]),
        }

    def decode_one(self, entry: JsonDict) -> np.ndarray:
        torch = _require_torch()
        model = self.model()
        with self.imports(), _cuda_sync_guard(torch, self.device), torch.no_grad():
            decoded = model.decompress(entry["strings"], entry["shape"])["x_hat"]
        cropped = _crop_symmetric(torch, decoded, tuple(entry["padding"]))
        return _torch_to_image(torch, cropped, tuple(entry["original_shape"]))


class _EvcRunner(_BaseRunner):
    repo_modules = ("src",)

    def model(self):
        if self._model is not None:
            return self._model
        torch = _require_torch()
        model_name = str(self.params.get("model_name") or "EVC_SS")
        self.model_label = model_name
        _validate_checkpoint_file(Path(self.checkpoint))
        with self.imports():
            try:
                from src.models import build_model
                from src.utils.stream_helper import get_state_dict
            except Exception as exc:
                raise RuntimeError("EVC requires the official DCVC/DCVC-family/EVC repo") from exc
            try:
                state = get_state_dict(self.checkpoint)
            except Exception as exc:
                raise RuntimeError("EVC checkpoint load failed for %s: %s" % (self.checkpoint, _exception_text(exc))) from exc
            try:
                model = build_model(model_name, ec_thread=bool(self.params.get("ec_thread", False)))
                model.load_state_dict(state, verbose=False)
                if hasattr(model, "set_rate"):
                    model.set_rate(int(self.params.get("rate_idx", 0)))
                if hasattr(model, "update"):
                    try:
                        model.update(force=True)
                    except TypeError:
                        model.update()
            except Exception as exc:
                detail = _exception_text(exc)
                if "MLCodec" in detail or "mlcodec" in detail:
                    raise RuntimeError(
                        "EVC MLCodec entropy extension is missing or failed to import. "
                        "Build the DCVC-family/EVC extension under src/build with CMake, then rerun. "
                        "Original error: %s" % detail
                    ) from exc
                raise RuntimeError("EVC model setup failed for %s: %s" % (model_name, detail)) from exc
            self._model = model.to(self.device).eval()
        return self._model

    def _q_scale(self):
        torch = _require_torch()
        with self.imports():
            from src.utils.stream_helper import get_state_dict
            state = get_state_dict(self.checkpoint)
        for key in ("q_scale", "student.q_scale", "teacher.q_scale"):
            if key in state:
                q_scales = state[key].reshape(-1)
                index = min(max(int(self.params.get("rate_idx", 0)), 0), int(q_scales.numel()) - 1)
                return q_scales[index].to(torch.float32).to(self.device)
        raise RuntimeError("EVC checkpoint does not contain q_scale; cannot select rate_idx")

    def encode_one(self, image: np.ndarray) -> JsonDict:
        torch = _require_torch()
        model = self.model()
        tensor = _image_to_torch(torch, image, self.device)
        padded, padding = _pad_bottom_right(torch, tensor, 64)
        q_scale = self._q_scale()
        try:
            with _cuda_sync_guard(torch, self.device), torch.no_grad():
                encoded = model.compress(padded, q_scale)
        except ImportError as exc:
            raise RuntimeError("EVC actual bitstream mode requires building the repo's C++ entropy coder extension") from exc
        return {
            "bit_stream": encoded["bit_stream"],
            "height": int(padded.shape[2]),
            "width": int(padded.shape[3]),
            "original_shape": [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            "padding": padding,
            "q_scale": float(q_scale.detach().cpu().reshape(-1)[0]),
            "payload_bytes": len(encoded["bit_stream"]),
        }

    def decode_one(self, entry: JsonDict) -> np.ndarray:
        torch = _require_torch()
        model = self.model()
        with _cuda_sync_guard(torch, self.device), torch.no_grad():
            decoded = model.decompress(
                entry["bit_stream"],
                int(entry["height"]),
                int(entry["width"]),
                float(entry["q_scale"]),
            )["x_hat"]
        cropped = _crop_bottom_right(decoded, tuple(entry["padding"]))
        return _torch_to_image(torch, cropped, tuple(entry["original_shape"]))


def _required_path(params: JsonDict, key: str, label: str) -> str:
    value = str(params.get(key) or "").strip()
    if not value:
        raise RuntimeError("Configure Codec Settings: %s path is required" % label)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError("Configured %s does not exist: %s" % (label, path))
    return str(path)


def _payload_params(payload_params: JsonDict, decoder_params: JsonDict) -> JsonDict:
    params = dict(payload_params or {})
    for key, value in (decoder_params or {}).items():
        if key in {"repo_path", "checkpoint"} and not str(value or "").strip():
            continue
        params[key] = value
    return params


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to use upstream LIC codecs") from exc
    return torch


def _require_onnx_stack(require_onnx: bool = True):
    onnx = None
    if require_onnx:
        try:
            import onnx as onnx_module
        except ImportError as exc:
            raise RuntimeError(
                'Install with `python -m pip install "noema-lab[onnx]"` in an '
                "installed environment, or `uv sync --extra onnx` in a source "
                "checkout, to export EVC to ONNX"
            ) from exc
        onnx = onnx_module
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            'Install with `python -m pip install "noema-lab[onnx]"` in an '
            "installed environment, or `uv sync --extra onnx` in a source "
            "checkout, to run EVC ONNX Runtime recipes"
        ) from exc
    return onnx, ort


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


def _load_onnx_bundle(input_artifact, required_keys=("analysis_path", "synthesis_path")) -> JsonDict:
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
    if str(payload.get("codec") or "") != "evc":
        raise RuntimeError("EVC ONNX operation received a non-EVC ONNX bundle")
    for key in required_keys:
        path = payload.get(key)
        if not path:
            raise RuntimeError("ONNX bundle manifest is missing %s" % key)
        if not Path(str(path)).is_file():
            raise RuntimeError("ONNX bundle manifest points to missing %s: %s" % (key, path))
    return payload


def _evc_onnx_sessions(bundle: JsonDict, ort, providers: list, include_analysis: bool = True, include_synthesis: bool = True) -> JsonDict:
    paths = {
        "analysis": bundle.get("analysis_path"),
        "hyper_analysis": bundle.get("hyper_analysis_path"),
        "hyper_synthesis": bundle.get("hyper_synthesis_path"),
        "spatial_prior": bundle.get("spatial_prior_path"),
        "synthesis": bundle.get("synthesis_path"),
    }
    if not include_analysis:
        paths.pop("analysis", None)
        paths.pop("hyper_analysis", None)
    if not include_synthesis:
        paths.pop("synthesis", None)
    sessions = {}
    for key, path in paths.items():
        if not path:
            continue
        session = ort.InferenceSession(str(path), providers=providers)
        _assert_onnx_session_provider(session, providers, key)
        sessions[key] = session
    return sessions


def _evc_onnx_evidence(ort, sessions: JsonDict, provider: str) -> JsonDict:
    return {
        "runtime": "onnxruntime",
        "onnxruntime_version": str(getattr(ort, "__version__", "")),
        **onnxruntime_native_evidence(ort),
        "requested_provider": provider,
        "sessions": {
            key: {
                "active_providers": list(session.get_providers()),
                "model_path": str(getattr(session, "_model_path", "")),
                "inputs": [_onnx_value_info(item) for item in session.get_inputs()],
                "outputs": [_onnx_value_info(item) for item in session.get_outputs()],
            }
            for key, session in sessions.items()
        },
    }


def _onnx_value_info(value) -> JsonDict:
    return {
        "name": str(getattr(value, "name", "")),
        "type": str(getattr(value, "type", "")),
        "shape": [str(item) for item in (getattr(value, "shape", None) or [])],
    }


def _onnx_run(session, *arrays):
    inputs = session.get_inputs()
    if len(inputs) != len(arrays):
        if len(arrays) == 1:
            feed = {inputs[0].name: np.asarray(arrays[0], dtype=np.float32)}
        else:
            raise RuntimeError("ONNX session expected %d inputs but got %d" % (len(inputs), len(arrays)))
    else:
        feed = {item.name: np.asarray(array, dtype=np.float32) for item, array in zip(inputs, arrays)}
    return [np.asarray(item).astype(np.float32, copy=False) for item in session.run(None, feed)]


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


def _evc_analysis_wrapper(torch, model, q_scale):
    class AnalysisWrapper(torch.nn.Module):
        def __init__(self, inner, basic, scale) -> None:
            super().__init__()
            self.enc = inner
            self.register_buffer("q_basic", basic.detach().clone().to(torch.float32))
            self.register_buffer("q_scale", scale.detach().clone().reshape(1, 1, 1, 1).to(torch.float32))

        def forward(self, x):
            curr_q = torch.clamp_min(self.q_basic, 0.5) * self.q_scale
            return self.enc(x) / curr_q

    return AnalysisWrapper(model.enc, model.q_basic, q_scale)


def _evc_hyper_synthesis_wrapper(torch, model):
    class HyperSynthesisWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.hyper_dec = inner.hyper_dec
            self.y_prior_fusion = inner.y_prior_fusion

        def forward(self, z_hat):
            params = self.y_prior_fusion(self.hyper_dec(z_hat))
            return params.chunk(3, 1)

    return HyperSynthesisWrapper(model)


def _torch_to_numpy(torch, tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _evc_encode_dual_prior_onnx(
    torch,
    model,
    spatial_session,
    y_np: np.ndarray,
    q_step_np: np.ndarray,
    scales_np: np.ndarray,
    means_np: np.ndarray,
    device: str,
    timing_records,
    example_index: int,
    image_shape,
):
    y = torch.from_numpy(y_np.astype(np.float32, copy=False)).to(device)
    q_step = torch.from_numpy(q_step_np.astype(np.float32, copy=False)).to(device)
    scales = torch.from_numpy(scales_np.astype(np.float32, copy=False)).to(device)
    means = torch.from_numpy(means_np.astype(np.float32, copy=False)).to(device)
    dtype = y.dtype
    _, _, height, width = y.size()
    mask_0, mask_1 = model.get_mask(int(height), int(width), dtype, device)
    quant_step = torch.clamp_min(q_step, 0.5)
    y = y / quant_step
    y_0, y_1 = y.chunk(2, 1)
    scales_0, scales_1 = scales.chunk(2, 1)
    means_0, means_1 = means.chunk(2, 1)
    _y_res_0_0, y_q_0_0, y_hat_0_0, scales_hat_0_0 = model.process_with_mask(y_0, scales_0, means_0, mask_0)
    _y_res_1_1, y_q_1_1, y_hat_1_1, scales_hat_1_1 = model.process_with_mask(y_1, scales_1, means_1, mask_1)
    params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
    spatial = timed_call(
        timing_records,
        example_index,
        "encoder.model",
        lambda: _onnx_run(spatial_session, _torch_to_numpy(torch, params))[0],
        image_shape=image_shape,
    )
    scales_0, means_0, scales_1, means_1 = torch.from_numpy(spatial).to(device).chunk(4, 1)
    _y_res_0_1, y_q_0_1, _y_hat_0_1, scales_hat_0_1 = model.process_with_mask(y_0, scales_0, means_0, mask_1)
    _y_res_1_0, y_q_1_0, _y_hat_1_0, scales_hat_1_0 = model.process_with_mask(y_1, scales_1, means_1, mask_0)
    y_q_w_0 = y_q_0_0 + y_q_1_1
    y_q_w_1 = y_q_0_1 + y_q_1_0
    scales_w_0 = scales_hat_0_0 + scales_hat_1_1
    scales_w_1 = scales_hat_0_1 + scales_hat_1_0
    return y_q_w_0, y_q_w_1, scales_w_0, scales_w_1


def _evc_decode_dual_prior_onnx(
    torch,
    model,
    spatial_session,
    q_step_np: np.ndarray,
    scales_np: np.ndarray,
    means_np: np.ndarray,
    device: str,
    timing_records,
    example_index: int,
    image_shape,
):
    q_step = torch.from_numpy(q_step_np.astype(np.float32, copy=False)).to(device)
    scales = torch.from_numpy(scales_np.astype(np.float32, copy=False)).to(device)
    means = torch.from_numpy(means_np.astype(np.float32, copy=False)).to(device)
    dtype = means.dtype
    _, _, height, width = means.size()
    mask_0, mask_1 = model.get_mask(int(height), int(width), dtype, device)
    quant_step = torch.clamp_min(q_step, 0.5)
    scales_0, scales_1 = scales.chunk(2, 1)
    means_0, means_1 = means.chunk(2, 1)
    scales_r_0 = scales_0 * mask_0 + scales_1 * mask_1
    entropy_start = time.perf_counter()
    y_q_r_0 = model.gaussian_encoder.decode_stream(scales_r_0, dtype, device)
    append_measurement(timing_records, example_index, "decoder.symbol_decode", time.perf_counter() - entropy_start, image_shape=image_shape)
    y_hat_0_0 = (y_q_r_0 + means_0) * mask_0
    y_hat_1_1 = (y_q_r_0 + means_1) * mask_1
    params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
    spatial = timed_call(
        timing_records,
        example_index,
        "decoder.model",
        lambda: _onnx_run(spatial_session, _torch_to_numpy(torch, params))[0],
        image_shape=image_shape,
    )
    scales_0, means_0, scales_1, means_1 = torch.from_numpy(spatial).to(device).chunk(4, 1)
    scales_r_1 = scales_0 * mask_1 + scales_1 * mask_0
    entropy_start = time.perf_counter()
    y_q_r_1 = model.gaussian_encoder.decode_stream(scales_r_1, dtype, device)
    append_measurement(timing_records, example_index, "decoder.symbol_decode", time.perf_counter() - entropy_start, image_shape=image_shape)
    y_hat_0_1 = (y_q_r_1 + means_0) * mask_1
    y_hat_1_0 = (y_q_r_1 + means_1) * mask_0
    y_hat_0 = y_hat_0_0 + y_hat_0_1
    y_hat_1 = y_hat_1_1 + y_hat_1_0
    return torch.cat((y_hat_0, y_hat_1), dim=1) * quant_step


def _evc_z_size(runner, height: int, width: int):
    with runner.imports():
        from src.utils.stream_helper import get_downsampled_shape

        return get_downsampled_shape(height, width, 64)


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


@contextmanager
def _isolated_repo_import(repo_path: str, module_roots: Iterable[str]):
    roots = tuple(module_roots)
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in roots or any(name.startswith(root + ".") for root in roots)
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name in roots or any(name.startswith(root + ".") for root in roots):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


@contextmanager
def _cuda_sync_guard(torch, device: str):
    if str(device).startswith("cuda"):
        yield
        return
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *args, **kwargs: None
    try:
        yield
    finally:
        torch.cuda.synchronize = original


@contextmanager
def _torch_cuda_allocation_guard(torch, device: str):
    if str(device).startswith("cuda"):
        yield
        return
    names = ("ones", "zeros", "empty", "full", "rand", "randn", "tensor")
    originals = {name: getattr(torch, name) for name in names if hasattr(torch, name)}

    def wrap(fn):
        def wrapped(*args, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs = dict(kwargs)
                kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapped

    for name, fn in originals.items():
        setattr(torch, name, wrap(fn))
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch, name, fn)


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
        label="upstream learned-image-codec bitstream",
        backend="python_numpy",
    )
    metadata.update(boundary_metadata)
    return canonical, metadata


def _image_to_torch(torch, image: np.ndarray, device: str):
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def _torch_to_image(torch, tensor, original_shape) -> np.ndarray:
    _count, height, width, _channels = original_shape
    cropped = tensor[:, :, :height, :width].detach().cpu().clamp(0.0, 1.0)
    return (cropped[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _pad_symmetric(torch, tensor, multiple: int):
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    left = (new_w - width) // 2
    right = new_w - width - left
    top = (new_h - height) // 2
    bottom = new_h - height - top
    padded = torch.nn.functional.pad(tensor, (left, right, top, bottom), mode="constant", value=0)
    return padded, (left, right, top, bottom)


def _crop_symmetric(torch, tensor, padding):
    left, right, top, bottom = padding
    return torch.nn.functional.pad(tensor, (-left, -right, -top, -bottom), mode="constant", value=0)


def _pad_bottom_right(torch, tensor, multiple: int):
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    padding = (0, new_w - width, 0, new_h - height)
    return torch.nn.functional.pad(tensor, padding, mode="constant", value=0), padding


def _crop_bottom_right(tensor, padding):
    _left, right, _top, bottom = padding
    height = int(tensor.shape[2]) - int(bottom)
    width = int(tensor.shape[3]) - int(right)
    return tensor[:, :, :height, :width]


def _scale_table(torch, min_value, max_value, levels):
    return torch.exp(torch.linspace(math.log(min_value), math.log(max_value), int(levels)))


def _nested_byte_count(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_nested_byte_count(item) for item in value)
    return 0


def _bytes_to_bits(payload: bytes, backend: str = "auto") -> Tuple[np.ndarray, str]:
    return dataplane.bytes_to_bits(payload, backend)


def _bits_to_bytes(bits: np.ndarray, byte_count: int, backend: str = "auto") -> Tuple[bytes, str]:
    return dataplane.bits_to_bytes(bits, byte_count, backend)


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
