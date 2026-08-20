from __future__ import annotations

import importlib.util
import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)
from noema_lab.core.reproducibility import installed_dependency_version


JsonDict = Dict[str, Any]

CSI_KIND = "channel.miso_ofdm_csi.numpy"
FEEDBACK_CODE_KIND = "channel.csi_feedback_code.numpy"
FEEDBACK_RECEIVED_KIND = "channel.csi_feedback_received.numpy"
CSI_RECONSTRUCTION_KIND = "channel.miso_ofdm_csi_reconstruction.numpy"
PRECODER_KIND = "channel.miso_ofdm_precoder.numpy"

CSI_FEEDBACK_RUNTIMES = (
    "training_interface",
    "truncated_angular_delay",
    "identity",
    "learned_artifact",
)
_SIONNA_RNG_LOCK = threading.RLock()


def _artifact_runtime_schema(entrypoint: str) -> JsonDict:
    return {
        "runtime": {
            "type": "string",
            "default": "training_interface",
            "enum": list(CSI_FEEDBACK_RUNTIMES),
            "description": (
                "Use the architecture-neutral training slot, a matched-budget classical "
                "angular-delay truncation, the full-CSI identity upper bound, or a returned artifact."
            ),
        },
        "feedback_dimension": {
            "type": "integer",
            "default": 64,
            "minimum": 1,
            "description": "Number of real-valued feedback latents per CSI realization.",
        },
        "artifact_manifest_path": {
            "type": "string",
            "default": "",
            "description": "Registered schema-v2 trained-artifact manifest for this paired CSI codec.",
            "x-noema-ui": {
                "control": "trained_artifact",
                "label": "Trained artifact",
                "accept": ".zip,.noema-artifact,.yaml,.yml,.json,application/octet-stream",
                "visible_when": {"runtime": "learned_artifact"},
                "derived_params": [
                    "runtime",
                    "feedback_dimension",
                    "artifact_manifest_path",
                    "artifact_entrypoint",
                    "artifact_package_sha256",
                ],
            },
        },
        "artifact_entrypoint": {
            "type": "string",
            "default": entrypoint,
            "x-noema-ui": {"hidden": True},
        },
        "artifact_package_sha256": {
            "type": "string",
            "default": "",
            "x-noema-ui": {"hidden": True},
        },
    }


def _feedback_codec_materializations(role: str) -> list[JsonDict]:
    """Declare the operation parameter that activates each runtime branch."""

    payload: list[JsonDict] = []
    for runner in ("benchmark_run", "dataset_capture"):
        payload.extend(
            [
                {
                    "runner": runner,
                    "backend": "numpy",
                    "implementation": "identity_full_csi_%s" % role,
                    "status": "implemented",
                    "parameter_bindings": {"runtime": "identity"},
                },
                {
                    "runner": runner,
                    "backend": "numpy",
                    "implementation": "truncated_angular_delay_%s" % role,
                    "status": "implemented",
                    "parameter_bindings": {"runtime": "truncated_angular_delay"},
                },
                {
                    "runner": runner,
                    "backend": "onnxruntime",
                    "implementation": "portable_trained_artifact_%s" % role,
                    "status": "implemented",
                    "parameter_bindings": {"runtime": "learned_artifact"},
                },
            ]
        )
    payload.append(
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "trainable_csi_feedback_%s_slot" % role,
            "status": "implemented",
            "parameter_bindings": {"runtime": "training_interface"},
        }
    )
    return payload


class MisoOfdmCsiOperation(Operation):
    """Generate reproducible single-user MISO-OFDM downlink CSI.

    Sionna is the default and never silently falls back.  The NumPy generator is
    an explicit, statistically similar development backend with the same tensor
    contract, not a claim of sample-level equivalence.
    """

    id = "wireless.miso_ofdm_csi"
    name = "Correlated MISO-OFDM CSI realization"
    output_kinds = {"csi": CSI_KIND}
    differentiability = {
        "framework": "sionna",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": (
            "CSI realizations are exogenous captured training data. Gradients begin at the "
            "feedback encoder input rather than flowing into random channel generation."
        ),
    }
    backends = {
        "benchmark_run": ["sionna", "numpy"],
        "dataset_capture": ["sionna", "numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "sionna_tdl_correlated_miso_ofdm_csi",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "sionna",
            "implementation": "sionna_tdl_correlated_miso_ofdm_csi_capture",
            "status": "implemented",
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "explicit_numpy_correlated_tapped_delay_fallback",
            "status": "implemented",
            "notes": "Selected only when wireless_backend=numpy; Sionna requests never fall back silently.",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "explicit_numpy_correlated_tapped_delay_fallback",
            "status": "implemented",
        },
    ]
    equivalence = {
        "type": "statistical",
        "reason": (
            "Backends share dimensions, average normalization, spatial correlation, delay profile, "
            "and seed semantics but do not claim identical realizations."
        ),
    }
    formats = {"artifact": "npz", "tensor": "float32 RI"}
    params_schema = object_schema(
        {
            "sample_count": {"type": "integer", "default": 64, "minimum": 1},
            "tx_antennas": {"type": "integer", "default": 8, "minimum": 2},
            "ofdm_fft_size": {"type": "integer", "default": 32, "minimum": 8},
            "num_ofdm_symbols": {"type": "integer", "default": 1, "minimum": 1},
            "tdl_model": {
                "type": "string",
                "default": "A",
                "enum": ["A", "B", "C", "D", "E"],
            },
            "channel_tap_count": {
                "type": "integer",
                "default": 6,
                "minimum": 1,
                "description": "Tapped-delay count for the explicitly selected NumPy backend.",
            },
            "subcarrier_spacing_khz": {"type": "number", "default": 30.0, "minimum": 0.1},
            "carrier_frequency_ghz": {"type": "number", "default": 3.5, "minimum": 0.1},
            "delay_spread_ns": {"type": "number", "default": 100.0, "minimum": 0.1},
            "mobility_kmh": {"type": "number", "default": 0.0, "minimum": 0.0},
            "tx_correlation_coefficient": {
                "type": "number",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 0.999,
            },
            "normalize_channel": {"type": "boolean", "default": True},
            "downlink_snr_db": {"type": "number", "default": 10.0},
            "wireless_backend": {
                "type": "string",
                "default": "sionna",
                "enum": ["sionna", "numpy"],
            },
            "seed": {"type": "integer", "default": 23, "minimum": 0},
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = {
            "sionna": _sionna_available(),
            "numpy_fallback_requires_explicit_selection": True,
        }
        return payload

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Optional[Mapping[str, str]] = None,
    ) -> None:
        if str(params.get("wireless_backend") or "sionna") != "numpy":
            return
        neutral = {
            "tdl_model": ("A", str),
            "subcarrier_spacing_khz": (30.0, float),
            "carrier_frequency_ghz": (3.5, float),
            "delay_spread_ns": (100.0, float),
            "mobility_kmh": (0.0, float),
            "num_ofdm_symbols": (1, int),
        }
        changed = []
        for name, (expected, caster) in neutral.items():
            actual = caster(params.get(name) if params.get(name) is not None else expected)
            if actual != expected:
                changed.append("%s=%r" % (name, actual))
        if changed:
            raise OperationError(
                "wireless_backend=numpy is an abstract static correlated tapped-delay "
                "generator and does not implement these physical controls: %s. "
                "Select wireless_backend=sionna or restore the NumPy neutral values."
                % ", ".join(changed)
            )

    def run(self, ctx: OperationContext) -> OperationResult:
        sample_count = int(ctx.params.get("sample_count") or 64)
        tx_antennas = int(ctx.params.get("tx_antennas") or 8)
        fft_size = int(ctx.params.get("ofdm_fft_size") or 32)
        num_ofdm_symbols = int(ctx.params.get("num_ofdm_symbols") or 1)
        rho = float(ctx.params.get("tx_correlation_coefficient", 0.7))
        backend = str(ctx.params.get("wireless_backend") or "sionna")
        seed = ctx.seed("miso_ofdm_csi", default=23)
        if backend == "sionna":
            h_freq, backend_detail = _sionna_miso_ofdm_csi(
                sample_count,
                tx_antennas,
                fft_size,
                num_ofdm_symbols,
                ctx.params,
                seed,
            )
        elif backend == "numpy":
            self.validate_preflight(ctx.params)
            h_freq, backend_detail = _numpy_miso_ofdm_csi(
                sample_count,
                tx_antennas,
                fft_size,
                ctx.params,
                seed,
            )
        else:  # Protected by schema validation, retained for direct operation use.
            raise OperationError("Unknown CSI wireless backend: %s" % backend)

        csi_ri = _complex_to_ri(h_freq)
        snr_db = float(ctx.params.get("downlink_snr_db", 10.0))
        noise_variance = float(10.0 ** (-snr_db / 10.0))
        metadata = {
            "array": "csi_ri",
            "capture_record_axis": 0,
            "capture_record_unit": "miso_ofdm_channel_realization",
            "sample_count": sample_count,
            "csi_shape": [int(item) for item in csi_ri.shape],
            "csi_layout": "N_2RI_TXANT_SUBCARRIER",
            "csi_domain": "frequency",
            "csi_representation": "perfect_downlink_complex_frequency_response_real_imag",
            "csi_observation_assumption": "perfect_ue_csi_before_feedback",
            "duplexing": "fdd",
            "user_count": 1,
            "rx_antennas": 1,
            "tx_antennas": tx_antennas,
            "ofdm_fft_size": fft_size,
            "subcarrier_count": fft_size,
            "num_ofdm_symbols": num_ofdm_symbols if backend == "sionna" else None,
            "tdl_model": str(ctx.params.get("tdl_model") or "A") if backend == "sionna" else None,
            "channel_tap_count": int(ctx.params.get("channel_tap_count") or 6),
            "subcarrier_spacing_khz": (
                float(ctx.params.get("subcarrier_spacing_khz") or 30.0)
                if backend == "sionna"
                else None
            ),
            "carrier_frequency_ghz": (
                float(ctx.params.get("carrier_frequency_ghz") or 3.5)
                if backend == "sionna"
                else None
            ),
            "delay_spread_ns": (
                float(ctx.params.get("delay_spread_ns") or 100.0)
                if backend == "sionna"
                else None
            ),
            "mobility_kmh": (
                float(
                    ctx.params.get("mobility_kmh")
                    if ctx.params.get("mobility_kmh") is not None
                    else 0.0
                )
                if backend == "sionna"
                else None
            ),
            "tx_correlation_coefficient": rho,
            "normalize_channel": bool(ctx.params.get("normalize_channel", True)),
            "downlink_snr_db": snr_db,
            "snr_db": snr_db,
            "noise_variance": noise_variance,
            "wireless_backend": backend,
            "wireless_backend_detail": backend_detail,
            "channel_model_scope": (
                "3gpp_tdl_time_frequency_channel"
                if backend == "sionna"
                else "abstract_static_correlated_exponential_tapped_delay"
            ),
            "physical_channel_controls_applied": (
                [
                    "tdl_model",
                    "subcarrier_spacing_khz",
                    "carrier_frequency_ghz",
                    "delay_spread_ns",
                    "mobility_kmh",
                    "num_ofdm_symbols",
                ]
                if backend == "sionna"
                else []
            ),
            "seed": int(seed),
            "original_complex_dimension": int(tx_antennas * fft_size),
            "original_real_dimension": int(2 * tx_antennas * fft_size),
            "channel_response_preview": _channel_response_preview(h_freq),
        }
        path = ctx.output_path("csi", ".npz")
        np.savez_compressed(
            path,
            csi_ri=csi_ri,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"csi": artifact(CSI_KIND, path, metadata)},
            metrics={
                "channel.snr_db": snr_db,
                "channel.noise_variance": noise_variance,
                "channel.tx_antennas": tx_antennas,
                "channel.ofdm_fft_size": fft_size,
                "channel.csi_sample_count": sample_count,
                "channel.gain.average": float(np.mean(np.abs(h_freq) ** 2)),
            },
            metadata=metadata,
        )


class CsiFeedbackEncoderOperation(Operation):
    id = "model.csi_feedback_encoder"
    name = "CSI feedback encoder / returned-artifact slot"
    input_kinds = {"csi": [CSI_KIND]}
    output_kinds = {"feedback_code": FEEDBACK_CODE_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "The recipe declares the CSI-to-feedback interface; external training supplies its architecture.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = _feedback_codec_materializations("encoder")
    formats = {"artifact": "npz", "tensor": "float32", "checkpoint": "onnx"}
    trained_artifact_abi = {
        "component_id": "encoder",
        "component_role": "csi_feedback_encoder",
        "entrypoint_id": "encoder",
        "required_operation_inputs": ["csi"],
        "inputs": {
            "csi_ri": {
                "dtype": "float32",
                "shape": ["batch", 2, "tx_antenna", "subcarrier"],
            }
        },
        "outputs": {
            "feedback_code": {
                "dtype": "float32",
                "shape": ["batch", "feedback_dimension"],
            }
        },
        "binding_params": {
            "runtime": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "encoder",
        },
    }
    params_schema = object_schema(_artifact_runtime_schema("encoder"))

    def run(self, ctx: OperationContext) -> OperationResult:
        csi_ri, metadata = _load_named_array(ctx.require_input("csi"), "csi_ri")
        h_freq = _ri_to_complex(csi_ri, "CSI encoder input")
        runtime = _runtime(ctx.params)
        requested_dimension = int(ctx.params.get("feedback_dimension") or 64)
        original_dimension = int(csi_ri.shape[1] * csi_ri.shape[2] * csi_ri.shape[3])
        if runtime == "training_interface":
            raise OperationError(
                "CSI feedback runtime=training_interface is an export-only typed interface. "
                "Export and train the paired CSI codec, then select its returned learned artifact."
            )
        if runtime == "identity":
            code = csi_ri.reshape(csi_ri.shape[0], -1).astype(np.float32, copy=False)
            transform = "identity_full_csi_upper_bound"
        elif runtime == "truncated_angular_delay":
            if requested_dimension > original_dimension:
                raise OperationError(
                    "feedback_dimension=%d exceeds the original real CSI dimension %d"
                    % (requested_dimension, original_dimension)
                )
            code = _encode_angular_delay(h_freq, requested_dimension)
            transform = "unitary_angular_delay_low_delay_prefix"
        elif runtime == "learned_artifact":
            code = _run_artifact(
                ctx.params,
                default_entrypoint="encoder",
                inputs={"csi_ri": csi_ri.astype(np.float32, copy=False)},
                output_name="feedback_code",
                role="CSI feedback encoder",
            )
            if code.ndim != 2 or int(code.shape[0]) != int(csi_ri.shape[0]):
                raise OperationError(
                    "CSI feedback encoder artifact output must have shape [batch, feedback_dimension]; got %s"
                    % (tuple(code.shape),)
                )
            if int(code.shape[1]) != requested_dimension:
                raise OperationError(
                    "CSI feedback encoder artifact returned dimension %d, expected feedback_dimension=%d"
                    % (int(code.shape[1]), requested_dimension)
                )
            code = code.astype(np.float32, copy=False)
            transform = "learned_artifact"
        else:  # pragma: no cover - guarded by _runtime
            raise OperationError("Unsupported CSI feedback encoder runtime: %s" % runtime)

        effective_dimension = int(code.shape[1])
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "feedback_code",
                "capture_record_axis": 0,
                "encoder_runtime": runtime,
                "feedback_transform": transform,
                "feedback_latent_normalization": (
                    "divide_angular_delay_coefficients_by_sqrt_subcarrier_count"
                    if runtime == "truncated_angular_delay"
                    else "none"
                ),
                "angular_delay_feedback_scale": (
                    float(math.sqrt(h_freq.shape[-1]))
                    if runtime == "truncated_angular_delay"
                    else 1.0
                ),
                "requested_feedback_dimension": requested_dimension,
                "feedback_dimension": effective_dimension,
                "original_csi_shape": [int(item) for item in csi_ri.shape],
                "original_real_dimension": original_dimension,
                "latent_fraction": float(effective_dimension / float(original_dimension)),
                "compression_factor": float(original_dimension / float(effective_dimension)),
                "feedback_bit_accounting": "not_yet_transported",
            }
        )
        path = ctx.output_path("feedback_code", ".npz")
        np.savez_compressed(
            path,
            feedback_code=code,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"feedback_code": artifact(FEEDBACK_CODE_KIND, path, output_metadata)},
            metrics={
                "csi_feedback.feedback_dimension": effective_dimension,
                "csi_feedback.original_real_dimension": original_dimension,
                "csi_feedback.latent_fraction": output_metadata["latent_fraction"],
                "csi_feedback.compression_factor": output_metadata["compression_factor"],
            },
            metadata=output_metadata,
        )


class CsiFeedbackLinkOperation(Operation):
    id = "channel.csi_feedback_link"
    name = "Explicit CSI feedback link"
    input_kinds = {"feedback_code": [FEEDBACK_CODE_KIND]}
    output_kinds = {"received_code": FEEDBACK_RECEIVED_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": False,
        "exportable": True,
        "reason": (
            "The ideal link is differentiable; fixed-bit quantization requires an externally chosen "
            "surrogate such as a straight-through estimator during training."
        ),
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    formats = {"artifact": "npz", "tensor": "float32"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "ideal_noiseless",
                "enum": ["ideal_noiseless", "uniform_quantized"],
            },
            "bits_per_latent": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 24,
            },
            "clip_value": {"type": "number", "default": 1.0, "minimum": 1e-12},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        code, metadata = _load_named_array(ctx.require_input("feedback_code"), "feedback_code")
        code = np.asarray(code, dtype=np.float32)
        if code.ndim != 2:
            raise OperationError("CSI feedback code must have shape [sample, feedback_dimension]")
        mode = str(ctx.params.get("mode") or "ideal_noiseless")
        metrics: JsonDict = {
            "csi_feedback.feedback_dimension": int(code.shape[1]),
        }
        if mode == "ideal_noiseless":
            received = code.copy()
            bits_per_latent = None
            bits_per_sample = None
            clip_value = None
            quantization_level_count = None
            quantization_mse = 0.0
            clipped_fraction = 0.0
            accounting = "real_valued_noiseless_latents_no_finite_bit_claim"
        elif mode == "uniform_quantized":
            bits_per_latent = int(ctx.params.get("bits_per_latent") or 8)
            clip_value = float(ctx.params.get("clip_value") or 1.0)
            quantization_level_count = 1 << bits_per_latent
            levels = quantization_level_count - 1
            clipped = np.clip(code, -clip_value, clip_value)
            indices = np.rint((clipped + clip_value) * levels / (2.0 * clip_value))
            received = ((indices * (2.0 * clip_value) / levels) - clip_value).astype(
                np.float32,
                copy=False,
            )
            bits_per_sample = int(code.shape[1]) * bits_per_latent
            quantization_mse = float(np.mean((received.astype(np.float64) - code) ** 2))
            clipped_fraction = float(np.mean(np.abs(code) > clip_value))
            accounting = "fixed_length_uniform_scalar_quantization"
            metrics.update(
                {
                    "csi_feedback.feedback_bits_per_sample": bits_per_sample,
                    "csi_feedback.quantization_mse": quantization_mse,
                    "csi_feedback.quantization_clipped_fraction": clipped_fraction,
                }
            )
        else:  # Protected by schema validation.
            raise OperationError("Unknown CSI feedback-link mode: %s" % mode)

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "received_code",
                "capture_record_axis": 0,
                "feedback_link_mode": mode,
                "feedback_dimension": int(code.shape[1]),
                "bits_per_latent": bits_per_latent,
                "clip_value": clip_value,
                "quantization_level_count": quantization_level_count,
                "feedback_bits_per_sample": bits_per_sample,
                "feedback_bit_accounting": accounting,
                "quantization_mse": quantization_mse,
                "quantization_clipped_fraction": clipped_fraction,
            }
        )
        path = ctx.output_path("received_code", ".npz")
        np.savez_compressed(
            path,
            received_code=received,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"received_code": artifact(FEEDBACK_RECEIVED_KIND, path, output_metadata)},
            metrics=metrics,
            metadata=output_metadata,
        )


class CsiFeedbackDecoderOperation(Operation):
    id = "model.csi_feedback_decoder"
    name = "CSI feedback decoder / returned-artifact slot"
    input_kinds = {"received_code": [FEEDBACK_RECEIVED_KIND, FEEDBACK_CODE_KIND]}
    output_kinds = {"reconstruction": CSI_RECONSTRUCTION_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "The recipe declares the feedback-to-CSI interface; external training supplies its architecture.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = _feedback_codec_materializations("decoder")
    formats = {"artifact": "npz", "tensor": "float32", "checkpoint": "onnx"}
    trained_artifact_abi = {
        "component_id": "decoder",
        "component_role": "csi_feedback_decoder",
        "entrypoint_id": "decoder",
        "required_operation_inputs": ["received_code"],
        "inputs": {
            "feedback_code": {
                "dtype": "float32",
                "shape": ["batch", "feedback_dimension"],
            }
        },
        "outputs": {
            "csi_hat_ri": {
                "dtype": "float32",
                "shape": ["batch", 2, "tx_antenna", "subcarrier"],
            }
        },
        "binding_params": {
            "runtime": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "decoder",
        },
    }
    params_schema = object_schema(_artifact_runtime_schema("decoder"))

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("received_code")
        array_name = "received_code" if input_artifact.kind == FEEDBACK_RECEIVED_KIND else "feedback_code"
        code, metadata = _load_named_array(input_artifact, array_name)
        code = np.asarray(code, dtype=np.float32)
        if code.ndim != 2:
            raise OperationError("CSI feedback decoder input must have shape [sample, feedback_dimension]")
        runtime = _runtime(ctx.params)
        if runtime == "training_interface":
            raise OperationError(
                "CSI feedback runtime=training_interface is an export-only typed interface. "
                "Export and train the paired CSI codec, then select its returned learned artifact."
            )
        shape = _original_csi_shape(metadata, int(code.shape[0]))
        original_dimension = int(np.prod(shape[1:]))
        requested_dimension = int(ctx.params.get("feedback_dimension") or 64)
        if runtime == "identity":
            if int(code.shape[1]) != original_dimension:
                raise OperationError(
                    "Identity CSI decoder requires %d real values, got %d"
                    % (original_dimension, int(code.shape[1]))
                )
            csi_hat_ri = code.reshape(shape).astype(np.float32, copy=False)
        elif runtime == "truncated_angular_delay":
            if int(code.shape[1]) != requested_dimension:
                raise OperationError(
                    "Truncated CSI decoder expected feedback_dimension=%d, got %d"
                    % (requested_dimension, int(code.shape[1]))
                )
            csi_hat_ri = _decode_angular_delay(code, shape)
        elif runtime == "learned_artifact":
            csi_hat_ri = _run_artifact(
                ctx.params,
                default_entrypoint="decoder",
                inputs={"feedback_code": code},
                output_name="csi_hat_ri",
                role="CSI feedback decoder",
            ).astype(np.float32, copy=False)
            if tuple(csi_hat_ri.shape) != tuple(shape):
                raise OperationError(
                    "CSI feedback decoder artifact output must have shape %s, got %s"
                    % (tuple(shape), tuple(csi_hat_ri.shape))
                )
        else:  # pragma: no cover - guarded by _runtime
            raise OperationError("Unsupported CSI feedback decoder runtime: %s" % runtime)

        _ri_to_complex(csi_hat_ri, "CSI decoder output")
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "csi_hat_ri",
                "capture_record_axis": 0,
                "decoder_runtime": runtime,
                "feedback_dimension": int(code.shape[1]),
                "original_csi_shape": [int(item) for item in shape],
                "original_real_dimension": original_dimension,
                "latent_fraction": float(code.shape[1] / float(original_dimension)),
                "compression_factor": float(original_dimension / float(code.shape[1])),
            }
        )
        path = ctx.output_path("reconstruction", ".npz")
        np.savez_compressed(
            path,
            csi_hat_ri=csi_hat_ri,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"reconstruction": artifact(CSI_RECONSTRUCTION_KIND, path, output_metadata)},
            metrics={
                "csi_feedback.feedback_dimension": int(code.shape[1]),
                "csi_feedback.original_real_dimension": original_dimension,
                "csi_feedback.latent_fraction": output_metadata["latent_fraction"],
                "csi_feedback.compression_factor": output_metadata["compression_factor"],
            },
            metadata=output_metadata,
        )


class CsiMrtPrecoderOperation(Operation):
    id = "model.csi_mrt_precoder"
    name = "MRT precoder from reconstructed CSI"
    input_kinds = {"reconstruction": [CSI_RECONSTRUCTION_KIND]}
    output_kinds = {"precoder": PRECODER_KIND}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Per-subcarrier normalized MRT is differentiable with respect to reconstructed CSI.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    formats = {"artifact": "npz", "tensor": "float32 RI"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        csi_hat_ri, metadata = _load_named_array(
            ctx.require_input("reconstruction"), "csi_hat_ri"
        )
        h_hat = _ri_to_complex(csi_hat_ri, "MRT reconstructed CSI")
        norm = np.sqrt(np.sum(np.abs(h_hat) ** 2, axis=1, keepdims=True))
        weights = np.divide(
            h_hat,
            np.maximum(norm, 1e-12),
            out=np.zeros_like(h_hat),
            where=norm > 1e-12,
        )
        zero = np.squeeze(norm, axis=1) <= 1e-12
        if np.any(zero):
            sample_indices, subcarrier_indices = np.where(zero)
            weights[sample_indices, 0, subcarrier_indices] = 1.0 + 0.0j
        weights_ri = _complex_to_ri(weights)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "weights_ri",
                "capture_record_axis": 0,
                "precoder": "per_subcarrier_mrt_from_reconstructed_csi",
                "precoder_normalization": "unit_l2_norm_per_subcarrier",
            }
        )
        path = ctx.output_path("precoder", ".npz")
        np.savez_compressed(
            path,
            weights_ri=weights_ri,
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"precoder": artifact(PRECODER_KIND, path, output_metadata)},
            metadata=output_metadata,
        )


class CsiFeedbackMetricsOperation(Operation):
    id = "metrics.csi_feedback"
    name = "CSI feedback and downlink metrics"
    input_kinds = {
        "true_csi": [CSI_KIND],
        "reconstruction": [CSI_RECONSTRUCTION_KIND],
        "precoder": [PRECODER_KIND],
    }
    output_kinds = {"report": "metrics.report"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "NMSE and reconstructed-CSI MRT achievable rate have direct differentiable tensor forms.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    formats = {"artifact": "json", "tensor": "structured metrics"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        true_ri, true_metadata = _load_named_array(ctx.require_input("true_csi"), "csi_ri")
        reconstructed_ri, reconstruction_metadata = _load_named_array(
            ctx.require_input("reconstruction"), "csi_hat_ri"
        )
        weights_ri, precoder_metadata = _load_named_array(
            ctx.require_input("precoder"), "weights_ri"
        )
        true_h = _ri_to_complex(true_ri, "true CSI")
        reconstructed_h = _ri_to_complex(reconstructed_ri, "reconstructed CSI")
        weights = _ri_to_complex(weights_ri, "CSI precoder")
        if true_h.shape != reconstructed_h.shape or true_h.shape != weights.shape:
            raise OperationError(
                "CSI metric shapes must match; true=%s reconstructed=%s precoder=%s"
                % (true_h.shape, reconstructed_h.shape, weights.shape)
            )

        axes = (1, 2)
        reference_energy = np.sum(np.abs(true_h) ** 2, axis=axes)
        error_energy = np.sum(np.abs(reconstructed_h - true_h) ** 2, axis=axes)
        per_sample_nmse = error_energy / np.maximum(reference_energy, 1e-12)
        nmse = float(np.mean(per_sample_nmse))
        nmse_db = float(10.0 * np.log10(max(nmse, 1e-15)))

        true_flat = true_h.reshape(true_h.shape[0], -1)
        reconstructed_flat = reconstructed_h.reshape(reconstructed_h.shape[0], -1)
        inner = np.abs(np.sum(np.conjugate(true_flat) * reconstructed_flat, axis=1))
        cosine_denominator = np.linalg.norm(true_flat, axis=1) * np.linalg.norm(
            reconstructed_flat, axis=1
        )
        per_sample_cosine = np.divide(
            inner,
            np.maximum(cosine_denominator, 1e-12),
            out=np.zeros_like(inner, dtype=np.float64),
            where=cosine_denominator > 1e-12,
        )
        phase_invariant_cosine = float(np.mean(per_sample_cosine))

        snr_db = float(true_metadata.get("downlink_snr_db", true_metadata.get("snr_db", 10.0)))
        snr_linear = float(10.0 ** (snr_db / 10.0))
        effective_gain = np.abs(np.sum(true_h * np.conjugate(weights), axis=1)) ** 2
        perfect_gain = np.sum(np.abs(true_h) ** 2, axis=1)
        per_subcarrier_rate = np.log2(1.0 + snr_linear * effective_gain)
        perfect_per_subcarrier_rate = np.log2(1.0 + snr_linear * perfect_gain)
        per_sample_rate = np.mean(per_subcarrier_rate, axis=1)
        per_sample_perfect_rate = np.mean(perfect_per_subcarrier_rate, axis=1)
        achieved_rate = float(np.mean(per_sample_rate))
        perfect_rate = float(np.mean(per_sample_perfect_rate))
        retention = float(achieved_rate / max(perfect_rate, 1e-12))
        loss = float(perfect_rate - achieved_rate)

        original_dimension = int(np.prod(true_ri.shape[1:]))
        feedback_dimension = int(
            reconstruction_metadata.get("feedback_dimension") or original_dimension
        )
        latent_fraction = float(feedback_dimension / float(original_dimension))
        compression_factor = float(original_dimension / float(feedback_dimension))
        metrics: JsonDict = {
            "csi_feedback.nmse": nmse,
            "csi_feedback.nmse_db": nmse_db,
            "csi_feedback.phase_invariant_cosine": phase_invariant_cosine,
            "csi_feedback.achieved_spectral_efficiency_bps_hz": achieved_rate,
            "csi_feedback.perfect_csi_spectral_efficiency_bps_hz": perfect_rate,
            "csi_feedback.spectral_efficiency_retention": retention,
            "csi_feedback.spectral_efficiency_loss_bps_hz": loss,
            "csi_feedback.feedback_dimension": feedback_dimension,
            "csi_feedback.original_real_dimension": original_dimension,
            "csi_feedback.latent_fraction": latent_fraction,
            "csi_feedback.compression_factor": compression_factor,
            "channel.snr_db": snr_db,
            "task.score": retention,
        }
        feedback_bits = reconstruction_metadata.get("feedback_bits_per_sample")
        if feedback_bits is not None:
            metrics["csi_feedback.feedback_bits_per_sample"] = int(feedback_bits)

        per_sample_nmse_db = 10.0 * np.log10(np.maximum(per_sample_nmse, 1e-15))
        preview = _csi_feedback_preview(
            true_h,
            reconstructed_h,
            per_sample_nmse_db,
            per_sample_cosine,
            per_sample_rate,
            per_sample_perfect_rate,
            reconstruction_metadata,
            feedback_dimension,
            original_dimension,
        )
        rows = [
            {
                "sample": int(index),
                "nmse": float(per_sample_nmse[index]),
                "nmse_db": float(per_sample_nmse_db[index]),
                "phase_invariant_cosine": float(per_sample_cosine[index]),
                "achieved_spectral_efficiency_bps_hz": float(per_sample_rate[index]),
                "perfect_csi_spectral_efficiency_bps_hz": float(
                    per_sample_perfect_rate[index]
                ),
            }
            for index in range(min(128, int(true_h.shape[0])))
        ]
        report_metadata = {
            "true_csi_metadata": true_metadata,
            "reconstruction_metadata": reconstruction_metadata,
            "precoder_metadata": precoder_metadata,
            "csi_feedback_preview": preview,
            "nmse_definition": "mean_sample(||H_hat-H||_F^2 / ||H||_F^2)",
            "phase_invariant_cosine_definition": (
                "mean_sample(|<H,H_hat>| / (||H||_F ||H_hat||_F))"
            ),
            "rate_definition": "mean_sample_subcarrier(log2(1 + snr_linear * |h^H w_hat|^2))",
            "perfect_rate_definition": (
                "mean_sample_subcarrier(log2(1 + snr_linear * ||h||_2^2))"
            ),
            "feedback_bit_accounting": reconstruction_metadata.get("feedback_bit_accounting"),
        }
        payload = {
            "schema_version": 1,
            "family": "csi_feedback",
            "metrics": metrics,
            "rows": rows,
            "metadata": report_metadata,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report_metadata)},
            metrics=metrics,
            metadata=report_metadata,
        )


def _runtime(params: Mapping[str, Any]) -> str:
    runtime = str(params.get("runtime") or "training_interface")
    if runtime not in CSI_FEEDBACK_RUNTIMES:
        raise OperationError(
            "CSI feedback runtime must be one of %s; got %s"
            % (", ".join(CSI_FEEDBACK_RUNTIMES), runtime)
        )
    return runtime


def _load_named_array(input_artifact, expected_name: str) -> Tuple[np.ndarray, JsonDict]:
    path = Path(input_artifact.path)
    if not path.is_file():
        raise OperationError("CSI feedback input artifact is missing: %s" % path)
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            if expected_name not in payload:
                raise OperationError(
                    "CSI feedback artifact %s is missing array `%s`" % (path, expected_name)
                )
            array = np.asarray(payload[expected_name])
            metadata = (
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="CSI feedback artifact metadata_json",
                )
                if "metadata_json" in payload
                else {}
            )
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("Could not read CSI feedback artifact %s: %s" % (path, exc)) from exc
    merged = dict(input_artifact.metadata or {})
    merged.update(metadata)
    return array, merged


def _complex_to_ri(values: np.ndarray) -> np.ndarray:
    complex_values = np.asarray(values, dtype=np.complex64)
    if complex_values.ndim != 3:
        raise OperationError(
            "MISO-OFDM CSI must have shape [sample, tx_antenna, subcarrier]; got %s"
            % (complex_values.shape,)
        )
    return np.stack([complex_values.real, complex_values.imag], axis=1).astype(
        np.float32,
        copy=False,
    )


def _ri_to_complex(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 4 or int(array.shape[1]) != 2:
        raise OperationError(
            "%s must have RI shape [sample, 2, tx_antenna, subcarrier]; got %s"
            % (label, array.shape)
        )
    return (array[:, 0] + 1j * array[:, 1]).astype(np.complex64, copy=False)


def _original_csi_shape(metadata: Mapping[str, Any], sample_count: int) -> Tuple[int, int, int, int]:
    raw = metadata.get("original_csi_shape") or metadata.get("csi_shape")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise OperationError("CSI feedback metadata is missing original_csi_shape=[N,2,Nt,Nf]")
    try:
        shape = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise OperationError("CSI original shape metadata must contain integers") from exc
    if shape[0] != sample_count or shape[1] != 2 or any(item <= 0 for item in shape):
        raise OperationError("Invalid CSI original shape metadata: %s" % (shape,))
    return shape


def _encode_angular_delay(h_freq: np.ndarray, feedback_dimension: int) -> np.ndarray:
    delay = np.fft.ifft(h_freq, axis=-1, norm="ortho")
    angular_delay = np.fft.fft(delay, axis=1, norm="ortho")
    ordered = np.stack(
        [angular_delay.real, angular_delay.imag], axis=-1
    ).transpose(0, 2, 1, 3)
    scale = math.sqrt(float(h_freq.shape[-1]))
    return (ordered.reshape(ordered.shape[0], -1)[:, :feedback_dimension] / scale).astype(
        np.float32,
        copy=False,
    )


def _decode_angular_delay(code: np.ndarray, shape: Tuple[int, int, int, int]) -> np.ndarray:
    sample_count, _, tx_antennas, subcarriers = shape
    full = np.zeros((sample_count, 2 * tx_antennas * subcarriers), dtype=np.float32)
    # This fixed shape-derived scaling is invertible and consumes no side
    # information. It keeps the classical coefficients on the same bounded
    # feedback-link scale used by tanh-bounded learned encoders.
    full[:, : code.shape[1]] = code * math.sqrt(float(subcarriers))
    ordered = full.reshape(sample_count, subcarriers, tx_antennas, 2)
    angular_delay = ordered[..., 0] + 1j * ordered[..., 1]
    angular_delay = angular_delay.transpose(0, 2, 1)
    delay = np.fft.ifft(angular_delay, axis=1, norm="ortho")
    h_freq = np.fft.fft(delay, axis=-1, norm="ortho")
    return _complex_to_ri(h_freq.astype(np.complex64, copy=False))


def _run_artifact(
    params: Mapping[str, Any],
    *,
    default_entrypoint: str,
    inputs: Mapping[str, np.ndarray],
    output_name: str,
    role: str,
) -> np.ndarray:
    from noema_lab.core.trained_artifact_runtime import run_trained_artifact_entrypoint

    manifest_value = str(params.get("artifact_manifest_path") or "").strip()
    if not manifest_value:
        raise OperationError("%s runtime=learned_artifact requires artifact_manifest_path" % role)
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_file():
        raise OperationError("%s trained-artifact manifest does not exist: %s" % (role, manifest_path))
    entrypoint = str(params.get("artifact_entrypoint") or default_entrypoint).strip()
    try:
        outputs = run_trained_artifact_entrypoint(
            manifest_path,
            entrypoint,
            inputs,
            expected_package_sha256=str(
                params.get("artifact_package_sha256") or ""
            ),
        )
    except Exception as exc:
        raise OperationError("%s trained-artifact inference failed: %s" % (role, exc)) from exc
    if output_name not in outputs:
        raise OperationError("%s artifact did not return `%s`" % (role, output_name))
    return np.asarray(outputs[output_name], dtype=np.float32)


def _spatial_correlation(tx_antennas: int, rho: float) -> np.ndarray:
    indices = np.arange(tx_antennas)
    return np.power(float(rho), np.abs(indices[:, None] - indices[None, :])).astype(
        np.complex64
    )


def _sionna_miso_ofdm_csi(
    sample_count: int,
    tx_antennas: int,
    fft_size: int,
    num_ofdm_symbols: int,
    params: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, str]:
    if not _sionna_available():
        raise OperationError(
            "wireless_backend=sionna was requested but Sionna is unavailable; "
            'install with `python -m pip install "noema-lab[wireless]"` in an '
            "installed environment, or `uv sync --extra wireless` in a source "
            "checkout, or explicitly select wireless_backend=numpy"
        )
    try:
        import torch  # type: ignore
        from sionna.phy import config as sionna_config  # type: ignore
        from sionna.phy.channel import GenerateOFDMChannel  # type: ignore
        from sionna.phy.channel.tr38901 import TDL  # type: ignore
        from sionna.phy.ofdm import ResourceGrid  # type: ignore
    except ImportError as exc:
        raise OperationError("Installed Sionna does not expose TDL OFDM channel generation") from exc

    with _SIONNA_RNG_LOCK:
        sionna_config.seed = int(seed)
        resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=float(params.get("subcarrier_spacing_khz") or 30.0) * 1e3,
        )
        speed_mps = float(
            params.get("mobility_kmh")
            if params.get("mobility_kmh") is not None
            else 0.0
        ) / 3.6
        channel_model = TDL(
            model=str(params.get("tdl_model") or "A"),
            delay_spread=float(params.get("delay_spread_ns") or 100.0) * 1e-9,
            carrier_frequency=float(params.get("carrier_frequency_ghz") or 3.5) * 1e9,
            min_speed=speed_mps,
            max_speed=speed_mps,
            num_rx_ant=1,
            num_tx_ant=tx_antennas,
            tx_corr_mat=torch.as_tensor(
                _spatial_correlation(
                    tx_antennas,
                    float(params.get("tx_correlation_coefficient", 0.7)),
                ),
                dtype=torch.complex64,
            ),
        )
        generator = GenerateOFDMChannel(
            channel_model,
            resource_grid,
            normalize_channel=bool(params.get("normalize_channel", True)),
        )
        block_count = int(math.ceil(sample_count / float(num_ofdm_symbols)))
        full = (
            generator(max(1, block_count))
            .detach()
            .cpu()
            .numpy()
            .astype(np.complex64)
        )
    # [B, rx, rx_ant, tx, tx_ant, OFDM symbol, subcarrier]
    selected = full[:, 0, 0, 0, :, :, :].transpose(0, 2, 1, 3)
    h_freq = selected.reshape(-1, tx_antennas, fft_size)[:sample_count]
    return h_freq.astype(np.complex64, copy=False), "sionna.phy.TDL+GenerateOFDMChannel"


def _numpy_miso_ofdm_csi(
    sample_count: int,
    tx_antennas: int,
    fft_size: int,
    params: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, str]:
    rng = np.random.RandomState(seed)
    tap_count = min(fft_size, int(params.get("channel_tap_count") or 6))
    pdp = np.exp(-np.arange(tap_count, dtype=np.float64))
    pdp /= np.sum(pdp)
    raw = (
        rng.normal(size=(sample_count, tx_antennas, tap_count))
        + 1j * rng.normal(size=(sample_count, tx_antennas, tap_count))
    ) / math.sqrt(2.0)
    raw *= np.sqrt(pdp)[None, None, :]
    correlation = _spatial_correlation(
        tx_antennas, float(params.get("tx_correlation_coefficient", 0.7))
    )
    cholesky = np.linalg.cholesky(correlation + 1e-7 * np.eye(tx_antennas))
    taps = np.einsum("ij,njl->nil", cholesky, raw)
    h_freq = np.fft.fft(taps, n=fft_size, axis=-1).astype(np.complex64)
    if bool(params.get("normalize_channel", True)):
        average = np.mean(np.abs(h_freq) ** 2, axis=(1, 2), keepdims=True)
        h_freq = h_freq / np.sqrt(np.maximum(average, 1e-12))
    return h_freq.astype(np.complex64, copy=False), "numpy_correlated_tapped_delay_ofdm"


def _sionna_available() -> bool:
    version = installed_dependency_version("sionna") or ""
    major = version.split(".", 1)[0]
    return (
        importlib.util.find_spec("sionna") is not None
        and importlib.util.find_spec("torch") is not None
        and major.isdigit()
        and int(major) >= 2
    )


def _channel_response_preview(h_freq: np.ndarray, limit: int = 64) -> JsonDict:
    values = np.asarray(h_freq[0], dtype=np.complex64).reshape(-1)
    indices = np.linspace(0, max(0, values.size - 1), num=min(limit, values.size)).round().astype(
        np.int64
    )
    sample = values[indices]
    return {
        "kind": "miso_ofdm_frequency_response",
        "x_axis": "antenna_subcarrier",
        "source_count": int(values.size),
        "indices": [int(item) for item in indices.tolist()],
        "magnitude": [float(item) for item in np.abs(sample).tolist()],
        "phase_rad": [float(item) for item in np.angle(sample).tolist()],
    }


def _csi_feedback_preview(
    true_h: np.ndarray,
    reconstructed_h: np.ndarray,
    nmse_db: np.ndarray,
    cosine: np.ndarray,
    achieved_rate: np.ndarray,
    perfect_rate: np.ndarray,
    metadata: Mapping[str, Any],
    feedback_dimension: int,
    original_dimension: int,
) -> JsonDict:
    antenna_count = min(16, int(true_h.shape[1]))
    subcarrier_count = min(64, int(true_h.shape[2]))
    sample_previews = []
    for sample_index in range(min(8, int(true_h.shape[0]))):
        true_magnitude = np.abs(
            true_h[sample_index, :antenna_count, :subcarrier_count]
        )
        reconstructed_magnitude = np.abs(
            reconstructed_h[sample_index, :antenna_count, :subcarrier_count]
        )
        absolute_error = np.abs(
            reconstructed_h[sample_index, :antenna_count, :subcarrier_count]
            - true_h[sample_index, :antenna_count, :subcarrier_count]
        )
        sample_previews.append(
            {
                "sample_index": sample_index,
                "true_magnitude": true_magnitude.astype(np.float32).tolist(),
                "reconstructed_magnitude": reconstructed_magnitude.astype(np.float32).tolist(),
                "absolute_error": absolute_error.astype(np.float32).tolist(),
            }
        )
    trace_count = min(128, int(true_h.shape[0]))
    return {
        "sample_index": 0,
        "axes": {"rows": "tx_antenna", "columns": "subcarrier"},
        "samples": sample_previews,
        # Retain the first sample at the top level for simple consumers while
        # exposing a bounded selector-ready sample collection above.
        "true_magnitude": sample_previews[0]["true_magnitude"],
        "reconstructed_magnitude": sample_previews[0]["reconstructed_magnitude"],
        "absolute_error": sample_previews[0]["absolute_error"],
        "per_sample_nmse_db": [float(item) for item in nmse_db[:trace_count].tolist()],
        "per_sample_phase_invariant_cosine": [
            float(item) for item in cosine[:trace_count].tolist()
        ],
        "per_sample_achieved_spectral_efficiency_bps_hz": [
            float(item) for item in achieved_rate[:trace_count].tolist()
        ],
        "per_sample_perfect_csi_spectral_efficiency_bps_hz": [
            float(item) for item in perfect_rate[:trace_count].tolist()
        ],
        "feedback_dimension": feedback_dimension,
        "original_real_dimension": original_dimension,
        "latent_fraction": float(feedback_dimension / float(original_dimension)),
        "compression_factor": float(original_dimension / float(feedback_dimension)),
        "feedback_link_mode": metadata.get("feedback_link_mode"),
        "bits_per_latent": metadata.get("bits_per_latent"),
        "clip_value": metadata.get("clip_value"),
        "quantization_level_count": metadata.get("quantization_level_count"),
        "feedback_bits_per_sample": metadata.get("feedback_bits_per_sample"),
        "feedback_bit_accounting": metadata.get("feedback_bit_accounting"),
    }
