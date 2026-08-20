from __future__ import annotations

import importlib.util
import json
import math
import re
import struct
import threading
import time
import zlib
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core import dataplane
from noema_lab.core.boundaries import (
    validate_channel_bits,
    validate_channel_symbols,
    validate_rate_accounting_point,
)
from noema_lab.core.capture_layout import (
    CaptureRecordLayout,
    CaptureRecordLayoutError,
    explicit_capture_record_layout,
    remove_capture_record_metadata,
    remove_explicit_capture_record_layout,
    set_uniform_capture_record_layout,
)
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    derive_seed,
    installed_dependency_version,
)

JsonDict = Dict[str, Any]

WIRELESS_CHANNELS = [
    "awgn",
    "flat_rayleigh",
    "interference_awgn",
    "mimo_flat",
    "ofdm_tdl",
    "ofdm_cdl",
    "urban_micro",
]
WIRELESS_BACKENDS = ["auto", "numpy", "sionna"]
NOISE_MODES = ["snr_at_unit_power", "fixed_variance"]
RECEIVER_PROCESSING_MODES = ["matched", "none"]
CHANNEL_STATE_MODES = ["none", "explicit"]


# Sionna owns process-global configured random streams. LocalExecutor can dispatch
# independent DAG branches in threads, so seeding and executing a stochastic
# Sionna block must be one critical section.  NumPy/Torch paths use local RNGs
# and never acquire this lock.
_SIONNA_RNG_LOCK = threading.RLock()


def _validated_capture_layout(
    metadata: Mapping[str, Any],
    element_count: int,
    label: str,
) -> CaptureRecordLayout | None:
    try:
        return explicit_capture_record_layout(
            metadata,
            element_count,
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc


def _rewrite_uniform_capture_layout(
    output_metadata: JsonDict,
    layout: CaptureRecordLayout | None,
    output_element_count: int,
    label: str,
) -> None:
    try:
        set_uniform_capture_record_layout(
            output_metadata,
            layout,
            output_element_count,
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc


def _sionna_rng_serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _SIONNA_RNG_LOCK:
            return function(*args, **kwargs)

    return wrapped


def _wireless_channel_materializations() -> list[JsonDict]:
    """Enumerate only channel/backend/receiver contracts we execute."""

    payload: list[JsonDict] = []

    def add(
        runner: str,
        backend: str,
        channel: str,
        receiver_processing: str,
        channel_state_mode: str,
        implementation: str,
    ) -> None:
        payload.append(
            {
                "runner": runner,
                "backend": backend,
                "implementation": implementation,
                "status": "implemented",
                "parameter_bindings": {
                    "channel": channel,
                    "receiver_processing": receiver_processing,
                    "channel_state_mode": channel_state_mode,
                },
            }
        )

    for runner in ("benchmark_run", "dataset_capture"):
        for channel in WIRELESS_CHANNELS:
            add(
                runner,
                "numpy",
                channel,
                "matched",
                "none",
                "numpy_%s_matched" % channel,
            )
        for channel in ("awgn", "interference_awgn", "flat_rayleigh"):
            add(
                runner,
                "numpy",
                channel,
                "none",
                "none",
                "numpy_%s_unprocessed" % channel,
            )
        for channel in ("awgn", "flat_rayleigh", "mimo_flat"):
            add(
                runner,
                "sionna",
                channel,
                "matched",
                "none",
                "sionna_%s_matched_artifact" % channel,
            )
        for channel in ("awgn", "flat_rayleigh"):
            add(
                runner,
                "sionna",
                channel,
                "none",
                "none",
                "sionna_%s_unprocessed_artifact" % channel,
            )
        add(
            runner,
            "sionna",
            "ofdm_tdl",
            "matched",
            "explicit",
            "sionna_explicit_tdl_ofdm_matched_artifact",
        )

    # Sionna 2.x PHY blocks are PyTorch modules and preserve the same sender
    # gradients as the native Torch materializations for these two channels.
    for channel in ("awgn", "flat_rayleigh"):
        for receiver_processing in ("matched", "none"):
            add(
                "differentiable_export",
                "torch",
                channel,
                receiver_processing,
                "none",
                "torch_%s_%s_module" % (channel, receiver_processing),
            )
            payload[-1]["parameter_bindings"]["wireless_backend"] = "auto"
            add(
                "differentiable_export",
                "sionna",
                channel,
                receiver_processing,
                "none",
                "sionna_%s_%s_pytorch_module"
                % (channel, receiver_processing),
            )
    return payload


def _backend_param(default: str = "auto") -> JsonDict:
    return {"data_plane_backend": dataplane.backend_schema(default)}


def _wireless_backend_schema() -> JsonDict:
    return {
        "type": "string",
        "default": "auto",
        "enum": WIRELESS_BACKENDS,
        "description": (
            "Wireless implementation. auto resolves deterministically to the "
            "compatible NumPy materialization for ordinary sampled channels; "
            "select sionna explicitly to require Sionna. A bound explicit "
            "Sionna OFDM channel-state artifact necessarily selects Sionna."
        ),
    }


def _digital_modem_materializations(stage: str, *, include_auto: bool) -> list[JsonDict]:
    """Declare modulation/backend combinations implemented by hard modems."""

    numpy_modulations = ["bpsk", "qpsk", "qam16"]
    if include_auto:
        numpy_modulations.insert(0, "auto")
    payload: list[JsonDict] = []
    for runner in ("benchmark_run", "dataset_capture"):
        for modulation in numpy_modulations:
            payload.append(
                {
                    "runner": runner,
                    "backend": "numpy",
                    "implementation": "numpy_%s_%s" % (modulation, stage),
                    "status": "implemented",
                    "parameter_bindings": {"modulation": modulation},
                }
            )
        for modulation in ("bpsk", "qpsk"):
            payload.append(
                {
                    "runner": runner,
                    "backend": "cpp",
                    "implementation": "cpp_%s_%s" % (modulation, stage),
                    "status": "implemented",
                    "parameter_bindings": {"modulation": modulation},
                }
            )
    return payload


def _validate_wireless_runtime_contract(
    channel: str,
    wireless_backend: str,
    receiver_processing: str,
    channel_state_mode: str,
    *,
    has_channel_state: bool,
) -> None:
    if receiver_processing not in RECEIVER_PROCESSING_MODES:
        raise OperationError(
            "Unknown wireless receiver_processing: %s" % receiver_processing
        )
    if channel_state_mode not in CHANNEL_STATE_MODES:
        raise OperationError(
            "Unknown wireless channel_state_mode: %s" % channel_state_mode
        )
    if has_channel_state != (channel_state_mode == "explicit"):
        raise OperationError(
            "wireless.channel channel_state_mode=%s requires %s channel_state input"
            % (
                channel_state_mode,
                "an explicit" if channel_state_mode == "explicit" else "no",
            )
        )
    if channel_state_mode == "explicit" and channel != "ofdm_tdl":
        raise OperationError(
            "Explicit channel_state is currently supported only for channel=ofdm_tdl"
        )
    if channel_state_mode == "explicit" and wireless_backend not in {"auto", "sionna"}:
        raise OperationError(
            "Explicit OFDM channel state requires wireless_backend=sionna (or auto)"
        )
    if receiver_processing == "none" and channel not in {
        "awgn",
        "interference_awgn",
        "flat_rayleigh",
    }:
        raise OperationError(
            "receiver_processing=none is not implemented for channel=%s" % channel
        )
    if wireless_backend == "sionna" and channel_state_mode == "none" and channel not in {
        "awgn",
        "flat_rayleigh",
        "mimo_flat",
    }:
        raise OperationError(
            "wireless_backend=sionna has no sampled materialization for channel=%s"
            % channel
        )


def _noise_control_schema() -> JsonDict:
    return {
        "noise_mode": {
            "type": "string",
            "default": "snr_at_unit_power",
            "enum": NOISE_MODES,
            "description": "Choose whether physical-channel noise is derived from unit-power SNR or set directly as variance.",
        },
        "snr_db": {
            "type": "number",
            "default": 12.0,
            "description": "Unit-power reference SNR used to derive the physical-channel noise variance.",
            "x-noema-effective-when": {"noise_mode": "snr_at_unit_power"},
            "x-noema-ui": {"visible_when": {"noise_mode": "snr_at_unit_power"}},
        },
        "noise_variance": {
            "type": "number",
            "minimum": 1e-12,
            "description": "Physical-channel noise variance used when noise mode is fixed variance.",
            "x-noema-effective-when": {"noise_mode": "fixed_variance"},
            "x-noema-ui": {"visible_when": {"noise_mode": "fixed_variance"}},
        },
    }


def _physical_channel_owned(schema: JsonDict) -> JsonDict:
    payload = dict(schema)
    payload["x-noema-ui"] = {
        "inherited_from_consumer": {"op": "wireless.channel", "input": "channel_state"}
    }
    return payload


def _resolve_noise_control(params: JsonDict) -> Tuple[str, float, float]:
    mode = str(params.get("noise_mode") or "snr_at_unit_power")
    if mode not in NOISE_MODES:
        raise OperationError("noise_mode must be one of: %s" % ", ".join(NOISE_MODES))
    if mode == "fixed_variance":
        raw = params.get("noise_variance")
        if raw is None:
            raise OperationError("noise_mode=fixed_variance requires noise_variance")
        noise_variance = float(raw)
        if not math.isfinite(noise_variance) or noise_variance <= 0.0:
            raise OperationError("noise_variance must be a finite value greater than zero")
        reference_snr_db = -10.0 * math.log10(noise_variance)
        return mode, noise_variance, reference_snr_db
    reference_snr_db = float(params.get("snr_db", 12.0))
    return mode, _noise_variance_from_snr(reference_snr_db), reference_snr_db


def _snr_db_from_signal_and_noise(signal_power: float, noise_variance: float) -> float:
    return 10.0 * math.log10(max(float(signal_power), 1e-12) / max(float(noise_variance), 1e-12))


class IndicesToBitsOperation(Operation):
    id = "channel.indices_to_bits"
    name = "Pack semantic indices into bits"
    input_kinds = {"indices": ["semantic.indices.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(_backend_param())

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("indices")
        indices, metadata = _load_indices(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(indices.size),
            "Semantic-index input to %s" % ctx.step_id,
        )
        codebook_size = int(metadata.get("codebook_size") or max(int(indices.max()) + 1, 2))
        bits_per_index = int(math.ceil(math.log(max(codebook_size, 2), 2)))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = dataplane.indices_to_bits(indices, bits_per_index, backend)
        bit_metadata = dict(metadata)
        bit_metadata.update(
            {
                "codebook_size": codebook_size,
                "bits_per_index": bits_per_index,
                "indices_shape": list(indices.shape),
                "bit_count": int(bits.size),
                "bit_role": "payload",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "payload_coder": "fixed_index_bitpack",
                "payload_coder_type": "fixed_width_bit_packing",
                "payload_coder_label": "Fixed index bit-packing",
                "payload_coder_lossless": True,
                "entropy_coded": False,
                "data_plane_backend": selected_backend,
            }
        )
        _rewrite_uniform_capture_layout(
            bit_metadata,
            capture_layout,
            int(bits.size),
            "Semantic index bit-packing at %s" % ctx.step_id,
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(bit_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, bit_metadata)},
            metrics={"channel.payload_bit_count": int(bits.size)},
            metadata={
                "bit_count": int(bits.size),
                "bits_per_index": bits_per_index,
                "payload_coder": "fixed_index_bitpack",
                "payload_coder_type": "fixed_width_bit_packing",
                "payload_coder_label": "Fixed index bit-packing",
                "payload_coder_lossless": True,
                "entropy_coded": False,
                "data_plane_backend": selected_backend,
            },
        )


class BitsToIndicesOperation(Operation):
    id = "channel.bits_to_indices"
    name = "Unpack bits into semantic indices"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"indices": "semantic.indices.numpy"}
    params_schema = object_schema(
        {
            "invalid_policy": {
                "type": "string",
                "default": "mod",
                "enum": ["mod", "clamp"],
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "Packed-index input to %s" % ctx.step_id,
        )
        policy = str(ctx.params.get("invalid_policy", "mod"))
        bits_per_index = int(metadata["bits_per_index"])
        shape = tuple(int(item) for item in metadata["indices_shape"])
        codebook_size = int(metadata["codebook_size"])
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        indices, invalid_fraction, selected_backend = dataplane.bits_to_indices(bits, bits_per_index, shape, codebook_size, policy, backend)
        index_metadata = dict(metadata)
        index_metadata["invalid_index_fraction"] = invalid_fraction
        index_metadata["data_plane_backend"] = selected_backend
        _rewrite_uniform_capture_layout(
            index_metadata,
            capture_layout,
            int(indices.size),
            "Semantic index unpacking at %s" % ctx.step_id,
        )
        path = ctx.output_path("indices", ".npz")
        np.savez_compressed(path, indices=indices, metadata_json=json.dumps(index_metadata))
        return OperationResult(
            outputs={"indices": artifact("semantic.indices.numpy", path, index_metadata)},
            metrics={"channel.invalid_index_fraction": invalid_fraction},
            metadata={"invalid_index_fraction": invalid_fraction, "data_plane_backend": selected_backend},
        )


class WirelessDigitalLinkOperation(Operation):
    id = "wireless.digital_link"
    name = "Digital modulation over a simulated wireless link"
    input_kinds = {
        "bits": ["channel.payload_bits.numpy", "channel.coded_bits.numpy", "channel.bits.numpy"]
    }
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    backends = {
        "benchmark_run": ["numpy", "sionna"],
        "dataset_capture": ["numpy", "sionna"],
        "differentiable_export": [],
    }
    params_schema = object_schema(
        {
            "modulation": {
                "type": "string",
                "default": "qpsk",
                "enum": ["bpsk", "qpsk", "qam16"],
            },
            "channel": {
                "type": "string",
                "default": "awgn",
                "enum": WIRELESS_CHANNELS,
            },
            **_noise_control_schema(),
            "wireless_backend": _wireless_backend_schema(),
            "receiver_processing": {
                "type": "string",
                "default": "matched",
                "enum": RECEIVER_PROCESSING_MODES,
            },
            "tx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "rx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "ofdm_fft_size": {"type": "integer", "default": 64, "minimum": 8},
            "num_ofdm_symbols": {"type": "integer", "default": 14, "minimum": 1},
            "subcarrier_spacing_khz": {"type": "number", "default": 15.0, "minimum": 0.1},
            "carrier_frequency_ghz": {"type": "number", "default": 3.5, "minimum": 0.1},
            "mobility_kmh": {"type": "number", "default": 3.0, "minimum": 0.0},
            "interferers": {"type": "integer", "default": 0, "minimum": 0},
            "interference_sir_db": {"type": "number", "default": 18.0},
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": "Optional explicit operation seed. Omit it to derive the stream from the recipe master seed.",
            },
            **_backend_param(),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _sionna_availability(optional=True)
        return payload

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        _resolve_noise_control(dict(params))

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        requested = str(params.get("wireless_backend") or "auto").strip().lower()
        sionna = _sionna_availability(optional=True)
        if sionna.get("available") is False and not sionna.get("optional"):
            return sionna
        if requested == "sionna":
            return sionna
        return {
            "available": True,
            "optional": False,
            "backend": "numpy",
            "missing": [],
        }

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        modulation = str(ctx.params.get("modulation", "qpsk"))
        channel = str(ctx.params.get("channel", "awgn"))
        receiver_processing = str(
            ctx.params.get("receiver_processing") or "matched"
        )
        noise_mode, noise_var, reference_snr_db = _resolve_noise_control(ctx.params)
        effective_snr_db = _snr_db_from_signal_and_noise(1.0, noise_var)
        seed = ctx.seed("wireless_digital_link")
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        _validate_wireless_runtime_contract(
            channel,
            str(ctx.params.get("wireless_backend") or "auto"),
            receiver_processing,
            "none",
            has_channel_state=False,
        )
        rx_bits, link_metrics = _transmit_bits(bits, modulation, channel, reference_snr_db, seed, backend, ctx.params)
        if int(rx_bits.size) != int(bits.size):
            raise OperationError(
                "wireless.digital_link changed bit cardinality from %d to %d"
                % (int(bits.size), int(rx_bits.size))
            )
        output_metadata = dict(metadata)
        output_metadata.setdefault("wireless_history", [])
        output_metadata["wireless_history"] = list(output_metadata["wireless_history"]) + [
            {
                "modulation": modulation,
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "noise_variance": noise_var,
                "seed": seed,
                "ber": link_metrics["ber"],
                "wireless_backend": link_metrics["wireless_backend"],
                "backend_detail": link_metrics["backend_detail"],
                "requested_wireless_backend": link_metrics[
                    "requested_wireless_backend"
                ],
                "data_plane_backend": link_metrics["data_plane_backend"],
            }
        ]
        output_metadata.update(
            {
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "noise_variance": noise_var,
                "channel_response_preview": link_metrics.get("channel_response_preview"),
                "wireless_backend": link_metrics["wireless_backend"],
                "wireless_backend_detail": link_metrics["backend_detail"],
                "requested_wireless_backend": link_metrics[
                    "requested_wireless_backend"
                ],
                "wireless_preset": link_metrics.get("preset", channel),
                "tx_antennas": link_metrics.get("tx_antennas"),
                "rx_antennas": link_metrics.get("rx_antennas"),
                "ofdm_fft_size": link_metrics.get("ofdm_fft_size"),
                "num_ofdm_symbols": link_metrics.get("num_ofdm_symbols"),
                "interferers": link_metrics.get("interferers"),
                "interference_sir_db": link_metrics.get("interference_sir_db"),
                "carrier_frequency_ghz": link_metrics.get("carrier_frequency_ghz"),
                "mobility_kmh": link_metrics.get("mobility_kmh"),
                "channel_use_count": link_metrics["channel_use_count"],
                "payload_symbol_count": link_metrics["symbol_count"],
                "grid_padding_symbol_count": link_metrics[
                    "grid_padding_symbol_count"
                ],
                "receiver_processing": receiver_processing,
                "data_plane_backend": link_metrics["data_plane_backend"],
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=rx_bits, metadata_json=json.dumps(output_metadata))
        metrics = {
            "channel.ber": link_metrics["ber"],
            "channel.snr_db": reference_snr_db,
            "channel.reference_snr_db": reference_snr_db,
            "channel.effective_snr_db": effective_snr_db,
            "channel.noise_variance": noise_var,
            "channel.symbol_count": link_metrics["symbol_count"],
            "channel.channel_use_count": link_metrics["channel_use_count"],
            "channel.payload_symbol_count": link_metrics["symbol_count"],
            "channel.grid_padding_symbol_count": link_metrics[
                "grid_padding_symbol_count"
            ],
            "channel.backend.%s"
            % _metric_label(str(link_metrics["wireless_backend"])): 1,
        }
        for key in [
            "tx_antennas",
            "rx_antennas",
            "ofdm_fft_size",
            "num_ofdm_symbols",
            "interferers",
            "interference_sir_db",
            "carrier_frequency_ghz",
            "mobility_kmh",
            "multipath_taps",
        ]:
            value = link_metrics.get(key)
            if value is not None:
                metrics["channel.%s" % key] = value
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics=metrics,
            metadata={
                "modulation": modulation,
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "noise_variance": noise_var,
                "wireless_backend": link_metrics["wireless_backend"],
                "backend_detail": link_metrics["backend_detail"],
                "requested_wireless_backend": link_metrics[
                    "requested_wireless_backend"
                ],
                "data_plane_backend": link_metrics["data_plane_backend"],
            },
        )


class BitBoundaryCheckpointOperation(Operation):
    id = "channel.bit_boundary"
    name = "Canonical bit boundary checkpoint"
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "This fixed-point check validates discrete uint8 bits; it is exact for benchmarking but stops gradients.",
    }
    backends = {"benchmark_run": ["numpy", "cpp"], "dataset_capture": ["numpy", "cpp"], "differentiable_export": []}
    equivalence = {"type": "exact", "reason": "Canonical bit boundaries must preserve unpacked uint8 bit values exactly."}
    formats = {"artifact": "npz", "tensor": "none"}
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"bits": "channel.bits.numpy"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "boundary"},
            "role": {"type": "string", "default": "channel_boundary"},
            "expected_bit_count": {"type": "integer", "default": 0, "minimum": 0},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(input_artifact.path, input_artifact.metadata)
        label = _metric_label(str(ctx.params.get("label") or "boundary"))
        role = str(ctx.params.get("role") or "channel_boundary")
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = _require_canonical_bits(raw_bits, "%s/%s" % (ctx.step_id, label), backend)
        expected = int(ctx.params.get("expected_bit_count") or 0)
        if expected > 0 and int(bits.size) != expected:
            raise OperationError(
                "Bit boundary %s expected %d bits but received %d bits"
                % (label, expected, int(bits.size))
            )

        output_metadata = dict(metadata)
        output_metadata.pop("source_item_payload_byte_counts", None)
        output_metadata.update(
            {
                "bit_count": int(bits.size),
                "boundary_contract": "channel.bits",
                "bit_role": role,
                "fixed_point": label,
                "fixed_point_label": label,
                "fixed_point_bit_count": int(bits.size),
                "fixed_point_byte_count": int(math.ceil(float(bits.size) / 8.0)),
                "fixed_point_dtype": "uint8",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        output_metadata.setdefault("fixed_points", [])
        output_metadata["fixed_points"] = list(output_metadata["fixed_points"]) + [
            {
                "label": label,
                "role": role,
                "bit_count": int(bits.size),
                "byte_count": int(math.ceil(float(bits.size) / 8.0)),
                "dtype": "uint8",
                "storage": "unpacked_uint8",
                "data_plane_backend": selected_backend,
            }
        ]

        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(output_metadata))
        metrics = _fixed_bit_metrics(label, int(bits.size))
        metrics.update(
            {
                "channel.fixed.bit_count": int(bits.size),
                "channel.fixed.byte_count": int(math.ceil(float(bits.size) / 8.0)),
            }
        )
        return OperationResult(
            outputs={"bits": artifact("channel.bits.numpy", path, output_metadata)},
            metrics=metrics,
            metadata={
                "label": label,
                "bit_count": int(bits.size),
                "byte_count": int(math.ceil(float(bits.size) / 8.0)),
                "dtype": "uint8",
                "data_plane_backend": selected_backend,
            },
        )


class PayloadPassthroughEncoderOperation(Operation):
    id = "channel.payload_passthrough_encoder"
    name = "Payload bitstream pass-through encoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        start = time.perf_counter()
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        output_metadata = _payload_passthrough_metadata(metadata, "payload_encoder", int(bits.size))
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits.astype(np.uint8, copy=False), metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.payload_bit_count": int(bits.size)},
            metadata={
                "bit_count": int(bits.size),
                "payload_passthrough": True,
                "payload_coder": output_metadata["payload_coder"],
                "payload_coder_type": output_metadata["payload_coder_type"],
                "payload_coder_label": output_metadata["payload_coder_label"],
                "payload_coder_lossless": output_metadata["payload_coder_lossless"],
                "codec_timing": _passthrough_timing("encoder", time.perf_counter() - start),
            },
        )


class PayloadPassthroughDecoderOperation(Operation):
    id = "channel.payload_passthrough_decoder"
    name = "Payload bitstream pass-through decoder"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        start = time.perf_counter()
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        output_metadata = _payload_passthrough_metadata(metadata, "payload_decoder", int(bits.size))
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits.astype(np.uint8, copy=False), metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.payload_bit_count": int(bits.size)},
            metadata={
                "bit_count": int(bits.size),
                "payload_passthrough": True,
                "payload_coder": output_metadata["payload_coder"],
                "payload_coder_type": output_metadata["payload_coder_type"],
                "payload_coder_label": output_metadata["payload_coder_label"],
                "payload_coder_lossless": output_metadata["payload_coder_lossless"],
                "codec_timing": _passthrough_timing("decoder", time.perf_counter() - start),
            },
        )


class SymbolBoundaryCheckpointOperation(Operation):
    id = "channel.symbol_boundary"
    name = "Canonical complex-symbol boundary checkpoint"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Differentiable export treats this fixed-point check as a typed identity for continuous symbols; benchmark execution validates and serializes complex64 symbols.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    equivalence = {"type": "numerical", "tolerance": {"atol": 0.0, "rtol": 0.0}, "reason": "Symbol boundaries are identity checks; numeric implementations must preserve complex symbol values."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    input_kinds = {
        "symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "symbol_boundary"},
            "role": {"type": "string", "default": "channel_symbol_boundary"},
            "expected_symbol_count": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        label = _metric_label(str(ctx.params.get("label") or "symbol_boundary"))
        role = str(ctx.params.get("role") or "channel_symbol_boundary")
        symbols = _require_canonical_symbols(raw_symbols, "%s/%s" % (ctx.step_id, label))
        expected = int(ctx.params.get("expected_symbol_count") or 0)
        if expected > 0 and int(symbols.size) != expected:
            raise OperationError(
                "Symbol boundary %s expected %d symbols but received %d symbols"
                % (label, expected, int(symbols.size))
            )
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "symbol_count": int(symbols.size),
                "boundary_contract": "channel.symbols",
                "channel_use_count": int(symbols.size),
                "symbol_role": role,
                "fixed_point": label,
                "fixed_point_label": label,
                "fixed_point_symbol_count": int(symbols.size),
                "fixed_point_dtype": "complex64",
                "symbol_storage": "complex64",
            }
        )
        output_metadata.setdefault("fixed_points", [])
        output_metadata["fixed_points"] = list(output_metadata["fixed_points"]) + [
            {
                "label": label,
                "role": role,
                "symbol_count": int(symbols.size),
                "dtype": "complex64",
                "storage": "complex64",
            }
        ]
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        metrics = _fixed_symbol_metrics(label, int(symbols.size))
        metrics.update({"channel.fixed.symbol_count": int(symbols.size), "channel.channel_use_count": int(symbols.size)})
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics=metrics,
            metadata={"label": label, "symbol_count": int(symbols.size), "dtype": "complex64"},
        )


def _symbol_power_trace_metadata(symbols: np.ndarray, label: str, unit: str = "normalized", max_points: int = 128) -> Tuple[JsonDict, float, float]:
    flat = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    power = (np.abs(flat) ** 2).astype(np.float64, copy=False)
    total = float(np.sum(power)) if int(power.size) else 0.0
    average = float(np.mean(power)) if int(power.size) else 0.0
    max_points = max(1, int(max_points))
    if int(power.size):
        stride = max(1, int(math.ceil(float(power.size) / float(max_points))))
        indices = np.arange(0, int(power.size), stride, dtype=np.int64)[:max_points]
        values = power[indices]
    else:
        indices = np.zeros((0,), dtype=np.int64)
        values = np.zeros((0,), dtype=np.float64)
    preview = {
        "kind": "tx_power_trace",
        "label": label,
        "x_axis": "symbol",
        "unit": unit,
        "indices": [int(item) for item in indices.tolist()],
        "values": [float(item) for item in values.tolist()],
        "source_count": int(power.size),
        "average_power": average,
        "total_energy": total,
    }
    return preview, average, total


class SymbolPowerIdentityOperation(Operation):
    id = "channel.symbol_power_identity"
    name = "Transmit power disabled pass-through"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Disabled TX power is a continuous-symbol identity block; it preserves the PHY skeleton and does not change transmit power.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "numpy_symbol_identity", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "numpy_symbol_identity", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_identity", "status": "implemented"},
    ]
    equivalence = {
        "type": "exact",
        "reason": "The disabled TX power block must pass canonical complex symbols through unchanged.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    input_kinds = {
        "symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "tx_power_off"},
            "power_unit": {"type": "string", "default": "normalized", "enum": ["normalized", "mW"]},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        symbols = _require_canonical_symbols(raw_symbols, ctx.step_id).astype(np.complex64, copy=False)
        label = _metric_label(str(ctx.params.get("label") or "tx_power_off"))
        unit = str(ctx.params.get("power_unit") or "normalized")
        power_preview, power_after, total_energy = _symbol_power_trace_metadata(symbols, label, unit=unit)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "symbol_count": int(symbols.size),
                "boundary_contract": "channel.symbols",
                "channel_use_count": int(symbols.size),
                "symbol_storage": "complex64",
                "tx_power_enabled": False,
                "tx_power_mode": "off",
                "power_before": power_after,
                "power_after": power_after,
                "power_total_energy": total_energy,
                "power_unit": unit,
                "tx_power_preview": power_preview,
            }
        )
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics={
                "channel.tx_power.%s.before" % label: power_after,
                "channel.tx_power.%s.after" % label: power_after,
                "channel.tx_power.%s.total_energy" % label: total_energy,
                "channel.tx_power.before": power_after,
                "channel.tx_power.after": power_after,
                "channel.tx_power.average": power_after,
                "channel.tx_power.total_energy": total_energy,
                "channel.symbol_count": int(symbols.size),
                "channel.channel_use_count": int(symbols.size),
            },
            metadata={
                "label": label,
                "mode": "off",
                "symbol_count": int(symbols.size),
                "power_after": power_after,
                "total_energy": total_energy,
                "power_unit": unit,
            },
        )


class SymbolPowerNormalizeOperation(Operation):
    id = "channel.symbol_power_normalize"
    name = "Transmit symbol power normalization"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Continuous-symbol normalization is a differentiable scale operation that fixes average transmit power before the physical channel.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {
            "runner": runner,
            "backend": backend,
            "implementation": "%s_%s_power_normalizer" % (backend, scope),
            "status": "implemented",
            "parameter_bindings": {"normalization_scope": scope},
        }
        for runner, backend in (
            ("benchmark_run", "numpy"),
            ("dataset_capture", "numpy"),
            ("differentiable_export", "torch"),
        )
        for scope in ("source_item", "global")
    ]
    equivalence = {
        "type": "numerical",
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "reason": "NumPy and Torch materializations should produce the same average-power-normalized symbols within floating-point tolerance.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    input_kinds = {
        "symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    params_schema = object_schema(
        {
            "target_power": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "description": "Normalized average complex-symbol power target applied by this normalizer; this is not a value in watts unless a physical link-budget conversion is defined.",
                "x-noema-ui": {"label": "Normalized average TX symbol power"},
            },
            "eps": {"type": "number", "default": 1e-12, "minimum": 0.0},
            "normalization_scope": {
                "type": "string",
                "default": "source_item",
                "enum": ["source_item", "global"],
                "description": "Normalize each declared source item independently, or normalize the complete tensor/stream as one global batch.",
            },
            "label": {"type": "string", "default": "tx_power_normalize"},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        symbols = _require_canonical_symbols(raw_symbols, ctx.step_id)
        label = _metric_label(str(ctx.params.get("label") or "tx_power_normalize"))
        target_power = max(0.0, float(ctx.params.get("target_power", 1.0)))
        eps = max(0.0, float(ctx.params.get("eps", 1e-12)))
        normalization_scope = str(
            ctx.params.get("normalization_scope") or "source_item"
        )
        item_partition = _source_item_symbol_partition(metadata, int(symbols.size))
        if normalization_scope == "global":
            item_partition = None
        source_item_power_before: List[float] = []
        source_item_power_after: List[float] = []
        source_item_power_scales: List[float] = []
        if int(symbols.size) == 0:
            power_before = 0.0
            scale = 1.0
            normalized = symbols.astype(np.complex64, copy=False)
        elif item_partition is not None:
            power_before = float(
                np.mean(np.abs(symbols.astype(np.complex64, copy=False)) ** 2)
            )
            normalized_rows: List[np.ndarray] = []
            offset = 0
            for item_count in item_partition:
                row = symbols[offset : offset + item_count].astype(
                    np.complex64, copy=False
                )
                row_power = float(np.mean(np.abs(row) ** 2))
                if target_power == 0.0:
                    row_scale = 0.0
                    normalized_row = np.zeros_like(row, dtype=np.complex64)
                elif row_power > eps:
                    row_scale = float(math.sqrt(target_power / row_power))
                    normalized_row = (
                        row * np.complex64(row_scale)
                    ).astype(np.complex64, copy=False)
                else:
                    row_scale = 1.0
                    normalized_row = row
                source_item_power_before.append(row_power)
                source_item_power_after.append(
                    float(np.mean(np.abs(normalized_row) ** 2))
                )
                source_item_power_scales.append(row_scale)
                normalized_rows.append(normalized_row)
                offset += item_count
            normalized = np.concatenate(normalized_rows).astype(
                np.complex64, copy=False
            )
            scale = None
        else:
            power_before = float(np.mean(np.abs(symbols.astype(np.complex64, copy=False)) ** 2))
            if target_power == 0.0:
                scale = 0.0
                normalized = np.zeros_like(symbols, dtype=np.complex64)
            elif power_before > eps:
                scale = float(math.sqrt(target_power / power_before))
                normalized = (symbols.astype(np.complex64, copy=False) * np.complex64(scale)).astype(np.complex64, copy=False)
            else:
                scale = 1.0
                normalized = symbols.astype(np.complex64, copy=False)
        power_preview, power_after, total_energy = _symbol_power_trace_metadata(normalized, label)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "symbol_count": int(normalized.size),
                "boundary_contract": "channel.symbols",
                "channel_use_count": int(normalized.size),
                "symbol_storage": "complex64",
                "power_normalized": True,
                "power_normalization_label": label,
                "power_normalization_target": target_power,
                "power_budget_unit": "average_symbol_power",
                "power_before": power_before,
                "power_after": power_after,
                "power_total_energy": total_energy,
                "power_unit": "normalized",
                "tx_power_preview": power_preview,
                "power_normalization_scale": scale,
            }
        )
        if normalization_scope == "source_item":
            output_metadata.update(
                {
                    "power_normalization_scope": "source_item",
                    "source_item_power_before": source_item_power_before,
                    "source_item_power_after": source_item_power_after,
                    "source_item_power_normalization_scales": source_item_power_scales,
                    "max_source_item_tx_power_after": max(
                        source_item_power_after, default=0.0
                    ),
                }
            )
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=normalized, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics={
                "channel.tx_power.%s.before" % label: power_before,
                "channel.tx_power.%s.after" % label: power_after,
                "channel.tx_power.%s.target" % label: target_power,
                "channel.tx_power.%s.total_energy" % label: total_energy,
                "channel.tx_power.before": power_before,
                "channel.tx_power.after": power_after,
                "channel.tx_power.average": power_after,
                "channel.tx_power.target": target_power,
                "channel.tx_power.total_energy": total_energy,
                "channel.symbol_count": int(normalized.size),
                "channel.channel_use_count": int(normalized.size),
                **(
                    {
                        "channel.max_source_item_tx_power.after": max(
                            source_item_power_after, default=0.0
                        ),
                        "channel.power_normalization.source_item_scope": 1,
                    }
                    if normalization_scope == "source_item"
                    else {}
                ),
            },
            metadata={
                "label": label,
                "symbol_count": int(normalized.size),
                "target_power": target_power,
                "power_before": power_before,
                "power_after": power_after,
                "total_energy": total_energy,
                "power_unit": "normalized",
                "scale": scale,
                "power_normalization_scope": (
                    normalization_scope
                ),
            },
        )



class OfdmChannelStateOperation(Operation):
    id = "wireless.ofdm_channel_state"
    name = "Sionna OFDM channel-state realization"
    input_kinds = {"symbols": ["channel.symbols.complex_numpy"]}
    output_kinds = {
        "symbols": "channel.symbols.complex_numpy",
        "state": "channel.ofdm_channel_state.numpy",
    }
    differentiability = {
        "framework": "sionna",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "The artifact runner freezes a reproducible Sionna TDL realization so the allocator and channel consume identical CSI. No differentiable materialization is registered.",
    }
    backends = {
        "benchmark_run": ["sionna"],
        "dataset_capture": ["sionna"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "sionna_3gpp_tdl_ofdm_frequency_response",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "sionna",
            "implementation": "sionna_3gpp_tdl_ofdm_frequency_response_capture",
            "status": "implemented",
        },
    ]
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "tdl_model": _physical_channel_owned({"type": "string", "default": "A", "enum": ["A", "B", "C", "D", "E"]}),
            "ofdm_fft_size": _physical_channel_owned({"type": "integer", "default": 16, "minimum": 8}),
            "num_ofdm_symbols": _physical_channel_owned({"type": "integer", "default": 4, "minimum": 1}),
            "capacity_multiplier": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 16,
                "description": "Reserve additional OFDM channel-state capacity for allocation-aware variable-rate transport.",
            },
            "subcarrier_spacing_khz": _physical_channel_owned({"type": "number", "default": 15.0, "minimum": 0.1}),
            "carrier_frequency_ghz": _physical_channel_owned({"type": "number", "default": 3.5, "minimum": 0.1}),
            "delay_spread_ns": _physical_channel_owned({"type": "number", "default": 100.0, "minimum": 0.1}),
            "mobility_kmh": _physical_channel_owned({"type": "number", "default": 3.0, "minimum": 0.0}),
            "noise_variance": _physical_channel_owned({
                "type": "number",
                "default": 0.1,
                "minimum": 1e-12,
                "description": "Noise reference attached to this CSI realization; the downstream physical channel must use the same variance.",
            }),
            "average_power_budget": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "description": "Authoritative normalized average transmit-power budget per OFDM subcarrier for this channel-state scenario.",
            },
            "normalize_channel": _physical_channel_owned({"type": "boolean", "default": True}),
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": "Optional explicit operation seed. Omit it to derive the stream from the recipe master seed.",
            },
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _sionna_availability(optional=False)
        return payload

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        symbols, metadata = _load_symbols(input_artifact.path, input_artifact.metadata)
        fft_size = max(8, int(ctx.params.get("ofdm_fft_size") or 16))
        num_ofdm_symbols = max(1, int(ctx.params.get("num_ofdm_symbols") or 4))
        capacity_multiplier = max(1, min(16, int(ctx.params.get("capacity_multiplier") or 1)))
        grid_size = fft_size * num_ofdm_symbols
        block_count = max(
            1,
            int(math.ceil(float(max(int(symbols.size), 1) * capacity_multiplier) / float(grid_size))),
        )
        seed = ctx.seed("ofdm_channel_state")
        state = _generate_sionna_ofdm_channel_state(block_count, ctx.params, seed)
        h_freq = state["h_freq"]
        gains = np.maximum(np.abs(h_freq) ** 2, 1e-12).astype(np.float32)
        noise_variance = float(
            ctx.params.get("noise_variance")
            if ctx.params.get("noise_variance") is not None
            else 0.1
        )
        if not math.isfinite(noise_variance) or noise_variance <= 0.0:
            raise OperationError(
                "wireless.ofdm_channel_state noise_variance must be finite and positive"
            )
        average_power_budget = float(
            ctx.params.get("average_power_budget")
            if ctx.params.get("average_power_budget") is not None
            else 1.0
        )
        if not math.isfinite(average_power_budget) or average_power_budget < 0.0:
            raise OperationError(
                "wireless.ofdm_channel_state average_power_budget must be finite and nonnegative"
            )
        total_power_budget = average_power_budget * float(fft_size)
        reference_snr_db = _snr_db_from_signal_and_noise(1.0, noise_variance)
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "channel_state_available": True,
                "channel_state_delivery": "explicit_graph_artifact",
                "channel_state_kind": "sionna_3gpp_tdl_ofdm",
                "wireless_backend": "sionna",
                "wireless_backend_detail": "sionna.phy.channel.tr38901.TDL+GenerateOFDMChannel",
                "tdl_model": state["tdl_model"],
                "ofdm_fft_size": fft_size,
                "num_ofdm_symbols": num_ofdm_symbols,
                "ofdm_block_count": block_count,
                "ofdm_grid_size": grid_size,
                "ofdm_capacity_multiplier": capacity_multiplier,
                "ofdm_resource_element_capacity": block_count * grid_size,
                "subcarrier_spacing_khz": state["subcarrier_spacing_khz"],
                "carrier_frequency_ghz": state["carrier_frequency_ghz"],
                "delay_spread_ns": state["delay_spread_ns"],
                "mobility_kmh": state["mobility_kmh"],
                "channel_gain_definition": "abs_sionna_tdl_frequency_response_squared",
                "allocation_axis": "subcarrier",
                "snapshot_axis": "ofdm_symbol_channel_state",
                "allocation_granularity": "per_subcarrier_per_ofdm_symbol_state",
                "noise_variance": noise_variance,
                "average_power_budget": average_power_budget,
                "total_power_budget": total_power_budget,
                "reference_snr_db": reference_snr_db,
                "snr_db": reference_snr_db,
                "channel_state_seed": int(seed),
                "csi_representation": "perfect_complex_frequency_response",
                "transmitter_csi_assumption": "perfect_instantaneous",
                "channel_response_preview": _response_preview_from_values(
                    h_freq.reshape(-1), "subcarrier", "sionna_tdl_ofdm_frequency_response"
                ),
            }
        )
        symbols_path = ctx.output_path("symbols", ".npz")
        state_path = ctx.output_path("state", ".npz")
        state_metadata = dict(output_metadata)
        # The channel-state tensor has its own [snapshot, subcarrier] record
        # axis.  A source-symbol record shape describes a different array and
        # must never leak into this artifact.
        remove_capture_record_metadata(state_metadata)
        state_metadata.update(
            {
                "array": "gains",
                "capture_record_axis": 0,
                "capture_record_unit": "ofdm_symbol_channel_state",
                "snapshot_count": block_count * num_ofdm_symbols,
            }
        )
        np.savez_compressed(
            symbols_path,
            symbols=symbols.astype(np.complex64, copy=False),
            metadata_json=json.dumps(output_metadata, sort_keys=True),
        )
        np.savez_compressed(
            state_path,
            h_freq=h_freq,
            gains=gains.reshape(-1, fft_size),
            metadata_json=json.dumps(state_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "symbols": artifact("channel.symbols.complex_numpy", symbols_path, output_metadata),
                "state": artifact("channel.ofdm_channel_state.numpy", state_path, state_metadata),
            },
            metrics={
                "channel.reference_snr_db": reference_snr_db,
                "channel.noise_variance": noise_variance,
                "channel.ofdm_fft_size": fft_size,
                "channel.num_ofdm_symbols": num_ofdm_symbols,
                "channel.ofdm_state_count": block_count * num_ofdm_symbols,
                "channel.ofdm_resource_element_capacity": block_count * grid_size,
                "channel.gain.average": float(np.mean(gains)),
            },
            metadata=output_metadata,
        )


class SymbolPowerAllocatorOperation(Operation):
    id = "model.symbol_power_allocator"
    name = "Budgeted symbol power allocator"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": True,
        "exportable": True,
        "reason": "Continuous-symbol power allocation is differentiable. Noema can benchmark a frozen, CSI-conditioned artifact behind an architecture-neutral runtime ABI while model training remains outside the benchmark executor.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "fixed_power_allocator", "status": "implemented", "parameter_bindings": {"policy": "fixed"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "snr_sigmoid_power_allocator", "status": "implemented", "parameter_bindings": {"policy": "snr_sigmoid"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "closed_form_water_filling_allocator", "status": "implemented", "parameter_bindings": {"policy": "water_filling"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "water_filling_on_observed_csi", "status": "implemented", "parameter_bindings": {"policy": "observed_csi_water_filling"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "correlation_shrunk_water_filling", "status": "implemented", "parameter_bindings": {"policy": "robust_csi_water_filling"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "causal_complex_ar_prediction_plus_water_filling", "status": "implemented", "parameter_bindings": {"policy": "causal_ar_water_filling"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "causal_complex_ar_prediction_plus_box_constrained_water_filling", "status": "implemented", "parameter_bindings": {"policy": "causal_ar_box_water_filling"}},
        {"runner": "benchmark_run", "backend": "onnxruntime", "implementation": "portable_trained_artifact_policy_plus_noema_projection", "status": "implemented", "parameter_bindings": {"policy": "learned_artifact"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "fixed_power_allocator", "status": "implemented", "parameter_bindings": {"policy": "fixed"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "snr_sigmoid_power_allocator", "status": "implemented", "parameter_bindings": {"policy": "snr_sigmoid"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "closed_form_water_filling_allocator", "status": "implemented", "parameter_bindings": {"policy": "water_filling"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "water_filling_on_observed_csi", "status": "implemented", "parameter_bindings": {"policy": "observed_csi_water_filling"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "correlation_shrunk_water_filling", "status": "implemented", "parameter_bindings": {"policy": "robust_csi_water_filling"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "causal_complex_ar_prediction_plus_water_filling", "status": "implemented", "parameter_bindings": {"policy": "causal_ar_water_filling"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "causal_complex_ar_prediction_plus_box_constrained_water_filling", "status": "implemented", "parameter_bindings": {"policy": "causal_ar_box_water_filling"}},
        {"runner": "dataset_capture", "backend": "onnxruntime", "implementation": "portable_trained_artifact_policy_plus_noema_projection", "status": "implemented", "parameter_bindings": {"policy": "learned_artifact"}},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_budgeted_symbol_allocator", "status": "implemented"},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "safe_npz_csi_deepset_checkpoint", "status": "implemented", "parameter_bindings": {"policy": "learned_checkpoint"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "safe_npz_csi_deepset_checkpoint", "status": "implemented", "parameter_bindings": {"policy": "learned_checkpoint"}},
    ]
    equivalence = {
        "type": "numerical",
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "reason": "Reference NumPy and Torch policies should select the same group power targets and scaled symbols for fixed parameters.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor", "checkpoint": "noema_csi_power_deepset_npz_v1"}
    learned_runtime_feature = "channel_gain"
    trained_artifact_abi = {
        "component_id": "policy",
        "component_role": "power_policy",
        "entrypoint_id": "power_policy",
        "required_operation_inputs": ["channel_state"],
        "inputs": {
            "channel_gain": {"dtype": "float32", "shape": ["batch", "subcarrier"]},
            "noise_variance": {"dtype": "float32", "shape": ["batch", 1]},
            "average_power_budget": {"dtype": "float32", "shape": ["batch", 1]},
        },
        "outputs": {
            "allocation_scores": {"dtype": "float32", "shape": ["batch", "subcarrier"]},
        },
        "constraint_adapter": "noema_exact_simplex_projection_v1",
        "binding_params": {
            "policy": "learned_artifact",
            "granularity": "per_subcarrier",
            "budget_mode": "fixed_average",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "power_policy",
        },
    }
    input_kinds = {
        "symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    optional_input_kinds = {
        "channel_state": ["channel.ofdm_channel_state.numpy"],
        "bits": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.bits.numpy",
        ],
    }
    output_kinds = {
        "symbols": "channel.symbols.complex_numpy",
        "allocation": "channel.power_allocation.numpy",
    }
    params_schema = object_schema(
        {
            "policy": {
                "type": "string",
                "default": "snr_sigmoid",
                "enum": [
                    "fixed",
                    "snr_sigmoid",
                    "water_filling",
                    "observed_csi_water_filling",
                    "robust_csi_water_filling",
                    "causal_ar_water_filling",
                    "causal_ar_box_water_filling",
                    "learned_checkpoint",
                    "learned_artifact",
                ],
                "title": "Policy",
                "description": "Power-allocation method. Model selection is shown only for learned policies.",
                "x-noema-ui": {
                    "enum_labels": {
                        "fixed": "Equal power",
                        "snr_sigmoid": "SNR-adaptive sigmoid",
                        "water_filling": "Water filling (perfect current CSI)",
                        "observed_csi_water_filling": "Water filling on observed CSI",
                        "robust_csi_water_filling": "Uncertainty-shrunk water filling",
                        "causal_ar_water_filling": "Complex-AR CSI prediction + water filling",
                        "causal_ar_box_water_filling": "Complex-AR CSI prediction + bounded water filling",
                        "learned_checkpoint": "Learned model · NPZ checkpoint",
                        "learned_artifact": "Learned model · registered artifact",
                    }
                },
            },
            "granularity": {"type": "string", "default": "global", "enum": ["global", "per_symbol", "per_subcarrier", "per_stream"]},
            "budget_mode": {"type": "string", "default": "fixed_average", "enum": ["fixed_average", "variable_average"]},
            "snr_db": {
                "type": "number",
                "default": 12.0,
                "description": "Fallback reference SNR used only by the SNR-adaptive sigmoid policy when no explicit channel state is connected.",
                "x-noema-ui": {
                    "visible_when": {"policy": "snr_sigmoid"},
                    "hidden_when_input_connected": "channel_state",
                },
            },
            "target_power": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "description": "Normalized average complex-symbol power budget per resource element; the per-state sum constraint is this value times the subcarrier count.",
                "x-noema-ui": {"label": "Normalized average TX symbol power"},
            },
            "model_batch_size": {
                "type": "integer",
                "default": 1024,
                "minimum": 1,
                "description": "Maximum number of independent channel-state examples evaluated in one learned-model inference call. Chunking preserves the complete experiment batch and output order.",
                "x-noema-ui": {
                    "label": "Model batch size",
                    "allow_sweep": False,
                    "enabled_when": {
                        "policy": ["learned_checkpoint", "learned_artifact"],
                    },
                },
            },
            "min_power": {"type": "number", "default": 0.25, "minimum": 0.0},
            "max_power": {"type": "number", "default": 2.0, "minimum": 0.0},
            "midpoint_snr_db": {"type": "number", "default": 12.0},
            "slope_db": {"type": "number", "default": 4.0, "minimum": 1e-6},
            "allocation_contrast": {"type": "number", "default": 0.6, "minimum": 0.0},
            "csi_gain_shrinkage": {
                "type": "number",
                "default": 0.6,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "For uncertainty-shrunk water filling, blend observed gains "
                    "toward their per-state mean before allocating power."
                ),
                "x-noema-ui": {
                    "visible_when": {"policy": "robust_csi_water_filling"},
                    "label": "Observed-gain confidence",
                },
            },
            "csi_prediction_horizon_ofdm_symbols": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": (
                    "For complex-AR prediction, forecast this many OFDM symbols "
                    "past the newest causal CSI snapshot. Zero uses the CSI "
                    "artifact's declared feedback delay."
                ),
                "x-noema-ui": {
                    "visible_when": {
                        "policy": [
                            "causal_ar_water_filling",
                            "causal_ar_box_water_filling",
                        ]
                    },
                    "label": "Prediction horizon",
                },
            },
            "csi_prediction_gain_confidence": {
                "type": "number",
                "default": 0.4,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Blend complex-AR predicted gains toward their per-state "
                    "frequency mean before water filling."
                ),
                "x-noema-ui": {
                    "visible_when": {
                        "policy": [
                            "causal_ar_water_filling",
                            "causal_ar_box_water_filling",
                        ]
                    },
                    "label": "Predicted-gain confidence",
                },
            },
            "allocation_lower_power_ratio": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Minimum per-subcarrier power relative to the mean budget "
                    "for bounded causal-AR water filling."
                ),
                "x-noema-ui": {
                    "visible_when": {"policy": "causal_ar_box_water_filling"},
                    "label": "Lower relative power bound",
                },
            },
            "allocation_upper_power_ratio": {
                "type": "number",
                "default": 1.9,
                "minimum": 1.0,
                "description": (
                    "Maximum per-subcarrier power relative to the mean budget "
                    "for bounded causal-AR water filling."
                ),
                "x-noema-ui": {
                    "visible_when": {"policy": "causal_ar_box_water_filling"},
                    "label": "Upper relative power bound",
                },
            },
            "subcarrier_count": {"type": "integer", "default": 64, "minimum": 1},
            "stream_count": {"type": "integer", "default": 1, "minimum": 1},
            "eps": {"type": "number", "default": 1e-12, "minimum": 0.0},
            "checkpoint_path": {
                "type": "string",
                "default": "",
                "description": "Frozen safe-NPZ CSI allocator checkpoint used by policy=learned_checkpoint.",
                "x-noema-ui": {
                    "control": "trained_artifact",
                    "label": "Learned checkpoint",
                    "accept": ".npz,application/octet-stream",
                    "visible_when": {"policy": "learned_checkpoint"},
                    "derived_params": [
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_format",
                        "checkpoint_strict",
                        "checkpoint_max_bytes",
                    ],
                },
            },
            "checkpoint_sha256": {
                "type": "string",
                "default": "",
                "description": "Required lowercase SHA-256 of the frozen allocator checkpoint.",
                "x-noema-ui": {"hidden": True},
            },
            "checkpoint_format": {
                "type": "string",
                "default": "noema_csi_power_deepset_npz_v1",
                "enum": ["noema_csi_power_deepset_npz_v1"],
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
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "description": "Registered trained-artifact manifest implementing the architecture-neutral allocator slot ABI.",
                "x-noema-ui": {
                    "control": "trained_artifact",
                    "label": "Trained artifact",
                    "accept": ".zip,.noema-artifact,.yaml,.yml,.json,application/octet-stream",
                    "visible_when": {"policy": "learned_artifact"},
                    "derived_params": [
                        "policy",
                        "artifact_manifest_path",
                        "artifact_entrypoint",
                        "artifact_package_sha256",
                    ],
                },
            },
            "artifact_entrypoint": {
                "type": "string",
                "default": "power_policy",
                "description": "Entrypoint in the registered artifact manifest used for allocator inference.",
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "description": "Registry-independent digest of the complete trained-artifact package bound into the execution plan.",
                "x-noema-ui": {"hidden": True},
            },
            "label": {"type": "string", "default": "tx_power_allocate"},
            "power_unit": {"type": "string", "default": "normalized", "enum": ["normalized", "mW"]},
            "transport_mode": {
                "type": "string",
                "default": "fixed_modulation",
                "enum": ["fixed_modulation", "allocation_aware_bit_loading"],
                "description": "Optionally convert the coded-bit stream into a CSI/allocation-aware OFDM modulation map.",
            },
            "bit_loading_bpsk_min_snr_db": {"type": "number", "default": 6.0},
            "bit_loading_qpsk_min_snr_db": {"type": "number", "default": 10.0},
            "bit_loading_qam16_min_snr_db": {"type": "number", "default": 17.0},
            "bit_loading_qam64_min_snr_db": {"type": "number", "default": 23.0},
            "bit_loading_max_bits_per_symbol": {
                "type": "integer",
                "default": 6,
                "enum": [1, 2, 4, 6],
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(np.asarray(raw_symbols).size),
            "Power allocator symbol input at %s" % ctx.step_id,
        )
        transport_bits = None
        if "bits" in ctx.inputs:
            transport_bits, _transport_bit_metadata = _load_bits(
                ctx.inputs["bits"].path,
                ctx.inputs["bits"].metadata,
            )
        channel_state = None
        if "channel_state" in ctx.inputs:
            channel_state = _load_ofdm_channel_state_artifact(ctx.inputs["channel_state"])
            state_metadata = {
                key: value
                for key, value in channel_state["metadata"].items()
                if key
                not in {
                    "array",
                    "axes",
                    "capture_record_count",
                    "capture_record_shape",
                    "capture_record_axis",
                    "record_axis",
                    "sample_axis",
                    "capture_record_unit",
                }
            }
            metadata = {**metadata, **state_metadata}
        symbols = _require_canonical_symbols(raw_symbols, ctx.step_id)
        label = _metric_label(str(ctx.params.get("label") or "tx_power_allocate"))
        policy = str(ctx.params.get("policy") or "snr_sigmoid")
        if policy not in {
            "fixed",
            "snr_sigmoid",
            "water_filling",
            "observed_csi_water_filling",
            "robust_csi_water_filling",
            "causal_ar_water_filling",
            "causal_ar_box_water_filling",
            "learned_checkpoint",
            "learned_artifact",
        }:
            raise OperationError("Unknown symbol-power allocator policy: %s" % policy)
        if (
            policy == "water_filling"
            and channel_state is not None
            and str(
                channel_state["metadata"].get("csi_role") or ""
            ).strip()
            == "delayed_noisy_transmitter_observation"
        ):
            raise OperationError(
                "policy=water_filling declares perfect current CSI, but this "
                "allocator receives delayed/noisy transmitter CSI; use "
                "policy=observed_csi_water_filling for the mismatched baseline"
            )
        granularity = str(ctx.params.get("granularity") or "global")
        if granularity not in {"global", "per_symbol", "per_subcarrier", "per_stream"}:
            granularity = "global"
        budget_mode = str(ctx.params.get("budget_mode") or "fixed_average")
        if budget_mode not in {"fixed_average", "variable_average"}:
            budget_mode = "fixed_average"
        snr_db = float(metadata.get("reference_snr_db", ctx.params.get("snr_db", 12.0)))
        raw_target_power = float(ctx.params.get("target_power", 1.0))
        if not math.isfinite(raw_target_power) or raw_target_power < 0.0:
            raise OperationError(
                "Symbol-power allocator requires finite nonnegative target_power"
            )
        target_power = raw_target_power
        model_batch_size = max(1, int(ctx.params.get("model_batch_size") or 1024))
        min_power = max(0.0, float(ctx.params.get("min_power", 0.25)))
        max_power = max(min_power, float(ctx.params.get("max_power", 2.0)))
        midpoint = float(ctx.params.get("midpoint_snr_db", 12.0))
        slope = max(1e-6, abs(float(ctx.params.get("slope_db", 4.0))))
        contrast = max(0.0, float(ctx.params.get("allocation_contrast", 0.6)))
        subcarrier_count = max(1, int(ctx.params.get("subcarrier_count", 64)))
        stream_count = max(1, int(ctx.params.get("stream_count", 1)))
        eps = max(0.0, float(ctx.params.get("eps", 1e-12)))
        unit = str(ctx.params.get("power_unit") or "normalized")
        transport_mode = str(ctx.params.get("transport_mode") or "fixed_modulation")
        if transport_mode not in {"fixed_modulation", "allocation_aware_bit_loading"}:
            raise OperationError("Unknown symbol-power allocator transport_mode: %s" % transport_mode)
        bit_loading_thresholds = {
            1: float(ctx.params.get("bit_loading_bpsk_min_snr_db", 6.0)),
            2: float(ctx.params.get("bit_loading_qpsk_min_snr_db", 10.0)),
            4: float(ctx.params.get("bit_loading_qam16_min_snr_db", 17.0)),
            6: float(ctx.params.get("bit_loading_qam64_min_snr_db", 23.0)),
        }
        max_bits_per_symbol = int(ctx.params.get("bit_loading_max_bits_per_symbol") or 6)
        if max_bits_per_symbol not in {1, 2, 4, 6}:
            raise OperationError("bit_loading_max_bits_per_symbol must be one of 1, 2, 4, or 6")
        enabled_orders = [order for order in (1, 2, 4, 6) if order <= max_bits_per_symbol]
        enabled_thresholds = [bit_loading_thresholds[order] for order in enabled_orders]
        if any(right <= left for left, right in zip(enabled_thresholds, enabled_thresholds[1:])):
            raise OperationError("Allocation-aware bit-loading SNR thresholds must increase with modulation order")
        if transport_mode == "allocation_aware_bit_loading":
            if channel_state is None:
                raise OperationError("allocation_aware_bit_loading requires explicit OFDM channel_state input")
            if transport_bits is None:
                raise OperationError("allocation_aware_bit_loading requires the coded bits input")
        csi_required_policies = {
            "water_filling",
            "observed_csi_water_filling",
            "robust_csi_water_filling",
            "causal_ar_water_filling",
            "causal_ar_box_water_filling",
            "learned_checkpoint",
            "learned_artifact",
        }
        if policy in csi_required_policies and channel_state is None:
            raise OperationError(
                "policy=%s requires the explicit channel_state input from wireless.ofdm_channel_state" % policy
            )
        if policy in csi_required_policies and budget_mode != "fixed_average":
            raise OperationError("policy=%s requires budget_mode=fixed_average" % policy)
        if policy in {"learned_checkpoint", "learned_artifact"} and granularity != "per_subcarrier":
            raise OperationError("policy=%s requires granularity=per_subcarrier" % policy)
        if channel_state is not None:
            granularity = "per_subcarrier"
            subcarrier_count = int(channel_state["gains"].shape[-1])
        alpha = 0.5 if policy == "fixed" else 1.0 / (1.0 + math.exp(-((snr_db - midpoint) / slope)))
        selected_power = target_power
        if budget_mode == "variable_average" and policy != "fixed":
            selected_power = max_power - alpha * (max_power - min_power)
        if channel_state is not None:
            state_power_budget = channel_state["metadata"].get(
                "average_power_budget"
            )
            if state_power_budget is not None:
                authoritative_power_budget = float(state_power_budget)
                if (
                    not math.isfinite(authoritative_power_budget)
                    or authoritative_power_budget < 0.0
                ):
                    raise OperationError(
                        "OFDM channel-state average_power_budget must be finite and nonnegative"
                    )
                if not math.isclose(
                    selected_power,
                    authoritative_power_budget,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise OperationError(
                        "Symbol-power allocator target %.12g contradicts the OFDM channel-state average_power_budget %.12g"
                        % (selected_power, authoritative_power_budget)
                    )
        source_symbols = symbols.astype(np.complex64, copy=False)
        flat = source_symbols.reshape(-1)
        power_before = float(np.mean(np.abs(flat) ** 2)) if int(flat.size) else 0.0
        water_levels = None
        allocation_gains = None
        modulation_orders = None
        learned_checkpoint_info: JsonDict = {}
        transport_info: JsonDict = {}
        if channel_state is not None and int(flat.size):
            gains_grid = np.maximum(np.asarray(channel_state["gains"], dtype=np.float64), 1e-12)
            if gains_grid.ndim != 3:
                raise OperationError("OFDM channel-state gains must have shape [block, ofdm_symbol, subcarrier]")
            grid_size = int(np.prod(gains_grid.shape))
            padded = _pad_complex(flat.astype(np.complex64, copy=False), grid_size)
            if int(padded.size) != grid_size:
                raise OperationError(
                    "OFDM channel state covers %d symbols but allocator received %d; regenerate CSI after changing the payload"
                    % (grid_size, int(flat.size))
                )
            symbol_grid = padded.reshape(gains_grid.shape)
            total_power_per_state = selected_power * float(subcarrier_count)
            if policy in {
                "water_filling",
                "observed_csi_water_filling",
                "robust_csi_water_filling",
                "causal_ar_water_filling",
                "causal_ar_box_water_filling",
            }:
                from noema_lab.ops.ai_phy import _water_filling_allocation

                noise_variance = max(1e-12, float(metadata.get("noise_variance") or 1.0))
                allocation_power = np.zeros_like(gains_grid, dtype=np.float64)
                water_levels = np.zeros(gains_grid.shape[:-1], dtype=np.float64)
                allocation_input_gains = gains_grid
                if policy == "robust_csi_water_filling":
                    shrinkage = float(ctx.params.get("csi_gain_shrinkage", 0.6))
                    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
                        raise OperationError(
                            "csi_gain_shrinkage must be finite and between 0 and 1"
                        )
                    state_mean_gain = np.mean(
                        gains_grid, axis=-1, keepdims=True
                    )
                    allocation_input_gains = (
                        shrinkage * gains_grid
                        + (1.0 - shrinkage) * state_mean_gain
                    )
                elif policy in {
                    "causal_ar_water_filling",
                    "causal_ar_box_water_filling",
                }:
                    csi_history = channel_state.get("csi_history")
                    if csi_history is None:
                        raise OperationError(
                            "causal-AR water filling requires a "
                            "channel-state artifact containing causal complex "
                            "csi_history"
                        )
                    history = np.asarray(csi_history, dtype=np.float32)
                    if (
                        history.ndim != 5
                        or int(history.shape[-1]) != 2
                        or int(history.shape[-2]) != subcarrier_count
                        or int(history.shape[-3]) < 2
                        or tuple(history.shape[:2]) != tuple(gains_grid.shape[:2])
                    ):
                        raise OperationError(
                            "causal complex CSI history must have shape "
                            "[block,ofdm_symbol,history,subcarrier,2] aligned "
                            "with the observed state; got %s"
                            % (list(history.shape),)
                        )
                    declared_delay = int(
                        metadata.get("feedback_delay_ofdm_symbols") or 0
                    )
                    requested_horizon = int(
                        ctx.params.get(
                            "csi_prediction_horizon_ofdm_symbols"
                        )
                        or 0
                    )
                    prediction_horizon = (
                        requested_horizon
                        if requested_horizon > 0
                        else declared_delay
                    )
                    prediction_confidence = float(
                        ctx.params.get("csi_prediction_gain_confidence", 0.4)
                    )
                    if (
                        not math.isfinite(prediction_confidence)
                        or not 0.0 <= prediction_confidence <= 1.0
                    ):
                        raise OperationError(
                            "csi_prediction_gain_confidence must be finite "
                            "and between 0 and 1"
                        )
                    allocation_input_gains = _causal_complex_ar_predicted_gains(
                        history,
                        horizon=prediction_horizon,
                        confidence=prediction_confidence,
                        eps=max(eps, 1e-30),
                    )
                for state_index in np.ndindex(gains_grid.shape[:-1]):
                    if policy == "causal_ar_box_water_filling":
                        lower_ratio = float(
                            ctx.params.get("allocation_lower_power_ratio", 0.1)
                        )
                        upper_ratio = float(
                            ctx.params.get("allocation_upper_power_ratio", 1.9)
                        )
                        state_power, water_level = (
                            _box_constrained_water_filling_allocation(
                                allocation_input_gains[state_index],
                                noise_variance,
                                total_power_per_state,
                                lower_power=lower_ratio * selected_power,
                                upper_power=upper_ratio * selected_power,
                            )
                        )
                    else:
                        state_power, water_level = _water_filling_allocation(
                            allocation_input_gains[state_index],
                            noise_variance,
                            total_power_per_state,
                        )
                    allocation_power[state_index] = state_power
                    water_levels[state_index] = water_level
            elif policy == "learned_checkpoint":
                from noema_lab.ops.channel.power_allocator_checkpoint import (
                    CHECKPOINT_FORMAT,
                    infer_csi_power_allocation,
                    load_csi_power_allocator_checkpoint,
                )

                checkpoint_format = str(ctx.params.get("checkpoint_format") or CHECKPOINT_FORMAT)
                if checkpoint_format != CHECKPOINT_FORMAT:
                    raise OperationError("Unsupported learned power-allocation checkpoint format: %s" % checkpoint_format)
                checkpoint = load_csi_power_allocator_checkpoint(
                    str(ctx.params.get("checkpoint_path") or ""),
                    str(ctx.params.get("checkpoint_sha256") or ""),
                    strict=bool(ctx.params.get("checkpoint_strict", True)),
                    max_bytes=max(1, int(ctx.params.get("checkpoint_max_bytes") or 67108864)),
                )
                noise_variance = float(metadata.get("noise_variance") or 0.0)
                gain_rows = gains_grid.reshape(-1, subcarrier_count)
                inference_example_count = int(gain_rows.shape[0])
                inference_batch_count = int(math.ceil(inference_example_count / float(model_batch_size)))
                allocation_rows = np.empty_like(gain_rows, dtype=np.float64)
                for batch_start in range(0, inference_example_count, model_batch_size):
                    batch_stop = min(inference_example_count, batch_start + model_batch_size)
                    allocation_rows[batch_start:batch_stop] = infer_csi_power_allocation(
                        checkpoint,
                        gain_rows[batch_start:batch_stop],
                        noise_variance,
                        selected_power,
                        eps=max(eps, 1e-30),
                    )
                allocation_power = allocation_rows.reshape(gains_grid.shape)
                training_provenance = dict(checkpoint.metadata.get("training") or {})
                learned_checkpoint_info = {
                    "checkpoint_path": str(checkpoint.path),
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_format": CHECKPOINT_FORMAT,
                    "checkpoint_hidden_dim": checkpoint.hidden_dim,
                    "checkpoint_training": training_provenance,
                    "model_batch_size": model_batch_size,
                    "model_batch_count": inference_batch_count,
                    "model_input_example_count": inference_example_count,
                }
            elif policy == "learned_artifact":
                from noema_lab.core.trained_artifact_runtime import (
                    run_trained_artifact_entrypoint,
                )
                from noema_lab.ops.portable_onnx import project_power_scores

                manifest_path = str(ctx.params.get("artifact_manifest_path") or "").strip()
                entrypoint = str(ctx.params.get("artifact_entrypoint") or "power_policy").strip()
                package_sha256 = str(
                    ctx.params.get("artifact_package_sha256") or ""
                ).strip()
                if not manifest_path:
                    raise OperationError("policy=learned_artifact requires artifact_manifest_path")
                noise_variance = float(metadata.get("noise_variance") or 0.0)
                gain_rows = gains_grid.reshape(-1, subcarrier_count).astype(np.float32, copy=False)
                inference_example_count = int(gain_rows.shape[0])
                inference_batch_count = int(math.ceil(inference_example_count / float(model_batch_size)))
                learned_runtime_feature = str(
                    getattr(self, "learned_runtime_feature", "channel_gain")
                    or "channel_gain"
                )
                if learned_runtime_feature == "csi_history":
                    csi_history = channel_state.get("csi_history")
                    if csi_history is None:
                        raise OperationError(
                            "model.causal_csi_power_allocator requires a "
                            "channel_state artifact containing causal complex "
                            "csi_history"
                        )
                    history_rows = np.asarray(
                        csi_history, dtype=np.float32
                    ).reshape(
                        inference_example_count,
                        int(csi_history.shape[2]),
                        subcarrier_count,
                        2,
                    )
                    artifact_feature_inputs = {
                        "csi_history": np.ascontiguousarray(history_rows)
                    }
                elif learned_runtime_feature == "channel_gain":
                    artifact_feature_inputs = {
                        "channel_gain": np.ascontiguousarray(gain_rows)
                    }
                else:
                    raise OperationError(
                        "Unsupported learned allocator runtime feature: %s"
                        % learned_runtime_feature
                    )
                artifact_outputs = run_trained_artifact_entrypoint(
                    Path(manifest_path),
                    entrypoint,
                    {
                        **artifact_feature_inputs,
                        "noise_variance": np.full((inference_example_count, 1), noise_variance, dtype=np.float32),
                        "average_power_budget": np.full((inference_example_count, 1), selected_power, dtype=np.float32),
                    },
                    expected_package_sha256=package_sha256,
                    inference_batch_size=model_batch_size,
                )
                if "allocation_scores" not in artifact_outputs:
                    raise OperationError(
                        "Power-allocation artifact entrypoint must return allocation_scores"
                    )
                allocation_rows = project_power_scores(
                    np.asarray(artifact_outputs["allocation_scores"]),
                    selected_power,
                )
                if allocation_rows.shape != gain_rows.shape:
                    raise OperationError(
                        "Power-allocation artifact returned shape %s; expected %s"
                        % (list(allocation_rows.shape), list(gain_rows.shape))
                    )
                allocation_power = allocation_rows.reshape(gains_grid.shape)
                artifact_manifest_sha256 = file_sha256(Path(manifest_path))
                learned_checkpoint_info = {
                    "artifact_manifest_path": manifest_path,
                    "artifact_manifest_sha256": artifact_manifest_sha256,
                    "artifact_package_sha256": package_sha256,
                    "checkpoint_sha256": artifact_manifest_sha256,
                    "artifact_entrypoint": entrypoint,
                    "artifact_runtime": "portable",
                    "constraint_adapter": "noema_exact_simplex_projection_v1",
                    "model_batch_size": model_batch_size,
                    "model_batch_count": inference_batch_count,
                    "model_input_example_count": inference_example_count,
                    "model_input_representation": (
                        "causal_complex_csi_history_iq"
                        if learned_runtime_feature == "csi_history"
                        else "channel_power_gain"
                    ),
                }
            else:
                base_targets = _power_allocator_targets(
                    subcarrier_count, selected_power, policy, alpha, contrast, eps
                )
                allocation_power = np.broadcast_to(base_targets, gains_grid.shape).astype(np.float64, copy=True)
            if transport_mode == "allocation_aware_bit_loading":
                allocated, modulation_orders, transport_info = _allocation_aware_ofdm_modulate(
                    np.asarray(transport_bits, dtype=np.uint8),
                    gains_grid,
                    allocation_power,
                    max(1e-12, float(metadata.get("noise_variance") or 1.0)),
                    bit_loading_thresholds,
                    max_bits_per_symbol,
                    subcarrier_count,
                )
            else:
                scales_grid = np.sqrt(np.maximum(allocation_power, 0.0)).astype(np.float32)
                allocated = (symbol_grid * scales_grid.astype(np.complex64)).reshape(-1)[: flat.size]
                allocated = allocated.reshape(source_symbols.shape).astype(np.complex64, copy=False)
            group_targets = np.mean(allocation_power, axis=(0, 1))
            group_count = int(subcarrier_count)
            scale_preview = [float(item) for item in np.sqrt(group_targets)[: min(16, group_count)].tolist()]
            allocation_gains = gains_grid.reshape(-1, subcarrier_count)
            allocation_power = allocation_power.reshape(-1, subcarrier_count)
            if water_levels is not None:
                water_levels = water_levels.reshape(-1)
        elif int(flat.size) == 0:
            allocated = source_symbols
            scale_preview = [1.0]
            group_targets = np.asarray([selected_power], dtype=np.float64)
            group_count = 1
            allocation_power = group_targets.reshape(1, -1)
        else:
            group_ids, group_count = _power_allocator_groups(int(flat.size), granularity, subcarrier_count, stream_count)
            group_targets = _power_allocator_targets(group_count, selected_power, policy, alpha, contrast, eps)
            allocated_flat = np.zeros_like(flat, dtype=np.complex64)
            scales = np.ones((group_count,), dtype=np.float64)
            for group_index in range(group_count):
                mask = group_ids == group_index
                group_symbols = flat[mask]
                group_power = float(np.mean(np.abs(group_symbols) ** 2)) if int(group_symbols.size) else 0.0
                group_target = float(group_targets[group_index])
                if group_target == 0.0:
                    scales[group_index] = 0.0
                    allocated_flat[mask] = np.zeros_like(group_symbols, dtype=np.complex64)
                elif group_power > eps:
                    scales[group_index] = float(math.sqrt(group_target / group_power))
                    allocated_flat[mask] = (group_symbols * np.complex64(scales[group_index])).astype(np.complex64, copy=False)
                else:
                    scales[group_index] = 1.0
                    allocated_flat[mask] = group_symbols
            allocated = allocated_flat.reshape(source_symbols.shape).astype(np.complex64, copy=False)
            scale_preview = [float(item) for item in scales[: min(16, int(scales.size))].tolist()]
            allocation_power = group_targets.reshape(1, -1)
        power_preview, power_after, total_energy = _symbol_power_trace_metadata(allocated, label, unit=unit)
        target_preview = [float(item) for item in group_targets[: min(16, int(group_targets.size))].tolist()]
        power_preview.update(
            {
                "allocation_granularity": granularity,
                "allocation_group_count": int(group_count),
                "allocation_target_power_preview": target_preview,
                "allocation_scale_preview": scale_preview,
            }
        )
        output_metadata = dict(metadata)
        allocation_metadata = dict(metadata)
        remove_capture_record_metadata(allocation_metadata)
        _remove_source_item_partition_metadata(allocation_metadata)
        allocation_metadata.update(
            {
                "array": "power",
                "capture_record_axis": 0,
                "capture_record_unit": "ofdm_symbol_channel_state",
                "policy": (
                    "theoretical_water_filling"
                    if policy == "water_filling"
                    else "water_filling_on_observed_csi"
                    if policy == "observed_csi_water_filling"
                    else "uncertainty_shrunk_water_filling"
                    if policy == "robust_csi_water_filling"
                    else "causal_complex_ar_prediction_water_filling"
                    if policy == "causal_ar_water_filling"
                    else "causal_complex_ar_prediction_box_constrained_water_filling"
                    if policy == "causal_ar_box_water_filling"
                    else "learned_csi_power_allocator"
                    if policy in {"learned_checkpoint", "learned_artifact"}
                    else policy
                ),
                "objective": (
                    "maximize_parallel_channel_sum_rate_with_current_csi"
                    if policy == "water_filling"
                    else "maximize_parallel_channel_sum_rate_with_observed_csi"
                    if policy == "observed_csi_water_filling"
                    else "uncertainty_shrunk_observed_csi_heuristic"
                    if policy == "robust_csi_water_filling"
                    else "predict_current_csi_from_causal_history_then_maximize_shannon_sum_rate"
                    if policy == "causal_ar_water_filling"
                    else "predict_current_csi_then_box_constrained_shannon_water_filling"
                    if policy == "causal_ar_box_water_filling"
                    else "trained_artifact_declared_objective"
                    if policy in {"learned_checkpoint", "learned_artifact"}
                    else "configured_power_policy"
                ),
                "constraint": "fixed_average_power" if budget_mode == "fixed_average" else "variable_average_power",
                "allocation_axis": "subcarrier" if granularity == "per_subcarrier" else granularity,
                "allocation_granularity": (
                    "per_subcarrier_per_ofdm_symbol_state"
                    if channel_state is not None
                    else granularity
                ),
                "snapshot_axis": "ofdm_symbol_channel_state" if channel_state is not None else "allocation_run",
                "snapshot_count": int(allocation_power.shape[0]),
                "subcarrier_count": int(allocation_power.shape[1]),
                "target_power": selected_power,
                "total_power": selected_power * float(allocation_power.shape[1]),
                "noise_variance": float(metadata.get("noise_variance") or 1.0),
                "reference_snr_db": snr_db,
                "transport_mode": transport_mode,
            }
        )
        if transport_info:
            allocation_metadata.update(transport_info)
        if policy in {
            "causal_ar_water_filling",
            "causal_ar_box_water_filling",
        }:
            configured_horizon = int(
                ctx.params.get("csi_prediction_horizon_ofdm_symbols") or 0
            )
            allocation_metadata.update(
                {
                    "csi_prediction_model": "per_subcarrier_complex_ar1",
                    "csi_prediction_history_order": "oldest_to_newest",
                    "csi_prediction_horizon_ofdm_symbols": (
                        configured_horizon
                        if configured_horizon > 0
                        else int(
                            metadata.get("feedback_delay_ofdm_symbols") or 0
                        )
                    ),
                    "csi_prediction_gain_confidence": float(
                        ctx.params.get(
                            "csi_prediction_gain_confidence", 0.4
                        )
                    ),
                }
            )
            if policy == "causal_ar_box_water_filling":
                allocation_metadata.update(
                    {
                        "allocation_lower_power_ratio": float(
                            ctx.params.get("allocation_lower_power_ratio", 0.1)
                        ),
                        "allocation_upper_power_ratio": float(
                            ctx.params.get("allocation_upper_power_ratio", 1.9)
                        ),
                    }
                )
        if learned_checkpoint_info:
            allocation_metadata.update(learned_checkpoint_info)
        if allocation_gains is not None:
            from noema_lab.ops.ai_phy import _resource_allocation_preview

            allocation_metadata["resource_allocation_preview"] = _resource_allocation_preview(
                allocation_gains,
                allocation_power,
                float(allocation_metadata["noise_variance"]),
                allocation_metadata,
                allocation_metadata,
                water_levels,
            )
        output_metadata.update(
            {
                "symbol_count": int(allocated.size),
                "boundary_contract": "channel.symbols",
                "channel_use_count": int(allocated.size),
                "symbol_storage": "complex64",
                "power_allocated": True,
                "power_allocator_label": label,
                "power_allocator_policy": policy,
                "power_allocator_granularity": granularity,
                "power_allocator_budget_mode": budget_mode,
                "power_budget_unit": "average_symbol_power",
                "power_allocator_snr_db": snr_db,
                "power_allocator_selected_power": selected_power,
                "power_allocator_min_power": min_power,
                "power_allocator_max_power": max_power,
                "power_allocator_midpoint_snr_db": midpoint,
                "power_allocator_slope_db": slope,
                "power_allocator_allocation_contrast": contrast,
                "power_allocator_subcarrier_count": subcarrier_count,
                "power_allocator_stream_count": stream_count,
                "power_allocator_group_count": int(group_count),
                "power_before": power_before,
                "power_after": power_after,
                "power_total_energy": total_energy,
                "power_unit": unit,
                "power_unit_note": "normalized linear symbol power unless the recipe declares calibrated physical units",
                "tx_power_preview": power_preview,
                "power_allocation_scale_preview": scale_preview,
                "power_allocation_target_preview": target_preview,
                "transport_mode": transport_mode,
            }
        )
        if transport_info:
            output_metadata.update(transport_info)
        if learned_checkpoint_info:
            output_metadata.update(learned_checkpoint_info)
        if allocation_metadata.get("resource_allocation_preview"):
            output_metadata["resource_allocation_preview"] = allocation_metadata["resource_allocation_preview"]
        if int(allocated.size) != int(np.asarray(raw_symbols).size):
            # Allocation-aware bit loading schedules a new representation whose
            # records are not the source symbol records.
            remove_capture_record_metadata(output_metadata)
            _remove_source_item_partition_metadata(output_metadata)
            output_metadata.update(
                {
                    "source_item_partition_preserved": False,
                    "source_item_partition_removed_reason": (
                        "allocation_aware_bit_loading_reschedules_the_flat_bit_stream"
                    ),
                }
            )
        elif capture_layout is not None:
            # The fixed-modulation allocator is cardinality preserving.  The
            # loader already validated the exact source layout.
            if capture_layout.element_count != int(allocated.size):
                raise OperationError(
                    "Power allocator changed an explicit capture-record layout"
                )
        path = ctx.output_path("symbols", ".npz")
        allocation_path = ctx.output_path("allocation", ".npz")
        symbols_payload = {
            "symbols": allocated,
            "metadata_json": json.dumps(output_metadata, sort_keys=True),
        }
        allocation_payload = {
            "power": allocation_power.astype(np.float32),
            "metadata_json": json.dumps(allocation_metadata, sort_keys=True),
        }
        if modulation_orders is not None:
            allocation_payload["modulation_order_bits_per_symbol"] = modulation_orders.astype(np.uint8, copy=False)
        if channel_state is not None:
            allocation_payload["gains"] = allocation_gains.astype(np.float32, copy=False)
        if water_levels is not None:
            allocation_payload["water_level"] = water_levels.astype(np.float32, copy=False)
        np.savez_compressed(path, **symbols_payload)
        np.savez_compressed(allocation_path, **allocation_payload)
        return OperationResult(
            outputs={
                "symbols": artifact("channel.symbols.complex_numpy", path, output_metadata),
                "allocation": artifact("channel.power_allocation.numpy", allocation_path, allocation_metadata),
            },
            metrics={
                "channel.tx_power.%s.before" % label: power_before,
                "channel.tx_power.%s.after" % label: power_after,
                "channel.tx_power.%s.selected" % label: selected_power,
                "channel.tx_power.%s.total_energy" % label: total_energy,
                "channel.tx_power.%s.snr_db" % label: snr_db,
                "channel.tx_power.before": power_before,
                "channel.tx_power.after": power_after,
                "channel.tx_power.average": power_after,
                "channel.tx_power.selected": selected_power,
                "channel.tx_power.total_energy": total_energy,
                "channel.tx_power.snr_db": snr_db,
                "channel.tx_power.group_count": int(group_count),
                **(
                    {
                        "channel.tx_power.model_batch_size": int(learned_checkpoint_info["model_batch_size"]),
                        "channel.tx_power.model_batch_count": int(learned_checkpoint_info["model_batch_count"]),
                        "channel.tx_power.model_input_example_count": int(
                            learned_checkpoint_info["model_input_example_count"]
                        ),
                    }
                    if learned_checkpoint_info
                    else {}
                ),
                "channel.symbol_count": int(allocated.size),
                "channel.channel_use_count": int(allocated.size),
                "channel.scheduled_resource_element_count": int(allocated.size),
                "channel.data_bearing_resource_element_count": int(
                    transport_info.get("adaptive_data_resource_element_count", allocated.size)
                ),
            },
            metadata={
                "label": label,
                "policy": policy,
                "granularity": granularity,
                "budget_mode": budget_mode,
                "snr_db": snr_db,
                "selected_power": selected_power,
                "power_before": power_before,
                "power_after": power_after,
                "total_energy": total_energy,
                "power_unit": unit,
                "group_count": int(group_count),
                "symbol_count": int(allocated.size),
                "transport_mode": transport_mode,
                **learned_checkpoint_info,
                **transport_info,
            },
        )


class CausalCsiPowerAllocatorOperation(SymbolPowerAllocatorOperation):
    """Power allocator slot whose learned ABI consumes causal complex CSI."""

    id = "model.causal_csi_power_allocator"
    name = "Causal-history symbol power allocator"
    learned_runtime_feature = "csi_history"
    trained_artifact_abi = {
        "component_id": "policy",
        "component_role": "power_policy",
        "entrypoint_id": "power_policy",
        "required_operation_inputs": ["channel_state"],
        "inputs": {
            "csi_history": {
                "dtype": "float32",
                "shape": ["batch", "history", "subcarrier", 2],
            },
            "noise_variance": {
                "dtype": "float32",
                "shape": ["batch", 1],
            },
            "average_power_budget": {
                "dtype": "float32",
                "shape": ["batch", 1],
            },
        },
        "outputs": {
            "allocation_scores": {
                "dtype": "float32",
                "shape": ["batch", "subcarrier"],
            },
        },
        "constraint_adapter": "noema_exact_simplex_projection_v1",
        "binding_params": {
            "policy": "learned_artifact",
            "granularity": "per_subcarrier",
            "budget_mode": "fixed_average",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "power_policy",
        },
    }


def _causal_complex_ar_predicted_gains(
    history_iq: np.ndarray,
    *,
    horizon: int,
    confidence: float,
    eps: float,
) -> np.ndarray:
    """Predict later per-subcarrier gains from an oldest-to-newest CSI history.

    A complex AR(1) coefficient is fitted independently for every aligned
    state/subcarrier using only consecutive causal history samples. Its
    magnitude is capped at one for stable multi-step prediction. The resulting
    gains are shrunk toward their per-state frequency mean before allocation,
    which makes the classical baseline robust to short, noisy histories.
    """

    history = np.asarray(history_iq, dtype=np.float64)
    complex_history = history[..., 0] + 1j * history[..., 1]
    previous = complex_history[..., :-1, :]
    following = complex_history[..., 1:, :]
    denominator = np.sum(np.abs(previous) ** 2, axis=-2) + float(eps)
    coefficient = (
        np.sum(following * np.conjugate(previous), axis=-2)
        / denominator
    )
    magnitude = np.abs(coefficient)
    coefficient = coefficient / np.maximum(magnitude, 1.0)
    predicted = complex_history[..., -1, :] * np.power(
        coefficient,
        max(0, int(horizon)),
    )
    predicted_gains = np.maximum(np.abs(predicted) ** 2, float(eps))
    state_mean = np.mean(predicted_gains, axis=-1, keepdims=True)
    return (
        float(confidence) * predicted_gains
        + (1.0 - float(confidence)) * state_mean
    )


def _box_constrained_water_filling_allocation(
    gains: np.ndarray,
    noise_variance: float,
    total_power: float,
    *,
    lower_power: float,
    upper_power: float,
) -> tuple[np.ndarray, float]:
    """Water filling with a fixed lower and upper power bound per tone."""

    raw_gain = np.asarray(gains, dtype=np.float64).reshape(-1)
    if raw_gain.size < 1 or not np.all(np.isfinite(raw_gain)):
        raise OperationError(
            "Box-constrained water-filling gains must be a finite nonempty array"
        )
    gain = np.maximum(raw_gain, 1e-12)
    noise = float(noise_variance)
    total = float(total_power)
    lower = float(lower_power)
    upper = float(upper_power)
    if (
        not all(math.isfinite(value) for value in (noise, total, lower, upper))
        or noise <= 0.0
        or total < 0.0
        or lower < 0.0
        or upper < lower
    ):
        raise OperationError("Box-constrained water-filling parameters are invalid")
    count = int(gain.size)
    tolerance = 1e-10 * max(1.0, abs(total))
    if total < lower * count - tolerance or total > upper * count + tolerance:
        raise OperationError(
            "Box-constrained water-filling bounds do not contain the power budget"
        )
    inverse_gain_noise = noise / gain
    left = float(np.min(inverse_gain_noise + lower)) - max(1.0, upper)
    right = float(np.max(inverse_gain_noise + upper)) + max(1.0, upper)
    for _ in range(96):
        level = 0.5 * (left + right)
        allocated = np.clip(level - inverse_gain_noise, lower, upper)
        if float(np.sum(allocated)) < total:
            left = level
        else:
            right = level
    water_level = 0.5 * (left + right)
    allocated = np.clip(water_level - inverse_gain_noise, lower, upper)
    # Remove the final floating-point residual without leaving the box.
    for _ in range(4):
        residual = total - float(np.sum(allocated))
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            eligible = allocated < upper - tolerance
        else:
            eligible = allocated > lower + tolerance
        eligible_count = int(np.count_nonzero(eligible))
        if eligible_count == 0:
            break
        allocated[eligible] += residual / float(eligible_count)
        allocated = np.clip(allocated, lower, upper)
    if abs(float(np.sum(allocated)) - total) > 1e-8 * max(1.0, total):
        raise OperationError("Box-constrained water filling did not close the budget")
    return allocated.astype(np.float64, copy=False), float(water_level)


def _power_allocator_groups(symbol_count: int, granularity: str, subcarrier_count: int, stream_count: int) -> Tuple[np.ndarray, int]:
    symbol_count = max(0, int(symbol_count))
    if symbol_count == 0:
        return np.zeros((0,), dtype=np.int64), 1
    if granularity == "per_symbol":
        return np.arange(symbol_count, dtype=np.int64), symbol_count
    if granularity == "per_subcarrier":
        count = max(1, min(int(subcarrier_count), symbol_count))
        return (np.arange(symbol_count, dtype=np.int64) % count), count
    if granularity == "per_stream":
        count = max(1, min(int(stream_count), symbol_count))
        return (np.arange(symbol_count, dtype=np.int64) % count), count
    return np.zeros((symbol_count,), dtype=np.int64), 1


def _power_allocator_targets(group_count: int, selected_power: float, policy: str, alpha: float, contrast: float, eps: float) -> np.ndarray:
    group_count = max(1, int(group_count))
    selected_power = max(0.0, float(selected_power))
    if group_count == 1 or policy == "fixed" or contrast <= 0.0 or selected_power == 0.0:
        return np.full((group_count,), selected_power, dtype=np.float64)
    positions = np.linspace(-1.0, 1.0, group_count, dtype=np.float64)
    tilt = (float(alpha) - 0.5) * 2.0 * float(contrast)
    weights = np.exp(tilt * positions)
    mean_weight = float(np.mean(weights)) if int(weights.size) else 1.0
    if mean_weight <= eps:
        weights = np.ones_like(weights, dtype=np.float64)
    else:
        weights = weights / mean_weight
    return (selected_power * weights).astype(np.float64, copy=False)


def _allocation_aware_ofdm_modulate(
    bits: np.ndarray,
    gains_grid: np.ndarray,
    power_grid: np.ndarray,
    noise_variance: float,
    thresholds_db: Dict[int, float],
    max_bits_per_symbol: int,
    subcarrier_count: int,
) -> Tuple[np.ndarray, np.ndarray, JsonDict]:
    source_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    gains = np.maximum(np.asarray(gains_grid, dtype=np.float64).reshape(-1), 1e-12)
    power = np.maximum(np.asarray(power_grid, dtype=np.float64).reshape(-1), 0.0)
    if gains.shape != power.shape:
        raise OperationError("Allocation-aware OFDM modulation requires matching gain and power grids")
    effective_snr_db = 10.0 * np.log10(np.maximum(gains * power / max(noise_variance, 1e-12), 1e-15))
    orders = np.zeros((int(power.size),), dtype=np.uint8)
    for order in (1, 2, 4, 6):
        if order <= int(max_bits_per_symbol):
            orders[effective_snr_db >= float(thresholds_db[order])] = np.uint8(order)
    orders[power <= 1e-12] = np.uint8(0)
    capacity = np.cumsum(orders.astype(np.int64), dtype=np.int64)
    required_bits = int(source_bits.size)
    if required_bits == 0:
        return (
            np.zeros((0,), dtype=np.complex64),
            np.zeros_like(orders, dtype=np.uint8),
            {
                "modulation": "allocation_aware",
                "allocation_aware_transport": True,
                "adaptive_original_bit_count": 0,
                "adaptive_padded_bit_count": 0,
                "adaptive_bit_padding_count": 0,
                "adaptive_scheduled_resource_element_count": 0,
                "adaptive_data_resource_element_count": 0,
                "adaptive_scheduled_ofdm_symbol_count": 0,
                "adaptive_payload_bits_per_resource_element": 0.0,
                "adaptive_modulation_order_histogram": {},
                "adaptive_bit_loading_thresholds_db": {
                    str(order): float(thresholds_db[order]) for order in (1, 2, 4, 6)
                },
            },
        )
    available_bits = int(capacity[-1]) if int(capacity.size) else 0
    if available_bits < required_bits:
        raise OperationError(
            "Allocation-aware OFDM grid can carry %d coded bits but the payload requires %d; "
            "increase channel_state.capacity_multiplier, lower the noise, or relax the bit-loading thresholds"
            % (available_bits, required_bits)
        )
    last_data_index = int(np.searchsorted(capacity, required_bits, side="left"))
    fft_size = max(1, int(subcarrier_count))
    scheduled_count = min(
        int(orders.size),
        int(math.ceil(float(last_data_index + 1) / float(fft_size))) * fft_size,
    )
    scheduled_orders = orders[:scheduled_count].copy()
    if last_data_index + 1 < scheduled_count:
        scheduled_orders[last_data_index + 1 :] = np.uint8(0)
    mapped_capacity = int(np.sum(scheduled_orders.astype(np.int64)))
    unit_symbols = _modulate_variable_orders(source_bits, scheduled_orders)
    allocated = (
        unit_symbols * np.sqrt(power[:scheduled_count]).astype(np.complex64, copy=False)
    ).astype(np.complex64, copy=False)
    full_orders = np.zeros_like(orders, dtype=np.uint8)
    full_orders[:scheduled_count] = scheduled_orders
    histogram = {
        str(order): int(np.count_nonzero(scheduled_orders == order))
        for order in (0, 1, 2, 4, 6)
    }
    data_resource_count = int(np.count_nonzero(scheduled_orders))
    return allocated, full_orders, {
        "modulation": "allocation_aware",
        "allocation_aware_transport": True,
        "adaptive_original_bit_count": required_bits,
        "adaptive_padded_bit_count": mapped_capacity,
        "adaptive_bit_padding_count": mapped_capacity - required_bits,
        "adaptive_scheduled_resource_element_count": scheduled_count,
        "adaptive_data_resource_element_count": data_resource_count,
        "adaptive_scheduled_ofdm_symbol_count": scheduled_count // fft_size,
        "adaptive_payload_bits_per_resource_element": float(required_bits) / float(scheduled_count),
        "adaptive_modulation_order_histogram": histogram,
        "adaptive_bit_loading_thresholds_db": {
            str(order): float(thresholds_db[order]) for order in (1, 2, 4, 6)
        },
        "adaptive_max_bits_per_symbol": int(max_bits_per_symbol),
        "bits_per_symbol": float(required_bits) / float(scheduled_count),
        "bit_count": required_bits,
        "padded_bit_count": mapped_capacity,
    }


def _modulate_variable_orders(bits: np.ndarray, orders: np.ndarray) -> np.ndarray:
    source_bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    orders = np.asarray(orders, dtype=np.uint8).reshape(-1)
    offsets = np.cumsum(orders.astype(np.int64), dtype=np.int64) - orders.astype(np.int64)
    mapped_count = int(np.sum(orders.astype(np.int64)))
    padded = np.zeros((mapped_count,), dtype=np.uint8)
    padded[: min(int(source_bits.size), mapped_count)] = source_bits[:mapped_count]
    symbols = np.zeros((int(orders.size),), dtype=np.complex64)
    for order in (1, 2, 4, 6):
        positions = np.flatnonzero(orders == order)
        if not int(positions.size):
            continue
        groups = padded[offsets[positions, None] + np.arange(order, dtype=np.int64)[None, :]]
        if order == 1:
            values = 1.0 - 2.0 * groups[:, 0].astype(np.float32)
            symbols[positions] = values.astype(np.complex64)
            continue
        axis_bits = order // 2
        weights = (2 ** np.arange(axis_bits - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
        i_index = np.sum(groups[:, :axis_bits].astype(np.int64) * weights, axis=1)
        q_index = np.sum(groups[:, axis_bits:].astype(np.int64) * weights, axis=1)
        if order == 2:
            i_values = 1.0 - 2.0 * i_index.astype(np.float32)
            q_values = 1.0 - 2.0 * q_index.astype(np.float32)
            normalization = np.sqrt(2.0)
        else:
            level_count = 2 ** axis_bits
            levels = np.arange(-(level_count - 1), level_count, 2, dtype=np.float32)
            i_values = levels[i_index]
            q_values = levels[q_index]
            normalization = np.sqrt((2.0 / 3.0) * (level_count ** 2 - 1))
        symbols[positions] = ((i_values + 1j * q_values) / normalization).astype(np.complex64)
    return symbols


def _demodulate_variable_orders(symbols: np.ndarray, orders: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    symbols = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    orders = np.asarray(orders, dtype=np.uint8).reshape(-1)
    if symbols.shape != orders.shape:
        raise OperationError("Allocation-aware demodulation requires one modulation order per received resource element")
    offsets = np.cumsum(orders.astype(np.int64), dtype=np.int64) - orders.astype(np.int64)
    bit_count = int(np.sum(orders.astype(np.int64)))
    bits = np.zeros((bit_count,), dtype=np.uint8)
    for order in (1, 2, 4, 6):
        positions = np.flatnonzero(orders == order)
        if not int(positions.size):
            continue
        selected = symbols[positions]
        if order == 1:
            groups = (selected.real < 0.0).astype(np.uint8).reshape(-1, 1)
        else:
            axis_bits = order // 2
            if order == 2:
                i_index = (selected.real < 0.0).astype(np.int64)
                q_index = (selected.imag < 0.0).astype(np.int64)
            else:
                level_count = 2 ** axis_bits
                normalization = np.sqrt((2.0 / 3.0) * (level_count ** 2 - 1))
                scaled_i = selected.real.astype(np.float64) * normalization
                scaled_q = selected.imag.astype(np.float64) * normalization
                i_index = np.clip(
                    np.rint((scaled_i + float(level_count - 1)) / 2.0), 0, level_count - 1
                ).astype(np.int64)
                q_index = np.clip(
                    np.rint((scaled_q + float(level_count - 1)) / 2.0), 0, level_count - 1
                ).astype(np.int64)
            groups = np.zeros((int(positions.size), order), dtype=np.uint8)
            for bit_index in range(axis_bits):
                shift = axis_bits - bit_index - 1
                groups[:, bit_index] = ((i_index >> shift) & 1).astype(np.uint8)
                groups[:, axis_bits + bit_index] = ((q_index >> shift) & 1).astype(np.uint8)
        target_indices = offsets[positions, None] + np.arange(order, dtype=np.int64)[None, :]
        bits[target_indices.reshape(-1)] = groups.reshape(-1)
    llr = np.where(bits == 0, 1.0, -1.0).astype(np.float32)
    return bits, llr


def _required_finite_metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    owner: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if key not in metadata or metadata.get(key) is None:
        raise OperationError("%s metadata is missing required %s" % (owner, key))
    try:
        value = float(metadata[key])
    except (TypeError, ValueError) as exc:
        raise OperationError(
            "%s metadata %s must be numeric" % (owner, key)
        ) from exc
    if not math.isfinite(value):
        raise OperationError("%s metadata %s must be finite" % (owner, key))
    if minimum is not None:
        invalid = value <= minimum if exclusive_minimum else value < minimum
        if invalid:
            qualifier = "greater than" if exclusive_minimum else "at least"
            raise OperationError(
                "%s metadata %s must be %s %.12g"
                % (owner, key, qualifier, minimum)
            )
    return value


def _reject_conflicting_metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    expected: float,
    owner: str,
) -> None:
    if key not in metadata or metadata.get(key) is None:
        return
    actual = _required_finite_metadata_float(metadata, key, owner)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise OperationError(
            "%s metadata %s %.12g contradicts authoritative channel-state value %.12g"
            % (owner, key, actual, expected)
        )


class OfdmPowerAllocationMetricsOperation(Operation):
    id = "metrics.ofdm_power_allocation"
    name = "OFDM power-allocation metrics"
    input_kinds = {
        "state": ["channel.ofdm_channel_state.numpy"],
        "allocation": ["channel.power_allocation.numpy"],
    }
    optional_input_kinds = {
        "reference": ["channel.payload_bits.numpy", "channel.bits.numpy"],
        "candidate": ["channel.payload_bits.numpy", "channel.bits.numpy"],
        "symbols": ["channel.symbols.complex_numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "transport_block_size_bits": {
                "type": "integer",
                "default": 1024,
                "minimum": 1,
                "description": "Payload bits per independently scored transport block for empirical BLER and goodput.",
            },
            "outage_target_spectral_efficiency_bps_hz": {
                "type": "number",
                "default": 2.0,
                "minimum": 0.0,
                "description": "Target instantaneous Shannon spectral efficiency used for the fading-outage statistic.",
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        state_arrays, state_metadata = _load_npz_arrays(ctx.require_input("state").path)
        allocation_arrays, allocation_metadata = _load_npz_arrays(ctx.require_input("allocation").path)
        if "gains" not in state_arrays or "power" not in allocation_arrays:
            raise OperationError("OFDM allocation metrics require channel gains and allocated power arrays")
        gains = np.asarray(state_arrays["gains"], dtype=np.float64)
        power = np.asarray(allocation_arrays["power"], dtype=np.float64)
        if gains.ndim != 2 or any(int(size) == 0 for size in gains.shape):
            raise OperationError(
                "OFDM channel gains must be a non-empty 2-D [state, subcarrier] array"
            )
        if power.ndim != 2 or any(int(size) == 0 for size in power.shape):
            raise OperationError(
                "OFDM power allocation must be a non-empty 2-D [state, subcarrier] array"
            )
        if not bool(np.all(np.isfinite(gains))):
            raise OperationError("OFDM channel gains must contain only finite values")
        if bool(np.any(gains < 0.0)):
            raise OperationError("OFDM channel gains must be nonnegative")
        if not bool(np.all(np.isfinite(power))):
            raise OperationError("OFDM power allocation must contain only finite values")
        if bool(np.any(power < 0.0)):
            raise OperationError(
                "OFDM power allocation is infeasible because it contains negative power"
            )
        if gains.shape != power.shape:
            raise OperationError(
                "OFDM state/allocation shape mismatch: gains %s, power %s" % (gains.shape, power.shape)
            )
        noise_variance = _required_finite_metadata_float(
            state_metadata,
            "noise_variance",
            "OFDM channel state",
            minimum=0.0,
            exclusive_minimum=True,
        )
        target_power = _required_finite_metadata_float(
            state_metadata,
            "average_power_budget",
            "OFDM channel state",
            minimum=0.0,
        )
        total_power = target_power * float(gains.shape[1])
        if not math.isfinite(total_power):
            raise OperationError(
                "OFDM channel-state power budget is too large to represent"
            )
        _reject_conflicting_metadata_float(
            state_metadata,
            "total_power_budget",
            total_power,
            "OFDM channel state",
        )
        _reject_conflicting_metadata_float(
            allocation_metadata,
            "noise_variance",
            noise_variance,
            "OFDM allocation",
        )
        _reject_conflicting_metadata_float(
            allocation_metadata,
            "target_power",
            target_power,
            "OFDM allocation",
        )
        _reject_conflicting_metadata_float(
            allocation_metadata,
            "total_power",
            total_power,
            "OFDM allocation",
        )
        power_sums = np.sum(power, axis=1)
        if not bool(np.all(np.isfinite(power_sums))):
            raise OperationError("OFDM power-allocation row sums must be finite")
        power_error = np.abs(power_sums - total_power)
        feasibility_tolerance = max(
            1e-6, 1e-7 * max(1.0, abs(total_power))
        )
        if bool(np.any(power_error > feasibility_tolerance)):
            worst_state = int(np.argmax(power_error))
            raise OperationError(
                "OFDM power allocation violates the channel-state sum-power budget "
                "at state %d: allocated %.12g, required %.12g, tolerance %.12g"
                % (
                    worst_state,
                    float(power_sums[worst_state]),
                    total_power,
                    feasibility_tolerance,
                )
            )
        rates = np.log2(1.0 + gains * power / noise_variance)
        if not bool(np.all(np.isfinite(rates))):
            raise OperationError(
                "OFDM allocation produced non-finite Shannon-rate values"
            )
        sum_rates = np.sum(rates, axis=1)
        spectral_efficiencies = np.mean(rates, axis=1)
        outage_target = float(
            ctx.params.get(
                "outage_target_spectral_efficiency_bps_hz", 2.0
            )
        )
        if not math.isfinite(outage_target) or outage_target < 0.0:
            raise OperationError(
                "OFDM outage target must be finite and nonnegative"
            )
        from noema_lab.ops.ai_phy import _water_filling_allocation

        oracle_power = np.zeros_like(power, dtype=np.float64)
        oracle_gains = np.maximum(gains, 1e-12)
        for state_index in range(int(gains.shape[0])):
            oracle_power[state_index], _water_level = _water_filling_allocation(
                oracle_gains[state_index], noise_variance, total_power
            )
        oracle_rates = np.log2(1.0 + gains * oracle_power / noise_variance)
        oracle_spectral_efficiencies = np.mean(oracle_rates, axis=1)
        state_oracle_gaps = oracle_spectral_efficiencies - spectral_efficiencies
        oracle_tolerances = np.maximum(
            1e-7,
            1e-8 * np.maximum(1.0, np.abs(oracle_spectral_efficiencies)),
        )
        if bool(np.any(state_oracle_gaps < -oracle_tolerances)):
            invalid_state = int(np.argmin(state_oracle_gaps))
            raise OperationError(
                "OFDM candidate exceeds the water-filling upper bound at state %d "
                "by %.12g bps/Hz"
                % (invalid_state, float(-state_oracle_gaps[invalid_state]))
            )
        mean_spectral_efficiency = float(np.mean(spectral_efficiencies))
        mean_oracle_spectral_efficiency = float(np.mean(oracle_spectral_efficiencies))
        raw_optimality_gap = (
            mean_oracle_spectral_efficiency - mean_spectral_efficiency
        )
        optimality_gap = max(0.0, raw_optimality_gap)
        relative_optimality_gap = (
            optimality_gap / mean_oracle_spectral_efficiency
            if mean_oracle_spectral_efficiency > 1e-12
            else 0.0
        )
        oracle_power_rmse = float(np.sqrt(np.mean((power - oracle_power) ** 2)))
        oracle_power_normalized_rmse = oracle_power_rmse / max(
            float(np.mean(oracle_power)), 1e-12
        )
        marginal_rate_derivative = oracle_gains / (
            math.log(2.0) * (noise_variance + oracle_gains * power)
        )
        kkt_residuals = np.zeros((int(power.shape[0]),), dtype=np.float64)
        kkt_active_threshold = max(1e-12, target_power * 1e-9)
        for state_index in range(int(power.shape[0])):
            state_active = power[state_index] > kkt_active_threshold
            if not np.any(state_active):
                continue
            lagrange_level = float(np.mean(marginal_rate_derivative[state_index][state_active]))
            scale = max(abs(lagrange_level), 1e-12)
            active_residual = float(
                np.max(
                    np.abs(
                        marginal_rate_derivative[state_index][state_active] - lagrange_level
                    )
                )
                / scale
            )
            inactive_residual = 0.0
            if np.any(~state_active):
                inactive_residual = float(
                    np.max(
                        np.maximum(
                            marginal_rate_derivative[state_index][~state_active] - lagrange_level,
                            0.0,
                        )
                    )
                    / scale
                )
            kkt_residuals[state_index] = max(active_residual, inactive_residual)
        power_scale = max(abs(total_power), 1e-12)
        active_threshold = max(1e-12, target_power * 1e-9)
        active_mask = power > active_threshold
        mean_power = np.mean(power, axis=1)
        allocation_cv = np.divide(
            np.std(power, axis=1),
            mean_power,
            out=np.zeros_like(mean_power),
            where=mean_power > 1e-12,
        )
        allocated_power_sum = np.sum(power, axis=1)
        power_weighted_gain = np.divide(
            np.sum(gains * power, axis=1),
            allocated_power_sum,
            out=np.zeros_like(allocated_power_sum),
            where=allocated_power_sum > 1e-12,
        )
        unweighted_gain = np.mean(gains, axis=1)
        gain_weighted_lift = np.divide(
            power_weighted_gain,
            unweighted_gain,
            out=np.ones_like(power_weighted_gain),
            where=unweighted_gain > 1e-12,
        ) - 1.0
        delivery_metrics: JsonDict = {}
        delivery_report: JsonDict = {}
        if all(name in ctx.inputs for name in ("reference", "candidate", "symbols")):
            reference_bits, _reference_metadata = _load_bits(
                ctx.inputs["reference"].path, ctx.inputs["reference"].metadata
            )
            candidate_bits, _candidate_metadata = _load_bits(
                ctx.inputs["candidate"].path, ctx.inputs["candidate"].metadata
            )
            tx_symbols, tx_symbol_metadata = _load_symbols(
                ctx.inputs["symbols"].path, ctx.inputs["symbols"].metadata
            )
            compare_count = min(int(reference_bits.size), int(candidate_bits.size))
            payload_error_count = int(
                np.count_nonzero(reference_bits[:compare_count] != candidate_bits[:compare_count])
            ) + abs(int(reference_bits.size) - int(candidate_bits.size))
            block_size = max(1, int(ctx.params.get("transport_block_size_bits") or 1024))
            scored_bit_count = max(int(reference_bits.size), int(candidate_bits.size))
            block_count = int(math.ceil(float(scored_bit_count) / float(block_size))) if scored_bit_count else 0
            block_error_count = 0
            delivered_bits = 0
            for block_index in range(block_count):
                start = block_index * block_size
                stop = min(start + block_size, scored_bit_count)
                reference_block = reference_bits[start : min(stop, int(reference_bits.size))]
                candidate_block = candidate_bits[start : min(stop, int(candidate_bits.size))]
                block_failed = int(reference_block.size) != int(candidate_block.size)
                block_compare_count = min(int(reference_block.size), int(candidate_block.size))
                if block_compare_count and np.any(
                    reference_block[:block_compare_count] != candidate_block[:block_compare_count]
                ):
                    block_failed = True
                if block_failed:
                    block_error_count += 1
                else:
                    delivered_bits += int(reference_block.size)
            block_error_rate = float(block_error_count) / float(block_count) if block_count else 0.0
            delivery_success = bool(block_error_count == 0 and int(reference_bits.size) == int(candidate_bits.size))
            scheduled_resources = int(
                tx_symbol_metadata.get("adaptive_scheduled_resource_element_count") or tx_symbols.size
            )
            data_resources = int(
                tx_symbol_metadata.get("adaptive_data_resource_element_count") or tx_symbols.size
            )
            payload_bits = int(reference_bits.size)
            tx_energy = float(np.sum(np.abs(tx_symbols.astype(np.complex64, copy=False)) ** 2))
            offered_payload_rate = (
                float(payload_bits) / float(scheduled_resources) if scheduled_resources else 0.0
            )
            achieved_goodput = (
                float(delivered_bits) / float(scheduled_resources) if scheduled_resources else 0.0
            )
            energy_efficiency = float(delivered_bits) / tx_energy if tx_energy > 1e-12 else 0.0
            resource_utilization = (
                float(data_resources) / float(scheduled_resources) if scheduled_resources else 0.0
            )
            delivery_metrics = {
                "channel.offered_payload_bits_per_resource_element": offered_payload_rate,
                "channel.achieved_payload_goodput_bits_per_resource_element": achieved_goodput,
                "channel.payload_delivery_block_error_rate": block_error_rate,
                "channel.payload_delivery_block_count": block_count,
                "channel.payload_delivery_block_error_count": block_error_count,
                "channel.payload_delivery_success": 1 if delivery_success else 0,
                "channel.delivered_payload_bit_count": delivered_bits,
                "channel.scheduled_resource_element_count": scheduled_resources,
                "channel.data_bearing_resource_element_count": data_resources,
                "channel.resource_element_utilization": resource_utilization,
                "channel.tx_energy_for_payload": tx_energy,
                "channel.payload_energy_efficiency_bits_per_normalized_energy": energy_efficiency,
                "channel.goodput_to_theoretical_shannon_efficiency_ratio": (
                    achieved_goodput / mean_spectral_efficiency
                    if mean_spectral_efficiency > 1e-12
                    else 0.0
                ),
            }
            if delivered_bits:
                delivery_metrics["channel.tx_energy_per_delivered_payload_bit"] = (
                    tx_energy / float(delivered_bits)
                )
            delivery_report = {
                "payload_bit_count": payload_bits,
                "candidate_bit_count": int(candidate_bits.size),
                "payload_error_count": payload_error_count,
                "payload_delivery_success": delivery_success,
                "transport_block_size_bits": block_size,
                "transport_block_count": block_count,
                "transport_block_error_count": block_error_count,
                "transport_block_error_rate": block_error_rate,
                "delivered_payload_bit_count": delivered_bits,
                "scheduled_resource_element_count": scheduled_resources,
                "data_bearing_resource_element_count": data_resources,
                "resource_element_utilization": resource_utilization,
                "tx_energy_for_payload": tx_energy,
                "offered_payload_bits_per_resource_element": offered_payload_rate,
                "achieved_payload_goodput_bits_per_resource_element": achieved_goodput,
                "payload_energy_efficiency_bits_per_normalized_energy": energy_efficiency,
                "tx_energy_per_delivered_payload_bit": (
                    tx_energy / float(delivered_bits) if delivered_bits else None
                ),
            }
        metrics = {
            "resource.theoretical_shannon_sum_bits_per_ofdm_symbol": float(np.mean(sum_rates)),
            "resource.theoretical_shannon_spectral_efficiency_bps_hz": mean_spectral_efficiency,
            "resource.theoretical_min_state_spectral_efficiency_bps_hz": float(np.min(spectral_efficiencies)),
            "resource.theoretical_p05_state_spectral_efficiency_bps_hz": float(
                np.percentile(spectral_efficiencies, 5.0)
            ),
            "resource.theoretical_outage_probability": float(
                np.mean(spectral_efficiencies < outage_target)
            ),
            "resource.theoretical_outage_target_spectral_efficiency_bps_hz": outage_target,
            "resource.water_filling_oracle_spectral_efficiency_bps_hz": mean_oracle_spectral_efficiency,
            "resource.water_filling_optimality_gap_bps_hz": optimality_gap,
            "resource.water_filling_relative_optimality_gap": relative_optimality_gap,
            "resource.water_filling_power_rmse": oracle_power_rmse,
            "resource.water_filling_power_normalized_rmse": oracle_power_normalized_rmse,
            "resource.water_filling_kkt_normalized_residual": float(np.mean(kkt_residuals)),
            "resource.average_transmit_power_budget": target_power,
            "resource.power_constraint.max_abs_error": float(np.max(power_error)),
            "resource.power_constraint.max_relative_error": float(np.max(power_error) / power_scale),
            "resource.power_constraint.max_negative_violation": 0.0,
            "resource.active_resource_element_fraction": float(np.mean(active_mask)),
            "resource.allocation_power_coefficient_of_variation": float(np.mean(allocation_cv)),
            "resource.allocated_power_weighted_channel_gain_lift": float(np.mean(gain_weighted_lift)),
            "channel.reference_snr_db": _snr_db_from_signal_and_noise(
                1.0, noise_variance
            ),
            "channel.noise_variance": noise_variance,
            "task.score": mean_spectral_efficiency,
            **delivery_metrics,
        }
        preview = allocation_metadata.get("resource_allocation_preview")
        rows = [
            {
                "state": int(index),
                "theoretical_shannon_sum_bits_per_ofdm_symbol": float(sum_rates[index]),
                "theoretical_shannon_spectral_efficiency_bps_hz": float(spectral_efficiencies[index]),
                "active_subcarrier_count": int(np.count_nonzero(active_mask[index])),
                "allocation_power_coefficient_of_variation": float(allocation_cv[index]),
                "allocated_power_weighted_channel_gain_lift": float(gain_weighted_lift[index]),
            }
            for index in range(min(8, int(sum_rates.size)))
        ]
        report = {
            "schema_version": 1,
            "metric_family": "resource_allocation",
            "rows": rows,
            "metrics": metrics,
            "metadata": {
                "metric_scope": (
                    "measured_allocation_aware_ofdm_payload_delivery_plus_parallel_gaussian_reference"
                    if delivery_metrics
                    else "parallel_gaussian_channel_oracle_not_achieved_qpsk_payload_throughput"
                ),
                "rate_formula": "log2(1 + |h[k]|^2 * p[k] / noise_variance)",
                "water_filling_oracle_formula": "p_star[k] = max(mu - noise_variance/|h[k]|^2, 0), with sum_k(p_star[k]) = total_power",
                "spectral_efficiency_reduction": "mean_over_equal_bandwidth_subcarriers_then_mean_over_channel_states",
                "outage_definition": "fraction_of_channel_states_with_mean_subcarrier_shannon_spectral_efficiency_below_declared_target",
                "outage_target_spectral_efficiency_bps_hz": outage_target,
                "water_filling_optimality_gap_definition": "mean_oracle_shannon_spectral_efficiency_minus_policy_shannon_spectral_efficiency",
                "task_score_definition": "mean_parallel_gaussian_channel_shannon_spectral_efficiency_bps_hz",
                "transport_diagnostics": "finite_modulation_bit_loading_BER_BLER_and_goodput_are_reported_but_are_not_the_water_filling_optimization_objective",
                "water_filling_kkt_residual_definition": "mean_state_max_normalized_stationarity_or_inactive_complementarity_residual",
                "active_power_threshold": active_threshold,
                "power_budget_authority": "channel_state.average_power_budget",
                "noise_authority": "channel_state.noise_variance",
                "sum_power_feasibility_tolerance": feasibility_tolerance,
                "allocation_power_coefficient_of_variation_formula": "std_k(p[k]) / mean_k(p[k])",
                "allocated_power_weighted_channel_gain_lift_formula": "(sum_k(|h[k]|^2*p[k]) / sum_k(p[k])) / mean_k(|h[k]|^2) - 1",
                "state_metadata": state_metadata,
                "allocation_metadata": allocation_metadata,
                "resource_allocation_preview": preview,
                "delivery_measurement": delivery_report,
            },
            "resource_allocation_preview": preview,
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics=metrics,
            metadata=report,
        )


class IdentitySymbolLinkOperation(Operation):
    id = "channel.identity_symbol_link"
    name = "Disabled channel identity symbol link"
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Disabled continuous-symbol channels are identity maps in differentiable export.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    equivalence = {"type": "exact", "reason": "Identity PHY materializations must return the same continuous symbols."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    input_kinds = {
        "symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    params_schema = object_schema({"label": {"type": "string", "default": "identity_symbol_channel"}})

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        symbols = _require_canonical_symbols(raw_symbols, ctx.step_id)
        label = _metric_label(str(ctx.params.get("label") or "identity_symbol_channel"))
        power_unit = str(metadata.get("power_unit") or "normalized")
        rx_power_preview, rx_power_average, rx_power_total_energy = _symbol_power_trace_metadata(
            symbols, "rx_power", unit=power_unit
        )
        rx_power_preview["kind"] = "rx_output_power_trace"
        rx_antenna_power_preview = dict(rx_power_preview)
        rx_antenna_power_preview["kind"] = "rx_antenna_power_trace"
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "channel_mode": "disabled_identity",
                "boundary_contract": "channel.symbols",
                "identity_channel": True,
                "symbol_count": int(symbols.size),
                "channel_use_count": int(symbols.size),
                "symbol_storage": "complex64",
                "rx_power_preview": rx_power_preview,
                "rx_power_average": rx_power_average,
                "rx_power_total_energy": rx_power_total_energy,
                "rx_output_power_preview": rx_power_preview,
                "rx_output_power_average": rx_power_average,
                "rx_output_power_total_energy": rx_power_total_energy,
                "rx_antenna_power_preview": rx_antenna_power_preview,
                "rx_antenna_power_average": rx_power_average,
                "rx_antenna_power_total_energy": rx_power_total_energy,
                "rx_power_unit": power_unit,
                "channel_equalized": False,
            }
        )
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics={
                "channel.identity_link.symbol_count": int(symbols.size),
                "channel.channel_use_count": int(symbols.size),
                "channel.rx_power.average": rx_power_average,
                "channel.rx_power.total_energy": rx_power_total_energy,
                "channel.rx_output_power.average": rx_power_average,
                "channel.rx_output_power.total_energy": rx_power_total_energy,
                "channel.rx_antenna_power.average": rx_power_average,
                "channel.rx_antenna_power.total_energy": rx_power_total_energy,
                "channel.disabled": 1,
            },
            metadata={
                "label": label,
                "symbol_count": int(symbols.size),
                "channel_mode": "disabled_identity",
                "rx_power_average": rx_power_average,
                "rx_power_total_energy": rx_power_total_energy,
                "rx_antenna_power_average": rx_power_average,
                "rx_antenna_power_total_energy": rx_power_total_energy,
                "rx_output_power_average": rx_power_average,
                "rx_output_power_total_energy": rx_power_total_energy,
                "rx_power_unit": power_unit,
                "channel_equalized": False,
            },
        )


class SymbolCountMatchOperation(Operation):
    id = "channel.symbol_count_match"
    name = "Channel symbol-count consistency check"
    input_kinds = {
        "reference": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ],
        "candidate": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema({"label": {"type": "string", "default": "symbol_channel_io"}})

    def run(self, ctx: OperationContext) -> OperationResult:
        reference_artifact = ctx.require_input("reference")
        candidate_artifact = ctx.require_input("candidate")
        reference, reference_metadata = _load_symbols_raw(reference_artifact.path, reference_artifact.metadata)
        candidate, candidate_metadata = _load_symbols_raw(candidate_artifact.path, candidate_artifact.metadata)
        label = _metric_label(str(ctx.params.get("label") or "symbol_channel_io"))
        reference = _require_canonical_symbols(reference, "%s/%s/reference" % (ctx.step_id, label))
        candidate = _require_canonical_symbols(candidate, "%s/%s/candidate" % (ctx.step_id, label))
        reference_count = int(reference.size)
        candidate_count = int(candidate.size)
        delta = candidate_count - reference_count
        if delta != 0:
            raise OperationError(
                "Channel symbol-count mismatch at %s: reference has %d symbols, candidate has %d symbols"
                % (label, reference_count, candidate_count)
            )
        report = {
            "schema_version": 1,
            "metric_family": "symbol_count_match",
            "label": label,
            "reference_symbol_count": reference_count,
            "candidate_symbol_count": candidate_count,
            "length_delta": delta,
            "reference_fixed_point": reference_metadata.get("fixed_point_label") or reference_metadata.get("fixed_point"),
            "candidate_fixed_point": candidate_metadata.get("fixed_point_label") or candidate_metadata.get("fixed_point"),
            "canonical_dtype": "complex64",
            "symbol_storage": "complex64",
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics={
                "channel.fixed.%s.symbol_count_match" % label: 1,
                "channel.fixed.%s.reference_symbol_count" % label: reference_count,
                "channel.fixed.%s.candidate_symbol_count" % label: candidate_count,
                "channel.fixed.%s.length_delta" % label: delta,
            },
            metadata=report,
        )


class IdentityBitLinkOperation(Operation):
    id = "channel.identity_link"
    name = "Disabled channel identity bit link"
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "This is a discrete bitstream bypass. Disabled physical channels for differentiable symbol paths should use channel.identity_symbol_link.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    equivalence = {"type": "exact", "reason": "Bit-perfect bypass materializations must preserve unpacked uint8 bits exactly."}
    formats = {"artifact": "npz", "tensor": "none"}
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"bits": "channel.bits.numpy"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "identity_channel"},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(input_artifact.path, input_artifact.metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = _require_canonical_bits(raw_bits, ctx.step_id, backend)
        label = _metric_label(str(ctx.params.get("label") or "identity_channel"))
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "channel_mode": "disabled_identity",
                "boundary_contract": "channel.bits",
                "identity_channel": True,
                "bit_count": int(bits.size),
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=bits, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.bits.numpy", path, output_metadata)},
            metrics={
                "channel.identity_link.bit_count": int(bits.size),
                "channel.identity_link.byte_count": int(math.ceil(float(bits.size) / 8.0)),
                "channel.disabled": 1,
            },
            metadata={"label": label, "bit_count": int(bits.size), "channel_mode": "disabled_identity", "data_plane_backend": selected_backend},
        )


class IdentityModulateOperation(Operation):
    id = "modulation.identity_modulate"
    name = "Identity PHY bit-to-symbol mapper"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.coded_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": False,
        "exportable": True,
        "reason": "Artifact execution maps discrete uint8 bits to 0/1 complex symbols. Differentiable export may materialize this as an identity over continuous logits/symbols, but gradients through hard bits are surrogate-only.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    equivalence = {"type": "exact", "reason": "Identity PHY modulation stores each bit as one finite complex symbol with real value 0 or 1."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema({"label": {"type": "string", "default": "identity_phy_modulator"}, **_backend_param()})

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(input_artifact.path, input_artifact.metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = _require_canonical_bits(raw_bits, ctx.step_id, backend)
        label = _metric_label(str(ctx.params.get("label") or "identity_phy_modulator"))
        symbols = bits.astype(np.float32, copy=False).astype(np.complex64, copy=False)
        source_counts = _source_item_stage_bit_counts(metadata, int(bits.size))
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "modulation": "identity_phy",
                "identity_phy": True,
                "identity_phy_stage": "modulator",
                "identity_phy_label": label,
                "boundary_contract": "channel.symbols",
                "bit_count": int(bits.size),
                "padded_bit_count": int(bits.size),
                "symbol_count": int(symbols.size),
                "channel_use_count": int(symbols.size),
                "bits_per_symbol": 1,
                "symbol_storage": "complex64",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "data_plane_backend": selected_backend,
            }
        )
        source_item_uses_per_pixel: List[float] = []
        if source_counts is not None:
            output_metadata.update(
                {
                    "source_item_symbol_counts": [
                        int(item) for item in source_counts
                    ],
                    "source_item_modulator_input_bit_counts": [
                        int(item) for item in source_counts
                    ],
                    "source_item_padded_bit_counts": [
                        int(item) for item in source_counts
                    ],
                }
            )
            source_item_pixels = _source_item_pixel_counts(
                metadata, len(source_counts)
            )
            if source_item_pixels is not None:
                source_item_uses_per_pixel = [
                    float(use_count) / float(pixel_count)
                    for use_count, pixel_count in zip(
                        source_counts, source_item_pixels
                    )
                ]
                output_metadata["source_item_pixel_counts"] = source_item_pixels
                output_metadata[
                    "source_item_channel_uses_per_pixel"
                ] = source_item_uses_per_pixel
                output_metadata["max_source_item_channel_uses_per_pixel"] = max(
                    source_item_uses_per_pixel
                )
        pixel_count = _pixel_count_from_metadata(metadata)
        if pixel_count:
            output_metadata.update(
                {
                    "pixel_count": pixel_count,
                    "channel_uses_per_pixel": float(symbols.size) / float(pixel_count),
                    "rate_coded_bpp": float(bits.size) / float(pixel_count),
                    "rate_padded_bpp": float(bits.size) / float(pixel_count),
                }
            )
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        metrics = {
            "channel.coded_bit_count": int(bits.size),
            "channel.transmitted_bit_count": int(bits.size),
            "channel.symbol_count": int(symbols.size),
            "channel.channel_use_count": int(symbols.size),
            "channel.bits_per_symbol": 1,
            "channel.disabled": 1,
            "channel.identity_phy": 1,
        }
        if pixel_count:
            metrics.update(
                {
                    "rate.coded_bpp": float(bits.size) / float(pixel_count),
                    "rate.padded_bpp": float(bits.size) / float(pixel_count),
                    "channel.uses_per_pixel": float(symbols.size) / float(pixel_count),
                }
            )
        if source_item_uses_per_pixel:
            metrics["channel.max_source_item_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics=metrics,
            metadata={"label": label, "modulation": "identity_phy", "bit_count": int(bits.size), "data_plane_backend": selected_backend},
        )


class IdentityDemodulateOperation(Operation):
    id = "demodulation.identity_demodulate"
    name = "Identity PHY symbol-to-bit mapper"
    input_kinds = {"rx_symbols": ["channel.symbols.complex_numpy", "channel.rx_symbols.complex_numpy"]}
    output_kinds = {
        "bits": "channel.demod_bits.numpy",
        "llr": "channel.llr.numpy",
    }
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": False,
        "exportable": True,
        "reason": "Artifact execution thresholds 0/1 identity PHY symbols back to bits. Differentiable export may keep a continuous soft/logit materialization until a loss, but hard bit decisions are surrogate-only.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    equivalence = {"type": "exact", "reason": "When paired with identity modulation and identity channel, demodulation recovers the original uint8 bit vector exactly."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema({"label": {"type": "string", "default": "identity_phy_demodulator"}, **_backend_param()})

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("rx_symbols")
        raw_symbols, metadata = _load_symbols_raw(input_artifact.path, input_artifact.metadata)
        symbols = _require_canonical_symbols(raw_symbols, ctx.step_id)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bit_count = int(metadata.get("bit_count") or symbols.size)
        if bit_count > int(symbols.size):
            raise OperationError(
                "Identity PHY demodulator expected at least %d symbols for %d bits, got %d"
                % (bit_count, bit_count, int(symbols.size))
            )
        bits = (symbols[:bit_count].real >= 0.5).astype(np.uint8, copy=False)
        bits, selected_backend = _require_canonical_bits(bits, ctx.step_id, backend)
        llr = ((symbols[:bit_count].real.astype(np.float32, copy=False) - 0.5) * 2.0).astype(np.float32, copy=False)
        label = _metric_label(str(ctx.params.get("label") or "identity_phy_demodulator"))
        bit_metadata = dict(metadata)
        bit_metadata.update(
            {
                "demodulation": "identity_phy",
                "identity_phy": True,
                "identity_phy_stage": "demodulator",
                "identity_phy_label": label,
                "boundary_contract": "channel.bits",
                "bit_count": int(bits.size),
                "bit_role": "demodulated",
                "llr_kind": "identity_phy_signed_distance",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        bits_path = ctx.output_path("bits", ".npz")
        llr_path = ctx.output_path("llr", ".npz")
        np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(bit_metadata))
        np.savez_compressed(llr_path, llr=llr, metadata_json=json.dumps(bit_metadata))
        return OperationResult(
            outputs={
                "bits": artifact("channel.demod_bits.numpy", bits_path, bit_metadata),
                "llr": artifact("channel.llr.numpy", llr_path, bit_metadata),
            },
            metrics={
                "channel.demod_bit_count": int(bits.size),
                "channel.disabled": 1,
                "channel.identity_phy": 1,
            },
            metadata={"label": label, "demodulation": "identity_phy", "bit_count": int(bits.size), "data_plane_backend": selected_backend},
        )


_CRC32_PACKET_PROTOCOL_V4 = "noema.source_item_crc32.v4"
_CRC32_PACKET_HEADER_BITS = 160
_CRC32_PACKET_HEADER_CRC_BITS = 32
_CRC32_PACKET_HEADER_REPETITION_FACTOR = 1
_CRC32_PACKET_CRC_BITS = 32
_U32_MAX = (1 << 32) - 1


def _build_crc32_packet_contract(
    source_item_payload_bit_counts: List[int], packet_payload_bits: int
) -> JsonDict:
    """Build the canonical, content-addressed packet ownership contract."""

    packet_payload_bits = int(packet_payload_bits)
    item_counts = [int(value) for value in source_item_payload_bit_counts]
    if packet_payload_bits < 8 or packet_payload_bits > _U32_MAX:
        raise OperationError("CRC32 packet payload length is outside the uint32 protocol range")
    if len(item_counts) > _U32_MAX:
        raise OperationError("CRC32 source-item count exceeds the uint32 protocol range")
    if any(value <= 0 or value > _U32_MAX for value in item_counts):
        raise OperationError(
            "CRC32 source-item payload lengths must be positive uint32 values"
        )

    packet_total_bits = (
        _CRC32_PACKET_HEADER_BITS
        + _CRC32_PACKET_HEADER_CRC_BITS
        + packet_payload_bits
        + _CRC32_PACKET_CRC_BITS
    )
    ordered_packet_layout: List[JsonDict] = []
    source_item_packet_counts: List[int] = []
    for source_item_index, item_count in enumerate(item_counts):
        item_packet_counts = _packet_payload_counts(item_count, packet_payload_bits)
        source_item_packet_counts.append(len(item_packet_counts))
        item_bit_offset = 0
        for valid_payload_bits in item_packet_counts:
            packet_index = len(ordered_packet_layout)
            ordered_packet_layout.append(
                {
                    "packet_index": packet_index,
                    "packet_bit_offset": packet_index * packet_total_bits,
                    "source_item_index": source_item_index,
                    "source_item_count": len(item_counts),
                    "item_bit_offset": item_bit_offset,
                    "valid_payload_bits": int(valid_payload_bits),
                    "item_total_payload_bits": item_count,
                }
            )
            item_bit_offset += int(valid_payload_bits)

    return {
        "schema_version": 1,
        "protocol": _CRC32_PACKET_PROTOCOL_V4,
        "packet_payload_bits": packet_payload_bits,
        "packet_header_logical_bits": _CRC32_PACKET_HEADER_BITS,
        "packet_header_crc_logical_bits": _CRC32_PACKET_HEADER_CRC_BITS,
        "packet_header_repetition_factor": _CRC32_PACKET_HEADER_REPETITION_FACTOR,
        "packet_header_bits": _CRC32_PACKET_HEADER_BITS,
        "packet_header_crc_bits": _CRC32_PACKET_HEADER_CRC_BITS,
        "packet_crc_bits": _CRC32_PACKET_CRC_BITS,
        "packet_total_bits": packet_total_bits,
        "packet_count": len(ordered_packet_layout),
        "source_item_count": len(item_counts),
        "source_item_payload_bit_counts": item_counts,
        "source_item_packet_counts": source_item_packet_counts,
        "ordered_packet_layout": ordered_packet_layout,
    }


def _validated_crc32_packet_contract(
    metadata: Mapping[str, Any], received_bit_count: int
) -> JsonDict:
    """Validate the immutable control-plane contract before using its layout."""

    raw_contract = metadata.get("packet_contract")
    if not isinstance(raw_contract, dict):
        raise OperationError("CRC32 v4 metadata is missing packet_contract")
    declared_sha256 = metadata.get("packet_contract_sha256")
    if not isinstance(declared_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", declared_sha256
    ) is None:
        raise OperationError("CRC32 v4 metadata has an invalid packet_contract_sha256")
    try:
        actual_sha256 = canonical_json_sha256(raw_contract)
    except (TypeError, ValueError) as exc:
        raise OperationError("CRC32 v4 packet_contract is not canonical JSON") from exc
    if actual_sha256 != declared_sha256:
        raise OperationError("CRC32 v4 packet_contract hash does not match its contents")

    raw_item_counts = raw_contract.get("source_item_payload_bit_counts")
    raw_packet_payload_bits = raw_contract.get("packet_payload_bits")
    if not isinstance(raw_item_counts, list):
        raise OperationError("CRC32 v4 packet_contract has no source-item lengths")
    if type(raw_packet_payload_bits) is not int:
        raise OperationError("CRC32 v4 packet_contract packet_payload_bits must be an integer")
    if any(type(value) is not int for value in raw_item_counts):
        raise OperationError("CRC32 v4 packet_contract source-item lengths must be integers")
    canonical_contract = _build_crc32_packet_contract(
        raw_item_counts, raw_packet_payload_bits
    )
    if raw_contract != canonical_contract:
        raise OperationError("CRC32 v4 packet_contract is not the canonical packet layout")

    packet_count = int(canonical_contract["packet_count"])
    packet_total_bits = int(canonical_contract["packet_total_bits"])
    expected_bit_count = packet_count * packet_total_bits
    if int(received_bit_count) != expected_bit_count:
        raise OperationError(
            "Received framed bit count %d does not match the v4 packet contract (%d)"
            % (int(received_bit_count), expected_bit_count)
        )

    # These fields are redundant conveniences. If a producer retained them,
    # they may not contradict the hash-bound contract; they are not required
    # for recovery.
    redundant_fields = {
        "packet_protocol": canonical_contract["protocol"],
        "packet_count": packet_count,
        "packet_payload_bits": canonical_contract["packet_payload_bits"],
        "packet_header_bits": canonical_contract["packet_header_bits"],
        "packet_header_crc_bits": canonical_contract["packet_header_crc_bits"],
        "packet_header_logical_bits": canonical_contract[
            "packet_header_logical_bits"
        ],
        "packet_header_crc_logical_bits": canonical_contract[
            "packet_header_crc_logical_bits"
        ],
        "packet_header_repetition_factor": canonical_contract[
            "packet_header_repetition_factor"
        ],
        "packet_crc_bits": canonical_contract["packet_crc_bits"],
        "packet_total_bits": packet_total_bits,
        "packet_source_item_counts": canonical_contract[
            "source_item_packet_counts"
        ],
        "source_item_count": canonical_contract["source_item_count"],
        "source_item_payload_bit_counts": canonical_contract[
            "source_item_payload_bit_counts"
        ],
        "source_item_framed_bit_counts": [
            int(count) * packet_total_bits
            for count in canonical_contract["source_item_packet_counts"]
        ],
    }
    for field, expected in redundant_fields.items():
        if field in metadata and metadata[field] != expected:
            raise OperationError(
                "CRC32 v4 metadata field %s contradicts packet_contract" % field
            )
    return canonical_contract


def _crc32_packet_header(layout: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            _u32_to_bits(int(layout["source_item_index"])),
            _u32_to_bits(int(layout["source_item_count"])),
            _u32_to_bits(int(layout["item_bit_offset"])),
            _u32_to_bits(int(layout["valid_payload_bits"])),
            _u32_to_bits(int(layout["item_total_payload_bits"])),
        ]
    ).astype(np.uint8, copy=False)


class Crc32PacketizeOperation(Operation):
    id = "channel.packetize_crc32"
    name = "Packetize payload bits with CRC32"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "CRC packetization is a hard digital transport operation.",
    }
    params_schema = object_schema(
        {
            "packet_payload_bits": {
                "type": "integer",
                "default": 4096,
                "minimum": 8,
                "description": "Payload bits protected by one CRC32. The final packet is padded but its valid length is checked.",
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(input_artifact.path, input_artifact.metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = _require_canonical_bits(raw_bits, ctx.step_id, backend)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "CRC32 packetizer input at %s" % ctx.step_id,
        )
        packet_payload_bits = int(ctx.params.get("packet_payload_bits") or 4096)
        if packet_payload_bits < 8:
            raise OperationError("packet_payload_bits must be at least 8")

        payload_count = int(bits.size)
        raw_item_counts = metadata.get("source_item_payload_bit_counts")
        if isinstance(raw_item_counts, list):
            item_counts = [int(item) for item in raw_item_counts]
        elif capture_layout is not None:
            item_counts = [
                int(capture_layout.elements_per_record)
            ] * int(capture_layout.count)
        else:
            item_counts = [payload_count] if payload_count else []
        if any(item <= 0 for item in item_counts) or sum(item_counts) != payload_count:
            raise OperationError(
                "source_item_payload_bit_counts must be positive and sum to the payload bit count"
            )
        declared_item_count = metadata.get("source_item_count")
        if declared_item_count is not None and int(declared_item_count) != len(item_counts):
            raise OperationError(
                "source_item_count does not match source_item_payload_bit_counts"
            )
        if capture_layout is not None and (
            len(item_counts) != capture_layout.count
            or any(
                int(item) != capture_layout.elements_per_record
                for item in item_counts
            )
        ):
            raise OperationError(
                "CRC32 packetizer source-item metadata conflicts with the explicit "
                "capture-record layout"
            )

        # Protocol v4 keeps the in-band header compact and moves ownership,
        # cardinality, exact lengths, and ordering into a content-addressed
        # control-plane contract. The receiver validates any recoverable header
        # against this contract but never needs a header to allocate an item.
        packet_contract = _build_crc32_packet_contract(
            item_counts, packet_payload_bits
        )
        packet_contract_sha256 = canonical_json_sha256(packet_contract)
        packet_header_bits = int(packet_contract["packet_header_bits"])
        packet_header_crc_bits = int(packet_contract["packet_header_crc_bits"])
        packet_crc_bits = int(packet_contract["packet_crc_bits"])
        packet_total_bits = int(packet_contract["packet_total_bits"])
        packet_counts_by_item = list(packet_contract["source_item_packet_counts"])
        source_item_payload_offsets: List[int] = []
        payload_offset = 0
        for item_count in item_counts:
            source_item_payload_offsets.append(payload_offset)
            payload_offset += item_count

        packets: List[np.ndarray] = []
        for layout in packet_contract["ordered_packet_layout"]:
            source_item_index = int(layout["source_item_index"])
            item_bit_offset = int(layout["item_bit_offset"])
            valid_count = int(layout["valid_payload_bits"])
            absolute_offset = (
                source_item_payload_offsets[source_item_index] + item_bit_offset
            )
            segment = bits[
                absolute_offset : absolute_offset + valid_count
            ].astype(np.uint8, copy=False)
            header = _crc32_packet_header(layout)
            header_crc = _u32_to_bits(_crc32_bits(header, int(header.size)))
            padded = np.zeros((packet_payload_bits,), dtype=np.uint8)
            padded[:valid_count] = segment
            protected = np.concatenate([header, header_crc, padded]).astype(
                np.uint8, copy=False
            )
            crc_value = _crc32_bits(protected, int(protected.size))
            packets.append(np.concatenate([protected, _u32_to_bits(crc_value)]))
        packetized = (
            np.concatenate(packets).astype(np.uint8, copy=False)
            if packets
            else np.zeros((0,), dtype=np.uint8)
        )
        packet_count = int(packet_contract["packet_count"])
        framing_header_bit_count = (
            packet_header_bits + packet_header_crc_bits
        ) * packet_count
        header_crc_overhead_bit_count = packet_header_crc_bits * packet_count
        crc_overhead_bit_count = packet_crc_bits * packet_count
        padding_bit_count = packet_payload_bits * packet_count - payload_count
        pixel_count = _pixel_count_from_metadata(metadata)

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "protected_digital_baseline": "crc32_packetized",
                "packetizer": "crc32",
                "packetized": True,
                "packet_protocol": _CRC32_PACKET_PROTOCOL_V4,
                "packet_contract": packet_contract,
                "packet_contract_sha256": packet_contract_sha256,
                "packet_count": int(packet_count),
                "packet_payload_bits": int(packet_payload_bits),
                "packet_header_bits": packet_header_bits,
                "packet_header_crc_bits": packet_header_crc_bits,
                "packet_header_logical_bits": _CRC32_PACKET_HEADER_BITS,
                "packet_header_crc_logical_bits": _CRC32_PACKET_HEADER_CRC_BITS,
                "packet_header_repetition_factor": _CRC32_PACKET_HEADER_REPETITION_FACTOR,
                "packet_header_schema": "source_item_index_u32,source_item_count_u32,item_bit_offset_u32,valid_payload_bits_u32,item_total_payload_bits_u32,header_crc32",
                "packet_crc_bits": packet_crc_bits,
                "packet_total_bits": int(packet_total_bits),
                "packet_integrity_scope": "header_and_fixed_payload_area_including_padding",
                "packet_crc_reference": "received_crc_only",
                "packet_source_item_counts": packet_counts_by_item,
                "source_item_framed_bit_counts": [
                    int(count * packet_total_bits)
                    for count in packet_counts_by_item
                ],
                "source_item_count": len(item_counts),
                "source_item_payload_bit_counts": item_counts,
                "original_payload_bit_count": int(payload_count),
                "payload_bit_count": int(payload_count),
                "framing_header_bit_count": int(framing_header_bit_count),
                "header_crc_overhead_bit_count": int(
                    header_crc_overhead_bit_count
                ),
                "padding_bit_count": int(padding_bit_count),
                "framed_bit_count": int(packetized.size),
                "packetized_bit_count": int(packetized.size),
                "crc_overhead_bit_count": int(crc_overhead_bit_count),
                "bit_count": int(packetized.size),
                "bit_role": "packetized_payload",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(packetized.size),
            "CRC32 packetizer output at %s" % ctx.step_id,
        )
        if pixel_count:
            output_metadata.update(
                {
                    "rate_payload_bpp": float(payload_count) / float(pixel_count),
                    "rate_framed_bpp": float(packetized.size) / float(pixel_count),
                    "framing_padding_bpp": float(padding_bit_count) / float(pixel_count),
                }
            )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=packetized, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={
                "channel.payload_bit_count": int(payload_count),
                "channel.packet_count": int(packet_count),
                "channel.packet_payload_bits": int(packet_payload_bits),
                "channel.framing_header_bit_count": int(framing_header_bit_count),
                "channel.header_crc_overhead_bit_count": int(
                    header_crc_overhead_bit_count
                ),
                "channel.padding_bit_count": int(padding_bit_count),
                "channel.framed_bit_count": int(packetized.size),
                "channel.packetized_bit_count": int(packetized.size),
                "channel.crc_overhead_bit_count": int(crc_overhead_bit_count),
                **(
                    {
                        "rate.payload_bpp": float(payload_count) / float(pixel_count),
                        "rate.framed_bpp": float(packetized.size) / float(pixel_count),
                    }
                    if pixel_count
                    else {}
                ),
            },
            metadata={
                "packetizer": "crc32",
                "packet_count": int(packet_count),
                "packet_payload_bits": int(packet_payload_bits),
                "packet_protocol": _CRC32_PACKET_PROTOCOL_V4,
                "packet_contract_sha256": packet_contract_sha256,
                "packetized_bit_count": int(packetized.size),
                "data_plane_backend": selected_backend,
            },
        )


class Crc32PacketizeV2Operation(Crc32PacketizeOperation):
    """Packetize payload bits while preserving the framed-bit semantic kind.

    The legacy operation remains available under ``channel.packetize_crc32``
    with its historical payload-bit output contract.  This versioned operation
    reuses the same packet protocol and bytes but exposes the result as framed
    bits so strict accounting ports cannot accept a generic bit checkpoint.
    """

    id = "channel.packetize_crc32.v2"
    name = "Packetize payload bits with CRC32 (framed-bit contract v2)"
    input_kinds = {"bits": ["channel.payload_bits.numpy"]}
    output_kinds = {"bits": "channel.framed_bits.numpy"}

    def run(self, ctx: OperationContext) -> OperationResult:
        result = super().run(ctx)
        legacy_output = result.outputs["bits"]
        return OperationResult(
            outputs={
                "bits": artifact(
                    "channel.framed_bits.numpy",
                    legacy_output.path,
                    legacy_output.metadata,
                )
            },
            metrics=result.metrics,
            metadata=result.metadata,
        )


class Crc32CheckOperation(Operation):
    id = "channel.crc32_check"
    name = "Check CRC32 packets and recover payload bits"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.bits.numpy", "channel.demod_bits.numpy"]}
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "CRC checking and packet erasure are hard non-differentiable decisions.",
    }
    params_schema = object_schema(
        {
            "on_decode_failure": {
                "type": "string",
                "default": "gray_image",
                "enum": ["gray_image", "erasure", "report_outage"],
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(
            input_artifact.path, input_artifact.metadata
        )
        backend = dataplane.normalize_backend(
            ctx.params.get("data_plane_backend", "auto")
        )
        bits, selected_backend = _require_canonical_bits(
            raw_bits, ctx.step_id, backend
        )
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "CRC32 checker input at %s" % ctx.step_id,
        )
        if not metadata.get("packetized"):
            raise OperationError(
                "CRC32 check requires packetized bits produced by "
                "channel.packetize_crc32"
            )
        if str(metadata.get("packet_protocol") or "") != _CRC32_PACKET_PROTOCOL_V4:
            raise OperationError(
                "CRC32 check requires packet protocol %s"
                % _CRC32_PACKET_PROTOCOL_V4
            )

        packet_contract = _validated_crc32_packet_contract(
            metadata, int(bits.size)
        )
        packet_payload_bits = int(packet_contract["packet_payload_bits"])
        packet_header_bits = int(packet_contract["packet_header_bits"])
        packet_header_crc_bits = int(packet_contract["packet_header_crc_bits"])
        packet_total_bits = int(packet_contract["packet_total_bits"])
        packet_count = int(packet_contract["packet_count"])
        item_counts = [
            int(value)
            for value in packet_contract["source_item_payload_bit_counts"]
        ]
        source_item_count = int(packet_contract["source_item_count"])
        framed_counts = [
            int(value)
            for value in list(
                metadata.get("source_item_framed_bit_counts") or []
            )
        ]
        if capture_layout is not None and (
            source_item_count != capture_layout.count
            or len(framed_counts) != capture_layout.count
            or any(
                count != capture_layout.elements_per_record
                for count in framed_counts
            )
        ):
            raise OperationError(
                "CRC32 checker packet contract conflicts with the explicit "
                "capture-record layout"
            )
        failure_policy = str(
            ctx.params.get("on_decode_failure") or "gray_image"
        )
        if failure_policy not in {"gray_image", "erasure", "report_outage"}:
            raise OperationError(
                "Unknown CRC32 on_decode_failure policy: %s" % failure_policy
            )
        failed_item_bit_policy = (
            "best_effort_received_payload"
            if failure_policy == "report_outage"
            else "zero_entire_item"
        )

        # Allocation and ownership come exclusively from the hash-bound
        # contract. A destroyed header therefore becomes a packet failure for a
        # known item, never an unknown-length output or a framing exception.
        recovered_rows = [
            np.zeros((item_count,), dtype=np.uint8)
            for item_count in item_counts
        ]
        passed_by_item = [0] * source_item_count
        failed_by_item = [0] * source_item_count
        passed = 0
        failed = 0
        unrecoverable_header_packets = 0
        header_contract_mismatches = 0
        for layout in packet_contract["ordered_packet_layout"]:
            source_item_index = int(layout["source_item_index"])
            item_bit_offset = int(layout["item_bit_offset"])
            valid_count = int(layout["valid_payload_bits"])
            start = int(layout["packet_bit_offset"])
            header = bits[start : start + packet_header_bits]
            header_crc_start = start + packet_header_bits
            header_crc_segment = bits[
                header_crc_start : header_crc_start + packet_header_crc_bits
            ]
            payload_start = header_crc_start + packet_header_crc_bits
            payload_segment = bits[
                payload_start : payload_start + packet_payload_bits
            ]
            crc_segment = bits[
                payload_start + packet_payload_bits : start + packet_total_bits
            ]

            received_header_crc = _bits_to_u32(header_crc_segment)
            computed_header_crc = _crc32_bits(header, int(header.size))
            header_crc_valid = bool(
                received_header_crc == computed_header_crc
            )
            expected_header = _crc32_packet_header(layout)
            header_matches_contract = bool(
                header_crc_valid
                and np.array_equal(header, expected_header)
            )
            received_crc = _bits_to_u32(crc_segment)
            protected = np.concatenate(
                [header, header_crc_segment, payload_segment]
            ).astype(np.uint8, copy=False)
            computed_crc = _crc32_bits(protected, int(protected.size))
            packet_crc_valid = bool(received_crc == computed_crc)
            packet_ok = bool(
                packet_crc_valid
                and header_crc_valid
                and header_matches_contract
            )

            if not header_crc_valid:
                unrecoverable_header_packets += 1
            elif not header_matches_contract:
                header_contract_mismatches += 1
            if packet_ok:
                recovered_rows[source_item_index][
                    item_bit_offset : item_bit_offset + valid_count
                ] = payload_segment[:valid_count]
                passed_by_item[source_item_index] += 1
                passed += 1
            else:
                if failure_policy == "report_outage":
                    recovered_rows[source_item_index][
                        item_bit_offset : item_bit_offset + valid_count
                    ] = payload_segment[:valid_count]
                failed_by_item[source_item_index] += 1
                failed += 1

        source_item_outage = [
            1 if failed_count else 0 for failed_count in failed_by_item
        ]
        if failure_policy in {"gray_image", "erasure"}:
            for item_index, item_failed in enumerate(source_item_outage):
                if item_failed:
                    recovered_rows[item_index].fill(0)

        recovered = (
            np.concatenate(recovered_rows).astype(np.uint8, copy=False)
            if recovered_rows
            else np.zeros((0,), dtype=np.uint8)
        )
        expected_payload_bit_count = sum(item_counts)
        if int(recovered.size) != expected_payload_bit_count:
            raise OperationError(
                "CRC32 v4 recovery violated its exact-length packet contract"
            )

        packet_success_rate = (
            float(passed) / float(packet_count) if packet_count else 1.0
        )
        packet_outage_rate = (
            float(failed) / float(packet_count) if packet_count else 0.0
        )
        failed_source_items = int(sum(source_item_outage))
        source_item_success_rate = (
            float(source_item_count - failed_source_items)
            / float(source_item_count)
            if source_item_count
            else 1.0
        )
        source_item_outage_rate = 1.0 - source_item_success_rate
        payload_outage = 1 if failed_source_items else 0

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "packetized": False,
                "packetizer": "crc32",
                "crc_checked": True,
                "packet_contract_validated": True,
                "crc_passed_packet_count": int(passed),
                "crc_failed_packet_count": int(failed),
                "packet_success_rate": packet_success_rate,
                "packet_outage_rate": packet_outage_rate,
                "source_item_success_rate": source_item_success_rate,
                "source_item_outage_rate": source_item_outage_rate,
                "source_item_outage": source_item_outage,
                "source_item_count": source_item_count,
                "source_item_payload_bit_counts": item_counts,
                "source_item_payload_byte_counts": [
                    int(count // 8) if count % 8 == 0 else 0
                    for count in item_counts
                ],
                "source_item_length_source": "immutable_packet_contract",
                "source_item_packet_pass_counts": passed_by_item,
                "source_item_packet_fail_counts": failed_by_item,
                "unrecoverable_header_packet_count": int(
                    unrecoverable_header_packets
                ),
                "header_contract_mismatch_packet_count": int(
                    header_contract_mismatches
                ),
                "unassigned_failed_packet_count": 0,
                "payload_outage": int(payload_outage),
                "packet_success": int(not payload_outage),
                "outage": int(payload_outage),
                "on_decode_failure": failure_policy,
                "failed_item_bit_policy": failed_item_bit_policy,
                "bit_count": int(recovered.size),
                "payload_bit_count": int(recovered.size),
                "bit_role": "payload",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(recovered.size),
            "CRC32 checker output at %s" % ctx.step_id,
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(
            path, bits=recovered, metadata_json=json.dumps(output_metadata)
        )
        return OperationResult(
            outputs={
                "bits": artifact(
                    "channel.payload_bits.numpy", path, output_metadata
                )
            },
            metrics={
                "channel.packet_success_rate": packet_success_rate,
                "channel.packet_outage_rate": packet_outage_rate,
                "channel.source_item_success_rate": source_item_success_rate,
                "channel.outage_rate": source_item_outage_rate,
                "channel.packet_success": int(not payload_outage),
                "channel.outage": int(payload_outage),
                "channel.packet_count": packet_count,
                "channel.crc_passed_packet_count": int(passed),
                "channel.crc_failed_packet_count": int(failed),
                "channel.failed_source_item_count": failed_source_items,
                "channel.source_item_count": source_item_count,
                "channel.unrecoverable_header_packet_count": int(
                    unrecoverable_header_packets
                ),
                "channel.header_contract_mismatch_packet_count": int(
                    header_contract_mismatches
                ),
                "channel.unassigned_failed_packet_count": 0,
                "channel.payload_bit_count": int(recovered.size),
            },
            metadata={
                "packetizer": "crc32",
                "packet_protocol": _CRC32_PACKET_PROTOCOL_V4,
                "packet_contract_sha256": metadata[
                    "packet_contract_sha256"
                ],
                "packet_success_rate": packet_success_rate,
                "source_item_success_rate": source_item_success_rate,
                "source_item_outage_rate": source_item_outage_rate,
                "outage": int(payload_outage),
                "on_decode_failure": failure_policy,
                "failed_item_bit_policy": failed_item_bit_policy,
                "data_plane_backend": selected_backend,
            },
        )


class CapacityOracleDigitalLinkOperation(Operation):
    id = "channel.capacity_oracle_digital_link"
    name = "Capacity-oracle protected digital AWGN link"
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"bits": "channel.bits.numpy"}
    differentiability = {
        "framework": "none",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "This is a theoretical protected-digital reference, not a differentiable physical link or implemented LDPC decoder.",
    }
    params_schema = object_schema(
        {
            "snr_db": {"type": "number", "default": 12.0},
            "channel": {"type": "string", "default": "awgn", "enum": ["awgn"]},
            "modulation": {"type": "string", "default": "qpsk", "enum": ["qpsk"]},
            "ldpc_rate": {"type": "number", "default": 0.5, "minimum": 0.01},
            "on_decode_failure": {
                "type": "string",
                "default": "gray_image",
                "enum": ["gray_image", "erasure", "report_outage"],
            },
            "label": {"type": "string", "default": "capacity_oracle_awgn"},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        raw_bits, metadata = _load_bits_raw(input_artifact.path, input_artifact.metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = _require_canonical_bits(raw_bits, ctx.step_id, backend)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "Capacity-oracle input at %s" % ctx.step_id,
        )
        snr_db = float(ctx.params.get("snr_db", 12.0))
        ldpc_rate = float(ctx.params.get("ldpc_rate", 0.5))
        if ldpc_rate <= 0.0 or ldpc_rate > 1.0:
            raise OperationError("Capacity-oracle digital link requires 0 < ldpc_rate <= 1")
        bits_per_symbol = 2
        payload_bits = int(bits.size)
        raw_item_counts = metadata.get("source_item_payload_bit_counts")
        item_counts = (
            [int(item) for item in raw_item_counts]
            if isinstance(raw_item_counts, list)
            else (
                [capture_layout.elements_per_record] * capture_layout.count
                if capture_layout is not None
                else [payload_bits]
            )
        )
        if (
            any(item <= 0 for item in item_counts)
            or sum(item_counts) != payload_bits
        ):
            raise OperationError(
                "Capacity-oracle source-item payload lengths are invalid"
            )
        if capture_layout is not None and (
            len(item_counts) != capture_layout.count
            or any(
                int(item) != capture_layout.elements_per_record
                for item in item_counts
            )
        ):
            raise OperationError(
                "Capacity-oracle source-item metadata conflicts with the explicit "
                "capture-record layout"
            )
        snr_linear = 10.0 ** (snr_db / 10.0)
        item_coded_bits: List[int] = []
        item_channel_uses: List[int] = []
        item_capacity_bits: List[float] = []
        item_success: List[bool] = []
        rx_rows: List[np.ndarray] = []
        offset = 0
        for item_count in item_counts:
            coded_count = int(math.ceil(float(item_count) / ldpc_rate))
            use_count = int(
                math.ceil(float(coded_count) / float(bits_per_symbol))
            )
            capacity_count = float(use_count) * math.log2(1.0 + snr_linear)
            success = bool(float(item_count) <= capacity_count + 1e-9)
            row = bits[offset : offset + item_count]
            rx_rows.append(
                row.astype(np.uint8, copy=True)
                if success
                else np.zeros_like(row, dtype=np.uint8)
            )
            item_coded_bits.append(coded_count)
            item_channel_uses.append(use_count)
            item_capacity_bits.append(capacity_count)
            item_success.append(success)
            offset += item_count
        coded_bits = int(sum(item_coded_bits))
        channel_uses = int(sum(item_channel_uses))
        padded_bits = int(channel_uses * bits_per_symbol)
        capacity_bits = float(sum(item_capacity_bits))
        success = bool(all(item_success))
        source_item_outage = [0 if item else 1 for item in item_success]
        source_item_success_rate = float(sum(item_success)) / float(len(item_success))
        source_item_outage_rate = 1.0 - source_item_success_rate
        failure_policy = str(ctx.params.get("on_decode_failure") or "gray_image")
        rx_bits = np.concatenate(rx_rows).astype(np.uint8, copy=False)
        pixel_count = _pixel_count_from_metadata(metadata)
        source_item_pixels = _source_item_pixel_counts(metadata, len(item_counts))
        source_item_uses_per_pixel = (
            [
                float(use_count) / float(item_pixel_count)
                for use_count, item_pixel_count in zip(
                    item_channel_uses, source_item_pixels
                )
            ]
            if source_item_pixels is not None
            else []
        )
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "protected_digital_baseline": "capacity_oracle",
                "protected_digital_baseline_label": "Capacity-oracle protected digital AWGN link",
                "protected_digital_note": "Theoretical per-source-item reference: each payload is delivered perfectly only when its source_bits <= its channel_uses * log2(1 + snr). This is not an implemented LDPC/CRC chain.",
                "channel": "awgn",
                "channel_mode": "capacity_oracle_awgn",
                "modulation": "qpsk",
                "bits_per_symbol": bits_per_symbol,
                "ldpc_rate": ldpc_rate,
                "snr_db": snr_db,
                "snr_linear": snr_linear,
                "source_bit_count": payload_bits,
                "payload_bit_count": int(metadata.get("payload_bit_count") or payload_bits),
                "framed_bit_count": int(metadata.get("framed_bit_count") or payload_bits),
                "coded_bit_count": coded_bits,
                "padded_bit_count": padded_bits,
                "transmitted_bit_count": padded_bits,
                "bit_count": int(rx_bits.size),
                "channel_use_count": channel_uses,
                "capacity_bits": capacity_bits,
                "capacity_oracle_accounting_scope": "source_item",
                "source_item_count": len(item_counts),
                "source_item_payload_bit_counts": item_counts,
                "source_item_coded_bit_counts": item_coded_bits,
                "source_item_channel_use_counts": item_channel_uses,
                "source_item_capacity_bits": item_capacity_bits,
                "source_item_outage": source_item_outage,
                "source_item_success_rate": source_item_success_rate,
                "source_item_outage_rate": source_item_outage_rate,
                "packet_success": int(success),
                "outage": int(not success),
                "on_decode_failure": failure_policy,
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        if source_item_uses_per_pixel:
            output_metadata["source_item_pixel_counts"] = source_item_pixels
            output_metadata[
                "source_item_channel_uses_per_pixel"
            ] = source_item_uses_per_pixel
            output_metadata["max_source_item_channel_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        if pixel_count:
            output_metadata["pixel_count"] = pixel_count
            output_metadata["rate_payload_bpp"] = float(payload_bits) / float(pixel_count)
            output_metadata["rate_framed_bpp"] = float(payload_bits) / float(pixel_count)
            output_metadata["rate_coded_bpp"] = float(coded_bits) / float(pixel_count)
            output_metadata["rate_padded_bpp"] = float(padded_bits) / float(pixel_count)
            output_metadata["channel_uses_per_pixel"] = float(channel_uses) / float(pixel_count)
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=rx_bits, metadata_json=json.dumps(output_metadata))
        metrics = {
            "channel.snr_db": snr_db,
            "channel.awgn.capacity_bits": capacity_bits,
            "channel.packet_success_rate": source_item_success_rate,
            "channel.source_item_success_rate": source_item_success_rate,
            "channel.outage_rate": source_item_outage_rate,
            "channel.packet_success": 1 if success else 0,
            "channel.outage": 0 if success else 1,
            "channel.payload_bit_count": payload_bits,
            "channel.coded_bit_count": coded_bits,
            "channel.padded_bit_count": padded_bits,
            "channel.transmitted_bit_count": padded_bits,
            "channel.channel_use_count": channel_uses,
            "channel.bits_per_symbol": bits_per_symbol,
            "channel.ldpc_rate": ldpc_rate,
            "channel.code_rate": ldpc_rate,
            "channel.capacity_oracle": 1,
            "channel.protected_digital.theoretical_reference": 1,
            "channel.capacity_oracle.source_item_accounting": 1,
        }
        if pixel_count:
            metrics["rate.payload_bpp"] = float(payload_bits) / float(pixel_count)
            metrics["rate.framed_bpp"] = float(payload_bits) / float(pixel_count)
            metrics["rate.coded_bpp"] = float(coded_bits) / float(pixel_count)
            metrics["rate.padded_bpp"] = float(padded_bits) / float(pixel_count)
            metrics["channel.uses_per_pixel"] = float(channel_uses) / float(pixel_count)
        if source_item_uses_per_pixel:
            metrics["channel.max_source_item_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        return OperationResult(
            outputs={"bits": artifact("channel.bits.numpy", path, output_metadata)},
            metrics=metrics,
            metadata={
                "protected_digital_baseline": "capacity_oracle",
                "channel": "awgn",
                "snr_db": snr_db,
                "ldpc_rate": ldpc_rate,
                "modulation": "qpsk",
                "channel_use_count": channel_uses,
                "transmitted_bit_count": padded_bits,
                "packet_success": int(success),
                "outage": int(not success),
                "source_item_success_rate": source_item_success_rate,
                "source_item_outage_rate": source_item_outage_rate,
                "capacity_oracle_accounting_scope": "source_item",
                "on_decode_failure": failure_policy,
                "data_plane_backend": selected_backend,
            },
        )


class BitCountMatchOperation(Operation):
    id = "channel.bit_count_match"
    name = "Channel bit-count consistency check"
    input_kinds = {
        "reference": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
        "candidate": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.bits.numpy",
        ],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema(
        {
            "label": {"type": "string", "default": "channel_io"},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        reference_artifact = ctx.require_input("reference")
        candidate_artifact = ctx.require_input("candidate")
        reference, reference_metadata = _load_bits_raw(reference_artifact.path, reference_artifact.metadata)
        candidate, candidate_metadata = _load_bits_raw(candidate_artifact.path, candidate_artifact.metadata)
        label = _metric_label(str(ctx.params.get("label") or "channel_io"))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        reference, selected_backend = _require_canonical_bits(reference, "%s/%s/reference" % (ctx.step_id, label), backend)
        candidate, selected_backend_candidate = _require_canonical_bits(candidate, "%s/%s/candidate" % (ctx.step_id, label), backend)
        reference_count = int(reference.size)
        candidate_count = int(candidate.size)
        delta = candidate_count - reference_count
        if delta != 0:
            raise OperationError(
                "Channel bit-count mismatch at %s: reference has %d bits, candidate has %d bits"
                % (label, reference_count, candidate_count)
            )
        report = {
            "schema_version": 1,
            "metric_family": "bit_count_match",
            "label": label,
            "reference_bit_count": reference_count,
            "candidate_bit_count": candidate_count,
            "length_delta": delta,
            "reference_fixed_point": reference_metadata.get("fixed_point_label") or reference_metadata.get("fixed_point"),
            "candidate_fixed_point": candidate_metadata.get("fixed_point_label") or candidate_metadata.get("fixed_point"),
            "canonical_dtype": "uint8",
            "bit_storage": "unpacked_uint8",
            "data_plane_backend": selected_backend if selected_backend == selected_backend_candidate else "%s/%s" % (selected_backend, selected_backend_candidate),
        }
        path = ctx.output_path("report", ".json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return OperationResult(
            outputs={"report": artifact("metrics.report", path, report)},
            metrics={
                "channel.fixed.%s.bit_count_match" % label: 1,
                "channel.fixed.%s.reference_bit_count" % label: reference_count,
                "channel.fixed.%s.candidate_bit_count" % label: candidate_count,
                "channel.fixed.%s.length_delta" % label: delta,
            },
            metadata=report,
        )


class IdentityChannelEncoderOperation(Operation):
    id = "channel.identity_encoder"
    name = "Identity channel encoder"
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.framed_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"coded_bits": "channel.coded_bits.numpy"}
    params_schema = object_schema(_backend_param())

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = dataplane.require_canonical_bits(bits, ctx.step_id, backend)
        output_metadata = _coding_metadata(
            metadata=metadata,
            scheme="identity",
            input_bit_count=int(bits.size),
            output_bit_count=int(bits.size),
            code_rate=1.0,
        )
        source_counts = _source_item_stage_bit_counts(metadata, int(bits.size))
        if source_counts is not None:
            output_metadata["source_item_coded_bit_counts"] = source_counts
        pixel_count = _pixel_count_from_metadata(metadata)
        if pixel_count:
            output_metadata["rate_coded_bpp"] = float(bits.size) / float(pixel_count)
        path = ctx.output_path("coded_bits", ".npz")
        np.savez_compressed(path, bits=bits.astype(np.uint8), metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"coded_bits": artifact("channel.coded_bits.numpy", path, output_metadata)},
            metrics={
                "channel.code_rate": 1.0,
                "channel.payload_bit_count": int(
                    metadata.get("payload_bit_count") or bits.size
                ),
                "channel.channel_code_input_bit_count": int(bits.size),
                "channel.coded_bit_count": int(bits.size),
                **(
                    {"rate.coded_bpp": float(bits.size) / float(pixel_count)}
                    if pixel_count
                    else {}
                ),
            },
            metadata={"scheme": "identity", "coded_bit_count": int(bits.size), "data_plane_backend": selected_backend},
        )


class RepetitionChannelEncoderOperation(Operation):
    id = "channel.repetition_encoder"
    name = "Repetition channel encoder"
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.framed_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"coded_bits": "channel.coded_bits.numpy"}
    params_schema = object_schema(
        {
            "factor": {"type": "integer", "default": 3, "minimum": 1},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "Repetition encoder input at %s" % ctx.step_id,
        )
        factor = int(ctx.params.get("factor", 3))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        coded, selected_backend = dataplane.repetition_encode(bits, factor, backend)
        output_metadata = _coding_metadata(
            metadata=metadata,
            scheme="repetition",
            input_bit_count=int(bits.size),
            output_bit_count=int(coded.size),
            code_rate=1.0 / float(factor),
        )
        output_metadata["repetition_factor"] = factor
        output_metadata["data_plane_backend"] = selected_backend
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(coded.size),
            "Repetition encoder output at %s" % ctx.step_id,
        )
        source_counts = _source_item_stage_bit_counts(metadata, int(bits.size))
        if source_counts is not None:
            output_metadata["source_item_coded_bit_counts"] = [
                int(count * factor) for count in source_counts
            ]
        pixel_count = _pixel_count_from_metadata(metadata)
        if pixel_count:
            output_metadata["rate_coded_bpp"] = float(coded.size) / float(pixel_count)
        path = ctx.output_path("coded_bits", ".npz")
        np.savez_compressed(path, bits=coded, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"coded_bits": artifact("channel.coded_bits.numpy", path, output_metadata)},
            metrics={
                "channel.code_rate": 1.0 / float(factor),
                "channel.payload_bit_count": int(
                    metadata.get("payload_bit_count") or bits.size
                ),
                "channel.channel_code_input_bit_count": int(bits.size),
                "channel.coded_bit_count": int(coded.size),
                **(
                    {"rate.coded_bpp": float(coded.size) / float(pixel_count)}
                    if pixel_count
                    else {}
                ),
            },
            metadata={"scheme": "repetition", "factor": factor, "data_plane_backend": selected_backend},
        )


class IdentityChannelDecoderOperation(Operation):
    id = "channel.identity_decoder"
    name = "Identity channel decoder"
    input_kinds = {
        "coded_bits": [
            "channel.coded_bits.numpy",
            "channel.demod_bits.numpy",
            "channel.payload_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(_backend_param())

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("coded_bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "Identity decoder input at %s" % ctx.step_id,
        )
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        bits, selected_backend = dataplane.require_canonical_bits(bits, ctx.step_id, backend)
        payload_count = _channel_code_input_bit_count(metadata, int(metadata.get("bit_count") or bits.size))
        decoded = bits[:payload_count].astype(np.uint8)
        if capture_layout is not None and int(decoded.size) != int(bits.size):
            raise OperationError(
                "Identity decoder cannot truncate an explicit capture-record layout"
            )
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "bit_count": int(decoded.size),
                "bit_role": "payload",
                "decoder": "identity",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=decoded, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.decoded_bit_count": int(decoded.size)},
            metadata={"scheme": "identity", "data_plane_backend": selected_backend},
        )


class RepetitionChannelDecoderOperation(Operation):
    id = "channel.repetition_decoder"
    name = "Repetition channel decoder"
    input_kinds = {
        "coded_bits": ["channel.coded_bits.numpy", "channel.demod_bits.numpy", "channel.bits.numpy"]
    }
    output_kinds = {"bits": "channel.payload_bits.numpy"}
    params_schema = object_schema(
        {
            "factor": {"type": "integer", "default": 0, "minimum": 0},
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("coded_bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "Repetition decoder input at %s" % ctx.step_id,
        )
        factor = int(ctx.params.get("factor", 0))
        if factor <= 0:
            factor = int(metadata.get("repetition_factor") or 1)
        fallback_payload_count = (int(bits.size) // max(int(factor), 1))
        payload_count = _channel_code_input_bit_count(metadata, fallback_payload_count)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        decoded, selected_backend = dataplane.repetition_decode(bits, factor, payload_count, backend)
        if capture_layout is not None and (
            capture_layout.elements_per_record % factor
            or int(decoded.size)
            != capture_layout.count
            * (capture_layout.elements_per_record // factor)
        ):
            raise OperationError(
                "Repetition decoder cannot preserve the explicit capture-record layout"
            )
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "bit_count": int(decoded.size),
                "bit_role": "payload",
                "decoder": "repetition",
                "repetition_factor": factor,
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
            }
        )
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(decoded.size),
            "Repetition decoder output at %s" % ctx.step_id,
        )
        path = ctx.output_path("bits", ".npz")
        np.savez_compressed(path, bits=decoded, metadata_json=json.dumps(output_metadata))
        return OperationResult(
            outputs={"bits": artifact("channel.payload_bits.numpy", path, output_metadata)},
            metrics={"channel.decoded_bit_count": int(decoded.size)},
            metadata={"scheme": "repetition", "factor": factor, "data_plane_backend": selected_backend},
        )


class DigitalModulateOperation(Operation):
    id = "modulation.digital_modulate"
    name = "Digital bit-to-symbol modulator"
    input_kinds = {"bits": ["channel.payload_bits.numpy", "channel.coded_bits.numpy", "channel.bits.numpy"]}
    output_kinds = {"symbols": "channel.symbols.complex_numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Hard bit-to-constellation mapping is discrete; differentiable exports should use a differentiable modem surrogate or Sionna block.",
    }
    backends = {"benchmark_run": ["numpy", "cpp"], "dataset_capture": ["numpy", "cpp"], "differentiable_export": []}
    materializations = _digital_modem_materializations("modulator", include_auto=False)
    equivalence = {"type": "numerical", "tolerance": {"atol": 1e-7, "rtol": 1e-6}, "reason": "Hard mapper implementations should produce the same normalized constellation symbols within floating-point tolerance."}
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "modulation": {
                "type": "string",
                "default": "qpsk",
                "enum": ["bpsk", "qpsk", "qam16"],
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        bits, metadata = _load_bits(input_artifact.path, input_artifact.metadata)
        modulation = str(ctx.params.get("modulation", "qpsk"))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        source_counts = _source_item_stage_bit_counts(metadata, int(bits.size))
        capture_record_count: Optional[int] = None
        declared_capture_count = metadata.get("capture_record_count")
        declared_capture_shape = metadata.get("capture_record_shape")
        if declared_capture_count is not None or declared_capture_shape is not None:
            if (
                not isinstance(declared_capture_count, int)
                or isinstance(declared_capture_count, bool)
                or declared_capture_count <= 0
                or not isinstance(declared_capture_shape, (list, tuple))
                or not declared_capture_shape
                or any(
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    or item <= 0
                    for item in declared_capture_shape
                )
            ):
                raise OperationError(
                    "Digital modulator input has an invalid explicit capture-record layout"
                )
            record_bit_count = int(
                np.prod(declared_capture_shape, dtype=np.int64)
            )
            if declared_capture_count * record_bit_count != int(bits.size):
                raise OperationError(
                    "Digital modulator input capture-record layout does not match its bit count"
                )
            capture_record_count = int(declared_capture_count)
            capture_counts = [record_bit_count] * capture_record_count
            if source_counts is None:
                source_counts = capture_counts
            elif [int(value) for value in source_counts] != capture_counts:
                raise OperationError(
                    "Digital modulator source-item and capture-record layouts conflict"
                )
        source_item_symbol_counts: List[int] = []
        source_item_uses_per_pixel: List[float] = []
        if source_counts is not None:
            symbol_rows: List[np.ndarray] = []
            padded_rows: List[np.ndarray] = []
            selected_backend = "python_numpy"
            bits_per_symbol = 0
            offset = 0
            for count in source_counts:
                row_symbols, row_padded, row_bps, selected_backend = _modulate(
                    bits[offset : offset + count], modulation, backend
                )
                symbol_rows.append(row_symbols)
                padded_rows.append(row_padded)
                source_item_symbol_counts.append(int(row_symbols.size))
                bits_per_symbol = row_bps
                offset += count
            symbols = np.concatenate(symbol_rows).astype(np.complex64, copy=False)
            padded_bits = np.concatenate(padded_rows).astype(np.uint8, copy=False)
        else:
            symbols, padded_bits, bits_per_symbol, selected_backend = _modulate(
                bits, modulation, backend
            )
        pixel_count = _pixel_count_from_metadata(metadata)
        output_metadata = dict(metadata)
        output_metadata.pop("capture_record_count", None)
        output_metadata.pop("capture_record_shape", None)
        output_metadata.update(
            {
                "modulation": modulation,
                "bits_per_symbol": bits_per_symbol,
                "bit_count": int(bits.size),
                "padded_bit_count": int(padded_bits.size),
                "symbol_count": int(symbols.size),
                "data_plane_backend": selected_backend,
            }
        )
        if source_item_symbol_counts:
            output_metadata["source_item_symbol_counts"] = source_item_symbol_counts
            output_metadata["source_item_modulator_input_bit_counts"] = [
                int(item) for item in source_counts or []
            ]
            output_metadata["source_item_padded_bit_counts"] = [
                int(item * bits_per_symbol) for item in source_item_symbol_counts
            ]
            source_item_pixels = _source_item_pixel_counts(
                metadata, len(source_item_symbol_counts)
            )
            if source_item_pixels is not None:
                source_item_uses_per_pixel = [
                    float(symbol_count) / float(pixel_count)
                    for symbol_count, pixel_count in zip(
                        source_item_symbol_counts, source_item_pixels
                    )
                ]
                output_metadata["source_item_pixel_counts"] = source_item_pixels
                output_metadata[
                    "source_item_channel_uses_per_pixel"
                ] = source_item_uses_per_pixel
                output_metadata["max_source_item_channel_uses_per_pixel"] = max(
                    source_item_uses_per_pixel
                )
        if capture_record_count is not None:
            if (
                len(source_item_symbol_counts) != capture_record_count
                or len(set(source_item_symbol_counts)) != 1
            ):
                raise OperationError(
                    "Digital modulator could not preserve the declared capture-record layout"
                )
            output_metadata["capture_record_count"] = capture_record_count
            output_metadata["capture_record_shape"] = [
                int(source_item_symbol_counts[0])
            ]
        if pixel_count:
            output_metadata["pixel_count"] = pixel_count
            output_metadata["channel_uses_per_pixel"] = float(symbols.size) / float(pixel_count)
        path = ctx.output_path("symbols", ".npz")
        np.savez_compressed(path, symbols=symbols, metadata_json=json.dumps(output_metadata))
        metrics = {
            "channel.coded_bit_count": int(bits.size),
            "channel.transmitted_bit_count": int(padded_bits.size),
            "channel.modulation_padding_bit_count": int(
                padded_bits.size - bits.size
            ),
            "channel.symbol_count": int(symbols.size),
            "channel.channel_use_count": int(symbols.size),
            "channel.bits_per_symbol": bits_per_symbol,
        }
        if pixel_count:
            metrics.update(
                {
                    "rate.coded_bpp": float(bits.size) / float(pixel_count),
                    "rate.padded_bpp": float(padded_bits.size)
                    / float(pixel_count),
                    "channel.uses_per_pixel": float(symbols.size)
                    / float(pixel_count),
                }
            )
        if source_item_uses_per_pixel:
            metrics["channel.max_source_item_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        return OperationResult(
            outputs={"symbols": artifact("channel.symbols.complex_numpy", path, output_metadata)},
            metrics=metrics,
            metadata={"modulation": modulation, "data_plane_backend": selected_backend},
        )


class WirelessChannelOperation(Operation):
    id = "wireless.channel"
    name = "Wireless channel over complex symbols"
    thread_safe = True
    input_kinds = {"symbols": ["channel.symbols.complex_numpy"]}
    optional_input_kinds = {
        "channel_state": ["channel.ofdm_channel_state.numpy"],
    }
    output_kinds = {"rx_symbols": "channel.rx_symbols.complex_numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Native Torch and Sionna 2.x/PyTorch AWGN and flat-Rayleigh materializations preserve gradients to transmitted symbols.",
    }
    backends = {"benchmark_run": ["numpy", "sionna"], "dataset_capture": ["numpy", "sionna"], "differentiable_export": ["torch", "sionna"]}
    materializations = _wireless_channel_materializations()
    equivalence = {
        "type": "statistical",
        "reason": "Only materializations with the same bound channel, receiver_processing, and channel_state_mode share a statistical-equivalence obligation; random samples need not be identical.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "channel": {
                "type": "string",
                "default": "awgn",
                "enum": WIRELESS_CHANNELS,
            },
            **_noise_control_schema(),
            "wireless_backend": _wireless_backend_schema(),
            "receiver_processing": {
                "type": "string",
                "default": "matched",
                "enum": RECEIVER_PROCESSING_MODES,
                "description": "matched applies the preset's declared receiver (perfect-CSI equalization for fading presets); none returns the raw supported receive signal.",
            },
            "fading_scope": {
                "type": "string",
                "default": "symbol",
                "enum": ["symbol", "source_item"],
                "description": (
                    "For flat Rayleigh fading, symbol draws an independent gain "
                    "per complex symbol; source_item holds one gain constant over "
                    "each source item. Source-item mode requires preserved item "
                    "boundaries."
                ),
                "x-noema-effective-when": {"channel": "flat_rayleigh"},
                "x-noema-ui": {"visible_when": {"channel": "flat_rayleigh"}},
            },
            "channel_state_mode": {
                "type": "string",
                "default": "none",
                "enum": CHANNEL_STATE_MODES,
                "description": "explicit requires a bound channel_state artifact; none forbids one.",
            },
            "tx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "rx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "ofdm_fft_size": {"type": "integer", "default": 64, "minimum": 8},
            "num_ofdm_symbols": {"type": "integer", "default": 14, "minimum": 1},
            "subcarrier_spacing_khz": {"type": "number", "default": 15.0, "minimum": 0.1},
            "carrier_frequency_ghz": {"type": "number", "default": 3.5, "minimum": 0.1},
            "mobility_kmh": {"type": "number", "default": 3.0, "minimum": 0.0},
            "tdl_model": {"type": "string", "default": "A", "enum": ["A", "B", "C", "D", "E"]},
            "delay_spread_ns": {"type": "number", "default": 100.0, "minimum": 0.1},
            "normalize_channel": {"type": "boolean", "default": True},
            "interferers": {"type": "integer", "default": 0, "minimum": 0},
            "interference_sir_db": {"type": "number", "default": 18.0},
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": "Optional explicit operation seed. Omit it to derive the stream from the recipe master seed.",
            },
            **_backend_param(),
        }
    )

    def describe(self) -> JsonDict:
        payload = super().describe()
        payload["availability"] = _sionna_availability(optional=True)
        payload["wireless_presets"] = {
            "awgn": "Complex AWGN baseline.",
            "flat_rayleigh": "Flat Rayleigh fading with perfect one-tap equalization.",
            "interference_awgn": "AWGN plus controllable co-channel interference.",
            "mimo_flat": "Flat Rayleigh MIMO receive diversity / perfect-CSI equalization.",
            "ofdm_tdl": "OFDM-grid tapped-delay-line emulation with perfect frequency-domain equalization.",
            "ofdm_cdl": "Clustered-delay-line inspired OFDM emulation.",
            "urban_micro": "Urban microcell preset: OFDM TDL plus mobility and interference defaults.",
        }
        return payload

    def validate_preflight(
        self,
        params: Mapping[str, Any],
        inputs: Mapping[str, str] | None = None,
    ) -> None:
        values = dict(params)
        _resolve_noise_control(values)

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        requested = str(params.get("wireless_backend") or "auto").strip().lower()
        explicit_state = (
            str(params.get("channel_state_mode") or "none").strip().lower()
            == "explicit"
        )
        sionna = _sionna_availability(optional=True)
        # A provider may report an operation-wide outage (no ``optional``
        # marker), in which case even the local selection is not runnable.
        if sionna.get("available") is False and not sionna.get("optional"):
            return sionna
        if requested == "sionna" or (requested == "auto" and explicit_state):
            return sionna
        return {
            "available": True,
            "optional": False,
            "backend": "numpy",
            "missing": [],
        }

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("symbols")
        symbols, metadata = _load_symbols(input_artifact.path, input_artifact.metadata)
        channel_state = None
        if "channel_state" in ctx.inputs:
            channel_state = _load_ofdm_channel_state_artifact(ctx.inputs["channel_state"])
        channel = str(ctx.params.get("channel", "awgn"))
        receiver_processing = str(
            ctx.params.get("receiver_processing") or "matched"
        )
        channel_state_mode = str(ctx.params.get("channel_state_mode") or "none")
        noise_mode, noise_var, reference_snr_db = _resolve_noise_control(ctx.params)
        tx_power_average = float(np.mean(np.abs(symbols) ** 2)) if int(symbols.size) else 0.0
        seed = ctx.seed("wireless_channel")
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        wireless_backend = str(ctx.params.get("wireless_backend") or "auto")
        _validate_wireless_runtime_contract(
            channel,
            wireless_backend,
            receiver_processing,
            channel_state_mode,
            has_channel_state=channel_state is not None,
        )
        rng = np.random.RandomState(seed)
        item_partition = _source_item_symbol_partition(metadata, int(symbols.size))
        fading_scope = str(ctx.params.get("fading_scope") or "symbol")
        if (
            channel == "flat_rayleigh"
            and fading_scope == "source_item"
            and item_partition is None
        ):
            raise OperationError(
                "flat_rayleigh fading_scope=source_item requires preserved "
                "source-item symbol boundaries"
            )
        source_item_channel_seeds: List[int] = []
        source_item_uses_per_pixel: List[float] = []
        source_item_tx_power_average: List[float] = []
        if item_partition is not None:
            source_item_pixels = _source_item_pixel_counts(
                metadata, len(item_partition)
            )
            if source_item_pixels is not None:
                source_item_uses_per_pixel = [
                    float(symbol_count) / float(pixel_count)
                    for symbol_count, pixel_count in zip(
                        item_partition, source_item_pixels
                    )
                ]
            item_offset = 0
            for symbol_count in item_partition:
                item_symbols = symbols[item_offset : item_offset + symbol_count]
                source_item_tx_power_average.append(
                    float(np.mean(np.abs(item_symbols) ** 2))
                )
                item_offset += symbol_count
        if channel_state is not None:
            if channel != "ofdm_tdl":
                raise OperationError(
                    "Explicit Sionna TDL channel state cannot be applied to wireless.channel configured as %s" % channel
                )
            if wireless_backend not in {"auto", "sionna"}:
                raise OperationError(
                    "A Sionna OFDM channel-state realization must be applied with wireless_backend=sionna"
                )
            state_noise = float(channel_state["metadata"].get("noise_variance") or noise_var)
            if not math.isclose(state_noise, noise_var, rel_tol=1e-9, abs_tol=1e-12):
                raise OperationError(
                    "wireless.channel noise_variance %.12g does not match the CSI scenario noise_variance %.12g"
                    % (noise_var, state_noise)
                )
            rx_symbols, channel_report = _apply_sionna_realized_ofdm_channel(
                symbols,
                channel_state["h_freq"],
                noise_var,
                ctx.params,
                seed,
            )
        else:
            if item_partition is not None:
                (
                    rx_symbols,
                    channel_report,
                    source_item_channel_seeds,
                ) = _apply_wireless_channel_per_item(
                    symbols,
                    metadata,
                    item_partition,
                    channel,
                    reference_snr_db,
                    ctx.params,
                    backend,
                    wireless_backend,
                    seed,
                )
            else:
                rx_symbols, channel_report = _apply_wireless_channel(
                    symbols,
                    channel,
                    reference_snr_db,
                    rng,
                    ctx.params,
                    backend,
                    wireless_backend,
                    seed,
                )
        reported_backend = str(
            channel_report.get("wireless_backend")
            or channel_report.get("backend")
            or ""
        )
        if reported_backend in {"python_numpy", "cpp_native"}:
            selected_backend = "numpy"
            selected_data_plane_backend = reported_backend
            selected_wireless_backend_detail = (
                "numpy_wireless/%s_dataplane" % reported_backend
            )
        elif reported_backend == "numpy":
            selected_backend = "numpy"
            selected_data_plane_backend = str(
                channel_report.get("data_plane_backend") or "python_numpy"
            )
            selected_wireless_backend_detail = str(
                channel_report.get("backend_detail") or "numpy_wireless"
            )
        elif reported_backend == "sionna":
            selected_backend = "sionna"
            selected_data_plane_backend = str(
                channel_report.get("data_plane_backend") or "torch"
            )
            selected_wireless_backend_detail = str(
                channel_report.get("backend_detail") or "sionna_wireless"
            )
        else:
            raise OperationError(
                "wireless.channel reported an unknown runtime backend: %s"
                % (reported_backend or "<missing>")
            )
        if int(np.asarray(rx_symbols).size) != int(symbols.size):
            raise OperationError(
                "wireless.channel changed payload symbol cardinality from %d to %d"
                % (int(symbols.size), int(np.asarray(rx_symbols).size))
            )
        payload_symbol_count = int(symbols.size)
        executed_channel_use_count = int(
            channel_report.get("executed_channel_use_count")
            or payload_symbol_count
        )
        if executed_channel_use_count < payload_symbol_count:
            raise OperationError(
                "wireless.channel executed use count cannot be smaller than its payload symbol count"
            )
        grid_padding_symbol_count = int(
            channel_report.get("grid_padding_symbol_count")
            or (executed_channel_use_count - payload_symbol_count)
        )
        source_item_executed_counts = [
            int(item)
            for item in list(
                channel_report.get("source_item_executed_channel_use_counts")
                or []
            )
        ]
        if item_partition is not None and not source_item_executed_counts:
            source_item_executed_counts = [int(item) for item in item_partition]
        if item_partition is not None and (
            len(source_item_executed_counts) != len(item_partition)
            or sum(source_item_executed_counts) > executed_channel_use_count
        ):
            raise OperationError(
                "Per-item channel-use accounting exceeds the physical realization"
            )
        shared_grid_overhead_channel_use_count = (
            executed_channel_use_count - sum(source_item_executed_counts)
            if item_partition is not None
            else grid_padding_symbol_count
        )
        source_item_charged_counts = [
            int(item + shared_grid_overhead_channel_use_count)
            for item in source_item_executed_counts
        ]
        if item_partition is not None:
            source_item_pixels = _source_item_pixel_counts(
                metadata, len(item_partition)
            )
            if source_item_pixels is not None:
                source_item_uses_per_pixel = [
                    float(use_count) / float(pixel_count)
                    for use_count, pixel_count in zip(
                        source_item_charged_counts, source_item_pixels
                    )
                ]
        noise_var = float(channel_report.get("noise_variance") or noise_var)
        channel_gain_average = float(channel_report.get("channel_gain_average", 1.0))
        tx_effective_snr_db = _snr_db_from_signal_and_noise(tx_power_average, noise_var)
        received_signal_power_average = float(
            channel_report.get("received_signal_power_average", tx_power_average * channel_gain_average)
        )
        effective_snr_db = _snr_db_from_signal_and_noise(received_signal_power_average, noise_var)
        power_unit = str(metadata.get("power_unit") or "normalized")
        tx_total_energy = float(np.sum(np.abs(symbols) ** 2))
        tx_power_per_executed_use = (
            tx_total_energy / float(executed_channel_use_count)
            if executed_channel_use_count
            else 0.0
        )
        rx_power_preview, rx_power_average, rx_power_total_energy = _symbol_power_trace_metadata(
            rx_symbols, "rx_power", unit=power_unit
        )
        rx_power_preview["kind"] = "rx_output_power_trace"
        rx_antenna_power_preview = channel_report.get("rx_antenna_power_preview") or rx_power_preview
        rx_antenna_power_average = float(channel_report.get("rx_antenna_power_average", rx_power_average))
        rx_antenna_power_total_energy = float(channel_report.get("rx_antenna_power_total_energy", rx_power_total_energy))
        rx_equalized_power_preview = channel_report.get("rx_equalized_power_preview")
        rx_equalized_power_average = channel_report.get("rx_equalized_power_average")
        rx_equalized_power_total_energy = channel_report.get("rx_equalized_power_total_energy")
        component_power_metric_names = {
            "rx_antenna_signal_power_average": "channel.rx_antenna_signal_power.average",
            "rx_antenna_noise_power_average": "channel.rx_antenna_noise_power.average",
            "rx_antenna_component_power_average": "channel.rx_antenna_component_power.average",
            "rx_antenna_signal_noise_cross_power_average": (
                "channel.rx_antenna_signal_noise_cross_power.average"
            ),
            "post_equalizer_signal_power_average": "channel.post_equalizer_signal_power.average",
            "post_equalizer_noise_power_average": "channel.post_equalizer_noise_power.average",
            "post_equalizer_component_power_average": "channel.post_equalizer_component_power.average",
            "post_equalizer_signal_noise_cross_power_average": (
                "channel.post_equalizer_signal_noise_cross_power.average"
            ),
        }
        component_power_values = {
            metadata_key: float(channel_report[metadata_key])
            for metadata_key in component_power_metric_names
            if channel_report.get(metadata_key) is not None
        }
        channel_equalized = bool(channel_report.get("channel_equalized", False))
        pixel_count = _pixel_count_from_metadata(metadata)
        output_metadata = dict(metadata)
        if channel_state is not None:
            output_metadata.update(
                {
                    "channel_state_shared": True,
                    "channel_state_kind": "sionna_3gpp_tdl_ofdm",
                    "channel_state_seed": channel_state["metadata"].get("channel_state_seed"),
                    "transmitter_csi_assumption": channel_state["metadata"].get(
                        "transmitter_csi_assumption", "perfect_instantaneous"
                    ),
                }
            )
        output_metadata.setdefault("wireless_history", [])
        output_metadata["wireless_history"] = list(output_metadata["wireless_history"]) + [
            {
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "tx_effective_snr_db": tx_effective_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "seed": seed,
                "noise_variance": noise_var,
                "wireless_backend": selected_backend,
                "preset": channel_report.get("preset", channel),
                "backend_detail": selected_wireless_backend_detail,
                "requested_wireless_backend": channel_report.get(
                    "requested_wireless_backend", wireless_backend
                ),
                "data_plane_backend": selected_data_plane_backend,
                "receiver_processing": receiver_processing,
                "fading_scope": str(
                    channel_report.get("fading_scope")
                    or ctx.params.get("fading_scope")
                    or "symbol"
                ),
                "channel_state_mode": channel_state_mode,
            }
        ]
        output_metadata.update(
            {
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "tx_effective_snr_db": tx_effective_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "noise_variance": noise_var,
                "tx_power_average": tx_power_average,
                "tx_power_per_executed_use": tx_power_per_executed_use,
                "tx_total_energy": tx_total_energy,
                "received_signal_power_average": received_signal_power_average,
                "channel_response_preview": channel_report.get("channel_response_preview"),
                "rx_power_preview": rx_power_preview,
                "rx_power_average": rx_power_average,
                "rx_power_total_energy": rx_power_total_energy,
                "rx_output_power_preview": rx_power_preview,
                "rx_output_power_average": rx_power_average,
                "rx_output_power_total_energy": rx_power_total_energy,
                "rx_antenna_power_preview": rx_antenna_power_preview,
                "rx_antenna_power_average": rx_antenna_power_average,
                "rx_antenna_power_total_energy": rx_antenna_power_total_energy,
                "rx_power_unit": power_unit,
                "channel_equalized": channel_equalized,
                "equalizer": channel_report.get("equalizer"),
                "receiver_processing": receiver_processing,
                "fading_scope": str(
                    channel_report.get("fading_scope")
                    or ctx.params.get("fading_scope")
                    or "symbol"
                ),
                "channel_state_mode": channel_state_mode,
                "wireless_backend": selected_backend,
                "wireless_backend_detail": selected_wireless_backend_detail,
                "requested_wireless_backend": channel_report.get(
                    "requested_wireless_backend", wireless_backend
                ),
                "wireless_preset": channel_report.get("preset", channel),
                "tx_antennas": channel_report.get("tx_antennas"),
                "rx_antennas": channel_report.get("rx_antennas"),
                "ofdm_fft_size": channel_report.get("ofdm_fft_size"),
                "num_ofdm_symbols": channel_report.get("num_ofdm_symbols"),
                "interferers": channel_report.get("interferers"),
                "interference_sir_db": channel_report.get("interference_sir_db"),
                "carrier_frequency_ghz": channel_report.get("carrier_frequency_ghz"),
                "mobility_kmh": channel_report.get("mobility_kmh"),
                "payload_symbol_count": payload_symbol_count,
                "channel_use_count": executed_channel_use_count,
                "grid_padding_symbol_count": grid_padding_symbol_count,
                "shared_grid_overhead_channel_use_count": (
                    shared_grid_overhead_channel_use_count
                ),
                "tx_power_per_executed_use": tx_power_per_executed_use,
                "tx_total_energy": tx_total_energy,
                "data_plane_backend": selected_data_plane_backend,
                **component_power_values,
            }
        )
        if item_partition is not None:
            output_metadata["source_item_symbol_counts"] = [
                int(item) for item in item_partition
            ]
            output_metadata[
                "source_item_tx_power_average"
            ] = source_item_tx_power_average
            output_metadata["source_item_channel_use_counts"] = (
                source_item_charged_counts
            )
            output_metadata["source_item_payload_channel_use_counts"] = (
                source_item_executed_counts
            )
            output_metadata["max_source_item_tx_power_average"] = max(
                source_item_tx_power_average
            )
        if source_item_uses_per_pixel:
            output_metadata[
                "source_item_channel_uses_per_pixel"
            ] = source_item_uses_per_pixel
            output_metadata["max_source_item_channel_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        if source_item_channel_seeds:
            output_metadata.update(
                {
                    "source_item_channel_seeds": [
                        int(item) for item in source_item_channel_seeds
                    ],
                    "source_item_channel_identity_keys": list(
                        channel_report.get("source_item_identity_keys") or []
                    ),
                    "channel_realization_scope": "source_item_identity",
                    "channel_realization_identity_field": channel_report.get(
                        "source_item_identity_field"
                    ),
                    "channel_realization_order_invariant": bool(
                        channel_report.get("channel_realization_order_invariant")
                    ),
                }
            )
            for key in (
                "source_item_channel_gain_real",
                "source_item_channel_gain_imag",
                "source_item_channel_gain_magnitude",
                "source_item_channel_gain_power",
            ):
                values = channel_report.get(key)
                if isinstance(values, list):
                    output_metadata[key] = [float(value) for value in values]
        if rx_equalized_power_preview is not None:
            output_metadata["rx_equalized_power_preview"] = rx_equalized_power_preview
        if rx_equalized_power_average is not None:
            output_metadata["rx_equalized_power_average"] = float(rx_equalized_power_average)
            output_metadata["post_equalizer_output_power_average"] = float(rx_equalized_power_average)
        if rx_equalized_power_total_energy is not None:
            output_metadata["rx_equalized_power_total_energy"] = float(rx_equalized_power_total_energy)
        if pixel_count:
            output_metadata["pixel_count"] = pixel_count
            output_metadata["channel_uses_per_pixel"] = (
                float(executed_channel_use_count) / float(pixel_count)
            )
        path = ctx.output_path("rx_symbols", ".npz")
        np.savez_compressed(path, symbols=rx_symbols, metadata_json=json.dumps(output_metadata))
        metrics = {
            "channel.snr_db": reference_snr_db,
            "channel.reference_snr_db": reference_snr_db,
            "channel.tx_effective_snr_db": tx_effective_snr_db,
            "channel.effective_snr_db": effective_snr_db,
            "channel.tx_power_at_channel.average": tx_power_average,
            "channel.tx_power_per_executed_use.average": tx_power_per_executed_use,
            "channel.tx_energy.total": tx_total_energy,
            "channel.received_signal_power.average": received_signal_power_average,
            "channel.noise_variance": noise_var,
            "channel.rx_power.average": rx_power_average,
            "channel.rx_power.total_energy": rx_power_total_energy,
            "channel.rx_output_power.average": rx_power_average,
            "channel.rx_output_power.total_energy": rx_power_total_energy,
            "channel.rx_antenna_power.average": rx_antenna_power_average,
            "channel.rx_antenna_power.total_energy": rx_antenna_power_total_energy,
            "channel.symbol_count": payload_symbol_count,
            "channel.payload_symbol_count": payload_symbol_count,
            "channel.channel_use_count": executed_channel_use_count,
            "channel.grid_padding_symbol_count": grid_padding_symbol_count,
            "channel.shared_grid_overhead_channel_use_count": (
                shared_grid_overhead_channel_use_count
            ),
            "channel.backend.%s" % _metric_label(selected_backend): 1,
        }
        if rx_equalized_power_average is not None:
            metrics["channel.rx_equalized_power.average"] = float(rx_equalized_power_average)
            metrics["channel.post_equalizer_output_power.average"] = float(rx_equalized_power_average)
        if rx_equalized_power_total_energy is not None:
            metrics["channel.rx_equalized_power.total_energy"] = float(rx_equalized_power_total_energy)
        for metadata_key, metric_name in component_power_metric_names.items():
            if metadata_key in component_power_values:
                metrics[metric_name] = component_power_values[metadata_key]
        if channel_report.get("channel_gain_average") is not None:
            metrics["channel.gain.average"] = float(channel_report.get("channel_gain_average"))
        gain_magnitudes = channel_report.get(
            "source_item_channel_gain_magnitude"
        )
        if isinstance(gain_magnitudes, list) and gain_magnitudes:
            metrics["channel.gain_magnitude.minimum"] = float(
                min(gain_magnitudes)
            )
            metrics["channel.gain_magnitude.maximum"] = float(
                max(gain_magnitudes)
            )
        gain_powers = channel_report.get("source_item_channel_gain_power")
        if isinstance(gain_powers, list) and gain_powers:
            metrics["channel.gain_power.minimum"] = float(min(gain_powers))
            metrics["channel.gain_power.maximum"] = float(max(gain_powers))
        if pixel_count:
            metrics["channel.uses_per_pixel"] = (
                float(executed_channel_use_count) / float(pixel_count)
            )
        if source_item_uses_per_pixel:
            metrics["channel.max_source_item_uses_per_pixel"] = max(
                source_item_uses_per_pixel
            )
        if source_item_tx_power_average:
            metrics["channel.max_source_item_tx_power.average"] = max(
                source_item_tx_power_average
            )
        if source_item_channel_seeds:
            metrics["channel.per_item_seeded_realization"] = 1
        for key in [
            "tx_antennas",
            "rx_antennas",
            "ofdm_fft_size",
            "num_ofdm_symbols",
            "interferers",
            "interference_sir_db",
            "carrier_frequency_ghz",
            "mobility_kmh",
            "multipath_taps",
        ]:
            value = channel_report.get(key)
            if value is not None:
                metrics["channel.%s" % key] = value
        return OperationResult(
            outputs={"rx_symbols": artifact("channel.rx_symbols.complex_numpy", path, output_metadata)},
            metrics=metrics,
            metadata={
                "channel": channel,
                "snr_db": reference_snr_db,
                "reference_snr_db": reference_snr_db,
                "tx_effective_snr_db": tx_effective_snr_db,
                "effective_snr_db": effective_snr_db,
                "noise_mode": noise_mode,
                "noise_variance": noise_var,
                "tx_power_average": tx_power_average,
                "tx_power_per_executed_use": tx_power_per_executed_use,
                "tx_total_energy": tx_total_energy,
                "received_signal_power_average": received_signal_power_average,
                "wireless_backend": selected_backend,
                "backend_detail": selected_wireless_backend_detail,
                "wireless_backend_detail": selected_wireless_backend_detail,
                "requested_wireless_backend": channel_report.get(
                    "requested_wireless_backend", wireless_backend
                ),
                "rx_power_average": rx_power_average,
                "rx_power_total_energy": rx_power_total_energy,
                "rx_antenna_power_average": rx_antenna_power_average,
                "rx_antenna_power_total_energy": rx_antenna_power_total_energy,
                "rx_equalized_power_average": float(rx_equalized_power_average) if rx_equalized_power_average is not None else None,
                "rx_equalized_power_total_energy": float(rx_equalized_power_total_energy) if rx_equalized_power_total_energy is not None else None,
                "post_equalizer_output_power_average": float(rx_equalized_power_average) if rx_equalized_power_average is not None else None,
                "rx_power_unit": power_unit,
                "channel_equalized": channel_equalized,
                "equalizer": channel_report.get("equalizer"),
                "receiver_processing": receiver_processing,
                "fading_scope": str(
                    channel_report.get("fading_scope")
                    or ctx.params.get("fading_scope")
                    or "symbol"
                ),
                "channel_state_mode": channel_state_mode,
                "data_plane_backend": selected_data_plane_backend,
                "payload_symbol_count": payload_symbol_count,
                "channel_use_count": executed_channel_use_count,
                "grid_padding_symbol_count": grid_padding_symbol_count,
                "shared_grid_overhead_channel_use_count": (
                    shared_grid_overhead_channel_use_count
                ),
                "channel_realization_scope": (
                    "source_item_identity" if source_item_channel_seeds else "batch"
                ),
                "channel_realization_order_invariant": bool(
                    source_item_channel_seeds
                    and channel_report.get("channel_realization_order_invariant")
                ),
                **component_power_values,
            },
        )


_RECEIVER_IQ_IMPAIRMENT_METADATA_KEY = "receiver_iq_impairment"


def _receiver_iq_forward_matrix(
    *,
    gain_imbalance_db: float,
    quadrature_error_deg: float,
    phase_offset_deg: float,
) -> np.ndarray:
    """Build the real 2x2 map of a fixed, memoryless receiver front end."""

    gain_ratio = 10.0 ** (float(gain_imbalance_db) / 20.0)
    i_gain = math.sqrt(gain_ratio)
    q_gain = 1.0 / i_gain
    quadrature_error = math.radians(float(quadrature_error_deg))
    phase_offset = math.radians(float(phase_offset_deg))
    rotation = np.asarray(
        [
            [math.cos(phase_offset), -math.sin(phase_offset)],
            [math.sin(phase_offset), math.cos(phase_offset)],
        ],
        dtype=np.float64,
    )
    imbalance = np.asarray(
        [
            [i_gain, 0.0],
            [
                q_gain * math.sin(quadrature_error),
                q_gain * math.cos(quadrature_error),
            ],
        ],
        dtype=np.float64,
    )
    matrix = imbalance @ rotation
    determinant = float(np.linalg.det(matrix))
    if not bool(np.all(np.isfinite(matrix))) or abs(determinant) < 1e-8:
        raise OperationError(
            "Receiver I/Q impairment parameters produce a singular or non-finite transform"
        )
    return matrix


def _receiver_iq_impairment_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    raw = metadata.get(_RECEIVER_IQ_IMPAIRMENT_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OperationError("Receiver I/Q impairment metadata must be an object")
    matrix = np.asarray(raw.get("forward_matrix"), dtype=np.float64)
    offset = np.asarray(raw.get("dc_offset"), dtype=np.float64)
    if tuple(matrix.shape) != (2, 2) or tuple(offset.shape) != (2,):
        raise OperationError(
            "Receiver I/Q impairment metadata requires a 2x2 forward_matrix and two-element dc_offset"
        )
    determinant = float(np.linalg.det(matrix))
    if (
        not bool(np.all(np.isfinite(matrix)))
        or not bool(np.all(np.isfinite(offset)))
        or abs(determinant) < 1e-8
    ):
        raise OperationError(
            "Receiver I/Q impairment metadata contains a singular or non-finite transform"
        )
    return matrix, offset


def _apply_receiver_iq_impairment(
    symbols: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    features = np.stack(
        [
            np.asarray(symbols).real.astype(np.float64, copy=False),
            np.asarray(symbols).imag.astype(np.float64, copy=False),
        ],
        axis=1,
    )
    impaired = features @ np.asarray(matrix, dtype=np.float64).T
    impaired += np.asarray(offset, dtype=np.float64).reshape(1, 2)
    return np.ascontiguousarray(
        impaired[:, 0] + 1j * impaired[:, 1],
        dtype=np.complex64,
    )


def _compensate_receiver_iq_impairment(
    symbols: np.ndarray,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    impairment = _receiver_iq_impairment_from_metadata(metadata)
    if impairment is None:
        raise OperationError(
            "Calibrated I/Q oracle requires receiver I/Q impairment metadata"
        )
    matrix, offset = impairment
    features = np.stack(
        [
            np.asarray(symbols).real.astype(np.float64, copy=False),
            np.asarray(symbols).imag.astype(np.float64, copy=False),
        ],
        axis=1,
    )
    compensated = (features - offset.reshape(1, 2)) @ np.linalg.inv(matrix).T
    return np.ascontiguousarray(
        compensated[:, 0] + 1j * compensated[:, 1],
        dtype=np.complex64,
    )


class ReceiverIqImpairmentOperation(Operation):
    """Apply one stable, device-specific affine distortion in the receiver I/Q plane."""

    id = "hardware.receiver_iq_imbalance"
    name = "Receiver I/Q front-end impairment"
    thread_safe = True
    input_kinds = {"rx_symbols": ["channel.rx_symbols.complex_numpy"]}
    output_kinds = {"rx_symbols": "channel.rx_symbols.complex_numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "The fixed affine I/Q transform is differentiable, while its simulated calibration parameters remain frozen scenario settings.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {
            "runner": runner,
            "backend": "torch" if runner == "differentiable_export" else "numpy",
            "implementation": "fixed_affine_receiver_iq_impairment",
            "status": "implemented",
        }
        for runner in ("benchmark_run", "dataset_capture", "differentiable_export")
    ]
    equivalence = {
        "type": "numerical",
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "reason": "All materializations implement the same frozen real 2x2 I/Q transform and DC offset.",
    }
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "gain_imbalance_db": {
                "type": "number",
                "default": 5.0,
                "minimum": -20.0,
                "maximum": 20.0,
                "description": "I-to-Q amplitude-gain ratio in dB.",
            },
            "quadrature_error_deg": {
                "type": "number",
                "default": 12.0,
                "minimum": -45.0,
                "maximum": 45.0,
            },
            "phase_offset_deg": {
                "type": "number",
                "default": 20.0,
                "minimum": -180.0,
                "maximum": 180.0,
            },
            "dc_offset_i": {
                "type": "number",
                "default": 0.18,
                "minimum": -2.0,
                "maximum": 2.0,
            },
            "dc_offset_q": {
                "type": "number",
                "default": -0.12,
                "minimum": -2.0,
                "maximum": 2.0,
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("rx_symbols")
        symbols, metadata = _load_symbols(
            input_artifact.path, input_artifact.metadata
        )
        symbols = _require_canonical_symbols(symbols, ctx.step_id)
        capture_layout = _validated_capture_layout(
            metadata,
            int(symbols.size),
            "Receiver I/Q impairment input at %s" % ctx.step_id,
        )
        parameters = {
            "gain_imbalance_db": float(
                ctx.params.get("gain_imbalance_db", 5.0)
            ),
            "quadrature_error_deg": float(
                ctx.params.get("quadrature_error_deg", 12.0)
            ),
            "phase_offset_deg": float(
                ctx.params.get("phase_offset_deg", 20.0)
            ),
            "dc_offset_i": float(ctx.params.get("dc_offset_i", 0.18)),
            "dc_offset_q": float(ctx.params.get("dc_offset_q", -0.12)),
        }
        if not all(math.isfinite(value) for value in parameters.values()):
            raise OperationError(
                "Receiver I/Q impairment parameters must be finite"
            )
        matrix = _receiver_iq_forward_matrix(
            gain_imbalance_db=parameters["gain_imbalance_db"],
            quadrature_error_deg=parameters["quadrature_error_deg"],
            phase_offset_deg=parameters["phase_offset_deg"],
        )
        offset = np.asarray(
            [parameters["dc_offset_i"], parameters["dc_offset_q"]],
            dtype=np.float64,
        )
        impaired = _apply_receiver_iq_impairment(symbols, matrix, offset)
        output_metadata = dict(metadata)
        output_metadata[_RECEIVER_IQ_IMPAIRMENT_METADATA_KEY] = {
            "schema_version": 1,
            "kind": "fixed_affine_iq_frontend",
            "forward_matrix": matrix.tolist(),
            "dc_offset": offset.tolist(),
            **parameters,
            "simulation_truth": True,
            "forwarded_to_learned_runtime": False,
        }
        output_metadata["receiver_frontend"] = "fixed_iq_imbalance"
        _rewrite_uniform_capture_layout(
            output_metadata,
            capture_layout,
            int(impaired.size),
            "Receiver I/Q impairment output at %s" % ctx.step_id,
        )
        path = ctx.output_path("rx_symbols", ".npz")
        np.savez_compressed(
            path,
            symbols=impaired,
            metadata_json=json.dumps(output_metadata),
        )
        return OperationResult(
            outputs={
                "rx_symbols": artifact(
                    "channel.rx_symbols.complex_numpy",
                    path,
                    output_metadata,
                )
            },
            metrics={
                "receiver.frontend.iq_gain_imbalance_db": parameters[
                    "gain_imbalance_db"
                ],
                "receiver.frontend.quadrature_error_deg": parameters[
                    "quadrature_error_deg"
                ],
                "receiver.frontend.phase_offset_deg": parameters[
                    "phase_offset_deg"
                ],
                "receiver.frontend.dc_offset_magnitude": float(
                    np.linalg.norm(offset)
                ),
            },
            metadata={
                "receiver_frontend": "fixed_iq_imbalance",
                _RECEIVER_IQ_IMPAIRMENT_METADATA_KEY: output_metadata[
                    _RECEIVER_IQ_IMPAIRMENT_METADATA_KEY
                ],
            },
        )


_QPSK_DECISION_PREVIEW_GRID_SIZE = 64
_QPSK_DECISION_PREVIEW_LIMIT = 2.25


def _qpsk_decision_probe() -> Tuple[np.ndarray, np.ndarray]:
    """Return a fixed I/Q grid used only for comparable result evidence."""

    axis = np.linspace(
        -_QPSK_DECISION_PREVIEW_LIMIT,
        _QPSK_DECISION_PREVIEW_LIMIT,
        _QPSK_DECISION_PREVIEW_GRID_SIZE,
        dtype=np.float32,
    )
    i_grid, q_grid = np.meshgrid(axis, axis)
    features = np.stack([i_grid.reshape(-1), q_grid.reshape(-1)], axis=1)
    return axis, np.ascontiguousarray(features, dtype=np.float32)


def _qpsk_reference_decision_preview(
    *,
    receiver_mode: str,
    receiver_label: str,
) -> JsonDict:
    _axis, features = _qpsk_decision_probe()
    class_ids = (
        2 * (features[:, 0] < 0.0).astype(np.uint8)
        + (features[:, 1] < 0.0).astype(np.uint8)
    )
    return _qpsk_decision_preview(
        class_ids,
        receiver_mode=receiver_mode,
        receiver_label=receiver_label,
    )


def _qpsk_calibrated_iq_decision_preview(
    metadata: Mapping[str, Any],
    *,
    receiver_mode: str,
    receiver_label: str,
) -> JsonDict:
    _axis, features = _qpsk_decision_probe()
    probe_symbols = np.ascontiguousarray(
        features[:, 0] + 1j * features[:, 1],
        dtype=np.complex64,
    )
    compensated = _compensate_receiver_iq_impairment(probe_symbols, metadata)
    compensated_features = np.stack(
        [compensated.real, compensated.imag],
        axis=1,
    )
    class_ids = (
        2 * (compensated_features[:, 0] < 0.0).astype(np.uint8)
        + (compensated_features[:, 1] < 0.0).astype(np.uint8)
    )
    return _qpsk_decision_preview(
        class_ids,
        receiver_mode=receiver_mode,
        receiver_label=receiver_label,
    )


def _qpsk_preview_with_observed_constellation(
    preview: JsonDict,
    metadata: Mapping[str, Any],
) -> JsonDict:
    impairment = _receiver_iq_impairment_from_metadata(metadata)
    if impairment is None:
        return preview
    matrix, offset = impairment
    ideal = np.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    ) / math.sqrt(2.0)
    observed = ideal @ matrix.T + offset.reshape(1, 2)
    updated = dict(preview)
    updated["constellation"] = [
        {
            "class_id": index,
            "bits": bits,
            "i": float(point[0]),
            "q": float(point[1]),
        }
        for index, (bits, point) in enumerate(
            zip(("00", "01", "10", "11"), observed)
        )
    ]
    updated["coordinate_space"] = "impaired_received_iq"
    updated["receiver_frontend"] = {
        "kind": "fixed_affine_iq_frontend",
        "simulation_truth_forwarded_to_learned_runtime": False,
    }
    return updated


def _qpsk_decision_preview_from_logits(
    logits: np.ndarray,
    *,
    receiver_mode: str,
    receiver_label: str,
    model_sha256: str | None = None,
) -> JsonDict:
    values = np.asarray(logits, dtype=np.float32)
    expected = (_QPSK_DECISION_PREVIEW_GRID_SIZE**2, 2)
    if tuple(values.shape) != expected:
        raise OperationError(
            "QPSK decision-region logits must have shape %s, got %s"
            % (expected, tuple(values.shape))
        )
    class_ids = (
        2 * (values[:, 0] < 0.0).astype(np.uint8)
        + (values[:, 1] < 0.0).astype(np.uint8)
    )
    return _qpsk_decision_preview(
        class_ids,
        receiver_mode=receiver_mode,
        receiver_label=receiver_label,
        model_sha256=model_sha256,
    )


def _qpsk_decision_preview(
    class_ids: np.ndarray,
    *,
    receiver_mode: str,
    receiver_label: str,
    model_sha256: str | None = None,
) -> JsonDict:
    values = np.asarray(class_ids, dtype=np.uint8).reshape(-1)
    expected_size = _QPSK_DECISION_PREVIEW_GRID_SIZE**2
    if int(values.size) != expected_size or np.any(values > 3):
        raise OperationError(
            "QPSK decision-region classes must contain %d values in [0, 3]"
            % expected_size
        )
    rows = values.reshape(
        _QPSK_DECISION_PREVIEW_GRID_SIZE,
        _QPSK_DECISION_PREVIEW_GRID_SIZE,
    )
    limit = float(_QPSK_DECISION_PREVIEW_LIMIT)
    center = float(1.0 / np.sqrt(2.0))
    preview: JsonDict = {
        "schema_version": 1,
        "kind": "memoryless_qpsk_iq_decision_regions",
        "receiver_mode": str(receiver_mode),
        "receiver_label": str(receiver_label),
        "coordinate_space": "received_iq",
        "decision_rule": "negative_logit_is_bit_one",
        "grid": {
            "i_min": -limit,
            "i_max": limit,
            "q_min": -limit,
            "q_max": limit,
            "width": _QPSK_DECISION_PREVIEW_GRID_SIZE,
            "height": _QPSK_DECISION_PREVIEW_GRID_SIZE,
            "class_rows": [
                "".join(str(int(value)) for value in row.tolist()) for row in rows
            ],
        },
        "constellation": [
            {"class_id": 0, "bits": "00", "i": center, "q": center},
            {"class_id": 1, "bits": "01", "i": center, "q": -center},
            {"class_id": 2, "bits": "10", "i": -center, "q": center},
            {"class_id": 3, "bits": "11", "i": -center, "q": -center},
        ],
        "conditioning": {"kind": "none", "memoryless": True},
    }
    if model_sha256:
        preview["model_sha256"] = str(model_sha256)
    return preview


def _demodulate_source_items(
    symbols: np.ndarray,
    metadata: Mapping[str, Any],
    modulation: str,
    backend: str,
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    raw_symbol_counts = metadata.get("source_item_symbol_counts")
    raw_bit_counts = metadata.get("source_item_modulator_input_bit_counts")
    if not isinstance(raw_bit_counts, list):
        raw_bit_counts = metadata.get("source_item_coded_bit_counts")
    if isinstance(raw_symbol_counts, list) and isinstance(raw_bit_counts, list):
        symbol_counts = [int(item) for item in raw_symbol_counts]
        bit_counts = [int(item) for item in raw_bit_counts]
        if (
            symbol_counts
            and len(symbol_counts) == len(bit_counts)
            and all(item > 0 for item in symbol_counts + bit_counts)
            and sum(symbol_counts) == int(symbols.size)
        ):
            bit_rows: List[np.ndarray] = []
            llr_rows: List[np.ndarray] = []
            symbol_offset = 0
            selected_backend = "python_numpy"
            noise_variance = float(metadata.get("noise_variance") or 1.0)
            for symbol_count, bit_count in zip(symbol_counts, bit_counts):
                row_symbols = symbols[
                    symbol_offset : symbol_offset + symbol_count
                ]
                row_bits, selected_backend = _demodulate(
                    row_symbols, modulation, backend
                )
                row_llr = _approximate_llr(
                    row_symbols, modulation, row_bits, noise_variance
                )
                if bit_count > int(row_bits.size):
                    raise OperationError(
                        "Per-item demodulation bit count exceeds its symbol capacity"
                    )
                bit_rows.append(row_bits[:bit_count])
                llr_rows.append(row_llr[:bit_count])
                symbol_offset += symbol_count
            return (
                np.concatenate(bit_rows).astype(np.uint8, copy=False),
                np.concatenate(llr_rows).astype(np.float32, copy=False),
                selected_backend,
                int(sum(bit_counts)),
            )
    bits, selected_backend = _demodulate(symbols, modulation, backend)
    llr = _approximate_llr(
        symbols,
        modulation,
        bits,
        float(metadata.get("noise_variance") or 1.0),
    )
    return bits, llr, selected_backend, int(metadata.get("bit_count") or bits.size)


def _capture_demodulation_partitions(
    metadata: Mapping[str, Any],
    layout: CaptureRecordLayout | None,
    *,
    symbol_count: int,
    label: str,
) -> Tuple[List[int], List[int]] | None:
    if layout is None:
        return None
    raw_symbol_counts = metadata.get("source_item_symbol_counts")
    raw_bit_counts = metadata.get("source_item_modulator_input_bit_counts")
    if not isinstance(raw_bit_counts, list):
        raw_bit_counts = metadata.get("source_item_coded_bit_counts")
    if not isinstance(raw_symbol_counts, list) or not isinstance(
        raw_bit_counts, list
    ):
        raise OperationError(
            "%s requires per-record symbol and bit counts to preserve an explicit "
            "capture-record layout" % label
        )
    if any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in [*raw_symbol_counts, *raw_bit_counts]
    ):
        raise OperationError(
            "%s per-record symbol and bit counts must be integers" % label
        )
    symbol_counts = [int(item) for item in raw_symbol_counts]
    bit_counts = [int(item) for item in raw_bit_counts]
    if (
        len(symbol_counts) != layout.count
        or len(bit_counts) != layout.count
        or any(item <= 0 for item in symbol_counts + bit_counts)
        or any(item != layout.elements_per_record for item in symbol_counts)
        or sum(symbol_counts) != int(symbol_count)
        or len(set(bit_counts)) != 1
    ):
        raise OperationError(
            "%s per-record symbol/bit accounting conflicts with the explicit "
            "capture-record layout" % label
        )
    return symbol_counts, bit_counts


def _validate_demodulated_capture_records(
    metadata: Mapping[str, Any],
    layout: CaptureRecordLayout | None,
    *,
    symbol_count: int,
    bit_count: int,
    label: str,
) -> List[int] | None:
    partitions = _capture_demodulation_partitions(
        metadata,
        layout,
        symbol_count=symbol_count,
        label=label,
    )
    if partitions is None:
        return None
    _symbol_counts, bit_counts = partitions
    if sum(bit_counts) != int(bit_count):
        raise OperationError(
            "%s output bit count conflicts with its per-record accounting" % label
        )
    return bit_counts


def _trim_qpsk_capture_receiver_outputs(
    bits: np.ndarray,
    llr: np.ndarray,
    metadata: Mapping[str, Any],
    layout: CaptureRecordLayout | None,
    *,
    symbol_count: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, int] | None:
    partitions = _capture_demodulation_partitions(
        metadata,
        layout,
        symbol_count=symbol_count,
        label=label,
    )
    if partitions is None:
        return None
    symbol_counts, bit_counts = partitions
    raw_bits = np.asarray(bits).reshape(-1)
    raw_llr = np.asarray(llr).reshape(-1)
    if int(raw_bits.size) != int(symbol_count * 2) or int(raw_llr.size) != int(
        symbol_count * 2
    ):
        raise OperationError(
            "%s QPSK receiver output does not provide two decisions per symbol"
            % label
        )
    bit_rows: List[np.ndarray] = []
    llr_rows: List[np.ndarray] = []
    offset = 0
    for item_symbol_count, item_bit_count in zip(symbol_counts, bit_counts):
        capacity = int(item_symbol_count * 2)
        if item_bit_count > capacity:
            raise OperationError(
                "%s capture record requires more bits than its QPSK symbols carry"
                % label
            )
        bit_rows.append(raw_bits[offset : offset + item_bit_count])
        llr_rows.append(raw_llr[offset : offset + item_bit_count])
        offset += capacity
    return (
        np.concatenate(bit_rows).astype(np.uint8, copy=False),
        np.concatenate(llr_rows).astype(np.float32, copy=False),
        int(sum(bit_counts)),
    )


def _trim_qpsk_receiver_outputs(
    bits: np.ndarray,
    llr: np.ndarray,
    metadata: Mapping[str, Any],
    layout: CaptureRecordLayout | None,
    *,
    symbol_count: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove QPSK padding without crossing source-item boundaries."""

    capture_trimmed = _trim_qpsk_capture_receiver_outputs(
        bits,
        llr,
        metadata,
        layout,
        symbol_count=symbol_count,
        label=label,
    )
    if capture_trimmed is not None:
        return capture_trimmed

    raw_bits = np.asarray(bits).reshape(-1)
    raw_llr = np.asarray(llr).reshape(-1)
    expected_capacity = int(symbol_count * 2)
    if int(raw_bits.size) != expected_capacity or int(raw_llr.size) != expected_capacity:
        raise OperationError(
            "%s QPSK receiver output does not provide two decisions per symbol"
            % label
        )

    raw_symbol_counts = metadata.get("source_item_symbol_counts")
    raw_bit_counts = metadata.get("source_item_modulator_input_bit_counts")
    if not isinstance(raw_bit_counts, list):
        raw_bit_counts = metadata.get("source_item_coded_bit_counts")
    has_symbol_partition = raw_symbol_counts is not None
    has_bit_partition = raw_bit_counts is not None
    if has_symbol_partition or has_bit_partition:
        if not isinstance(raw_symbol_counts, list) or not isinstance(
            raw_bit_counts, list
        ):
            raise OperationError(
                "%s requires both source-item symbol and bit counts for QPSK "
                "padding removal" % label
            )
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in [*raw_symbol_counts, *raw_bit_counts]
        ):
            raise OperationError(
                "%s source-item symbol and bit counts must be integers" % label
            )
        symbol_counts = [int(item) for item in raw_symbol_counts]
        bit_counts = [int(item) for item in raw_bit_counts]
        if (
            not symbol_counts
            or len(symbol_counts) != len(bit_counts)
            or any(item <= 0 for item in symbol_counts + bit_counts)
            or sum(symbol_counts) != int(symbol_count)
            or any(
                bit_count > 2 * symbol_count_item
                for symbol_count_item, bit_count in zip(symbol_counts, bit_counts)
            )
        ):
            raise OperationError(
                "%s source-item QPSK symbol/bit accounting is invalid" % label
            )
        bit_rows: List[np.ndarray] = []
        llr_rows: List[np.ndarray] = []
        offset = 0
        for item_symbol_count, item_bit_count in zip(symbol_counts, bit_counts):
            bit_rows.append(raw_bits[offset : offset + item_bit_count])
            llr_rows.append(raw_llr[offset : offset + item_bit_count])
            offset += int(item_symbol_count * 2)
        return (
            np.concatenate(bit_rows).astype(np.uint8, copy=False),
            np.concatenate(llr_rows).astype(np.float32, copy=False),
            int(sum(bit_counts)),
        )

    raw_bit_count = metadata.get("bit_count")
    if raw_bit_count is None:
        bit_count = expected_capacity
    elif not isinstance(raw_bit_count, int) or isinstance(raw_bit_count, bool):
        raise OperationError("%s metadata bit_count must be an integer" % label)
    else:
        bit_count = int(raw_bit_count)
    if bit_count < 0 or bit_count > expected_capacity:
        raise OperationError(
            "%s metadata bit_count exceeds its QPSK symbol capacity" % label
        )
    return (
        raw_bits[:bit_count].astype(np.uint8, copy=False),
        raw_llr[:bit_count].astype(np.float32, copy=False),
        bit_count,
    )


def _demodulate_ofdm_qpsk_with_current_csi(
    symbols: np.ndarray,
    metadata: Mapping[str, Any],
    allocation_artifact,
    channel_state_artifact,
    capture_layout: CaptureRecordLayout | None,
    *,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, int, JsonDict]:
    """Compute positive-for-zero QPSK LLRs for a perfect-ZF OFDM receiver."""

    allocation_arrays, allocation_metadata = _load_npz_arrays(
        allocation_artifact.path
    )
    if "power" not in allocation_arrays:
        raise OperationError("%s allocation artifact is missing power" % label)
    if "modulation_order_bits_per_symbol" in allocation_arrays:
        raise OperationError(
            "%s current-CSI soft demodulation requires fixed QPSK transport"
            % label
        )
    if str(allocation_metadata.get("transport_mode") or "") != "fixed_modulation":
        raise OperationError(
            "%s current-CSI soft demodulation requires allocation "
            "transport_mode=fixed_modulation" % label
        )

    channel_state = _load_ofdm_channel_state_artifact(channel_state_artifact)
    state_metadata = channel_state["metadata"]
    csi_role = str(state_metadata.get("csi_role") or "").strip()
    if csi_role != "actual_current_channel_state":
        raise OperationError(
            "%s requires channel_state with csi_role="
            "actual_current_channel_state; got %s"
            % (label, csi_role or "<missing>")
        )
    if state_metadata.get("channel_application_state") is not True:
        raise OperationError(
            "%s actual current channel_state must declare "
            "channel_application_state=true" % label
        )
    if state_metadata.get("transmitter_visible") is not False:
        raise OperationError(
            "%s receiver channel_state must declare transmitter_visible=false"
            % label
        )
    if metadata.get("channel_equalized") is not True or str(
        metadata.get("equalizer") or ""
    ) != "perfect_csi_ofdm_zero_forcing_one_tap":
        raise OperationError(
            "%s current-CSI QPSK LLR formula requires post-ZF symbols from "
            "perfect_csi_ofdm_zero_forcing_one_tap" % label
        )
    if metadata.get("channel_state_shared") is not True:
        raise OperationError(
            "%s received symbols must declare channel_state_shared=true" % label
        )

    state_seed = state_metadata.get("channel_state_seed")
    receiver_state_seed = metadata.get("channel_state_seed")
    if state_seed is None or receiver_state_seed is None:
        raise OperationError(
            "%s requires channel_state_seed on the current state and received "
            "symbols" % label
        )
    if int(state_seed) != int(receiver_state_seed):
        raise OperationError(
            "%s received symbols and current channel_state use different "
            "channel_state_seed values" % label
        )

    h_freq = np.asarray(channel_state["h_freq"], dtype=np.complex64)
    power = np.asarray(allocation_arrays["power"], dtype=np.float64)
    expected_power_shape = (
        int(h_freq.shape[0] * h_freq.shape[1]),
        int(h_freq.shape[2]),
    )
    if power.ndim != 2 or tuple(int(value) for value in power.shape) != expected_power_shape:
        raise OperationError(
            "%s state/allocation shape mismatch: h_freq %s requires power %s, "
            "got %s"
            % (label, h_freq.shape, expected_power_shape, power.shape)
        )
    if not bool(np.all(np.isfinite(h_freq))):
        raise OperationError("%s channel_state h_freq must be finite" % label)
    if not bool(np.all(np.isfinite(power))) or bool(np.any(power < 0.0)):
        raise OperationError(
            "%s allocation power must contain finite nonnegative values" % label
        )
    if not bool(np.all(np.isfinite(symbols))):
        raise OperationError("%s received symbols must be finite" % label)
    resource_element_capacity = int(h_freq.size)
    symbol_count = int(symbols.size)
    if symbol_count > resource_element_capacity:
        raise OperationError(
            "%s received %d symbols but current state/allocation cover only %d "
            "resource elements" % (label, symbol_count, resource_element_capacity)
        )
    if metadata.get("payload_symbol_count") is not None and int(
        metadata["payload_symbol_count"]
    ) != symbol_count:
        raise OperationError(
            "%s received-symbol metadata payload_symbol_count is not aligned "
            "with the rx array" % label
        )
    if metadata.get("channel_use_count") is not None and int(
        metadata["channel_use_count"]
    ) != resource_element_capacity:
        raise OperationError(
            "%s received-symbol metadata channel_use_count is not aligned with "
            "the current OFDM state" % label
        )
    if metadata.get("grid_padding_symbol_count") is not None and int(
        metadata["grid_padding_symbol_count"]
    ) != resource_element_capacity - symbol_count:
        raise OperationError(
            "%s received-symbol metadata grid_padding_symbol_count is not "
            "aligned with the current OFDM state" % label
        )

    noise_variance = _required_finite_metadata_float(
        state_metadata,
        "noise_variance",
        "%s current channel state" % label,
        minimum=0.0,
        exclusive_minimum=True,
    )
    for owner, owner_metadata in (
        ("received symbols", metadata),
        ("power allocation", allocation_metadata),
    ):
        declared_noise = _required_finite_metadata_float(
            owner_metadata,
            "noise_variance",
            "%s %s" % (label, owner),
            minimum=0.0,
            exclusive_minimum=True,
        )
        if not math.isclose(
            declared_noise,
            noise_variance,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise OperationError(
                "%s %s noise_variance %.12g contradicts current channel-state "
                "noise_variance %.12g"
                % (label, owner, declared_noise, noise_variance)
            )

    h_flat = h_freq.reshape(-1)[:symbol_count].astype(np.complex128, copy=False)
    power_flat = power.reshape(-1)[:symbol_count]
    rx = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    llr_scale = (
        2.0
        * np.sqrt(2.0 * power_flat)
        * np.square(np.abs(h_flat))
        / noise_variance
    )
    llr_pairs = np.stack(
        [llr_scale * rx.real, llr_scale * rx.imag], axis=-1
    )
    llr_pairs[power_flat == 0.0, :] = 0.0
    if not bool(np.all(np.isfinite(llr_pairs))):
        raise OperationError("%s computed non-finite current-CSI QPSK LLRs" % label)
    raw_llr = llr_pairs.reshape(-1).astype(np.float32, copy=False)
    raw_bits = (raw_llr < 0.0).astype(np.uint8)
    bits, llr, bit_count = _trim_qpsk_receiver_outputs(
        raw_bits,
        raw_llr,
        metadata,
        capture_layout,
        symbol_count=symbol_count,
        label=label,
    )
    details: JsonDict = {
        "llr_kind": "max_log_qpsk_post_zf_current_csi",
        "llr_formula": (
            "LLR_[I,Q](k)=2*sqrt(2*p(k))*abs(h(k))^2*"
            "[Re(z(k)),Im(z(k))]/noise_variance"
        ),
        "llr_observation_model": "z(k)=sqrt(p(k))*x(k)+n(k)/h(k)",
        "llr_noise_variance_definition": (
            "noise_variance=E[abs(n)^2] at the receive antenna before ZF"
        ),
        "llr_bit_order": "resource_element_major_then_qpsk_i_bit_q_bit",
        "llr_positive_value_bit": 0,
        "llr_zero_power_behavior": "zero_llr_erasure",
        "receiver_csi_role": csi_role,
        "receiver_csi_assumption": "perfect_current_complex_frequency_response",
        "receiver_equalizer_assumption": (
            "perfect_csi_ofdm_zero_forcing_one_tap"
        ),
        "receiver_channel_state_seed": int(state_seed),
        "channel_state_aware_soft_demodulation": True,
    }
    return bits, llr, bit_count, details


class DigitalDemodulateOperation(Operation):
    id = "demodulation.digital_demodulate"
    name = "Digital symbol-to-bit demodulator"
    input_kinds = {
        "rx_symbols": [
            "channel.symbols.complex_numpy",
            "channel.rx_symbols.complex_numpy",
        ]
    }
    optional_input_kinds = {
        "allocation": ["channel.power_allocation.numpy"],
        "channel_state": ["channel.ofdm_channel_state.numpy"],
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Hard demodulation decisions stop gradients; use soft/differentiable demodulation or dataset capture mode for neural receiver training.",
    }
    backends = {"benchmark_run": ["numpy", "cpp"], "dataset_capture": ["numpy", "cpp"], "differentiable_export": []}
    materializations = _digital_modem_materializations("demodulator", include_auto=True)
    equivalence = {"type": "numerical", "tolerance": {"atol": 1e-7, "rtol": 1e-6}, "reason": "Demapper materializations should produce matching hard bits and comparable LLR-like reliability values for the same symbols."}
    formats = {"artifact": "npz", "tensor": "none"}
    output_kinds = {
        "bits": "channel.demod_bits.numpy",
        "llr": "channel.llr.numpy",
    }
    params_schema = object_schema(
        {
            "modulation": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "bpsk", "qpsk", "qam16"],
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("rx_symbols")
        symbols, metadata = _load_symbols(input_artifact.path, input_artifact.metadata)
        capture_layout = _validated_capture_layout(
            metadata,
            int(symbols.size),
            "Digital demodulator input at %s" % ctx.step_id,
        )
        modulation = str(ctx.params.get("modulation", "auto"))
        if modulation == "auto":
            modulation = str(metadata.get("modulation", "qpsk"))
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        adaptive_transport = False
        current_csi_details: JsonDict = {}
        if "channel_state" in ctx.inputs:
            if "allocation" not in ctx.inputs:
                raise OperationError(
                    "Digital demodulator channel_state input requires allocation"
                )
            if modulation != "qpsk":
                raise OperationError(
                    "Current-CSI soft demodulation requires fixed modulation=qpsk"
                )
            bits, llr, bit_count, current_csi_details = (
                _demodulate_ofdm_qpsk_with_current_csi(
                    symbols,
                    metadata,
                    ctx.inputs["allocation"],
                    ctx.inputs["channel_state"],
                    capture_layout,
                    label="Digital demodulator at %s" % ctx.step_id,
                )
            )
            selected_backend = "python_numpy"
        elif "allocation" in ctx.inputs:
            allocation_arrays, allocation_metadata = _load_npz_arrays(ctx.inputs["allocation"].path)
            if "modulation_order_bits_per_symbol" in allocation_arrays:
                adaptive_transport = True
                orders = np.asarray(
                    allocation_arrays["modulation_order_bits_per_symbol"], dtype=np.uint8
                ).reshape(-1)[: int(symbols.size)]
                power = np.asarray(allocation_arrays.get("power"), dtype=np.float64).reshape(-1)[: int(symbols.size)]
                if int(orders.size) != int(symbols.size) or int(power.size) != int(symbols.size):
                    raise OperationError(
                        "Allocation-aware demodulation map does not cover every received resource element"
                    )
                normalized_symbols = np.zeros_like(symbols, dtype=np.complex64)
                active = (orders > 0) & (power > 1e-12)
                normalized_symbols[active] = (
                    symbols[active] / np.sqrt(power[active]).astype(np.complex64, copy=False)
                ).astype(np.complex64, copy=False)
                bits, llr = _demodulate_variable_orders(normalized_symbols, orders)
                modulation = "allocation_aware"
                selected_backend = "python_numpy"
                bit_count = int(
                    allocation_metadata.get("adaptive_original_bit_count")
                    or metadata.get("bit_count")
                    or bits.size
                )
            else:
                bits, llr, selected_backend, bit_count = _demodulate_source_items(
                    symbols, metadata, modulation, backend
                )
        else:
            bits, llr, selected_backend, bit_count = _demodulate_source_items(
                symbols, metadata, modulation, backend
            )
        bits = bits[:bit_count].astype(np.uint8)
        llr = llr[:bit_count].astype(np.float32)
        _validate_demodulated_capture_records(
            metadata,
            capture_layout,
            symbol_count=int(symbols.size),
            bit_count=int(bits.size),
            label="Digital demodulator at %s" % ctx.step_id,
        )
        bit_metadata = dict(metadata)
        bit_metadata.update(
            {
                "bit_count": int(bits.size),
                "bit_role": "demodulated",
                "demodulation": modulation,
                "llr_kind": "approximate_hard_reliability",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": selected_backend,
                "allocation_aware_transport": adaptive_transport,
                **current_csi_details,
            }
        )
        _rewrite_uniform_capture_layout(
            bit_metadata,
            capture_layout,
            int(bits.size),
            "Digital demodulator output at %s" % ctx.step_id,
        )
        if modulation == "qpsk" and not adaptive_transport:
            decision_preview = _qpsk_reference_decision_preview(
                receiver_mode=(
                    "current_csi_post_zf_qpsk"
                    if current_csi_details
                    else "analytical_qpsk"
                ),
                receiver_label=(
                    "Current-CSI post-ZF QPSK demapper"
                    if current_csi_details
                    else "Analytical QPSK demapper"
                ),
            )
            if current_csi_details:
                decision_preview["conditioning"] = {
                    "kind": "per_resource_element_current_csi_and_power",
                    "channel_state_role": "actual_current_channel_state",
                    "allocation_required": True,
                }
            bit_metadata["receiver_decision_preview"] = decision_preview
        bits_path = ctx.output_path("bits", ".npz")
        llr_path = ctx.output_path("llr", ".npz")
        np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(bit_metadata))
        np.savez_compressed(llr_path, llr=llr, metadata_json=json.dumps(bit_metadata))
        result_metrics: JsonDict = {
            "channel.demod_bit_count": int(bits.size),
            "channel.allocation_aware_transport": int(adaptive_transport),
        }
        result_metadata: JsonDict = {
            "modulation": modulation,
            "allocation_aware_transport": adaptive_transport,
            "data_plane_backend": selected_backend,
        }
        if current_csi_details:
            result_metrics["channel.current_csi_soft_demodulation"] = 1
            result_metadata["channel_state_aware_soft_demodulation"] = True
        return OperationResult(
            outputs={
                "bits": artifact("channel.demod_bits.numpy", bits_path, bit_metadata),
                "llr": artifact("channel.llr.numpy", llr_path, bit_metadata),
            },
            metrics=result_metrics,
            metadata=result_metadata,
        )


class NeuralReceiverAdapterOperation(Operation):
    id = "demodulation.neural_receiver_adapter"
    name = "Neural receiver adapter"
    input_kinds = {"rx_symbols": ["channel.rx_symbols.complex_numpy"]}
    output_kinds = {
        "bits": "channel.demod_bits.numpy",
        "llr": "channel.llr.numpy",
    }
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": True,
        "exportable": True,
        "reason": "Checkpoint-backed neural receivers can be trained/exported; artifact benchmark mode records hard decisions and treats them as gradient-stopping evidence.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "reference_qpsk_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "reference_qpsk"},
            "notes": "Built-in reference QPSK receiver for smoke benchmarks.",
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "calibrated_iq_oracle_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "oracle_frontend_calibrated"},
            "notes": "Diagnostic oracle that inverts the simulated fixed I/Q front-end transform before QPSK demapping.",
        },
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "linear_npz_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "linear_npz"},
            "notes": "Tiny linear NPZ checkpoint adapter for smoke benchmarks.",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "reference_qpsk_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "reference_qpsk"},
            "notes": "Produces demodulated bits and LLR-like logits while preserving rx-symbol capture compatibility.",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "calibrated_iq_oracle_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "oracle_frontend_calibrated"},
            "notes": "Uses simulation-only front-end calibration metadata for a diagnostic upper-bound receiver.",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "linear_npz_receiver",
            "status": "implemented",
            "parameter_bindings": {"mode": "linear_npz"},
            "notes": "Runs the linear NPZ checkpoint adapter while preserving capture compatibility.",
        },
        {
            "runner": "benchmark_run",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"mode": "learned_artifact"},
            "notes": "Runs a schema-v2, hash-pinned neural-receiver artifact through its operation-owned tensor ABI.",
        },
        {
            "runner": "dataset_capture",
            "backend": "onnxruntime",
            "implementation": "portable_trained_artifact_runtime",
            "status": "implemented",
            "parameter_bindings": {"mode": "learned_artifact"},
            "notes": "The same returned artifact can be used in ordinary capture recipes without a task-specific runner.",
        },
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "neural_receiver_training_endpoint",
            "status": "implemented",
            "notes": "Training-contract export exposes this typed receiver slot; the researcher supplies the trainable module.",
        },
    ]
    equivalence = {
        "type": "behavioral",
        "reason": "External neural receiver checkpoints are compared by declared BER/BLER behavior rather than exact implementation internals.",
    }
    formats = {
        "artifact": "npz",
        "tensor": "torch.Tensor",
        "checkpoint": "npz|onnx",
    }
    trained_artifact_abi = {
        "component_id": "receiver",
        "component_role": "neural_receiver",
        "entrypoint_id": "neural_receiver",
        "required_operation_inputs": ["rx_symbols"],
        "inputs": {
            "rx_symbols_ri": {
                "dtype": "float32",
                "shape": ["symbol", 2],
            },
        },
        "outputs": {
            "bit_llr": {
                "dtype": "float32",
                "shape": ["symbol", 2],
            },
        },
        "binding_params": {
            "mode": "learned_artifact",
            "modulation": "qpsk",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "neural_receiver",
        },
    }
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "reference_qpsk",
                "enum": [
                    "reference_qpsk",
                    "oracle_frontend_calibrated",
                    "linear_npz",
                    "learned_artifact",
                ],
            },
            "modulation": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "qpsk"],
            },
            "checkpoint_path": {"type": "string", "default": ""},
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "description": "Registered schema-v2 trained artifact implementing the neural-receiver slot ABI.",
                "x-noema-ui": {
                    "control": "trained_artifact",
                    "label": "Trained artifact",
                    "accept": ".zip,.noema-artifact,.yaml,.yml,.json,application/octet-stream",
                    "visible_when": {"mode": "learned_artifact"},
                    "derived_params": [
                        "mode",
                        "modulation",
                        "artifact_manifest_path",
                        "artifact_entrypoint",
                        "artifact_package_sha256",
                    ],
                },
            },
            "artifact_entrypoint": {
                "type": "string",
                "default": "neural_receiver",
                "description": "Entrypoint in the registered trained artifact used for inference.",
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "description": "Registry-independent digest of the complete trained-artifact package bound into the execution plan.",
                "x-noema-ui": {"hidden": True},
            },
            **_backend_param(),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("rx_symbols")
        symbols, metadata = _load_symbols(input_artifact.path, input_artifact.metadata)
        symbols = _require_canonical_symbols(symbols, ctx.step_id)
        capture_layout = _validated_capture_layout(
            metadata,
            int(symbols.size),
            "Neural receiver input at %s" % ctx.step_id,
        )
        mode = str(ctx.params.get("mode") or "reference_qpsk")
        modulation = str(ctx.params.get("modulation") or "auto")
        if modulation == "auto":
            modulation = str(metadata.get("modulation") or "qpsk")
        if modulation != "qpsk":
            raise OperationError("Neural receiver adapter MVP supports qpsk only, got %s" % modulation)
        backend = dataplane.normalize_backend(ctx.params.get("data_plane_backend", "auto"))
        checkpoint_path = str(ctx.params.get("checkpoint_path") or "").strip()
        checkpoint_sha = None
        artifact_package_sha256 = ""
        decision_preview = None
        if mode == "reference_qpsk":
            bits, selected_backend = _demodulate(symbols, modulation, backend)
            llr = _approximate_llr(symbols, modulation, bits, float(metadata.get("noise_variance") or 1.0))
            frontend_impaired = (
                _receiver_iq_impairment_from_metadata(metadata) is not None
            )
            adapter_label = (
                "Uncompensated QPSK hard-decision receiver"
                if frontend_impaired
                else "Reference QPSK hard-decision receiver"
            )
            decision_preview = _qpsk_reference_decision_preview(
                receiver_mode=mode,
                receiver_label=(
                    "Uncompensated QPSK demapper"
                    if frontend_impaired
                    else "Analytical QPSK demapper"
                ),
            )
        elif mode == "oracle_frontend_calibrated":
            compensated_symbols = _compensate_receiver_iq_impairment(
                symbols, metadata
            )
            bits, selected_backend = _demodulate(
                compensated_symbols, modulation, backend
            )
            llr = _approximate_llr(
                compensated_symbols,
                modulation,
                bits,
                float(metadata.get("noise_variance") or 1.0),
            )
            adapter_label = "Calibrated I/Q oracle receiver"
            decision_preview = _qpsk_calibrated_iq_decision_preview(
                metadata,
                receiver_mode=mode,
                receiver_label=adapter_label,
            )
        elif mode == "linear_npz":
            if not checkpoint_path:
                raise OperationError("linear_npz neural receiver adapter requires params.checkpoint_path")
            path = Path(checkpoint_path).expanduser()
            if not path.exists():
                raise OperationError("Neural receiver checkpoint does not exist: %s" % path)
            with np.load(str(path), allow_pickle=False) as payload:
                if "weight" in payload:
                    weight = payload["weight"].astype(np.float32, copy=False)
                elif "weights" in payload:
                    weight = payload["weights"].astype(np.float32, copy=False)
                else:
                    raise OperationError("linear_npz checkpoint must contain a 2x2 `weight` array")
                bias = payload["bias"].astype(np.float32, copy=False) if "bias" in payload else np.zeros((2,), dtype=np.float32)
            if tuple(weight.shape) != (2, 2):
                raise OperationError("linear_npz checkpoint weight must have shape [2, 2], got %s" % (tuple(weight.shape),))
            if tuple(bias.shape) != (2,):
                raise OperationError("linear_npz checkpoint bias must have shape [2], got %s" % (tuple(bias.shape),))
            if not bool(np.all(np.isfinite(weight))):
                raise OperationError(
                    "linear_npz checkpoint weight must contain only finite values"
                )
            if not bool(np.all(np.isfinite(bias))):
                raise OperationError(
                    "linear_npz checkpoint bias must contain only finite values"
                )
            features = np.stack([symbols.real.astype(np.float32, copy=False), symbols.imag.astype(np.float32, copy=False)], axis=1)
            logits = features @ weight.T + bias.reshape(1, 2)
            if not bool(np.all(np.isfinite(logits))):
                raise OperationError(
                    "linear_npz neural receiver produced non-finite logits"
                )
            bits = (logits.reshape(-1) < 0.0).astype(np.uint8, copy=False)
            llr = logits.reshape(-1).astype(np.float32, copy=False)
            selected_backend = "python_numpy"
            checkpoint_sha = file_sha256(path)
            adapter_label = "Linear NPZ neural receiver adapter"
            _axis, probe_features = _qpsk_decision_probe()
            probe_logits = probe_features @ weight.T + bias.reshape(1, 2)
            if not bool(np.all(np.isfinite(probe_logits))):
                raise OperationError(
                    "linear_npz neural receiver decision probe produced non-finite logits"
                )
            decision_preview = _qpsk_decision_preview_from_logits(
                probe_logits,
                receiver_mode=mode,
                receiver_label=adapter_label,
                model_sha256=checkpoint_sha,
            )
        elif mode == "learned_artifact":
            from noema_lab.core.trained_artifact_runtime import (
                run_trained_artifact_entrypoint,
            )

            manifest_value = str(ctx.params.get("artifact_manifest_path") or "").strip()
            entrypoint = str(
                ctx.params.get("artifact_entrypoint") or "neural_receiver"
            ).strip()
            artifact_package_sha256 = str(
                ctx.params.get("artifact_package_sha256") or ""
            ).strip()
            if not manifest_value:
                raise OperationError(
                    "learned_artifact neural receiver requires params.artifact_manifest_path"
                )
            manifest_path = Path(manifest_value).expanduser()
            if not manifest_path.is_file():
                raise OperationError(
                    "Neural receiver trained-artifact manifest does not exist: %s"
                    % manifest_path
                )
            features = np.stack(
                [
                    symbols.real.astype(np.float32, copy=False),
                    symbols.imag.astype(np.float32, copy=False),
                ],
                axis=1,
            )
            try:
                artifact_outputs = run_trained_artifact_entrypoint(
                    manifest_path,
                    entrypoint,
                    {"rx_symbols_ri": features},
                    expected_package_sha256=artifact_package_sha256,
                )
            except Exception as exc:
                raise OperationError(
                    "Neural receiver trained-artifact inference failed: %s" % exc
                ) from exc
            if "bit_llr" not in artifact_outputs:
                raise OperationError(
                    "Neural receiver trained artifact did not return `bit_llr`"
                )
            bit_llr = np.asarray(artifact_outputs["bit_llr"], dtype=np.float32)
            expected_shape = (int(symbols.size), 2)
            if tuple(bit_llr.shape) != expected_shape:
                raise OperationError(
                    "Neural receiver artifact bit_llr must have shape %s, got %s"
                    % (expected_shape, tuple(bit_llr.shape))
                )
            if not bool(np.all(np.isfinite(bit_llr))):
                raise OperationError(
                    "Neural receiver artifact bit_llr must contain only finite values"
                )
            llr = bit_llr.reshape(-1).astype(np.float32, copy=False)
            bits = (llr < 0.0).astype(np.uint8, copy=False)
            selected_backend = "onnxruntime"
            checkpoint_path = str(manifest_path)
            checkpoint_sha = file_sha256(manifest_path)
            adapter_label = "Portable learned neural receiver"
            _axis, probe_features = _qpsk_decision_probe()
            try:
                probe_outputs = run_trained_artifact_entrypoint(
                    manifest_path,
                    entrypoint,
                    {"rx_symbols_ri": probe_features},
                    expected_package_sha256=artifact_package_sha256,
                )
            except Exception as exc:
                raise OperationError(
                    "Neural receiver decision-region probe failed: %s" % exc
                ) from exc
            if "bit_llr" not in probe_outputs:
                raise OperationError(
                    "Neural receiver decision-region probe did not return `bit_llr`"
                )
            probe_llr = np.asarray(probe_outputs["bit_llr"], dtype=np.float32)
            expected_probe_shape = (int(probe_features.shape[0]), 2)
            if tuple(probe_llr.shape) != expected_probe_shape:
                raise OperationError(
                    "Neural receiver decision-region probe bit_llr must have shape %s, got %s"
                    % (expected_probe_shape, tuple(probe_llr.shape))
                )
            if not bool(np.all(np.isfinite(probe_llr))):
                raise OperationError(
                    "Neural receiver decision-region probe returned non-finite bit_llr"
                )
            decision_preview = _qpsk_decision_preview_from_logits(
                probe_llr,
                receiver_mode=mode,
                receiver_label=adapter_label,
                model_sha256=checkpoint_sha,
            )
        else:
            raise OperationError("Unknown neural receiver adapter mode: %s" % mode)
        if decision_preview is not None:
            decision_preview = _qpsk_preview_with_observed_constellation(
                decision_preview,
                metadata,
            )
        trimmed_capture = _trim_qpsk_capture_receiver_outputs(
            bits,
            llr,
            metadata,
            capture_layout,
            symbol_count=int(symbols.size),
            label="Neural receiver at %s" % ctx.step_id,
        )
        if trimmed_capture is not None:
            bits, llr, bit_count = trimmed_capture
            declared_bit_count = metadata.get("bit_count")
            if declared_bit_count is not None and int(declared_bit_count) != bit_count:
                raise OperationError(
                    "Neural receiver per-record bit accounting conflicts with "
                    "the declared aggregate bit_count"
                )
        else:
            bit_count = int(metadata.get("bit_count") or bits.size)
        bits = bits[:bit_count].astype(np.uint8, copy=False)
        llr = llr[:bit_count].astype(np.float32, copy=False)
        receiver_runtime_backend = selected_backend
        bits, bit_validation_backend = _require_canonical_bits(
            bits,
            ctx.step_id,
            backend,
        )
        bit_metadata = dict(metadata)
        bit_metadata.update(
            {
                "bit_count": int(bits.size),
                "bit_role": "demodulated",
                "demodulation": "neural_receiver_adapter",
                "receiver_family": "neural_receiver",
                "receiver_adapter": mode,
                "receiver_adapter_label": adapter_label,
                "modulation": modulation,
                "llr_kind": "receiver_logit_or_maxlog_llr",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "data_plane_backend": bit_validation_backend,
                "bit_validation_backend": bit_validation_backend,
                "receiver_runtime_backend": receiver_runtime_backend,
            }
        )
        if checkpoint_path:
            bit_metadata["checkpoint_path"] = checkpoint_path
        if checkpoint_sha:
            bit_metadata["checkpoint_sha256"] = checkpoint_sha
            bit_metadata["artifact_manifest_sha256"] = checkpoint_sha
        if artifact_package_sha256:
            bit_metadata["artifact_package_sha256"] = artifact_package_sha256
        if decision_preview is not None:
            bit_metadata["receiver_decision_preview"] = decision_preview
        _rewrite_uniform_capture_layout(
            bit_metadata,
            capture_layout,
            int(bits.size),
            "Neural receiver output at %s" % ctx.step_id,
        )
        bits_path = ctx.output_path("bits", ".npz")
        llr_path = ctx.output_path("llr", ".npz")
        np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(bit_metadata))
        np.savez_compressed(llr_path, llr=llr, metadata_json=json.dumps(bit_metadata))
        metrics = {
            "channel.demod_bit_count": int(bits.size),
            "receiver.neural_receiver_adapter": 1,
            "receiver.neural_receiver.%s" % _metric_label(mode): 1,
        }
        return OperationResult(
            outputs={
                "bits": artifact("channel.demod_bits.numpy", bits_path, bit_metadata),
                "llr": artifact("channel.llr.numpy", llr_path, bit_metadata),
            },
            metrics=metrics,
            metadata={
                "mode": mode,
                "modulation": modulation,
                "checkpoint_sha256": checkpoint_sha,
                "artifact_manifest_sha256": checkpoint_sha,
                "artifact_package_sha256": artifact_package_sha256 or None,
                "data_plane_backend": bit_validation_backend,
                "receiver_runtime_backend": receiver_runtime_backend,
            },
        )


def _load_indices(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        indices = payload["indices"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Digital bit artifact metadata_json",
                )
            )
    _validated_capture_layout(
        metadata,
        int(np.asarray(indices).size),
        "Semantic-index artifact %s" % path,
    )
    return indices.astype(np.int64, copy=False), metadata


def _load_bits(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    bits, metadata = _load_bits_raw(path, fallback_metadata)
    canonical, _validation_metadata = validate_channel_bits(
        np.asarray(bits),
        label="channel bit artifact %s" % path,
        backend="python_numpy",
    )
    return canonical, metadata


def _load_bits_raw(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        bits = payload["bits"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Digital symbol artifact metadata_json",
                )
            )
    _validated_capture_layout(
        metadata,
        int(np.asarray(bits).size),
        "Channel bit artifact %s" % path,
    )
    return bits, metadata


def _require_canonical_bits(bits: np.ndarray, label: str, backend: str = "auto") -> Tuple[np.ndarray, str]:
    canonical, metadata = validate_channel_bits(bits, label=label, backend=backend)
    return canonical, str(metadata["data_plane_backend"])


def _require_canonical_symbols(symbols: np.ndarray, label: str) -> np.ndarray:
    canonical, _metadata = validate_channel_symbols(symbols, label=label, representation="complex")
    return canonical


def _metric_label(label: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", label.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "boundary"


def _payload_passthrough_metadata(metadata: JsonDict, stage: str, bit_count: int) -> JsonDict:
    output_metadata = dict(metadata)
    output_metadata.update(
        {
            "bit_count": bit_count,
            "bit_role": "payload",
            "payload_bit_count": int(metadata.get("payload_bit_count") or bit_count),
            "payload_passthrough": True,
            "payload_passthrough_stage": stage,
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "payload_coder": metadata.get("payload_coder") or "payload_passthrough",
            "payload_coder_type": metadata.get("payload_coder_type") or "passthrough",
            "payload_coder_label": metadata.get("payload_coder_label") or "Payload bits passthrough",
            "payload_coder_lossless": bool(metadata.get("payload_coder_lossless", True)),
        }
    )
    return output_metadata


def _pixel_count_from_metadata(metadata: JsonDict) -> int:
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, (list, tuple)) and shapes:
        total = 0
        for shape in shapes:
            if isinstance(shape, (list, tuple)) and len(shape) >= 4:
                total += int(shape[0] or 1) * int(shape[1]) * int(shape[2])
        if total > 0:
            return int(total)
    # Continuous-symbol codecs such as DeepJSCC describe their source tensor as
    # ``image_shape`` because ``shape`` belongs to the encoded symbol tensor.
    # It is still the same [N,H,W,C] source-image accounting boundary.
    shape = (
        metadata.get("original_shape")
        or metadata.get("image_shape")
        or metadata.get("storage_shape")
        or metadata.get("shape")
    )
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        return int(shape[0] or 1) * int(shape[1]) * int(shape[2])
    return 0


def _source_item_pixel_counts(
    metadata: Mapping[str, Any], item_count: int
) -> List[int] | None:
    """Resolve the source-pixel denominator for each declared source item."""

    raw_shapes = metadata.get("original_shapes")
    if isinstance(raw_shapes, (list, tuple)):
        shapes = list(raw_shapes)
        repeat_count = max(1, int(metadata.get("repeat_count") or 1))
        if len(shapes) * repeat_count == item_count and len(shapes) != item_count:
            shapes = shapes * repeat_count
        if len(shapes) == item_count:
            counts: List[int] = []
            for shape in shapes:
                if not isinstance(shape, (list, tuple)):
                    return None
                if len(shape) >= 4:
                    count = int(shape[0] or 1) * int(shape[1]) * int(shape[2])
                elif len(shape) == 3:
                    count = int(shape[0]) * int(shape[1])
                else:
                    return None
                if count <= 0:
                    return None
                counts.append(count)
            return counts
    shape = (
        metadata.get("original_shape")
        or metadata.get("image_shape")
        or metadata.get("storage_shape")
    )
    if (
        isinstance(shape, (list, tuple))
        and len(shape) >= 4
        and int(shape[0] or 1) == item_count
    ):
        per_item = int(shape[1]) * int(shape[2])
        if per_item > 0:
            return [per_item] * item_count
    return None


def _passthrough_timing(role: str, duration_s: float) -> JsonDict:
    stage = "%s.total" % role
    return {
        "schema_version": 2,
        "enabled": True,
        "scope": "operation_stage",
        "measurement_protocol": "single_execution",
        "warmup_runs": 0,
        "timed_runs": 1,
        "clock": "perf_counter",
        "runner": "payload_passthrough",
        "role": role,
        "measurements": [
            {
                "stage": stage,
                "duration_s": float(duration_s),
                "unit": "seconds",
            }
        ],
        "notes": {
            stage: "No separate payload codec is exposed by this adapter; payload bits pass through unchanged.",
        },
    }


def _fixed_bit_metrics(label: str, bit_count: int) -> JsonDict:
    canonical_bit_count = validate_rate_accounting_point("channel.fixed.%s.bit_count" % label, bit_count, "bits")
    byte_count = int(math.ceil(float(canonical_bit_count) / 8.0))
    validate_rate_accounting_point("channel.fixed.%s.byte_count" % label, byte_count, "bytes")
    return {
        "channel.fixed.%s.bit_count" % label: int(canonical_bit_count),
        "channel.fixed.%s.byte_count" % label: byte_count,
        "channel.fixed.%s.bits_per_element" % label: 1,
        "channel.fixed.%s.dtype_uint8" % label: 1,
    }


def _fixed_symbol_metrics(label: str, symbol_count: int) -> JsonDict:
    canonical_symbol_count = validate_rate_accounting_point(
        "channel.fixed.%s.symbol_count" % label,
        symbol_count,
        "symbols",
    )
    return {
        "channel.fixed.%s.symbol_count" % label: int(canonical_symbol_count),
        "channel.fixed.%s.channel_use_count" % label: int(canonical_symbol_count),
        "channel.fixed.%s.dtype_complex64" % label: 1,
    }


@_sionna_rng_serialized
def _generate_sionna_ofdm_channel_state(block_count: int, params: JsonDict, seed: int) -> JsonDict:
    if not _sionna_available():
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to generate Sionna OFDM channel state"
        )
    try:
        from sionna.phy import config as sionna_config  # type: ignore
        from sionna.phy.channel import GenerateOFDMChannel  # type: ignore
        from sionna.phy.channel.tr38901 import TDL  # type: ignore
        from sionna.phy.ofdm import ResourceGrid  # type: ignore
    except ImportError as exc:
        raise OperationError("The installed Sionna package does not expose TDL OFDM channel generation") from exc
    fft_size = max(8, int(params.get("ofdm_fft_size") or 16))
    num_ofdm_symbols = max(1, int(params.get("num_ofdm_symbols") or 4))
    subcarrier_spacing_khz = float(params.get("subcarrier_spacing_khz") or 15.0)
    carrier_frequency_ghz = float(params.get("carrier_frequency_ghz") or 3.5)
    delay_spread_ns = float(params.get("delay_spread_ns") or 100.0)
    mobility_kmh = float(params.get("mobility_kmh") or 0.0)
    tdl_model = str(params.get("tdl_model") or "A")
    sionna_config.seed = int(seed)
    resource_grid = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=subcarrier_spacing_khz * 1e3,
        precision="single",
        device="cpu",
    )
    speed_mps = mobility_kmh / 3.6
    channel_model = TDL(
        model=tdl_model,
        delay_spread=delay_spread_ns * 1e-9,
        carrier_frequency=carrier_frequency_ghz * 1e9,
        min_speed=speed_mps,
        max_speed=speed_mps,
        num_rx_ant=1,
        num_tx_ant=1,
        precision="single",
        device="cpu",
    )
    generator = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        normalize_channel=bool(params.get("normalize_channel", True)),
        precision="single",
        device="cpu",
    )
    tensor = generator(max(1, int(block_count)))
    full = tensor.detach().cpu().numpy().astype(np.complex64)
    h_freq = full[:, 0, 0, 0, 0, :, :]
    return {
        "h_freq": h_freq,
        "tdl_model": tdl_model,
        "subcarrier_spacing_khz": subcarrier_spacing_khz,
        "carrier_frequency_ghz": carrier_frequency_ghz,
        "delay_spread_ns": delay_spread_ns,
        "mobility_kmh": mobility_kmh,
    }


def _load_npz_arrays(path) -> Tuple[Dict[str, np.ndarray], JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files if name != "metadata_json"}
        metadata = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="Digital array artifact metadata_json",
            )
            if "metadata_json" in payload
            else {}
        )
    return arrays, metadata


def _load_ofdm_channel_state_artifact(input_artifact) -> JsonDict:
    arrays, metadata = _load_npz_arrays(input_artifact.path)
    if "h_freq" not in arrays:
        raise OperationError("OFDM channel-state artifact is missing h_freq")
    h_freq = np.asarray(arrays["h_freq"], dtype=np.complex64)
    if h_freq.ndim != 3:
        raise OperationError("OFDM channel-state h_freq must have shape [block, ofdm_symbol, subcarrier]")
    gains = np.maximum(np.abs(h_freq) ** 2, 1e-12).astype(np.float32)
    if "gains" in arrays:
        stored_gains = np.asarray(arrays["gains"], dtype=np.float32).reshape(-1, h_freq.shape[-1])
        expected_gains = gains.reshape(-1, h_freq.shape[-1])
        if stored_gains.shape != expected_gains.shape or not np.allclose(
            stored_gains, expected_gains, rtol=1e-5, atol=1e-7
        ):
            raise OperationError("OFDM channel-state gains do not match abs(h_freq)^2")
    csi_history = None
    if "csi_history" in arrays:
        csi_history = np.asarray(arrays["csi_history"], dtype=np.float32)
        expected_prefix = (
            int(h_freq.shape[0]),
            int(h_freq.shape[1]),
        )
        if (
            csi_history.ndim != 5
            or tuple(int(value) for value in csi_history.shape[:2])
            != expected_prefix
            or int(csi_history.shape[2]) < 1
            or int(csi_history.shape[3]) != int(h_freq.shape[2])
            or int(csi_history.shape[4]) != 2
        ):
            raise OperationError(
                "OFDM causal csi_history must have shape "
                "[block, ofdm_symbol, history, subcarrier, 2] aligned with "
                "h_freq"
            )
        if not bool(np.all(np.isfinite(csi_history))):
            raise OperationError(
                "OFDM causal csi_history must contain only finite values"
            )
        newest = (
            csi_history[:, :, -1, :, 0]
            + 1j * csi_history[:, :, -1, :, 1]
        ).astype(np.complex64, copy=False)
        if not np.allclose(
            newest, h_freq, rtol=1e-5, atol=1e-7
        ):
            raise OperationError(
                "OFDM causal csi_history newest snapshot does not match h_freq"
            )
    return {
        "h_freq": h_freq,
        "gains": gains,
        "csi_history": csi_history,
        "metadata": metadata,
    }


def _load_symbols(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    symbols, metadata = _load_symbols_raw(path, fallback_metadata)
    return symbols.astype(np.complex64, copy=False), metadata


def _load_symbols_raw(path, fallback_metadata: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        symbols = payload["symbols"]
        metadata = dict(fallback_metadata)
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Digital LLR artifact metadata_json",
                )
            )
    _validated_capture_layout(
        metadata,
        int(np.asarray(symbols).size),
        "Channel symbol artifact %s" % path,
    )
    return symbols, metadata


def _source_item_stage_bit_counts(
    metadata: Mapping[str, Any], total_count: int
) -> List[int] | None:
    declared_totals: List[str] = []
    for key in (
        "source_item_coded_bit_counts",
        "source_item_framed_bit_counts",
        "source_item_payload_bit_counts",
    ):
        if key not in metadata or metadata.get(key) is None:
            continue
        raw = metadata.get(key)
        if not isinstance(raw, list) or not raw:
            raise OperationError("%s must be a non-empty list" % key)
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in raw
        ):
            raise OperationError("%s must contain integer counts" % key)
        counts = [int(item) for item in raw]
        if any(item <= 0 for item in counts):
            raise OperationError(
                "%s must contain positive counts" % key
            )
        declared_total = int(sum(counts))
        if declared_total == int(total_count):
            return counts
        declared_totals.append("%s=%d" % (key, declared_total))
    if declared_totals:
        raise OperationError(
            "No source-item bit-count declaration matches the current %d-bit "
            "representation (%s)"
            % (int(total_count), ", ".join(declared_totals))
        )
    return None


def _coding_metadata(
    metadata: JsonDict,
    scheme: str,
    input_bit_count: int,
    output_bit_count: int,
    code_rate: float,
) -> JsonDict:
    output_metadata = dict(metadata)
    output_metadata["payload_bit_count"] = int(
        metadata.get("payload_bit_count") or metadata.get("bit_count") or input_bit_count
    )
    output_metadata.update(
        {
            "bit_count": int(output_bit_count),
            "coded_bit_count": int(output_bit_count),
            "bit_role": "coded",
            "bit_storage": "unpacked_uint8",
            "channel_bit_array": "unpacked_uint8",
            "bits_per_element": 1,
            "coding_scheme": scheme,
            "channel_code_input_bit_count": int(input_bit_count),
            "channel_code_output_bit_count": int(output_bit_count),
            "code_rate": float(code_rate),
        }
    )
    output_metadata.setdefault("coding_history", [])
    output_metadata["coding_history"] = list(output_metadata["coding_history"]) + [
        {
            "scheme": scheme,
            "input_bit_count": int(input_bit_count),
            "output_bit_count": int(output_bit_count),
            "code_rate": float(code_rate),
        }
    ]
    return output_metadata


def _channel_code_input_bit_count(metadata: JsonDict, fallback: int) -> int:
    value = metadata.get("channel_code_input_bit_count")
    if value is not None:
        return int(value)
    history = metadata.get("coding_history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict) and latest.get("input_bit_count") is not None:
            return int(latest["input_bit_count"])
    return int(fallback)


def _packet_payload_counts(bit_count: int, packet_payload_bits: int) -> list:
    if bit_count <= 0:
        return []
    counts = []
    remaining = int(bit_count)
    while remaining > 0:
        take = min(int(packet_payload_bits), remaining)
        counts.append(int(take))
        remaining -= take
    return counts


def _crc32_bits(bits: np.ndarray, valid_count: int) -> int:
    valid = np.ascontiguousarray(bits[: int(valid_count)].astype(np.uint8, copy=False))
    packed = np.packbits(valid).tobytes()
    payload = struct.pack(">I", int(valid_count)) + packed
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def _u32_to_bits(value: int) -> np.ndarray:
    data = np.frombuffer(struct.pack(">I", int(value) & 0xFFFFFFFF), dtype=np.uint8)
    return np.unpackbits(data).astype(np.uint8, copy=False)


def _bits_to_u32(bits: np.ndarray) -> int:
    array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if int(array.size) < 32:
        array = np.pad(array, (0, 32 - int(array.size)), mode="constant")
    packed = np.packbits(array[:32]).astype(np.uint8, copy=False).tobytes()
    return int(struct.unpack(">I", packed)[0])


def _transmit_bits(
    bits: np.ndarray,
    modulation: str,
    channel: str,
    snr_db: float,
    seed: int,
    backend: str = "auto",
    params: JsonDict | None = None,
) -> Tuple[np.ndarray, JsonDict]:
    params = params or {}
    rng = np.random.RandomState(seed)
    symbols, padded_bits, bits_per_symbol, mod_backend = _modulate(bits, modulation, backend)
    noisy_symbols, channel_report = _apply_wireless_channel(
        symbols,
        channel,
        snr_db,
        rng,
        params,
        backend,
        str(params.get("wireless_backend") or "auto"),
        seed,
    )
    reported_channel_backend = str(
        channel_report.get("wireless_backend")
        or channel_report.get("backend")
        or ""
    )
    if reported_channel_backend in {"python_numpy", "cpp_native", "numpy"}:
        channel_backend = "numpy"
        channel_data_plane_backend = str(
            channel_report.get("data_plane_backend")
            or (
                reported_channel_backend
                if reported_channel_backend in {"python_numpy", "cpp_native"}
                else "python_numpy"
            )
        )
        channel_backend_detail = str(
            channel_report.get("backend_detail")
            or "numpy_wireless/%s_dataplane" % channel_data_plane_backend
        )
    elif reported_channel_backend == "sionna":
        channel_backend = "sionna"
        channel_data_plane_backend = str(
            channel_report.get("data_plane_backend") or "torch"
        )
        channel_backend_detail = str(
            channel_report.get("backend_detail") or "sionna_wireless"
        )
    else:
        raise OperationError(
            "wireless.digital_link reported an unknown runtime backend: %s"
            % (reported_channel_backend or "<missing>")
        )
    rx_padded, demod_backend = _demodulate(noisy_symbols, modulation, backend)
    rx_bits = rx_padded[: bits.size].astype(np.uint8)
    error_count, ber_backend = dataplane.bit_error_count(rx_bits, bits, backend)
    ber = float(error_count) / float(bits.size) if bits.size else 0.0
    backends = {
        "modulator": mod_backend,
        "wireless_channel": channel_data_plane_backend,
        "demodulator": demod_backend,
        "ber": ber_backend,
    }
    report = {
        "ber": ber,
        "symbol_count": int(symbols.size),
        "channel_use_count": int(
            channel_report.get("executed_channel_use_count") or symbols.size
        ),
        "grid_padding_symbol_count": int(
            channel_report.get("grid_padding_symbol_count") or 0
        ),
        "bits_per_symbol": bits_per_symbol,
        "padded_bit_count": int(padded_bits.size),
        "wireless_backend": channel_backend,
        "requested_wireless_backend": channel_report[
            "requested_wireless_backend"
        ],
        "noise_variance": channel_report.get("noise_variance"),
        "channel_response_preview": channel_report.get("channel_response_preview"),
        "backend_detail": channel_backend_detail,
        "preset": channel_report.get("preset", channel),
        "data_plane_backend": _join_backends(backends),
        "data_plane_backends": backends,
    }
    for key in [
        "tx_antennas",
        "rx_antennas",
        "ofdm_fft_size",
        "num_ofdm_symbols",
        "interferers",
        "interference_sir_db",
        "carrier_frequency_ghz",
        "mobility_kmh",
        "multipath_taps",
    ]:
        value = channel_report.get(key)
        if value is not None:
            report[key] = value
    return rx_bits, report


def _modulate(bits: np.ndarray, modulation: str, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, int, str]:
    if modulation == "bpsk":
        symbols, padded, selected_backend = dataplane.bpsk_modulate(bits, backend)
        return symbols, padded, 1, selected_backend
    if modulation == "qpsk":
        symbols, padded, selected_backend = dataplane.qpsk_modulate(bits, backend)
        return symbols, padded, 2, selected_backend
    if modulation == "qam16":
        if backend == "cpp_native":
            raise OperationError("data_plane_backend=cpp_native is not implemented for qam16 modulation yet")
        padded = _pad_bits(bits, 4)
        groups = padded.reshape(-1, 4)
        i_value = groups[:, 0].astype(np.int64) * 2 + groups[:, 1].astype(np.int64)
        q_value = groups[:, 2].astype(np.int64) * 2 + groups[:, 3].astype(np.int64)
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
        symbols = (levels[i_value] + 1j * levels[q_value]) / np.sqrt(10.0)
        return symbols.astype(np.complex64), padded, 4, "python_numpy"
    raise RuntimeError("Unknown modulation: %s" % modulation)


def _apply_channel(
    symbols: np.ndarray,
    channel: str,
    snr_db: float,
    rng,
    backend: str = "auto",
    receiver_processing: str = "matched",
    fading_scope: str = "symbol",
) -> Tuple[np.ndarray, str, JsonDict]:
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / max(snr_linear, 1e-12)
    scale = float(np.sqrt(noise_var / 2.0))
    noise_real = rng.randn(symbols.size).astype(np.float32)
    noise_imag = rng.randn(symbols.size).astype(np.float32)
    if channel == "awgn":
        rx, selected_backend = dataplane.awgn_apply(symbols, noise_real, noise_imag, scale, backend)
        antenna_preview, antenna_power, antenna_total = _symbol_power_trace_metadata(rx, "rx_antenna")
        antenna_preview["kind"] = "rx_antenna_power_trace"
        return rx, selected_backend, {
            "channel_response_preview": _response_preview_from_values(
                np.ones(min(int(symbols.size), 64), dtype=np.complex64),
                "symbol",
                "unit_awgn",
            ),
            "rx_antenna_power_preview": antenna_preview,
            "rx_antenna_power_average": antenna_power,
            "rx_antenna_power_total_energy": antenna_total,
            "channel_equalized": False,
        }
    if channel == "flat_rayleigh":
        if backend == "cpp_native":
            raise OperationError("data_plane_backend=cpp_native is not implemented for flat_rayleigh channel yet")
        noise = scale * (noise_real + 1j * noise_imag)
        if fading_scope == "source_item":
            fading_value = np.complex64(
                (float(rng.randn()) + 1j * float(rng.randn())) / np.sqrt(2.0)
            )
            if abs(fading_value) < 1e-6:
                fading_value = np.complex64(1e-6 + 0j)
            fading = np.full(symbols.size, fading_value, dtype=np.complex64)
        elif fading_scope == "symbol":
            fading = (
                rng.randn(symbols.size).astype(np.float32)
                + 1j * rng.randn(symbols.size).astype(np.float32)
            ) / np.sqrt(2.0)
        else:
            raise OperationError(
                "flat_rayleigh fading_scope must be symbol or source_item"
            )
        fading[np.abs(fading) < 1e-6] = 1e-6 + 0j
        rx_antenna = (fading * symbols + noise).astype(np.complex64, copy=False)
        equalize = receiver_processing == "matched"
        rx_output = (
            (rx_antenna / fading).astype(np.complex64, copy=False)
            if equalize
            else rx_antenna
        )
        antenna_preview, antenna_power, antenna_total = _symbol_power_trace_metadata(rx_antenna, "rx_antenna")
        antenna_preview["kind"] = "rx_antenna_power_trace"
        report = {
            "channel_response_preview": _response_preview_from_values(
                fading,
                "symbol",
                (
                    "slow_rayleigh_source_item"
                    if fading_scope == "source_item"
                    else (
                        "flat_rayleigh_equalized"
                        if equalize
                        else "flat_rayleigh_raw"
                    )
                ),
            ),
            "rx_antenna_power_preview": antenna_preview,
            "rx_antenna_power_average": antenna_power,
            "rx_antenna_power_total_energy": antenna_total,
            "channel_equalized": equalize,
            "equalizer": "perfect_csi_one_tap" if equalize else None,
            "receiver_processing": receiver_processing,
            "fading_scope": fading_scope,
            "channel_gain_average": float(np.mean(np.abs(fading) ** 2)) if int(fading.size) else 0.0,
        }
        if fading_scope == "source_item" and int(fading.size):
            report.update(
                {
                    "channel_gain_real": float(fading[0].real),
                    "channel_gain_imag": float(fading[0].imag),
                    "channel_gain_magnitude": float(abs(fading[0])),
                    "channel_gain_power": float(abs(fading[0]) ** 2),
                }
            )
        if equalize:
            equalized_preview, equalized_power, equalized_total = (
                _symbol_power_trace_metadata(rx_output, "rx_equalized")
            )
            equalized_preview["kind"] = "rx_equalized_power_trace"
            report.update(
                {
                    "rx_equalized_power_preview": equalized_preview,
                    "rx_equalized_power_average": equalized_power,
                    "rx_equalized_power_total_energy": equalized_total,
                }
            )
        return rx_output, "python_numpy", report
    raise RuntimeError("Unknown channel: %s" % channel)


def _apply_wireless_channel(
    symbols: np.ndarray,
    channel: str,
    snr_db: float,
    rng,
    params: JsonDict,
    data_plane_backend: str = "auto",
    wireless_backend: str = "auto",
    seed: int = 0,
) -> Tuple[np.ndarray, JsonDict]:
    if channel not in WIRELESS_CHANNELS:
        raise RuntimeError("Unknown channel: %s" % channel)
    requested = str(wireless_backend or "auto").strip().lower()
    if requested not in WIRELESS_BACKENDS:
        raise OperationError(
            "Unknown wireless_backend %r; expected one of %s"
            % (wireless_backend, ", ".join(WIRELESS_BACKENDS))
        )
    if requested == "sionna":
        return _apply_sionna_or_raise(symbols, channel, snr_db, rng, params, seed)
    rx, report = _apply_numpy_realistic_channel(
        symbols,
        channel,
        snr_db,
        rng,
        params,
        data_plane_backend,
    )
    report["requested_wireless_backend"] = requested
    return rx, report


def _source_item_symbol_partition(
    metadata: Mapping[str, Any], total_count: int
) -> List[int] | None:
    """Return a validated source-item symbol partition when one is declared.

    A wireless realization may only be made order invariant when the modem has
    preserved exact item boundaries.  Silently guessing boundaries would make
    the reported per-item seed scope stronger than the actual data plane.
    """

    raw = metadata.get("source_item_symbol_counts")
    if not isinstance(raw, list):
        raw_shape = metadata.get("symbol_shape")
        if isinstance(raw_shape, (list, tuple)) and raw_shape:
            try:
                shape = [int(item) for item in raw_shape]
            except (TypeError, ValueError) as exc:
                raise OperationError("symbol_shape must contain integers") from exc
            item_count = shape[0]
            expected_total = int(np.prod(shape, dtype=np.int64))
            if (
                item_count > 0
                and expected_total == int(total_count)
                and int(total_count) % item_count == 0
            ):
                declared_shapes = metadata.get("original_shapes")
                image_shape = metadata.get("image_shape")
                try:
                    image_item_count = (
                        int(image_shape[0])
                        if isinstance(image_shape, (list, tuple)) and image_shape
                        else None
                    )
                except (TypeError, ValueError) as exc:
                    raise OperationError("image_shape must contain integers") from exc
                count_is_bound = bool(
                    isinstance(declared_shapes, (list, tuple))
                    and len(declared_shapes) == item_count
                ) or bool(
                    image_item_count == item_count
                )
                if count_is_bound:
                    return [int(total_count) // item_count] * item_count
        return None
    try:
        counts = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise OperationError("source_item_symbol_counts must contain integers") from exc
    if not counts or any(item <= 0 for item in counts):
        raise OperationError("source_item_symbol_counts must contain positive counts")
    if sum(counts) != int(total_count):
        raise OperationError(
            "source_item_symbol_counts do not cover the transmitted symbol stream"
        )
    return counts


def _remove_source_item_partition_metadata(
    metadata: MutableMapping[str, Any],
) -> None:
    """Drop source-item fields after a transform destroys their partition.

    Allocation-aware bit loading maps one flat coded-bit stream onto a
    variable-order OFDM schedule. A resource element can therefore cross an
    input-item boundary, so forwarding the modem's old per-item symbol counts
    would assert a partition that no longer covers the output representation.
    """

    for key in tuple(metadata):
        if str(key).startswith(("source_item_", "max_source_item_")):
            metadata.pop(key, None)


def _source_item_identity_keys(
    metadata: Mapping[str, Any], item_count: int
) -> Tuple[List[str], str | None]:
    for field in ("source_item_ids", "image_ids", "sample_ids"):
        raw = metadata.get(field)
        if not isinstance(raw, list) or len(raw) != item_count:
            continue
        keys = [str(item) for item in raw]
        if all(key for key in keys):
            return keys, field
    return ["source_item_%d" % index for index in range(item_count)], None


def _apply_wireless_channel_per_item(
    symbols: np.ndarray,
    metadata: Mapping[str, Any],
    item_partition: List[int],
    channel: str,
    snr_db: float,
    params: JsonDict,
    data_plane_backend: str,
    wireless_backend: str,
    seed: int,
) -> Tuple[np.ndarray, JsonDict, List[int]]:
    identity_keys, identity_field = _source_item_identity_keys(
        metadata, len(item_partition)
    )
    occurrences: Dict[str, int] = {}
    item_seeds: List[int] = []
    rx_rows: List[np.ndarray] = []
    reports: List[JsonDict] = []
    offset = 0
    for identity, count in zip(identity_keys, item_partition):
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        item_seed = derive_seed(
            int(seed),
            "wireless.channel.source_item",
            identity,
            "occurrence_%d" % occurrence,
        )
        row = symbols[offset : offset + count]
        row_rx, row_report = _apply_wireless_channel(
            row,
            channel,
            snr_db,
            np.random.RandomState(item_seed),
            params,
            data_plane_backend,
            wireless_backend,
            item_seed,
        )
        if int(row_rx.size) != count:
            raise OperationError(
                "Per-item wireless channel changed the source-item symbol count"
            )
        item_seeds.append(int(item_seed))
        rx_rows.append(np.asarray(row_rx, dtype=np.complex64))
        reports.append(dict(row_report))
        offset += count
    report = _aggregate_source_item_channel_reports(reports, item_partition)
    report.update(
        {
            "source_item_identity_keys": identity_keys,
            "source_item_identity_field": identity_field,
            "source_item_channel_seeds": item_seeds,
            # Without stable IDs the implementation is still isolated per item,
            # but an index key (or duplicated identity) cannot support a strict
            # batch-order invariance claim.
            "channel_realization_order_invariant": bool(
                identity_field is not None
                and len(set(identity_keys)) == len(identity_keys)
            ),
        }
    )
    return (
        np.concatenate(rx_rows).astype(np.complex64, copy=False),
        report,
        item_seeds,
    )


def _aggregate_source_item_channel_reports(
    reports: List[JsonDict], item_counts: List[int]
) -> JsonDict:
    if not reports or len(reports) != len(item_counts):
        raise OperationError("Per-item wireless channel report inventory is invalid")
    backends = {str(report.get("backend") or "python_numpy") for report in reports}
    if len(backends) != 1:
        raise OperationError("Per-item wireless channel selected inconsistent backends")
    result = dict(reports[0])
    total = float(sum(item_counts))
    weighted_keys = {
        "noise_variance",
        "channel_gain_average",
        "received_signal_power_average",
        "rx_antenna_power_average",
        "rx_equalized_power_average",
        "rx_antenna_signal_power_average",
        "rx_antenna_noise_power_average",
        "rx_antenna_component_power_average",
        "rx_antenna_signal_noise_cross_power_average",
        "post_equalizer_signal_power_average",
        "post_equalizer_noise_power_average",
        "post_equalizer_component_power_average",
        "post_equalizer_signal_noise_cross_power_average",
    }
    total_keys = {
        "rx_antenna_power_total_energy",
        "rx_equalized_power_total_energy",
    }
    for key in weighted_keys:
        values = [report.get(key) for report in reports]
        if all(isinstance(value, (int, float)) for value in values):
            result[key] = float(
                sum(float(value) * count for value, count in zip(values, item_counts))
                / total
            )
    for key in total_keys:
        values = [report.get(key) for report in reports]
        if all(isinstance(value, (int, float)) for value in values):
            result[key] = float(sum(float(value) for value in values))
    for key in (
        "channel_gain_real",
        "channel_gain_imag",
        "channel_gain_magnitude",
        "channel_gain_power",
    ):
        values = [report.get(key) for report in reports]
        if all(isinstance(value, (int, float)) for value in values):
            result["source_item_%s" % key] = [
                float(value) for value in values
            ]
    result["source_item_report_count"] = len(reports)
    executed_counts = [
        int(report.get("executed_channel_use_count") or item_count)
        for report, item_count in zip(reports, item_counts)
    ]
    if any(value < count for value, count in zip(executed_counts, item_counts)):
        raise OperationError(
            "Wireless channel reported fewer executed uses than payload symbols"
        )
    result.update(
        {
            "source_item_executed_channel_use_counts": executed_counts,
            "payload_symbol_count": int(sum(item_counts)),
            "executed_channel_use_count": int(sum(executed_counts)),
            "grid_padding_symbol_count": int(
                sum(executed_counts) - sum(item_counts)
            ),
        }
    )
    return result


def _apply_sionna_or_raise(
    symbols: np.ndarray,
    channel: str,
    snr_db: float,
    rng,
    params: JsonDict,
    seed: int,
) -> Tuple[np.ndarray, JsonDict]:
    if not _sionna_available():
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna wireless backend"
        )
    try:
        return _apply_sionna_channel(symbols, channel, snr_db, rng, params, seed)
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("Sionna wireless backend failed for %s: %s" % (channel, exc)) from exc


def _apply_sionna_channel(
    symbols: np.ndarray,
    channel: str,
    snr_db: float,
    rng,
    params: JsonDict,
    seed: int,
) -> Tuple[np.ndarray, JsonDict]:
    if channel in {"ofdm_tdl", "ofdm_cdl", "urban_micro", "interference_awgn"}:
        raise OperationError(
            "Sionna wireless backend is currently registered for awgn, flat_rayleigh, and mimo_flat; "
            "use wireless_backend=numpy for the %s preset until a native Sionna OFDM/interference adapter is added"
            % channel
        )
    if channel == "awgn":
        rx, report = _apply_sionna_awgn(symbols, snr_db, seed)
        report.update({"preset": "awgn"})
        return rx, report
    if channel in {"flat_rayleigh", "mimo_flat"}:
        tx_antennas = max(1, int(params.get("tx_antennas") or 1))
        rx_antennas = max(1, int(params.get("rx_antennas") or (2 if channel == "mimo_flat" else 1)))
        receiver_processing = str(
            params.get("receiver_processing") or "matched"
        )
        rx, report = _apply_sionna_mimo_flat(
            symbols,
            snr_db,
            tx_antennas,
            rx_antennas,
            seed,
            equalize=receiver_processing == "matched",
        )
        report.update({"preset": channel})
        return rx, report
    raise RuntimeError("Unknown channel: %s" % channel)


@_sionna_rng_serialized
def _apply_sionna_realized_ofdm_channel(
    symbols: np.ndarray,
    h_freq: np.ndarray,
    noise_variance: float,
    params: JsonDict,
    seed: int,
) -> Tuple[np.ndarray, JsonDict]:
    if not _sionna_available():
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna OFDM backend"
        )
    try:
        from sionna.phy import config as sionna_config  # type: ignore
        from sionna.phy.channel import ApplyOFDMChannel  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise OperationError("The installed Sionna package does not expose its OFDM channel blocks") from exc
    state = np.asarray(h_freq, dtype=np.complex64)
    if state.ndim != 3:
        raise OperationError("Sionna OFDM frequency response must have shape [block, ofdm_symbol, subcarrier]")
    block_count, num_ofdm_symbols, fft_size = state.shape
    capacity = int(block_count * num_ofdm_symbols * fft_size)
    padded = _pad_complex(np.asarray(symbols, dtype=np.complex64).reshape(-1), capacity)
    if int(padded.size) != capacity:
        raise OperationError(
            "Sionna OFDM state covers %d symbols but wireless.channel received %d"
            % (capacity, int(symbols.size))
        )
    x = padded.reshape(block_count, 1, 1, num_ofdm_symbols, fft_size)
    h = state.reshape(block_count, 1, 1, 1, 1, num_ofdm_symbols, fft_size)
    sionna_config.seed = int(seed)
    layer = ApplyOFDMChannel(precision="single", device="cpu")
    y_tensor = layer(
        torch.as_tensor(x, dtype=torch.complex64),
        torch.as_tensor(h, dtype=torch.complex64),
        torch.as_tensor(float(noise_variance), dtype=torch.float32),
    )
    antenna = y_tensor.detach().cpu().numpy().astype(np.complex64)[:, 0, 0, :, :]
    transmitted_grid = padded.reshape(block_count, num_ofdm_symbols, fft_size)
    antenna_signal_grid = transmitted_grid * state
    antenna_noise_grid = antenna - antenna_signal_grid
    safe_h = state.copy()
    safe_h[np.abs(safe_h) < 1e-8] = np.complex64(1e-8 + 0j)
    equalized_grid = antenna / safe_h
    equalized_signal_grid = antenna_signal_grid / safe_h
    equalized_noise_grid = antenna_noise_grid / safe_h
    equalized = equalized_grid.reshape(-1)[: symbols.size].astype(np.complex64, copy=False)
    valid_antenna_signal = antenna_signal_grid.reshape(-1)[: symbols.size]
    valid_antenna_noise = antenna_noise_grid.reshape(-1)[: symbols.size]
    valid_equalized_signal = equalized_signal_grid.reshape(-1)[: symbols.size]
    valid_equalized_noise = equalized_noise_grid.reshape(-1)[: symbols.size]
    antenna_signal_power = (
        float(np.mean(np.abs(valid_antenna_signal) ** 2)) if int(symbols.size) else 0.0
    )
    antenna_noise_power = (
        float(np.mean(np.abs(valid_antenna_noise) ** 2)) if int(symbols.size) else 0.0
    )
    equalized_signal_power = (
        float(np.mean(np.abs(valid_equalized_signal) ** 2)) if int(symbols.size) else 0.0
    )
    equalized_noise_power = (
        float(np.mean(np.abs(valid_equalized_noise) ** 2)) if int(symbols.size) else 0.0
    )
    antenna_preview, antenna_power, antenna_total = _symbol_power_trace_metadata(
        antenna.reshape(-1)[: symbols.size], "rx_antenna"
    )
    equalized_preview, equalized_power, equalized_total = _symbol_power_trace_metadata(
        equalized, "rx_equalized"
    )
    antenna_preview["kind"] = "rx_antenna_power_trace"
    equalized_preview["kind"] = "rx_equalized_power_trace"
    return equalized, {
        "backend": "sionna",
        "backend_detail": "sionna.tdl.GenerateOFDMChannel+ApplyOFDMChannel.pytorch.perfect_csi",
        "data_plane_backend": "torch",
        "preset": "ofdm_tdl",
        "noise_variance": float(noise_variance),
        "channel_response_preview": _response_preview_from_values(
            state.reshape(-1), "subcarrier", "sionna_tdl_ofdm_frequency_response"
        ),
        "rx_antenna_power_preview": antenna_preview,
        "rx_antenna_power_average": antenna_power,
        "rx_antenna_power_total_energy": antenna_total,
        "rx_antenna_signal_power_average": antenna_signal_power,
        "rx_antenna_noise_power_average": antenna_noise_power,
        "rx_antenna_component_power_average": antenna_signal_power + antenna_noise_power,
        "rx_antenna_signal_noise_cross_power_average": (
            antenna_power - antenna_signal_power - antenna_noise_power
        ),
        "rx_equalized_power_preview": equalized_preview,
        "rx_equalized_power_average": equalized_power,
        "rx_equalized_power_total_energy": equalized_total,
        "post_equalizer_signal_power_average": equalized_signal_power,
        "post_equalizer_noise_power_average": equalized_noise_power,
        "post_equalizer_component_power_average": equalized_signal_power + equalized_noise_power,
        "post_equalizer_signal_noise_cross_power_average": (
            equalized_power - equalized_signal_power - equalized_noise_power
        ),
        "channel_equalized": True,
        "equalizer": "perfect_csi_ofdm_zero_forcing_one_tap",
        "channel_gain_average": float(np.mean(np.abs(state) ** 2)),
        "received_signal_power_average": antenna_signal_power,
        "ofdm_fft_size": int(fft_size),
        "num_ofdm_symbols": int(num_ofdm_symbols),
        "multipath_taps": None,
        "subcarrier_spacing_khz": float(params.get("subcarrier_spacing_khz") or 15.0),
        "carrier_frequency_ghz": float(params.get("carrier_frequency_ghz") or 3.5),
        "mobility_kmh": float(params.get("mobility_kmh") or 0.0),
        "payload_symbol_count": int(symbols.size),
        "executed_channel_use_count": int(capacity),
        "grid_padding_symbol_count": int(capacity - symbols.size),
        "ofdm_grid_block_count": int(block_count),
    }


@_sionna_rng_serialized
def _apply_sionna_awgn(symbols: np.ndarray, snr_db: float, seed: int) -> Tuple[np.ndarray, JsonDict]:
    try:
        from sionna.phy.channel import AWGN  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna wireless backend"
        ) from exc
    _set_sionna_seed(seed)
    noise_var = _noise_variance_from_snr(snr_db)
    layer = AWGN()
    try:
        y = layer(
            torch.as_tensor(symbols.astype(np.complex64), dtype=torch.complex64),
            torch.as_tensor(noise_var, dtype=torch.float32),
        )
        output = y.detach().cpu().numpy().astype(np.complex64)
    except Exception as exc:
        raise OperationError(
            "Sionna PyTorch AWGN execution failed: %s" % exc
        ) from exc
    if not bool(
        np.all(np.isfinite(output.real)) and np.all(np.isfinite(output.imag))
    ):
        raise OperationError("Sionna PyTorch AWGN returned non-finite symbols")
    return output, {
        "backend": "sionna",
        "backend_detail": "sionna.awgn.pytorch",
        "data_plane_backend": "torch",
        "requested_wireless_backend": "sionna",
        "noise_variance": noise_var,
        "channel_response_preview": _response_preview_from_values(
            np.ones(min(int(symbols.size), 64), dtype=np.complex64),
            "symbol",
            "unit_awgn",
        ),
    }


@_sionna_rng_serialized
def _apply_sionna_mimo_flat(
    symbols: np.ndarray,
    snr_db: float,
    tx_antennas: int,
    rx_antennas: int,
    seed: int,
    *,
    equalize: bool = True,
) -> Tuple[np.ndarray, JsonDict]:
    try:
        from sionna.phy.channel import FlatFadingChannel  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna wireless backend"
        ) from exc
    _set_sionna_seed(seed)
    noise_var = _noise_variance_from_snr(snr_db)
    stream_count = max(1, int(tx_antennas))
    rx_count = max(1, int(rx_antennas))
    x_np = np.zeros((symbols.size, stream_count), dtype=np.complex64)
    x_np[:, 0] = symbols.astype(np.complex64)
    channel = FlatFadingChannel(stream_count, rx_count, return_channel=True)
    try:
        y, h = channel(
            torch.as_tensor(x_np, dtype=torch.complex64),
            torch.as_tensor(noise_var, dtype=torch.float32),
        )
    except Exception as exc:
        raise OperationError(
            "Sionna PyTorch flat-fading execution failed: %s" % exc
        ) from exc
    y_np = y.detach().cpu().numpy().astype(np.complex64)
    h_np = h.detach().cpu().numpy().astype(np.complex64)
    if not bool(
        np.all(np.isfinite(y_np.real))
        and np.all(np.isfinite(y_np.imag))
        and np.all(np.isfinite(h_np.real))
        and np.all(np.isfinite(h_np.imag))
    ):
        raise OperationError(
            "Sionna PyTorch flat-fading channel returned non-finite tensors"
        )
    if not equalize and (stream_count != 1 or rx_count != 1):
        raise OperationError(
            "Unprocessed Sionna flat fading requires one transmit and one receive antenna"
        )
    recovered = (
        _perfect_mimo_equalize(y_np, h_np)
        if equalize
        else y_np.reshape(-1)
    )
    antenna_preview, antenna_power, antenna_total = _symbol_power_trace_metadata(y_np, "rx_antenna")
    antenna_preview["kind"] = "rx_antenna_power_trace"
    report = {
        "backend": "sionna",
        "backend_detail": (
            "sionna.flat_fading_perfect_csi.pytorch"
            if equalize
            else "sionna.flat_fading_unprocessed.pytorch"
        ),
        "data_plane_backend": "torch",
        "requested_wireless_backend": "sionna",
        "noise_variance": noise_var,
        "channel_response_preview": _response_preview_from_values(
            _effective_mimo_response(h_np),
            "symbol",
            "mimo_flat_perfect_csi",
        ),
        "rx_antenna_power_preview": antenna_preview,
        "rx_antenna_power_average": antenna_power,
        "rx_antenna_power_total_energy": antenna_total,
        "channel_equalized": bool(equalize),
        "equalizer": "perfect_csi_mimo" if equalize else None,
        "receiver_processing": "matched" if equalize else "none",
        "channel_gain_average": float(np.mean(np.abs(h_np) ** 2)) if int(h_np.size) else 0.0,
        "tx_antennas": int(tx_antennas),
        "rx_antennas": int(rx_antennas),
    }
    if equalize:
        equalized_preview, equalized_power, equalized_total = (
            _symbol_power_trace_metadata(recovered, "rx_equalized")
        )
        equalized_preview["kind"] = "rx_equalized_power_trace"
        report.update(
            {
                "rx_equalized_power_preview": equalized_preview,
                "rx_equalized_power_average": equalized_power,
                "rx_equalized_power_total_energy": equalized_total,
            }
        )
    return recovered.astype(np.complex64), report


def _apply_numpy_realistic_channel(
    symbols: np.ndarray,
    channel: str,
    snr_db: float,
    rng,
    params: JsonDict,
    data_plane_backend: str = "auto",
) -> Tuple[np.ndarray, JsonDict]:
    receiver_processing = str(params.get("receiver_processing") or "matched")
    if channel in {"awgn", "flat_rayleigh"}:
        rx, backend, response_report = _apply_channel(
            symbols,
            channel,
            snr_db,
            rng,
            data_plane_backend,
            receiver_processing,
            str(params.get("fading_scope") or "symbol"),
        )
        return rx, {
            "backend": backend,
            "backend_detail": backend,
            "preset": channel,
            "noise_variance": _noise_variance_from_snr(snr_db),
            **response_report,
        }
    if channel == "interference_awgn":
        interferers = max(1, int(params.get("interferers") or 1))
        sir_db = float(params.get("interference_sir_db") if params.get("interference_sir_db") is not None else 18.0)
        rx = _add_awgn(symbols, snr_db, rng)
        rx = _add_interference(rx, sir_db, interferers, rng)
        return rx.astype(np.complex64), {
            "backend": "python_numpy",
            "backend_detail": "numpy_awgn_plus_cochannel_interference",
            "preset": channel,
            "noise_variance": _noise_variance_from_snr(snr_db),
            "channel_response_preview": _response_preview_from_values(
                np.ones(min(int(symbols.size), 64), dtype=np.complex64),
                "symbol",
                "unit_awgn_plus_interference",
            ),
            "interferers": interferers,
            "interference_sir_db": sir_db,
            "channel_equalized": False,
            "equalizer": None,
            "receiver_processing": receiver_processing,
        }
    if channel == "mimo_flat":
        tx_antennas = max(1, int(params.get("tx_antennas") or 1))
        rx_antennas = max(1, int(params.get("rx_antennas") or 2))
        rx, response, antenna = _numpy_mimo_flat(symbols, snr_db, tx_antennas, rx_antennas, rng)
        antenna_preview, antenna_power, antenna_total = _symbol_power_trace_metadata(antenna, "rx_antenna")
        equalized_preview, equalized_power, equalized_total = _symbol_power_trace_metadata(rx, "rx_equalized")
        antenna_preview["kind"] = "rx_antenna_power_trace"
        equalized_preview["kind"] = "rx_equalized_power_trace"
        return rx.astype(np.complex64), {
            "backend": "python_numpy",
            "backend_detail": "numpy_flat_mimo_perfect_csi",
            "preset": channel,
            "noise_variance": _noise_variance_from_snr(snr_db),
            "channel_response_preview": _response_preview_from_values(response, "symbol", "mimo_flat_perfect_csi"),
            "rx_antenna_power_preview": antenna_preview,
            "rx_antenna_power_average": antenna_power,
            "rx_antenna_power_total_energy": antenna_total,
            "rx_equalized_power_preview": equalized_preview,
            "rx_equalized_power_average": equalized_power,
            "rx_equalized_power_total_energy": equalized_total,
            "channel_equalized": True,
            "equalizer": "perfect_csi_mimo",
            "channel_gain_average": float(np.mean(np.abs(response) ** 2)) if int(response.size) else 0.0,
            "tx_antennas": tx_antennas,
            "rx_antennas": rx_antennas,
        }
    if channel in {"ofdm_tdl", "ofdm_cdl", "urban_micro"}:
        rx, report = _numpy_ofdm_multipath(symbols, channel, snr_db, rng, params)
        if channel == "urban_micro":
            interferers = max(1, int(params.get("interferers") or 1))
            sir_db = float(params.get("interference_sir_db") if params.get("interference_sir_db") is not None else 20.0)
            rx = _add_interference(rx, sir_db, interferers, rng)
            report.update({"interferers": interferers, "interference_sir_db": sir_db})
        return rx.astype(np.complex64), report
    raise RuntimeError("Unknown channel: %s" % channel)


def _noise_variance_from_snr(snr_db: float) -> float:
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    return 1.0 / max(snr_linear, 1e-12)


def _add_awgn(symbols: np.ndarray, snr_db: float, rng) -> np.ndarray:
    noise_var = _noise_variance_from_snr(snr_db)
    scale = float(np.sqrt(noise_var / 2.0))
    noise = scale * (rng.randn(symbols.size).astype(np.float32) + 1j * rng.randn(symbols.size).astype(np.float32))
    return (symbols + noise).astype(np.complex64)


def _add_interference(symbols: np.ndarray, sir_db: float, interferers: int, rng) -> np.ndarray:
    signal_power = float(np.mean(np.abs(symbols) ** 2)) if symbols.size else 1.0
    total_interference_power = signal_power / max(10.0 ** (float(sir_db) / 10.0), 1e-12)
    if total_interference_power <= 0.0:
        return symbols.astype(np.complex64)
    scale = math.sqrt(total_interference_power / max(int(interferers), 1) / 2.0)
    interference = np.zeros(symbols.shape, dtype=np.complex64)
    n = np.arange(symbols.size, dtype=np.float32)
    for index in range(max(int(interferers), 1)):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        offset = rng.uniform(-0.04, 0.04)
        tone = np.exp(1j * (2.0 * math.pi * offset * n + phase)).astype(np.complex64)
        random_symbols = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=symbols.size).astype(np.float32)
        interference += scale * random_symbols.astype(np.complex64) * tone
    return (symbols + interference).astype(np.complex64)


def _numpy_mimo_flat(symbols: np.ndarray, snr_db: float, tx_antennas: int, rx_antennas: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx_count = max(1, int(tx_antennas))
    rx_count = max(1, int(rx_antennas))
    x = np.zeros((symbols.size, tx_count), dtype=np.complex64)
    x[:, 0] = symbols.astype(np.complex64)
    h = (
        rng.randn(symbols.size, rx_count, tx_count).astype(np.float32)
        + 1j * rng.randn(symbols.size, rx_count, tx_count).astype(np.float32)
    ) / np.sqrt(2.0)
    noise_var = _noise_variance_from_snr(snr_db)
    noise_scale = float(np.sqrt(noise_var / 2.0))
    y = np.einsum("nrt,nt->nr", h, x)
    noise = noise_scale * (
        rng.randn(symbols.size, rx_count).astype(np.float32)
        + 1j * rng.randn(symbols.size, rx_count).astype(np.float32)
    )
    antenna = y + noise
    return _perfect_mimo_equalize(antenna, h), _effective_mimo_response(h), antenna


def _perfect_mimo_equalize(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    recovered = np.empty((y.shape[0],), dtype=np.complex64)
    for index in range(y.shape[0]):
        estimate = np.linalg.pinv(h[index]).dot(y[index])
        recovered[index] = estimate[0]
    return recovered


def _numpy_ofdm_multipath(symbols: np.ndarray, channel: str, snr_db: float, rng, params: JsonDict) -> Tuple[np.ndarray, JsonDict]:
    fft_size = max(8, int(params.get("ofdm_fft_size") or 64))
    num_symbols = max(1, int(params.get("num_ofdm_symbols") or 14))
    grid_size = fft_size * num_symbols
    padded = _pad_complex(symbols.astype(np.complex64), grid_size)
    grid = padded.reshape(-1, num_symbols, fft_size)
    if channel == "ofdm_cdl":
        power_profile = np.array([0.0, -1.5, -4.0, -7.0, -11.0, -16.0], dtype=np.float32)
    elif channel == "urban_micro":
        power_profile = np.array([0.0, -2.0, -5.0, -9.0, -14.0], dtype=np.float32)
    else:
        power_profile = np.array([0.0, -3.0, -6.0, -9.0], dtype=np.float32)
    tap_power = 10.0 ** (power_profile / 10.0)
    tap_power = tap_power / np.sum(tap_power)
    taps = (
        rng.randn(grid.shape[0], len(tap_power)).astype(np.float32)
        + 1j * rng.randn(grid.shape[0], len(tap_power)).astype(np.float32)
    ) * np.sqrt(tap_power[None, :] / 2.0)
    h_freq = np.fft.fft(taps, n=fft_size, axis=1).astype(np.complex64)
    h_freq[np.abs(h_freq) < 1e-8] = 1e-6 + 0j
    faded = grid * h_freq[:, None, :]
    noisy = _add_awgn(faded.reshape(-1), snr_db, rng).reshape(faded.shape)
    equalized = noisy / h_freq[:, None, :]
    mobility_kmh = float(params.get("mobility_kmh") if params.get("mobility_kmh") is not None else (30.0 if channel == "urban_micro" else 3.0))
    if mobility_kmh > 0:
        phase_step = (mobility_kmh / 3.6) * 1e-4
        phase = np.exp(1j * phase_step * np.arange(equalized.size, dtype=np.float32)).reshape(equalized.shape)
        equalized = equalized * phase.astype(np.complex64)
    return equalized.reshape(-1)[: symbols.size].astype(np.complex64), {
        "backend": "python_numpy",
        "backend_detail": "numpy_ofdm_frequency_selective_perfect_csi",
        "preset": channel,
        "noise_variance": _noise_variance_from_snr(snr_db),
        "channel_response_preview": _response_preview_from_values(
            h_freq.reshape(-1),
            "subcarrier",
            "ofdm_frequency_response",
        ),
        "ofdm_fft_size": fft_size,
        "num_ofdm_symbols": num_symbols,
        "multipath_taps": int(len(tap_power)),
        "subcarrier_spacing_khz": float(params.get("subcarrier_spacing_khz") or 15.0),
        "carrier_frequency_ghz": float(params.get("carrier_frequency_ghz") or 3.5),
        "mobility_kmh": mobility_kmh,
        "payload_symbol_count": int(symbols.size),
        "executed_channel_use_count": int(padded.size),
        "grid_padding_symbol_count": int(padded.size - symbols.size),
        "ofdm_grid_block_count": int(grid.shape[0]),
    }


def _pad_complex(values: np.ndarray, width: int) -> np.ndarray:
    remainder = values.size % width
    if remainder == 0:
        return values
    return np.pad(values, (0, width - remainder), mode="constant").astype(np.complex64)


def _effective_mimo_response(h: np.ndarray) -> np.ndarray:
    array = np.asarray(h, dtype=np.complex64)
    if array.ndim == 3:
        branch = array[:, :, 0].reshape(array.shape[0], -1)
        magnitude = np.sqrt(np.mean(np.abs(branch) ** 2, axis=1)).astype(np.float32)
        return magnitude.astype(np.complex64)
    return array.reshape(-1).astype(np.complex64)


def _response_preview_from_values(values: np.ndarray, x_axis: str, kind: str, limit: int = 64) -> JsonDict:
    array = np.asarray(values, dtype=np.complex64).reshape(-1)
    if array.size == 0:
        return {"kind": kind, "x_axis": x_axis, "magnitude": [], "phase_rad": []}
    if array.size > limit:
        indices = np.linspace(0, array.size - 1, num=limit).round().astype(np.int64)
        sample = array[indices]
    else:
        indices = np.arange(array.size, dtype=np.int64)
        sample = array
    return {
        "kind": kind,
        "x_axis": x_axis,
        "source_count": int(array.size),
        "indices": [int(item) for item in indices.tolist()],
        "magnitude": [float(item) for item in np.abs(sample).astype(np.float32).tolist()],
        "phase_rad": [float(item) for item in np.angle(sample).astype(np.float32).tolist()],
    }


def _sionna_available() -> bool:
    version = installed_dependency_version("sionna") or ""
    major = version.split(".", 1)[0]
    return (
        importlib.util.find_spec("sionna") is not None
        and importlib.util.find_spec("torch") is not None
        and major.isdigit()
        and int(major) >= 2
    )


def _set_sionna_seed(seed: int) -> None:
    """Reset Sionna's random streams for a reproducible operation realization."""
    try:
        from sionna.phy import config as sionna_config  # type: ignore
    except ImportError as exc:
        raise OperationError(
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna 2.x wireless backend"
        ) from exc
    sionna_config.seed = int(seed)


def _sionna_availability(optional: bool = False) -> JsonDict:
    if _sionna_available():
        return {"available": True, "extra": "wireless", "missing": []}
    payload = {
        "available": False,
        "extra": "wireless",
        "missing": ["sionna"],
        "reason": (
            'Install optional dependencies with `python -m pip install "noema-lab[wireless]"` '
            "in an installed environment, or `uv sync --extra wireless` from a source checkout, "
            "to use the Sionna wireless backend"
        ),
    }
    if optional:
        payload["optional"] = True
    return payload


def _demodulate(symbols: np.ndarray, modulation: str, backend: str = "auto") -> Tuple[np.ndarray, str]:
    if modulation == "bpsk":
        return dataplane.bpsk_demodulate(symbols, backend)
    if modulation == "qpsk":
        return dataplane.qpsk_demodulate(symbols, backend)
    if modulation == "qam16":
        if backend == "cpp_native":
            raise OperationError("data_plane_backend=cpp_native is not implemented for qam16 demodulation yet")
        scaled = symbols * np.sqrt(10.0)
        i_values = _nearest_qam16_values(scaled.real)
        q_values = _nearest_qam16_values(scaled.imag)
        bits = np.empty((symbols.size, 4), dtype=np.uint8)
        bits[:, 0] = (i_values // 2).astype(np.uint8)
        bits[:, 1] = (i_values % 2).astype(np.uint8)
        bits[:, 2] = (q_values // 2).astype(np.uint8)
        bits[:, 3] = (q_values % 2).astype(np.uint8)
        return bits.reshape(-1), "python_numpy"
    raise RuntimeError("Unknown modulation: %s" % modulation)


def _join_backends(backends: JsonDict) -> str:
    values = [str(value) for value in backends.values()]
    first = values[0] if values else "python_numpy"
    if all(value == first for value in values):
        return first
    return ",".join("%s=%s" % (key, value) for key, value in backends.items())


def _approximate_llr(
    symbols: np.ndarray,
    modulation: str,
    hard_bits: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    maxlog = _maxlog_llr(symbols, modulation, noise_variance)
    if maxlog is not None:
        return maxlog
    scale = 2.0 / max(float(noise_variance), 1e-12)
    if modulation == "bpsk":
        reliability = np.abs(symbols.real) * scale
    elif modulation == "qpsk":
        reliability = np.empty((symbols.size, 2), dtype=np.float32)
        reliability[:, 0] = np.abs(symbols.real) * scale
        reliability[:, 1] = np.abs(symbols.imag) * scale
        reliability = reliability.reshape(-1)
    elif modulation == "qam16":
        scaled = symbols * np.sqrt(10.0)
        boundary_distance_i = np.minimum.reduce(
            [
                np.abs(scaled.real + 2.0),
                np.abs(scaled.real),
                np.abs(scaled.real - 2.0),
            ]
        )
        boundary_distance_q = np.minimum.reduce(
            [
                np.abs(scaled.imag + 2.0),
                np.abs(scaled.imag),
                np.abs(scaled.imag - 2.0),
            ]
        )
        reliability = np.empty((symbols.size, 4), dtype=np.float32)
        reliability[:, 0] = np.abs(scaled.real) * scale
        reliability[:, 1] = boundary_distance_i * scale
        reliability[:, 2] = np.abs(scaled.imag) * scale
        reliability[:, 3] = boundary_distance_q * scale
        reliability = reliability.reshape(-1)
    else:
        raise RuntimeError("Unknown modulation: %s" % modulation)
    signs = 1.0 - 2.0 * hard_bits.astype(np.float32)
    return signs * reliability.astype(np.float32)


def _maxlog_llr(symbols: np.ndarray, modulation: str, noise_variance: float) -> np.ndarray | None:
    constellation, labels = _constellation_numpy(modulation)
    if constellation is None or labels is None:
        return None
    rx = symbols.astype(np.complex64, copy=False).reshape(-1)
    distances = np.abs(rx[:, None] - constellation[None, :]) ** 2
    variance = max(float(noise_variance), 1e-12)
    llrs = []
    for bit_index in range(labels.shape[1]):
        mask0 = labels[:, bit_index] == 0
        mask1 = ~mask0
        d0 = np.min(distances[:, mask0], axis=1)
        d1 = np.min(distances[:, mask1], axis=1)
        llrs.append((d1 - d0) / variance)
    return np.stack(llrs, axis=-1).reshape(-1).astype(np.float32, copy=False)


def _constellation_numpy(modulation: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
    value = str(modulation or "qpsk").lower().replace("-", "")
    if value in {"pam2", "bpsk"}:
        return (
            np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64),
            np.array([[0], [1]], dtype=np.uint8),
        )
    if value in {"qpsk", "qam4"}:
        scale = 1.0 / np.sqrt(2.0)
        return (
            np.array([1.0 + 1j, 1.0 - 1j, -1.0 + 1j, -1.0 - 1j], dtype=np.complex64) * scale,
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8),
        )
    if value in {"qam16", "16qam"}:
        levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
        bit_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        points = []
        labels = []
        for i_index in range(4):
            for q_index in range(4):
                points.append(levels[i_index] + 1j * levels[q_index])
                labels.append(np.concatenate([bit_pairs[i_index], bit_pairs[q_index]]))
        return np.asarray(points, dtype=np.complex64) / np.sqrt(10.0), np.asarray(labels, dtype=np.uint8)
    return None, None


def _nearest_qam16_values(values: np.ndarray) -> np.ndarray:
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    distances = np.abs(values[:, None] - levels[None, :])
    return np.argmin(distances, axis=1).astype(np.int64)


def _pad_bits(bits: np.ndarray, width: int) -> np.ndarray:
    bits = bits.astype(np.uint8)
    remainder = bits.size % width
    if remainder == 0:
        return bits
    return np.pad(bits, (0, width - remainder), mode="constant")
