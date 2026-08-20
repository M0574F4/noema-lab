from __future__ import annotations

"""Pilot-aided QPSK framing, carrier impairment, and phase-tracking receivers.

The learned receiver deliberately receives only observations available to a real
receiver: the impaired packet and its public pilot context.  Simulator phase
truth is a separate optional artifact used exclusively by the oracle mode.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from noema_lab.core import dataplane
from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.boundaries import validate_channel_bits
from noema_lab.core.capture_layout import (
    CaptureRecordLayout,
    CaptureRecordLayoutError,
    explicit_capture_record_layout,
    remove_capture_record_metadata,
    set_uniform_capture_record_layout,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationResult,
    object_schema,
)


JsonDict = Dict[str, Any]

PILOT_CONTEXT_KIND = "channel.qpsk_pilot_context.numpy"
PHASE_TRUTH_KIND = "channel.carrier_phase_truth.numpy"
PHASE_DIAGNOSTICS_KIND = "receiver.phase_tracking_diagnostics.numpy"
RECEIVER_MODES = (
    "uncompensated",
    "pilot_interpolation",
    "pilot_smoothing",
    "decision_directed_pll",
    "oracle",
    "learned_artifact",
)


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
    metadata: JsonDict,
    layout: CaptureRecordLayout | None,
    element_count: int,
    label: str,
) -> None:
    try:
        set_uniform_capture_record_layout(
            metadata,
            layout,
            element_count,
            label=label,
        )
    except CaptureRecordLayoutError as exc:
        raise OperationError(str(exc)) from exc


def _load_npz(path: Path, fallback_metadata: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], JsonDict]:
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name])
                for name in payload.files
                if name != "metadata_json"
            }
            metadata = dict(fallback_metadata or {})
            if "metadata_json" in payload.files:
                embedded = decode_strict_json_object(
                    str(payload["metadata_json"].item()),
                    label="Phase-tracking metadata_json",
                )
                metadata.update(dict(embedded))
    except Exception as exc:
        raise OperationError("Could not read phase-tracking NPZ artifact %s: %s" % (path, exc)) from exc
    return arrays, metadata


def _save_npz(path: Path, metadata: Mapping[str, Any], **arrays: np.ndarray) -> None:
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=json.dumps(dict(metadata), sort_keys=True),
    )


def _required_array(arrays: Mapping[str, np.ndarray], name: str, label: str) -> np.ndarray:
    if name not in arrays:
        raise OperationError("%s artifact is missing `%s`" % (label, name))
    return np.asarray(arrays[name])


def _packet_layout(metadata: Mapping[str, Any], element_count: int, *, label: str) -> Tuple[int, int]:
    packet_count = int(
        metadata.get("packet_count")
        or metadata.get("example_count")
        or metadata.get("batch_size")
        or 1
    )
    elements_per_packet = int(
        metadata.get("frame_symbols_per_packet")
        or metadata.get("symbols_per_packet")
        or 0
    )
    if packet_count < 1:
        raise OperationError("%s packet_count must be positive" % label)
    if elements_per_packet <= 0:
        if int(element_count) % packet_count:
            raise OperationError(
                "%s element count %d is not divisible by packet_count %d"
                % (label, int(element_count), packet_count)
            )
        elements_per_packet = int(element_count) // packet_count
    if packet_count * elements_per_packet != int(element_count):
        raise OperationError(
            "%s declares %d packets x %d elements but stores %d"
            % (label, packet_count, elements_per_packet, int(element_count))
        )
    return packet_count, elements_per_packet


def _qpsk_slice(symbols: np.ndarray) -> np.ndarray:
    real = np.where(np.real(symbols) < 0.0, -1.0, 1.0)
    imag = np.where(np.imag(symbols) < 0.0, -1.0, 1.0)
    return ((real + 1j * imag) / math.sqrt(2.0)).astype(np.complex64)


class QpskPilotModulateOperation(Operation):
    id = "modulation.qpsk_pilot_modulate"
    name = "QPSK pilot-frame modulator"
    input_kinds = {
        "bits": [
            "channel.payload_bits.numpy",
            "channel.coded_bits.numpy",
            "channel.bits.numpy",
        ]
    }
    output_kinds = {
        "symbols": "channel.symbols.complex_numpy",
        "pilot_context": PILOT_CONTEXT_KIND,
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "stop",
        "trainable_params": False,
        "exportable": False,
        "reason": "Hard QPSK mapping and discrete pilot insertion are benchmark framing operations.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {
        "type": "exact",
        "reason": "The same bits, packet layout, and pilot seed define an exact framed symbol stream.",
    }
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "preamble_symbols": {
                "type": "integer",
                "default": 16,
                "minimum": 2,
            },
            "pilot_spacing_data_symbols": {
                "type": "integer",
                "default": 16,
                "minimum": 1,
            },
            "pilot_seed": {
                "type": "integer",
                "default": 1701,
                "minimum": 0,
            },
            "data_plane_backend": dataplane.backend_schema("auto"),
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("bits")
        arrays, metadata = _load_npz(input_artifact.path, input_artifact.metadata)
        raw_bits = _required_array(arrays, "bits", "QPSK modulator")
        bits, selected_backend = dataplane.require_canonical_bits(
            raw_bits,
            ctx.step_id,
            ctx.params.get("data_plane_backend", "auto"),
        )
        capture_layout = _validated_capture_layout(
            metadata,
            int(bits.size),
            "QPSK pilot modulator input at %s" % ctx.step_id,
        )
        packet_count = int(
            metadata.get("example_count")
            or metadata.get("batch_size")
            or metadata.get("packet_count")
            or 1
        )
        bit_count_per_packet = int(
            metadata.get("bit_count_per_example")
            or metadata.get("bits_per_packet")
            or 0
        )
        if bit_count_per_packet <= 0:
            if bits.size % max(packet_count, 1):
                raise OperationError("QPSK pilot framing cannot infer equal packet lengths")
            bit_count_per_packet = int(bits.size) // max(packet_count, 1)
        if packet_count < 1 or packet_count * bit_count_per_packet != int(bits.size):
            raise OperationError("QPSK pilot framing requires equal, complete bit packets")
        if capture_layout is not None and (
            capture_layout.count != packet_count
            or capture_layout.elements_per_record != bit_count_per_packet
        ):
            raise OperationError(
                "QPSK pilot packet metadata conflicts with the explicit "
                "capture-record layout"
            )
        if bit_count_per_packet % 2:
            raise OperationError(
                "QPSK pilot framing requires an even bit_count_per_example; got %d"
                % bit_count_per_packet
            )

        preamble_symbols = int(ctx.params.get("preamble_symbols", 16))
        spacing = int(ctx.params.get("pilot_spacing_data_symbols", 16))
        pilot_seed = int(ctx.params.get("pilot_seed", 1701))
        if preamble_symbols < 2 or spacing < 1:
            raise OperationError("QPSK pilot framing requires at least two preamble pilots and positive spacing")

        data_symbols_per_packet = bit_count_per_packet // 2
        trailing_pilots = int(math.ceil(data_symbols_per_packet / float(spacing)))
        pilot_symbols_per_packet = preamble_symbols + trailing_pilots
        frame_symbols_per_packet = data_symbols_per_packet + pilot_symbols_per_packet
        packet_bits = bits.reshape(packet_count, bit_count_per_packet)
        data_symbols = np.empty(
            (packet_count, data_symbols_per_packet), dtype=np.complex64
        )
        for packet_index in range(packet_count):
            ctx.raise_if_cancelled()
            mapped, padded, _backend = dataplane.qpsk_modulate(
                packet_bits[packet_index], selected_backend
            )
            if int(padded.size) != bit_count_per_packet:
                raise OperationError("QPSK pilot framing unexpectedly padded an even packet")
            data_symbols[packet_index] = mapped

        frames = np.empty(
            (packet_count, frame_symbols_per_packet), dtype=np.complex64
        )
        context = np.zeros(
            (packet_count, frame_symbols_per_packet, 3), dtype=np.float32
        )
        pilot_rng = np.random.RandomState(pilot_seed)
        for packet_index in range(packet_count):
            ctx.raise_if_cancelled()
            pilot_bits = pilot_rng.randint(
                0, 2, size=2 * pilot_symbols_per_packet
            ).astype(np.uint8)
            pilot_symbols, _padded, _backend = dataplane.qpsk_modulate(
                pilot_bits, "python_numpy"
            )
            cursor = 0
            pilot_cursor = 0
            frames[packet_index, :preamble_symbols] = pilot_symbols[:preamble_symbols]
            context[packet_index, :preamble_symbols, 0] = 1.0
            context[packet_index, :preamble_symbols, 1] = np.real(
                pilot_symbols[:preamble_symbols]
            )
            context[packet_index, :preamble_symbols, 2] = np.imag(
                pilot_symbols[:preamble_symbols]
            )
            cursor += preamble_symbols
            pilot_cursor += preamble_symbols
            for data_start in range(0, data_symbols_per_packet, spacing):
                data_stop = min(data_symbols_per_packet, data_start + spacing)
                count = data_stop - data_start
                frames[packet_index, cursor : cursor + count] = data_symbols[
                    packet_index, data_start:data_stop
                ]
                cursor += count
                pilot_symbol = pilot_symbols[pilot_cursor]
                frames[packet_index, cursor] = pilot_symbol
                context[packet_index, cursor, 0] = 1.0
                context[packet_index, cursor, 1] = float(np.real(pilot_symbol))
                context[packet_index, cursor, 2] = float(np.imag(pilot_symbol))
                cursor += 1
                pilot_cursor += 1
            if cursor != frame_symbols_per_packet or pilot_cursor != pilot_symbols_per_packet:
                raise OperationError("Internal QPSK pilot-frame accounting mismatch")

        common_metadata = dict(metadata)
        common_metadata.update(
            {
                "modulation": "qpsk",
                "bits_per_symbol": 2,
                "bit_count": int(bits.size),
                "packet_count": packet_count,
                "bits_per_packet": bit_count_per_packet,
                "bit_count_per_example": bit_count_per_packet,
                "data_symbols_per_packet": data_symbols_per_packet,
                "pilot_symbols_per_packet": pilot_symbols_per_packet,
                "frame_symbols_per_packet": frame_symbols_per_packet,
                "data_symbol_count": int(packet_count * data_symbols_per_packet),
                "pilot_symbol_count": int(packet_count * pilot_symbols_per_packet),
                "symbol_count": int(frames.size),
                "channel_use_count": int(frames.size),
                "preamble_symbols": preamble_symbols,
                "pilot_spacing_data_symbols": spacing,
                "pilot_seed": pilot_seed,
                "pilot_overhead_fraction": float(pilot_symbols_per_packet)
                / float(frame_symbols_per_packet),
                "effective_payload_bits_per_channel_use": float(bit_count_per_packet)
                / float(frame_symbols_per_packet),
                "data_plane_backend": selected_backend,
            }
        )
        symbol_metadata = dict(common_metadata)
        symbol_metadata.update(
            {
                "array": "symbols",
                "axes": ["flattened_packet_frame_symbol"],
                "capture_record_count": packet_count,
                "capture_record_shape": [frame_symbols_per_packet],
            }
        )
        context_metadata = dict(common_metadata)
        remove_capture_record_metadata(context_metadata)
        context_metadata.update(
            {
                "array": "pilot_context",
                "axes": ["packet", "frame_symbol", "pilot_feature"],
                "pilot_features": [
                    "pilot_mask",
                    "known_pilot_real",
                    "known_pilot_imag",
                ],
                "capture_record_axis": 0,
            }
        )
        symbols_path = ctx.output_path("symbols", ".npz")
        context_path = ctx.output_path("pilot_context", ".npz")
        _save_npz(
            symbols_path,
            symbol_metadata,
            symbols=frames.reshape(-1).astype(np.complex64, copy=False),
        )
        _save_npz(context_path, context_metadata, pilot_context=context)
        return OperationResult(
            outputs={
                "symbols": artifact(
                    "channel.symbols.complex_numpy",
                    symbols_path,
                    symbol_metadata,
                ),
                "pilot_context": artifact(
                    PILOT_CONTEXT_KIND,
                    context_path,
                    context_metadata,
                ),
            },
            metrics={
                "channel.transmitted_bit_count": int(bits.size),
                "channel.data_symbol_count": int(packet_count * data_symbols_per_packet),
                "channel.pilot_symbol_count": int(packet_count * pilot_symbols_per_packet),
                "channel.symbol_count": int(frames.size),
                "channel.channel_use_count": int(frames.size),
                "channel.bits_per_symbol": 2,
                "channel.pilot_overhead_fraction": float(pilot_symbols_per_packet)
                / float(frame_symbols_per_packet),
                "channel.effective_payload_bits_per_channel_use": float(bit_count_per_packet)
                / float(frame_symbols_per_packet),
            },
            metadata={
                "modulation": "qpsk",
                "packet_count": packet_count,
                "frame_symbols_per_packet": frame_symbols_per_packet,
                "pilot_symbols_per_packet": pilot_symbols_per_packet,
                "data_plane_backend": selected_backend,
            },
        )


class CarrierPhaseImpairmentOperation(Operation):
    id = "wireless.carrier_phase_impairment"
    name = "Residual carrier phase impairment"
    input_kinds = {"rx_symbols": ["channel.rx_symbols.complex_numpy"]}
    output_kinds = {
        "rx_symbols": "channel.rx_symbols.complex_numpy",
        "phase_truth": PHASE_TRUTH_KIND,
    }
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "The seeded carrier process is a benchmark impairment with explicit simulation truth.",
    }
    backends = {
        "benchmark_run": ["numpy"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    equivalence = {
        "type": "exact",
        "reason": "The same packet layout, parameters, and seed define the exact carrier trajectory.",
    }
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "initial_phase_min_rad": {
                "type": "number",
                "default": -math.pi,
            },
            "initial_phase_max_rad": {
                "type": "number",
                "default": math.pi,
            },
            "cfo_min_cycles_per_symbol": {
                "type": "number",
                "default": -0.01,
            },
            "cfo_max_cycles_per_symbol": {
                "type": "number",
                "default": 0.01,
            },
            "phase_noise_increment_std_rad": {
                "type": "number",
                "default": 0.04,
                "minimum": 0.0,
            },
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        input_artifact = ctx.require_input("rx_symbols")
        arrays, metadata = _load_npz(input_artifact.path, input_artifact.metadata)
        symbols = _required_array(arrays, "symbols", "carrier input").astype(
            np.complex64, copy=False
        ).reshape(-1)
        capture_layout = _validated_capture_layout(
            metadata,
            int(symbols.size),
            "Carrier-impairment input at %s" % ctx.step_id,
        )
        packet_count, frame_symbols = _packet_layout(
            metadata, symbols.size, label="carrier input"
        )
        if capture_layout is not None and (
            capture_layout.count != packet_count
            or capture_layout.elements_per_record != frame_symbols
        ):
            raise OperationError(
                "Carrier-impairment packet metadata conflicts with the explicit "
                "capture-record layout"
            )
        phase_min = float(ctx.params.get("initial_phase_min_rad", -math.pi))
        phase_max = float(ctx.params.get("initial_phase_max_rad", math.pi))
        cfo_min = float(ctx.params.get("cfo_min_cycles_per_symbol", -0.01))
        cfo_max = float(ctx.params.get("cfo_max_cycles_per_symbol", 0.01))
        phase_noise_std = float(
            ctx.params.get("phase_noise_increment_std_rad", 0.04)
        )
        values = (phase_min, phase_max, cfo_min, cfo_max, phase_noise_std)
        if any(not math.isfinite(value) for value in values):
            raise OperationError("Carrier impairment parameters must be finite")
        if phase_max < phase_min:
            raise OperationError("initial_phase_max_rad must be >= initial_phase_min_rad")
        if cfo_max < cfo_min:
            raise OperationError("cfo_max_cycles_per_symbol must be >= cfo_min_cycles_per_symbol")
        if phase_noise_std < 0.0:
            raise OperationError("phase_noise_increment_std_rad must be nonnegative")

        seed = ctx.seed("carrier_phase_impairment")
        rng = np.random.RandomState(seed)
        initial_phase = rng.uniform(phase_min, phase_max, size=packet_count).astype(
            np.float64
        )
        cfo = rng.uniform(cfo_min, cfo_max, size=packet_count).astype(np.float64)
        increments = rng.normal(
            0.0,
            phase_noise_std,
            size=(packet_count, max(frame_symbols - 1, 0)),
        ).astype(np.float64)
        random_walk = np.zeros((packet_count, frame_symbols), dtype=np.float64)
        if frame_symbols > 1:
            random_walk[:, 1:] = np.cumsum(increments, axis=1)
        time_index = np.arange(frame_symbols, dtype=np.float64).reshape(1, -1)
        phase = (
            initial_phase.reshape(-1, 1)
            + (2.0 * math.pi) * cfo.reshape(-1, 1) * time_index
            + random_walk
        )
        framed = symbols.reshape(packet_count, frame_symbols)
        impaired = (
            framed * np.exp(1j * phase).astype(np.complex64)
        ).astype(np.complex64, copy=False)

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "array": "symbols",
                "carrier_impairment": "random_phase_cfo_wiener_phase_noise",
                "carrier_phase_seed": int(seed),
                "initial_phase_min_rad": phase_min,
                "initial_phase_max_rad": phase_max,
                "cfo_min_cycles_per_symbol": cfo_min,
                "cfo_max_cycles_per_symbol": cfo_max,
                "phase_noise_increment_std_rad": phase_noise_std,
                "initial_phase_rad_preview": [
                    float(value) for value in initial_phase[:8]
                ],
                "cfo_cycles_per_symbol_preview": [
                    float(value) for value in cfo[:8]
                ],
                "phase_truth_available": True,
            }
        )
        truth_metadata = dict(output_metadata)
        remove_capture_record_metadata(truth_metadata)
        preview_count = min(frame_symbols, 64)
        preview_indices = np.unique(
            np.linspace(0, frame_symbols - 1, num=preview_count, dtype=np.int64)
        )
        truth_metadata.update(
            {
                "array": "phase_rad",
                "axes": ["packet", "frame_symbol"],
                "capture_record_axis": 0,
                "simulation_truth": True,
                "runtime_receiver_input": False,
                # A bounded first-packet trace lets the Results surface compare
                # estimators without loading a potentially large truth tensor.
                # It remains metadata of the separate truth artifact and is never
                # inspected by the learned receiver path.
                "phase_truth_preview": {
                    "packet_index": 0,
                    "frame_symbol_index": [
                        int(index) for index in preview_indices.tolist()
                    ],
                    "true_phase_rad": [
                        float(phase[0, index]) for index in preview_indices
                    ],
                },
            }
        )
        symbols_path = ctx.output_path("rx_symbols", ".npz")
        truth_path = ctx.output_path("phase_truth", ".npz")
        _save_npz(
            symbols_path,
            output_metadata,
            symbols=impaired.reshape(-1),
        )
        _save_npz(
            truth_path,
            truth_metadata,
            phase_rad=phase.astype(np.float32),
        )
        return OperationResult(
            outputs={
                "rx_symbols": artifact(
                    "channel.rx_symbols.complex_numpy",
                    symbols_path,
                    output_metadata,
                ),
                "phase_truth": artifact(
                    PHASE_TRUTH_KIND,
                    truth_path,
                    truth_metadata,
                ),
            },
            metrics={
                "channel.carrier.initial_phase_abs_mean_rad": float(
                    np.mean(np.abs(initial_phase))
                ),
                "channel.carrier.cfo_abs_mean_cycles_per_symbol": float(
                    np.mean(np.abs(cfo))
                ),
                "channel.carrier.phase_noise_increment_std_rad": phase_noise_std,
                "channel.carrier.phase_drift_rms_rad": float(
                    np.sqrt(np.mean((phase - phase[:, :1]) ** 2))
                ),
            },
            metadata={
                "carrier_impairment": output_metadata["carrier_impairment"],
                "seed": int(seed),
                "packet_count": packet_count,
                "frame_symbols_per_packet": frame_symbols,
            },
        )


class PhaseTrackingReceiverAdapterOperation(Operation):
    id = "demodulation.phase_tracking_receiver_adapter"
    name = "Pilot-aided QPSK phase-tracking receiver"
    input_kinds = {
        "rx_symbols": ["channel.rx_symbols.complex_numpy"],
        "pilot_context": [PILOT_CONTEXT_KIND],
    }
    optional_input_kinds = {"phase_truth": [PHASE_TRUTH_KIND]}
    output_kinds = {
        "bits": "channel.demod_bits.numpy",
        "llr": "channel.llr.numpy",
        "diagnostics": PHASE_DIAGNOSTICS_KIND,
    }
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": True,
        "exportable": True,
        "reason": "The adapter exposes a packet-context portable receiver slot while classical modes remain reproducible baselines.",
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch"],
    }
    materializations = [
        *[
            {
                "runner": runner,
                "backend": "numpy",
                "implementation": "%s_phase_tracking_receiver" % mode,
                "status": "implemented",
                "parameter_bindings": {"mode": mode},
            }
            for runner in ("benchmark_run", "dataset_capture")
            for mode in RECEIVER_MODES
            if mode != "learned_artifact"
        ],
        *[
            {
                "runner": runner,
                "backend": "onnxruntime",
                "implementation": "portable_packet_context_receiver",
                "status": "implemented",
                "parameter_bindings": {"mode": "learned_artifact"},
                "notes": "Simulator phase truth is not passed to the learned entrypoint.",
            }
            for runner in ("benchmark_run", "dataset_capture")
        ],
        {
            "runner": "differentiable_export",
            "backend": "torch",
            "implementation": "phase_tracking_receiver_training_endpoint",
            "status": "implemented",
        },
    ]
    equivalence = {
        "type": "behavioral",
        "reason": "Phase-tracking receivers are compared by paired BER/BLER behavior, not sample-identical internal estimates.",
    }
    formats = {
        "artifact": "npz",
        "tensor": "torch.Tensor",
        "checkpoint": "onnx",
    }
    trained_artifact_abi = {
        "component_id": "receiver",
        "component_role": "phase_tracking_receiver",
        "entrypoint_id": "phase_tracking_receiver",
        "required_operation_inputs": ["rx_symbols", "pilot_context"],
        "inputs": {
            "receiver_features_v3": {
                "dtype": "float32",
                "shape": ["packet", "frame_symbol", 11],
            }
        },
        "outputs": {
            "residual_phase_rad": {
                "dtype": "float32",
                "shape": ["packet", "frame_symbol"],
            }
        },
        "binding_params": {
            "mode": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "phase_tracking_receiver",
        },
    }
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "pilot_smoothing",
                "enum": list(RECEIVER_MODES),
            },
            "pilot_smoothing_nearest_pilots": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": (
                    "Number of nearest public pilots used by the noncausal "
                    "local-linear packet smoother."
                ),
            },
            "pll_alpha": {
                "type": "number",
                "default": 0.12,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "pll_beta": {
                "type": "number",
                "default": 0.005,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "description": "Registered schema-v2 artifact implementing the packet-context phase-tracking receiver ABI.",
                "x-noema-ui": {
                    "control": "trained_artifact",
                    "label": "Trained artifact",
                    "accept": ".zip,.noema-artifact,.yaml,.yml,.json,application/octet-stream",
                    "visible_when": {"mode": "learned_artifact"},
                    "derived_params": [
                        "mode",
                        "artifact_manifest_path",
                        "artifact_entrypoint",
                        "artifact_package_sha256",
                    ],
                },
            },
            "artifact_entrypoint": {
                "type": "string",
                "default": "phase_tracking_receiver",
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "x-noema-ui": {"hidden": True},
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        rx_artifact = ctx.require_input("rx_symbols")
        rx_arrays, metadata = _load_npz(rx_artifact.path, rx_artifact.metadata)
        flat_rx = _required_array(rx_arrays, "symbols", "phase-tracking receiver").astype(
            np.complex64, copy=False
        ).reshape(-1)
        capture_layout = _validated_capture_layout(
            metadata,
            int(flat_rx.size),
            "Phase-tracking receiver input at %s" % ctx.step_id,
        )
        context_artifact = ctx.require_input("pilot_context")
        context_arrays, context_metadata = _load_npz(
            context_artifact.path, context_artifact.metadata
        )
        pilot_context = _required_array(
            context_arrays, "pilot_context", "pilot context"
        ).astype(np.float32, copy=False)
        if pilot_context.ndim != 3 or pilot_context.shape[2] != 3:
            raise OperationError(
                "pilot_context must have shape [packet, frame_symbol, 3], got %s"
                % (tuple(pilot_context.shape),)
            )
        packet_count, frame_symbols = int(pilot_context.shape[0]), int(
            pilot_context.shape[1]
        )
        if int(flat_rx.size) != packet_count * frame_symbols:
            raise OperationError(
                "Received symbols do not match pilot-context packet layout: %d vs %s"
                % (int(flat_rx.size), tuple(pilot_context.shape[:2]))
            )
        if capture_layout is not None and (
            capture_layout.count != packet_count
            or capture_layout.elements_per_record != frame_symbols
        ):
            raise OperationError(
                "Phase-tracking receiver packet context conflicts with the "
                "explicit capture-record layout"
            )
        rx = flat_rx.reshape(packet_count, frame_symbols)
        pilot_mask = pilot_context[:, :, 0] > 0.5
        known_pilots = (
            pilot_context[:, :, 1].astype(np.float32)
            + 1j * pilot_context[:, :, 2].astype(np.float32)
        ).astype(np.complex64)
        if np.any(np.sum(pilot_mask, axis=1) < 2):
            raise OperationError("Every packet needs at least two known pilots")
        if np.any(np.abs(known_pilots[pilot_mask]) < 1e-6):
            raise OperationError("Active pilot-context symbols must be non-zero")
        data_mask = ~pilot_mask
        data_counts = np.sum(data_mask, axis=1)
        if not np.all(data_counts == data_counts[0]):
            raise OperationError("Every phase-tracking packet must use the same pilot layout")
        data_symbols_per_packet = int(data_counts[0])
        mode = str(ctx.params.get("mode") or "pilot_smoothing")
        if mode not in RECEIVER_MODES:
            raise OperationError("Unknown phase-tracking receiver mode: %s" % mode)
        smoothing_nearest_pilots = int(
            ctx.params.get("pilot_smoothing_nearest_pilots", 5)
        )
        if smoothing_nearest_pilots < 2:
            raise OperationError(
                "pilot_smoothing_nearest_pilots must be at least 2"
            )

        phase_estimate: np.ndarray | None = None
        selected_backend = "python_numpy"
        manifest_sha = None
        if mode == "learned_artifact":
            # Deliberately construct the learned ABI from observations and public
            # pilot context only.  Do not load, inspect, or forward phase_truth.
            from noema_lab.core.trained_artifact_runtime import (
                run_trained_artifact_entrypoint,
            )

            manifest_value = str(
                ctx.params.get("artifact_manifest_path") or ""
            ).strip()
            entrypoint = str(
                ctx.params.get("artifact_entrypoint")
                or "phase_tracking_receiver"
            ).strip()
            if not manifest_value:
                raise OperationError(
                    "learned_artifact phase-tracking receiver requires artifact_manifest_path"
                )
            manifest_path = Path(manifest_value).expanduser()
            if not manifest_path.is_file():
                raise OperationError(
                    "Phase-tracking trained-artifact manifest does not exist: %s"
                    % manifest_path
                )
            smoothing_phase = _pilot_smoothed_phase(
                rx,
                pilot_mask,
                known_pilots,
                nearest_pilots=smoothing_nearest_pilots,
            )
            features = _receiver_features_v3(
                rx,
                pilot_mask,
                known_pilots,
                nearest_pilots=smoothing_nearest_pilots,
                smoothing_phase=smoothing_phase,
            )
            try:
                artifact_outputs = run_trained_artifact_entrypoint(
                    manifest_path,
                    entrypoint,
                    {"receiver_features_v3": features},
                    expected_package_sha256=str(
                        ctx.params.get("artifact_package_sha256") or ""
                    ),
                )
            except Exception as exc:
                raise OperationError(
                    "Phase-tracking trained-artifact inference failed: %s" % exc
                ) from exc
            if "residual_phase_rad" not in artifact_outputs:
                raise OperationError(
                    "Phase-tracking trained artifact did not return `residual_phase_rad`"
                )
            residual_phase = np.asarray(
                artifact_outputs["residual_phase_rad"],
                dtype=np.float32,
            )
            if tuple(residual_phase.shape) != (packet_count, frame_symbols):
                raise OperationError(
                    "residual_phase_rad must have shape %s, got %s"
                    % (
                        (packet_count, frame_symbols),
                        tuple(residual_phase.shape),
                    )
                )
            if not np.all(np.isfinite(residual_phase)):
                raise OperationError("residual_phase_rad contains non-finite values")
            phase_estimate = np.unwrap(
                smoothing_phase
                + residual_phase.astype(np.float64, copy=False),
                axis=1,
            )
            selected_backend = "onnxruntime"
            manifest_sha = file_sha256(manifest_path)
        else:
            if mode == "uncompensated":
                phase_estimate = np.zeros(rx.shape, dtype=np.float64)
            elif mode == "pilot_interpolation":
                phase_estimate = _pilot_interpolated_phase(
                    rx, pilot_mask, known_pilots
                )
            elif mode == "pilot_smoothing":
                phase_estimate = _pilot_smoothed_phase(
                    rx,
                    pilot_mask,
                    known_pilots,
                    nearest_pilots=smoothing_nearest_pilots,
                )
            elif mode == "decision_directed_pll":
                phase_estimate = _decision_directed_pll_phase(
                    rx,
                    pilot_mask,
                    known_pilots,
                    alpha=float(ctx.params.get("pll_alpha", 0.12)),
                    beta=float(ctx.params.get("pll_beta", 0.005)),
                )
            elif mode == "oracle":
                if "phase_truth" not in ctx.inputs:
                    raise OperationError(
                        "oracle phase-tracking receiver requires phase_truth"
                    )
                phase_estimate = _load_phase_truth(
                    ctx.inputs["phase_truth"], packet_count, frame_symbols
                )
            else:  # pragma: no cover - protected by RECEIVER_MODES
                raise OperationError("Unsupported receiver mode %s" % mode)
        if phase_estimate is None:  # pragma: no cover - guarded by mode handling
            raise OperationError("Phase-tracking receiver did not produce a phase estimate")
        corrected = rx * np.exp(-1j * phase_estimate).astype(np.complex64)
        data_symbols = corrected[data_mask].reshape(
            packet_count, data_symbols_per_packet
        )
        noise_variance = max(
            float(metadata.get("noise_variance") or 1.0), 1e-12
        )
        scale = float(2.0 * math.sqrt(2.0) / noise_variance)
        data_llr = np.stack(
            [
                np.real(data_symbols) * scale,
                np.imag(data_symbols) * scale,
            ],
            axis=-1,
        ).astype(np.float32)

        bit_count = int(
            metadata.get("bit_count")
            or context_metadata.get("bit_count")
            or packet_count * data_symbols_per_packet * 2
        )
        if bit_count > int(data_llr.size):
            raise OperationError(
                "Receiver produced %d bit logits for declared bit_count %d"
                % (int(data_llr.size), bit_count)
            )
        bit_count_per_packet = int(
            metadata.get("bits_per_packet")
            or metadata.get("bit_count_per_example")
            or 0
        )
        if bit_count_per_packet <= 0:
            if bit_count % packet_count:
                raise OperationError(
                    "Phase-tracking bit_count cannot be divided into equal packets"
                )
            bit_count_per_packet = bit_count // packet_count
        if (
            packet_count * bit_count_per_packet != bit_count
            or bit_count_per_packet > data_symbols_per_packet * 2
        ):
            raise OperationError(
                "Phase-tracking per-packet bit accounting is inconsistent"
            )
        llr = np.concatenate(
            [
                data_llr[packet_index].reshape(-1)[:bit_count_per_packet]
                for packet_index in range(packet_count)
            ]
        ).astype(np.float32, copy=False)
        raw_bits = (llr < 0.0).astype(np.uint8, copy=False)
        bits, validation_metadata = validate_channel_bits(
            raw_bits,
            label=ctx.step_id,
            backend="auto",
        )
        if int(bits.size) % packet_count:
            raise OperationError(
                "Receiver produced %d bits that cannot be divided across %d packets"
                % (int(bits.size), packet_count)
            )
        bit_count_per_packet = int(bits.size) // packet_count
        diagnostics, diagnostics_metadata = _receiver_diagnostics(
            mode=mode,
            rx=rx,
            data_mask=data_mask,
            phase_estimate=phase_estimate,
            manifest_sha=manifest_sha,
            smoothing_nearest_pilots=smoothing_nearest_pilots,
        )
        bit_metadata = dict(metadata)
        bit_metadata.update(
            {
                "array": "bits",
                "axes": ["flattened_packet_data_bit"],
                "boundary_contract": "channel.bits",
                "bit_count": int(bits.size),
                "bit_count_per_example": bit_count_per_packet,
                "bits_per_packet": bit_count_per_packet,
                "byte_count": int(validation_metadata["byte_count"]),
                "capture_record_count": packet_count,
                "capture_record_shape": [bit_count_per_packet],
                "dtype": "uint8",
                "shape": [int(bits.size)],
                "bit_role": "demodulated",
                "demodulation": "phase_tracking_receiver_adapter",
                "receiver_family": "pilot_aided_qpsk_phase_tracking",
                "receiver_mode": mode,
                "llr_kind": "analytical_qpsk_maxlog_after_phase_tracking",
                "bit_storage": "unpacked_uint8",
                "channel_bit_array": "unpacked_uint8",
                "bits_per_element": 1,
                "storage": "unpacked_uint8",
                "data_plane_backend": selected_backend,
                "bit_validation_backend": validation_metadata.get(
                    "data_plane_backend"
                ),
            }
        )
        if mode in {"pilot_smoothing", "learned_artifact"}:
            bit_metadata["pilot_smoothing_nearest_pilots"] = (
                smoothing_nearest_pilots
            )
        if mode == "learned_artifact":
            bit_metadata["learned_receiver_feature_contract"] = (
                "receiver_features_v3"
            )
            bit_metadata["learned_receiver_output"] = "residual_phase_rad"
        if manifest_sha:
            bit_metadata["trained_artifact_manifest_sha256"] = manifest_sha
        output_capture_layout = capture_layout or CaptureRecordLayout(
            count=packet_count,
            shape=(frame_symbols,),
        )
        _rewrite_uniform_capture_layout(
            bit_metadata,
            output_capture_layout,
            int(bits.size),
            "Phase-tracking receiver output at %s" % ctx.step_id,
        )
        llr_metadata = dict(bit_metadata)
        llr_metadata["array"] = "llr"
        llr_metadata["dtype"] = "float32"
        llr_metadata["shape"] = [int(llr.size)]
        llr_metadata["storage"] = "float32"
        bits_path = ctx.output_path("bits", ".npz")
        llr_path = ctx.output_path("llr", ".npz")
        diagnostics_path = ctx.output_path("diagnostics", ".npz")
        _save_npz(bits_path, bit_metadata, bits=bits)
        _save_npz(llr_path, llr_metadata, llr=llr)
        _save_npz(
            diagnostics_path,
            diagnostics_metadata,
            phase_estimate_rad=diagnostics,
        )
        safe_mode = re.sub(r"[^a-z0-9]+", "_", mode.lower()).strip("_")
        return OperationResult(
            outputs={
                "bits": artifact(
                    "channel.demod_bits.numpy", bits_path, bit_metadata
                ),
                "llr": artifact("channel.llr.numpy", llr_path, llr_metadata),
                "diagnostics": artifact(
                    PHASE_DIAGNOSTICS_KIND,
                    diagnostics_path,
                    diagnostics_metadata,
                ),
            },
            metrics={
                "channel.demod_bit_count": int(bits.size),
                "receiver.phase_tracking.%s" % safe_mode: 1,
                "receiver.phase_tracking.packet_count": packet_count,
                "receiver.phase_tracking.data_symbol_count": int(
                    packet_count * data_symbols_per_packet
                ),
            },
            metadata={
                "receiver_mode": mode,
                "packet_context": True,
                "phase_truth_forwarded_to_learned_runtime": False,
                "data_plane_backend": selected_backend,
                **(
                    {
                        "pilot_smoothing_nearest_pilots": (
                            smoothing_nearest_pilots
                        )
                    }
                    if mode in {"pilot_smoothing", "learned_artifact"}
                    else {}
                ),
                **(
                    {"learned_receiver_feature_contract": "receiver_features_v3"}
                    if mode == "learned_artifact"
                    else {}
                ),
            },
        )


def _receiver_features_v3(
    rx: np.ndarray,
    pilot_mask: np.ndarray,
    known_pilots: np.ndarray,
    *,
    nearest_pilots: int = 5,
    smoothing_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Build the versioned packet-context residual-tracking ABI.

    The first two channels are corrected by the noncausal pilot smoother.  A
    zero-output residual model therefore has exactly the same hard decisions
    as ``pilot_smoothing``.  The remaining observable channels are raw I/Q,
    the pilot mask, unit-phasor pilot innovations relative to the smoother,
    the smoother phasor, and the normalized QPSK fourth-power cue.  Simulator
    phase truth is deliberately absent.
    """

    pilot_observation = np.zeros(rx.shape, dtype=np.complex64)
    pilot_observation[pilot_mask] = (
        rx[pilot_mask] * np.conj(known_pilots[pilot_mask])
    ).astype(np.complex64, copy=False)
    if smoothing_phase is None:
        smoothing_phase = _pilot_smoothed_phase(
            rx,
            pilot_mask,
            known_pilots,
            nearest_pilots=nearest_pilots,
        )
    elif tuple(smoothing_phase.shape) != tuple(rx.shape):
        raise OperationError(
            "Pilot-smoothing phase must match received packet shape %s; got %s"
            % (tuple(rx.shape), tuple(smoothing_phase.shape))
        )
    smoothing_phasor = np.exp(1j * smoothing_phase).astype(
        np.complex64,
        copy=False,
    )
    corrected = (
        rx * np.conj(smoothing_phasor)
    ).astype(np.complex64, copy=False)
    pilot_innovation = np.zeros(rx.shape, dtype=np.complex64)
    pilot_innovation[pilot_mask] = _normalized_complex_phasor(
        pilot_observation[pilot_mask]
        * np.conj(smoothing_phasor[pilot_mask])
    )
    fourth_power_cue = _normalized_complex_phasor(
        -(corrected.astype(np.complex64, copy=False) ** 4)
    )
    return np.stack(
        [
            np.real(corrected),
            np.imag(corrected),
            np.real(rx),
            np.imag(rx),
            pilot_mask.astype(np.float32),
            np.real(pilot_innovation),
            np.imag(pilot_innovation),
            np.real(smoothing_phasor),
            np.imag(smoothing_phasor),
            np.real(fourth_power_cue),
            np.imag(fourth_power_cue),
        ],
        axis=-1,
    ).astype(np.float32)


def _normalized_complex_phasor(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.complex64)
    magnitude = np.abs(source)
    normalized = np.zeros(source.shape, dtype=np.complex64)
    valid = magnitude > 1e-12
    normalized[valid] = source[valid] / magnitude[valid]
    return normalized


def _pilot_interpolated_phase(
    rx: np.ndarray,
    pilot_mask: np.ndarray,
    known_pilots: np.ndarray,
) -> np.ndarray:
    estimate = np.empty(rx.shape, dtype=np.float64)
    full_index = np.arange(rx.shape[1], dtype=np.float64)
    for packet_index in range(rx.shape[0]):
        indices = np.flatnonzero(pilot_mask[packet_index])
        observations = np.unwrap(
            np.angle(
                rx[packet_index, indices]
                * np.conj(known_pilots[packet_index, indices])
            )
        )
        estimate[packet_index] = np.interp(full_index, indices, observations)
    return estimate


def _pilot_smoothed_phase(
    rx: np.ndarray,
    pilot_mask: np.ndarray,
    known_pilots: np.ndarray,
    *,
    nearest_pilots: int = 5,
) -> np.ndarray:
    """Fit a noncausal local line at every pilot, then interpolate the packet.

    Each pilot estimate uses the configured number of pilots nearest in frame
    time, including future observations when available.  The local regression
    suppresses pilot phase noise while retaining slow CFO and phase evolution.
    """

    neighbor_count = int(nearest_pilots)
    if neighbor_count < 2:
        raise OperationError("Pilot smoothing requires at least two nearest pilots")
    estimate = np.empty(rx.shape, dtype=np.float64)
    full_index = np.arange(rx.shape[1], dtype=np.float64)
    for packet_index in range(rx.shape[0]):
        indices = np.flatnonzero(pilot_mask[packet_index])
        if indices.size < 2:
            raise OperationError("Pilot smoothing requires at least two pilots per packet")
        observations = np.unwrap(
            np.angle(
                rx[packet_index, indices]
                * np.conj(known_pilots[packet_index, indices])
            )
        )
        local_count = min(neighbor_count, int(indices.size))
        smoothed_observations = np.empty(observations.shape, dtype=np.float64)
        for pilot_position, center_index in enumerate(indices):
            distances = np.abs(indices - center_index)
            nearest = np.lexsort((indices, distances))[:local_count]
            centered_index = (
                indices[nearest].astype(np.float64) - float(center_index)
            )
            design = np.column_stack(
                (centered_index, np.ones(centered_index.shape, dtype=np.float64))
            )
            _slope, intercept = np.linalg.lstsq(
                design,
                observations[nearest],
                rcond=None,
            )[0]
            smoothed_observations[pilot_position] = float(intercept)
        estimate[packet_index] = np.interp(
            full_index,
            indices,
            smoothed_observations,
        )
    return estimate


def _decision_directed_pll_phase(
    rx: np.ndarray,
    pilot_mask: np.ndarray,
    known_pilots: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise OperationError("PLL alpha and beta must lie in [0, 1]")
    estimate = np.empty(rx.shape, dtype=np.float64)
    for packet_index in range(rx.shape[0]):
        pilot_indices = np.flatnonzero(pilot_mask[packet_index])
        preamble_end = 1
        while (
            preamble_end < pilot_indices.size
            and pilot_indices[preamble_end] == pilot_indices[preamble_end - 1] + 1
        ):
            preamble_end += 1
        init_indices = pilot_indices[:preamble_end]
        init_phase = np.unwrap(
            np.angle(
                rx[packet_index, init_indices]
                * np.conj(known_pilots[packet_index, init_indices])
            )
        )
        if init_indices.size >= 2:
            slope, intercept = np.polyfit(
                init_indices.astype(np.float64), init_phase, 1
            )
        else:  # pragma: no cover - operation requires at least two preamble pilots
            slope, intercept = 0.0, float(init_phase[0])
        phase_state = float(intercept)
        frequency_state = float(slope)
        for symbol_index in range(rx.shape[1]):
            predicted = phase_state if symbol_index == 0 else phase_state + frequency_state
            corrected = rx[packet_index, symbol_index] * np.exp(-1j * predicted)
            reference = (
                known_pilots[packet_index, symbol_index]
                if pilot_mask[packet_index, symbol_index]
                else _qpsk_slice(np.asarray([corrected]))[0]
            )
            phase_error = float(np.angle(corrected * np.conj(reference)))
            phase_state = predicted + alpha * phase_error
            frequency_state = frequency_state + beta * phase_error
            estimate[packet_index, symbol_index] = phase_state
    return estimate


def _load_phase_truth(
    input_artifact,
    packet_count: int,
    frame_symbols: int,
) -> np.ndarray:
    arrays, _metadata = _load_npz(input_artifact.path, input_artifact.metadata)
    phase = _required_array(arrays, "phase_rad", "carrier phase truth").astype(
        np.float64, copy=False
    )
    expected = (packet_count, frame_symbols)
    if tuple(phase.shape) != expected:
        raise OperationError(
            "phase_truth must have shape %s, got %s"
            % (expected, tuple(phase.shape))
        )
    if not np.all(np.isfinite(phase)):
        raise OperationError("phase_truth contains non-finite values")
    return phase


def _receiver_diagnostics(
    *,
    mode: str,
    rx: np.ndarray,
    data_mask: np.ndarray,
    phase_estimate: np.ndarray | None,
    manifest_sha: str | None,
    smoothing_nearest_pilots: int,
) -> Tuple[np.ndarray, JsonDict]:
    packet_index = 0
    data_indices = np.flatnonzero(data_mask[packet_index])
    representative_index = int(data_indices[len(data_indices) // 2])
    received_point = rx[packet_index, representative_index]
    metadata: JsonDict = {
        "array": "phase_estimate_rad",
        "axes": ["packet", "frame_symbol"],
        "receiver_mode": mode,
        "receiver_decision_preview": {
            "schema_version": 1,
            "receiver_mode": mode,
            "context": "one_held_out_packet_position_with_other_packet_samples_fixed",
            "representative_packet_index": packet_index,
            "representative_frame_symbol_index": representative_index,
            "received_point": [
                float(np.real(received_point)),
                float(np.imag(received_point)),
            ],
            "boundary_kind": (
                "context_dependent_learned"
                if mode == "learned_artifact"
                else "phase_rotated_orthogonal_lines"
            ),
        },
        "phase_truth_used_for_decisions": mode == "oracle",
        "phase_truth_forwarded_to_learned_runtime": False,
    }
    if mode in {"pilot_smoothing", "learned_artifact"}:
        metadata["pilot_smoothing_nearest_pilots"] = int(
            smoothing_nearest_pilots
        )
    if mode == "learned_artifact":
        metadata["learned_receiver_feature_contract"] = "receiver_features_v3"
        metadata["learned_receiver_output"] = "residual_phase_rad"
    if manifest_sha:
        metadata["trained_artifact_manifest_sha256"] = manifest_sha
    if phase_estimate is None:
        values = np.empty((rx.shape[0], 0), dtype=np.float32)
        metadata["phase_estimate_available"] = False
    else:
        values = phase_estimate.astype(np.float32, copy=False)
        angle = float(phase_estimate[packet_index, representative_index])
        metadata["phase_estimate_available"] = True
        metadata["receiver_decision_preview"]["boundary_angles_rad"] = [
            float(angle % math.pi),
            float((angle + math.pi / 2.0) % math.pi),
        ]
        sample_indices = np.linspace(
            0,
            phase_estimate.shape[1] - 1,
            min(128, phase_estimate.shape[1]),
            dtype=np.int64,
        )
        metadata["phase_tracking_preview"] = {
            "schema_version": 1,
            "receiver_mode": mode,
            "frame_symbol_index": [int(value) for value in sample_indices],
            "estimated_phase_rad": [
                float(phase_estimate[packet_index, value])
                for value in sample_indices
            ],
        }
    return values, metadata
