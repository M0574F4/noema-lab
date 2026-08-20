from __future__ import annotations

import importlib.util
import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core.artifacts import artifact, file_sha256
from noema_lab.core.structured_input import decode_strict_json_object
from noema_lab.core.operations import Operation, OperationContext, OperationError, OperationResult, object_schema
from noema_lab.core.reproducibility import installed_dependency_version

JsonDict = Dict[str, Any]
RESULT_PREVIEW_LIMIT = 16
_SIONNA_RNG_LOCK = threading.RLock()


def _wireless_backend_schema() -> JsonDict:
    return {
        "type": "string",
        "default": "auto",
        "enum": ["auto", "numpy", "sionna"],
        "description": (
            "Wireless implementation. auto resolves deterministically to "
            "NumPy; select sionna explicitly to require Sionna, with no "
            "runtime fallback."
        ),
    }


def _wireless_runtime_availability(params: Mapping[str, Any]) -> JsonDict:
    requested = str(params.get("wireless_backend") or "auto").strip().lower()
    if requested != "sionna":
        return {
            "available": True,
            "optional": False,
            "backend": "numpy",
            "missing": [],
        }
    available = _sionna_available()
    return {
        "available": available,
        "optional": False,
        "backend": "sionna",
        "missing": [] if available else ["sionna>=2.0.1", "torch>=2.9.1"],
        "reason": (
            ""
            if available
            else (
                'Install with `python -m pip install "noema-lab[wireless]"` in '
                "an installed environment, or `uv sync --extra wireless` in "
                "a source checkout, to use the Sionna wireless backend"
            )
        ),
    }


class AiPhyChannelRealizationSourceOperation(Operation):
    """Materialize channel truth independently from pilots and receiver noise."""

    id = "source.ai_phy_channel_realization"
    name = "AI-PHY channel realization source"
    output_kinds = {"channel": "ai_phy.channel_realization.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Channel truth is a fixed-seed benchmark input, separate from the pilot observation model.",
    }
    backends = {
        "benchmark_run": ["numpy", "sionna"],
        "dataset_capture": ["numpy", "sionna"],
        "differentiable_export": ["torch", "sionna"],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "numpy_channel_realization",
            "status": "implemented",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "numpy_channel_realization",
            "status": "implemented",
        },
        {
            "runner": "benchmark_run",
            "backend": "sionna",
            "implementation": "sionna_3gpp_tdl_ofdm_channel",
            "status": "implemented",
            "parameter_bindings": {"scenario": "mimo_ofdm"},
        },
        {
            "runner": "dataset_capture",
            "backend": "sionna",
            "implementation": "sionna_3gpp_tdl_ofdm_channel",
            "status": "implemented",
            "parameter_bindings": {"scenario": "mimo_ofdm"},
        },
    ]
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "scenario": {
                "type": "string",
                "default": "pilot_awgn",
                "enum": [
                    "pilot_awgn",
                    "flat_siso",
                    "mimo_ofdm_pilot",
                    "mimo_ofdm",
                    "mimo_ofdm_3gpp_tdl",
                ],
            },
            "example_count": {"type": "integer", "default": 32, "minimum": 1},
            "rx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "tx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "subcarriers": {"type": "integer", "default": 1, "minimum": 1},
            "channel_tap_count": {"type": "integer", "default": 4, "minimum": 1},
            "wireless_backend": _wireless_backend_schema(),
            "tdl_model": {
                "type": "string",
                "default": "C",
                "enum": ["A", "B", "C", "D", "E"],
            },
            "subcarrier_spacing_khz": {
                "type": "number",
                "default": 30.0,
                "minimum": 0.1,
            },
            "carrier_frequency_ghz": {
                "type": "number",
                "default": 3.5,
                "minimum": 0.1,
            },
            "delay_spread_ns": {
                "type": "number",
                "default": 300.0,
                "minimum": 0.1,
            },
            "mobility_kmh": {
                "type": "number",
                "default": 30.0,
                "minimum": 0.0,
            },
            "normalize_channel": {"type": "boolean", "default": True},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        scenario = str(params.get("scenario") or "pilot_awgn")
        if not (
            scenario == "mimo_ofdm_3gpp_tdl"
            or (
                scenario == "mimo_ofdm"
                and str(params.get("wireless_backend") or "auto") == "sionna"
            )
        ):
            return {
                "available": True,
                "optional": False,
                "backend": "numpy",
                "missing": [],
            }
        return _wireless_runtime_availability(
            {**dict(params), "wireless_backend": "sionna"}
        )

    def run(self, ctx: OperationContext) -> OperationResult:
        scenario = str(ctx.params.get("scenario") or "pilot_awgn")
        n = int(_param(ctx.params, "example_count", 32))
        rx = int(_param(ctx.params, "rx_antennas", 1))
        tx = int(_param(ctx.params, "tx_antennas", 1))
        subcarriers = int(_param(ctx.params, "subcarriers", 1))
        seed = ctx.seed("ai_phy_channel_realization")
        rng = np.random.RandomState(seed)
        if scenario in {"pilot_awgn", "flat_siso"}:
            # Flat Rayleigh fading.  Retain explicit RX, TX and subcarrier axes,
            # even in the scalar SISO case, so downstream contracts never need
            # shape-dependent special cases.
            flat = _complex_normal(rng, (n, rx, tx, 1))
            h_true = np.repeat(flat, subcarriers, axis=3)
            tap_count = 1
            scenario_kind = "flat_rayleigh_pilot_channel"
            backend = "numpy"
            backend_detail = "numpy.flat_rayleigh"
        elif scenario == "mimo_ofdm_3gpp_tdl" or (
            scenario == "mimo_ofdm"
            and str(ctx.params.get("wireless_backend") or "auto") == "sionna"
        ):
            h_true, backend_detail = _sionna_mimo_ofdm_channel_realization(
                example_count=n,
                rx_antennas=rx,
                tx_antennas=tx,
                subcarriers=subcarriers,
                params=ctx.params,
                seed=seed,
            )
            tap_count = int(_param(ctx.params, "channel_tap_count", 4))
            scenario_kind = "sionna_3gpp_frequency_selective_mimo_ofdm"
            backend = "sionna"
        else:
            # A short exponentially decaying tapped-delay line gives a smooth,
            # frequency-selective MIMO-OFDM response suitable for comb pilots.
            tap_count = min(int(_param(ctx.params, "channel_tap_count", 4)), subcarriers)
            pdp = np.exp(-np.arange(tap_count, dtype=np.float64))
            pdp /= np.sum(pdp)
            taps = _complex_normal(rng, (n, rx, tx, tap_count)) * np.sqrt(pdp)[None, None, None, :]
            h_true = np.fft.fft(taps, n=subcarriers, axis=3).astype(np.complex64)
            scenario_kind = "frequency_selective_mimo_ofdm"
            backend = "numpy"
            backend_detail = "numpy.exponential_tapped_delay"
        metadata = {
            "array": "h_true",
            "capture_record_axis": 0,
            "capture_record_unit": "mimo_ofdm_channel_realization",
            "dataset": "synthetic_ai_phy_channel_realizations",
            "scenario": scenario,
            "scenario_kind": scenario_kind,
            "split": "fixed_seed",
            "example_count": n,
            "rx_antennas": rx,
            "tx_antennas": tx,
            "subcarriers": subcarriers,
            "channel_tap_count": tap_count,
            "channel_shape": list(h_true.shape),
            "channel_axes": ["example", "rx_antenna", "tx_antenna", "subcarrier"],
            "channel_variance": 1.0,
            "measured_channel_variance": float(np.mean(np.abs(h_true) ** 2)),
            "seed": int(seed),
            "wireless_backend": backend,
            "wireless_backend_detail": backend_detail,
            "tdl_model": (
                str(ctx.params.get("tdl_model") or "C")
                if backend == "sionna"
                else None
            ),
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
                float(ctx.params.get("delay_spread_ns") or 300.0)
                if backend == "sionna"
                else None
            ),
            "mobility_kmh": (
                float(ctx.params.get("mobility_kmh") or 0.0)
                if backend == "sionna"
                else None
            ),
        }
        path = ctx.output_path("channel", ".npz")
        np.savez_compressed(path, h_true=h_true.astype(np.complex64), metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(
            outputs={"channel": artifact("ai_phy.channel_realization.numpy", path, metadata)},
            metrics={"ai_phy.example_count": n, "ai_phy.channel_element_count": int(h_true.size)},
            metadata=metadata,
        )


class AiPhyPilotPatternSourceOperation(Operation):
    """Create deterministic orthogonal unit or comb pilot symbols."""

    id = "source.ai_phy_pilot_pattern"
    name = "AI-PHY pilot pattern source"
    output_kinds = {"pilots": "ai_phy.pilot_pattern.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "scenario": {"type": "string", "default": "unit", "enum": ["unit", "comb"]},
            "tx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "subcarriers": {"type": "integer", "default": 1, "minimum": 1},
            "pilot_spacing": {"type": "integer", "default": 4, "minimum": 1},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        scenario = str(ctx.params.get("scenario") or "unit")
        tx = int(_param(ctx.params, "tx_antennas", 1))
        subcarriers = int(_param(ctx.params, "subcarriers", 1))
        spacing = max(1, int(_param(ctx.params, "pilot_spacing", 4)))
        seed = ctx.seed("ai_phy_pilot_pattern")
        rng = np.random.RandomState(seed)
        mask = np.zeros((tx, subcarriers), dtype=bool)
        if scenario == "unit":
            mask[:] = True
            orthogonalization_axis = "implicit_tx_slot"
        else:
            available_offsets = min(spacing, subcarriers)
            if tx > available_offsets:
                raise OperationError(
                    "Orthogonal comb pilots require tx_antennas <= "
                    "min(pilot_spacing, subcarriers); got tx_antennas=%d, "
                    "pilot_spacing=%d, subcarriers=%d"
                    % (tx, spacing, subcarriers)
                )
            for tx_index in range(tx):
                offset = tx_index
                mask[tx_index, offset::spacing] = True
            if np.any(np.sum(mask, axis=0) > 1):
                raise OperationError(
                    "Orthogonal comb pilot construction produced overlapping transmitter masks"
                )
            orthogonalization_axis = "subcarrier"
        # QPSK phases make division by the known pilot observable and catch the
        # common but incorrect implementation that simply copies received y.
        phases = rng.randint(0, 4, size=(tx, subcarriers))
        symbols = np.exp(0.5j * np.pi * phases).astype(np.complex64)
        symbols[~mask] = 0.0
        metadata = {
            "scenario": scenario,
            "pattern_kind": "all_subcarriers" if scenario == "unit" else "orthogonal_comb",
            "tx_antennas": tx,
            "subcarriers": subcarriers,
            "pilot_spacing": 1 if scenario == "unit" else spacing,
            "pilot_count": int(np.count_nonzero(mask)),
            "pilot_density": float(np.mean(mask)),
            "pilot_axes": ["tx_antenna", "subcarrier"],
            "orthogonal_tx_slots": True,
            "orthogonalization_axis": orthogonalization_axis,
            "seed": int(seed),
        }
        path = ctx.output_path("pilots", ".npz")
        np.savez_compressed(
            path,
            pilots=symbols,
            pilot_mask=mask.astype(np.uint8),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"pilots": artifact("ai_phy.pilot_pattern.numpy", path, metadata)},
            metrics={"pilot.count": int(np.count_nonzero(mask)), "pilot.density": float(np.mean(mask))},
            metadata=metadata,
        )


class PilotObservationOperation(Operation):
    """Apply pilots to channel truth and materialize the noisy observation."""

    id = "wireless.pilot_observation"
    name = "Noisy pilot observation"
    input_kinds = {
        "channel": ["ai_phy.channel_realization.numpy"],
        "pilots": ["ai_phy.pilot_pattern.numpy"],
    }
    output_kinds = {
        "problem": "ai_phy.channel_estimation_problem.numpy",
        "observation": "ai_phy.channel_estimation_observation.numpy",
        "pilot_ls": "ai_phy.channel_estimation_sparse_pilot_ls.numpy",
        "pilot_mask": "ai_phy.channel_estimation_pilot_mask.numpy",
        "ls_estimate": "ai_phy.channel_estimation_ls_estimate.numpy",
        "noise_variance": "ai_phy.channel_noise_variance.numpy",
        "truth": "ai_phy.channel_estimation_truth.numpy",
    }
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Pilot multiplication and additive noise have a direct differentiable materialization.",
    }
    backends = {"benchmark_run": ["numpy", "sionna"], "dataset_capture": ["numpy", "sionna"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "snr_db": {"type": "number", "default": 12.0},
            "wireless_backend": _wireless_backend_schema(),
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _wireless_runtime_availability(params)

    def run(self, ctx: OperationContext) -> OperationResult:
        h_true, channel_metadata = _load_npz(ctx.require_input("channel").path, "h_true")
        pilot_data, pilot_metadata = _load_all_npz(ctx.require_input("pilots").path)
        pilots = np.asarray(pilot_data["pilots"], dtype=np.complex64)
        mask = np.asarray(pilot_data["pilot_mask"], dtype=bool)
        if h_true.ndim != 4:
            raise OperationError("Channel realization must have shape [example, rx, tx, subcarrier]")
        expected = (int(h_true.shape[2]), int(h_true.shape[3]))
        if pilots.shape != expected or mask.shape != expected:
            raise OperationError(
                "Pilot shape %s must exactly match channel [tx, subcarrier] shape %s" % (pilots.shape, expected)
            )
        snr_db = float(_param(ctx.params, "snr_db", 12.0))
        noise_var = _noise_variance_from_snr(snr_db)
        seed = ctx.seed("pilot_observation")
        rng = np.random.RandomState(seed)
        clean = h_true * pilots[None, None, :, :]
        noisy, selected_backend = _add_awgn(clean, noise_var, rng, str(ctx.params.get("wireless_backend") or "auto"))
        requested_backend = str(ctx.params.get("wireless_backend") or "auto")
        backend_detail = (
            "sionna.awgn.pytorch"
            if selected_backend == "sionna"
            else "numpy.complex_awgn"
        )
        observations = np.where(mask[None, None, :, :], noisy, 0.0).astype(np.complex64)
        metadata = dict(channel_metadata)
        metadata.update(
            {
                "observation_model": "orthogonal_pilot_multiplication_plus_complex_awgn",
                "snr_db": snr_db,
                "noise_variance": noise_var,
                "wireless_backend": selected_backend,
                "wireless_backend_detail": backend_detail,
                "requested_wireless_backend": requested_backend,
                "pilot_pattern": pilot_metadata.get("pattern_kind", pilot_metadata.get("scenario", "unknown")),
                "pilot_spacing": int(pilot_metadata.get("pilot_spacing") or 1),
                "pilot_density": float(np.mean(mask)),
                "seed": int(seed),
            }
        )
        problem_metadata = dict(metadata)
        problem_metadata.update(
            {
                "array": "observations",
                "capture_record_axis": 0,
                "capture_record_unit": "pilot_observation",
            }
        )
        observation_metadata = dict(problem_metadata)
        truth_metadata = dict(metadata)
        truth_metadata.update(
            {
                "array": "h_true",
                "capture_record_axis": 0,
                "capture_record_unit": "channel_truth",
            }
        )
        estimation_problem = {
            "observations": observations,
            "pilots": pilots,
            "pilot_mask": mask,
        }
        pilot_ls = _sparse_pilot_ls_channel_estimate(estimation_problem)
        pilot_mask_values = np.broadcast_to(
            mask[None, :, :],
            (int(h_true.shape[0]), int(mask.shape[0]), int(mask.shape[1])),
        ).astype(np.float32)
        ls_estimate = _interpolate_channel_from_pilots(pilot_ls, mask)
        pilot_ls_metadata = dict(metadata)
        pilot_ls_metadata.update(
            {
                "array": "pilot_ls",
                "capture_record_axis": 0,
                "capture_record_unit": "sparse_pilot_least_squares_observation",
                "preprocessing_owner": "noema_operation",
                "preprocessing": "known_pilot_division_without_interpolation",
            }
        )
        pilot_mask_metadata = dict(metadata)
        pilot_mask_metadata.update(
            {
                "array": "pilot_mask",
                "capture_record_axis": 0,
                "capture_record_unit": "pilot_observation_mask",
                "mask_axes": ["example", "tx_antenna", "subcarrier"],
            }
        )
        ls_metadata = dict(metadata)
        ls_metadata.update(
            {
                "array": "ls_estimate",
                "capture_record_axis": 0,
                "capture_record_unit": "least_squares_channel_estimate",
                "preprocessing_owner": "noema_operation",
                "preprocessing": "pilot_division_and_frequency_interpolation",
            }
        )
        noise_values = np.full((int(h_true.shape[0]), 1), noise_var, dtype=np.float32)
        noise_metadata = dict(metadata)
        noise_metadata.update(
            {
                "array": "noise_variance",
                "capture_record_axis": 0,
                "capture_record_unit": "channel_noise_variance",
            }
        )
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            path,
            h_true=h_true.astype(np.complex64),
            observations=observations,
            pilots=pilots,
            pilot_mask=mask.astype(np.uint8),
            metadata_json=json.dumps(problem_metadata, sort_keys=True),
        )
        observation_path = ctx.output_path("observation", ".npz")
        np.savez_compressed(
            observation_path,
            observations=observations,
            pilots=pilots,
            pilot_mask=mask.astype(np.uint8),
            metadata_json=json.dumps(observation_metadata, sort_keys=True),
        )
        pilot_ls_path = ctx.output_path("pilot_ls", ".npz")
        np.savez_compressed(
            pilot_ls_path,
            pilot_ls=pilot_ls.astype(np.complex64),
            metadata_json=json.dumps(pilot_ls_metadata, sort_keys=True),
        )
        pilot_mask_path = ctx.output_path("pilot_mask", ".npz")
        np.savez_compressed(
            pilot_mask_path,
            pilot_mask=pilot_mask_values,
            metadata_json=json.dumps(pilot_mask_metadata, sort_keys=True),
        )
        truth_path = ctx.output_path("truth", ".npz")
        np.savez_compressed(
            truth_path,
            h_true=h_true.astype(np.complex64),
            metadata_json=json.dumps(truth_metadata, sort_keys=True),
        )
        ls_path = ctx.output_path("ls_estimate", ".npz")
        np.savez_compressed(
            ls_path,
            ls_estimate=ls_estimate.astype(np.complex64),
            metadata_json=json.dumps(ls_metadata, sort_keys=True),
        )
        noise_path = ctx.output_path("noise_variance", ".npz")
        np.savez_compressed(
            noise_path,
            noise_variance=noise_values,
            metadata_json=json.dumps(noise_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "problem": artifact(
                    "ai_phy.channel_estimation_problem.numpy",
                    path,
                    problem_metadata,
                ),
                "observation": artifact(
                    "ai_phy.channel_estimation_observation.numpy",
                    observation_path,
                    observation_metadata,
                ),
                "pilot_ls": artifact(
                    "ai_phy.channel_estimation_sparse_pilot_ls.numpy",
                    pilot_ls_path,
                    pilot_ls_metadata,
                ),
                "pilot_mask": artifact(
                    "ai_phy.channel_estimation_pilot_mask.numpy",
                    pilot_mask_path,
                    pilot_mask_metadata,
                ),
                "ls_estimate": artifact(
                    "ai_phy.channel_estimation_ls_estimate.numpy",
                    ls_path,
                    ls_metadata,
                ),
                "noise_variance": artifact(
                    "ai_phy.channel_noise_variance.numpy",
                    noise_path,
                    noise_metadata,
                ),
                "truth": artifact(
                    "ai_phy.channel_estimation_truth.numpy",
                    truth_path,
                    truth_metadata,
                ),
            },
            metrics={"channel.snr_db": snr_db, "channel.noise_variance": noise_var, "pilot.density": float(np.mean(mask))},
            metadata=problem_metadata,
        )


class AiPhyPilotChannelSourceOperation(Operation):
    id = "source.ai_phy_pilot_channel"
    name = "AI-PHY pilot channel scenario"
    output_kinds = {"problem": "ai_phy.channel_estimation_problem.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Scenario generation is a benchmark data source; differentiable exports should materialize the channel model directly.",
    }
    backends = {"benchmark_run": ["numpy", "sionna"], "dataset_capture": ["numpy", "sionna"], "differentiable_export": ["torch", "sionna"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "numpy_pilot_awgn", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "numpy_pilot_awgn", "status": "implemented"},
        {"runner": "benchmark_run", "backend": "sionna", "implementation": "sionna_awgn_pilot_observation", "status": "implemented", "notes": "Selected only when wireless_backend=sionna; Sionna failures are fatal."},
        {"runner": "dataset_capture", "backend": "sionna", "implementation": "sionna_awgn_pilot_observation", "status": "implemented", "notes": "Selected only when wireless_backend=sionna; Sionna failures are fatal."},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_pilot_channel_scenario", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_pilot_channel_scenario", "status": "implemented"},
    ]
    equivalence = {"type": "statistical", "reason": "Pilot-observation backends must match the declared channel/noise distribution and SNR, not sample-identical noise."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "scenario": {"type": "string", "default": "pilot_awgn", "enum": ["pilot_awgn", "mimo_ofdm_pilot"]},
            "example_count": {"type": "integer", "default": 32, "minimum": 1},
            "rx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "tx_antennas": {"type": "integer", "default": 1, "minimum": 1},
            "subcarriers": {"type": "integer", "default": 1, "minimum": 1},
            "snr_db": {"type": "number", "default": 12.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
            "wireless_backend": _wireless_backend_schema(),
        }
    )

    def runtime_availability(self, params: Mapping[str, Any]) -> JsonDict:
        return _wireless_runtime_availability(params)

    def run(self, ctx: OperationContext) -> OperationResult:
        scenario = str(ctx.params.get("scenario") or "pilot_awgn")
        example_count = int(_param(ctx.params, "example_count", 32))
        rx_antennas = int(_param(ctx.params, "rx_antennas", 1))
        tx_antennas = int(_param(ctx.params, "tx_antennas", 1))
        subcarriers = int(_param(ctx.params, "subcarriers", 1))
        snr_db = float(_param(ctx.params, "snr_db", 12.0))
        seed = ctx.seed("ai_phy_pilot_channel")
        rng = np.random.RandomState(seed)
        h = _complex_normal(rng, (example_count, rx_antennas, tx_antennas, subcarriers))
        noise_var = _noise_variance_from_snr(snr_db)
        backend = str(ctx.params.get("wireless_backend") or "auto")
        observations, selected_backend = _add_awgn(h, noise_var, rng, backend)
        backend_detail = (
            "sionna.awgn.pytorch"
            if selected_backend == "sionna"
            else "numpy.complex_awgn"
        )
        metadata = {
            "dataset": "synthetic_pilot_channel",
            "scenario": scenario,
            "split": "fixed_seed",
            "example_count": int(example_count),
            "rx_antennas": int(rx_antennas),
            "tx_antennas": int(tx_antennas),
            "subcarriers": int(subcarriers),
            "channel_shape": list(h.shape),
            "channel_axes": ["example", "rx_antenna", "tx_antenna", "subcarrier"],
            "channel_variance": 1.0,
            "snr_db": snr_db,
            "noise_variance": noise_var,
            "seed": int(seed),
            "wireless_backend": selected_backend,
            "wireless_backend_detail": backend_detail,
            "requested_wireless_backend": backend,
            "sionna_backed": selected_backend == "sionna",
        }
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(path, h_true=h.astype(np.complex64), observations=observations.astype(np.complex64), metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(
            outputs={"problem": artifact("ai_phy.channel_estimation_problem.numpy", path, metadata)},
            metrics={"channel.snr_db": snr_db, "channel.noise_variance": noise_var, "ai_phy.example_count": int(example_count)},
            metadata=metadata,
        )


class LsChannelEstimatorOperation(Operation):
    id = "model.ls_channel_estimator"
    name = "LS channel estimator baseline"
    input_kinds = {
        "problem": [
            "ai_phy.channel_estimation_problem.numpy",
            "ai_phy.channel_estimation_observation.numpy",
        ]
    }
    output_kinds = {"estimate": "ai_phy.channel_estimate.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False, "reason": "Least-squares benchmark baseline is an artifact operation."}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    equivalence = {"type": "numerical", "tolerance": {"atol": 1e-7, "rtol": 1e-6}}
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_all_npz(ctx.require_input("problem").path)
        estimate = _least_squares_channel_estimate(problem)
        out_metadata = dict(metadata)
        out_metadata.update({"estimator": "ls", "estimator_label": "Least squares pilot estimator"})
        path = ctx.output_path("estimate", ".npz")
        np.savez_compressed(path, h_hat=estimate, metadata_json=json.dumps(out_metadata, sort_keys=True))
        return OperationResult(outputs={"estimate": artifact("ai_phy.channel_estimate.numpy", path, out_metadata)}, metadata=out_metadata)


class ChannelEstimatorAdapterOperation(Operation):
    id = "model.channel_estimator_adapter"
    name = "Channel estimator adapter"
    input_kinds = {
        "problem": [
            "ai_phy.channel_estimation_problem.numpy",
            "ai_phy.channel_estimation_observation.numpy",
        ]
    }
    output_kinds = {"estimate": "ai_phy.channel_estimate.numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": True,
        "exportable": True,
        "reason": (
            "The estimator slot accepts a portable learned artifact while "
            "classical LS and exponential-PDP LMMSE modes remain reproducible baselines."
        ),
    }
    backends = {
        "benchmark_run": ["numpy", "onnxruntime"],
        "dataset_capture": ["numpy", "onnxruntime"],
        "differentiable_export": ["torch", "sionna"],
    }
    trained_artifact_abi = {
        "component_id": "estimator",
        "component_role": "mimo_ofdm_channel_estimator",
        "entrypoint_id": "channel_estimator",
        "required_operation_inputs": ["problem"],
        "inputs": {
            "pilot_ls_ri": {
                "dtype": "float32",
                "shape": [
                    "batch",
                    "rx_antenna",
                    "tx_antenna",
                    "subcarrier",
                    2,
                ],
            },
            "pilot_mask": {
                "dtype": "float32",
                "shape": [
                    "batch",
                    "tx_antenna",
                    "subcarrier",
                ],
            },
            "ls_estimate_ri": {
                "dtype": "float32",
                "shape": [
                    "batch",
                    "rx_antenna",
                    "tx_antenna",
                    "subcarrier",
                    2,
                ],
            },
            "noise_variance": {
                "dtype": "float32",
                "shape": ["batch", 1],
            },
        },
        "outputs": {
            "h_hat_ri": {
                "dtype": "float32",
                "shape": [
                    "batch",
                    "rx_antenna",
                    "tx_antenna",
                    "subcarrier",
                    2,
                ],
            },
        },
        "binding_params": {
            "mode": "learned_artifact",
            "artifact_manifest_path": "trained_artifact.yaml",
            "artifact_entrypoint": "channel_estimator",
        },
    }
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "least_squares", "status": "implemented", "parameter_bindings": {"mode": "least_squares"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "linear_mmse_reference", "status": "implemented", "parameter_bindings": {"mode": "linear_mmse_reference"}},
        {"runner": "benchmark_run", "backend": "onnxruntime", "implementation": "portable_trained_artifact_runtime", "status": "implemented", "parameter_bindings": {"mode": "learned_artifact"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "least_squares", "status": "implemented", "parameter_bindings": {"mode": "least_squares"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "linear_mmse_reference", "status": "implemented", "parameter_bindings": {"mode": "linear_mmse_reference"}},
        {"runner": "dataset_capture", "backend": "onnxruntime", "implementation": "portable_trained_artifact_runtime", "status": "implemented", "parameter_bindings": {"mode": "learned_artifact"}},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_channel_estimator_endpoint", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_receiver_training_endpoint", "status": "implemented"},
    ]
    equivalence = {"type": "behavioral", "reason": "Adapter implementations are compared by NMSE/task metrics under the declared benchmark protocol."}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "linear_mmse_reference",
                "enum": [
                    "least_squares",
                    "linear_mmse_reference",
                    "learned_artifact",
                ],
                "title": "Reference method",
                "description": (
                    "Runnable estimator. The learned mode uses a returned "
                    "portable artifact; the other modes are fixed baselines."
                ),
            },
            "artifact_manifest_path": {
                "type": "string",
                "default": "",
                "description": (
                    "Registered schema-v2 trained artifact implementing the "
                    "MIMO-OFDM channel-estimator ABI."
                ),
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
                "default": "channel_estimator",
                "x-noema-ui": {"hidden": True},
            },
            "artifact_package_sha256": {
                "type": "string",
                "default": "",
                "x-noema-ui": {"hidden": True},
            },
            "lmmse_assumed_tap_count": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "title": "Assumed LMMSE taps",
                "description": (
                    "Fixed exponential-PDP prior used by the practical LMMSE "
                    "baseline. It is not adapted from hidden channel truth."
                ),
                "x-noema-ui": {
                    "visible_when": {"mode": "linear_mmse_reference"},
                },
            },
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_all_npz(ctx.require_input("problem").path)
        mode = str(ctx.params.get("mode") or "linear_mmse_reference")
        ls_estimate = _least_squares_channel_estimate(problem)
        checkpoint_sha = None
        selected_backend = "numpy"
        if mode == "least_squares":
            estimate = ls_estimate
            reference_estimator = "least_squares_with_pilot_interpolation"
            estimator_label = "Least-squares reference"
            shrinkage = None
        elif mode == "linear_mmse_reference":
            noise_var = float(metadata.get("noise_variance") or 0.0)
            signal_var = max(float(metadata.get("channel_variance") or 1.0), 1e-12)
            shrinkage = signal_var / (signal_var + max(noise_var, 0.0))
            estimate = _linear_mmse_channel_estimate(
                problem,
                noise_variance=max(noise_var, 0.0),
                channel_variance=signal_var,
                channel_tap_count=int(
                    ctx.params.get("lmmse_assumed_tap_count") or 4
                ),
            )
            reference_estimator = (
                "fixed_exponential_pdp_frequency_covariance_linear_mmse"
            )
            estimator_label = "Fixed-prior exponential-PDP LMMSE"
        elif mode == "learned_artifact":
            manifest_value = str(
                ctx.params.get("artifact_manifest_path") or ""
            ).strip()
            entrypoint = str(
                ctx.params.get("artifact_entrypoint") or "channel_estimator"
            ).strip()
            if not manifest_value:
                raise OperationError(
                    "learned_artifact channel estimator requires "
                    "params.artifact_manifest_path"
                )
            manifest_path = Path(manifest_value).expanduser()
            if not manifest_path.is_file():
                raise OperationError(
                    "Channel-estimator trained-artifact manifest does not exist: %s"
                    % manifest_path
                )
            ls_ri = np.stack(
                [ls_estimate.real, ls_estimate.imag], axis=-1
            ).astype(np.float32)
            pilot_ls = _sparse_pilot_ls_channel_estimate(problem)
            pilot_ls_ri = np.stack(
                [pilot_ls.real, pilot_ls.imag], axis=-1
            ).astype(np.float32)
            raw_mask = np.asarray(problem.get("pilot_mask"), dtype=bool)
            if raw_mask.ndim != 2:
                raise OperationError(
                    "Learned channel estimator requires a [tx, subcarrier] pilot mask"
                )
            pilot_mask = np.broadcast_to(
                raw_mask[None, :, :],
                (int(ls_ri.shape[0]), int(raw_mask.shape[0]), int(raw_mask.shape[1])),
            ).astype(np.float32)
            noise_variance = np.full(
                (int(ls_ri.shape[0]), 1),
                float(metadata.get("noise_variance") or 0.0),
                dtype=np.float32,
            )
            try:
                from noema_lab.core.trained_artifact_runtime import (
                    run_trained_artifact_entrypoint,
                )

                outputs = run_trained_artifact_entrypoint(
                    manifest_path,
                    entrypoint,
                    {
                        "pilot_ls_ri": pilot_ls_ri,
                        "pilot_mask": pilot_mask,
                        "ls_estimate_ri": ls_ri,
                        "noise_variance": noise_variance,
                    },
                    expected_package_sha256=str(
                        ctx.params.get("artifact_package_sha256") or ""
                    ),
                )
            except Exception as exc:
                raise OperationError(
                    "Channel-estimator trained-artifact inference failed: %s"
                    % exc
                ) from exc
            if "h_hat_ri" not in outputs:
                raise OperationError(
                    "Channel-estimator trained artifact did not return `h_hat_ri`"
                )
            estimate_ri = np.asarray(outputs["h_hat_ri"], dtype=np.float32)
            if tuple(estimate_ri.shape) != tuple(ls_ri.shape):
                raise OperationError(
                    "Channel-estimator artifact h_hat_ri must have shape %s, got %s"
                    % (tuple(ls_ri.shape), tuple(estimate_ri.shape))
                )
            if not np.all(np.isfinite(estimate_ri)):
                raise OperationError(
                    "Channel-estimator artifact returned non-finite values"
                )
            estimate = (
                estimate_ri[..., 0] + 1j * estimate_ri[..., 1]
            ).astype(np.complex64)
            reference_estimator = "portable_learned_channel_estimator"
            estimator_label = "Portable learned MIMO-OFDM estimator"
            shrinkage = None
            checkpoint_sha = file_sha256(manifest_path)
            selected_backend = "onnxruntime"
        else:
            raise OperationError("Unsupported channel-estimator adapter mode `%s`" % mode)
        out_metadata = dict(metadata)
        out_metadata.update(
            {
                "estimator": "adapter",
                "adapter_mode": mode,
                "estimator_label": estimator_label,
                "reference_estimator": reference_estimator,
                "data_plane_backend": selected_backend,
            }
        )
        if checkpoint_sha:
            out_metadata["checkpoint_sha256"] = checkpoint_sha
            out_metadata["artifact_manifest_sha256"] = checkpoint_sha
        if shrinkage is not None:
            out_metadata["lmmse_shrinkage"] = float(shrinkage)
            out_metadata["lmmse_assumed_tap_count"] = int(
                ctx.params.get("lmmse_assumed_tap_count") or 4
            )
        path = ctx.output_path("estimate", ".npz")
        np.savez_compressed(path, h_hat=estimate.astype(np.complex64), metadata_json=json.dumps(out_metadata, sort_keys=True))
        return OperationResult(outputs={"estimate": artifact("ai_phy.channel_estimate.numpy", path, out_metadata)}, metadata=out_metadata)


class ChannelEstimationMetricsOperation(Operation):
    id = "metrics.channel_estimation"
    name = "Channel-estimation metrics"
    input_kinds = {
        "problem": [
            "ai_phy.channel_estimation_problem.numpy",
            "ai_phy.channel_estimation_truth.numpy",
        ],
        "estimate": ["ai_phy.channel_estimate.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        h_true, metadata = _load_npz(ctx.require_input("problem").path, "h_true")
        h_hat, estimate_metadata = _load_npz(ctx.require_input("estimate").path, "h_hat")
        if h_hat.shape != h_true.shape:
            raise OperationError(
                "Channel estimate shape %s must exactly match channel truth shape %s" % (h_hat.shape, h_true.shape)
            )
        denom = float(np.mean(np.abs(h_true) ** 2)) if h_true.size else 1.0
        mse = float(np.mean(np.abs(h_hat - h_true) ** 2)) if h_true.size else 0.0
        nmse = mse / max(denom, 1e-12)
        score = 1.0 / (1.0 + nmse)
        nmse_db = float(10.0 * math.log10(max(nmse, 1e-12)))
        correlation_denominator = math.sqrt(
            float(np.vdot(h_true.reshape(-1), h_true.reshape(-1)).real)
            * float(np.vdot(h_hat.reshape(-1), h_hat.reshape(-1)).real)
        )
        complex_correlation = (
            float(
                abs(
                    np.vdot(
                        h_true.reshape(-1),
                        h_hat.reshape(-1),
                    )
                )
                / correlation_denominator
            )
            if correlation_denominator > 1e-30
            else 0.0
        )
        zf_rate, oracle_zf_rate = _zf_spectral_efficiency(
            h_true,
            h_hat,
            noise_variance=float(metadata.get("noise_variance") or 1.0),
        )
        zf_retention = (
            zf_rate / oracle_zf_rate if oracle_zf_rate > 1e-12 else 0.0
        )
        metrics = {
            "channel_estimation.mse": mse,
            "channel_estimation.nmse": nmse,
            "channel_estimation.nmse_db": nmse_db,
            "channel_estimation.complex_correlation": complex_correlation,
            "mimo.channel_estimation.nmse": nmse,
            "mimo.channel_estimation.nmse_db": nmse_db,
            "mimo.channel_estimation.zf_spectral_efficiency_bps_hz": zf_rate,
            "mimo.channel_estimation.perfect_csi_zf_spectral_efficiency_bps_hz": oracle_zf_rate,
            "mimo.channel_estimation.zf_rate_retention": zf_retention,
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "pilot.density": float(metadata.get("pilot_density") or 0.0),
            "task.score": score,
        }
        preview = _channel_estimation_preview(
            h_true,
            h_hat,
            metadata=metadata,
            estimate_metadata=estimate_metadata,
        )
        report_metadata = {
            "problem_metadata": metadata,
            "estimate_metadata": estimate_metadata,
            "channel_estimation_preview": preview,
        }
        rows = [
            {
                "id": "mean",
                "mse": mse,
                "nmse": nmse,
                "nmse_db": nmse_db,
                "complex_correlation": complex_correlation,
                "zf_spectral_efficiency_bps_hz": zf_rate,
                "perfect_csi_zf_spectral_efficiency_bps_hz": oracle_zf_rate,
                "zf_rate_retention": zf_retention,
            }
        ]
        return _write_report(
            ctx,
            "channel_estimation",
            rows,
            metrics,
            report_metadata,
        )


class BeamformingScenarioSourceOperation(Operation):
    id = "source.beamforming_scenario"
    name = "Beamforming scenario source"
    output_kinds = {"problem": "ai_phy.beamforming_problem.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "synthetic_miso_channel_batch", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "synthetic_miso_channel_batch", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_miso_channel_batch", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_miso_channel_batch", "status": "planned", "notes": "Use Sionna PHY/RT channel materializations as this suite matures."},
    ]
    equivalence = {"type": "statistical"}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema({"example_count": {"type": "integer", "default": 32, "minimum": 1}, "tx_antennas": {"type": "integer", "default": 8, "minimum": 2}, "snr_db": {"type": "number", "default": 10.0}, "seed": {"type": "integer", "default": 0, "minimum": 0}})

    def run(self, ctx: OperationContext) -> OperationResult:
        n = int(_param(ctx.params, "example_count", 32))
        tx = int(_param(ctx.params, "tx_antennas", 8))
        snr_db = float(_param(ctx.params, "snr_db", 10.0))
        seed = ctx.seed("beamforming_scenario")
        rng = np.random.RandomState(seed)
        channels = _complex_normal(rng, (n, tx))
        codebook = _dft_codebook(tx).astype(np.complex64)
        metadata = {"dataset": "synthetic_beamforming", "split": "fixed_seed", "example_count": n, "tx_antennas": tx, "snr_db": snr_db, "seed": int(seed)}
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(path, channels=channels.astype(np.complex64), codebook=codebook, metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(outputs={"problem": artifact("ai_phy.beamforming_problem.numpy", path, metadata)}, metrics={"channel.snr_db": snr_db, "ai_phy.example_count": n}, metadata=metadata)


class MrtBeamformerOperation(Operation):
    id = "model.mrt_beamformer"
    name = "MRT/codebook beamformer baseline"
    input_kinds = {"problem": ["ai_phy.beamforming_problem.numpy"]}
    output_kinds = {"decision": "ai_phy.beamforming_decision.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        channels, metadata = _load_npz(ctx.require_input("problem").path, "channels")
        weights = _mrt_weights(channels)
        return _write_beam_decision(ctx, weights, metadata, "mrt", "Maximum-ratio transmission baseline")


class BeamformingAdapterOperation(Operation):
    id = "model.beamforming_adapter"
    name = "Beamforming/precoding adapter"
    input_kinds = {"problem": ["ai_phy.beamforming_problem.numpy"]}
    output_kinds = {"decision": "ai_phy.beamforming_decision.numpy"}
    differentiability = {"framework": "torch", "gradient": "surrogate", "trainable_params": True, "exportable": True, "reason": "Surrogate-gradient metadata describes this adapter only when retained as downstream support. Portable replacement is not available until a trained-artifact ABI and runtime binding are implemented."}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "maximum_ratio_transmission", "status": "implemented", "parameter_bindings": {"mode": "mrt"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "codebook_sweep_reference", "status": "implemented", "parameter_bindings": {"mode": "codebook_sweep_reference"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "maximum_ratio_transmission", "status": "implemented", "parameter_bindings": {"mode": "mrt"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "codebook_sweep_reference", "status": "implemented", "parameter_bindings": {"mode": "codebook_sweep_reference"}},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_beam_policy_endpoint", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_precoder_training_endpoint", "status": "planned"},
    ]
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "codebook_sweep_reference",
                "enum": ["mrt", "codebook_sweep_reference"],
                "title": "Reference method",
                "description": "Runnable beamforming method used until a trained beam-policy artifact is bound to this research slot.",
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        channels = data["channels"].astype(np.complex64)
        mode = str(ctx.params.get("mode") or "codebook_sweep_reference")
        if mode == "mrt":
            weights = _mrt_weights(channels)
            label = "Maximum-ratio transmission reference"
        elif mode == "codebook_sweep_reference":
            codebook = data["codebook"].astype(np.complex64)
            gains = np.abs(channels @ np.conjugate(codebook).T) ** 2
            indices = np.argmax(gains, axis=1)
            weights = codebook[indices]
            label = "DFT-codebook sweep reference"
        else:
            raise OperationError("Unsupported beamforming adapter mode `%s`" % mode)
        return _write_beam_decision(ctx, weights, metadata, mode, label)


class BeamformingMetricsOperation(Operation):
    id = "metrics.beamforming"
    name = "Beamforming metrics"
    input_kinds = {"problem": ["ai_phy.beamforming_problem.numpy"], "decision": ["ai_phy.beamforming_decision.numpy"]}
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        channels, metadata = _load_npz(ctx.require_input("problem").path, "channels")
        weights, decision_metadata = _load_npz(ctx.require_input("decision").path, "weights")
        weights, norm_error = _validated_beamforming_weights(channels, weights)
        gain = np.abs(np.sum(channels * np.conjugate(weights), axis=1)) ** 2
        optimum = np.sum(np.abs(channels) ** 2, axis=1)
        snr_linear = 10.0 ** (float(metadata.get("snr_db") or 0.0) / 10.0)
        rates = np.log2(1.0 + snr_linear * gain)
        norm_gain = np.clip(gain / np.maximum(optimum, 1e-12), 0.0, 1.0)
        metrics = {
            "beamforming.spectral_efficiency_bps_hz": float(np.mean(rates)),
            "beamforming.normalized_gain": float(np.mean(norm_gain)),
            "beamforming.array_gain_db": _db(float(np.mean(gain))),
            "beamforming.unit_norm_max_error": norm_error,
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "task.score": float(np.mean(norm_gain)),
        }
        rows = [{"id": int(i), "rate_bps_hz": float(rates[i]), "normalized_gain": float(norm_gain[i])} for i in range(min(8, len(rates)))]
        return _write_report(ctx, "beamforming", rows, metrics, {"problem_metadata": metadata, "decision_metadata": decision_metadata})


class LocalizationGeometrySourceOperation(Operation):
    """Materialize anchor/target geometry without embedding measurement noise."""

    id = "source.localization_geometry"
    name = "Generated localization geometry"
    output_kinds = {"scene": "ai_phy.localization_scene.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "example_count": {
                "type": "integer",
                "default": 16,
                "minimum": 1,
                "title": "Generated examples",
                "description": "Number of synthetic target positions generated for this run.",
            },
            "area_m": {
                "type": "number",
                "default": 20.0,
                "minimum": 1.0,
                "title": "Square area side (m)",
                "description": "Side length of the generated two-dimensional localization area.",
            },
            "anchor_count": {
                "type": "integer",
                "default": 4,
                "minimum": 3,
                "title": "Perimeter anchors",
                "description": "Number of anchors generated at equal intervals around the square perimeter.",
            },
            "position_mode": {
                "type": "string",
                "default": "random_uniform",
                "enum": ["random_uniform", "fixed"],
                "title": "Target placement",
                "description": "Generate seeded random targets or repeat one user-selected target position.",
            },
            "target_x_m": {
                "type": "number",
                "default": 10.0,
                "minimum": 0.0,
                "title": "Fixed target x (m)",
                "description": "Used only for fixed placement; must lie inside the configured area.",
            },
            "target_y_m": {
                "type": "number",
                "default": 10.0,
                "minimum": 0.0,
                "title": "Fixed target y (m)",
                "description": "Used only for fixed placement; must lie inside the configured area.",
            },
            "seed": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "title": "Generation seed",
                "description": "Controls repeatable random target placement.",
            },
        }
    )
    params_schema.update(
        {
            "title": "Generated localization scene",
            "description": "Generates anchors and target positions in a synthetic square environment; no dataset is imported.",
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        n = int(_param(ctx.params, "example_count", 16))
        area = float(_param(ctx.params, "area_m", 20.0))
        anchor_count = int(_param(ctx.params, "anchor_count", 4))
        position_mode = str(_param(ctx.params, "position_mode", "random_uniform"))
        seed = ctx.seed("localization_geometry")
        rng = np.random.RandomState(seed)
        anchors = _square_perimeter_anchors(area, anchor_count)
        if position_mode == "random_uniform":
            positions = rng.uniform(0.15 * area, 0.85 * area, size=(n, 2)).astype(np.float32)
        elif position_mode == "fixed":
            target_x_m = float(_param(ctx.params, "target_x_m", area / 2.0))
            target_y_m = float(_param(ctx.params, "target_y_m", area / 2.0))
            if not 0.0 <= target_x_m <= area or not 0.0 <= target_y_m <= area:
                raise OperationError(
                    "Fixed localization target (%.3f, %.3f) m must lie inside [0, %.3f] m on both axes"
                    % (target_x_m, target_y_m, area)
                )
            positions = np.repeat(
                np.asarray([[target_x_m, target_y_m]], dtype=np.float32),
                n,
                axis=0,
            )
        else:
            raise OperationError("Unknown localization position_mode `%s`" % position_mode)
        metadata = {
            "dataset": "synthetic_localization_geometry",
            "data_origin": "synthetic_generated",
            "scenario_kind": "two_dimensional_multilateration",
            "split": "fixed_seed",
            "example_count": n,
            "area_m": area,
            "anchor_count": anchor_count,
            "position_mode": position_mode,
            "coordinate_system": "cartesian_xy_metres",
            "seed": int(seed),
            "sionna_rt_contract": "range_geometry_v1",
        }
        path = ctx.output_path("scene", ".npz")
        np.savez_compressed(
            path,
            anchors=anchors.astype(np.float32),
            positions=positions,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"scene": artifact("ai_phy.localization_scene.numpy", path, metadata)},
            metrics={"localization.example_count": n, "localization.anchor_count": anchor_count},
            metadata=metadata,
        )


class RangeObservationOperation(Operation):
    """Generate range observations whose error variance is explicitly SNR-driven."""

    id = "wireless.range_observation"
    name = "Noisy range observation"
    input_kinds = {"scene": ["ai_phy.localization_scene.numpy"]}
    output_kinds = {
        "problem": "ai_phy.localization_problem.numpy",
        "observation": "ai_phy.localization_observation.numpy",
        "truth": "ai_phy.localization_truth.numpy",
    }
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "Euclidean range and additive measurement noise have a direct differentiable materialization.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "snr_db": {"type": "number", "default": 20.0},
            "range_noise_floor_m": {"type": "number", "default": 0.05, "minimum": 0.0},
            "nlos_probability": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
            "nlos_bias_m": {"type": "number", "default": 1.5, "minimum": 0.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        scene, scene_metadata = _load_all_npz(ctx.require_input("scene").path)
        anchors = np.asarray(scene["anchors"], dtype=np.float32)
        positions = np.asarray(scene["positions"], dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or positions.ndim != 2 or positions.shape[1] != 2:
            raise OperationError("Localization scene anchors and positions must have exact [count, 2] shapes")
        true_ranges = np.linalg.norm(positions[:, None, :] - anchors[None, :, :], axis=2).astype(np.float32)
        snr_db = float(_param(ctx.params, "snr_db", 20.0))
        floor_m = float(_param(ctx.params, "range_noise_floor_m", 0.05))
        nlos_probability = float(_param(ctx.params, "nlos_probability", 0.0))
        nlos_bias_m = float(_param(ctx.params, "nlos_bias_m", 1.5))
        # Relate measurement standard deviation to amplitude SNR.  The quarter-
        # area reference is fixed by scene geometry, while the optional floor
        # represents clock/quantization error that does not vanish at high SNR.
        reference_range_m = max(float(scene_metadata.get("area_m") or 1.0) * 0.25, 1e-6)
        snr_sigma_m = reference_range_m * 10.0 ** (-snr_db / 20.0)
        sigma_m = math.sqrt(floor_m ** 2 + snr_sigma_m ** 2)
        seed = ctx.seed("range_observation")
        rng = np.random.RandomState(seed)
        gaussian_error = rng.normal(0.0, sigma_m, size=true_ranges.shape).astype(np.float32)
        nlos_mask = rng.uniform(size=true_ranges.shape) < nlos_probability
        nlos_bias = np.where(nlos_mask, nlos_bias_m, 0.0).astype(np.float32)
        ranges = np.maximum(true_ranges + gaussian_error + nlos_bias, 0.0).astype(np.float32)
        metadata = dict(scene_metadata)
        metadata.update(
            {
                "observation_model": "range_awgn_with_optional_positive_nlos_bias",
                "snr_db": snr_db,
                "range_reference_m": reference_range_m,
                "range_noise_floor_m": floor_m,
                "range_noise_from_snr_m": snr_sigma_m,
                "range_noise_m": sigma_m,
                "nlos_probability": nlos_probability,
                "nlos_bias_m": nlos_bias_m,
                "nlos_fraction": float(np.mean(nlos_mask)),
                "seed": int(seed),
            }
        )
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            path,
            anchors=anchors,
            positions=positions,
            ranges=ranges,
            true_ranges=true_ranges,
            nlos_mask=nlos_mask.astype(np.uint8),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        observation_path = ctx.output_path("observation", ".npz")
        np.savez_compressed(
            observation_path,
            anchors=anchors,
            ranges=ranges,
            nlos_mask=nlos_mask.astype(np.uint8),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        truth_path = ctx.output_path("truth", ".npz")
        np.savez_compressed(
            truth_path,
            anchors=anchors,
            positions=positions,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "problem": artifact("ai_phy.localization_problem.numpy", path, metadata),
                "observation": artifact("ai_phy.localization_observation.numpy", observation_path, metadata),
                "truth": artifact("ai_phy.localization_truth.numpy", truth_path, metadata),
            },
            metrics={
                "channel.snr_db": snr_db,
                "localization.range_noise_m": sigma_m,
                "localization.nlos_fraction": float(np.mean(nlos_mask)),
            },
            metadata=metadata,
        )


class LocalizationScenarioSourceOperation(Operation):
    id = "source.localization_sensing_scenario"
    name = "Localization/sensing geometry source"
    output_kinds = {"problem": "ai_phy.localization_problem.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "geometric_range_measurements", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "geometric_range_measurements", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_range_geometry", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_rt_scene_endpoint", "status": "planned", "notes": "Future Sionna RT scenes should materialize this same range/AoA target contract."},
    ]
    params_schema = object_schema({"example_count": {"type": "integer", "default": 16, "minimum": 1}, "area_m": {"type": "number", "default": 20.0, "minimum": 1.0}, "range_noise_m": {"type": "number", "default": 0.25, "minimum": 0.0}, "snr_db": {"type": "number", "default": 20.0}, "seed": {"type": "integer", "default": 0, "minimum": 0}})

    def run(self, ctx: OperationContext) -> OperationResult:
        n = int(_param(ctx.params, "example_count", 16))
        area = float(_param(ctx.params, "area_m", 20.0))
        sigma = float(_param(ctx.params, "range_noise_m", 0.25))
        snr_db = float(_param(ctx.params, "snr_db", 20.0))
        seed = ctx.seed("localization_sensing")
        rng = np.random.RandomState(seed)
        anchors = np.array([[0, 0], [area, 0], [0, area], [area, area]], dtype=np.float32)
        positions = rng.uniform(0.15 * area, 0.85 * area, size=(n, 2)).astype(np.float32)
        ranges = np.linalg.norm(positions[:, None, :] - anchors[None, :, :], axis=2).astype(np.float32)
        noisy = ranges + rng.normal(0.0, sigma, size=ranges.shape).astype(np.float32)
        metadata = {"dataset": "synthetic_localization_geometry", "split": "fixed_seed", "example_count": n, "area_m": area, "range_noise_m": sigma, "snr_db": snr_db, "seed": int(seed), "sionna_rt_contract": "range_geometry_v1"}
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(path, anchors=anchors, positions=positions, ranges=noisy.astype(np.float32), true_ranges=ranges, metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(outputs={"problem": artifact("ai_phy.localization_problem.numpy", path, metadata)}, metrics={"channel.snr_db": snr_db, "localization.range_noise_m": sigma}, metadata=metadata)


class TrilaterationLocalizationOperation(Operation):
    id = "model.trilateration_localizer"
    name = "Trilateration localization baseline"
    input_kinds = {
        "problem": ["ai_phy.localization_problem.numpy", "ai_phy.localization_observation.numpy"]
    }
    output_kinds = {"estimate": "ai_phy.localization_estimate.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        positions = _linear_trilateration(data["anchors"], data["ranges"])
        return _write_position_estimate(ctx, positions, metadata, "trilateration", data)


class LocalizationAdapterOperation(Operation):
    id = "model.localization_adapter"
    name = "Localization/sensing adapter"
    input_kinds = {
        "problem": ["ai_phy.localization_problem.numpy", "ai_phy.localization_observation.numpy"]
    }
    output_kinds = {"estimate": "ai_phy.localization_estimate.numpy"}
    differentiability = {"framework": "torch", "gradient": "surrogate", "trainable_params": True, "exportable": True, "reason": "Surrogate-gradient metadata describes this adapter only when retained as downstream support. Portable replacement is not available until a trained-artifact ABI and runtime binding are implemented."}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "linear_trilateration", "status": "implemented", "parameter_bindings": {"mode": "trilateration"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "centroid_regularized_trilateration", "status": "implemented", "parameter_bindings": {"mode": "regularized_trilateration"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "linear_trilateration", "status": "implemented", "parameter_bindings": {"mode": "trilateration"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "centroid_regularized_trilateration", "status": "implemented", "parameter_bindings": {"mode": "regularized_trilateration"}},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_localization_endpoint", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_rt_localization_endpoint", "status": "planned"},
    ]
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "regularized_trilateration",
                "enum": ["trilateration", "regularized_trilateration"],
                "title": "Reference method",
                "description": "Runnable localization method used until a trained localization artifact is bound to this research slot.",
            }
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        positions = _linear_trilateration(data["anchors"], data["ranges"])
        mode = str(ctx.params.get("mode") or "regularized_trilateration")
        if mode == "regularized_trilateration":
            centroid = np.mean(data["anchors"], axis=0, keepdims=True)
            positions = 0.95 * positions + 0.05 * centroid
        elif mode != "trilateration":
            raise OperationError("Unsupported localization adapter mode `%s`" % mode)
        return _write_position_estimate(ctx, positions, metadata, mode, data)


class LocalizationMetricsOperation(Operation):
    id = "metrics.localization"
    name = "Localization metrics"
    input_kinds = {
        "problem": ["ai_phy.localization_problem.numpy", "ai_phy.localization_truth.numpy"],
        "estimate": ["ai_phy.localization_estimate.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        estimate, estimate_metadata = _load_npz(ctx.require_input("estimate").path, "positions")
        errors = np.linalg.norm(estimate - data["positions"], axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(errors))
        metrics = {"localization.rmse_m": rmse, "localization.mae_m": mae, "localization.p90_error_m": float(np.percentile(errors, 90)), "channel.snr_db": float(metadata.get("snr_db") or 0.0), "task.score": 1.0 / (1.0 + rmse)}
        rows = [{"id": int(i), "error_m": float(errors[i])} for i in range(min(8, len(errors)))]
        return _write_report(
            ctx,
            "localization",
            rows,
            metrics,
            {
                "problem_metadata": metadata,
                "estimate_metadata": estimate_metadata,
                "localization_preview": _localization_preview(data, estimate),
            },
        )


class AoaSceneSourceOperation(Operation):
    """Far-field, single-source azimuth scene for a uniform linear array."""

    id = "source.aoa_scene"
    name = "Generated AoA far-field scene"
    output_kinds = {"scene": "ai_phy.aoa_scene.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "example_count": {
                "type": "integer",
                "default": 16,
                "minimum": 1,
                "title": "Generated examples",
                "description": "Number of synthetic far-field source angles generated for this run.",
            },
            "angle_mode": {
                "type": "string",
                "default": "random_uniform",
                "enum": ["random_uniform", "fixed"],
                "title": "Source-angle mode",
                "description": "Generate seeded random bearings or repeat one user-selected bearing.",
            },
            "angle_min_deg": {
                "type": "number",
                "default": -60.0,
                "minimum": -89.0,
                "maximum": 89.0,
                "title": "Minimum generated bearing (deg)",
                "description": "Lower bound for random broadside azimuth generation.",
            },
            "angle_max_deg": {
                "type": "number",
                "default": 60.0,
                "minimum": -89.0,
                "maximum": 89.0,
                "title": "Maximum generated bearing (deg)",
                "description": "Upper bound for random broadside azimuth generation.",
            },
            "source_angle_deg": {
                "type": "number",
                "default": 0.0,
                "minimum": -89.0,
                "maximum": 89.0,
                "title": "Fixed source bearing (deg)",
                "description": "Used only in fixed mode; 0 degrees is array broadside.",
            },
            "seed": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "title": "Generation seed",
                "description": "Controls repeatable random source-angle generation.",
            },
        }
    )
    params_schema.update(
        {
            "title": "Generated AoA scene",
            "description": "Generates a synthetic single-source, far-field, narrowband AoA scene; no dataset is imported.",
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        n = int(_param(ctx.params, "example_count", 16))
        angle_mode = str(_param(ctx.params, "angle_mode", "random_uniform"))
        angle_min = float(_param(ctx.params, "angle_min_deg", -60.0))
        angle_max = float(_param(ctx.params, "angle_max_deg", 60.0))
        if angle_max <= angle_min:
            raise OperationError("angle_max_deg must be greater than angle_min_deg")
        seed = ctx.seed("aoa_scene")
        rng = np.random.RandomState(seed)
        if angle_mode == "random_uniform":
            angles_deg = rng.uniform(angle_min, angle_max, size=n).astype(np.float32)
        elif angle_mode == "fixed":
            source_angle_deg = float(_param(ctx.params, "source_angle_deg", 0.0))
            if not -89.0 <= source_angle_deg <= 89.0:
                raise OperationError("Fixed AoA source bearing must lie inside the physical (-90, 90) degree ULA range")
            if not angle_min <= source_angle_deg <= angle_max:
                raise OperationError(
                    "Fixed AoA source bearing %.3f degrees must lie inside the configured [%.3f, %.3f] degree search range"
                    % (source_angle_deg, angle_min, angle_max)
                )
            angles_deg = np.full((n,), source_angle_deg, dtype=np.float32)
        else:
            raise OperationError("Unknown AoA angle_mode `%s`" % angle_mode)
        metadata = {
            "dataset": "synthetic_ula_aoa",
            "data_origin": "synthetic_generated",
            "scenario_kind": "single_source_far_field_narrowband",
            "split": "fixed_seed",
            "example_count": n,
            "source_count": 1,
            "angle_mode": angle_mode,
            "angle_convention": "broadside_azimuth_degrees",
            "angle_min_deg": angle_min,
            "angle_max_deg": angle_max,
            "seed": int(seed),
            "sionna_rt_contract": "arrival_angle_v1",
        }
        path = ctx.output_path("scene", ".npz")
        np.savez_compressed(path, angles_deg=angles_deg, metadata_json=json.dumps(metadata, sort_keys=True))
        return OperationResult(
            outputs={"scene": artifact("ai_phy.aoa_scene.numpy", path, metadata)},
            metrics={"aoa.example_count": n},
            metadata=metadata,
        )


class UlaArrayObservationOperation(Operation):
    """Simulate noisy narrowband snapshots from a half-wavelength ULA."""

    id = "wireless.ula_array_observation"
    name = "ULA noisy array observation"
    input_kinds = {"scene": ["ai_phy.aoa_scene.numpy"]}
    output_kinds = {
        "problem": "ai_phy.aoa_problem.numpy",
        "observation": "ai_phy.aoa_observation.numpy",
        "truth": "ai_phy.aoa_truth.numpy",
    }
    differentiability = {
        "framework": "torch",
        "gradient": "full",
        "trainable_params": False,
        "exportable": True,
        "reason": "ULA steering, source mixing and additive noise have direct tensor materializations.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "antenna_count": {"type": "integer", "default": 8, "minimum": 2},
            "snapshot_count": {"type": "integer", "default": 64, "minimum": 2},
            "snr_db": {"type": "number", "default": 15.0},
            "element_spacing_wavelengths": {"type": "number", "default": 0.5, "minimum": 0.05, "maximum": 0.5},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        angles_deg, scene_metadata = _load_npz(ctx.require_input("scene").path, "angles_deg")
        angles_deg = np.asarray(angles_deg, dtype=np.float32).reshape(-1)
        antenna_count = int(_param(ctx.params, "antenna_count", 8))
        snapshot_count = int(_param(ctx.params, "snapshot_count", 64))
        snr_db = float(_param(ctx.params, "snr_db", 15.0))
        spacing = float(_param(ctx.params, "element_spacing_wavelengths", 0.5))
        noise_var = _noise_variance_from_snr(snr_db)
        seed = ctx.seed("ula_array_observation")
        rng = np.random.RandomState(seed)
        steering = _ula_steering_vectors(angles_deg, antenna_count, spacing)
        source_symbols = _complex_normal(rng, (angles_deg.size, snapshot_count))
        clean = steering[:, :, None] * source_symbols[:, None, :]
        noise = math.sqrt(noise_var) * _complex_normal(rng, clean.shape)
        snapshots = (clean + noise).astype(np.complex64)
        metadata = dict(scene_metadata)
        metadata.update(
            {
                "observation_model": "single_source_ula_narrowband_complex_awgn",
                "antenna_count": antenna_count,
                "snapshot_count": snapshot_count,
                "element_spacing_wavelengths": spacing,
                "snr_db": snr_db,
                "noise_variance": noise_var,
                "array_axes": ["example", "antenna", "snapshot"],
                "seed": int(seed),
            }
        )
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            path,
            snapshots=snapshots,
            angles_deg=angles_deg,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        observation_path = ctx.output_path("observation", ".npz")
        np.savez_compressed(
            observation_path,
            snapshots=snapshots,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        truth_path = ctx.output_path("truth", ".npz")
        np.savez_compressed(
            truth_path,
            angles_deg=angles_deg,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={
                "problem": artifact("ai_phy.aoa_problem.numpy", path, metadata),
                "observation": artifact("ai_phy.aoa_observation.numpy", observation_path, metadata),
                "truth": artifact("ai_phy.aoa_truth.numpy", truth_path, metadata),
            },
            metrics={"channel.snr_db": snr_db, "channel.noise_variance": noise_var, "aoa.snapshot_count": snapshot_count},
            metadata=metadata,
        )


class MusicAoaEstimatorOperation(Operation):
    id = "model.music_aoa_estimator"
    name = "MUSIC AoA estimator baseline"
    input_kinds = {"problem": ["ai_phy.aoa_problem.numpy", "ai_phy.aoa_observation.numpy"]}
    output_kinds = {"estimate": "ai_phy.aoa_estimate.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Classical sample-covariance eigendecomposition and grid search benchmark.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema(
        {
            "grid_size": {"type": "integer", "default": 721, "minimum": 2},
            "grid_step_deg": {"type": "number", "minimum": 0.01, "maximum": 5.0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_all_npz(ctx.require_input("problem").path)
        if ctx.params.get("grid_step_deg") is not None:
            grid_step = float(ctx.params["grid_step_deg"])
        else:
            angle_min = float(metadata.get("angle_min_deg") if metadata.get("angle_min_deg") is not None else -89.0)
            angle_max = float(metadata.get("angle_max_deg") if metadata.get("angle_max_deg") is not None else 89.0)
            grid_step = (angle_max - angle_min) / float(max(int(_param(ctx.params, "grid_size", 721)) - 1, 1))
        estimates, grid, spectra = _estimate_aoa_grid(problem["snapshots"], metadata, grid_step, method="music")
        return _write_aoa_estimate(
            ctx,
            estimates,
            grid,
            spectra,
            metadata,
            "music",
            problem.get("angles_deg"),
        )


class AoaEstimatorAdapterOperation(Operation):
    id = "model.aoa_estimator_adapter"
    name = "AoA estimator adapter"
    input_kinds = {"problem": ["ai_phy.aoa_problem.numpy", "ai_phy.aoa_observation.numpy"]}
    output_kinds = {"estimate": "ai_phy.aoa_estimate.numpy"}
    differentiability = {
        "framework": "torch",
        "gradient": "surrogate",
        "trainable_params": True,
        "exportable": True,
        "reason": "Surrogate-gradient metadata describes this adapter only when retained as downstream support. Portable replacement is not available until a trained-artifact ABI and runtime binding are implemented.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch", "sionna"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "music_spatial_spectrum_reference", "status": "implemented", "parameter_bindings": {"mode": "music"}},
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "bartlett_spatial_spectrum_reference", "status": "implemented", "parameter_bindings": {"mode": "bartlett_reference"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "music_spatial_spectrum_reference", "status": "implemented", "parameter_bindings": {"mode": "music"}},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "bartlett_spatial_spectrum_reference", "status": "implemented", "parameter_bindings": {"mode": "bartlett_reference"}},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_aoa_estimator_endpoint", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_rt_aoa_training_endpoint", "status": "planned"},
    ]
    formats = {"artifact": "npz", "tensor": "torch.Tensor"}
    params_schema = object_schema(
        {
            "mode": {
                "type": "string",
                "default": "bartlett_reference",
                "enum": ["music", "bartlett_reference"],
                "title": "Reference method",
                "description": "Runnable AoA method used until a trained estimator artifact is bound to this research slot.",
            },
            "grid_step_deg": {"type": "number", "default": 0.25, "minimum": 0.05, "maximum": 5.0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        problem, metadata = _load_all_npz(ctx.require_input("problem").path)
        grid_step = float(_param(ctx.params, "grid_step_deg", 0.25))
        mode = str(ctx.params.get("mode") or "bartlett_reference")
        if mode == "music":
            method = "music"
        elif mode == "bartlett_reference":
            method = "bartlett"
        else:
            raise OperationError("Unsupported AoA-estimator adapter mode `%s`" % mode)
        estimates, grid, spectra = _estimate_aoa_grid(problem["snapshots"], metadata, grid_step, method=method)
        return _write_aoa_estimate(
            ctx,
            estimates,
            grid,
            spectra,
            metadata,
            mode,
            problem.get("angles_deg"),
        )


class AoaEstimationMetricsOperation(Operation):
    id = "metrics.aoa_estimation"
    name = "AoA-estimation metrics"
    input_kinds = {
        "problem": ["ai_phy.aoa_problem.numpy", "ai_phy.aoa_truth.numpy"],
        "estimate": ["ai_phy.aoa_estimate.numpy"],
    }
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        truth, problem_metadata = _load_npz(ctx.require_input("problem").path, "angles_deg")
        estimate, estimate_metadata = _load_npz(ctx.require_input("estimate").path, "angles_deg")
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
        if estimate.shape != truth.shape:
            raise OperationError("AoA estimate shape %s must exactly match truth shape %s" % (estimate.shape, truth.shape))
        errors = np.abs(estimate - truth)
        rmse = float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0
        mae = float(np.mean(errors)) if errors.size else 0.0
        median = float(np.median(errors)) if errors.size else 0.0
        p90 = float(np.percentile(errors, 90)) if errors.size else 0.0
        metrics = {
            "aoa.rmse_deg": rmse,
            "aoa.mae_deg": mae,
            "aoa.median_error_deg": median,
            "aoa.p90_error_deg": p90,
            "localization.aoa_rmse_deg": rmse,
            "channel.snr_db": float(problem_metadata.get("snr_db") or 0.0),
            "task.score": 1.0 / (1.0 + rmse),
        }
        rows = [
            {"id": int(index), "true_angle_deg": float(truth[index]), "estimated_angle_deg": float(estimate[index]), "error_deg": float(errors[index])}
            for index in range(min(16, int(errors.size)))
        ]
        return _write_report(
            ctx,
            "aoa_estimation",
            rows,
            metrics,
            {
                "problem_metadata": problem_metadata,
                "estimate_metadata": estimate_metadata,
                "aoa_preview": _aoa_preview(truth, estimate, problem_metadata),
            },
        )


class ResourceAllocationScenarioSourceOperation(Operation):
    id = "source.resource_allocation_scenario"
    name = "OFDM subcarrier power-allocation scenario source"
    output_kinds = {"problem": "ai_phy.resource_allocation_problem.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "synthetic_ofdm_frequency_response", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "synthetic_ofdm_frequency_response", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "torch_ofdm_resource_state", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "sionna", "implementation": "sionna_sys_resource_state", "status": "planned"},
    ]
    params_schema = object_schema(
        {
            "snapshot_count": {"type": "integer", "default": 32, "minimum": 1},
            "subcarrier_count": {"type": "integer", "default": 16, "minimum": 2},
            "channel_tap_count": {"type": "integer", "default": 4, "minimum": 1},
            "snr_db": {"type": "number", "default": 10.0},
            "total_power": {"type": "number", "default": 1.0, "minimum": 0.0},
            "seed": {"type": "integer", "default": 0, "minimum": 0},
        }
    )

    def run(self, ctx: OperationContext) -> OperationResult:
        snapshots = int(_param(ctx.params, "snapshot_count", 32))
        subcarriers = int(_param(ctx.params, "subcarrier_count", 16))
        tap_count = int(_param(ctx.params, "channel_tap_count", 4))
        snr_db = float(_param(ctx.params, "snr_db", 10.0))
        total_power = float(_param(ctx.params, "total_power", 1.0))
        noise_variance = _noise_variance_from_snr(snr_db)
        seed = ctx.seed("resource_allocation")
        rng = np.random.RandomState(seed)
        power_delay_profile = np.exp(-np.arange(tap_count, dtype=np.float64))
        power_delay_profile /= np.sum(power_delay_profile)
        channel_taps = _complex_normal(rng, (snapshots, tap_count)) * np.sqrt(power_delay_profile)[None, :]
        frequency_response = np.fft.fft(channel_taps, n=subcarriers, axis=1)
        gains = np.maximum(np.abs(frequency_response) ** 2, 1e-12).astype(np.float32)
        metadata = {
            "dataset": "synthetic_ofdm_power_allocation",
            "split": "fixed_seed",
            "scenario_kind": "ofdm_frequency_selective_parallel_channels",
            "snapshot_count": snapshots,
            "snapshot_axis": "ofdm_channel_state",
            "subcarrier_count": subcarriers,
            "allocation_axis": "subcarrier",
            "allocation_granularity": "per_subcarrier_per_channel_snapshot",
            "channel_tap_count": tap_count,
            "channel_gain_definition": "abs_frequency_response_squared",
            "snr_db": snr_db,
            "snr_definition": "unit_tx_power_unit_mean_gain_over_noise_variance",
            "noise_variance": noise_variance,
            "seed": int(seed),
            "total_power": total_power,
            "power_budget_unit": "total_power_per_ofdm_channel_snapshot",
        }
        path = ctx.output_path("problem", ".npz")
        np.savez_compressed(
            path,
            gains=gains,
            channel_taps=channel_taps.astype(np.complex64),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"problem": artifact("ai_phy.resource_allocation_problem.numpy", path, metadata)},
            metrics={
                "channel.snr_db": snr_db,
                "channel.noise_variance": noise_variance,
                "ai_phy.snapshot_count": snapshots,
                "ai_phy.parallel_channel_count": subcarriers,
            },
            metadata=metadata,
        )


class EqualPowerAllocationOperation(Operation):
    id = "model.equal_power_allocator"
    name = "Equal-power allocation baseline"
    input_kinds = {"problem": ["ai_phy.resource_allocation_problem.numpy"]}
    output_kinds = {"decision": "ai_phy.resource_allocation_decision.numpy"}
    differentiability = {"framework": "numpy", "gradient": "none", "trainable_params": False, "exportable": False}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        gains, metadata = _load_npz(ctx.require_input("problem").path, "gains")
        total_power = max(0.0, float(metadata.get("total_power") or 0.0))
        power = np.full_like(gains, total_power / float(gains.shape[1]), dtype=np.float32)
        return _write_power_decision(ctx, power, metadata, "equal_power")


class WaterFillingPowerAllocationOperation(Operation):
    id = "model.water_filling_power_allocator"
    name = "Theoretical water-filling power allocator"
    input_kinds = {"problem": ["ai_phy.resource_allocation_problem.numpy"]}
    output_kinds = {"decision": "ai_phy.resource_allocation_decision.numpy"}
    differentiability = {
        "framework": "numpy",
        "gradient": "none",
        "trainable_params": False,
        "exportable": False,
        "reason": "Closed-form capacity-optimal oracle for parallel Gaussian fading channels under a fixed sum-power constraint.",
    }
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": []}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "closed_form_parallel_channel_water_filling", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "closed_form_parallel_channel_water_filling_labels", "status": "implemented"},
    ]
    equivalence = {
        "type": "numerical",
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "reason": "Allocations must satisfy the fixed sum-power constraint and the KKT water-level solution.",
    }
    formats = {"artifact": "npz", "tensor": "none"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        gains = np.maximum(np.asarray(data["gains"], dtype=np.float64), 1e-12)
        total_power = max(0.0, float(metadata.get("total_power") or 0.0))
        noise_variance = float(metadata.get("noise_variance") or _noise_variance_from_snr(float(metadata.get("snr_db") or 0.0)))
        power = np.zeros_like(gains, dtype=np.float64)
        water_levels = np.zeros((gains.shape[0],), dtype=np.float64)
        for index, sample_gains in enumerate(gains):
            sample_power, water_level = _water_filling_allocation(sample_gains, noise_variance, total_power)
            power[index] = sample_power
            water_levels[index] = water_level
        active_threshold = max(1e-12, (total_power / float(max(gains.shape[1], 1))) * 1e-9)
        active_mask = power > active_threshold
        out_metadata = dict(metadata)
        out_metadata.update(
            {
                "policy": "theoretical_water_filling",
                "objective": "maximize_parallel_channel_sum_rate",
                "constraint": "fixed_sum_power",
                "mean_active_resource_element_fraction": float(np.mean(active_mask)),
                "active_power_threshold": active_threshold,
                "max_power_constraint_error": float(np.max(np.abs(np.sum(power, axis=1) - total_power))),
            }
        )
        path = ctx.output_path("decision", ".npz")
        np.savez_compressed(
            path,
            power=power.astype(np.float32),
            water_level=water_levels.astype(np.float32),
            active_mask=active_mask.astype(np.uint8),
            metadata_json=json.dumps(out_metadata, sort_keys=True),
        )
        return OperationResult(
            outputs={"decision": artifact("ai_phy.resource_allocation_decision.numpy", path, out_metadata)},
            metrics={
                "resource.water_filling.active_resource_element_fraction": float(np.mean(active_mask)),
                "resource.power_constraint.max_abs_error": float(np.max(np.abs(np.sum(power, axis=1) - total_power))),
            },
            metadata=out_metadata,
        )


class ResourceAllocationAdapterOperation(Operation):
    id = "model.resource_allocation_adapter"
    name = "Resource allocation adapter"
    input_kinds = {"problem": ["ai_phy.resource_allocation_problem.numpy"]}
    output_kinds = {"decision": "ai_phy.resource_allocation_decision.numpy"}
    differentiability = {"framework": "torch", "gradient": "surrogate", "trainable_params": True, "exportable": True, "reason": "Surrogate-gradient metadata describes this adapter only when retained as downstream support. Portable replacement is not available until a trained-artifact ABI and runtime binding are implemented."}
    backends = {"benchmark_run": ["numpy"], "dataset_capture": ["numpy"], "differentiable_export": ["torch"]}
    materializations = [
        {"runner": "benchmark_run", "backend": "numpy", "implementation": "weighted_gain_softmax_reference", "status": "implemented"},
        {"runner": "dataset_capture", "backend": "numpy", "implementation": "weighted_gain_softmax_reference", "status": "implemented"},
        {"runner": "differentiable_export", "backend": "torch", "implementation": "trainable_resource_policy_endpoint", "status": "implemented"},
    ]
    params_schema = object_schema({"temperature": {"type": "number", "default": 1.0, "minimum": 0.01}})

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        logits = np.log(np.maximum(data["gains"], 1e-12)) / float(_param(ctx.params, "temperature", 1.0))
        logits = logits - np.max(logits, axis=1, keepdims=True)
        power = np.exp(logits)
        total_power = max(0.0, float(metadata.get("total_power") or 0.0))
        power = total_power * power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-12)
        return _write_power_decision(ctx, power.astype(np.float32), metadata, "resource_allocation_adapter")


class ResourceAllocationMetricsOperation(Operation):
    id = "metrics.resource_allocation"
    name = "Resource allocation metrics"
    input_kinds = {"problem": ["ai_phy.resource_allocation_problem.numpy"], "decision": ["ai_phy.resource_allocation_decision.numpy"]}
    output_kinds = {"report": "metrics.report"}
    params_schema = object_schema()

    def run(self, ctx: OperationContext) -> OperationResult:
        data, metadata = _load_all_npz(ctx.require_input("problem").path)
        decision, decision_metadata = _load_all_npz(ctx.require_input("decision").path)
        power = decision["power"]
        noise_variance = float(metadata.get("noise_variance") or _noise_variance_from_snr(float(metadata.get("snr_db") or 0.0)))
        rates = np.log2(1.0 + data["gains"] * power / max(noise_variance, 1e-12))
        sum_rate = np.sum(rates, axis=1)
        spectral_efficiency = np.mean(rates, axis=1)
        subcarrier_rates = np.mean(rates, axis=0)
        fairness = (float(np.sum(subcarrier_rates)) ** 2) / max(float(len(subcarrier_rates)) * float(np.sum(subcarrier_rates ** 2)), 1e-12)
        total_power = float(metadata.get("total_power") or 0.0)
        power_error = np.abs(np.sum(power, axis=1) - total_power)
        power_scale = max(abs(total_power), 1e-12)
        active_threshold = max(1e-12, (total_power / float(max(power.shape[1], 1))) * 1e-9)
        active_mask = power > active_threshold
        metrics = {
            "resource.theoretical_shannon_sum_bits_per_ofdm_symbol": float(np.mean(sum_rate)),
            "resource.theoretical_shannon_spectral_efficiency_bps_hz": float(np.mean(spectral_efficiency)),
            "resource.theoretical_min_subcarrier_spectral_efficiency_bps_hz": float(np.min(subcarrier_rates)),
            "resource.fairness_jain": fairness,
            "resource.power_constraint.max_abs_error": float(np.max(power_error)),
            "resource.power_constraint.max_relative_error": float(np.max(power_error) / power_scale),
            "resource.active_resource_element_fraction": float(np.mean(active_mask)),
            "channel.snr_db": float(metadata.get("snr_db") or 0.0),
            "channel.noise_variance": noise_variance,
            "task.score": float(np.mean(spectral_efficiency)),
        }
        preview = _resource_allocation_preview(
            data["gains"],
            power,
            noise_variance,
            metadata,
            decision_metadata,
            decision.get("water_level"),
        )
        rows = [
            {
                "snapshot": int(i),
                "theoretical_shannon_sum_bits_per_ofdm_symbol": float(sum_rate[i]),
                "theoretical_shannon_spectral_efficiency_bps_hz": float(spectral_efficiency[i]),
                "active_subcarrier_count": int(np.count_nonzero(active_mask[i])),
            }
            for i in range(min(8, len(sum_rate)))
        ]
        return _write_report(
            ctx,
            "resource_allocation",
            rows,
            metrics,
            {
                "problem_metadata": metadata,
                "decision_metadata": decision_metadata,
                "resource_allocation_preview": preview,
                "metric_scope": "parallel_gaussian_channel_oracle_not_achieved_modulated_payload_throughput",
                "rate_formula": "log2(1 + |h[k]|^2 * p[k] / noise_variance)",
                "active_power_threshold": active_threshold,
            },
        )


def _resource_allocation_preview(
    gains: np.ndarray,
    power: np.ndarray,
    noise_variance: float,
    problem_metadata: JsonDict,
    decision_metadata: JsonDict,
    water_levels: Optional[np.ndarray] = None,
) -> JsonDict:
    gains = np.maximum(np.asarray(gains, dtype=np.float64), 1e-12)
    power = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    noise_variance = max(float(noise_variance), 1e-12)
    snapshot_count = min(int(gains.shape[0]), 32)
    levels = None if water_levels is None else np.asarray(water_levels, dtype=np.float64).reshape(-1)
    snapshots = []
    for index in range(snapshot_count):
        unit_snr_linear = gains[index] / noise_variance
        inverse_unit_snr = noise_variance / gains[index]
        row = {
            "index": index,
            "channel_gain": gains[index].tolist(),
            "unit_power_snr_db": (10.0 * np.log10(np.maximum(unit_snr_linear, 1e-12))).tolist(),
            "inverse_unit_snr": inverse_unit_snr.tolist(),
            "allocated_power": power[index].tolist(),
        }
        if levels is not None and index < int(levels.size):
            row["water_level"] = float(levels[index])
        snapshots.append(row)
    return {
        "schema_version": 1,
        "scenario_kind": str(problem_metadata.get("scenario_kind") or "parallel_gaussian_channels"),
        "allocation_axis": str(problem_metadata.get("allocation_axis") or "parallel_channel"),
        "allocation_granularity": str(problem_metadata.get("allocation_granularity") or "per_channel_per_snapshot"),
        "snapshot_axis": str(problem_metadata.get("snapshot_axis") or "channel_state"),
        "snapshot_count": int(gains.shape[0]),
        "preview_snapshot_count": snapshot_count,
        "channel_count": int(gains.shape[1]),
        "noise_variance": noise_variance,
        "reference_snr_db": float(problem_metadata.get("snr_db") or 0.0),
        "total_power": float(problem_metadata.get("total_power") or 0.0),
        "policy": str(decision_metadata.get("policy") or "power_allocator"),
        "formula": "allocated_power=max(water_level-noise_variance/channel_gain,0)",
        "snapshots": snapshots,
    }


def _param(params: JsonDict, name: str, default: Any) -> Any:
    value = params.get(name)
    return default if value is None else value


def _least_squares_channel_estimate(problem: Dict[str, np.ndarray]) -> np.ndarray:
    pilot_estimates = _sparse_pilot_ls_channel_estimate(problem)
    if "pilots" not in problem and "pilot_mask" not in problem:
        return pilot_estimates
    mask = np.asarray(problem.get("pilot_mask"), dtype=bool)
    return _interpolate_channel_from_pilots(pilot_estimates, mask)


def _sparse_pilot_ls_channel_estimate(
    problem: Dict[str, np.ndarray],
) -> np.ndarray:
    if "observations" not in problem:
        raise OperationError("Channel-estimation problem is missing `observations`")
    observations = np.asarray(problem["observations"], dtype=np.complex64)
    # Legacy fused problems use unit pilots on every element.
    if "pilots" not in problem and "pilot_mask" not in problem:
        return observations.astype(np.complex64, copy=False)
    if observations.ndim != 4:
        raise OperationError("Pilot observations must have shape [example, rx, tx, subcarrier]")
    pilots = np.asarray(problem.get("pilots"), dtype=np.complex64)
    mask = np.asarray(problem.get("pilot_mask"), dtype=bool)
    expected = (int(observations.shape[2]), int(observations.shape[3]))
    if pilots.shape != expected or mask.shape != expected:
        raise OperationError("Pilot arrays must exactly match observation [tx, subcarrier] shape %s" % (expected,))
    if np.any(np.abs(pilots[mask]) <= 1e-12):
        raise OperationError("Active pilot symbols must be non-zero for least-squares estimation")
    pilot_estimates = np.zeros_like(observations, dtype=np.complex64)
    pilot_estimates[:, :, mask] = observations[:, :, mask] / pilots[mask]
    return pilot_estimates


def _linear_mmse_channel_estimate(
    problem: Dict[str, np.ndarray],
    *,
    noise_variance: float,
    channel_variance: float,
    channel_tap_count: int,
) -> np.ndarray:
    """Bayesian frequency-domain LMMSE using the declared exponential TDL prior."""

    if "pilots" not in problem or "pilot_mask" not in problem:
        observations = np.asarray(problem.get("observations"), dtype=np.complex64)
        return (
            float(channel_variance)
            / max(float(channel_variance) + float(noise_variance), 1e-12)
            * observations
        ).astype(np.complex64)
    observations = np.asarray(problem.get("observations"), dtype=np.complex64)
    pilots = np.asarray(problem.get("pilots"), dtype=np.complex64)
    mask = np.asarray(problem.get("pilot_mask"), dtype=bool)
    if observations.ndim != 4:
        raise OperationError(
            "Pilot observations must have shape [example, rx, tx, subcarrier]"
        )
    expected = (int(observations.shape[2]), int(observations.shape[3]))
    if pilots.shape != expected or mask.shape != expected:
        raise OperationError(
            "Pilot arrays must exactly match observation [tx, subcarrier] shape %s"
            % (expected,)
        )
    subcarrier_count = int(observations.shape[-1])
    tap_count = max(1, min(int(channel_tap_count), subcarrier_count))
    tap_power = np.exp(-np.arange(tap_count, dtype=np.float64))
    tap_power *= float(channel_variance) / float(np.sum(tap_power))
    subcarrier_indices = np.arange(subcarrier_count, dtype=np.float64)
    differences = subcarrier_indices[:, None] - subcarrier_indices[None, :]
    tap_indices = np.arange(tap_count, dtype=np.float64)
    covariance = np.sum(
        tap_power[None, None, :]
        * np.exp(
            -2j
            * np.pi
            * differences[:, :, None]
            * tap_indices[None, None, :]
            / float(subcarrier_count)
        ),
        axis=2,
    )
    output = np.empty_like(observations, dtype=np.complex64)
    for tx_index in range(int(observations.shape[2])):
        pilot_indices = np.flatnonzero(mask[tx_index])
        if pilot_indices.size == 0:
            raise OperationError("Every transmit antenna needs at least one pilot")
        active_pilots = pilots[tx_index, pilot_indices]
        if np.any(np.abs(active_pilots) <= 1e-12):
            raise OperationError("Active pilot symbols must be non-zero for LMMSE estimation")
        covariance_pp = covariance[np.ix_(pilot_indices, pilot_indices)]
        regularized = covariance_pp + float(noise_variance) * np.eye(
            pilot_indices.size, dtype=np.complex128
        )
        cross_covariance = covariance[:, pilot_indices]
        try:
            weights = np.linalg.solve(
                regularized.T,
                cross_covariance.T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise OperationError("LMMSE pilot covariance is singular") from exc
        pilot_ls = (
            observations[:, :, tx_index, pilot_indices]
            / active_pilots[None, None, :]
        )
        output[:, :, tx_index, :] = np.einsum(
            "kp,nrp->nrk",
            weights,
            pilot_ls,
        ).astype(np.complex64)
    if not np.all(np.isfinite(output)):
        raise OperationError("LMMSE channel estimate contains non-finite values")
    return output


def _interpolate_channel_from_pilots(pilot_estimates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.empty_like(pilot_estimates, dtype=np.complex64)
    subcarrier_axis = np.arange(pilot_estimates.shape[3], dtype=np.float64)
    for tx_index in range(pilot_estimates.shape[2]):
        pilot_indices = np.flatnonzero(mask[tx_index])
        if pilot_indices.size == 0:
            raise OperationError("Every transmit antenna needs at least one pilot")
        for example_index in range(pilot_estimates.shape[0]):
            for rx_index in range(pilot_estimates.shape[1]):
                values = pilot_estimates[example_index, rx_index, tx_index, pilot_indices]
                if pilot_indices.size == 1:
                    output[example_index, rx_index, tx_index, :] = values[0]
                    continue
                real = np.interp(subcarrier_axis, pilot_indices, np.real(values))
                imag = np.interp(subcarrier_axis, pilot_indices, np.imag(values))
                output[example_index, rx_index, tx_index, :] = (real + 1j * imag).astype(np.complex64)
    return output


def _complex_normal(rng: np.random.RandomState, shape: Tuple[int, ...]) -> np.ndarray:
    return ((rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)).astype(np.complex64)


def _noise_variance_from_snr(snr_db: float) -> float:
    return float(10.0 ** (-float(snr_db) / 10.0))


def _sionna_mimo_ofdm_channel_realization(
    *,
    example_count: int,
    rx_antennas: int,
    tx_antennas: int,
    subcarriers: int,
    params: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, str]:
    if not _sionna_available():
        raise OperationError(
            'Install optional dependencies with `python -m pip install '
            '"noema-lab[wireless]"` in an installed environment, or '
            "`uv sync --extra wireless` in a source checkout, to generate "
            "3GPP TDL MIMO-OFDM channel realizations"
        )
    try:
        from sionna.phy import config as sionna_config  # type: ignore
        from sionna.phy.channel import GenerateOFDMChannel  # type: ignore
        from sionna.phy.channel.tr38901 import TDL  # type: ignore
        from sionna.phy.ofdm import ResourceGrid  # type: ignore
    except ImportError as exc:
        raise OperationError(
            "The installed Sionna package does not expose 3GPP TDL OFDM "
            "channel generation"
        ) from exc

    fft_size = max(8, int(subcarriers))
    subcarrier_spacing_khz = float(
        params.get("subcarrier_spacing_khz") or 30.0
    )
    carrier_frequency_ghz = float(
        params.get("carrier_frequency_ghz") or 3.5
    )
    delay_spread_ns = float(params.get("delay_spread_ns") or 300.0)
    mobility_kmh = float(
        params.get("mobility_kmh")
        if params.get("mobility_kmh") is not None
        else 30.0
    )
    tdl_model = str(params.get("tdl_model") or "C")
    with _SIONNA_RNG_LOCK:
        try:
            sionna_config.seed = int(seed)
            resource_grid = ResourceGrid(
                num_ofdm_symbols=1,
                fft_size=fft_size,
                subcarrier_spacing=subcarrier_spacing_khz * 1e3,
            )
            speed_mps = mobility_kmh / 3.6
            model = TDL(
                model=tdl_model,
                delay_spread=delay_spread_ns * 1e-9,
                carrier_frequency=carrier_frequency_ghz * 1e9,
                min_speed=speed_mps,
                max_speed=speed_mps,
                num_rx_ant=max(1, int(rx_antennas)),
                num_tx_ant=max(1, int(tx_antennas)),
            )
            generator = GenerateOFDMChannel(
                model,
                resource_grid,
                normalize_channel=bool(
                    params.get("normalize_channel", True)
                ),
            )
            full = (
                generator(max(1, int(example_count)))
                .detach()
                .cpu()
                .numpy()
                .astype(np.complex64)
            )
        except Exception as exc:
            raise OperationError(
                "Sionna 3GPP TDL MIMO-OFDM generation failed: %s" % exc
            ) from exc
    if full.ndim != 7:
        raise OperationError(
            "Sionna TDL output must have seven axes, got shape %s"
            % (tuple(full.shape),)
        )
    # [batch, rx, rx_ant, tx, tx_ant, OFDM symbol, subcarrier]
    selected = full[:, 0, :, 0, :, 0, :]
    expected = (
        max(1, int(example_count)),
        max(1, int(rx_antennas)),
        max(1, int(tx_antennas)),
        fft_size,
    )
    if tuple(selected.shape) != expected:
        raise OperationError(
            "Sionna TDL output resolved to shape %s, expected %s"
            % (tuple(selected.shape), expected)
        )
    if not (
        np.all(np.isfinite(selected.real))
        and np.all(np.isfinite(selected.imag))
    ):
        raise OperationError(
            "Sionna TDL MIMO-OFDM channel contains non-finite values"
        )
    return (
        selected.astype(np.complex64, copy=False),
        "sionna.phy.channel.tr38901.TDL+GenerateOFDMChannel",
    )


def _sionna_available() -> bool:
    version = installed_dependency_version("sionna") or ""
    major = version.split(".", 1)[0]
    return (
        importlib.util.find_spec("sionna") is not None
        and importlib.util.find_spec("torch") is not None
        and major.isdigit()
        and int(major) >= 2
    )


def _add_awgn(values: np.ndarray, noise_var: float, rng: np.random.RandomState, backend: str) -> Tuple[np.ndarray, str]:
    requested = str(backend or "auto").strip().lower()
    if requested not in {"auto", "numpy", "sionna"}:
        raise OperationError(
            "Unknown wireless_backend %r; expected auto, numpy, or sionna"
            % backend
        )
    if requested == "sionna" and not _sionna_available():
        raise OperationError(
            'Install with `python -m pip install "noema-lab[wireless]"` in an '
            "installed environment, or `uv sync --extra wireless` in a source "
            "checkout, to use the Sionna wireless backend"
        )
    if requested == "sionna":
        with _SIONNA_RNG_LOCK:
            try:
                from sionna.phy.channel import AWGN  # type: ignore
                from sionna.phy import config as sionna_config  # type: ignore
                import torch  # type: ignore

                seed = int(rng.randint(0, 2**31 - 1))
                sionna_config.seed = seed
                layer = AWGN()
                y = layer(
                    torch.as_tensor(
                        values.astype(np.complex64),
                        dtype=torch.complex64,
                    ),
                    torch.as_tensor(float(noise_var), dtype=torch.float32),
                )
                output = y.detach().cpu().numpy().astype(np.complex64)
            except Exception as exc:
                raise OperationError(
                    "Sionna PyTorch AWGN execution failed: %s" % exc
                ) from exc
        if not bool(
            np.all(np.isfinite(output.real))
            and np.all(np.isfinite(output.imag))
        ):
            raise OperationError(
                "Sionna PyTorch AWGN returned non-finite values"
            )
        return output, "sionna"
    # ``_complex_normal`` already has unit complex variance (one half per real
    # component), so scaling by sqrt(noise_var) yields E[|n|^2]=noise_var.
    noise = np.sqrt(noise_var) * _complex_normal(rng, values.shape)
    return (values + noise).astype(np.complex64), "numpy"


def _zf_spectral_efficiency(
    h_true: np.ndarray,
    h_hat: np.ndarray,
    *,
    noise_variance: float,
) -> Tuple[float, float]:
    """Evaluate estimated-CSI and perfect-CSI linear ZF on the true channel."""

    truth = np.asarray(h_true, dtype=np.complex128)
    estimate = np.asarray(h_hat, dtype=np.complex128)
    if truth.ndim != 4 or estimate.shape != truth.shape:
        return 0.0, 0.0
    noise = max(float(noise_variance), 1e-12)
    true_matrices = truth.transpose(0, 3, 1, 2).reshape(
        -1, truth.shape[1], truth.shape[2]
    )
    estimated_matrices = estimate.transpose(0, 3, 1, 2).reshape(
        -1, estimate.shape[1], estimate.shape[2]
    )

    def rate(receiver_channels: np.ndarray) -> float:
        rates = []
        for actual, receiver_channel in zip(
            true_matrices, receiver_channels
        ):
            equalizer = np.linalg.pinv(receiver_channel, rcond=1e-5)
            effective = equalizer @ actual
            noise_enhancement = np.real(
                np.diag(equalizer @ np.conjugate(equalizer.T))
            )
            desired = np.abs(np.diag(effective)) ** 2
            interference = (
                np.sum(np.abs(effective) ** 2, axis=1) - desired
            )
            sinr = desired / np.maximum(
                interference + noise * noise_enhancement,
                1e-12,
            )
            rates.append(
                float(
                    np.sum(
                        np.log2(
                            1.0
                            + np.minimum(
                                np.maximum(sinr, 0.0),
                                1e12,
                            )
                        )
                    )
                )
            )
        return float(np.mean(rates)) if rates else 0.0

    return rate(estimated_matrices), rate(true_matrices)


def _channel_estimation_preview(
    h_true: np.ndarray,
    h_hat: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    estimate_metadata: Mapping[str, Any],
) -> JsonDict:
    truth = np.asarray(h_true, dtype=np.complex64)
    estimate = np.asarray(h_hat, dtype=np.complex64)
    if truth.ndim != 4 or estimate.shape != truth.shape or not truth.size:
        return {}
    sample_truth = truth[0]
    sample_estimate = estimate[0]
    rows_true = sample_truth.reshape(
        int(sample_truth.shape[0] * sample_truth.shape[1]),
        int(sample_truth.shape[2]),
    )
    rows_estimate = sample_estimate.reshape(rows_true.shape)
    phase_error = np.angle(
        rows_estimate * np.conjugate(rows_true)
    ).astype(np.float32)
    pilot_spacing = max(1, int(metadata.get("pilot_spacing") or 1))
    tx_antennas = int(sample_truth.shape[1])
    pilot_mask = np.zeros((tx_antennas, sample_truth.shape[2]), dtype=np.uint8)
    for tx_index in range(tx_antennas):
        pilot_mask[tx_index, tx_index::pilot_spacing] = 1
    repeated_mask = np.tile(
        pilot_mask,
        (int(sample_truth.shape[0]), 1),
    )
    return {
        "schema_version": 1,
        "sample_index": 0,
        "layout": "row_is_rx_tx_link,column_is_subcarrier",
        "link_labels": [
            "RX%d–TX%d" % (rx_index, tx_index)
            for rx_index in range(int(sample_truth.shape[0]))
            for tx_index in range(int(sample_truth.shape[1]))
        ],
        "subcarrier_indices": list(range(int(sample_truth.shape[2]))),
        "pilot_mask": repeated_mask.tolist(),
        "true_magnitude": np.abs(rows_true).astype(np.float32).tolist(),
        "estimated_magnitude": np.abs(rows_estimate).astype(
            np.float32
        ).tolist(),
        "absolute_error": np.abs(rows_estimate - rows_true).astype(
            np.float32
        ).tolist(),
        "true_phase_rad": np.angle(rows_true).astype(np.float32).tolist(),
        "estimated_phase_rad": np.angle(rows_estimate).astype(
            np.float32
        ).tolist(),
        "phase_error_rad": phase_error.tolist(),
        "estimator_mode": str(
            estimate_metadata.get("adapter_mode")
            or estimate_metadata.get("estimator")
            or "channel_estimator"
        ),
        "estimator_label": str(
            estimate_metadata.get("estimator_label")
            or "Channel estimator"
        ),
        "snr_db": float(metadata.get("snr_db") or 0.0),
        "pilot_density": float(metadata.get("pilot_density") or 0.0),
    }


def _load_all_npz(path) -> Tuple[Dict[str, np.ndarray], JsonDict]:
    with np.load(str(path), allow_pickle=False) as payload:
        metadata = (
            decode_strict_json_object(
                str(payload["metadata_json"]),
                label="AI-PHY artifact metadata_json",
            )
            if "metadata_json" in payload
            else {}
        )
        arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
    return arrays, metadata


def _load_npz(path, key: str) -> Tuple[np.ndarray, JsonDict]:
    arrays, metadata = _load_all_npz(path)
    if key not in arrays:
        raise OperationError("NPZ artifact %s is missing array `%s`" % (path, key))
    return arrays[key], metadata


def _write_report(ctx: OperationContext, family: str, rows, metrics: JsonDict, metadata: JsonDict) -> OperationResult:
    report = {"schema_version": 1, "metric_family": family, "rows": rows, "metrics": metrics, "metadata": metadata}
    path = ctx.output_path("report", ".json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return OperationResult(outputs={"report": artifact("metrics.report", path, report)}, metrics=metrics, metadata=report)


def _dft_codebook(size: int) -> np.ndarray:
    n = np.arange(size)
    codebook = []
    for k in range(size):
        codebook.append(np.exp(1j * 2.0 * np.pi * k * n / float(size)) / np.sqrt(float(size)))
    return np.asarray(codebook, dtype=np.complex64)


def _mrt_weights(channels: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(channels, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return channels / norms


def _validated_beamforming_weights(
    channels: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, float]:
    channels = np.asarray(channels)
    weights = np.asarray(weights)
    if channels.ndim != 2:
        raise OperationError(
            "beamforming problem channels must have shape [examples, tx_antennas]"
        )
    if weights.shape != channels.shape:
        raise OperationError(
            "beamforming decision weights shape %s does not match channel shape %s"
            % (tuple(weights.shape), tuple(channels.shape))
        )
    if not np.all(np.isfinite(channels)):
        raise OperationError("beamforming problem contains non-finite channel values")
    if not np.all(np.isfinite(weights)):
        raise OperationError("beamforming decision contains non-finite weights")
    norms = np.linalg.norm(weights, axis=1)
    if not np.all(np.isfinite(norms)):
        raise OperationError("beamforming decision has non-finite weight norms")
    norm_error = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    if norm_error > 1e-5:
        raise OperationError(
            "beamforming decision weights must be unit norm per example; "
            "maximum norm error is %.6g" % norm_error
        )
    return weights.astype(np.complex64, copy=False), norm_error


def _write_beam_decision(ctx: OperationContext, weights: np.ndarray, metadata: JsonDict, mode: str, label: str) -> OperationResult:
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise OperationError(
            "beamforming decision weights must have shape [examples, tx_antennas]"
        )
    expected_examples = int(metadata.get("example_count") or weights.shape[0])
    expected_antennas = int(metadata.get("tx_antennas") or weights.shape[1])
    reference_shape = np.zeros((expected_examples, expected_antennas), dtype=np.complex64)
    weights, norm_error = _validated_beamforming_weights(reference_shape, weights)
    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "beamformer": mode,
            "beamformer_label": label,
            "unit_norm_validated": True,
            "unit_norm_max_error": norm_error,
        }
    )
    path = ctx.output_path("decision", ".npz")
    np.savez_compressed(path, weights=weights.astype(np.complex64), metadata_json=json.dumps(out_metadata, sort_keys=True))
    return OperationResult(outputs={"decision": artifact("ai_phy.beamforming_decision.numpy", path, out_metadata)}, metadata=out_metadata)


def _linear_trilateration(anchors: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    a0 = anchors[0]
    rows = []
    for anchor in anchors[1:]:
        rows.append(2.0 * (anchor - a0))
    A = np.asarray(rows, dtype=np.float32)
    estimates = []
    for sample_ranges in ranges:
        b = []
        r0 = float(sample_ranges[0])
        for idx, anchor in enumerate(anchors[1:], start=1):
            ri = float(sample_ranges[idx])
            b.append(r0 ** 2 - ri ** 2 + float(np.dot(anchor, anchor) - np.dot(a0, a0)))
        x, *_ = np.linalg.lstsq(A, np.asarray(b, dtype=np.float32), rcond=None)
        estimates.append(x.astype(np.float32))
    return np.asarray(estimates, dtype=np.float32)


def _localization_preview(
    problem: Dict[str, np.ndarray],
    estimated_positions: np.ndarray,
) -> JsonDict:
    anchors = np.asarray(problem.get("anchors", []), dtype=np.float64)
    truth = np.asarray(problem.get("positions", []), dtype=np.float64)
    estimates = np.asarray(estimated_positions, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1:] != (2,):
        return {}
    if truth.ndim != 2 or truth.shape[1:] != (2,) or estimates.shape != truth.shape:
        return {}
    count = min(RESULT_PREVIEW_LIMIT, int(truth.shape[0]))
    errors = np.linalg.norm(estimates[:count] - truth[:count], axis=1)
    return {
        "coordinate_system": "cartesian_xy_metres",
        "anchors": anchors.tolist(),
        "true_positions": truth[:count].tolist(),
        "estimated_positions": estimates[:count].tolist(),
        "errors_m": errors.tolist(),
        "shown_example_count": count,
        "total_example_count": int(truth.shape[0]),
    }


def _write_position_estimate(
    ctx: OperationContext,
    positions: np.ndarray,
    metadata: JsonDict,
    estimator: str,
    problem: Optional[Dict[str, np.ndarray]] = None,
) -> OperationResult:
    out_metadata = dict(metadata)
    out_metadata.update({"estimator": estimator})
    if problem is not None:
        preview = _localization_preview(problem, positions)
        if preview:
            out_metadata["localization_preview"] = preview
    path = ctx.output_path("estimate", ".npz")
    np.savez_compressed(path, positions=positions.astype(np.float32), metadata_json=json.dumps(out_metadata, sort_keys=True))
    return OperationResult(outputs={"estimate": artifact("ai_phy.localization_estimate.numpy", path, out_metadata)}, metadata=out_metadata)


def _square_perimeter_anchors(area_m: float, anchor_count: int) -> np.ndarray:
    # Evenly space anchors around the square perimeter.  Four anchors resolve
    # exactly to the familiar corner-anchor benchmark.
    perimeter_positions = np.linspace(0.0, 4.0 * area_m, anchor_count, endpoint=False)
    anchors = []
    for distance in perimeter_positions:
        if distance < area_m:
            anchors.append((distance, 0.0))
        elif distance < 2.0 * area_m:
            anchors.append((area_m, distance - area_m))
        elif distance < 3.0 * area_m:
            anchors.append((3.0 * area_m - distance, area_m))
        else:
            anchors.append((0.0, 4.0 * area_m - distance))
    return np.asarray(anchors, dtype=np.float32)


def _ula_steering_vectors(angles_deg: np.ndarray, antenna_count: int, spacing_wavelengths: float) -> np.ndarray:
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64).reshape(-1))
    antenna_indices = np.arange(antenna_count, dtype=np.float64)
    phase = 2.0 * np.pi * float(spacing_wavelengths) * np.sin(angles)[:, None] * antenna_indices[None, :]
    return np.exp(1j * phase).astype(np.complex64)


def _estimate_aoa_grid(
    snapshots: np.ndarray,
    metadata: JsonDict,
    grid_step_deg: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    snapshots = np.asarray(snapshots, dtype=np.complex64)
    if snapshots.ndim != 3:
        raise OperationError("AoA snapshots must have shape [example, antenna, snapshot]")
    if snapshots.shape[1] < 2 or snapshots.shape[2] < 2:
        raise OperationError("AoA estimation requires at least two antennas and two snapshots")
    angle_min = float(metadata.get("angle_min_deg") if metadata.get("angle_min_deg") is not None else -89.0)
    angle_max = float(metadata.get("angle_max_deg") if metadata.get("angle_max_deg") is not None else 89.0)
    grid = np.arange(angle_min, angle_max + 0.5 * grid_step_deg, grid_step_deg, dtype=np.float64)
    spacing = float(metadata.get("element_spacing_wavelengths") or 0.5)
    steering = _ula_steering_vectors(grid, int(snapshots.shape[1]), spacing).astype(np.complex128)
    estimates = np.empty((snapshots.shape[0],), dtype=np.float32)
    spectra = np.empty((snapshots.shape[0], grid.size), dtype=np.float32)
    for index, sample in enumerate(snapshots):
        sample128 = sample.astype(np.complex128)
        covariance = sample128 @ np.conjugate(sample128).T / float(sample.shape[1])
        if method == "music":
            _, eigenvectors = np.linalg.eigh(covariance)
            noise_subspace = eigenvectors[:, :-1]
            response = np.conjugate(steering) @ noise_subspace
            spectrum = 1.0 / np.maximum(np.sum(np.abs(response) ** 2, axis=1), 1e-12)
        elif method == "bartlett":
            spectrum = np.real(np.einsum("gm,mn,gn->g", np.conjugate(steering), covariance, steering))
        else:
            raise OperationError("Unknown AoA grid estimator method `%s`" % method)
        spectrum = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
        spectrum /= max(float(np.max(spectrum)), 1e-12)
        spectra[index] = spectrum.astype(np.float32)
        estimates[index] = float(grid[int(np.argmax(spectrum))])
    return estimates, grid.astype(np.float32), spectra


def _write_aoa_estimate(
    ctx: OperationContext,
    angles_deg: np.ndarray,
    grid_deg: np.ndarray,
    spectra: np.ndarray,
    metadata: JsonDict,
    estimator: str,
    true_angles_deg: Optional[np.ndarray] = None,
) -> OperationResult:
    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "estimator": estimator,
            "grid_step_deg": float(grid_deg[1] - grid_deg[0]) if grid_deg.size > 1 else 0.0,
            "grid_point_count": int(grid_deg.size),
        }
    )
    if true_angles_deg is not None:
        preview = _aoa_preview(true_angles_deg, angles_deg, metadata)
        if preview:
            out_metadata["aoa_preview"] = preview
    path = ctx.output_path("estimate", ".npz")
    np.savez_compressed(
        path,
        angles_deg=np.asarray(angles_deg, dtype=np.float32),
        grid_deg=np.asarray(grid_deg, dtype=np.float32),
        spatial_spectrum=np.asarray(spectra, dtype=np.float32),
        metadata_json=json.dumps(out_metadata, sort_keys=True),
    )
    return OperationResult(outputs={"estimate": artifact("ai_phy.aoa_estimate.numpy", path, out_metadata)}, metadata=out_metadata)


def _aoa_preview(
    true_angles_deg: np.ndarray,
    estimated_angles_deg: np.ndarray,
    metadata: JsonDict,
) -> JsonDict:
    truth = np.asarray(true_angles_deg, dtype=np.float64).reshape(-1)
    estimates = np.asarray(estimated_angles_deg, dtype=np.float64).reshape(-1)
    if estimates.shape != truth.shape:
        return {}
    count = min(RESULT_PREVIEW_LIMIT, int(truth.size))
    return {
        "angle_convention": str(metadata.get("angle_convention") or "broadside_azimuth_degrees"),
        "true_angles_deg": truth[:count].tolist(),
        "estimated_angles_deg": estimates[:count].tolist(),
        "errors_deg": np.abs(estimates[:count] - truth[:count]).tolist(),
        "antenna_count": int(metadata.get("antenna_count") or 0),
        "element_spacing_wavelengths": float(metadata.get("element_spacing_wavelengths") or 0.5),
        "shown_example_count": count,
        "total_example_count": int(truth.size),
    }


def _write_power_decision(ctx: OperationContext, power: np.ndarray, metadata: JsonDict, policy: str) -> OperationResult:
    out_metadata = dict(metadata)
    out_metadata.update({"policy": policy})
    path = ctx.output_path("decision", ".npz")
    np.savez_compressed(path, power=power.astype(np.float32), metadata_json=json.dumps(out_metadata, sort_keys=True))
    return OperationResult(outputs={"decision": artifact("ai_phy.resource_allocation_decision.numpy", path, out_metadata)}, metadata=out_metadata)


def _water_filling_allocation(gains: np.ndarray, noise_variance: float, total_power: float) -> Tuple[np.ndarray, float]:
    gains = np.maximum(np.asarray(gains, dtype=np.float64).reshape(-1), 1e-12)
    total_power = max(0.0, float(total_power))
    inverse_quality = max(float(noise_variance), 1e-12) / gains
    if total_power == 0.0:
        return np.zeros_like(gains, dtype=np.float64), float(np.min(inverse_quality))
    ordered = np.sort(inverse_quality)
    water_level = float(ordered[0] + total_power)
    for active_count in range(1, int(ordered.size) + 1):
        candidate = float((total_power + np.sum(ordered[:active_count])) / active_count)
        if active_count == int(ordered.size) or candidate <= float(ordered[active_count]):
            water_level = candidate
            break
    power = np.maximum(water_level - inverse_quality, 0.0)
    allocated = float(np.sum(power))
    if allocated > 0.0:
        power *= total_power / allocated
    return power.astype(np.float64, copy=False), water_level


def _db(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-12))
