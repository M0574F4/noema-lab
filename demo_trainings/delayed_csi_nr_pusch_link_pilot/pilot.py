#!/usr/bin/env python3
"""Bounded development-only delayed-CSI NR-PUSCH transfer pilot.

This script never modifies the frozen NR-PUSCH publication case.  It runs a
small paired link-level pilot with a continuous TDL-C trajectory, NR LDPC/CRC,
DMRS receiver estimation, and one fixed per-PRB allocation over a PUSCH slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/noema-nr-pusch-link-pilot-mpl")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".noema" / "demos" / "delayed_csi_nr_pusch_link_pilot_v2"
MODEL = (
    ROOT
    / ".noema/training_exports/delayed_csi_codec_aligned_v1/seed_35023/"
    "artifacts/power_policy.onnx"
)

METHODS = (
    "equal_power",
    "stale_csi_box_water_filling",
    "causal_ar_box_water_filling",
    "learned_transfer_seed35023",
    "current_csi_box_water_filling_diagnostic",
)

CONFIG: dict[str, Any] = {
    "study_id": "delayed_csi_nr_pusch_link_pilot_v2",
    "status": "development_only_not_publication_evidence",
    "scope": "normalized_nr_framed_tdl_c_pusch_link_level_pilot",
    "carrier": {
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 15000.0,
        "outer_fft_size": 512,
        "carrier_resource_blocks": 25,
        "pusch_resource_blocks": 10,
        "pusch_subcarriers": 120,
        "pusch_start_resource_block": 7,
    },
    "pusch": {
        "mcs_table": 1,
        "mcs_index": 7,
        "modulation": "QPSK",
        "transport_block_size_bits": 1608,
        "slot_symbols": 14,
        "dmrs_symbols_zero_based": [2, 11],
        "decoder_iterations": 20,
        "ebno_db": 5.0,
    },
    "channel": {
        "model": "TDL-C300",
        "delay_spread_s": 300e-9,
        "maximum_path_delay_s": 2595e-9,
        "mobility_kmh": 120.0,
        "normalize_channel": True,
        "l_min": -6,
        "l_max": 26,
    },
    "transmitter_csi": {
        "interpretation": "fast reciprocal_or_SRS_like_CSI_abstraction",
        "history_length": 4,
        "feedback_delay_ofdm_symbols_to_slot_start": 5,
        "estimation_snr_db": 20.0,
        "current_or_future_forwarded_to_policy": False,
    },
    "allocation": {
        "granularity": "one_power_per_PUSCH_PRB_constant_over_slot",
        "lower_power": 0.5,
        "upper_power": 1.5,
        "mean_power": 1.0,
        "scale_data_and_dmrs": True,
        "paired_exact_waveform_energy_normalization": True,
    },
    "learned_transfer": {
        "source_model": str(MODEL.relative_to(ROOT)),
        "source_initialization_seed": 35023,
        "source_selected_epoch": 47,
        "adapter": "four_adjacent_tones_each_side_then_crop_and_PRB_project",
        "qualification": (
            "out_of_distribution_transfer_from_TDL_C1000_generic_codec_to_"
            "TDL_C300_NR_PUSCH; not the positive 16-tone mechanism model"
        ),
    },
    "smoke": {"seed": 280900001, "trajectories": 2},
    "pilot": {
        "unit_seed_start": 281000001,
        "units": 5,
        "trajectories_per_unit": 8,
        "promising_minimum_delivery_gain": 4,
        "equal_bler_interval": [0.2, 0.8],
        "minimum_gain_correlation": 0.5,
    },
    "resource_limits": {
        "maximum_unit_wall_seconds": 180.0,
        "maximum_peak_rss_bytes": 3 * 1024**3,
        "maximum_retained_bytes": 250 * 1024**2,
    },
    "methods": list(METHODS),
}


class PilotError(RuntimeError):
    pass


def _config_sha256() -> str:
    payload = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise PilotError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"required result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotError(f"expected JSON object: {path}")
    return value


def _noise_variance() -> float:
    # Same CP+DMRS-inclusive normalized definition as the frozen anchor.
    energy_per_information_bit = 1800.0 / 1608.0
    return energy_per_information_bit / (10.0 ** (5.0 / 10.0))


def _box_project(values: np.ndarray, lower: float = 0.5, upper: float = 1.5) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    elif x.ndim == 2:
        squeeze = False
    else:
        raise ValueError("box projection expects one or two dimensions")
    target = float(x.shape[1])
    low = np.min(x - upper, axis=1)
    high = np.max(x - lower, axis=1)
    for _ in range(70):
        shift = 0.5 * (low + high)
        projected = np.clip(x - shift[:, None], lower, upper)
        too_much = np.sum(projected, axis=1) > target
        low = np.where(too_much, shift, low)
        high = np.where(too_much, high, shift)
    result = np.clip(x - 0.5 * (low + high)[:, None], lower, upper)
    return result[0] if squeeze else result


def _box_water_filling(gains: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(gains, dtype=np.float64), 1e-12)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    else:
        squeeze = False
    noise = _noise_variance()
    floors = noise / values
    lower_power = float(CONFIG["allocation"]["lower_power"])
    upper_power = float(CONFIG["allocation"]["upper_power"])
    target = float(values.shape[1])
    low = np.min(floors + lower_power, axis=1) - upper_power
    high = np.max(floors + upper_power, axis=1) + target
    for _ in range(70):
        level = 0.5 * (low + high)
        power = np.clip(level[:, None] - floors, lower_power, upper_power)
        too_much = np.sum(power, axis=1) > target
        high = np.where(too_much, level, high)
        low = np.where(too_much, low, level)
    result = np.clip(0.5 * (low + high)[:, None] - floors, lower_power, upper_power)
    return result[0] if squeeze else result


def _simplex_power(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    rank = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = ordered - cumulative / rank > 0.0
    rho = np.maximum(np.sum(active, axis=1) - 1, 0)
    threshold = cumulative[np.arange(values.shape[0]), rho] / (rho + 1.0)
    fractions = np.maximum(values - threshold[:, None], 0.0)
    fractions /= np.maximum(np.sum(fractions, axis=1, keepdims=True), 1e-15)
    return fractions * float(values.shape[1])


def _rb_mean(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    if x.shape[-1] != 120:
        raise ValueError("expected 120 PUSCH subcarriers")
    return np.mean(x.reshape(*x.shape[:-1], 10, 12), axis=-1)


def _frequency_response_from_taps(taps: Any, torch: Any) -> Any:
    outer = torch.zeros((*taps.shape[:-1], 512), dtype=taps.dtype, device=taps.device)
    for tap_index, lag in enumerate(range(-6, 27)):
        outer[..., lag % 512] = taps[..., tap_index]
    return torch.fft.fftshift(torch.fft.fft(outer, dim=-1), dim=-1)


def _history_and_current(h_time: Any, torch: Any, csi_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    history_length = 4
    delay = 5
    nominal_symbol_samples = 512 + 36
    current_start = (history_length - 1 + delay) * nominal_symbol_samples
    history_indices = np.asarray(
        [int((index + 0.5) * nominal_symbol_samples) for index in range(history_length)],
        dtype=np.int64,
    )
    cp = np.asarray((40, 36, 36, 36, 36, 36, 36, 40, 36, 36, 36, 36, 36, 36))
    current_indices: list[int] = []
    cursor = current_start
    for cp_samples in cp:
        current_indices.append(int(cursor + cp_samples + 512 // 2))
        cursor += int(cp_samples) + 512

    taps = h_time[:, 0, 0, 0, 0]
    history_outer = _frequency_response_from_taps(taps[:, history_indices], torch)
    current_outer = _frequency_response_from_taps(taps[:, current_indices], torch)
    history_true = history_outer[..., 186:314].detach().cpu().numpy().astype(np.complex64)
    current = current_outer[..., 190:310].detach().cpu().numpy().astype(np.complex64)
    rng = np.random.default_rng(int(csi_seed))
    variance = float(np.mean(np.abs(history_true) ** 2)) / 100.0
    error = math.sqrt(variance / 2.0) * (
        rng.standard_normal(history_true.shape) + 1j * rng.standard_normal(history_true.shape)
    )
    observed = (history_true + error.astype(np.complex64)).astype(np.complex64)
    newest = observed[:, -1, 4:124]
    current_average_gain = np.mean(np.abs(current) ** 2, axis=1)
    old_gain = np.abs(newest) ** 2
    gain_corr = float(np.corrcoef(old_gain.reshape(-1), current_average_gain.reshape(-1))[0, 1])
    first_current = current[:, 0]
    complex_corr = abs(np.mean(first_current * np.conjugate(newest))) / math.sqrt(
        float(np.mean(np.abs(first_current) ** 2) * np.mean(np.abs(newest) ** 2))
    )
    return observed, current, current_average_gain, {
        "delayed_current_gain_correlation": gain_corr,
        "delayed_first_current_complex_correlation_magnitude": float(complex_corr),
        "csi_error_variance": variance,
        "history_last_sample_index": int(history_indices[-1]),
        "current_slot_start_sample_index": int(current_start),
        "causal_sample_gap": int(current_start - history_indices[-1]),
    }


def _learned_power(history: np.ndarray) -> np.ndarray:
    if not MODEL.is_file():
        raise PilotError(f"required transfer candidate is missing: {MODEL}")
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    history_ri = np.stack((history.real, history.imag), axis=-1).astype(np.float32)
    batch = history_ri.shape[0]
    scores = session.run(
        ["allocation_scores"],
        {
            "csi_history": history_ri,
            "noise_variance": np.full((batch, 1), _noise_variance(), dtype=np.float32),
            "average_power_budget": np.ones((batch, 1), dtype=np.float32),
        },
    )[0]
    tone_power = _simplex_power(np.asarray(scores[:, 4:124], dtype=np.float64))
    return _box_project(_rb_mean(tone_power))


def _allocations(history: np.ndarray, current_average_gain: np.ndarray) -> dict[str, np.ndarray]:
    newest_gain = np.abs(history[:, -1, 4:124]) ** 2
    stale_rb = _rb_mean(newest_gain)
    complex_history = history[..., 4:124]
    previous = complex_history[:, :-1]
    following = complex_history[:, 1:]
    coefficient = np.sum(following * np.conjugate(previous), axis=1) / (
        np.sum(np.abs(previous) ** 2, axis=1) + 1e-12
    )
    coefficient /= np.maximum(np.abs(coefficient), 1.0)
    predicted = complex_history[:, -1] * np.power(coefficient, 11)
    predicted_gain = np.abs(predicted) ** 2
    predicted_gain = 0.5 * predicted_gain + 0.5 * np.mean(predicted_gain, axis=1, keepdims=True)
    return {
        "equal_power": np.ones((history.shape[0], 10), dtype=np.float64),
        "stale_csi_box_water_filling": _box_water_filling(stale_rb),
        "causal_ar_box_water_filling": _box_water_filling(_rb_mean(predicted_gain)),
        "learned_transfer_seed35023": _learned_power(history),
        "current_csi_box_water_filling_diagnostic": _box_water_filling(_rb_mean(current_average_gain)),
    }


def _build_link():
    import torch
    from sionna.phy.channel import ApplyTimeChannel, GenerateTimeChannel
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.nr import (
        CarrierConfig,
        PUSCHConfig,
        PUSCHDMRSConfig,
        PUSCHReceiver,
        PUSCHTransmitter,
        TBConfig,
    )
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    carrier = CarrierConfig(
        n_cell_id=1,
        cyclic_prefix="normal",
        subcarrier_spacing=15,
        n_size_grid=25,
        slot_number=0,
        frame_number=0,
    )
    dmrs = PUSCHDMRSConfig(
        config_type=1,
        type_a_position=2,
        additional_position=1,
        length=1,
        dmrs_port_set=[0],
        num_cdm_groups_without_data=1,
    )
    tb = TBConfig(mcs_index=7, mcs_table=1, channel_type="PUSCH", n_id=1)
    pusch = PUSCHConfig(
        carrier_config=carrier,
        pusch_dmrs_config=dmrs,
        tb_config=tb,
        n_size_bwp=10,
        n_start_bwp=7,
        num_layers=1,
        num_antenna_ports=1,
        mapping_type="A",
        symbol_allocation=[0, 14],
        n_rnti=1,
        precoding="non-codebook",
        transform_precoding=False,
    )
    transmitter = PUSCHTransmitter(pusch, return_bits=False, output_domain="freq")
    receiver = PUSCHReceiver(
        transmitter,
        channel_estimator=None,
        return_tb_crc_status=True,
        input_domain="freq",
    )
    cp = np.asarray((40, 36, 36, 36, 36, 36, 36, 40, 36, 36, 36, 36, 36, 36), dtype=np.int32)
    modulator = OFDMModulator(cp)
    demodulator = OFDMDemodulator(fft_size=512, l_min=-6, cyclic_prefix_length=cp)
    history_samples = (4 - 1 + 5) * (512 + 36)
    total_samples = history_samples + 7680
    channel_model = TDL(
        model="C",
        delay_spread=300e-9,
        carrier_frequency=3.5e9,
        min_speed=120.0 / 3.6,
        max_speed=120.0 / 3.6,
        num_rx_ant=1,
        num_tx_ant=1,
    )
    generator = GenerateTimeChannel(
        channel_model,
        bandwidth=512 * 15000.0,
        num_time_samples=total_samples,
        l_min=-6,
        l_max=26,
        normalize_channel=True,
    )
    apply_channel = ApplyTimeChannel(num_time_samples=7680, l_tot=33)
    return torch, transmitter, receiver, modulator, demodulator, generator, apply_channel, history_samples


def _run_batch(seed: int, trajectories: int) -> dict[str, Any]:
    from sionna.phy import config as sionna_config

    started = time.perf_counter()
    (
        torch,
        transmitter,
        receiver,
        modulator,
        demodulator,
        generator,
        apply_channel,
        current_start,
    ) = _build_link()
    sionna_config.seed = int(seed) + 100000
    h_time = generator(int(trajectories))
    history, current, current_average_gain, correlations = _history_and_current(
        h_time, torch, int(seed) + 200000
    )
    allocation_by_method = _allocations(history, current_average_gain)
    allocations = np.stack([allocation_by_method[name] for name in METHODS], axis=1)
    if not np.all(np.isfinite(allocations)):
        raise PilotError("allocation contains nonfinite values")
    sum_error = float(np.max(np.abs(np.sum(allocations, axis=-1) - 10.0)))
    lower_violation = float(np.max(np.maximum(0.5 - allocations, 0.0)))
    upper_violation = float(np.max(np.maximum(allocations - 1.5, 0.0)))
    if max(sum_error, lower_violation, upper_violation) > 1e-6:
        raise PilotError("allocation constraints failed")

    rng = np.random.default_rng(int(seed) + 300000)
    bits = rng.integers(0, 2, size=(int(trajectories), 1608), dtype=np.uint8)
    expanded_bits = np.repeat(bits[:, None, :], len(METHODS), axis=1).reshape(-1, 1608)
    input_tensor = torch.as_tensor(expanded_bits[:, None, :], dtype=torch.float32)
    pusch_grid = transmitter(input_tensor)
    tone_power = np.repeat(allocations, 12, axis=-1).reshape(-1, 120)
    scale = torch.sqrt(torch.as_tensor(tone_power, dtype=torch.float32)).reshape(-1, 1, 1, 1, 120)
    allocated_grid = pusch_grid * scale
    outer_grid = torch.zeros(
        (expanded_bits.shape[0], 1, 1, 14, 512), dtype=torch.complex64
    )
    outer_grid[..., 190:310] = allocated_grid
    waveform = modulator(outer_grid)
    waveform = waveform.reshape(int(trajectories), len(METHODS), 1, 1, 7680)
    energy_before = torch.sum(torch.abs(waveform) ** 2, dim=(-1, -2, -3))
    reference_energy = energy_before[:, :1]
    normalization = torch.sqrt(reference_energy / torch.clamp(energy_before, min=1e-12))
    waveform = waveform * normalization[..., None, None, None]
    energy_after = torch.sum(torch.abs(waveform) ** 2, dim=(-1, -2, -3))
    waveform_flat = waveform.reshape(-1, 1, 1, 7680)

    h_current = h_time[..., current_start : current_start + 7680 + 32, :]
    h_paired = h_current.repeat_interleave(len(METHODS), dim=0)
    received_clean = apply_channel(waveform_flat, h_paired)
    noise_rng = np.random.default_rng(int(seed) + 400000)
    noise_base = math.sqrt(_noise_variance() / 2.0) * (
        noise_rng.standard_normal((int(trajectories), 1, 1, 7712))
        + 1j * noise_rng.standard_normal((int(trajectories), 1, 1, 7712))
    )
    noise = np.repeat(noise_base[:, None], len(METHODS), axis=1).reshape(-1, 1, 1, 7712)
    received = received_clean + torch.as_tensor(noise, dtype=torch.complex64)
    rx_outer = demodulator(received)
    rx_pusch = rx_outer[..., 190:310]
    decoded, crc = receiver(rx_pusch, _noise_variance())
    decoded_np = decoded.detach().cpu().numpy().reshape(int(trajectories), len(METHODS), 1608)
    crc_np = crc.detach().cpu().numpy().reshape(int(trajectories), len(METHODS)).astype(bool)
    bit_errors = np.count_nonzero(decoded_np.astype(np.uint8) != np.repeat(bits[:, None], len(METHODS), axis=1), axis=-1)
    peak = torch.max(torch.abs(waveform) ** 2, dim=-1).values.squeeze((-1, -2))
    mean = torch.mean(torch.abs(waveform) ** 2, dim=-1).squeeze((-1, -2))
    papr = (peak / torch.clamp(mean, min=1e-12)).detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for trajectory_index in range(int(trajectories)):
        row_methods: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            row_methods[method] = {
                "crc_pass": bool(crc_np[trajectory_index, method_index]),
                "bit_errors": int(bit_errors[trajectory_index, method_index]),
                "rb_power": allocations[trajectory_index, method_index].tolist(),
                "waveform_energy_before_normalization": float(energy_before[trajectory_index, method_index]),
                "waveform_energy_after_normalization": float(energy_after[trajectory_index, method_index]),
                "waveform_normalization": float(normalization[trajectory_index, method_index]),
                "papr_linear": float(papr[trajectory_index, method_index]),
            }
        rows.append({"trajectory_index": trajectory_index, "methods": row_methods})
    elapsed = time.perf_counter() - started
    max_energy_relative_error = float(
        torch.max(
            torch.abs(energy_after - reference_energy)
            / torch.clamp(reference_energy, min=1e-12)
        ).item()
    )
    return {
        "config_sha256": _config_sha256(),
        "seed": int(seed),
        "trajectory_count": int(trajectories),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "correlations": correlations,
        "constraints": {
            "maximum_RB_sum_power_error": sum_error,
            "maximum_lower_power_violation": lower_violation,
            "maximum_upper_power_violation": upper_violation,
            "maximum_paired_waveform_energy_relative_error": max_energy_relative_error,
            "equal_allocation_max_abs_error_from_one": float(
                np.max(np.abs(allocations[:, 0] - 1.0))
            ),
        },
        "rows": rows,
    }


def self_test() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(20, 10))
    projected = _box_project(values)
    assert np.max(np.abs(np.sum(projected, axis=1) - 10.0)) < 1e-10
    assert np.min(projected) >= 0.5 - 1e-12
    assert np.max(projected) <= 1.5 + 1e-12
    gains = rng.lognormal(size=(20, 10))
    wf = _box_water_filling(gains)
    assert np.max(np.abs(np.sum(wf, axis=1) - 10.0)) < 1e-10
    assert CONFIG["transmitter_csi"]["feedback_delay_ofdm_symbols_to_slot_start"] == 5
    assert MODEL.is_file()
    assert _sha256(MODEL) == "6758833388bc2b9acf75f8df7cac482118642ac3a8b72662144baae3d709714e"
    print("self-test passed")


def smoke() -> dict[str, Any]:
    path = OUTPUT / "smoke.json"
    if path.is_file():
        result = _read(path)
        print(f"reusing smoke: {path}")
        return result
    result = _run_batch(int(CONFIG["smoke"]["seed"]), int(CONFIG["smoke"]["trajectories"]))
    limits = CONFIG["resource_limits"]
    result["kind"] = "delayed_csi_nr_pusch_link_smoke"
    result["passed"] = bool(
        result["elapsed_seconds"] <= float(limits["maximum_unit_wall_seconds"])
        and result["peak_rss_bytes"] <= int(limits["maximum_peak_rss_bytes"])
        and result["correlations"]["delayed_current_gain_correlation"]
        >= float(CONFIG["pilot"]["minimum_gain_correlation"])
        and max(result["constraints"].values()) <= 1e-6
    )
    _write_new(path, result)
    print(
        "smoke passed=%s wall=%.2fs peak_rss=%.1fMiB gain_corr=%.3f"
        % (
            result["passed"],
            result["elapsed_seconds"],
            result["peak_rss_bytes"] / 1024**2,
            result["correlations"]["delayed_current_gain_correlation"],
        )
    )
    return result


def run(max_units: int | None = None) -> None:
    smoke_result = smoke()
    if not smoke_result.get("passed"):
        raise PilotError("infrastructure smoke did not pass")
    total_units = int(CONFIG["pilot"]["units"])
    limit = total_units if max_units is None else min(total_units, max(0, int(max_units)))
    completed = 0
    for unit_index in range(total_units):
        path = OUTPUT / "units" / ("unit_%02d.json" % (unit_index + 1))
        if path.is_file():
            continue
        if completed >= limit:
            break
        seed = int(CONFIG["pilot"]["unit_seed_start"]) + unit_index
        result = _run_batch(seed, int(CONFIG["pilot"]["trajectories_per_unit"]))
        result["kind"] = "delayed_csi_nr_pusch_link_pilot_unit"
        result["unit_index"] = unit_index + 1
        limits = CONFIG["resource_limits"]
        result["resource_guard_passed"] = bool(
            result["elapsed_seconds"] <= float(limits["maximum_unit_wall_seconds"])
            and result["peak_rss_bytes"] <= int(limits["maximum_peak_rss_bytes"])
            and max(result["constraints"].values()) <= 1e-6
        )
        _write_new(path, result)
        completed += 1
        print(
            "unit=%d trajectories=%d wall=%.2fs gain_corr=%.3f guard=%s"
            % (
                unit_index + 1,
                result["trajectory_count"],
                result["elapsed_seconds"],
                result["correlations"]["delayed_current_gain_correlation"],
                result["resource_guard_passed"],
            )
        )
        if not result["resource_guard_passed"]:
            raise PilotError("resource/constraint guard failed; stopping pilot")


def analyze() -> dict[str, Any]:
    paths = sorted((OUTPUT / "units").glob("unit_*.json"))
    if not paths:
        raise PilotError("no pilot units exist")
    units = [_read(path) for path in paths]
    rows = [row for unit in units for row in unit["rows"]]
    deliveries = {method: 0 for method in METHODS}
    bit_errors = {method: 0 for method in METHODS}
    per_method_crc: dict[str, list[bool]] = {method: [] for method in METHODS}
    for row in rows:
        for method in METHODS:
            outcome = row["methods"][method]
            passed = bool(outcome["crc_pass"])
            deliveries[method] += int(passed)
            bit_errors[method] += int(outcome["bit_errors"])
            per_method_crc[method].append(passed)
    count = len(rows)
    learned = np.asarray(per_method_crc["learned_transfer_seed35023"], dtype=np.int8)
    contrasts: dict[str, Any] = {}
    for comparator in ("equal_power", "stale_csi_box_water_filling", "causal_ar_box_water_filling"):
        other = np.asarray(per_method_crc[comparator], dtype=np.int8)
        contrasts[f"learned_minus_{comparator}"] = {
            "delivery_count_difference": int(np.sum(learned) - np.sum(other)),
            "delivery_rate_difference": float(np.mean(learned - other)),
            "paired_wins": int(np.sum((learned == 1) & (other == 0))),
            "paired_losses": int(np.sum((learned == 0) & (other == 1))),
            "paired_ties": int(np.sum(learned == other)),
        }
    minimum_gain = int(CONFIG["pilot"]["promising_minimum_delivery_gain"])
    equal_bler = 1.0 - deliveries["equal_power"] / count
    bler_low, bler_high = CONFIG["pilot"]["equal_bler_interval"]
    correlation = float(np.mean([unit["correlations"]["delayed_current_gain_correlation"] for unit in units]))
    all_units_complete = len(units) == int(CONFIG["pilot"]["units"])
    promising = bool(
        all_units_complete
        and contrasts["learned_minus_equal_power"]["delivery_count_difference"] >= minimum_gain
        and contrasts["learned_minus_stale_csi_box_water_filling"]["delivery_count_difference"] >= minimum_gain
        and contrasts["learned_minus_equal_power"]["paired_wins"]
        > contrasts["learned_minus_equal_power"]["paired_losses"]
        and contrasts["learned_minus_stale_csi_box_water_filling"]["paired_wins"]
        > contrasts["learned_minus_stale_csi_box_water_filling"]["paired_losses"]
        and float(bler_low) <= equal_bler <= float(bler_high)
        and correlation >= float(CONFIG["pilot"]["minimum_gain_correlation"])
        and all(bool(unit["resource_guard_passed"]) for unit in units)
    )
    report = {
        "kind": "delayed_csi_nr_pusch_link_pilot_analysis",
        "schema_version": 1,
        "config": CONFIG,
        "config_sha256": _config_sha256(),
        "pilot_complete": all_units_complete,
        "trajectory_count": count,
        "method_results": {
            method: {
                "crc_deliveries": deliveries[method],
                "attempts": count,
                "bler": 1.0 - deliveries[method] / count,
                "crc_gated_goodput_bits_per_attempt": 1608.0 * deliveries[method] / count,
                "decoded_bit_errors": bit_errors[method],
            }
            for method in METHODS
        },
        "paired_contrasts": contrasts,
        "mean_delayed_current_gain_correlation": correlation,
        "pilot_decision": "promising_for_new_prospective_study" if promising else "use_controlled_mechanism_result",
        "claim_boundary": (
            "Development-only normalized NR-framed link-level pilot; not publication evidence, "
            "not deployment throughput, and not NR conformance."
        ),
    }
    output = OUTPUT / "analysis.json"
    if output.exists():
        existing = _read(output)
        if existing != report:
            raise PilotError("existing analysis differs; refusing overwrite")
    else:
        _write_new(output, report)
    for method in METHODS:
        row = report["method_results"][method]
        print(f"{method:48s} {row['crc_deliveries']}/{count} BLER={row['bler']:.3f}")
    for name, row in contrasts.items():
        print(
            f"{name}: delta={row['delivery_count_difference']:+d}/{count} "
            f"wins={row['paired_wins']} losses={row['paired_losses']} ties={row['paired_ties']}"
        )
    print(f"pilot_decision={report['pilot_decision']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "smoke", "run", "analyze", "show"))
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "smoke":
        self_test()
        smoke()
    elif args.command == "run":
        run(args.max_units)
    elif args.command in {"analyze", "show"}:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
