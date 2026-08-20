#!/usr/bin/env python3
"""Standalone delayed-CSI predictive power-allocation experiment.

The experiment is intentionally isolated from the paper's completed delayed-CSI
campaign.  It uses a transparent two-path OFDM channel, fresh seed namespaces,
and a continuous Shannon-rate endpoint.  The learned policy receives exactly
two noisy channel reports, the newest of which is four slots old.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".noema" / "demos" / "delayed_csi_predictive_allocation_v2"

CONFIG: dict[str, Any] = {
    "study_id": "delayed_csi_predictive_allocation_v2",
    "channel": {
        "model": "two_path_ofdm_persistent_doppler_markov",
        "subcarriers": 16,
        "phase_states": 64,
        "second_path_relative_amplitude": 0.9,
        "second_path_delay_samples": 3,
        "phase_step_states_per_slot": 2,
        "velocity_flip_probability": 0.01,
        "feedback_delay_slots": 4,
        "history_snapshots": 2,
        "csi_error_variance_per_complex_subcarrier": 0.05,
    },
    "endpoint": {
        "name": "mean_achievable_spectral_efficiency",
        "units": "bit/s/Hz/subcarrier",
        "noise_variance": 1.0,
        "total_power": 16.0,
        "power_floor": 0.0,
    },
    "training": {
        "sample_seed": 211000001,
        "validation_seed": 212000001,
        "samples": 120000,
        "validation_samples": 20000,
        "initialization_seeds": [49009, 59011, 69017],
        "epochs": 25,
        "batch_size": 1024,
        "learning_rate": 0.002,
        "weight_decay": 0.00001,
        "hidden_width": 128,
    },
    "development": {
        "trajectory_seed_start": 213000001,
        "trajectories": 240,
        "decisions_per_trajectory": 128,
        "bootstrap_seed": 213900001,
        "bootstrap_resamples": 10000,
        "minimum_mean_gain_over_equal": 0.03,
        "minimum_mean_gain_over_stale_water_filling": 0.03,
        "maximum_independent_control_gain_over_equal": 0.02,
    },
    "heldout": {
        "trajectory_seed_start": 214000001,
        "trajectories": 500,
        "decisions_per_trajectory": 128,
        "independent_control_seed_start": 214900001,
        "bootstrap_seed": 215000001,
        "bootstrap_resamples": 20000,
        "familywise_confidence": 0.95,
        "one_sided_per_contrast_alpha": 0.025,
        "minimum_simultaneous_lower_bound": 0.03,
    },
    "methods": [
        "equal_power",
        "stale_csi_water_filling",
        "estimated_predictive_water_filling",
        "learned_predictive_allocator",
        "known_state_causal_optimum",
        "current_csi_water_filling_oracle",
    ],
}


class StudyError(RuntimeError):
    pass


class DelayedCsiPolicy(torch.nn.Module):
    """Small learned allocator with an exact nonnegative sum-power output."""

    def __init__(self, subcarriers: int = 16, hidden_width: int = 128) -> None:
        super().__init__()
        width = int(hidden_width)
        tones = int(subcarriers)
        self.subcarriers = tones
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4 * tones, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, tones),
        )
        final = self.network[-1]
        assert isinstance(final, torch.nn.Linear)
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def forward(self, delayed_history_ri: torch.Tensor) -> torch.Tensor:
        if delayed_history_ri.ndim != 2:
            raise ValueError("policy input must have shape [batch, 4*subcarriers]")
        logits = self.network(delayed_history_ri.float())
        total = float(CONFIG["endpoint"]["total_power"])
        return torch.softmax(logits, dim=1) * total


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256() -> str:
    return sha256_bytes(canonical_json(CONFIG))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StudyError(f"required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StudyError(f"expected a JSON object: {path}")
    return value


def state_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel = CONFIG["channel"]
    n = int(channel["subcarriers"])
    q_count = int(channel["phase_states"])
    amplitude = float(channel["second_path_relative_amplitude"])
    delay = int(channel["second_path_delay_samples"])
    phase = 2.0 * np.pi * np.arange(q_count, dtype=np.float64) / q_count
    tone = np.arange(n, dtype=np.float64)
    response = (
        1.0
        + amplitude
        * np.exp(1j * (phase[:, None] - 2.0 * np.pi * delay * tone[None, :] / n))
    ) / math.sqrt(1.0 + amplitude * amplitude)
    gains = np.abs(response) ** 2

    states = 2 * q_count
    transition = np.zeros((states, states), dtype=np.float64)
    step = int(channel["phase_step_states_per_slot"])
    flip = float(channel["velocity_flip_probability"])
    for q in range(q_count):
        for velocity_index, velocity in enumerate((-1, 1)):
            source = 2 * q + velocity_index
            for next_velocity, probability in ((velocity, 1.0 - flip), (-velocity, flip)):
                next_q = (q + step * next_velocity) % q_count
                next_index = 0 if next_velocity == -1 else 1
                transition[source, 2 * next_q + next_index] += probability
    horizon = int(channel["feedback_delay_slots"])
    delayed_transition = np.linalg.matrix_power(transition, horizon)
    return response.astype(np.complex128), gains.astype(np.float64), delayed_transition


def channel_response_for_state_ids(state_ids: np.ndarray) -> np.ndarray:
    response, _, _ = state_tables()
    q = np.asarray(state_ids, dtype=np.int64) // 2
    return response[q]


def encode_history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history)
    n = int(CONFIG["channel"]["subcarriers"])
    if values.shape[-2:] != (2, n):
        raise ValueError(f"history must end in [2,{n}], got {values.shape}")
    flattened = values.reshape(-1, 2 * n)
    return np.concatenate((flattened.real, flattened.imag), axis=1).astype(np.float32)


def water_filling(gains: np.ndarray) -> np.ndarray:
    """Exact Shannon water filling under the demo's common sum-power constraint."""

    values = np.maximum(np.asarray(gains, dtype=np.float64), 1e-12)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("gains must have shape [subcarrier] or [batch,subcarrier]")
    noise = float(CONFIG["endpoint"]["noise_variance"])
    total = float(CONFIG["endpoint"]["total_power"])
    floors = noise / values
    lower = np.min(floors, axis=1)
    upper = np.max(floors, axis=1) + total
    for _ in range(60):
        level = 0.5 * (lower + upper)
        power = np.maximum(level[:, None] - floors, 0.0)
        too_much = np.sum(power, axis=1) > total
        upper = np.where(too_much, level, upper)
        lower = np.where(too_much, lower, level)
    power = np.maximum(0.5 * (lower + upper)[:, None] - floors, 0.0)
    power *= total / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-15)
    return power[0] if squeeze else power


def spectral_efficiency(gains: np.ndarray, power: np.ndarray) -> np.ndarray:
    noise = float(CONFIG["endpoint"]["noise_variance"])
    return np.mean(np.log2(1.0 + np.asarray(gains) * np.asarray(power) / noise), axis=-1)


def _expected_marginal(
    power: np.ndarray,
    probabilities: np.ndarray,
    gains_by_state: np.ndarray,
) -> np.ndarray:
    noise = float(CONFIG["endpoint"]["noise_variance"])
    denominator = noise + gains_by_state * power[None, :]
    return np.sum(probabilities[:, None] * gains_by_state / denominator, axis=0)


def exact_causal_allocation_table() -> np.ndarray:
    """Solve the conditional expected-rate KKT equations for every old state."""

    _, gains_q, delayed_transition = state_tables()
    q_count, n = gains_q.shape
    gains_state = np.repeat(gains_q, 2, axis=0)
    total = float(CONFIG["endpoint"]["total_power"])
    table = np.zeros((2 * q_count, n), dtype=np.float64)
    for old_state, probabilities in enumerate(delayed_transition):
        active = probabilities > 1e-15
        probs = probabilities[active]
        possible_gains = gains_state[active]
        derivative_zero = _expected_marginal(np.zeros(n), probs, possible_gains)
        lambda_low = 0.0
        lambda_high = float(np.max(derivative_zero))
        chosen = np.full(n, total / n, dtype=np.float64)
        for _ in range(55):
            multiplier = 0.5 * (lambda_low + lambda_high)
            p_low = np.zeros(n, dtype=np.float64)
            p_high = np.full(n, total, dtype=np.float64)
            inactive = derivative_zero <= multiplier
            p_high[inactive] = 0.0
            for _ in range(45):
                midpoint = 0.5 * (p_low + p_high)
                derivative = _expected_marginal(midpoint, probs, possible_gains)
                needs_more_power = derivative > multiplier
                p_low = np.where(needs_more_power, midpoint, p_low)
                p_high = np.where(needs_more_power, p_high, midpoint)
            candidate = 0.5 * (p_low + p_high)
            if float(np.sum(candidate)) > total:
                lambda_low = multiplier
            else:
                lambda_high = multiplier
            chosen = candidate
        chosen *= total / max(float(np.sum(chosen)), 1e-15)
        table[old_state] = chosen
    return table


def predictive_tables() -> tuple[np.ndarray, np.ndarray]:
    _, gains_q, delayed_transition = state_tables()
    gains_state = np.repeat(gains_q, 2, axis=0)
    conditional_mean = delayed_transition @ gains_state
    predicted_wf = water_filling(conditional_mean)
    exact = exact_causal_allocation_table()
    return predicted_wf, exact


def _sample_next_states(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q_count = int(CONFIG["channel"]["phase_states"])
    step = int(CONFIG["channel"]["phase_step_states_per_slot"])
    flip_probability = float(CONFIG["channel"]["velocity_flip_probability"])
    q = states // 2
    velocity = np.where(states % 2 == 0, -1, 1)
    flips = rng.random(states.shape) < flip_probability
    velocity = np.where(flips, -velocity, velocity)
    q = (q + step * velocity) % q_count
    return (2 * q + (velocity == 1).astype(np.int64)).astype(np.int64)


def noisy_observation(state_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    true = channel_response_for_state_ids(state_ids)
    variance = float(CONFIG["channel"]["csi_error_variance_per_complex_subcarrier"])
    scale = math.sqrt(variance / 2.0)
    noise = scale * (rng.standard_normal(true.shape) + 1j * rng.standard_normal(true.shape))
    return (true + noise).astype(np.complex64)


def sample_independent_windows(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    q_count = int(CONFIG["channel"]["phase_states"])
    step = int(CONFIG["channel"]["phase_step_states_per_slot"])
    old_states = rng.integers(0, 2 * q_count, size=int(count), dtype=np.int64)
    old_q = old_states // 2
    old_velocity = np.where(old_states % 2 == 0, -1, 1)
    previous_q = (old_q - step * old_velocity) % q_count
    previous_states = 2 * previous_q + (old_velocity == 1).astype(np.int64)
    history = np.stack(
        (noisy_observation(previous_states, rng), noisy_observation(old_states, rng)),
        axis=1,
    )
    current_states = old_states.copy()
    for _ in range(int(CONFIG["channel"]["feedback_delay_slots"])):
        current_states = _sample_next_states(current_states, rng)
    _, gains_q, _ = state_tables()
    current_gains = gains_q[current_states // 2].astype(np.float32)
    return history, current_gains


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)


def _model_from_checkpoint(path: Path) -> DelayedCsiPolicy:
    if not path.is_file():
        raise StudyError(f"trained model is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("config_sha256") != config_sha256():
        raise StudyError("trained model configuration does not match this demo")
    model = DelayedCsiPolicy(
        subcarriers=int(CONFIG["channel"]["subcarriers"]),
        hidden_width=int(CONFIG["training"]["hidden_width"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def train(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    report_path = output_dir / "training.json"
    if model_path.exists() or report_path.exists():
        if not (model_path.is_file() and report_path.is_file()):
            raise StudyError("partial training output exists; refusing to overwrite it")
        report = read_json(report_path)
        if report.get("config_sha256") != config_sha256():
            raise StudyError("existing training output belongs to a different configuration")
        if report.get("model_sha256") != sha256_file(model_path):
            raise StudyError("existing model hash disagrees with training report")
        print(f"reusing locked training artifact: {model_path}")
        return report

    training = CONFIG["training"]
    history, gains = sample_independent_windows(int(training["samples"]), int(training["sample_seed"]))
    validation_history, validation_gains = sample_independent_windows(
        int(training["validation_samples"]), int(training["validation_seed"])
    )
    features = torch.from_numpy(encode_history(history))
    targets = torch.from_numpy(gains)
    validation_features = torch.from_numpy(encode_history(validation_history))
    validation_targets = torch.from_numpy(validation_gains)
    batch_size = int(training["batch_size"])
    best_rate = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_seed = -1
    best_epoch = -1
    histories: list[dict[str, Any]] = []
    started = time.perf_counter()
    for initialization_seed in training["initialization_seeds"]:
        seed = int(initialization_seed)
        _set_determinism(seed)
        model = DelayedCsiPolicy(
            subcarriers=int(CONFIG["channel"]["subcarriers"]),
            hidden_width=int(training["hidden_width"]),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(seed + 700000)
        seed_best = -float("inf")
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            order = torch.randperm(features.shape[0], generator=generator)
            loss_sum = 0.0
            seen = 0
            for start in range(0, features.shape[0], batch_size):
                indices = order[start : start + batch_size]
                x = features[indices]
                current_gains = targets[indices]
                power = model(x)
                rate = torch.mean(
                    torch.log2(
                        1.0
                        + current_gains
                        * power
                        / float(CONFIG["endpoint"]["noise_variance"])
                    ),
                    dim=1,
                )
                loss = -torch.mean(rate)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = int(indices.numel())
                loss_sum += float(loss.detach()) * count
                seen += count
            model.eval()
            with torch.no_grad():
                validation_power = model(validation_features)
                validation_rate = float(
                    torch.mean(
                        torch.mean(
                            torch.log2(
                                1.0
                                + validation_targets
                                * validation_power
                                / float(CONFIG["endpoint"]["noise_variance"])
                            ),
                            dim=1,
                        )
                    )
                )
            row = {
                "initialization_seed": seed,
                "epoch": epoch,
                "training_rate": -loss_sum / max(seen, 1),
                "validation_rate": validation_rate,
            }
            histories.append(row)
            if validation_rate > seed_best:
                seed_best = validation_rate
            if validation_rate > best_rate:
                best_rate = validation_rate
                best_seed = seed
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0 or epoch == int(training["epochs"]):
                print(
                    f"seed={seed} epoch={epoch:02d} train_rate={-loss_sum/max(seen,1):.6f} "
                    f"validation_rate={validation_rate:.6f}"
                )
        print(f"seed={seed} best_validation_rate={seed_best:.6f}")
    if best_state is None:
        raise StudyError("training produced no candidate")
    payload = {
        "config_sha256": config_sha256(),
        "selected_initialization_seed": best_seed,
        "selected_epoch": best_epoch,
        "state_dict": best_state,
    }
    torch.save(payload, model_path)
    report = {
        "kind": "delayed_csi_standalone_training",
        "schema_version": 1,
        "config": CONFIG,
        "config_sha256": config_sha256(),
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_file(model_path),
        "selected_initialization_seed": best_seed,
        "selected_epoch": best_epoch,
        "selected_validation_rate": best_rate,
        "elapsed_seconds": time.perf_counter() - started,
        "history": histories,
    }
    write_json(report_path, report)
    print(
        f"selected seed={best_seed} epoch={best_epoch} validation_rate={best_rate:.6f}; "
        f"saved {model_path}"
    )
    return report


def _trajectory_batch(
    *,
    seed_start: int,
    trajectories: int,
    decisions: int,
    independent_current_seed_start: int | None = None,
) -> dict[str, np.ndarray]:
    q_count = int(CONFIG["channel"]["phase_states"])
    delay = int(CONFIG["channel"]["feedback_delay_slots"])
    histories: list[np.ndarray] = []
    current_gains: list[np.ndarray] = []
    old_states_rows: list[np.ndarray] = []
    current_states_rows: list[np.ndarray] = []
    _, gains_q, _ = state_tables()
    for trajectory_index in range(int(trajectories)):
        rng = np.random.default_rng(int(seed_start) + trajectory_index)
        states = np.empty(int(decisions) + delay + 1, dtype=np.int64)
        states[0] = int(rng.integers(0, 2 * q_count))
        for time_index in range(1, states.size):
            states[time_index] = _sample_next_states(states[time_index - 1 : time_index], rng)[0]
        observed = noisy_observation(states, rng)
        history = np.stack((observed[:decisions], observed[1 : decisions + 1]), axis=1)
        old_states = states[1 : decisions + 1]
        actual_states = states[1 + delay : 1 + delay + decisions]
        if independent_current_seed_start is not None:
            control_rng = np.random.default_rng(int(independent_current_seed_start) + trajectory_index)
            actual_states = control_rng.integers(0, 2 * q_count, size=int(decisions), dtype=np.int64)
        histories.append(history)
        current_gains.append(gains_q[actual_states // 2])
        old_states_rows.append(old_states)
        current_states_rows.append(actual_states)
    return {
        "history": np.stack(histories).astype(np.complex64),
        "current_gains": np.stack(current_gains).astype(np.float64),
        "old_states": np.stack(old_states_rows).astype(np.int64),
        "current_states": np.stack(current_states_rows).astype(np.int64),
    }


def estimate_old_states(history: np.ndarray) -> np.ndarray:
    response, _, _ = state_tables()
    values = np.asarray(history)
    original_shape = values.shape[:-2]
    flattened = values.reshape(-1, 2, values.shape[-1])
    estimated_q = np.empty((flattened.shape[0], 2), dtype=np.int64)
    template_conjugate = np.conjugate(response)
    for start in range(0, flattened.shape[0], 4096):
        chunk = flattened[start : start + 4096]
        score = np.real(np.einsum("bhn,qn->bhq", chunk, template_conjugate, optimize=True))
        estimated_q[start : start + chunk.shape[0]] = np.argmax(score, axis=2)
    q_count = int(CONFIG["channel"]["phase_states"])
    step = int(CONFIG["channel"]["phase_step_states_per_slot"])
    delta = (estimated_q[:, 1] - estimated_q[:, 0]) % q_count
    distance_plus = np.minimum((delta - step) % q_count, (step - delta) % q_count)
    minus_step = (-step) % q_count
    distance_minus = np.minimum((delta - minus_step) % q_count, (minus_step - delta) % q_count)
    velocity_positive = distance_plus <= distance_minus
    state = 2 * estimated_q[:, 1] + velocity_positive.astype(np.int64)
    return state.reshape(original_shape)


def learned_power(model: DelayedCsiPolicy, history: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    features = encode_history(history)
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            rows.append(model(torch.from_numpy(features[start : start + batch_size])).cpu().numpy())
    power = np.concatenate(rows, axis=0).reshape(
        *history.shape[:-2], history.shape[-1]
    ).astype(np.float64)
    total = float(CONFIG["endpoint"]["total_power"])
    power *= total / np.maximum(np.sum(power, axis=-1, keepdims=True), 1e-15)
    return power


def _method_powers(model: DelayedCsiPolicy, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    history = batch["history"]
    gains = batch["current_gains"]
    trajectories, decisions, n = gains.shape
    predicted_table, exact_table = predictive_tables()
    estimated_states = estimate_old_states(history)
    total = float(CONFIG["endpoint"]["total_power"])
    return {
        "equal_power": np.full((trajectories, decisions, n), total / n, dtype=np.float64),
        "stale_csi_water_filling": water_filling(
            (np.abs(history[:, :, -1]) ** 2).reshape(-1, n)
        ).reshape(trajectories, decisions, n),
        "estimated_predictive_water_filling": predicted_table[estimated_states],
        "learned_predictive_allocator": learned_power(model, history),
        "known_state_causal_optimum": exact_table[batch["old_states"]],
        "current_csi_water_filling_oracle": water_filling(gains.reshape(-1, n)).reshape(trajectories, decisions, n),
    }


def delayed_correlations(batch: dict[str, np.ndarray]) -> dict[str, float]:
    response, _, _ = state_tables()
    old = response[batch["old_states"] // 2].reshape(-1)
    current = response[batch["current_states"] // 2].reshape(-1)
    complex_correlation = abs(np.mean(current * np.conjugate(old))) / math.sqrt(
        float(np.mean(np.abs(current) ** 2) * np.mean(np.abs(old) ** 2))
    )
    old_gain = np.abs(old) ** 2
    current_gain = np.abs(current) ** 2
    gain_correlation = float(np.corrcoef(old_gain, current_gain)[0, 1])
    return {
        "delayed_current_complex_correlation_magnitude": float(complex_correlation),
        "delayed_current_gain_correlation": gain_correlation,
    }


def bootstrap_interval(values: np.ndarray, seed: int, resamples: int) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    offset = 0
    while offset < int(resamples):
        count = min(1000, int(resamples) - offset)
        indices = rng.integers(0, samples.size, size=(count, samples.size))
        means[offset : offset + count] = np.mean(samples[indices], axis=1)
        offset += count
    return {
        "mean": float(np.mean(samples)),
        "two_sided_95_lower": float(np.quantile(means, 0.025)),
        "two_sided_95_upper": float(np.quantile(means, 0.975)),
        "one_sided_bonferroni_97p5_lower": float(np.quantile(means, 0.025)),
        "trajectory_win_fraction": float(np.mean(samples > 0.0)),
    }


def evaluate_batch(
    model: DelayedCsiPolicy,
    batch: dict[str, np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    powers = _method_powers(model, batch)
    trajectory_rate: dict[str, np.ndarray] = {}
    method_rows: dict[str, Any] = {}
    total = float(CONFIG["endpoint"]["total_power"])
    max_power_error = 0.0
    max_negative = 0.0
    for method, allocation in powers.items():
        state_rate = spectral_efficiency(batch["current_gains"], allocation)
        trajectory_rate[method] = np.mean(state_rate, axis=1)
        power_error = float(np.max(np.abs(np.sum(allocation, axis=-1) - total)))
        negative = float(np.max(np.maximum(-allocation, 0.0)))
        max_power_error = max(max_power_error, power_error)
        max_negative = max(max_negative, negative)
        method_rows[method] = {
            "mean_rate": float(np.mean(trajectory_rate[method])),
            "trajectory_standard_deviation": float(np.std(trajectory_rate[method], ddof=1)),
            "max_sum_power_error": power_error,
            "max_negative_power": negative,
        }
    learned = trajectory_rate["learned_predictive_allocator"]
    contrasts: dict[str, Any] = {}
    comparator_names = [
        "equal_power",
        "stale_csi_water_filling",
        "estimated_predictive_water_filling",
        "known_state_causal_optimum",
        "current_csi_water_filling_oracle",
    ]
    for index, comparator in enumerate(comparator_names):
        contrasts[f"learned_minus_{comparator}"] = bootstrap_interval(
            learned - trajectory_rate[comparator],
            int(bootstrap_seed) + 1009 * index,
            int(bootstrap_resamples),
        )
    oracle_state_rate = spectral_efficiency(
        batch["current_gains"], powers["current_csi_water_filling_oracle"]
    )
    oracle_violation = 0.0
    for method, allocation in powers.items():
        if method == "current_csi_water_filling_oracle":
            continue
        method_state_rate = spectral_efficiency(batch["current_gains"], allocation)
        oracle_violation = max(oracle_violation, float(np.max(method_state_rate - oracle_state_rate)))
    return {
        "trajectory_count": int(batch["history"].shape[0]),
        "decisions_per_trajectory": int(batch["history"].shape[1]),
        "correlations": delayed_correlations(batch),
        "methods": method_rows,
        "paired_contrasts": contrasts,
        "constraints": {
            "maximum_sum_power_error": max_power_error,
            "maximum_negative_power": max_negative,
            "maximum_current_csi_oracle_dominance_violation": oracle_violation,
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    evaluation = report.get("primary_evaluation", report)
    for method, row in evaluation["methods"].items():
        print(f"{method:42s} {row['mean_rate']:.6f} bit/s/Hz/subcarrier")
    for name, row in evaluation["paired_contrasts"].items():
        print(
            f"{name:58s} mean={row['mean']:+.6f} "
            f"95% CI=[{row['two_sided_95_lower']:+.6f},{row['two_sided_95_upper']:+.6f}]"
        )


def _development_gate_from_report(report: dict[str, Any]) -> bool:
    config = CONFIG["development"]
    primary = report["primary_evaluation"]
    control = report["independent_current_control"]
    equal_gain = primary["paired_contrasts"]["learned_minus_equal_power"]["mean"]
    stale_gain = primary["paired_contrasts"]["learned_minus_stale_csi_water_filling"]["mean"]
    control_gain = control["paired_contrasts"]["learned_minus_equal_power"]["mean"]
    return bool(
        equal_gain >= float(config["minimum_mean_gain_over_equal"])
        and stale_gain >= float(config["minimum_mean_gain_over_stale_water_filling"])
        # With an independent current channel, equal power is optimal in
        # expectation.  The causal policy must show no positive advantage;
        # being worse is an expected, valid negative-control outcome.
        and control_gain <= float(config["maximum_independent_control_gain_over_equal"])
        and primary["correlations"]["delayed_current_gain_correlation"] > 0.5
        and abs(control["correlations"]["delayed_current_gain_correlation"]) <= 0.05
        and primary["constraints"]["maximum_sum_power_error"] <= 1e-5
        and primary["constraints"]["maximum_negative_power"] <= 1e-12
        and primary["constraints"]["maximum_current_csi_oracle_dominance_violation"] <= 1e-9
    )


def _verify_development_report(output_dir: Path, report: dict[str, Any]) -> bool:
    if report.get("kind") != "delayed_csi_standalone_development":
        raise StudyError("development result has an unexpected kind")
    if int(report.get("schema_version", 0)) != 1:
        raise StudyError("development result has an unsupported schema version")
    if report.get("config_sha256") != config_sha256():
        raise StudyError("development result configuration hash mismatch")
    if report.get("model_sha256") != sha256_file(output_dir / "model.pt"):
        raise StudyError("development result model hash mismatch")
    recomputed = _development_gate_from_report(report)
    if bool(report.get("development_gate_passed")) != recomputed:
        raise StudyError("stored development gate disagrees with recomputed metrics")
    return recomputed


def development(output_dir: Path) -> dict[str, Any]:
    train(output_dir)
    model_path = output_dir / "model.pt"
    result_path = output_dir / "development.json"
    if result_path.is_file():
        report = read_json(result_path)
        _verify_development_report(output_dir, report)
        print(f"reusing development result: {result_path}")
        _print_summary(report)
        return report
    model = _model_from_checkpoint(model_path)
    config = CONFIG["development"]
    primary_batch = _trajectory_batch(
        seed_start=int(config["trajectory_seed_start"]),
        trajectories=int(config["trajectories"]),
        decisions=int(config["decisions_per_trajectory"]),
    )
    control_batch = _trajectory_batch(
        seed_start=int(config["trajectory_seed_start"]),
        trajectories=int(config["trajectories"]),
        decisions=int(config["decisions_per_trajectory"]),
        independent_current_seed_start=int(config["trajectory_seed_start"]) + 700000,
    )
    primary = evaluate_batch(
        model,
        primary_batch,
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
    )
    control = evaluate_batch(
        model,
        control_batch,
        bootstrap_seed=int(config["bootstrap_seed"]) + 500000,
        bootstrap_resamples=int(config["bootstrap_resamples"]),
    )
    report = {
        "kind": "delayed_csi_standalone_development",
        "schema_version": 1,
        "config_sha256": config_sha256(),
        "model_sha256": sha256_file(model_path),
        "development_gate_passed": False,
        "primary_evaluation": primary,
        "independent_current_control": control,
    }
    report["development_gate_passed"] = _development_gate_from_report(report)
    write_json(result_path, report)
    _print_summary(report)
    print(f"development_gate_passed={report['development_gate_passed']}")
    return report


def freeze_final(output_dir: Path) -> dict[str, Any]:
    development_report = development(output_dir)
    if not _verify_development_report(output_dir, development_report):
        raise StudyError("development gate did not pass; held-out execution is blocked")
    freeze_path = output_dir / "final_design_freeze.json"
    if freeze_path.is_file():
        freeze = read_json(freeze_path)
        _verify_freeze(output_dir, freeze)
        print(f"reusing final design freeze: {freeze_path}")
        return freeze
    model_path = output_dir / "model.pt"
    development_path = output_dir / "development.json"
    script_path = Path(__file__).resolve()
    freeze = {
        "kind": "delayed_csi_standalone_final_design_freeze",
        "schema_version": 1,
        "config": CONFIG,
        "config_sha256": config_sha256(),
        "script_path": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256_file(script_path),
        "model_sha256": sha256_file(model_path),
        "development_sha256": sha256_file(development_path),
        "heldout_outcomes_accessed": False,
        "prior_development_disclosure": (
            "A preliminary v1 development run on the same simulator coordinate was observed. "
            "Before any v2 held-out access, v2 corrected an independent-current control from "
            "an absolute-deviation rule to the scientifically appropriate no-positive-gain rule "
            "and used fresh training, validation, development, and held-out seed namespaces."
        ),
    }
    write_json(freeze_path, freeze)
    print(f"froze final design: {freeze_path}")
    return freeze


def _verify_freeze(output_dir: Path, freeze: dict[str, Any] | None = None) -> dict[str, Any]:
    freeze_path = output_dir / "final_design_freeze.json"
    value = freeze or read_json(freeze_path)
    checks = {
        "config_sha256": config_sha256(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "model_sha256": sha256_file(output_dir / "model.pt"),
        "development_sha256": sha256_file(output_dir / "development.json"),
    }
    for key, actual in checks.items():
        if value.get(key) != actual:
            raise StudyError(f"final freeze mismatch for {key}")
    return value


def _plot_results(report: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Equal", "Stale WF", "Predictive WF", "Learned", "Causal ref.", "Current-CSI\noracle"]
    keys = CONFIG["methods"]
    primary = report["primary_evaluation"]
    control = report["independent_current_control"]
    primary_values = [primary["methods"][key]["mean_rate"] for key in keys]
    control_values = [control["methods"][key]["mean_rate"] for key in keys]
    colors = ["#7f8c8d", "#d97706", "#2563eb", "#16a34a", "#7c3aed", "#111827"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharey=True)
    for axis, values, title in zip(
        axes,
        (primary_values, control_values),
        ("Correlated delayed CSI", "Independent-current control"),
    ):
        positions = np.arange(len(keys))
        axis.bar(positions, values, color=colors, width=0.76)
        axis.set_xticks(positions, labels, rotation=24, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean spectral efficiency (bit/s/Hz/subcarrier)")
    figure.suptitle("Predictive allocation from two delayed CSI snapshots")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=180)
    figure.savefig(output_dir / "comparison.svg")
    plt.close(figure)


def heldout(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "heldout_results.json"
    freeze = _verify_freeze(output_dir)
    if result_path.is_file():
        if not bool(freeze.get("heldout_outcomes_accessed")):
            raise StudyError("held-out result exists but the access state is false; refusing to trust it")
        expected_result_hash = freeze.get("heldout_result_sha256")
        if not expected_result_hash:
            raise StudyError("held-out access is recorded without a completed result hash; refusing rerun")
        if expected_result_hash != sha256_file(result_path):
            raise StudyError("held-out result hash disagrees with the recorded access state")
        report = read_json(result_path)
        if report.get("kind") != "delayed_csi_standalone_heldout_result":
            raise StudyError("held-out result has an unexpected kind")
        if report.get("config_sha256") != config_sha256():
            raise StudyError("held-out result configuration hash mismatch")
        if report.get("script_sha256") != freeze.get("script_sha256"):
            raise StudyError("held-out result script hash mismatch")
        if report.get("model_sha256") != freeze.get("model_sha256"):
            raise StudyError("held-out result model hash mismatch")
        print(f"held-out result already exists; not rerunning: {result_path}")
        _print_summary(report)
        return report
    if bool(freeze.get("heldout_outcomes_accessed")):
        raise StudyError("held-out access was already recorded but its result is missing; refusing rerun")
    freeze_accessed = dict(freeze)
    freeze_accessed["heldout_outcomes_accessed"] = True
    freeze_accessed["heldout_status"] = "running"
    write_json(output_dir / "final_design_freeze.json", freeze_accessed)
    freeze = freeze_accessed
    model = _model_from_checkpoint(output_dir / "model.pt")
    config = CONFIG["heldout"]
    primary_batch = _trajectory_batch(
        seed_start=int(config["trajectory_seed_start"]),
        trajectories=int(config["trajectories"]),
        decisions=int(config["decisions_per_trajectory"]),
    )
    control_batch = _trajectory_batch(
        seed_start=int(config["trajectory_seed_start"]),
        trajectories=int(config["trajectories"]),
        decisions=int(config["decisions_per_trajectory"]),
        independent_current_seed_start=int(config["independent_control_seed_start"]),
    )
    primary = evaluate_batch(
        model,
        primary_batch,
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
    )
    control = evaluate_batch(
        model,
        control_batch,
        bootstrap_seed=int(config["bootstrap_seed"]) + 500000,
        bootstrap_resamples=int(config["bootstrap_resamples"]),
    )
    equal_lower = primary["paired_contrasts"]["learned_minus_equal_power"][
        "one_sided_bonferroni_97p5_lower"
    ]
    stale_lower = primary["paired_contrasts"]["learned_minus_stale_csi_water_filling"][
        "one_sided_bonferroni_97p5_lower"
    ]
    threshold = float(config["minimum_simultaneous_lower_bound"])
    control_upper = control["paired_contrasts"]["learned_minus_equal_power"][
        "two_sided_95_upper"
    ]
    decision = (
        "passed"
        if equal_lower > threshold
        and stale_lower > threshold
        and control_upper <= float(
            CONFIG["development"]["maximum_independent_control_gain_over_equal"]
        )
        and primary["correlations"]["delayed_current_gain_correlation"] > 0.5
        and abs(control["correlations"]["delayed_current_gain_correlation"]) <= 0.05
        and primary["constraints"]["maximum_sum_power_error"] <= 1e-5
        and primary["constraints"]["maximum_negative_power"] <= 1e-12
        and primary["constraints"]["maximum_current_csi_oracle_dominance_violation"] <= 1e-9
        and control["constraints"]["maximum_sum_power_error"] <= 1e-5
        and control["constraints"]["maximum_negative_power"] <= 1e-12
        else "did_not_pass"
    )
    report = {
        "kind": "delayed_csi_standalone_heldout_result",
        "schema_version": 1,
        "config_sha256": config_sha256(),
        "script_sha256": freeze["script_sha256"],
        "model_sha256": freeze["model_sha256"],
        "primary_claim_decision": decision,
        "primary_claim": (
            "The learned causal allocator exceeds equal power and water filling on the newest "
            "delayed/noisy CSI report by more than 0.03 bit/s/Hz/subcarrier under both "
            "Bonferroni-adjusted one-sided lower confidence bounds."
        ),
        "scope_note": (
            "Controlled two-path OFDM mechanism case; this is not a field, NR-codec, or "
            "deployment benchmark. Current-CSI water filling is a noncausal oracle and is "
            "not a comparator the learned policy can legitimately beat."
        ),
        "prior_development_disclosure": freeze["prior_development_disclosure"],
        "primary_evaluation": primary,
        "independent_current_control": control,
    }
    write_json(result_path, report)
    _plot_results(report, output_dir)
    freeze_updated = dict(freeze)
    freeze_updated["heldout_status"] = "completed"
    freeze_updated["heldout_result_sha256"] = sha256_file(result_path)
    write_json(output_dir / "final_design_freeze.json", freeze_updated)
    _print_summary(report)
    print(f"primary_claim_decision={decision}")
    print(f"saved held-out result: {result_path}")
    return report


def self_test() -> None:
    response, gains, transition = state_tables()
    n = int(CONFIG["channel"]["subcarriers"])
    assert response.shape == (64, n)
    assert gains.shape == (64, n)
    assert transition.shape == (128, 128)
    assert np.allclose(np.sum(transition, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.mean(gains, axis=1), 1.0, atol=1e-12)
    random_gains = np.random.default_rng(7).lognormal(size=(20, n))
    power = water_filling(random_gains)
    assert np.max(np.abs(np.sum(power, axis=1) - float(CONFIG["endpoint"]["total_power"]))) < 1e-9
    equal = np.ones_like(power)
    assert np.min(spectral_efficiency(random_gains, power) - spectral_efficiency(random_gains, equal)) > -1e-10
    history, _ = sample_independent_windows(12, 9001)
    model = DelayedCsiPolicy()
    first = learned_power(model, history)
    unrelated_current = np.random.default_rng(9002).normal(size=(12, n))
    unrelated_current[:] *= 1000.0
    second = learned_power(model, history.copy())
    assert np.array_equal(first, second), unrelated_current.shape
    assert np.max(np.abs(np.sum(first, axis=1) - float(CONFIG["endpoint"]["total_power"]))) < 1e-5
    causal_batch = _trajectory_batch(seed_start=9010, trajectories=2, decisions=4)
    causal_before = _method_powers(model, causal_batch)["learned_predictive_allocator"]
    mutated_batch = dict(causal_batch)
    mutated_batch["current_gains"] = np.flip(causal_batch["current_gains"], axis=-1).copy() * 1000.0
    mutated_batch["current_states"] = np.flip(causal_batch["current_states"], axis=-1).copy()
    causal_after = _method_powers(model, mutated_batch)["learned_predictive_allocator"]
    assert np.array_equal(causal_before, causal_after)
    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("self-test", "train", "development", "freeze-final", "heldout", "all", "show"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.command in {"freeze-final", "heldout", "all", "show"} and output_dir != DEFAULT_OUTPUT.resolve():
        raise StudyError(
            "final freeze and held-out execution are restricted to the canonical output directory: "
            f"{DEFAULT_OUTPUT}"
        )
    if args.command == "self-test":
        self_test()
    elif args.command == "train":
        train(output_dir)
    elif args.command == "development":
        development(output_dir)
    elif args.command == "freeze-final":
        freeze_final(output_dir)
    elif args.command == "heldout":
        heldout(output_dir)
    elif args.command == "all":
        self_test()
        train(output_dir)
        development(output_dir)
        freeze_final(output_dir)
        heldout(output_dir)
    elif args.command == "show":
        if not (output_dir / "heldout_results.json").is_file():
            raise StudyError("no completed held-out result is available to show")
        result = heldout(output_dir)
        print(f"primary_claim_decision={result['primary_claim_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
