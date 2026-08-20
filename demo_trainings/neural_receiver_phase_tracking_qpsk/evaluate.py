from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from datamodule import held_out_split_integrity_report, load_capture_dataset
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data_config = dict(config.get("data") or {})
    recipe_config = dict(config.get("recipe") or {})
    training = dict(config.get("training") or {})
    evaluation_config = dict(config.get("evaluation") or {})
    pll_alpha = float(evaluation_config.get("pll_alpha", 0.12))
    pll_beta = float(evaluation_config.get("pll_beta", 0.005))
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    manifest = load_strict_yaml_or_json(manifest_path)
    if int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    component = dict((manifest.get("components") or [])[0])
    component_path = manifest_path.parent / str(component.get("path") or "")
    component_sha = hashlib.sha256(component_path.read_bytes()).hexdigest()
    if component_sha != str(component.get("sha256") or ""):
        raise ValueError("phase-tracking component SHA-256 does not match the manifest")

    dataset = load_capture_dataset(
        data_config.get("test_capture_dirs") or [],
        feature_tap=str(data_config.get("feature_tap") or "rx_symbols"),
        pilot_context_tap=str(
            data_config.get("pilot_context_tap") or "pilot_context"
        ),
        target_tap=str(data_config.get("target_tap") or "target_bits"),
        phase_truth_tap=str(data_config.get("phase_truth_tap") or ""),
        feature_reference=str(
            data_config.get("feature_reference")
            or recipe_config.get("feature_reference")
            or ""
        ),
        pilot_context_reference=str(
            data_config.get("pilot_context_reference")
            or recipe_config.get("pilot_context_reference")
            or ""
        ),
        target_reference=str(
            data_config.get("target_reference")
            or recipe_config.get("target_reference")
            or ""
        ),
        phase_truth_reference=str(
            data_config.get("phase_truth_reference") or ""
        ),
        pilot_smoothing_neighbors=int(
            data_config.get("pilot_smoothing_neighbors") or 5
        ),
        expected_split="test",
    )
    held_out_integrity = held_out_split_integrity_report(
        dataset,
        dict(dict(manifest.get("training") or {}).get("split_integrity") or {}),
    )
    session = ort.InferenceSession(
        str(component_path),
        providers=["CPUExecutionProvider"],
    )
    learned_outputs = session.run(
        ["residual_phase_rad"],
        {
            "receiver_features_v3": dataset.receiver_features.astype(
                np.float32,
                copy=False,
            )
        },
    )
    learned_residual_phase = _validated_learned_outputs(
        learned_outputs,
        expected_phase_shape=dataset.data_mask.shape,
    )

    raw = _raw_symbols(dataset.receiver_features)
    interpolation_phase = _pilot_interpolation_phase(dataset.receiver_features)
    smoothing_phase = _pilot_smoothing_phase(dataset.receiver_features)
    learned_phase = np.unwrap(
        smoothing_phase + learned_residual_phase,
        axis=1,
    )
    learned_corrected = raw * np.exp(-1j * learned_phase)
    learned_llr = _analytic_qpsk_llr(learned_corrected, dataset.snr_db)
    pll_phase = _decision_directed_pll_phase(
        dataset.receiver_features,
        alpha=pll_alpha,
        beta=pll_beta,
    )
    methods = {
        "uncompensated_qpsk": _summary(
            _hard_decisions(raw),
            dataset.target_bits,
            dataset.data_mask,
        ),
        "pilot_interpolation": _summary(
            _hard_decisions(raw * np.exp(-1j * interpolation_phase)),
            dataset.target_bits,
            dataset.data_mask,
        ),
        "pilot_smoothing": _summary(
            _hard_decisions(raw * np.exp(-1j * smoothing_phase)),
            dataset.target_bits,
            dataset.data_mask,
        ),
        "decision_directed_pll": _summary(
            _hard_decisions(raw * np.exp(-1j * pll_phase)),
            dataset.target_bits,
            dataset.data_mask,
        ),
        "learned_receiver": _summary(
            learned_llr < 0.0,
            dataset.target_bits,
            dataset.data_mask,
        ),
    }
    if dataset.phase_truth is not None:
        methods["oracle_phase"] = _summary(
            _hard_decisions(raw * np.exp(-1j * dataset.phase_truth)),
            dataset.target_bits,
            dataset.data_mask,
        )
        methods["pilot_interpolation"]["circular_phase_rmse_rad"] = _phase_rmse(
            interpolation_phase,
            dataset.phase_truth,
        )
        methods["pilot_smoothing"]["circular_phase_rmse_rad"] = _phase_rmse(
            smoothing_phase,
            dataset.phase_truth,
        )
        methods["decision_directed_pll"]["circular_phase_rmse_rad"] = _phase_rmse(
            pll_phase,
            dataset.phase_truth,
        )
        methods["learned_receiver"]["circular_phase_rmse_rad"] = _phase_rmse(
            learned_phase,
            dataset.phase_truth,
        )

    data_mask = dataset.data_mask[..., None]
    targets = dataset.target_bits.astype(np.float32, copy=False)
    logits_for_one = -learned_llr
    losses = (
        np.maximum(logits_for_one, 0.0)
        - logits_for_one * targets
        + np.log1p(np.exp(-np.abs(logits_for_one)))
    )
    learned_bce = float(np.sum(losses * data_mask) / np.sum(data_mask) / 2.0)
    metrics = {
        "split": "test",
        "bit_error_rate": methods["learned_receiver"]["bit_error_rate"],
        "binary_cross_entropy": learned_bce,
        "methods": methods,
        "comparison_scope": {
            "channel": (
                "pilot-aided QPSK with unknown packet phase, residual CFO, "
                "Wiener phase noise, and AWGN"
            ),
            "oracle_role": "diagnostic upper bound, not a deployable competitor",
            "learned_runtime_inputs": ["rx_symbols", "pilot_context"],
            "learned_output": "residual_phase_relative_to_pilot_smoothing",
            "phase_truth_forwarded_to_learned_runtime": False,
        },
        "split_integrity": held_out_integrity,
        "component_sha256": component_sha,
        "test_capture_schema_sha256": list(dataset.capture_schema_sha256),
    }
    per_snr = _per_snr_metrics(
        dataset=dataset,
        learned_decisions=learned_llr < 0.0,
        raw=raw,
        interpolation_phase=interpolation_phase,
        smoothing_phase=smoothing_phase,
        pll_phase=pll_phase,
    )
    if per_snr:
        metrics["per_snr"] = per_snr
    Path("evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _validated_learned_outputs(
    outputs,
    *,
    expected_phase_shape: tuple[int, ...],
) -> np.ndarray:
    if len(outputs) != 1:
        raise ValueError(
            "artifact returned %d outputs, expected residual_phase_rad"
            % len(outputs)
        )
    learned_residual_phase = np.asarray(outputs[0], dtype=np.float32)
    if learned_residual_phase.shape != expected_phase_shape:
        raise ValueError(
            "artifact returned residual_phase_rad %s, expected %s"
            % (learned_residual_phase.shape, expected_phase_shape)
        )
    if not np.all(np.isfinite(learned_residual_phase)):
        raise ValueError("artifact returned non-finite residual_phase_rad values")
    return learned_residual_phase


def _per_snr_metrics(
    *,
    dataset,
    learned_decisions: np.ndarray,
    raw: np.ndarray,
    interpolation_phase: np.ndarray,
    smoothing_phase: np.ndarray,
    pll_phase: np.ndarray,
) -> list[dict[str, object]]:
    if dataset.snr_db is None:
        return []
    values = np.asarray(dataset.snr_db, dtype=np.float64)
    if values.shape != (dataset.target_bits.shape[0],):
        return []
    rows = []
    for snr in sorted(float(value) for value in np.unique(values)):
        selected = values == snr
        methods = {
            "uncompensated_qpsk": _summary(
                _hard_decisions(raw[selected]),
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            ),
            "pilot_interpolation": _summary(
                _hard_decisions(
                    raw[selected] * np.exp(-1j * interpolation_phase[selected])
                ),
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            ),
            "pilot_smoothing": _summary(
                _hard_decisions(
                    raw[selected] * np.exp(-1j * smoothing_phase[selected])
                ),
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            ),
            "decision_directed_pll": _summary(
                _hard_decisions(raw[selected] * np.exp(-1j * pll_phase[selected])),
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            ),
            "learned_receiver": _summary(
                learned_decisions[selected],
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            ),
        }
        if dataset.phase_truth is not None:
            methods["oracle_phase"] = _summary(
                _hard_decisions(
                    raw[selected]
                    * np.exp(-1j * dataset.phase_truth[selected])
                ),
                dataset.target_bits[selected],
                dataset.data_mask[selected],
            )
        rows.append({"snr_db": snr, "methods": methods})
    return rows


def _raw_symbols(features: np.ndarray) -> np.ndarray:
    return features[..., 2].astype(np.float32) + 1j * features[..., 3].astype(
        np.float32
    )


def _hard_decisions(symbols: np.ndarray) -> np.ndarray:
    value = np.asarray(symbols)
    return np.stack((value.real < 0.0, value.imag < 0.0), axis=-1)


def _pilot_interpolation_phase(features: np.ndarray) -> np.ndarray:
    mask = features[..., 4] > 0.5
    smoothing_phasor = features[..., 7] + 1j * features[..., 8]
    observations = (features[..., 5] + 1j * features[..., 6]) * smoothing_phasor
    result = np.zeros(mask.shape, dtype=np.float32)
    sample_axis = np.arange(mask.shape[1], dtype=np.float64)
    for packet in range(mask.shape[0]):
        indices = np.flatnonzero(mask[packet])
        if not len(indices):
            continue
        phases = np.unwrap(np.angle(observations[packet, indices]))
        result[packet] = np.interp(
            sample_axis,
            indices.astype(np.float64),
            phases.astype(np.float64),
        ).astype(np.float32)
    return result


def _pilot_smoothing_phase(features: np.ndarray) -> np.ndarray:
    phasor = features[..., 7] + 1j * features[..., 8]
    return np.unwrap(np.angle(phasor), axis=1)


def _analytic_qpsk_llr(
    corrected: np.ndarray,
    snr_db: np.ndarray | None,
) -> np.ndarray:
    value = np.asarray(corrected)
    if snr_db is None:
        scale = np.ones((value.shape[0], 1), dtype=np.float32)
    else:
        noise_variance = np.power(
            10.0,
            -np.asarray(snr_db, dtype=np.float64) / 10.0,
        )
        scale = (2.0 * math.sqrt(2.0) / noise_variance).reshape(-1, 1)
    return np.stack(
        (value.real * scale, value.imag * scale),
        axis=-1,
    ).astype(np.float32, copy=False)


def _decision_directed_pll_phase(
    features: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("PLL alpha and beta must lie in [0, 1]")
    raw = _raw_symbols(features)
    mask = features[..., 4] > 0.5
    smoothing_phasor = features[..., 7] + 1j * features[..., 8]
    pilot_observation = (
        features[..., 5] + 1j * features[..., 6]
    ) * smoothing_phasor
    estimates = np.empty(raw.shape, dtype=np.float64)
    for packet in range(raw.shape[0]):
        pilot_indices = np.flatnonzero(mask[packet])
        preamble_end = 1
        while (
            preamble_end < pilot_indices.size
            and pilot_indices[preamble_end] == pilot_indices[preamble_end - 1] + 1
        ):
            preamble_end += 1
        init_indices = pilot_indices[:preamble_end]
        init_phase = np.unwrap(
            np.angle(pilot_observation[packet, init_indices])
        )
        if init_indices.size >= 2:
            slope, intercept = np.polyfit(
                init_indices.astype(np.float64),
                init_phase,
                1,
            )
        else:
            slope, intercept = 0.0, float(init_phase[0])
        phase_state = float(intercept)
        frequency_state = float(slope)
        for index in range(raw.shape[1]):
            predicted = (
                phase_state
                if index == 0
                else phase_state + frequency_state
            )
            corrected = raw[packet, index] * np.exp(-1j * predicted)
            if mask[packet, index]:
                error = float(
                    np.angle(
                        pilot_observation[packet, index]
                        * np.exp(-1j * predicted)
                    )
                )
            else:
                decision = (
                    (1.0 if corrected.real >= 0.0 else -1.0)
                    + 1j * (1.0 if corrected.imag >= 0.0 else -1.0)
                ) / math.sqrt(2.0)
                error = float(np.angle(corrected * np.conj(decision)))
            phase_state = predicted + alpha * error
            frequency_state = frequency_state + beta * error
            estimates[packet, index] = phase_state
    return estimates


def _summary(
    decisions: np.ndarray,
    targets: np.ndarray,
    data_mask: np.ndarray,
) -> dict[str, object]:
    selected_decisions = np.asarray(decisions)[np.asarray(data_mask)]
    selected_targets = np.asarray(targets).astype(bool)[np.asarray(data_mask)]
    errors = int(np.count_nonzero(selected_decisions != selected_targets))
    bit_count = int(selected_targets.size)
    ber = float(errors / max(1, bit_count))
    lower, upper = _wilson_interval(errors, bit_count)
    return {
        "bit_errors": errors,
        "bit_count": bit_count,
        "bit_error_rate": ber,
        "ber_95_percent_wilson_interval": [lower, upper],
    }


def _wilson_interval(errors: int, count: int) -> tuple[float, float]:
    if count < 1:
        return 0.0, 1.0
    z = 1.959963984540054
    probability = float(errors / count)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _phase_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    error = np.angle(np.exp(1j * (np.asarray(estimate) - np.asarray(truth))))
    return float(np.sqrt(np.mean(error * error)))


if __name__ == "__main__":
    raise SystemExit(main())
