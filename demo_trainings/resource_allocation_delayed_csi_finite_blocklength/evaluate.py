from __future__ import annotations

import hashlib
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from datamodule import (
    held_out_split_integrity_report,
    load_capture_dataset,
    paired_cluster_evidence,
)

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


METHOD_LABELS = {
    "equal_power": "Equal power",
    "observed_csi_water_filling": "Water filling on delayed/noisy CSI",
    "robust_csi_water_filling": "Uncertainty-shrunk water filling",
    "causal_ar_water_filling": "Complex-AR prediction + water filling",
    "learned_allocator": "Learned reliability-aware allocator",
    "current_csi_shannon_diagnostic": (
        "Current-CSI Shannon water filling (diagnostic only)"
    ),
    "current_csi_finite_blocklength_numerical_diagnostic": (
        "Perfect-current-CSI finite-blocklength numerical reference"
    ),
}


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    if not isinstance(config, dict):
        raise ValueError("train_config.yaml must contain a mapping")
    data = dict(config.get("data") or {})
    objective = dict(config.get("objective") or {})
    evaluation = dict(config.get("evaluation") or {})
    training = dict(config.get("training") or {})
    delayed_tap = str(
        data.get("csi_history_tap")
        or data.get("delayed_csi_tap")
        or data.get("feature_tap")
        or "csi_history"
    )
    current_tap = str(data.get("current_csi_tap") or "current_csi")
    dataset = load_capture_dataset(
        data.get("test_capture_dirs") or [],
        delayed_csi_tap=delayed_tap,
        current_csi_tap=current_tap,
        expected_split="test",
    )
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    manifest, session, artifact_evidence = _load_policy_session(manifest_path)
    training_provenance = dict(manifest.get("training") or {})
    integrity = held_out_split_integrity_report(
        dataset,
        dict(training_provenance.get("split_integrity") or {}),
    )

    blocklength = int(objective.get("blocklength_channel_uses") or 128)
    target_rate = float(objective.get("target_rate_bps_hz") or 2.0)
    third_order = bool(objective.get("include_third_order_term", True))
    budgets = _positive_values(
        evaluation.get("average_power_budgets", [0.4, 0.8, 1.4])
    )
    noises = _positive_values(evaluation.get("noise_variances", [0.2]))
    shrinkage = float(evaluation.get("robust_gain_shrinkage", 0.6))
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("evaluation.robust_gain_shrinkage must be in [0,1]")
    prediction_horizon = int(
        evaluation.get("csi_prediction_horizon_ofdm_symbols", 5)
    )
    prediction_confidence = float(
        evaluation.get("csi_prediction_gain_confidence", 0.4)
    )
    if prediction_horizon < 0:
        raise ValueError(
            "evaluation.csi_prediction_horizon_ofdm_symbols must be nonnegative"
        )
    if (
        not math.isfinite(prediction_confidence)
        or not 0.0 <= prediction_confidence <= 1.0
    ):
        raise ValueError(
            "evaluation.csi_prediction_gain_confidence must be in [0,1]"
        )

    csi_history = dataset.csi_history.astype(np.float32)
    observed = dataset.latest_observed_gains.astype(np.float64)
    actual = dataset.current_gains.astype(np.float64)
    rows = []
    state_goodput_by_method: dict[str, list[np.ndarray]] = {
        method_id: [] for method_id in METHOD_LABELS
    }
    operating_points = [
        {
            "average_power_budget": float(budget),
            "noise_variance": float(noise),
        }
        for budget, noise in product(budgets, noises)
    ]
    for budget, noise in product(budgets, noises):
        powers, latencies = _policies(
            session,
            csi_history,
            observed,
            actual,
            budget=budget,
            noise=noise,
            shrinkage=shrinkage,
            prediction_horizon=prediction_horizon,
            prediction_confidence=prediction_confidence,
            blocklength=blocklength,
            target_rate=target_rate,
            third_order=third_order,
        )
        for method_id, power in powers.items():
            row = _metrics(
                actual,
                power,
                noise=noise,
                budget=budget,
                blocklength=blocklength,
                target_rate=target_rate,
                third_order=third_order,
            )
            state_goodput_by_method[method_id].append(
                np.asarray(
                    row.pop("_state_goodput_bps_hz"),
                    dtype=np.float64,
                )
            )
            row.update(
                {
                    "method_id": method_id,
                    "method_label": METHOD_LABELS[method_id],
                    "role": (
                        "candidate"
                        if method_id == "learned_allocator"
                        else (
                            "diagnostic"
                            if method_id
                            in {
                                "current_csi_shannon_diagnostic",
                                "current_csi_finite_blocklength_numerical_diagnostic",
                            }
                            else "baseline"
                        )
                    ),
                    "inference_seconds_per_state": float(
                        latencies[method_id] / max(actual.shape[0], 1)
                    ),
                }
            )
            rows.append(row)

    learned_by_point = {
        (row["average_power_budget"], row["noise_variance"]): row
        for row in rows
        if row["method_id"] == "learned_allocator"
    }
    for row in rows:
        learned = learned_by_point[
            (row["average_power_budget"], row["noise_variance"])
        ]
        row["learned_minus_this_expected_goodput_bps_hz"] = float(
            learned["expected_goodput_bps_hz"]
            - row["expected_goodput_bps_hz"]
        )

    validation_evidence = dict(
        training_provenance.get("demo_evidence_status") or {}
    )
    criteria = dict(validation_evidence.get("criteria") or {})
    test_evidence = paired_cluster_evidence(
        np.asarray(
            state_goodput_by_method["learned_allocator"],
            dtype=np.float64,
        ),
        {
            method_id: np.asarray(
                state_goodput_by_method[method_id],
                dtype=np.float64,
            )
            for method_id in (
                "equal_power",
                "observed_csi_water_filling",
                "robust_csi_water_filling",
                "causal_ar_water_filling",
            )
        },
        cluster_ids=dataset.trajectory_cluster_ids,
        cluster_method=dataset.trajectory_cluster_method,
        operating_points=operating_points,
        minimum_relative_improvement=float(
            criteria.get("minimum_relative_improvement", 0.005)
        ),
        maximum_relative_point_regression=float(
            criteria.get("maximum_relative_point_regression", 0.002)
        ),
        confidence_level=float(criteria.get("confidence_level", 0.95)),
        minimum_cluster_count=int(criteria.get("minimum_cluster_count", 30)),
    )
    test_evidence["scope"] = "held_out_test_confirmation"
    test_evidence["validation_evidence_status"] = str(
        validation_evidence.get("status") or "missing"
    )

    report = {
        "schema_version": 1,
        "kind": "noema.delayed_csi_finite_blocklength_held_out_evaluation",
        "comparison_scope": {
            "channel": (
                "temporally correlated Sionna TDL OFDM with delayed/noisy "
                "transmitter CSI"
            ),
            "objective": (
                "normal-approximation expected goodput at fixed blocklength and rate"
            ),
            "blocklength_channel_uses": blocklength,
            "target_rate_bps_hz": target_rate,
            "runtime_inputs": [
                "causal_delayed_noisy_complex_csi_history",
                "noise_variance",
                "average_power_budget",
            ],
            "current_channel_forwarded_to_learned_runtime": False,
            "allocation_labels_used_during_training": False,
            "diagnostic_policy_note": (
                "Current-CSI water filling optimizes Shannon capacity. The "
                "separate perfect-current-CSI numerical reference directly "
                "optimizes the modeled finite-blocklength objective but is "
                "non-deployable and is not claimed to be a global optimum."
            ),
        },
        "trained_artifact": {
            "path": str(manifest_path),
            "schema_version": int(manifest.get("schema_version") or 0),
            "manifest_sha256": _sha256(manifest_path),
            "contract": dict(manifest.get("contract") or {}),
            "components": [artifact_evidence],
        },
        "training_provenance": training_provenance,
        "demo_evidence_status": test_evidence,
        "test_capture_schema_sha256": list(
            dataset.capture_schema_sha256
        ),
        "split_integrity": integrity,
        "operating_points": rows,
    }
    Path("evaluation_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for row in rows:
        print(
            "method=%s budget=%g noise=%g goodput=%.7g predicted_bler=%.7g"
            % (
                row["method_id"],
                row["average_power_budget"],
                row["noise_variance"],
                row["expected_goodput_bps_hz"],
                row["predicted_bler"],
            )
        )
    print("saved held-out metrics: evaluation_metrics.json")
    print(
        "demo_evidence_status=%s relative_improvement=%.4g ci_lower=%.4g"
        % (
            test_evidence["status"],
            test_evidence["relative_improvement"],
            test_evidence["paired_cluster_confidence_interval"][
                "lower_bps_hz"
            ],
        )
    )
    if test_evidence["status"] != "passed":
        for reason in test_evidence["reasons"]:
            print("evidence note: %s" % reason)
    return 0


def _policies(
    session,
    csi_history: np.ndarray,
    observed: np.ndarray,
    actual: np.ndarray,
    *,
    budget: float,
    noise: float,
    shrinkage: float,
    prediction_horizon: int,
    prediction_confidence: float,
    blocklength: int,
    target_rate: float,
    third_order: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    count, subcarriers = observed.shape
    total_power = float(budget) * float(subcarriers)
    powers = {}
    latencies = {}

    start = time.perf_counter()
    powers["equal_power"] = np.full(
        observed.shape,
        float(budget),
        dtype=np.float64,
    )
    latencies["equal_power"] = time.perf_counter() - start

    start = time.perf_counter()
    powers["observed_csi_water_filling"] = np.stack(
        [_water_filling(row, noise, total_power) for row in observed],
        axis=0,
    )
    latencies["observed_csi_water_filling"] = time.perf_counter() - start

    start = time.perf_counter()
    robust_gains = (
        float(shrinkage) * observed
        + (1.0 - float(shrinkage))
        * np.mean(observed, axis=1, keepdims=True)
    )
    powers["robust_csi_water_filling"] = np.stack(
        [_water_filling(row, noise, total_power) for row in robust_gains],
        axis=0,
    )
    latencies["robust_csi_water_filling"] = time.perf_counter() - start

    start = time.perf_counter()
    predicted_gains = _causal_complex_ar_predicted_gains(
        csi_history,
        horizon=prediction_horizon,
        confidence=prediction_confidence,
    )
    powers["causal_ar_water_filling"] = np.stack(
        [_water_filling(row, noise, total_power) for row in predicted_gains],
        axis=0,
    )
    latencies["causal_ar_water_filling"] = time.perf_counter() - start

    inputs = {
        "csi_history": np.ascontiguousarray(
            csi_history, dtype=np.float32
        ),
        "noise_variance": np.full((count, 1), noise, dtype=np.float32),
        "average_power_budget": np.full(
            (count, 1),
            budget,
            dtype=np.float32,
        ),
    }
    start = time.perf_counter()
    scores = session.run(["allocation_scores"], inputs)[0]
    powers["learned_allocator"] = _project_power_scores(scores, budget)
    latencies["learned_allocator"] = time.perf_counter() - start

    start = time.perf_counter()
    powers["current_csi_shannon_diagnostic"] = np.stack(
        [_water_filling(row, noise, total_power) for row in actual],
        axis=0,
    )
    latencies["current_csi_shannon_diagnostic"] = (
        time.perf_counter() - start
    )

    start = time.perf_counter()
    powers[
        "current_csi_finite_blocklength_numerical_diagnostic"
    ] = _finite_blocklength_current_csi_numerical_reference(
        actual,
        noise=noise,
        budget=budget,
        blocklength=blocklength,
        target_rate=target_rate,
        third_order=third_order,
        initial_power=powers["current_csi_shannon_diagnostic"],
    )
    latencies[
        "current_csi_finite_blocklength_numerical_diagnostic"
    ] = time.perf_counter() - start
    return powers, latencies


def _causal_complex_ar_predicted_gains(
    history_iq: np.ndarray,
    *,
    horizon: int,
    confidence: float,
) -> np.ndarray:
    history = np.asarray(history_iq, dtype=np.float64)
    complex_history = history[..., 0] + 1j * history[..., 1]
    previous = complex_history[:, :-1]
    following = complex_history[:, 1:]
    coefficient = (
        np.sum(following * np.conjugate(previous), axis=1)
        / (np.sum(np.abs(previous) ** 2, axis=1) + 1e-12)
    )
    coefficient = coefficient / np.maximum(np.abs(coefficient), 1.0)
    predicted = complex_history[:, -1] * np.power(
        coefficient,
        max(0, int(horizon)),
    )
    gains = np.maximum(np.abs(predicted) ** 2, 1e-12)
    mean_gain = np.mean(gains, axis=1, keepdims=True)
    return float(confidence) * gains + (1.0 - float(confidence)) * mean_gain


def _finite_blocklength_current_csi_numerical_reference(
    actual_gains: np.ndarray,
    *,
    noise: float,
    budget: float,
    blocklength: int,
    target_rate: float,
    third_order: bool,
    initial_power: np.ndarray,
) -> np.ndarray:
    """Numerically optimize the modeled objective with forbidden current CSI.

    This is a deterministic diagnostic reference, not a deployable policy and
    not a claim of global optimality. Multiple feasible initializations are
    optimized, and the best feasible allocation is retained independently for
    each current-channel state.
    """

    import torch

    gains = torch.as_tensor(
        np.asarray(actual_gains, dtype=np.float64),
        dtype=torch.float64,
    )
    count, subcarriers = gains.shape
    total_power = float(budget) * float(subcarriers)
    initializations = [
        np.asarray(initial_power, dtype=np.float64),
        np.full(
            (count, subcarriers),
            float(budget),
            dtype=np.float64,
        ),
    ]
    best_power = initializations[0].copy()
    best_score = _finite_blocklength_z_score_numpy(
        np.asarray(actual_gains, dtype=np.float64),
        best_power,
        noise=noise,
        blocklength=blocklength,
        target_rate=target_rate,
        third_order=third_order,
    )
    for initialization in initializations:
        logits = torch.nn.Parameter(
            torch.log(
                torch.as_tensor(
                    np.maximum(initialization, 1e-9),
                    dtype=torch.float64,
                )
            )
        )
        optimizer = torch.optim.Adam([logits], lr=0.06)
        for step in range(120):
            optimizer.zero_grad(set_to_none=True)
            power = torch.softmax(logits, dim=1) * total_power
            snr = gains * power / float(noise)
            capacity = torch.mean(torch.log2(1.0 + snr), dim=1)
            dispersion = torch.mean(
                (1.0 - torch.pow(1.0 + snr, -2.0))
                * (math.log2(math.e) ** 2),
                dim=1,
            )
            correction = (
                math.log2(float(blocklength))
                / (2.0 * float(blocklength))
                if third_order
                else 0.0
            )
            z = (
                (capacity - float(target_rate) + correction)
                * math.sqrt(float(blocklength))
                / torch.sqrt(torch.clamp(dispersion, min=1e-12))
            )
            (-torch.mean(z)).backward()
            optimizer.step()
            if step % 10 == 9 or step == 119:
                candidate_power = (
                    torch.softmax(logits.detach(), dim=1) * total_power
                ).cpu().numpy()
                candidate_score = _finite_blocklength_z_score_numpy(
                    np.asarray(actual_gains, dtype=np.float64),
                    candidate_power,
                    noise=noise,
                    blocklength=blocklength,
                    target_rate=target_rate,
                    third_order=third_order,
                )
                improved = candidate_score > best_score
                best_score[improved] = candidate_score[improved]
                best_power[improved] = candidate_power[improved]
    return best_power


def _finite_blocklength_z_score_numpy(
    gains: np.ndarray,
    power: np.ndarray,
    *,
    noise: float,
    blocklength: int,
    target_rate: float,
    third_order: bool,
) -> np.ndarray:
    snr = np.asarray(gains, dtype=np.float64) * np.asarray(
        power, dtype=np.float64
    ) / float(noise)
    capacity = np.mean(np.log2(1.0 + snr), axis=1)
    dispersion = np.mean(
        (1.0 - np.power(1.0 + snr, -2.0))
        * (math.log2(math.e) ** 2),
        axis=1,
    )
    correction = (
        math.log2(float(blocklength)) / (2.0 * float(blocklength))
        if third_order
        else 0.0
    )
    return (
        (capacity - float(target_rate) + correction)
        * math.sqrt(float(blocklength))
        / np.sqrt(np.maximum(dispersion, 1e-12))
    )


def _metrics(
    actual_gains: np.ndarray,
    power: np.ndarray,
    *,
    noise: float,
    budget: float,
    blocklength: int,
    target_rate: float,
    third_order: bool,
) -> dict[str, object]:
    snr = actual_gains * power / float(noise)
    capacity = np.mean(np.log2(1.0 + snr), axis=1)
    dispersion = np.mean(
        (1.0 - np.power(1.0 + snr, -2.0)) * (math.log2(math.e) ** 2),
        axis=1,
    )
    correction = (
        math.log2(float(blocklength)) / (2.0 * float(blocklength))
        if third_order
        else 0.0
    )
    z = (
        (capacity - float(target_rate) + correction)
        * math.sqrt(float(blocklength))
        / np.sqrt(np.maximum(dispersion, 1e-12))
    )
    bler = np.asarray(
        [
            0.5 * math.erfc(float(value) / math.sqrt(2.0))
            for value in z
        ],
        dtype=np.float64,
    )
    goodput = float(target_rate) * (1.0 - bler)
    total = float(budget) * float(power.shape[1])
    budget_error = np.abs(np.sum(power, axis=1) - total)
    return {
        "average_power_budget": float(budget),
        "noise_variance": float(noise),
        "sample_count": int(actual_gains.shape[0]),
        "expected_goodput_bps_hz": float(np.mean(goodput)),
        "predicted_bler": float(np.mean(bler)),
        "mean_capacity_bps_hz": float(np.mean(capacity)),
        "p05_goodput_bps_hz": float(np.percentile(goodput, 5.0)),
        "max_power_budget_error": float(np.max(budget_error)),
        "max_negative_power_violation": float(
            np.max(np.maximum(-power, 0.0))
        ),
        "_state_goodput_bps_hz": goodput,
    }


def _load_policy_session(manifest_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "evaluation requires onnxruntime; install requirements.txt"
        ) from exc
    manifest = load_strict_yaml_or_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or int(manifest.get("schema_version") or 0) != 2
    ):
        raise ValueError("trained artifact must use schema_version: 2")
    component = next(
        (
            dict(item)
            for item in manifest.get("components") or []
            if str(item.get("id") or "") == "policy"
        ),
        None,
    )
    if not component or str(component.get("format") or "").lower() != "onnx":
        raise ValueError("trained artifact is missing its policy ONNX component")
    path = (manifest_path.parent / str(component.get("path") or "")).resolve()
    actual_sha = _sha256(path) if path.is_file() else ""
    if not path.is_file() or actual_sha != str(component.get("sha256") or ""):
        raise ValueError("policy ONNX component failed SHA-256 verification")
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    expected_inputs = {
        "csi_history",
        "noise_variance",
        "average_power_budget",
    }
    actual_inputs = {item.name for item in session.get_inputs()}
    if actual_inputs != expected_inputs:
        raise ValueError(
            "policy runtime inputs are %s; expected delayed-CSI ABI %s"
            % (sorted(actual_inputs), sorted(expected_inputs))
        )
    evidence = {
        "id": str(component.get("id") or ""),
        "role": str(component.get("role") or ""),
        "path": str(component.get("path") or ""),
        "format": str(component.get("format") or ""),
        "sha256": str(component.get("sha256") or ""),
        "actual_sha256": actual_sha,
    }
    return manifest, session, evidence


def _project_power_scores(
    scores: np.ndarray,
    average_power_budget: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            "allocation_scores must be a finite [batch,subcarrier] array"
        )
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    rank = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = ordered - cumulative / rank > 0.0
    rho = np.maximum(np.sum(active, axis=1) - 1, 0)
    threshold = cumulative[np.arange(values.shape[0]), rho] / (rho + 1.0)
    fractions = np.maximum(values - threshold[:, None], 0.0)
    fractions /= np.maximum(np.sum(fractions, axis=1, keepdims=True), 1e-12)
    return fractions * (
        float(average_power_budget) * float(values.shape[1])
    )


def _water_filling(
    gains: np.ndarray,
    noise: float,
    total_power: float,
) -> np.ndarray:
    floors = float(noise) / np.maximum(
        np.asarray(gains, dtype=np.float64),
        1e-12,
    )
    ordered = np.sort(floors)
    water_level = 0.0
    for count in range(1, floors.size + 1):
        candidate = (
            float(total_power) + float(np.sum(ordered[:count]))
        ) / float(count)
        if count == floors.size or candidate <= ordered[count]:
            water_level = candidate
            break
    power = np.maximum(water_level - floors, 0.0)
    power *= float(total_power) / max(float(np.sum(power)), 1e-12)
    return power


def _positive_values(value) -> list[float]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    result = [float(item) for item in raw]
    if not result or any(
        not math.isfinite(item) or item <= 0.0 for item in result
    ):
        raise ValueError(
            "Operating-point values must be finite and greater than zero"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
