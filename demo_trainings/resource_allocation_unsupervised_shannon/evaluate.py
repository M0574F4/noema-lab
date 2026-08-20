from __future__ import annotations

import hashlib
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from datamodule import load_capture_dataset
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy_session(manifest_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("evaluation requires onnxruntime; install requirements.txt") from exc
    manifest = load_strict_yaml_or_json(manifest_path)
    if not isinstance(manifest, dict) or int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    component = next(
        (dict(row) for row in manifest.get("components") or [] if str(row.get("id") or "") == "policy"),
        None,
    )
    if not component or str(component.get("format") or "").lower() != "onnx":
        raise ValueError("trained artifact is missing its policy ONNX component")
    policy_path = (manifest_path.parent / str(component.get("path") or "")).resolve()
    actual_sha256 = _sha256(policy_path) if policy_path.is_file() else ""
    if not policy_path.is_file() or actual_sha256 != str(component.get("sha256") or ""):
        raise ValueError("policy ONNX component failed SHA-256 verification")
    evidence = {
        "id": str(component.get("id") or ""),
        "role": str(component.get("role") or ""),
        "path": str(component.get("path") or ""),
        "format": str(component.get("format") or ""),
        "sha256": str(component.get("sha256") or ""),
        "actual_sha256": actual_sha256,
    }
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    return manifest, session, evidence


def _project_power_scores(scores: np.ndarray, average_power_budget: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("allocation_scores must be a finite [batch, subcarrier] array")
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = ordered - cumulative / indices > 0.0
    rho = np.maximum(np.sum(active, axis=1) - 1, 0)
    theta = cumulative[np.arange(values.shape[0]), rho] / (rho + 1.0)
    fractions = np.maximum(values - theta[:, None], 0.0)
    fractions /= np.maximum(np.sum(fractions, axis=1, keepdims=True), 1e-12)
    return fractions * (float(average_power_budget) * float(values.shape[1]))


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    evaluation = dict(config.get("evaluation") or {})
    dataset = load_capture_dataset(
        data.get("test_capture_dirs") or [],
        feature_tap=str(data.get("feature_tap") or "channel_gains"),
        expected_split="test",
    )
    manifest_path = Path(
        str((config.get("training") or {}).get("artifact_manifest_path") or "trained_artifact.yaml")
    )
    manifest, policy_session, policy_evidence = _load_policy_session(manifest_path)
    budgets = [float(value) for value in evaluation.get("average_power_budgets", [0.5, 1.0, 2.0])]
    noises = [float(value) for value in evaluation.get("noise_variances", [0.2])]
    gains = dataset.gains.astype(np.float64)
    rows = []
    for budget, noise in product(budgets, noises):
        if budget <= 0.0 or noise <= 0.0:
            raise ValueError("Evaluation budgets and noise variances must be positive")
        policy_inputs = {
            "channel_gain": np.ascontiguousarray(gains, dtype=np.float32),
            "noise_variance": np.full((gains.shape[0], 1), noise, dtype=np.float32),
            "average_power_budget": np.full((gains.shape[0], 1), budget, dtype=np.float32),
        }
        start = time.perf_counter()
        scores = policy_session.run(["allocation_scores"], policy_inputs)[0]
        learned = _project_power_scores(scores, budget)
        learned_seconds = time.perf_counter() - start
        start = time.perf_counter()
        oracle = np.stack([_water_filling(row, noise, budget * gains.shape[1]) for row in gains], axis=0)
        oracle_seconds = time.perf_counter() - start
        rows.append(
            _metrics(
                gains,
                learned,
                oracle,
                noise,
                budget,
                learned_seconds,
                oracle_seconds,
            )
        )
    report = {
        "schema_version": 1,
        "kind": "noema.resource_allocation_held_out_evaluation",
        "oracle_used_during_training": False,
        "trained_artifact": {
            "path": str(manifest_path),
            "schema_version": int(manifest.get("schema_version") or 0),
            "manifest_sha256": _sha256(manifest_path),
            "contract": dict(manifest.get("contract") or {}),
            "components": [policy_evidence],
        },
        "training_provenance": dict(manifest.get("training") or {}),
        "test_capture_schema_sha256": list(dataset.capture_sha256),
        "operating_points": rows,
    }
    Path("evaluation_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for row in rows:
        print(
            "budget=%g noise=%g learned_se=%.7g oracle_se=%.7g relative_regret=%.4g active_f1=%.4g"
            % (
                row["average_power_budget"],
                row["noise_variance"],
                row["learned_spectral_efficiency_bps_hz"],
                row["water_filling_spectral_efficiency_bps_hz"],
                row["relative_rate_regret"],
                row["active_subcarrier_f1"],
            )
        )
    print("saved held-out metrics: evaluation_metrics.json")
    return 0


def _metrics(gains, learned, oracle, noise, budget, learned_seconds, oracle_seconds) -> dict:
    learned_state_rate = np.mean(np.log2(1.0 + gains * learned / noise), axis=1)
    oracle_state_rate = np.mean(np.log2(1.0 + gains * oracle / noise), axis=1)
    regret = oracle_state_rate - learned_state_rate
    relative = regret / np.maximum(oracle_state_rate, 1e-12)
    active_threshold = max(1e-8, budget * 1e-6)
    learned_active = learned > active_threshold
    oracle_active = oracle > active_threshold
    true_positive = float(np.sum(learned_active & oracle_active))
    false_positive = float(np.sum(learned_active & ~oracle_active))
    false_negative = float(np.sum(~learned_active & oracle_active))
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    total_power = budget * gains.shape[1]
    budget_error = np.abs(np.sum(learned, axis=1) - total_power)
    denominator = max(float(np.sqrt(np.mean(oracle**2))), 1e-12)
    kkt = [_kkt_residual(gains[index], learned[index], noise, active_threshold) for index in range(gains.shape[0])]
    confidence = 1.96 * float(np.std(regret, ddof=1)) / math.sqrt(max(gains.shape[0], 1)) if gains.shape[0] > 1 else 0.0
    return {
        "average_power_budget": float(budget),
        "noise_variance": float(noise),
        "sample_count": int(gains.shape[0]),
        "learned_spectral_efficiency_bps_hz": float(np.mean(learned_state_rate)),
        "water_filling_spectral_efficiency_bps_hz": float(np.mean(oracle_state_rate)),
        "capacity_ratio_learned_to_water_filling": float(np.mean(learned_state_rate) / max(np.mean(oracle_state_rate), 1e-12)),
        "absolute_rate_regret_bps_hz": float(np.mean(regret)),
        "absolute_rate_regret_95pct_ci_half_width": confidence,
        "relative_rate_regret": float(np.mean(relative)),
        "rate_regret_p05_bps_hz": float(np.percentile(regret, 5)),
        "rate_regret_p50_bps_hz": float(np.percentile(regret, 50)),
        "rate_regret_p95_bps_hz": float(np.percentile(regret, 95)),
        "allocation_power_normalized_rmse": float(np.sqrt(np.mean((learned - oracle) ** 2)) / denominator),
        "active_subcarrier_precision": precision,
        "active_subcarrier_recall": recall,
        "active_subcarrier_f1": f1,
        "learned_active_subcarrier_fraction": float(np.mean(learned_active)),
        "water_filling_active_subcarrier_fraction": float(np.mean(oracle_active)),
        "max_power_budget_error": float(np.max(budget_error)),
        "max_negative_power_violation": float(np.max(np.maximum(-learned, 0.0))),
        "mean_kkt_normalized_residual": float(np.mean(kkt)),
        "learned_inference_seconds_per_state": float(learned_seconds / gains.shape[0]),
        "water_filling_seconds_per_state": float(oracle_seconds / gains.shape[0]),
    }


def _water_filling(gains: np.ndarray, noise: float, total_power: float) -> np.ndarray:
    floors = float(noise) / np.maximum(np.asarray(gains, dtype=np.float64), 1e-12)
    sorted_floors = np.sort(floors)
    water_level = 0.0
    for count in range(1, floors.size + 1):
        candidate = (float(total_power) + float(np.sum(sorted_floors[:count]))) / float(count)
        if count == floors.size or candidate <= sorted_floors[count]:
            water_level = candidate
            break
    power = np.maximum(water_level - floors, 0.0)
    power *= float(total_power) / max(float(np.sum(power)), 1e-12)
    return power


def _kkt_residual(gains: np.ndarray, power: np.ndarray, noise: float, threshold: float) -> float:
    marginal = gains / (math.log(2.0) * (float(noise) + gains * power))
    active = power > threshold
    if not np.any(active):
        return float(np.max(marginal))
    common = float(np.mean(marginal[active]))
    active_error = float(np.max(np.abs(marginal[active] - common)))
    inactive_error = float(np.max(np.maximum(marginal[~active] - common, 0.0))) if np.any(~active) else 0.0
    return max(active_error, inactive_error) / max(abs(common), 1e-12)


if __name__ == "__main__":
    raise SystemExit(main())
