from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

from datamodule import (
    build_loader,
    feature_statistics,
    load_capture_dataset,
    paired_cluster_evidence,
    split_integrity_report,
)
from losses import (
    feasibility_metrics,
    finite_blocklength_state_metrics,
    negative_expected_finite_blocklength_goodput,
)
from model import (
    FrequencyResidualPowerAllocator,
    allocator_candidates,
    build_allocator,
    export_onnx_policy,
)

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


LOSS_ID = "resource.negative_expected_finite_blocklength_goodput"


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    if not isinstance(config, dict):
        raise ValueError("train_config.yaml must contain a mapping")
    data_config = dict(config.get("data") or {})
    objective = dict(config.get("objective") or {})
    training = dict(config.get("training") or {})
    evaluation = dict(config.get("evaluation") or {})
    if str(objective.get("loss") or "") != LOSS_ID:
        raise ValueError("This trainer requires objective.loss=%s" % LOSS_ID)
    if bool(
        objective.get(
            "uses_power_allocation_labels",
            objective.get("uses_allocation_labels", False),
        )
    ):
        raise ValueError("Power-allocation labels are not part of this demo")
    if bool(objective.get("uses_current_csi_at_runtime", False)):
        raise ValueError("Current CSI must never be a learned-policy runtime input")

    delayed_tap = str(
        data_config.get("csi_history_tap")
        or data_config.get("delayed_csi_tap")
        or data_config.get("feature_tap")
        or "csi_history"
    )
    current_tap = str(data_config.get("current_csi_tap") or "current_csi")
    if delayed_tap == current_tap:
        raise ValueError("Delayed and current CSI taps must be distinct")
    train_data = load_capture_dataset(
        data_config.get("train_capture_dirs") or [],
        delayed_csi_tap=delayed_tap,
        current_csi_tap=current_tap,
        expected_split="train",
    )
    validation_data = load_capture_dataset(
        data_config.get("validation_capture_dirs") or [],
        delayed_csi_tap=delayed_tap,
        current_csi_tap=current_tap,
        expected_split="validation",
    )
    if train_data.csi_history.shape[1:] != validation_data.csi_history.shape[1:]:
        raise ValueError(
            "Training and validation use different CSI-history shapes"
        )
    if train_data.current_gains.shape[1] != validation_data.current_gains.shape[1]:
        raise ValueError("Training and validation use different subcarrier counts")
    model_config = dict(config.get("model") or {})
    history_length = int(
        model_config.get("history_length")
        or data_config.get("csi_history_length")
        or train_data.csi_history.shape[1]
    )
    if history_length != int(train_data.csi_history.shape[1]):
        raise ValueError(
            "Configured model history_length=%d does not match captured "
            "history length %d"
            % (history_length, train_data.csi_history.shape[1])
        )
    integrity = split_integrity_report(train_data, validation_data)

    blocklength = int(objective.get("blocklength_channel_uses") or 128)
    target_rate = float(objective.get("target_rate_bps_hz") or 2.0)
    third_order = bool(objective.get("include_third_order_term", True))
    if blocklength < 16 or not math.isfinite(target_rate) or target_rate <= 0.0:
        raise ValueError(
            "Finite-blocklength training requires n>=16 and positive finite target rate"
        )
    budget_range = _positive_range(
        training.get("average_power_budget_range", [0.4, 1.4]),
        "average_power_budget_range",
    )
    noise_range = _positive_range(
        training.get("noise_variance_range", [0.2, 0.2]),
        "noise_variance_range",
    )
    validation_budgets = _positive_values(
        training.get(
            "validation_average_power_budgets",
            [budget_range[0], math.sqrt(budget_range[0] * budget_range[1]), budget_range[1]],
        )
    )
    validation_noises = _positive_values(
        training.get("validation_noise_variances", [noise_range[0]])
    )
    reference_budget = math.sqrt(budget_range[0] * budget_range[1])
    reference_noise = math.sqrt(noise_range[0] * noise_range[1])
    feature_mean, feature_scale = feature_statistics(
        train_data,
        reference_noise_variance=reference_noise,
        reference_average_power_budget=reference_budget,
    )

    device = _device(training)
    seeds = [int(value) for value in training.get("initialization_seeds", [23, 41])]
    if not seeds:
        raise ValueError("training.initialization_seeds must not be empty")
    candidates = allocator_candidates(model_config)
    epochs = max(1, int(training.get("epochs", 120)))
    batch_size = max(1, int(training.get("batch_size", 128)))
    workers = max(0, int(training.get("num_workers", 0)))
    patience = max(0, int(training.get("early_stopping_patience", 24)))
    minimum_relative_validation_improvement = float(
        training.get(
            "minimum_relative_validation_goodput_improvement",
            0.005,
        )
    )
    maximum_relative_point_regression = float(
        training.get(
            "maximum_relative_validation_point_regression",
            0.002,
        )
    )
    validation_confidence_level = float(
        training.get("validation_confidence_level", 0.95)
    )
    minimum_validation_cluster_count = int(
        training.get("minimum_validation_cluster_count", 30)
    )
    robust_gain_shrinkage = float(
        evaluation.get("robust_gain_shrinkage", 0.6)
    )
    csi_prediction_horizon = int(
        evaluation.get("csi_prediction_horizon_ofdm_symbols", 5)
    )
    csi_prediction_confidence = float(
        evaluation.get("csi_prediction_gain_confidence", 0.4)
    )
    operating_points = _operating_points(
        validation_budgets,
        validation_noises,
    )
    baseline_goodput_by_method = _deployable_baseline_goodputs(
        validation_data,
        operating_points=operating_points,
        blocklength=blocklength,
        target_rate=target_rate,
        third_order=third_order,
        robust_gain_shrinkage=robust_gain_shrinkage,
        csi_prediction_horizon=csi_prediction_horizon,
        csi_prediction_confidence=csi_prediction_confidence,
    )

    best_score = -float("inf")
    best_state = None
    best_candidate = None
    best_seed = None
    best_epoch = None
    best_evidence = None
    history_rows: list[dict] = []
    for candidate in candidates:
        for seed in seeds:
            _seed_everything(seed)
            model = build_allocator(
                candidate,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                history_length=history_length,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training.get("learning_rate", 1e-3)),
                weight_decay=float(training.get("weight_decay", 1e-5)),
            )
            loader = build_loader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                num_workers=workers,
            )
            baseline_validation = _validation_metrics(
                model,
                validation_data,
                budgets=validation_budgets,
                noises=validation_noises,
                blocklength=blocklength,
                target_rate=target_rate,
                third_order=third_order,
                batch_size=batch_size,
                workers=workers,
                seed=seed,
                device=device,
            )
            baseline_score = float(
                baseline_validation[
                    "validation_expected_goodput_bps_hz"
                ]
            )
            baseline_state_goodput = baseline_validation.pop(
                "_state_goodput_matrix"
            )
            baseline_evidence = paired_cluster_evidence(
                baseline_state_goodput,
                baseline_goodput_by_method,
                cluster_ids=validation_data.trajectory_cluster_ids,
                cluster_method=validation_data.trajectory_cluster_method,
                operating_points=operating_points,
                minimum_relative_improvement=(
                    minimum_relative_validation_improvement
                ),
                maximum_relative_point_regression=(
                    maximum_relative_point_regression
                ),
                confidence_level=validation_confidence_level,
                minimum_cluster_count=minimum_validation_cluster_count,
            )
            baseline_evidence["scope"] = "validation"
            history_rows.append(
                {
                    "candidate": candidate.id,
                    "seed": seed,
                    "epoch": 0,
                    "stage": "feasible_equal_power_initialization",
                    "accepted": False,
                    "demo_evidence_status": baseline_evidence,
                    "train_loss": None,
                    "train_expected_goodput_bps_hz": None,
                    **baseline_validation,
                }
            )
            print(
                "candidate=%s seed=%d epoch=0 stage=equal_power "
                "validation_goodput=%.7g validation_bler=%.7g"
                % (
                    candidate.id,
                    seed,
                    baseline_score,
                    baseline_validation["validation_predicted_bler"],
                )
            )
            candidate_best = baseline_score
            if baseline_score > best_score + 1e-9:
                best_score = baseline_score
                best_state = _cpu_state_dict(model)
                best_candidate = candidate
                best_seed = seed
                best_epoch = 0
                best_evidence = baseline_evidence
            epochs_without_improvement = 0
            for epoch in range(1, epochs + 1):
                model.train()
                train_losses = []
                train_goodput = []
                for delayed_gains, current_gains in loader:
                    delayed_gains = delayed_gains.to(device)
                    current_gains = current_gains.to(device)
                    noise = _sample_log_uniform(
                        noise_range,
                        delayed_gains.shape[0],
                        device,
                    )
                    budget = _sample_log_uniform(
                        budget_range,
                        delayed_gains.shape[0],
                        device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    power = model(delayed_gains, noise, budget)
                    loss = negative_expected_finite_blocklength_goodput(
                        power,
                        current_gains,
                        noise,
                        blocklength_channel_uses=blocklength,
                        target_rate_bps_hz=target_rate,
                        include_third_order_term=third_order,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Training loss became non-finite")
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu()))
                    train_goodput.append(float(-loss.detach().cpu()))

                validation = _validation_metrics(
                    model,
                    validation_data,
                    budgets=validation_budgets,
                    noises=validation_noises,
                    blocklength=blocklength,
                    target_rate=target_rate,
                    third_order=third_order,
                    batch_size=batch_size,
                    workers=workers,
                    seed=seed,
                    device=device,
                )
                row = {
                    "candidate": candidate.id,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "train_expected_goodput_bps_hz": float(
                        np.mean(train_goodput)
                    ),
                    **validation,
                }
                state_goodput = row.pop("_state_goodput_matrix")
                evidence = paired_cluster_evidence(
                    state_goodput,
                    baseline_goodput_by_method,
                    cluster_ids=validation_data.trajectory_cluster_ids,
                    cluster_method=validation_data.trajectory_cluster_method,
                    operating_points=operating_points,
                    minimum_relative_improvement=(
                        minimum_relative_validation_improvement
                    ),
                    maximum_relative_point_regression=(
                        maximum_relative_point_regression
                    ),
                    confidence_level=validation_confidence_level,
                    minimum_cluster_count=minimum_validation_cluster_count,
                )
                evidence["scope"] = "validation"
                score = float(row["validation_expected_goodput_bps_hz"])
                improvement = float(evidence["absolute_improvement_bps_hz"])
                relative_improvement = float(evidence["relative_improvement"])
                accepted = evidence["status"] == "passed"
                row.update(
                    {
                        "stage": "direct_goodput_training",
                        "accepted": accepted,
                        "validation_goodput_improvement": improvement,
                        "validation_relative_goodput_improvement": (
                            relative_improvement
                        ),
                        "demo_evidence_status": evidence,
                    }
                )
                history_rows.append(row)
                print(
                    "candidate=%s seed=%d epoch=%d train_goodput=%.7g "
                    "validation_goodput=%.7g relative_improvement=%.4g "
                    "ci_lower=%.4g accepted=%s "
                    "validation_bler=%.7g budget_error=%.3g"
                    % (
                        candidate.id,
                        seed,
                        epoch,
                        row["train_expected_goodput_bps_hz"],
                        row["validation_expected_goodput_bps_hz"],
                        relative_improvement,
                        evidence["paired_cluster_confidence_interval"][
                            "lower_bps_hz"
                        ],
                        accepted,
                        row["validation_predicted_bler"],
                        row["max_power_budget_error"],
                    )
                )
                if score > candidate_best + 1e-9:
                    candidate_best = score
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if accepted and score > best_score + 1e-9:
                    best_score = score
                    best_state = _cpu_state_dict(model)
                    best_candidate = candidate
                    best_seed = seed
                    best_epoch = epoch
                    best_evidence = evidence
                if patience and epochs_without_improvement >= patience:
                    print(
                        "early stopping candidate=%s seed=%d after epoch=%d"
                        % (candidate.id, seed, epoch)
                    )
                    break

    if (
        best_state is None
        or best_candidate is None
        or best_seed is None
        or best_epoch is None
        or best_evidence is None
    ):
        raise RuntimeError("Training produced no valid checkpoint")
    selected = build_allocator(
        best_candidate,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        history_length=history_length,
    )
    selected.load_state_dict(best_state)

    history_payload = {
        "schema_version": 1,
        "kind": "noema.delayed_csi_finite_blocklength_training_history",
        "objective": "expected_finite_blocklength_goodput",
        "blocklength_channel_uses": blocklength,
        "target_rate_bps_hz": target_rate,
        "uses_power_allocation_labels": False,
        "runtime_inputs": [
            "causal_delayed_noisy_complex_csi_history",
            "noise_variance",
            "average_power_budget",
        ],
        "training_only_environment_outcome": "aligned_current_channel_gain",
        "split_integrity": integrity,
        "selected": {
            "candidate": best_candidate.to_dict(),
            "seed": best_seed,
            "epoch": best_epoch,
            "validation_expected_goodput_bps_hz": best_score,
            "equal_power_fallback": best_epoch == 0,
            "demo_evidence_status": best_evidence,
        },
        "rows": history_rows,
    }
    Path("training_history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    training_manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    policy_path = Path(
        str(
            training.get("artifact_component_path")
            or "../artifacts/delayed_csi_power_policy.onnx"
        )
    )
    policy_sha = export_onnx_policy(
        selected,
        policy_path,
        subcarrier_count=int(train_data.current_gains.shape[1]),
        history_length=history_length,
    )
    provenance = {
        "source_recipe_sha256": str((config.get("recipe") or {}).get("sha256") or ""),
        "train_capture_schema_sha256": list(
            train_data.capture_schema_sha256
        ),
        "validation_capture_schema_sha256": list(
            validation_data.capture_schema_sha256
        ),
        "split_integrity": integrity,
        "initialization_seeds": seeds,
        "selected_seed": best_seed,
        "selected_epoch": best_epoch,
        "equal_power_fallback": best_epoch == 0,
        "demo_evidence_status": best_evidence,
        "selected_candidate": best_candidate.to_dict(),
        "best_validation_expected_goodput_bps_hz": best_score,
        "average_power_budget_range": list(budget_range),
        "noise_variance_range": list(noise_range),
        "blocklength_channel_uses": blocklength,
        "target_rate_bps_hz": target_rate,
        "include_third_order_term": third_order,
        "csi_history_length": history_length,
    }
    artifact_manifest = _write_trained_artifact_manifest(
        config,
        training_manifest_path,
        policy_path,
        policy_sha,
        provenance,
    )
    print(
        "selected candidate=%s seed=%d epoch=%d validation_goodput=%.7g"
        % (best_candidate.id, best_seed, best_epoch, best_score)
    )
    if best_epoch == 0:
        print(
            "selection safeguard retained the exact equal-power initialization; "
            "do not present this artifact as a learned improvement"
        )
    print("exported delayed-CSI policy ONNX: %s" % policy_path)
    print("component sha256: %s" % policy_sha)
    print("registered trained block artifact: %s" % artifact_manifest)
    return 0


def _validation_metrics(
    model: FrequencyResidualPowerAllocator,
    dataset,
    *,
    budgets: list[float],
    noises: list[float],
    blocklength: int,
    target_rate: float,
    third_order: bool,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    goodputs = []
    blers = []
    capacities = []
    point_values: dict[tuple[float, float], dict[str, list[float]]] = {}
    max_budget_error = 0.0
    max_negative = 0.0
    with torch.no_grad():
        for delayed, current in loader:
            delayed = delayed.to(device)
            current = current.to(device)
            for budget_value, noise_value in product(budgets, noises):
                budget = torch.full(
                    (delayed.shape[0],),
                    float(budget_value),
                    device=device,
                )
                noise = torch.full(
                    (delayed.shape[0],),
                    float(noise_value),
                    device=device,
                )
                power = model(delayed, noise, budget)
                goodput, bler, capacity, _ = finite_blocklength_state_metrics(
                    power,
                    current,
                    noise,
                    blocklength_channel_uses=blocklength,
                    target_rate_bps_hz=target_rate,
                    include_third_order_term=third_order,
                )
                goodputs.extend(goodput.cpu().tolist())
                blers.extend(bler.cpu().tolist())
                capacities.extend(capacity.cpu().tolist())
                key = (float(budget_value), float(noise_value))
                point = point_values.setdefault(
                    key,
                    {"goodput": [], "bler": [], "capacity": []},
                )
                point["goodput"].extend(goodput.cpu().tolist())
                point["bler"].extend(bler.cpu().tolist())
                point["capacity"].extend(capacity.cpu().tolist())
                feasibility = feasibility_metrics(power, budget)
                max_budget_error = max(
                    max_budget_error,
                    feasibility["max_power_budget_error"],
                )
                max_negative = max(
                    max_negative,
                    feasibility["max_negative_power_violation"],
                )
    return {
        "validation_expected_goodput_bps_hz": float(np.mean(goodputs)),
        "validation_predicted_bler": float(np.mean(blers)),
        "validation_mean_capacity_bps_hz": float(np.mean(capacities)),
        "validation_operating_points": [
            {
                "average_power_budget": budget,
                "noise_variance": noise,
                "expected_goodput_bps_hz": float(
                    np.mean(point_values[(budget, noise)]["goodput"])
                ),
                "predicted_bler": float(
                    np.mean(point_values[(budget, noise)]["bler"])
                ),
                "mean_capacity_bps_hz": float(
                    np.mean(point_values[(budget, noise)]["capacity"])
                ),
            }
            for budget, noise in sorted(point_values)
        ],
        "_state_goodput_matrix": np.asarray(
            [
                point_values[(budget, noise)]["goodput"]
                for budget, noise in sorted(point_values)
            ],
            dtype=np.float64,
        ),
        "max_power_budget_error": max_budget_error,
        "max_negative_power_violation": max_negative,
    }


def _operating_points(
    budgets: list[float],
    noises: list[float],
) -> list[dict[str, float]]:
    return [
        {
            "average_power_budget": float(budget),
            "noise_variance": float(noise),
        }
        for budget, noise in sorted(product(budgets, noises))
    ]


def _deployable_baseline_goodputs(
    dataset,
    *,
    operating_points: list[dict[str, float]],
    blocklength: int,
    target_rate: float,
    third_order: bool,
    robust_gain_shrinkage: float,
    csi_prediction_horizon: int,
    csi_prediction_confidence: float,
) -> dict[str, np.ndarray]:
    shrinkage = float(robust_gain_shrinkage)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("evaluation.robust_gain_shrinkage must be in [0,1]")
    observed = dataset.latest_observed_gains.astype(np.float64)
    actual = dataset.current_gains.astype(np.float64)
    robust = (
        shrinkage * observed
        + (1.0 - shrinkage) * np.mean(observed, axis=1, keepdims=True)
    )
    predicted = _causal_complex_ar_predicted_gains(
        dataset.csi_history,
        horizon=csi_prediction_horizon,
        confidence=csi_prediction_confidence,
    )
    result: dict[str, list[np.ndarray]] = {
        "equal_power": [],
        "observed_csi_water_filling": [],
        "robust_csi_water_filling": [],
        "causal_ar_water_filling": [],
    }
    subcarrier_count = observed.shape[1]
    for point in operating_points:
        budget = float(point["average_power_budget"])
        noise = float(point["noise_variance"])
        total_power = budget * float(subcarrier_count)
        powers = {
            "equal_power": np.full_like(observed, budget),
            "observed_csi_water_filling": np.stack(
                [
                    _water_filling(row, noise, total_power)
                    for row in observed
                ],
                axis=0,
            ),
            "robust_csi_water_filling": np.stack(
                [
                    _water_filling(row, noise, total_power)
                    for row in robust
                ],
                axis=0,
            ),
            "causal_ar_water_filling": np.stack(
                [
                    _water_filling(row, noise, total_power)
                    for row in predicted
                ],
                axis=0,
            ),
        }
        for method, power in powers.items():
            result[method].append(
                _finite_blocklength_goodput(
                    actual,
                    power,
                    noise=noise,
                    blocklength=blocklength,
                    target_rate=target_rate,
                    third_order=third_order,
                )
            )
    return {
        method: np.asarray(values, dtype=np.float64)
        for method, values in result.items()
    }


def _causal_complex_ar_predicted_gains(
    history_iq: np.ndarray,
    *,
    horizon: int,
    confidence: float,
) -> np.ndarray:
    if int(horizon) < 0:
        raise ValueError(
            "evaluation.csi_prediction_horizon_ofdm_symbols must be nonnegative"
        )
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "evaluation.csi_prediction_gain_confidence must be in [0,1]"
        )
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
        int(horizon),
    )
    gains = np.maximum(np.abs(predicted) ** 2, 1e-12)
    mean_gain = np.mean(gains, axis=1, keepdims=True)
    return float(confidence) * gains + (1.0 - float(confidence)) * mean_gain


def _finite_blocklength_goodput(
    actual_gain: np.ndarray,
    power: np.ndarray,
    *,
    noise: float,
    blocklength: int,
    target_rate: float,
    third_order: bool,
) -> np.ndarray:
    snr = actual_gain * power / float(noise)
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
    z = (
        (capacity - float(target_rate) + correction)
        * math.sqrt(float(blocklength))
        / np.sqrt(np.maximum(dispersion, 1e-12))
    )
    return float(target_rate) * (
        1.0
        - np.asarray(
            [
                0.5 * math.erfc(float(value) / math.sqrt(2.0))
                for value in z
            ],
            dtype=np.float64,
        )
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


def _write_trained_artifact_manifest(
    config: dict,
    manifest_path: Path,
    policy_path: Path,
    policy_sha: str,
    provenance: dict,
) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_source, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if contract_source.resolve() != contract_path.resolve():
        shutil.copyfile(contract_source, contract_path)
    contract_sha = _canonical_sha256(contract)
    contract_file_sha = _sha256(contract_path)
    try:
        portable_policy_path = str(
            policy_path.resolve().relative_to(manifest_path.parent.resolve())
        )
    except ValueError as exc:
        raise ValueError(
            "artifact components must be inside the trained-artifact package"
        ) from exc

    recipe_name = str(recipe.get("name") or "delayed_csi_resource_allocation")
    allocator_step = str(recipe.get("allocator_step") or "tx_power")
    artifact_id = str(
        training.get("artifact_id")
        or "%s.%s.delayed_csi_allocator"
        % (_safe_id(recipe_name), _safe_id(allocator_step))
    )
    artifact_label = str(
        training.get("artifact_label")
        or "Learned delayed-CSI allocator · %s" % recipe_name
    )
    operation = str(
        training.get("artifact_operation")
        or "model.causal_csi_power_allocator"
    )
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": artifact_id,
        "name": str(
            training.get("artifact_name")
            or "Reliability-aware delayed-CSI allocator for %s" % recipe_name
        ),
        "label": artifact_label,
        "description": (
            "History-aware frequency policy trained without allocation labels "
            "to maximize finite-blocklength expected goodput. Runtime receives "
            "only causal delayed/noisy complex CSI; aligned current CSI was a "
            "training-only environment outcome."
        ),
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(
                contract.get("version") or contract.get("schema_version") or 1
            ),
            "path": "training_contract.yaml",
            "sha256": contract_sha,
            "file_sha256": contract_file_sha,
        },
        "components": [
            {
                "id": "policy",
                "role": "power_policy",
                "path": portable_policy_path,
                "sha256": policy_sha,
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "power_policy",
                    "component": "policy",
                    "inputs": [
                        {
                            "name": "csi_history",
                            "dtype": "float32",
                            "shape": [
                                "batch",
                                int(provenance.get("csi_history_length") or 1),
                                "subcarrier",
                                2,
                            ],
                            "layout": "batch_history_subcarrier_iq",
                            "semantic": (
                                "causal_delayed_noisy_complex_csi_history"
                            ),
                        },
                        {
                            "name": "noise_variance",
                            "dtype": "float32",
                            "shape": ["batch", 1],
                            "layout": "batch_scalar",
                            "semantic": "complex_noise_variance",
                        },
                        {
                            "name": "average_power_budget",
                            "dtype": "float32",
                            "shape": ["batch", 1],
                            "layout": "batch_scalar",
                            "semantic": "average_power_per_subcarrier",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "allocation_scores",
                            "dtype": "float32",
                            "shape": ["batch", "subcarrier"],
                            "layout": "batch_subcarrier",
                            "semantic": "unconstrained_power_allocation_scores",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": operation,
                "label": artifact_label,
                "description": (
                    "Frozen reliability-aware allocation policy conditioned "
                    "only on transmitter-visible causal CSI history."
                ),
                "runtime_entrypoint": "power_policy",
                "required_inputs": ["channel_state"],
                "params": {
                    "policy": "learned_artifact",
                    "granularity": "per_subcarrier",
                    "budget_mode": "fixed_average",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "power_policy",
                },
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": str(config.get("training_template") or ""),
            "recipe_name": recipe_name,
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "step_id": allocator_step,
        },
        "training": {
            "framework": str(config.get("framework") or "torch"),
            "objective": LOSS_ID,
            "uses_power_allocation_labels": False,
            "current_csi_role": "training_only_environment_outcome",
            "current_csi_available_at_runtime": False,
            **provenance,
        },
        "evaluation": {
            "command": "cd reference_training && python evaluate.py",
            "metrics_path": "reference_training/evaluation_metrics.json",
            "comparators": [
                "equal_power",
                "water_filling_on_delayed_noisy_csi",
                "uncertainty_shrunk_water_filling",
                "complex_ar_prediction_plus_water_filling",
            ],
            "diagnostic_only": [
                "current_csi_shannon_water_filling_not_finite_block_optimum",
                "perfect_current_csi_finite_blocklength_numerical_reference_not_global_optimum",
            ],
        },
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def _find_training_contract(config: dict) -> tuple[Path, dict]:
    training = dict(config.get("training") or {})
    configured = str(
        training.get("contract_path")
        or config.get("training_contract_path")
        or ""
    ).strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict):
            raise ValueError("training_contract.yaml must contain a mapping")
        if str(payload.get("kind") or "") != "noema.trainable_slot_contract@1":
            raise ValueError(
                "training_contract.yaml is not a Noema trainable-slot contract"
            )
        if not str(payload.get("id") or "").strip():
            raise ValueError("training_contract.yaml requires an id")
        return path, payload
    raise FileNotFoundError(
        "training_contract.yaml is required; run this project inside its "
        "exported Noema training bundle"
    )


def _sample_log_uniform(
    bounds: tuple[float, float],
    count: int,
    device: torch.device,
) -> torch.Tensor:
    low, high = bounds
    if low == high:
        return torch.full(
            (count,),
            low,
            dtype=torch.float32,
            device=device,
        )
    uniform = torch.rand((count,), dtype=torch.float32, device=device)
    return torch.exp(
        math.log(low) + uniform * (math.log(high) - math.log(low))
    )


def _positive_range(value, name: str) -> tuple[float, float]:
    values = _positive_values(value)
    if len(values) == 1:
        return values[0], values[0]
    low, high = float(values[0]), float(values[-1])
    if high < low:
        raise ValueError("training.%s must be ordered low to high" % name)
    return low, high


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


def _cpu_state_dict(model: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: dict) -> torch.device:
    requested = str(config.get("device") or "cuda_if_available")
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: str) -> str:
    result = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in str(value)
    ).strip("_")
    return result or "model"


if __name__ == "__main__":
    raise SystemExit(main())
