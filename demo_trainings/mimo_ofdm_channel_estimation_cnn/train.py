from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from datamodule import (
    build_loader,
    fixed_prior_lmmse_estimate,
    load_capture_dataset,
    split_integrity_report,
)
from losses import normalized_complex_mse
from model import FrequencyResidualEstimator, export_onnx

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    root = Path(__file__).resolve().parent
    config = _mapping(root / "train_config.yaml")
    data_config = dict(config.get("data") or {})
    train = _dataset(root, data_config, "train")
    validation = _dataset(root, data_config, "validation")
    integrity = split_integrity_report({"train": train, "validation": validation})
    scenario = dict(config.get("scenario") or {})
    training = dict(config.get("training") or {})
    model_config = dict(config.get("model") or {})
    device = torch.device(
        "cuda"
        if str(training.get("device") or "") == "cuda_if_available"
        and torch.cuda.is_available()
        else "cpu"
    )
    epochs = int(training.get("epochs") or 30)
    batch_size = int(training.get("batch_size") or 64)
    patience = int(training.get("patience") or 8)
    regression_tolerance_db = float(
        training.get("snr_regression_tolerance_db") or 0.15
    )
    candidates = list(model_config.get("candidates") or [])
    seeds = [int(value) for value in training.get("initialization_seeds") or [23, 41]]
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    validation_ls = _evaluate_ls(validation)
    validation_lmmse = _evaluate_array(
        fixed_prior_lmmse_estimate(validation),
        validation,
    )
    validation_classical = _strongest_classical_by_snr(
        validation_ls,
        validation_lmmse,
    )
    strongest_classical_aggregate_db = min(
        float(validation_ls["nmse_db"]),
        float(validation_lmmse["nmse_db"]),
    )

    for candidate_index, candidate_value in enumerate(candidates):
        candidate = dict(candidate_value)
        for seed_index, seed in enumerate(seeds):
            _seed_everything(seed)
            model = _new_model(scenario, candidate).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training.get("learning_rate") or 1e-3),
                weight_decay=float(training.get("weight_decay") or 1e-5),
            )
            loader = build_loader(
                train,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            )
            best_for_run = math.inf
            stale = 0
            for epoch in range(0, epochs + 1):
                train_loss = None
                if epoch:
                    model.train()
                    train_losses = []
                    for pilot_value, pilot_mask, ls_value, noise, truth in loader:
                        pilot_value = pilot_value.to(device)
                        pilot_mask = pilot_mask.to(device)
                        ls_value = ls_value.to(device)
                        noise = noise.to(device)
                        truth = truth.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(
                            pilot_value,
                            pilot_mask,
                            ls_value,
                            noise,
                        )
                        loss = normalized_complex_mse(prediction, truth)
                        loss.backward()
                        optimizer.step()
                        train_losses.append(float(loss.detach().cpu()))
                    train_loss = float(np.mean(train_losses))
                validation_report = _evaluate(
                    model,
                    validation,
                    batch_size=batch_size,
                    device=device,
                )
                validation_nmse = float(validation_report["nmse"])
                validation_db = float(validation_report["nmse_db"])
                snr_guard = _snr_guard(
                    validation_report,
                    validation_classical,
                    tolerance_db=regression_tolerance_db,
                )
                aggregate_classic_delta_db = (
                    validation_db - strongest_classical_aggregate_db
                )
                snr_guard["aggregate_nmse_db_learned_minus_strongest_classical"] = (
                    aggregate_classic_delta_db
                )
                snr_guard["aggregate_classic_win"] = (
                    aggregate_classic_delta_db < 0.0
                )
                row = {
                    "candidate": str(candidate.get("id") or "residual_cnn"),
                    "seed": seed,
                    "epoch": epoch,
                    "stage": "ls_safe_initialization" if epoch == 0 else "training",
                    "train_nmse": train_loss,
                    "validation_nmse": validation_nmse,
                    "validation_nmse_db": validation_db,
                    "validation_by_snr": validation_report["by_snr"],
                    "classic_win_snr_bins": snr_guard["classic_win_snr_bins"],
                    **snr_guard,
                }
                history.append(row)
                print(
                    "candidate=%s seed=%d epoch=%d stage=%s train_nmse=%s "
                    "validation_nmse_db=%.5g classic_win_snr_bins=%d "
                    "aggregate_classic_delta_db=%.5g regressed_snr_bins=%d "
                    "worst_regression_db=%.5g"
                    % (
                        row["candidate"],
                        seed,
                        epoch,
                        row["stage"],
                        (
                            "baseline"
                            if train_loss is None
                            else "%.7g" % train_loss
                        ),
                        validation_db,
                        snr_guard["classic_win_snr_bins"],
                        aggregate_classic_delta_db,
                        snr_guard["regressed_snr_bins"],
                        snr_guard["worst_regression_db"],
                    )
                )
                rank = (
                    0 if aggregate_classic_delta_db < 0.0 else 1,
                    int(snr_guard["regressed_snr_bins"]),
                    float(snr_guard["worst_regression_beyond_tolerance_db"]),
                    validation_db,
                    candidate_index,
                    seed_index,
                    epoch,
                )
                if best is None or rank < best["rank"]:
                    best = {
                        "rank": rank,
                        "candidate": candidate,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_nmse": validation_nmse,
                        "validation_nmse_db": validation_db,
                        "validation_by_snr": validation_report["by_snr"],
                        "snr_guard": snr_guard,
                        "state_dict": copy.deepcopy(model.cpu().state_dict()),
                    }
                    model.to(device)
                if not epoch:
                    best_for_run = validation_nmse
                    continue
                if validation_nmse < best_for_run - 1e-5:
                    best_for_run = validation_nmse
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    print(
                        "early stopping candidate=%s seed=%d after epoch=%d"
                        % (row["candidate"], seed, epoch)
                    )
                    break
    if best is None:
        raise ValueError("No channel-estimation candidate was trained")

    candidate = dict(best["candidate"])
    selected = _new_model(scenario, candidate)
    selected.load_state_dict(best["state_dict"])
    component_path = _resolve(root, str(training["artifact_component_path"]))
    component_sha = export_onnx(
        selected,
        component_path,
        rx_antennas=int(scenario.get("rx_antennas") or 2),
        tx_antennas=int(scenario.get("tx_antennas") or 2),
        subcarriers=int(scenario.get("subcarriers") or 64),
    )
    history_payload = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_split_exposed_to_training": False,
        "split_integrity": integrity,
        "selected": {
            "candidate": str(candidate.get("id") or ""),
            "seed": int(best["seed"]),
            "epoch": int(best["epoch"]),
            "validation_nmse": float(best["validation_nmse"]),
            "validation_nmse_db": float(best["validation_nmse_db"]),
            "validation_by_snr": best["validation_by_snr"],
            "snr_guard": best["snr_guard"],
        },
        "validation_ls_by_snr": validation_ls["by_snr"],
        "validation_fixed_prior_lmmse_by_snr": validation_lmmse["by_snr"],
        "validation_strongest_classical_by_snr": validation_classical["by_snr"],
        "epochs": history,
        "component_sha256": component_sha,
    }
    history_path = root / "training_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = _write_manifest(
        root,
        config,
        component_path=component_path,
        component_sha=component_sha,
        history_path=history_path,
        history=history_payload,
    )
    print(
        "selected candidate=%s seed=%d epoch=%d validation_nmse_db=%.5g "
        "aggregate_classic_delta_db=%.5g classic_win_snr_bins=%d/6"
        % (
            candidate.get("id"),
            best["seed"],
            best["epoch"],
            best["validation_nmse_db"],
            best["snr_guard"][
                "aggregate_nmse_db_learned_minus_strongest_classical"
            ],
            best["snr_guard"]["classic_win_snr_bins"],
        )
    )
    print("exported channel estimator ONNX: %s" % component_path)
    print("component sha256: %s" % component_sha)
    print("registered trained block artifact: %s" % manifest_path)
    return 0


def _new_model(scenario: dict, candidate: dict) -> FrequencyResidualEstimator:
    return FrequencyResidualEstimator(
        rx_antennas=int(scenario.get("rx_antennas") or 2),
        tx_antennas=int(scenario.get("tx_antennas") or 2),
        subcarriers=int(scenario.get("subcarriers") or 64),
        pilot_spacing=int(scenario.get("pilot_spacing") or 4),
        hidden_channels=int(candidate.get("hidden_channels") or 48),
        depth=int(candidate.get("depth") or 5),
    )


def _evaluate(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    numerator = 0.0
    denominator = 0.0
    grouped: dict[float, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for pilot_value, pilot_mask, ls_value, noise, truth in loader:
            prediction = model(
                pilot_value.to(device),
                pilot_mask.to(device),
                ls_value.to(device),
                noise.to(device),
            )
            truth = truth.to(device)
            dimensions = tuple(range(1, prediction.ndim))
            sample_error = torch.sum((prediction - truth) ** 2, dim=dimensions)
            sample_power = torch.sum(truth**2, dim=dimensions)
            numerator += float(torch.sum(sample_error).cpu())
            denominator += float(torch.sum(sample_power).cpu())
            for variance, error, power in zip(
                noise.reshape(-1).tolist(),
                sample_error.detach().cpu().tolist(),
                sample_power.detach().cpu().tolist(),
            ):
                key = round(float(variance), 8)
                cell = grouped.setdefault(key, [0.0, 0.0])
                cell[0] += float(error)
                cell[1] += float(power)
    return _evaluation_payload(numerator, denominator, grouped)


def _evaluate_ls(dataset) -> dict[str, object]:
    return _evaluate_array(dataset.ls_estimate_ri, dataset)


def _evaluate_array(prediction: np.ndarray, dataset) -> dict[str, object]:
    error = np.sum(
        (prediction - dataset.channel_truth_ri) ** 2,
        axis=tuple(range(1, prediction.ndim)),
    )
    power = np.sum(
        dataset.channel_truth_ri**2,
        axis=tuple(range(1, dataset.channel_truth_ri.ndim)),
    )
    grouped: dict[float, list[float]] = {}
    for variance, sample_error, sample_power in zip(
        dataset.noise_variance.reshape(-1),
        error,
        power,
    ):
        key = round(float(variance), 8)
        cell = grouped.setdefault(key, [0.0, 0.0])
        cell[0] += float(sample_error)
        cell[1] += float(sample_power)
    return _evaluation_payload(float(np.sum(error)), float(np.sum(power)), grouped)


def _strongest_classical_by_snr(
    least_squares: dict[str, object],
    fixed_prior_lmmse: dict[str, object],
) -> dict[str, object]:
    ls_cells = dict(least_squares["by_snr"])
    lmmse_cells = dict(fixed_prior_lmmse["by_snr"])
    by_snr = {}
    for snr in sorted(ls_cells, key=float):
        candidates = {
            "least_squares": dict(ls_cells[snr]),
            "fixed_prior_lmmse": dict(lmmse_cells[snr]),
        }
        selected_name, selected = min(
            candidates.items(),
            key=lambda item: float(item[1]["nmse_db"]),
        )
        by_snr[snr] = {**selected, "method": selected_name}
    return {"by_snr": by_snr}


def _evaluation_payload(
    numerator: float,
    denominator: float,
    grouped: dict[float, list[float]],
) -> dict[str, object]:
    nmse = numerator / max(denominator, 1e-12)
    by_snr = {}
    for variance, (error, power) in sorted(grouped.items(), reverse=True):
        cell_nmse = error / max(power, 1e-12)
        snr_db = -10.0 * math.log10(max(float(variance), 1e-12))
        by_snr["%g" % round(snr_db, 6)] = {
            "noise_variance": float(variance),
            "nmse": float(cell_nmse),
            "nmse_db": float(10.0 * math.log10(max(cell_nmse, 1e-12))),
        }
    return {
        "nmse": float(nmse),
        "nmse_db": float(10.0 * math.log10(max(nmse, 1e-12))),
        "by_snr": by_snr,
    }


def _snr_guard(
    report: dict[str, object],
    baseline: dict[str, object],
    *,
    tolerance_db: float,
) -> dict[str, object]:
    candidate_cells = dict(report["by_snr"])
    baseline_cells = dict(baseline["by_snr"])
    deltas = {
        snr: float(candidate_cells[snr]["nmse_db"])
        - float(baseline_cells[snr]["nmse_db"])
        for snr in sorted(baseline_cells, key=float)
    }
    violations = {
        snr: delta
        for snr, delta in deltas.items()
        if delta > float(tolerance_db)
    }
    worst = max([0.0, *deltas.values()])
    return {
        "snr_nmse_db_learned_minus_strongest_classical": deltas,
        "classic_win_snr_bins": sum(delta < 0.0 for delta in deltas.values()),
        "snr_regression_tolerance_db": float(tolerance_db),
        "regressed_snr_bins": len(violations),
        "worst_regression_db": float(worst),
        "worst_regression_beyond_tolerance_db": float(
            max(0.0, worst - float(tolerance_db))
        ),
    }


def _dataset(root: Path, config: dict, split: str):
    return load_capture_dataset(
        [_resolve(root, value) for value in config["%s_capture_dirs" % split]],
        pilot_tap=str(config["pilot_tap"]),
        mask_tap=str(config["mask_tap"]),
        ls_tap=str(config["ls_tap"]),
        noise_tap=str(config["noise_tap"]),
        target_tap=str(config["target_tap"]),
        expected_split=split,
    )


def _write_manifest(
    root: Path,
    config: dict,
    *,
    component_path: Path,
    component_sha: str,
    history_path: Path,
    history: dict,
) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    scenario = dict(config.get("scenario") or {})
    manifest_path = _resolve(root, str(training["artifact_manifest_path"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path = manifest_path.parent / "training_contract.yaml"
    contract = _mapping(contract_path)
    component_relative = str(component_path.resolve().relative_to(manifest_path.parent.resolve()))
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": str(training["artifact_id"]),
        "name": str(training["artifact_name"]),
        "label": str(training["artifact_label"]),
        "description": (
            "Portable sparse-pilot dual-domain residual estimator trained on "
            "captured 2×2 MIMO-OFDM mixed-profile channels."
        ),
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(contract.get("version") or contract.get("schema_version") or 1),
            "path": "training_contract.yaml",
            "sha256": _canonical_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "estimator",
                "role": "mimo_ofdm_channel_estimator",
                "path": component_relative,
                "sha256": component_sha,
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "channel_estimator",
                    "component": "estimator",
                    "inputs": [
                        {
                            "name": "pilot_ls_ri",
                            "dtype": "float32",
                            "shape": ["batch", "rx_antenna", "tx_antenna", "subcarrier", 2],
                            "layout": "batch_rx_tx_subcarrier_real_imag",
                            "semantic": "sparse_divided_pilot_channel_observations",
                        },
                        {
                            "name": "pilot_mask",
                            "dtype": "float32",
                            "shape": ["batch", "tx_antenna", "subcarrier"],
                            "layout": "batch_tx_subcarrier",
                            "semantic": "binary_active_pilot_mask",
                        },
                        {
                            "name": "ls_estimate_ri",
                            "dtype": "float32",
                            "shape": ["batch", "rx_antenna", "tx_antenna", "subcarrier", 2],
                            "layout": "batch_rx_tx_subcarrier_real_imag",
                            "semantic": "operation_owned_least_squares_channel_estimate",
                        },
                        {
                            "name": "noise_variance",
                            "dtype": "float32",
                            "shape": ["batch", 1],
                            "layout": "batch_scalar",
                            "semantic": "complex_awgn_variance",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "h_hat_ri",
                            "dtype": "float32",
                            "shape": ["batch", "rx_antenna", "tx_antenna", "subcarrier", 2],
                            "layout": "batch_rx_tx_subcarrier_real_imag",
                            "semantic": "estimated_frequency_domain_mimo_channel",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": str(training["artifact_operation"]),
                "preferred_step_id": str(recipe["estimator_step"]),
                "label": str(training["artifact_label"]),
                "description": "Frozen MIMO-OFDM channel estimator.",
                "runtime_entrypoint": "channel_estimator",
                "required_inputs": ["problem"],
                "params": {
                    "mode": "learned_artifact",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "channel_estimator",
                },
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": str(config.get("training_template") or ""),
            "recipe_name": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "training_history": str(
                history_path.resolve().relative_to(manifest_path.parent.resolve())
            ),
            "training_history_sha256": _sha256(history_path),
            "validation_nmse_db": float(history["selected"]["validation_nmse_db"]),
            "test_split_exposed_to_training": False,
            "scenario": scenario,
        },
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _mapping(path: Path) -> dict:
    value = load_strict_yaml_or_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML/JSON object: %s" % path)
    return dict(value)


def _resolve(root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
