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

from datamodule import build_loader, feature_statistics, load_capture_dataset
from losses import feasibility_metrics, negative_shannon_spectral_efficiency, spectral_efficiency
from model import DeepSetPowerAllocator, export_onnx_policy
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config_path = Path("train_config.yaml")
    config = load_strict_yaml_or_json(config_path)
    data_config = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    objective = dict(config.get("objective") or {})
    if bool(objective.get("uses_oracle_labels", True)):
        raise ValueError("This template is label-free: objective.uses_oracle_labels must be false")
    feature_tap = str(data_config.get("feature_tap") or "channel_gains")
    train_data = load_capture_dataset(
        data_config.get("train_capture_dirs") or [],
        feature_tap=feature_tap,
        expected_split="train",
    )
    validation_data = load_capture_dataset(
        data_config.get("validation_capture_dirs") or [],
        feature_tap=feature_tap,
        expected_split="validation",
    )
    if train_data.gains.shape[1] != validation_data.gains.shape[1]:
        raise ValueError("Training and validation captures use different subcarrier counts")

    device = _device(training)
    seeds = [int(value) for value in training.get("initialization_seeds", [23, 41, 67])]
    if not seeds:
        raise ValueError("training.initialization_seeds must not be empty")
    budget_range = _positive_range(training.get("average_power_budget_range", [0.5, 2.0]), "average_power_budget_range")
    noise_range = _positive_range(training.get("noise_variance_range", [0.2, 0.2]), "noise_variance_range")
    validation_budgets = _positive_values(training.get("validation_average_power_budgets", [0.5, 1.0, 2.0]))
    validation_noises = _positive_values(training.get("validation_noise_variances", [0.2]))
    reference_budget = math.sqrt(budget_range[0] * budget_range[1])
    reference_noise = math.sqrt(noise_range[0] * noise_range[1])
    feature_mean, feature_scale = feature_statistics(
        train_data,
        reference_noise_variance=reference_noise,
        reference_average_power_budget=reference_budget,
    )
    hidden_dim = int((config.get("model") or {}).get("hidden_dim", 64))
    epochs = max(1, int(training.get("epochs", 100)))
    batch_size = max(1, int(training.get("batch_size", 256)))
    workers = max(0, int(training.get("num_workers", 0)))
    patience = max(0, int(training.get("early_stopping_patience", 20)))

    best_global_score = -float("inf")
    best_global_state = None
    selected_seed = None
    history: list[dict] = []
    for seed in seeds:
        _seed_everything(seed)
        model = DeepSetPowerAllocator(
            hidden_dim=hidden_dim,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
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
        best_seed_score = -float("inf")
        best_seed_state = None
        epochs_without_improvement = 0
        for epoch in range(epochs):
            model.train()
            losses = []
            train_se = []
            for gains in loader:
                gains = gains.to(device)
                noise = _sample_log_uniform(noise_range, gains.shape[0], device)
                budget = _sample_log_uniform(budget_range, gains.shape[0], device)
                optimizer.zero_grad(set_to_none=True)
                power = model(gains, noise, budget)
                loss = negative_shannon_spectral_efficiency(power, gains, noise)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                train_se.append(float(torch.mean(spectral_efficiency(power, gains, noise)).detach().cpu()))
            validation_score, validation_feasibility = _validation_score(
                model,
                validation_data,
                validation_budgets,
                validation_noises,
                batch_size,
                workers,
                seed,
                device,
            )
            row = {
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "train_spectral_efficiency_bps_hz": float(np.mean(train_se)),
                "validation_spectral_efficiency_bps_hz": validation_score,
                **validation_feasibility,
            }
            history.append(row)
            print(
                "seed=%d epoch=%d train_loss=%.7g validation_se_bps_hz=%.7g budget_error=%.3g"
                % (
                    seed,
                    epoch + 1,
                    row["train_loss"],
                    validation_score,
                    validation_feasibility["max_power_budget_error"],
                )
            )
            if validation_score > best_seed_score + 1e-10:
                best_seed_score = validation_score
                best_seed_state = _cpu_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                break
        if best_seed_state is not None and best_seed_score > best_global_score:
            best_global_score = best_seed_score
            best_global_state = best_seed_state
            selected_seed = seed

    if best_global_state is None or selected_seed is None:
        raise RuntimeError("Training produced no valid checkpoint")
    selected_model = DeepSetPowerAllocator(
        hidden_dim=hidden_dim,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    selected_model.load_state_dict(best_global_state)
    provenance = {
        "objective": "maximize_parallel_channel_shannon_spectral_efficiency",
        "loss": "resource.negative_shannon_spectral_efficiency",
        "supervised_labels_used": False,
        "water_filling_used_during_training": False,
        "checkpoint_selected_by": "validation_shannon_spectral_efficiency",
        "source_recipe_sha256": str((config.get("recipe") or {}).get("sha256") or ""),
        "train_capture_schema_sha256": list(train_data.capture_sha256),
        "validation_capture_schema_sha256": list(validation_data.capture_sha256),
        "initialization_seeds": seeds,
        "selected_seed": selected_seed,
        "best_validation_spectral_efficiency_bps_hz": best_global_score,
        "average_power_budget_range": list(budget_range),
        "noise_variance_range": list(noise_range),
    }
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    policy_path = manifest_path.parent / "artifacts" / "power_policy.onnx"
    policy_sha = export_onnx_policy(
        selected_model,
        policy_path,
        subcarrier_count=int(train_data.gains.shape[1]),
    )
    Path("training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    artifact_manifest_path = _write_trained_artifact_manifest(
        config,
        policy_path,
        policy_sha,
        provenance,
    )
    print(f"exported policy ONNX: {policy_path}")
    print(f"policy sha256: {policy_sha}")
    print(f"registered trained block artifact: {artifact_manifest_path}")
    print("return to Noema and select this artifact on a compatible model.symbol_power_allocator block")
    return 0


def _validation_score(
    model: DeepSetPowerAllocator,
    dataset,
    budgets: list[float],
    noises: list[float],
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed, num_workers=workers)
    scores = []
    max_budget_error = 0.0
    max_negative = 0.0
    with torch.no_grad():
        for gains in loader:
            gains = gains.to(device)
            for budget_value, noise_value in product(budgets, noises):
                budget = torch.full((gains.shape[0],), float(budget_value), device=device)
                noise = torch.full((gains.shape[0],), float(noise_value), device=device)
                power = model(gains, noise, budget)
                scores.append(float(torch.mean(spectral_efficiency(power, gains, noise)).cpu()))
                feasibility = feasibility_metrics(power, budget)
                max_budget_error = max(max_budget_error, feasibility["max_power_budget_error"])
                max_negative = max(max_negative, feasibility["max_negative_power_violation"])
    return float(np.mean(scores)), {
        "max_power_budget_error": max_budget_error,
        "max_negative_power_violation": max_negative,
    }


def _write_trained_artifact_manifest(
    config: dict,
    policy_path: Path,
    policy_sha: str,
    provenance: dict,
) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_source, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if contract_source.resolve() != contract_path.resolve():
        shutil.copyfile(contract_source, contract_path)
    contract_sha = _canonical_sha256(contract)
    contract_file_sha = _sha256(contract_path)
    try:
        portable_policy_path = str(policy_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError("artifact components must be inside the trained-artifact package") from exc
    recipe_name = str(recipe.get("name") or "resource_allocation")
    allocator_step = str(recipe.get("allocator_step") or "tx_power")
    artifact_id = str(
        training.get("artifact_id")
        or "%s.%s.csi_power_allocator" % (_safe_id(recipe_name), _safe_id(allocator_step))
    )
    artifact_name = str(
        training.get("artifact_name")
        or "Learned CSI power allocator for %s" % recipe_name
    )
    artifact_label = str(training.get("artifact_label") or "Learned · %s" % recipe_name)
    operation = str(training.get("artifact_operation") or "model.symbol_power_allocator")
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": artifact_id,
        "name": artifact_name,
        "label": artifact_label,
        "description": (
            "Label-free CSI-conditioned power allocator selected by validation Shannon "
            "spectral efficiency. Apply it to a compatible allocator block; the recipe "
            "continues to control power budget, channel, and noise."
        ),
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(contract.get("version") or contract.get("schema_version") or 1),
            "path": "training_contract.yaml",
            "sha256": contract_sha,
            "file_sha256": contract_file_sha,
        },
        "components": [
            {
                "id": "policy",
                "role": "power_policy",
                "path": portable_policy_path,
                "sha256": str(policy_sha),
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
                            "name": "channel_gain",
                            "dtype": "float32",
                            "shape": ["batch", "subcarrier"],
                            "layout": "batch_subcarrier",
                            "semantic": "per_subcarrier_channel_power_gain",
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
                "description": "Frozen learned per-subcarrier allocation policy conditioned on OFDM CSI.",
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
            "objective": str((config.get("objective") or {}).get("loss") or ""),
            "uses_oracle_labels": False,
            "supervised_labels_used": bool(provenance.get("supervised_labels_used")),
            "water_filling_used_during_training": bool(
                provenance.get("water_filling_used_during_training")
            ),
            "checkpoint_selected_by": str(provenance.get("checkpoint_selected_by") or ""),
            "source_recipe_sha256": str(provenance.get("source_recipe_sha256") or ""),
            "train_capture_schema_sha256": list(
                provenance.get("train_capture_schema_sha256") or []
            ),
            "validation_capture_schema_sha256": list(
                provenance.get("validation_capture_schema_sha256") or []
            ),
            "initialization_seeds": list(provenance.get("initialization_seeds") or []),
            "selected_seed": provenance.get("selected_seed"),
            "best_validation_spectral_efficiency_bps_hz": provenance.get(
                "best_validation_spectral_efficiency_bps_hz"
            ),
            "average_power_budget_range": list(
                provenance.get("average_power_budget_range") or []
            ),
            "noise_variance_range": list(provenance.get("noise_variance_range") or []),
        },
        "evaluation": {
            "command": "cd reference_training && python evaluate.py",
            "metrics_path": "reference_training/evaluation_metrics.json",
            "held_out_oracle": str((config.get("evaluation") or {}).get("held_out_oracle") or ""),
        },
    }
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def _find_training_contract(config: dict) -> tuple[Path, dict]:
    training = dict(config.get("training") or {})
    configured = str(training.get("contract_path") or config.get("training_contract_path") or "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict):
            raise ValueError("training_contract.yaml must contain a mapping")
        if not str(payload.get("kind") or "").startswith("noema.trainable_slot_contract"):
            raise ValueError("training_contract.yaml is not a Noema trainable-slot contract")
        if not str(payload.get("id") or "").strip():
            raise ValueError("training_contract.yaml requires an id")
        return path, payload
    raise FileNotFoundError(
        "training_contract.yaml is required to package the returned artifact; "
        "run this demonstration project inside its exported Noema training-contract bundle"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in str(value)).strip("_") or "model"


def _sample_log_uniform(bounds: tuple[float, float], count: int, device: torch.device) -> torch.Tensor:
    low, high = bounds
    if low == high:
        return torch.full((count,), low, dtype=torch.float32, device=device)
    uniform = torch.rand((count,), dtype=torch.float32, device=device)
    return torch.exp(math.log(low) + uniform * (math.log(high) - math.log(low)))


def _positive_range(value, name: str) -> tuple[float, float]:
    values = _positive_values(value)
    if len(values) == 1:
        return values[0], values[0]
    low, high = float(values[0]), float(values[-1])
    if high < low:
        raise ValueError(f"training.{name} must be ordered low to high")
    return low, high


def _positive_values(value) -> list[float]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    result = [float(item) for item in raw]
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError("Operating-point values must be finite and greater than zero")
    return result


def _cpu_state_dict(model: torch.nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


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


if __name__ == "__main__":
    raise SystemExit(main())
