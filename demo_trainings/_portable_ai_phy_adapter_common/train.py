from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

import task
from datamodule import build_loader, load_capture_dataset, split_fingerprint_report

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config = _mapping(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    seed = int(list(training.get("initialization_seeds") or [23])[0])
    _seed_everything(seed)
    splits = {
        name: load_capture_dataset(
            data.get("%s_capture_dirs" % name) or [],
            feature_tap=str(data.get("feature_tap") or ""),
            target_tap=str(data.get("target_tap") or ""),
            expected_split=name,
        )
        for name in ("train", "validation")
    }
    split_evidence = split_fingerprint_report(splits)
    device = _device(training)
    model = task.build_model(splits["train"].features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate") or 0.001),
        weight_decay=float(training.get("weight_decay") or 0.0),
    )
    epochs = max(1, int(training.get("epochs") or 40))
    batch_size = max(1, int(training.get("batch_size") or 128))
    patience = max(1, int(training.get("early_stopping_patience") or 8))
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        loader = build_loader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = task.training_loss(model, features, targets)
            if not torch.isfinite(loss):
                raise ValueError("training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss, validation_metrics = _evaluate_model(
            model,
            splits["validation"],
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
            }
        )
        print(
            "epoch %d/%d train=%.6f validation=%.6f"
            % (epoch, epochs, rows[-1]["train_loss"], validation_loss)
        )
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    component_path = Path(
        str(training.get("artifact_component_path") or "../artifacts/model.onnx")
    )
    component_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.from_numpy(splits["validation"].features[:2]).to(device)
    task.export_onnx(model, sample, component_path)
    component_sha = _sha256(component_path)
    history = {
        "schema_version": 1,
        "kind": "noema.reference_training_history",
        "task": task.TASK_ID,
        "objective": str((config.get("objective") or {}).get("loss") or ""),
        "initialization_seed": seed,
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "test_split_used_for_training_or_selection": False,
        "capture_splits": split_evidence,
        "epochs": rows,
    }
    history_path = Path("training_history.json")
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "candidate_count": 1,
        "selected_candidate": {
            "initialization_seed": seed,
            "epoch": best_epoch,
            "validation_loss": best_loss,
            "architecture": type(model).__name__,
        },
        "model_selection_history": {
            "path": "reference_training/training_history.json",
            "sha256": _sha256(history_path),
        },
        "capture_splits": split_evidence,
    }
    manifest_path = _write_manifest(
        config,
        component_path=component_path,
        component_sha=component_sha,
        provenance=provenance,
        runtime_inputs=task.runtime_inputs(splits["train"].features),
        runtime_outputs=task.runtime_outputs(splits["train"].features),
    )
    print("exported ONNX component: %s" % component_path)
    print("registered trained artifact: %s" % manifest_path)
    return 0


def _evaluate_model(model, dataset, *, batch_size: int, device: torch.device):
    losses = []
    predictions = []
    features_all = []
    targets_all = []
    model.eval()
    with torch.no_grad():
        for features, targets in build_loader(
            dataset, batch_size=batch_size, shuffle=False, seed=0
        ):
            features = features.to(device)
            targets = targets.to(device)
            losses.append(float(task.training_loss(model, features, targets).cpu()))
            predictions.append(task.predictions(model, features).cpu())
            features_all.append(features.cpu())
            targets_all.append(targets.cpu())
    metrics = task.metrics_from_predictions(
        torch.cat(predictions), torch.cat(features_all), torch.cat(targets_all)
    )
    return float(np.mean(losses)), metrics


def _write_manifest(
    config: dict[str, Any],
    *,
    component_path: Path,
    component_sha: str,
    provenance: dict[str, Any],
    runtime_inputs: list[dict[str, Any]],
    runtime_outputs: list[dict[str, Any]],
) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    manifest_path = Path(str(training.get("artifact_manifest_path") or "../trained_artifact.yaml"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_source, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if contract_source.resolve() != contract_path.resolve():
        shutil.copyfile(contract_source, contract_path)
    try:
        portable_component = str(component_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError("ONNX component must be inside the returned artifact package") from exc
    label = str(training.get("artifact_label") or "Learned %s" % task.TASK_ID)
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": str(training.get("artifact_id") or "external.%s" % task.ENTRYPOINT_ID),
        "name": str(training.get("artifact_name") or label),
        "label": label,
        "description": task.DESCRIPTION,
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(contract.get("version") or contract.get("schema_version") or 1),
            "path": "training_contract.yaml",
            "sha256": _canonical_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": task.COMPONENT_ID,
                "role": task.COMPONENT_ROLE,
                "path": portable_component,
                "sha256": component_sha,
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": task.ENTRYPOINT_ID,
                    "component": task.COMPONENT_ID,
                    "inputs": runtime_inputs,
                    "outputs": runtime_outputs,
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": task.OPERATION_ID,
                "preferred_step_id": str(recipe.get("replacement_step") or ""),
                "label": label,
                "description": task.DESCRIPTION,
                "runtime_entrypoint": task.ENTRYPOINT_ID,
                "required_inputs": list(task.REQUIRED_INPUTS),
                "params": task.binding_params(),
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": str(config.get("training_template") or ""),
            "recipe_name": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "step_id": str(recipe.get("replacement_step") or ""),
        },
        "training": {
            "framework": str(config.get("framework") or "torch"),
            "architecture": str((provenance.get("selected_candidate") or {}).get("architecture") or ""),
            "loss": str((config.get("objective") or {}).get("loss") or ""),
            "supervised_labels_used": bool(
                getattr(
                    task,
                    "SUPERVISED_LABELS_USED",
                    task.TASK_ID != "beamforming_precoding",
                )
            ),
            **provenance,
        },
        "evaluation": {
            "command": "cd reference_training && python evaluate.py",
            "metrics_path": "reference_training/evaluation_metrics.json",
            "primary_metric": task.PRIMARY_METRIC,
            "split": "test",
        },
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def _find_training_contract(config: Mapping[str, Any]):
    reference = dict(config.get("training_contract") or {})
    candidates = ([Path(str(reference["path"]))] if reference.get("path") else [])
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if path.is_file():
            payload = _mapping(path)
            if str(payload.get("kind") or "") != "noema.trainable_slot_contract@1":
                raise ValueError("%s is not a Noema trainable-slot contract" % path)
            return path, payload
    raise FileNotFoundError("training_contract.yaml is required to package the artifact")


def _mapping(path: Path) -> dict[str, Any]:
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping: %s" % path)
    return dict(payload)


def _device(training: Mapping[str, Any]) -> torch.device:
    requested = str(training.get("device") or "cuda_if_available")
    return torch.device("cuda" if requested == "cuda_if_available" and torch.cuda.is_available() else "cpu" if requested == "cuda_if_available" else requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
