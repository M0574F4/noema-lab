from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from datamodule import build_loader, load_capture_dataset, split_fingerprint_report
from losses import classification_loss, classification_metrics
from model import ReferenceModulationCNN1D, export_onnx_classifier
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    feature_tap = str(data.get("feature_tap") or "iq_frames")
    target_tap = str(data.get("target_tap") or "modulation_labels")
    train_data = load_capture_dataset(
        data.get("train_capture_dirs") or [],
        feature_tap=feature_tap,
        target_tap=target_tap,
        expected_split="train",
    )
    validation_data = load_capture_dataset(
        data.get("validation_capture_dirs") or [],
        feature_tap=feature_tap,
        target_tap=target_tap,
        expected_split="validation",
    )
    split_record_fingerprints = split_fingerprint_report(
        {"train": train_data, "validation": validation_data},
        include_record_sha256=True,
    )
    model_config = dict(config.get("model") or {})
    channels = tuple(
        int(value) for value in model_config.get("channels", [32, 64])
    )
    if len(channels) != 2:
        raise ValueError("model.channels must contain exactly two positive integers")
    dropout = float(model_config.get("dropout", 0.25))
    analytic_prior_weight = float(
        model_config.get("analytic_prior_weight", 0.35)
    )
    epochs = max(1, int(training.get("epochs", 20)))
    batch_size = max(1, int(training.get("batch_size", 64)))
    workers = max(0, int(training.get("num_workers", 0)))
    label_smoothing = float(training.get("label_smoothing", 0.03))
    residual_l2_weight = float(training.get("residual_l2_weight", 1e-3))
    correct_prior_residual_weight = float(
        training.get("correct_prior_residual_weight", 1e-2)
    )
    early_stopping_patience = max(
        1,
        int(training.get("early_stopping_patience", 8)),
    )
    rotation_augmentation = bool(
        training.get("rotation_augmentation", True)
    )
    seeds = [int(value) for value in training.get("initialization_seeds", [23, 41])]
    if not seeds:
        raise ValueError("training.initialization_seeds must not be empty")
    device = _device(training)
    best_key = (float("inf"), -1.0)
    best_state = None
    selected_seed = None
    selected_epoch = None
    history = []
    for seed in seeds:
        _seed_everything(seed)
        model = ReferenceModulationCNN1D(
            channels=channels,
            dropout=dropout,
            analytic_prior_weight=analytic_prior_weight,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training.get("learning_rate", 5e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        loader = build_loader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=workers,
        )
        validation = _evaluate(
            model,
            validation_data,
            batch_size=batch_size,
            workers=workers,
            seed=seed,
            device=device,
        )
        initial_row = {
            "seed": seed,
            "epoch": 0,
            "stage": "cumulant_prior",
            **{
                "validation_%s" % key: value
                for key, value in validation.items()
            },
        }
        history.append(initial_row)
        print(json.dumps(initial_row, sort_keys=True))
        best_key, best_state, selected_seed, selected_epoch = _consider_checkpoint(
            model,
            validation,
            seed=seed,
            epoch=0,
            best_key=best_key,
            best_state=best_state,
            selected_seed=selected_seed,
            selected_epoch=selected_epoch,
        )
        seed_best_cross_entropy = float(validation["cross_entropy"])
        stale_epochs = 0
        for epoch in range(epochs):
            model.train()
            train_objectives = []
            train_cross_entropies = []
            train_residual_rms = []
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                if rotation_augmentation:
                    features = _random_carrier_rotation(features)
                optimizer.zero_grad(set_to_none=True)
                logits, residual, analytic_scores = model.forward_components(
                    features
                )
                cross_entropy = classification_loss(
                    logits,
                    targets,
                    label_smoothing=label_smoothing,
                )
                loss = cross_entropy + residual_l2_weight * torch.mean(
                    residual.square()
                )
                prior_correct = (
                    torch.argmax(analytic_scores.detach(), dim=1) == targets
                )
                if torch.any(prior_correct):
                    loss = loss + correct_prior_residual_weight * torch.mean(
                        residual[prior_correct].square()
                    )
                loss.backward()
                optimizer.step()
                train_objectives.append(float(loss.detach().cpu()))
                train_cross_entropies.append(
                    float(cross_entropy.detach().cpu())
                )
                train_residual_rms.append(
                    float(torch.sqrt(torch.mean(residual.detach().square())).cpu())
                )
            validation = _evaluate(
                model,
                validation_data,
                batch_size=batch_size,
                workers=workers,
                seed=seed,
                device=device,
            )
            row = {
                "seed": seed,
                "epoch": epoch + 1,
                "stage": "residual_training",
                "train_cross_entropy": float(
                    np.mean(train_cross_entropies)
                ),
                "train_objective": float(np.mean(train_objectives)),
                "train_residual_rms": float(np.mean(train_residual_rms)),
                **{"validation_%s" % key: value for key, value in validation.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True))
            best_key, best_state, selected_seed, selected_epoch = _consider_checkpoint(
                model,
                validation,
                seed=seed,
                epoch=epoch + 1,
                best_key=best_key,
                best_state=best_state,
                selected_seed=selected_seed,
                selected_epoch=selected_epoch,
            )
            if validation["cross_entropy"] < seed_best_cross_entropy - 1e-4:
                seed_best_cross_entropy = float(validation["cross_entropy"])
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                print(
                    "early stopping seed=%d after epoch=%d"
                    % (seed, epoch + 1)
                )
                break
    if best_state is None or selected_seed is None or selected_epoch is None:
        raise RuntimeError("Training produced no valid modulation classifier")
    selected = ReferenceModulationCNN1D(
        channels=channels,
        dropout=dropout,
        analytic_prior_weight=analytic_prior_weight,
    )
    selected.load_state_dict(best_state)
    component_path = Path(str(training.get("artifact_component_path") or "artifacts/modulation_classifier.onnx"))
    component_sha = export_onnx_classifier(selected, component_path, train_data.iq_frames.shape[1])
    Path("training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = _write_manifest(
        config,
        component_path=component_path,
        component_sha=component_sha,
        provenance={
            "selected_seed": selected_seed,
            "selected_epoch": selected_epoch,
            "best_validation_balanced_accuracy": -best_key[1],
            "best_validation_cross_entropy": best_key[0],
            "initialization_seeds": seeds,
            "train_capture_schema_sha256": list(train_data.capture_schema_sha256),
            "validation_capture_schema_sha256": list(validation_data.capture_schema_sha256),
            "split_record_fingerprints": split_record_fingerprints,
        },
    )
    print("exported modulation classifier ONNX: %s" % component_path)
    print("component sha256: %s" % component_sha)
    print("registered trained block artifact: %s" % manifest_path)
    return 0


def _consider_checkpoint(
    model,
    validation: dict[str, float],
    *,
    seed: int,
    epoch: int,
    best_key: tuple[float, float],
    best_state,
    selected_seed,
    selected_epoch,
):
    key = (
        float(validation["cross_entropy"]),
        -float(validation["balanced_accuracy"]),
    )
    if key < best_key:
        return (
            key,
            {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            int(seed),
            int(epoch),
        )
    return best_key, best_state, selected_seed, selected_epoch


def _random_carrier_rotation(features: torch.Tensor) -> torch.Tensor:
    """Apply a label-preserving random common phase to every training frame."""

    phase = (
        2.0
        * torch.pi
        * torch.rand(
            (features.shape[0], 1),
            device=features.device,
            dtype=features.dtype,
        )
        - torch.pi
    )
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    in_phase = features[..., 0]
    quadrature = features[..., 1]
    return torch.stack(
        [
            cosine * in_phase - sine * quadrature,
            sine * in_phase + cosine * quadrature,
        ],
        dim=2,
    )


def _evaluate(model, dataset, *, batch_size: int, workers: int, seed: int, device: torch.device) -> dict[str, float]:
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed, num_workers=workers)
    logits = []
    labels = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            logits.append(model(features.to(device)).cpu())
            labels.append(targets.cpu())
    all_logits = torch.cat(logits, dim=0)
    all_labels = torch.cat(labels, dim=0)
    return {
        "cross_entropy": float(classification_loss(all_logits, all_labels)),
        **classification_metrics(all_logits, all_labels),
    }


def _write_manifest(config: dict, *, component_path: Path, component_sha: str, provenance: dict) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    manifest_path = Path(str(training.get("artifact_manifest_path") or "trained_artifact.yaml"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_contract, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if source_contract.resolve() != contract_path.resolve():
        shutil.copyfile(source_contract, contract_path)
    try:
        portable_component = str(component_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError("AMC ONNX component must be inside the returned artifact package") from exc
    operation = str(training.get("artifact_operation") or recipe.get("classifier_operation") or "model.modulation_classifier_adapter")
    label = str(training.get("artifact_label") or "Learned modulation classifier")
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": str(training.get("artifact_id") or "external.modulation_classifier"),
        "name": str(training.get("artifact_name") or "Learned modulation classifier"),
        "label": label,
        "description": (
            "Portable fixed-frame BPSK/QPSK/16-QAM classifier for unknown "
            "carrier phase and residual frequency offset, selected on captured "
            "validation data."
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
                "id": "classifier",
                "role": "modulation_classifier",
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
                    "id": "modulation_classifier",
                    "component": "classifier",
                    "inputs": [
                        {
                            "name": "iq_ri",
                            "dtype": "float32",
                            "shape": ["batch", "sample", 2],
                            "layout": "batch_symbol_real_imag",
                            "semantic": "synchronized_received_complex_baseband_frame",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "class_logits",
                            "dtype": "float32",
                            "shape": ["batch", 3],
                            "layout": "batch_class",
                            "semantic": "ordered_logits_bpsk_qpsk_qam16",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": operation,
                "preferred_step_id": str(recipe.get("classifier_step") or ""),
                "label": label,
                "description": (
                    "Frozen I/Q-only automatic modulation classifier for the "
                    "blind-carrier scenario."
                ),
                "runtime_entrypoint": "modulation_classifier",
                "required_inputs": ["observation"],
                "params": {
                    "mode": "learned_artifact",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "modulation_classifier",
                },
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": str(config.get("training_template") or ""),
            "recipe_name": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "step_id": str(recipe.get("classifier_step") or ""),
        },
        "training": {
            "framework": str(config.get("framework") or "torch"),
            "architecture": str((config.get("model") or {}).get("architecture") or ""),
            "loss": str((config.get("objective") or {}).get("loss") or ""),
            "supervised_labels_used": True,
            **provenance,
        },
        "evaluation": {
            "command": "cd reference_training && python evaluate.py",
            "metrics_path": "reference_training/evaluation_metrics.json",
            "primary_metric": "balanced_accuracy",
            "split": "test",
        },
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def _find_training_contract(config: dict) -> tuple[Path, dict]:
    reference = dict(config.get("training_contract") or {})
    candidates = []
    if reference.get("path"):
        candidates.append(Path(str(reference["path"])))
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict) or str(payload.get("kind") or "") != "noema.trainable_slot_contract@1":
            raise ValueError("training_contract.yaml is not a Noema trainable-slot contract")
        return path, payload
    raise FileNotFoundError("training_contract.yaml is required to package the artifact")


def _device(training: dict) -> torch.device:
    requested = str(training.get("device") or "cuda_if_available")
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
