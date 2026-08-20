from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import yaml

from datamodule import (
    CsiCaptureDataset,
    build_loader,
    datasets_from_config,
    load_data_contract,
    materialize_data_contract_inventory,
)
from losses import (
    csi_metrics_over_snrs,
    hybrid_csi_objective,
    normalized_reconstruction_mse,
)
from model import ARCHITECTURE, ReferenceCsiFeedbackAutoencoder, export_onnx_components

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str((config.get("training") or {}).get("device", "cuda_if_available"))
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config: Mapping[str, Any]) -> ReferenceCsiFeedbackAutoencoder:
    data = dict(config.get("data") or {})
    link = dict(config.get("feedback_link") or {})
    model = dict(config.get("model") or {})
    base_channels = model.get("base_channels")
    if base_channels is None:
        base_channels = model.get("hidden_channels", 24)
    return ReferenceCsiFeedbackAutoencoder(
        tx_antennas=int(data.get("tx_antennas") or 8),
        subcarrier_count=int(data.get("subcarrier_count") or 32),
        feedback_dimension=int(link.get("feedback_dimension") or 64),
        bits_per_latent=int(link.get("bits_per_latent") or 8),
        clip_value=float(link.get("clip_value") or 1.0),
        feedback_mode=str(link.get("mode") or "uniform_quantized"),
        base_channels=int(base_channels),
        refinement_blocks=int(model.get("refinement_blocks") or 2),
        latent_activation=str(model.get("latent_activation") or "tanh"),
        global_linear_domain=str(
            model.get("global_linear_domain") or "angular_delay"
        ),
        use_angular_delay_transform=bool(
            model.get("use_angular_delay_transform", True)
        ),
        use_per_latent_adaptor=bool(
            model.get("use_per_latent_adaptor", True)
        ),
    )


def _klt_initialization_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = dict(config.get("model") or {})
    initialization = dict(model.get("initialization") or {})
    kind = str(initialization.get("kind") or "none").strip().lower()
    if kind not in {"none", "training_split_klt"}:
        raise ValueError(
            "model.initialization.kind must be none or training_split_klt"
        )
    percentile = float(initialization.get("scale_percentile") or 99.0)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("model.initialization.scale_percentile must be in (0,100]")
    return {
        "kind": kind,
        "scale_percentile": percentile,
    }


def _fit_training_split_klt_prior(
    dataset: CsiCaptureDataset,
    model: ReferenceCsiFeedbackAutoencoder,
    *,
    scale_percentile: float,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Fit a deterministic KLT prior using only the declared training split."""

    transformed = []
    transform = model.encoder.transform.cpu().eval()
    with torch.no_grad():
        for start in range(0, len(dataset), max(1, int(batch_size))):
            batch = torch.from_numpy(
                dataset.csi_ri[start : start + max(1, int(batch_size))]
            )
            prior_input = (
                batch
                if model.encoder.global_linear_domain == "frequency"
                else transform.analysis(batch)
            )
            transformed.append(prior_input.flatten(start_dim=1).cpu().numpy())
    flattened = np.concatenate(transformed, axis=0).astype(np.float64)
    feedback_dimension = int(model.encoder.feedback_dimension)
    if feedback_dimension > int(flattened.shape[1]):
        raise ValueError(
            "feedback dimension %d exceeds CSI dimension %d"
            % (feedback_dimension, flattened.shape[1])
        )
    mean = flattened.mean(axis=0)
    centered = flattened - mean
    covariance = centered.T @ centered / float(max(1, flattened.shape[0] - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:feedback_dimension]
    basis = eigenvectors[:, order].T
    coefficients = centered @ basis.T
    scale = np.percentile(
        np.abs(coefficients),
        float(scale_percentile),
        axis=0,
    )
    scale = np.maximum(scale, 1e-6)
    return {
        "mean": mean.astype(np.float32),
        "basis": basis.astype(np.float32),
        "scale": scale.astype(np.float32),
        "explained_variance": eigenvalues[order].astype(np.float32),
    }


def _initialize_model_from_prior(
    model: ReferenceCsiFeedbackAutoencoder,
    prior: Mapping[str, np.ndarray],
) -> None:
    model.initialize_klt_prior(
        mean=torch.from_numpy(np.asarray(prior["mean"], dtype=np.float32)),
        basis=torch.from_numpy(np.asarray(prior["basis"], dtype=np.float32)),
        scale=torch.from_numpy(np.asarray(prior["scale"], dtype=np.float32)),
    )


def _optimizer_parameter_groups(
    model: ReferenceCsiFeedbackAutoencoder,
    *,
    learning_rate: float,
    weight_decay: float,
    has_klt_prior: bool,
) -> list[Dict[str, Any]]:
    if not has_klt_prior:
        return [
            {
                "name": "learned",
                "params": list(model.parameters()),
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        ]
    prior_prefixes = (
        "encoder.global_projection.",
        "decoder.global_expansion.",
        "encoder.latent_adaptor.",
        "decoder.latent_adaptor.",
    )
    prior_parameters = []
    residual_parameters = []
    for name, parameter in model.named_parameters():
        destination = (
            prior_parameters
            if any(name.startswith(prefix) for prefix in prior_prefixes)
            else residual_parameters
        )
        destination.append(parameter)
    return [
        {
            "name": "residual",
            "params": residual_parameters,
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
        {
            "name": "klt_prior",
            "params": prior_parameters,
            "lr": 0.0,
            "weight_decay": 0.0,
        },
    ]


def _objective_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    objective = dict(config.get("objective") or {})
    weights = dict(objective.get("weights") or {})
    configured_loss_name = str(objective.get("loss") or "normalized_reconstruction_mse")
    loss_name = configured_loss_name.rsplit(".", 1)[-1]
    if loss_name not in {"normalized_reconstruction_mse", "hybrid_nmse_mrt_rate"}:
        raise ValueError("Unsupported CSI-feedback objective: %s" % loss_name)
    snr_values = objective.get("snr_db_values") or [0, 5, 10, 15, 20]
    if not isinstance(snr_values, (list, tuple)) or not snr_values:
        raise ValueError("objective.snr_db_values must be a non-empty list")
    return {
        "loss": loss_name,
        "configured_loss": "csi.%s" % loss_name,
        "nmse_weight": float(weights.get("nmse", objective.get("nmse_weight", 0.25))),
        "direction_weight": float(
            weights.get(
                "subcarrier_direction_cosine",
                weights.get(
                    "phase_invariant_cosine",
                    objective.get("direction_weight", 0.25),
                ),
            )
        ),
        "rate_weight": float(
            weights.get("mrt_rate", objective.get("rate_weight", 0.5))
        ),
        "quantization_weight": float(objective.get("quantization_weight", 0.05)),
        "snr_db_values": [float(value) for value in snr_values],
    }


def _learning_rate_at_epoch(
    epoch: int,
    *,
    epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
    warmup_epochs: int,
) -> float:
    """Linear warmup followed by cosine decay, evaluated once per epoch."""

    warmup = max(0, min(int(warmup_epochs), int(epochs)))
    if warmup and int(epoch) < warmup:
        return float(base_learning_rate) * float(int(epoch) + 1) / float(warmup)
    cosine_epochs = max(1, int(epochs) - warmup)
    progress = min(
        1.0, max(0.0, float(int(epoch) - warmup) / float(cosine_epochs - 1 or 1))
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        float(minimum_learning_rate)
        + (float(base_learning_rate) - float(minimum_learning_rate)) * cosine
    )


def _selection_key(
    validation: Mapping[str, float],
    *,
    loss_name: str,
) -> tuple[float, float]:
    if loss_name == "hybrid_nmse_mrt_rate":
        return (
            -float(validation["mean_spectral_efficiency_retention"]),
            float(validation["nmse"]),
        )
    return (
        float(validation["nmse"]),
        -float(validation["mean_spectral_efficiency_retention"]),
    )


def _validation_is_better(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
    *,
    loss_name: str,
    rate_tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    if loss_name != "hybrid_nmse_mrt_rate":
        return _selection_key(candidate, loss_name=loss_name) < _selection_key(
            incumbent, loss_name=loss_name
        )
    candidate_rate = float(candidate["mean_spectral_efficiency_retention"])
    incumbent_rate = float(incumbent["mean_spectral_efficiency_retention"])
    tolerance = max(0.0, float(rate_tolerance))
    if candidate_rate > incumbent_rate + tolerance:
        return True
    if abs(candidate_rate - incumbent_rate) <= tolerance:
        return float(candidate["nmse"]) < float(incumbent["nmse"])
    return False


def _validation_metrics(
    model: ReferenceCsiFeedbackAutoencoder,
    loader,
    *,
    device: torch.device,
    snr_db_values: list[float],
) -> Dict[str, float]:
    model.eval()
    reconstructions = []
    targets = []
    with torch.no_grad():
        for csi_ri in loader:
            csi_ri = csi_ri.to(device)
            reconstruction = model(csi_ri, quantize=True)["reconstruction"]
            reconstructions.append(reconstruction.cpu())
            targets.append(csi_ri.cpu())
    if not targets:
        raise ValueError("Validation capture is empty")
    metrics = csi_metrics_over_snrs(
        torch.cat(reconstructions, dim=0),
        torch.cat(targets, dim=0),
        snr_db_values=snr_db_values,
    )
    return {key: float(value) for key, value in metrics.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", ".", str(value or "csi.feedback").lower()).strip(".")
    return text or "csi.feedback"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _find_training_contract(config: Mapping[str, Any]) -> tuple[Path, Dict[str, Any]]:
    configured = str(
        ((config.get("training_contract") or {}).get("path") or "")
    ).strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        (Path("training_contract.yaml"), Path("../training_contract.yaml"))
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict):
            raise ValueError("training_contract.yaml must contain a mapping")
        if not str(payload.get("kind") or "").startswith(
            "noema.trainable_slot_contract"
        ):
            raise ValueError(
                "training_contract.yaml is not a Noema trainable-slot contract"
            )
        if not str(payload.get("id") or "").strip():
            raise ValueError("training_contract.yaml requires an id")
        return path, payload
    raise FileNotFoundError(
        "training_contract.yaml is required to package the returned CSI-feedback artifact"
    )


def _package_relative_path(component_path: Path, manifest_path: Path) -> str:
    try:
        return str(component_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError as exc:
        raise ValueError(
            "artifact components must be inside the trained-artifact package"
        ) from exc


def _partition_summary(dataset: CsiCaptureDataset) -> Dict[str, Any]:
    digest = hashlib.sha256()
    for item in sorted(dataset.record_fingerprints):
        digest.update(item.encode("ascii"))
    return {
        "split": dataset.split,
        "records": len(dataset),
        "capture_schema_sha256": list(dataset.capture_sha256),
        "record_set_sha256": digest.hexdigest(),
    }


def _write_artifact_manifest(
    config: Mapping[str, Any],
    encoder_path: Path,
    decoder_path: Path,
    component_hashes: Mapping[str, str],
    *,
    best_seed: int,
    validation: Mapping[str, float],
    train_dataset: CsiCaptureDataset,
    validation_dataset: CsiCaptureDataset,
) -> Dict[str, Any]:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    data = dict(config.get("data") or {})
    link = dict(config.get("feedback_link") or {})
    model_config = dict(config.get("model") or {})
    objective = _objective_settings(config)
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    contract_source, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if contract_source.resolve() != contract_path.resolve():
        shutil.copyfile(contract_source, contract_path)
    data_contract_path = Path(
        str(data.get("contract_path") or "../data_contract.yaml")
    ).resolve()
    if not data_contract_path.is_file():
        raise FileNotFoundError("CSI data contract is required to package the artifact")
    if data_contract_path.parent != manifest_path.parent.resolve():
        raise ValueError(
            "CSI data contract must remain beside the returned trained artifact"
        )

    feedback_dimension = int(link.get("feedback_dimension") or 64)
    bits_per_latent = int(link.get("bits_per_latent") or 8)
    tx_antennas = int(data.get("tx_antennas") or 8)
    subcarriers = int(data.get("subcarrier_count") or 32)
    artifact_id = str(
        training.get("artifact_id")
        or "%s.csi.feedback.codec" % _safe_id(recipe.get("name", "csi.feedback"))
    )
    artifact_name = str(training.get("artifact_name") or "Learned CSI feedback codec")
    artifact_label = str(training.get("artifact_label") or artifact_name)
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": artifact_id,
        "name": artifact_name,
        "label": artifact_label,
        "description": "Jointly trained limited-feedback CSI encoder and decoder.",
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(
                contract.get("version") or contract.get("schema_version") or 1
            ),
            "path": "training_contract.yaml",
            "sha256": _canonical_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "encoder",
                "role": "csi_feedback_encoder",
                "path": _package_relative_path(encoder_path, manifest_path),
                "sha256": str(component_hashes["encoder_sha256"]),
                "format": "onnx",
            },
            {
                "id": "decoder",
                "role": "csi_feedback_decoder",
                "path": _package_relative_path(decoder_path, manifest_path),
                "sha256": str(component_hashes["decoder_sha256"]),
                "format": "onnx",
            },
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "encoder",
                    "component": "encoder",
                    "inputs": [
                        {
                            "name": "csi_ri",
                            "dtype": "float32",
                            "shape": ["batch", 2, "tx_antenna", "subcarrier"],
                            "layout": "N_ri_tx_subcarrier",
                            "semantic": "normalized_complex_CSI_real_imag",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "feedback_code",
                            "dtype": "float32",
                            "shape": ["batch", "feedback_dimension"],
                            "layout": "N_feedback_latent",
                            "semantic": "real_feedback_latents_before_frozen_link",
                        }
                    ],
                },
                {
                    "id": "decoder",
                    "component": "decoder",
                    "inputs": [
                        {
                            "name": "feedback_code",
                            "dtype": "float32",
                            "shape": ["batch", "feedback_dimension"],
                            "layout": "N_feedback_latent",
                            "semantic": "received_feedback_latents_after_frozen_link",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "csi_hat_ri",
                            "dtype": "float32",
                            "shape": ["batch", 2, "tx_antenna", "subcarrier"],
                            "layout": "N_ri_tx_subcarrier",
                            "semantic": "reconstructed_complex_CSI_real_imag",
                        }
                    ],
                },
            ],
        },
        "application": {"mode": "all_group_bindings"},
        "compatible_operations": [
            {
                "operation": "model.csi_feedback_encoder",
                "binding_group": "csi_feedback_codec",
                "role": "encoder",
                "preferred_step_id": str(
                    recipe.get("encoder_step") or "feedback_encoder"
                ),
                "label": artifact_label,
                "runtime_entrypoint": "encoder",
                "required_inputs": ["csi"],
                "params": {
                    "runtime": "learned_artifact",
                    "feedback_dimension": feedback_dimension,
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "encoder",
                },
            },
            {
                "operation": "model.csi_feedback_decoder",
                "binding_group": "csi_feedback_codec",
                "role": "decoder",
                "preferred_step_id": str(
                    recipe.get("decoder_step") or "feedback_decoder"
                ),
                "label": artifact_label,
                "runtime_entrypoint": "decoder",
                "required_inputs": ["received_code"],
                "params": {
                    "runtime": "learned_artifact",
                    "feedback_dimension": feedback_dimension,
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "decoder",
                },
            },
        ],
        "source": {
            "origin": "noema_optional_demo_training_scaffold",
            "recipe": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "training_template": str(config.get("training_template") or ""),
            "data_contract": {
                "path": "data_contract.yaml",
                "sha256": str(data.get("contract_sha256") or ""),
                "file_sha256": str(data.get("contract_file_sha256") or ""),
            },
        },
        "training": {
            "framework": "torch",
            "objective": str(objective["configured_loss"]),
            "objective_configuration": {
                "nmse_weight": float(objective["nmse_weight"]),
                "direction_weight": float(objective["direction_weight"]),
                "rate_weight": float(objective["rate_weight"]),
                "quantization_weight": float(objective["quantization_weight"]),
                "snr_db_values": list(objective["snr_db_values"]),
            },
            "architecture": ARCHITECTURE,
            "initialization": dict(model_config.get("initialization") or {}),
            "latent_activation": str(
                model_config.get("latent_activation") or "tanh"
            ),
            "global_linear_domain": str(
                model_config.get("global_linear_domain") or "angular_delay"
            ),
            "best_seed": int(best_seed),
            "feedback_constraint": {
                "mode": str(link.get("mode") or "uniform_quantized"),
                "feedback_dimension": feedback_dimension,
                "bits_per_latent": bits_per_latent,
                "feedback_bits_per_sample": feedback_dimension * bits_per_latent,
                "clip_value": float(link.get("clip_value") or 1.0),
            },
            "fixed_shape": {
                "tx_antennas": tx_antennas,
                "subcarrier_count": subcarriers,
            },
            "data_partitions": {
                "train": _partition_summary(train_dataset),
                "validation": _partition_summary(validation_dataset),
                "test_capture_used": False,
            },
        },
        "evaluation": {
            "selection_metric": (
                "validation_mean_spectral_efficiency_retention"
                if objective["loss"] == "hybrid_nmse_mrt_rate"
                else "validation_nmse"
            ),
            "selection_direction": (
                "maximize_rate_retention_then_minimize_nmse"
                if objective["loss"] == "hybrid_nmse_mrt_rate"
                else "minimize_nmse_then_maximize_rate_retention"
            ),
            "validation": {key: float(value) for key, value in validation.items()},
        },
    }
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return payload


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    if not isinstance(config, dict):
        raise ValueError("train_config.yaml must contain a mapping")
    load_data_contract(config)
    datasets = datasets_from_config(config, include_test=False)
    config = materialize_data_contract_inventory(config)
    training = dict(config.get("training") or {})
    objective = _objective_settings(config)
    batch_size = int(training.get("batch_size") or 256)
    num_workers = int(training.get("num_workers") or 0)
    device = _device(config)
    validation_loader = build_loader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    epochs = int(training.get("epochs") or 180)
    patience = int(
        training.get(
            "patience",
            training.get("early_stopping_patience", 35),
        )
    )
    learning_rate = float(training.get("learning_rate") or 1e-3)
    minimum_learning_rate = float(training.get("min_learning_rate") or 1e-5)
    learning_rate_warmup_epochs = int(
        training.get(
            "lr_warmup_epochs",
            training.get("learning_rate_warmup_epochs", 5),
        )
    )
    quantization_warmup_epochs = int(training.get("quantization_warmup_epochs") or 0)
    nmse_pretraining_epochs = int(training.get("nmse_pretraining_epochs") or 0)
    prior_freeze_epochs = int(training.get("prior_freeze_epochs") or 0)
    prior_learning_rate_scale = float(
        0.1
        if training.get("prior_learning_rate_scale") is None
        else training["prior_learning_rate_scale"]
    )
    checkpoint_rate_tolerance = float(
        1e-5
        if training.get("checkpoint_rate_tolerance") is None
        else training["checkpoint_rate_tolerance"]
    )
    if nmse_pretraining_epochs < 0 or nmse_pretraining_epochs > epochs:
        raise ValueError("training.nmse_pretraining_epochs must be in [0,epochs]")
    if prior_freeze_epochs < 0 or prior_freeze_epochs > epochs:
        raise ValueError("training.prior_freeze_epochs must be in [0,epochs]")
    if not 0.0 <= prior_learning_rate_scale <= 1.0:
        raise ValueError("training.prior_learning_rate_scale must be in [0,1]")
    if checkpoint_rate_tolerance < 0.0:
        raise ValueError("training.checkpoint_rate_tolerance must be non-negative")
    gradient_clip_norm = float(training.get("gradient_clip_norm") or 1.0)
    weight_decay = float(training.get("weight_decay") or 1e-5)
    optimizer_name = str(training.get("optimizer") or "AdamW").lower()
    if optimizer_name != "adamw":
        raise ValueError("This CSI demonstration project supports optimizer=AdamW")
    scheduler_name = str(
        training.get("scheduler")
        or training.get("learning_rate_schedule")
        or "cosine_annealing"
    ).lower()
    if scheduler_name not in {"cosine", "cosine_annealing"}:
        raise ValueError("This CSI demonstration project supports a cosine scheduler")
    seeds = [int(value) for value in training.get("initialization_seeds") or [23, 41]]
    history = []
    selected = None
    initialization = _klt_initialization_settings(config)
    prior = None
    if initialization["kind"] == "training_split_klt":
        print(
            "fitting training-only KLT prior from %d CSI records"
            % len(datasets["train"])
        )
        prior = _fit_training_split_klt_prior(
            datasets["train"],
            _build_model(config),
            scale_percentile=float(initialization["scale_percentile"]),
            batch_size=batch_size,
        )
        retained_variance = float(
            np.sum(prior["explained_variance"])
            / max(
                1e-12,
                np.sum(
                    np.var(
                        datasets["train"].csi_ri.reshape(len(datasets["train"]), -1),
                        axis=0,
                        ddof=1,
                    )
                ),
            )
        )
        print(
            "KLT prior ready: %d latents, scale percentile %.3g, "
            "retained variance %.4f"
            % (
                len(prior["scale"]),
                initialization["scale_percentile"],
                retained_variance,
            )
        )

    for seed in seeds:
        _seed_everything(seed)
        model = _build_model(config)
        if prior is not None:
            _initialize_model_from_prior(model, prior)
        model = model.to(device)
        train_loader = build_loader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
        optimizer = torch.optim.AdamW(
            _optimizer_parameter_groups(
                model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                has_klt_prior=prior is not None,
            ),
            lr=learning_rate,
        )
        best_state = None
        best_validation = None
        prior_validation = None
        if prior is not None:
            prior_validation = _validation_metrics(
                model,
                validation_loader,
                device=device,
                snr_db_values=objective["snr_db_values"],
            )
            best_validation = dict(prior_validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            history.append(
                {
                    "seed": seed,
                    "epoch": 0,
                    "stage": "training_split_klt_prior",
                    "learning_rate": 0.0,
                    "prior_learning_rate": 0.0,
                    "quantization_enabled": True,
                    **{
                        "validation_%s" % key: value
                        for key, value in prior_validation.items()
                    },
                }
            )
            print(
                "seed=%d epoch=0 stage=klt_prior validation_nmse=%.7f "
                "validation_mean_rate_retention=%.7f"
                % (
                    seed,
                    prior_validation["nmse"],
                    prior_validation["mean_spectral_efficiency_retention"],
                )
            )
        stale_epochs = 0
        for epoch in range(epochs):
            epoch_learning_rate = _learning_rate_at_epoch(
                epoch,
                epochs=epochs,
                base_learning_rate=learning_rate,
                minimum_learning_rate=minimum_learning_rate,
                warmup_epochs=learning_rate_warmup_epochs,
            )
            for parameter_group in optimizer.param_groups:
                if parameter_group.get("name") == "klt_prior":
                    parameter_group["lr"] = (
                        0.0
                        if epoch < prior_freeze_epochs
                        else epoch_learning_rate * prior_learning_rate_scale
                    )
                else:
                    parameter_group["lr"] = epoch_learning_rate
            quantize = epoch >= quantization_warmup_epochs
            nmse_pretraining = (
                objective["loss"] == "hybrid_nmse_mrt_rate"
                and epoch < nmse_pretraining_epochs
            )
            stage = (
                "quantized_nmse_pretraining"
                if nmse_pretraining
                else "mrt_rate_finetuning"
                if objective["loss"] == "hybrid_nmse_mrt_rate"
                else "nmse_training"
            )
            model.train()
            weighted_loss = 0.0
            weighted_components: Dict[str, float] = {}
            seen = 0
            for csi_ri in train_loader:
                csi_ri = csi_ri.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(csi_ri, quantize=quantize)
                if objective["loss"] == "hybrid_nmse_mrt_rate":
                    stage_weights = (
                        {
                            "nmse_weight": 1.0,
                            "direction_weight": 0.0,
                            "rate_weight": 0.0,
                            "quantization_weight": objective[
                                "quantization_weight"
                            ],
                        }
                        if nmse_pretraining
                        else objective
                    )
                    components = hybrid_csi_objective(
                        outputs["reconstruction"],
                        csi_ri,
                        feedback_code=outputs["feedback_code"],
                        received_code=outputs["received_code"],
                        snr_db_values=objective["snr_db_values"],
                        nmse_weight=stage_weights["nmse_weight"],
                        direction_weight=stage_weights["direction_weight"],
                        rate_weight=stage_weights["rate_weight"],
                        quantization_weight=stage_weights[
                            "quantization_weight"
                        ],
                    )
                    loss = components["loss"]
                else:
                    loss = normalized_reconstruction_mse(
                        outputs["reconstruction"], csi_ri
                    )
                    components = {"loss": loss, "nmse": loss}
                loss.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
                current_batch = int(csi_ri.shape[0])
                weighted_loss += float(loss.detach().cpu()) * current_batch
                for key, value in components.items():
                    weighted_components[key] = weighted_components.get(key, 0.0) + (
                        float(value.detach().cpu()) * current_batch
                    )
                seen += current_batch
            validation = _validation_metrics(
                model,
                validation_loader,
                device=device,
                snr_db_values=objective["snr_db_values"],
            )
            train_loss = weighted_loss / max(1, seen)
            row = {
                "seed": seed,
                "epoch": epoch + 1,
                "stage": stage,
                "learning_rate": epoch_learning_rate,
                "prior_learning_rate": next(
                    (
                        float(group["lr"])
                        for group in optimizer.param_groups
                        if group.get("name") == "klt_prior"
                    ),
                    epoch_learning_rate,
                ),
                "quantization_enabled": quantize,
                "train_loss": train_loss,
                **{
                    "train_%s" % key: value / max(1, seen)
                    for key, value in weighted_components.items()
                    if key != "loss"
                },
                **{"validation_%s" % key: value for key, value in validation.items()},
            }
            history.append(row)
            print(
                "seed=%d epoch=%d stage=%s train_loss=%.7f validation_nmse=%.7f "
                "validation_mean_rate_retention=%.7f lr=%.3g qat=%s"
                % (
                    seed,
                    epoch + 1,
                    stage,
                    train_loss,
                    validation["nmse"],
                    validation["mean_spectral_efficiency_retention"],
                    epoch_learning_rate,
                    "on" if quantize else "warmup",
                )
            )
            if _validation_is_better(
                validation,
                best_validation,
                loss_name=objective["loss"],
                rate_tolerance=checkpoint_rate_tolerance,
            ):
                best_validation = dict(validation)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                selection_stage_started = (
                    objective["loss"] != "hybrid_nmse_mrt_rate"
                    or epoch + 1 > nmse_pretraining_epochs
                )
                if selection_stage_started:
                    stale_epochs += 1
                    if patience >= 0 and stale_epochs >= patience:
                        break
            if (
                objective["loss"] == "hybrid_nmse_mrt_rate"
                and epoch + 1 == nmse_pretraining_epochs
            ):
                stale_epochs = 0
        if best_state is None or best_validation is None:
            raise RuntimeError("CSI-feedback training did not produce a candidate")
        candidate = _build_model(config)
        candidate.load_state_dict(best_state, strict=True)
        if _validation_is_better(
            best_validation,
            None if selected is None else selected["validation"],
            loss_name=objective["loss"],
            rate_tolerance=checkpoint_rate_tolerance,
        ):
            selected = {
                "model": candidate,
                "seed": seed,
                "validation": best_validation,
                "prior_validation": prior_validation,
            }

    if selected is None or not math.isfinite(float(selected["validation"]["nmse"])):
        raise RuntimeError(
            "CSI-feedback training did not produce a finite validation NMSE"
        )
    component_paths = [
        Path(str(item)) for item in training.get("artifact_component_paths") or []
    ]
    if len(component_paths) != 2:
        manifest_path = Path(
            str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
        )
        component_paths = [
            manifest_path.parent / "artifacts" / "csi_feedback_encoder.onnx",
            manifest_path.parent / "artifacts" / "csi_feedback_decoder.onnx",
        ]
    component_hashes = export_onnx_components(
        selected["model"], component_paths[0], component_paths[1]
    )
    _write_artifact_manifest(
        config,
        component_paths[0],
        component_paths[1],
        component_hashes,
        best_seed=int(selected["seed"]),
        validation=selected["validation"],
        train_dataset=datasets["train"],
        validation_dataset=datasets["validation"],
    )
    history_path = Path(str(training.get("history_path") or "training_history.json"))
    _write_json(
        history_path,
        {
            "schema_version": 1,
            "objective": str(objective["configured_loss"]),
            "objective_configuration": {
                "nmse_weight": objective["nmse_weight"],
                "direction_weight": objective["direction_weight"],
                "rate_weight": objective["rate_weight"],
                "quantization_weight": objective["quantization_weight"],
                "snr_db_values": objective["snr_db_values"],
            },
            "checkpoint_selection": (
                "maximum_validation_mean_spectral_efficiency_retention_then_minimum_nmse"
                if objective["loss"] == "hybrid_nmse_mrt_rate"
                else "minimum_validation_nmse_then_maximum_mean_rate_retention"
            ),
            "optimizer": "AdamW",
            "learning_rate_schedule": "linear_warmup_then_cosine",
            "quantization_warmup_epochs": quantization_warmup_epochs,
            "nmse_pretraining_epochs": nmse_pretraining_epochs,
            "prior_freeze_epochs": prior_freeze_epochs,
            "prior_learning_rate_scale": prior_learning_rate_scale,
            "checkpoint_rate_tolerance": checkpoint_rate_tolerance,
            "initialization": initialization,
            "gradient_clip_norm": gradient_clip_norm,
            "best_seed": int(selected["seed"]),
            "best_validation": selected["validation"],
            "klt_prior_validation": selected["prior_validation"],
            "test_capture_used": False,
            "epochs": history,
        },
    )
    print("exported encoder ONNX: %s" % component_paths[0])
    print("encoder sha256: %s" % component_hashes["encoder_sha256"])
    print("exported decoder ONNX: %s" % component_paths[1])
    print("decoder sha256: %s" % component_hashes["decoder_sha256"])
    print("wrote trained artifact: %s" % training.get("artifact_manifest_path"))
    if selected["prior_validation"] is not None:
        print(
            "selected validation delta versus KLT prior: rate_retention=%+.7f "
            "nmse=%+.7f"
            % (
                float(
                    selected["validation"][
                        "mean_spectral_efficiency_retention"
                    ]
                )
                - float(
                    selected["prior_validation"][
                        "mean_spectral_efficiency_retention"
                    ]
                ),
                float(selected["validation"]["nmse"])
                - float(selected["prior_validation"]["nmse"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
