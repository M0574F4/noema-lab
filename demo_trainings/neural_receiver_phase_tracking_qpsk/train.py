from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml

from datamodule import build_loader, load_capture_dataset, split_integrity_report
from losses import (
    bit_scores_from_residual,
    masked_bit_bce,
    masked_circular_phase_loss,
    phase_tracking_loss,
)
from model import build_receiver, export_onnx_receiver, receiver_candidates
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


CHECKPOINT_SELECTION_ORDER = (
    "material_validation_ber_improvement_required",
    "minimum_per_snr_pilot_smoothing_regressions",
    "minimum_validation_ber",
    "minimum_validation_phase_rmse",
    "minimum_worst_per_snr_ber_regression",
    "minimum_validation_phase_loss",
    "configured_candidate_order",
    "configured_seed_order",
    "earliest_epoch",
)


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data_config = dict(config.get("data") or {})
    recipe_config = dict(config.get("recipe") or {})
    training = dict(config.get("training") or {})
    common = {
        "feature_tap": str(data_config.get("feature_tap") or "rx_symbols"),
        "pilot_context_tap": str(
            data_config.get("pilot_context_tap") or "pilot_context"
        ),
        "target_tap": str(data_config.get("target_tap") or "target_bits"),
        "phase_truth_tap": str(data_config.get("phase_truth_tap") or ""),
        "feature_reference": str(
            data_config.get("feature_reference")
            or recipe_config.get("feature_reference")
            or ""
        ),
        "pilot_context_reference": str(
            data_config.get("pilot_context_reference")
            or recipe_config.get("pilot_context_reference")
            or ""
        ),
        "target_reference": str(
            data_config.get("target_reference")
            or recipe_config.get("target_reference")
            or ""
        ),
        "phase_truth_reference": str(
            data_config.get("phase_truth_reference") or ""
        ),
        "pilot_smoothing_neighbors": int(
            data_config.get("pilot_smoothing_neighbors") or 5
        ),
    }
    train_data = load_capture_dataset(
        data_config.get("train_capture_dirs") or [],
        expected_split="train",
        **common,
    )
    validation_data = load_capture_dataset(
        data_config.get("validation_capture_dirs") or [],
        expected_split="validation",
        **common,
    )
    split_integrity = split_integrity_report(
        {"train": train_data, "validation": validation_data}
    )
    if not np.any(train_data.residual_phase_mask):
        raise ValueError(
            "The phase-tracking reference trainer requires the capture-only "
            "residual phase target. Recapture the bundle with phase_truth enabled."
        )

    candidates = receiver_candidates(dict(config.get("model") or {}))
    seeds = [int(value) for value in training.get("initialization_seeds", [23, 41])]
    if not seeds:
        raise ValueError("training.initialization_seeds must not be empty")
    epochs = max(1, int(training.get("epochs", 48)))
    phase_pretraining_epochs = min(
        max(0, int(training.get("phase_pretraining_epochs", 16))),
        max(0, epochs - 1),
    )
    batch_size = max(1, int(training.get("batch_size", 4)))
    workers = max(0, int(training.get("num_workers", 0)))
    phase_weight = max(0.0, float(training.get("phase_loss_weight", 1.0)))
    minimum_ber_improvement = max(
        0.0,
        float(training.get("minimum_validation_ber_improvement", 0.001)),
    )
    early_stopping_patience = max(
        1,
        int(training.get("early_stopping_patience", 8)),
    )
    minimum_epochs = max(1, int(training.get("minimum_epochs", 8)))
    per_snr_margin = max(
        0.0,
        float(training.get("per_snr_noninferiority_margin", 0.001)),
    )
    device = _device(training)
    baseline_decisions = validation_data.receiver_features[..., :2] < 0.0
    baseline_ber = _masked_ber(baseline_decisions, validation_data)
    baseline_per_snr = _pilot_smoothing_ber_by_snr(validation_data)

    best_rank = None
    best_state = None
    selected_candidate = None
    selected_seed = None
    selected_epoch = None
    best_ber = float("inf")
    best_bce = float("inf")
    best_phase_loss = float("inf")
    best_phase_rmse = float("inf")
    best_per_snr = {}
    best_regression = {}
    selected_is_fallback = False
    history = []
    for candidate_index, candidate in enumerate(candidates):
        for seed_index, seed in enumerate(seeds):
            _seed_everything(seed)
            model = build_receiver(candidate).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training.get("learning_rate", 2e-3)),
                weight_decay=float(training.get("weight_decay", 1e-5)),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=max(2, early_stopping_patience // 2),
                min_lr=float(training.get("minimum_learning_rate", 1e-5)),
            )
            loader = build_loader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                num_workers=workers,
            )
            trial_best_rank = None
            trial_stage = None
            epochs_without_improvement = 0
            for epoch in range(epochs + 1):
                phase_only = 0 < epoch <= phase_pretraining_epochs
                stage = (
                    "pilot_smoothing_baseline"
                    if epoch == 0
                    else "phase_pretraining"
                    if phase_only
                    else "joint_finetuning"
                )
                if stage != trial_stage:
                    trial_stage = stage
                    trial_best_rank = None
                    epochs_without_improvement = 0
                train_total = None
                train_bce = None
                train_phase = None
                curriculum_weight = _curriculum_high_snr_weight(
                    training,
                    epoch=max(1, epoch),
                )
                if epoch > 0:
                    model.train()
                    total_losses = []
                    bit_losses = []
                    phase_losses = []
                    for batch in loader:
                        (
                            features,
                            targets,
                            data_mask,
                            residual_phase_target,
                            residual_phase_mask,
                            packet_snr_db,
                        ) = _unpack_batch(batch)
                        features = features.to(device)
                        targets = targets.to(device)
                        data_mask = data_mask.to(device)
                        residual_phase_target = residual_phase_target.to(device)
                        residual_phase_mask = residual_phase_mask.to(device)
                        packet_weights = _curriculum_packet_weights(
                            packet_snr_db,
                            training,
                            epoch=epoch,
                            device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        residual_phase = model(features)
                        loss, bit_loss, phase_loss, _ = phase_tracking_loss(
                            features,
                            targets,
                            data_mask,
                            residual_phase,
                            residual_phase_target,
                            residual_phase_mask,
                            phase_weight=phase_weight,
                            packet_weights=packet_weights,
                            phase_only=phase_only,
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        total_losses.append(float(loss.detach().cpu()))
                        bit_losses.append(float(bit_loss.detach().cpu()))
                        phase_losses.append(float(phase_loss.detach().cpu()))
                    train_total = float(np.mean(total_losses))
                    train_bce = float(np.mean(bit_losses))
                    train_phase = float(np.mean(phase_losses))
                (
                    validation_bce,
                    validation_ber,
                    validation_per_snr,
                    validation_phase_loss,
                    validation_phase_rmse,
                ) = _evaluate(
                    model,
                    validation_data,
                    batch_size=batch_size,
                    workers=workers,
                    seed=seed,
                    device=device,
                )
                regression = _per_snr_regression(
                    validation_per_snr,
                    baseline_per_snr,
                    margin=per_snr_margin,
                )
                ber_improvement = baseline_ber - validation_ber
                accepted = bool(
                    stage == "joint_finetuning"
                    and ber_improvement >= minimum_ber_improvement
                    and regression["regressed_bin_count"] == 0
                )
                row = {
                    "candidate": candidate.to_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "training_stage": stage,
                    "high_snr_curriculum_weight": curriculum_weight,
                    "train_loss": train_total,
                    "train_bce": train_bce,
                    "train_circular_phase_loss": train_phase,
                    "validation_bce": validation_bce,
                    "validation_ber": validation_ber,
                    "validation_circular_phase_loss": validation_phase_loss,
                    "validation_phase_rmse_rad": validation_phase_rmse,
                    "validation_ber_improvement_over_pilot_smoothing": ber_improvement,
                    "accepted_as_material_improvement": accepted,
                    "validation_ber_by_snr_db": validation_per_snr,
                    "pilot_smoothing_ber": baseline_ber,
                    "pilot_smoothing_ber_by_snr_db": baseline_per_snr,
                    "per_snr_regression": regression,
                }
                history.append(row)
                print(
                    "candidate=%s seed=%d epoch=%d stage=%s train_bce=%s "
                    "validation_ber=%.7g phase_rmse=%.7g improvement=%.7g "
                    "accepted=%s regressed_snr_bins=%d"
                    % (
                        candidate.id,
                        seed,
                        epoch,
                        stage,
                        "baseline" if train_bce is None else "%.7g" % train_bce,
                        validation_ber,
                        validation_phase_rmse,
                        ber_improvement,
                        accepted,
                        regression["regressed_bin_count"],
                    )
                )
                rank = _checkpoint_rank(
                    validation_ber=validation_ber,
                    validation_phase_loss=validation_phase_loss,
                    validation_phase_rmse=validation_phase_rmse,
                    regressed_snr_bins=regression["regressed_bin_count"],
                    worst_per_snr_regression=regression["worst_positive_delta"],
                    candidate_index=candidate_index,
                    seed_index=seed_index,
                    epoch_index=epoch,
                )
                state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                if accepted and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_ber = validation_ber
                    best_bce = validation_bce
                    best_phase_loss = validation_phase_loss
                    best_phase_rmse = validation_phase_rmse
                    best_per_snr = dict(validation_per_snr)
                    best_regression = dict(regression)
                    selected_candidate = candidate
                    selected_seed = seed
                    selected_epoch = epoch
                    selected_is_fallback = False
                    best_state = state
                stage_rank = (
                    (validation_phase_rmse, validation_phase_loss, epoch)
                    if phase_only
                    else rank
                )
                if trial_best_rank is None or stage_rank < trial_best_rank:
                    trial_best_rank = stage_rank
                    epochs_without_improvement = 0
                elif epoch > 0:
                    epochs_without_improvement += 1
                if stage == "joint_finetuning":
                    scheduler.step(validation_ber)
                if (
                    not phase_only
                    and epoch - phase_pretraining_epochs >= minimum_epochs
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        "early stopping candidate=%s seed=%d after epoch=%d"
                        % (candidate.id, seed, epoch)
                    )
                    break

    _require_learned_export_selection(
        best_state=best_state,
        selected_candidate=selected_candidate,
        selected_seed=selected_seed,
        selected_epoch=selected_epoch,
        selection_history=history,
    )
    selected = build_receiver(selected_candidate)
    selected.load_state_dict(best_state)
    component_path = Path(
        str(
            training.get("artifact_component_path")
            or "../artifacts/phase_tracking_receiver.onnx"
        )
    )
    component_sha = export_onnx_receiver(selected, component_path)
    provenance = {
        "selected_candidate": selected_candidate.to_dict(),
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "best_validation_ber": best_ber,
        "best_validation_bce": best_bce,
        "best_validation_circular_phase_loss": best_phase_loss,
        "best_validation_phase_rmse_rad": best_phase_rmse,
        "best_validation_ber_by_snr_db": best_per_snr,
        "selected_per_snr_regression": best_regression,
        "selected_epoch_is_pilot_smoothing_fallback": selected_is_fallback,
        "training_performed": True,
        "learned_checkpoint": True,
        "accepted_as_material_improvement": True,
        "candidate_search": [candidate.to_dict() for candidate in candidates],
        "selection_order": list(CHECKPOINT_SELECTION_ORDER),
        "initialization_seeds": seeds,
        "train_capture_schema_sha256": list(train_data.capture_schema_sha256),
        "validation_capture_schema_sha256": list(
            validation_data.capture_schema_sha256
        ),
        "split_integrity": split_integrity,
        "phase_truth_used_as_training_target": bool(
            np.any(train_data.residual_phase_mask)
        ),
        "phase_truth_used_at_runtime": False,
        "pilot_smoothing_baseline_ber": baseline_ber,
        "pilot_smoothing_baseline_ber_by_snr_db": baseline_per_snr,
        "pilot_smoothing_baseline_preserved_at_initialization": True,
        "minimum_validation_ber_improvement": minimum_ber_improvement,
        "per_snr_noninferiority_margin": per_snr_margin,
        "phase_pretraining_epochs": phase_pretraining_epochs,
        "phase_loss_weight": phase_weight,
        "high_snr_curriculum": dict(training.get("high_snr_curriculum") or {}),
    }
    Path("training_history.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "noema.phase_tracking_receiver_training_history",
                "split_integrity": split_integrity,
                "selection": {
                    "candidate": selected_candidate.to_dict(),
                    "seed": selected_seed,
                    "epoch": selected_epoch,
                    "validation_ber": best_ber,
                    "validation_bce": best_bce,
                    "validation_circular_phase_loss": best_phase_loss,
                    "validation_phase_rmse_rad": best_phase_rmse,
                    "validation_ber_by_snr_db": best_per_snr,
                    "per_snr_regression": best_regression,
                    "pilot_smoothing_fallback": selected_is_fallback,
                    "training_performed": True,
                    "learned_checkpoint": True,
                    "accepted_as_material_improvement": True,
                },
                "trials": history,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        config,
        component_path=component_path,
        component_sha=component_sha,
        provenance=provenance,
    )
    print(
        "selected candidate=%s seed=%d epoch=%d validation_ber=%.7g "
        "phase_rmse=%.7g fallback=%s"
        % (
            selected_candidate.id,
            selected_seed,
            selected_epoch,
            best_ber,
            best_phase_rmse,
            selected_is_fallback,
        )
    )
    print("exported phase-tracking receiver ONNX: %s" % component_path)
    print("component sha256: %s" % component_sha)
    print("registered trained block artifact: %s" % manifest_path)
    return 0


def _require_learned_export_selection(
    *,
    best_state,
    selected_candidate,
    selected_seed,
    selected_epoch,
    selection_history: list[Mapping[str, object]],
) -> None:
    """Refuse to package an initialization or pilot-smoothing-only checkpoint."""

    if (
        best_state is None
        or selected_candidate is None
        or selected_seed is None
        or isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch <= 0
    ):
        raise RuntimeError(
            "No trained phase-tracking checkpoint met the material validation "
            "improvement and per-SNR noninferiority criteria; refusing to export "
            "the untrained epoch-zero pilot-smoothing initialization as learned."
        )
    candidate_id = str(getattr(selected_candidate, "id", "") or "")
    accepted = any(
        bool(row.get("accepted_as_material_improvement"))
        and row.get("epoch") == selected_epoch
        and row.get("seed") == selected_seed
        and str((row.get("candidate") or {}).get("id") or "") == candidate_id
        for row in selection_history
        if isinstance(row, Mapping)
        and isinstance(row.get("candidate"), Mapping)
    )
    if not accepted:
        raise RuntimeError(
            "Selected phase-tracking checkpoint is not backed by an accepted "
            "trained validation-history row; refusing learned-artifact export."
        )


def _checkpoint_rank(
    *,
    validation_ber: float,
    validation_phase_loss: float,
    validation_phase_rmse: float,
    regressed_snr_bins: int = 0,
    worst_per_snr_regression: float = 0.0,
    candidate_index: int,
    seed_index: int,
    epoch_index: int,
) -> tuple[int, float, float, float, float, int, int, int]:
    return (
        int(regressed_snr_bins),
        float(validation_ber),
        float(validation_phase_rmse),
        max(0.0, float(worst_per_snr_regression)),
        float(validation_phase_loss),
        int(candidate_index),
        int(seed_index),
        int(epoch_index),
    )


def _evaluate(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float, dict[str, float], float, float]:
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    weighted_loss = 0.0
    data_bits = 0
    bit_errors = 0
    weighted_phase_loss = 0.0
    squared_phase_error = 0.0
    phase_count = 0
    packet_decisions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (
                features,
                targets,
                data_mask,
                residual_phase_target,
                residual_phase_mask,
                _,
            ) = _unpack_batch(batch)
            features = features.to(device)
            targets = targets.to(device)
            data_mask = data_mask.to(device)
            residual_phase_target = residual_phase_target.to(device)
            residual_phase_mask = residual_phase_mask.to(device)
            residual_phase = model(features)
            bit_llr = bit_scores_from_residual(features, residual_phase)
            count = int(data_mask.sum().cpu()) * 2
            weighted_loss += float(
                masked_bit_bce(bit_llr, targets, data_mask).cpu()
            ) * count
            decisions = bit_llr < 0.0
            packet_decisions.append(decisions.detach().cpu().numpy())
            expanded = data_mask.unsqueeze(-1).expand_as(decisions)
            bit_errors += int(torch.sum((decisions != targets.bool()) & expanded).cpu())
            data_bits += count
            count_phase = int(residual_phase_mask.sum().cpu())
            weighted_phase_loss += float(
                masked_circular_phase_loss(
                    residual_phase,
                    residual_phase_target,
                    residual_phase_mask,
                ).cpu()
            ) * count_phase
            wrapped_error = torch.atan2(
                torch.sin(residual_phase - residual_phase_target),
                torch.cos(residual_phase - residual_phase_target),
            )
            squared_phase_error += float(
                torch.sum(
                    wrapped_error.square()
                    * residual_phase_mask.to(dtype=wrapped_error.dtype)
                ).cpu()
            )
            phase_count += count_phase
    decisions_array = np.concatenate(packet_decisions, axis=0)
    return (
        float(weighted_loss / max(1, data_bits)),
        float(bit_errors / max(1, data_bits)),
        _ber_by_snr(decisions_array, dataset),
        float(weighted_phase_loss / max(1, phase_count)),
        float(np.sqrt(squared_phase_error / max(1, phase_count))),
    )


def _pilot_smoothing_ber_by_snr(dataset) -> dict[str, float]:
    decisions = dataset.receiver_features[..., :2] < 0.0
    return _ber_by_snr(decisions, dataset)


def _unpack_batch(batch):
    if len(batch) == 5:
        features, targets, data_mask, phase_target, phase_mask = batch
        return features, targets, data_mask, phase_target, phase_mask, None
    if len(batch) == 6:
        return tuple(batch)
    raise ValueError("phase-tracking loader batches must contain five or six tensors")


def _curriculum_config(training: Mapping[str, object]) -> dict[str, object]:
    configured = training.get("high_snr_curriculum")
    if configured is not None and not isinstance(configured, Mapping):
        raise ValueError("training.high_snr_curriculum must be a mapping")
    return dict(configured or {})


def _curriculum_high_snr_weight(
    training: Mapping[str, object],
    *,
    epoch: int,
) -> float:
    config = _curriculum_config(training)
    if not bool(config.get("enabled", True)):
        return 1.0
    start = max(0.0, float(config.get("start_weight", 2.0)))
    end = max(0.0, float(config.get("end_weight", 1.0)))
    anneal_epochs = max(1, int(config.get("anneal_epochs", 16)))
    progress = min(1.0, max(0.0, float(epoch - 1) / anneal_epochs))
    return float(start + progress * (end - start))


def _curriculum_packet_weights(
    packet_snr_db,
    training: Mapping[str, object],
    *,
    epoch: int,
    device: torch.device,
) -> torch.Tensor | None:
    if packet_snr_db is None:
        return None
    config = _curriculum_config(training)
    if not bool(config.get("enabled", True)):
        return None
    threshold = float(config.get("threshold_db", 6.0))
    high_weight = _curriculum_high_snr_weight(training, epoch=epoch)
    snr = torch.as_tensor(packet_snr_db, dtype=torch.float32, device=device)
    return torch.where(
        torch.isfinite(snr) & (snr >= threshold),
        torch.full_like(snr, high_weight),
        torch.ones_like(snr),
    )


def _ber_by_snr(decisions: np.ndarray, dataset) -> dict[str, float]:
    if dataset.snr_db is None:
        return {"all": _masked_ber(decisions, dataset)}
    snr_values = np.asarray(dataset.snr_db, dtype=np.float64)
    if snr_values.shape != (dataset.target_bits.shape[0],):
        return {"all": _masked_ber(decisions, dataset)}
    return {
        "%g" % float(snr): _masked_ber(decisions[selected], dataset, selected)
        for snr in np.unique(snr_values)
        for selected in [snr_values == snr]
    }


def _masked_ber(
    decisions: np.ndarray,
    dataset,
    selected_packets: np.ndarray | None = None,
) -> float:
    targets = dataset.target_bits
    mask = dataset.data_mask
    if selected_packets is not None:
        targets = targets[selected_packets]
        mask = mask[selected_packets]
    expanded = np.repeat(mask[..., None], 2, axis=-1)
    return float(
        np.count_nonzero((np.asarray(decisions) != targets.astype(bool)) & expanded)
        / max(1, int(np.count_nonzero(expanded)))
    )


def _per_snr_regression(
    learned: dict[str, float],
    interpolation: dict[str, float],
    *,
    margin: float,
) -> dict[str, object]:
    shared = sorted(set(learned).intersection(interpolation))
    deltas = {
        key: float(learned[key] - interpolation[key])
        for key in shared
    }
    positive = [max(0.0, value) for value in deltas.values()]
    return {
        "margin": float(margin),
        "delta_by_snr_db": deltas,
        "regressed_bin_count": sum(value > float(margin) for value in deltas.values()),
        "worst_positive_delta": max(positive, default=0.0),
    }


def _write_manifest(
    config: dict,
    *,
    component_path: Path,
    component_sha: str,
    provenance: dict,
) -> Path:
    training = dict(config.get("training") or {})
    recipe = dict(config.get("recipe") or {})
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_contract, contract = _find_training_contract(config)
    contract_path = manifest_path.parent / "training_contract.yaml"
    if source_contract.resolve() != contract_path.resolve():
        shutil.copyfile(source_contract, contract_path)
    try:
        portable_component = str(
            component_path.resolve().relative_to(manifest_path.parent.resolve())
        )
    except ValueError as exc:
        raise ValueError(
            "Phase-tracking ONNX component must be inside the returned artifact package"
        ) from exc
    artifact_label = str(
        training.get("artifact_label") or "Learned phase-tracking receiver"
    )
    operation = str(
        training.get("artifact_operation")
        or recipe.get("receiver_operation")
        or "demodulation.phase_tracking_receiver_adapter"
    )
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": str(
            training.get("artifact_id") or "external.phase_tracking_receiver"
        ),
        "name": str(
            training.get("artifact_name") or "Learned phase-tracking receiver"
        ),
        "label": artifact_label,
        "description": (
            "Portable packet-context residual phase tracker around Noema's "
            "deterministic pilot smoother."
        ),
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
                "id": "receiver",
                "role": "phase_tracking_receiver",
                "path": portable_component,
                "sha256": component_sha,
                "format": "onnx",
            }
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 2,
            "entrypoints": [
                {
                    "id": "phase_tracking_receiver",
                    "component": "receiver",
                    "inputs": [
                        {
                            "name": "receiver_features_v3",
                            "dtype": "float32",
                            "shape": ["packet", "frame_symbol", 11],
                            "layout": "packet_symbol_receiver_feature",
                            "semantic": (
                                "pilot_smoother_corrected_iq_raw_iq_pilot_"
                                "innovation_smoother_phasor_and_fourth_power_cue_v3"
                            ),
                        }
                    ],
                    "outputs": [
                        {
                            "name": "residual_phase_rad",
                            "dtype": "float32",
                            "shape": ["packet", "frame_symbol"],
                            "layout": "packet_symbol",
                            "semantic": "learned_residual_relative_to_pilot_smoothing_rad",
                        }
                    ],
                }
            ],
        },
        "application": {"mode": "single_binding"},
        "compatible_operations": [
            {
                "operation": operation,
                "preferred_step_id": str(recipe.get("receiver_step") or ""),
                "label": artifact_label,
                "description": (
                    "Temporal residual phase tracker using received symbols and public pilots."
                ),
                "runtime_entrypoint": "phase_tracking_receiver",
                "required_inputs": ["rx_symbols", "pilot_context"],
                "params": {
                    "mode": "learned_artifact",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "phase_tracking_receiver",
                },
            }
        ],
        "source": {
            "project_manifest": "project_manifest.yaml",
            "training_template": str(config.get("training_template") or ""),
            "recipe_name": str(recipe.get("name") or ""),
            "recipe_sha256": str(recipe.get("sha256") or ""),
            "step_id": str(recipe.get("receiver_step") or ""),
        },
        "training": {
            "framework": str(config.get("framework") or "torch"),
            "architecture": str(
                dict(provenance.get("selected_candidate") or {}).get(
                    "architecture"
                )
                or ""
            ),
            "loss": str((config.get("objective") or {}).get("loss") or ""),
            "supervised_labels_used": True,
            **provenance,
        },
        "evaluation": {
            "command": "cd reference_training && python evaluate.py",
            "metrics_path": "reference_training/evaluation_metrics.json",
            "primary_metric": "bit_error_rate",
            "split": "test",
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
    reference = dict(config.get("training_contract") or {})
    candidates = []
    if reference.get("path"):
        candidates.append(Path(str(reference["path"])))
    candidates.extend((Path("training_contract.yaml"), Path("../training_contract.yaml")))
    for path in candidates:
        if not path.is_file():
            continue
        payload = load_strict_yaml_or_json(path)
        if not isinstance(payload, dict):
            raise ValueError("training_contract.yaml must contain a mapping")
        if str(payload.get("kind") or "") != "noema.trainable_slot_contract@1":
            raise ValueError("training_contract.yaml is not a Noema trainable-slot contract")
        if not str(payload.get("id") or ""):
            raise ValueError("training_contract.yaml requires an id")
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
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
