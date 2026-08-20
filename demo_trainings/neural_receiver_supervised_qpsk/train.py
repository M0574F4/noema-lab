from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from datamodule import build_loader, load_capture_dataset, split_integrity_report
from losses import bit_bce
from model import build_receiver, export_onnx_receiver, receiver_candidates
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


CHECKPOINT_SELECTION_ORDER = (
    "minimum_validation_bce",
    "minimum_validation_ber",
    "configured_candidate_order",
    "configured_seed_order",
    "earliest_epoch",
)


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    feature_tap = str(data.get("feature_tap") or "rx_symbols")
    target_tap = str(data.get("target_tap") or "target_bits")
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
    split_integrity = split_integrity_report(
        {
            "train": train_data,
            "validation": validation_data,
        }
    )

    candidate_specs = receiver_candidates(dict(config.get("model") or {}))
    epochs = max(1, int(training.get("epochs", 12)))
    batch_size = max(1, int(training.get("batch_size", 512)))
    workers = max(0, int(training.get("num_workers", 0)))
    seeds = [int(value) for value in training.get("initialization_seeds", [23, 41])]
    if not seeds:
        raise ValueError("training.initialization_seeds must not be empty")
    device = _device(training)
    best_rank = None
    best_ber = float("inf")
    best_loss = float("inf")
    best_state = None
    selected_seed = None
    selected_epoch = None
    selected_candidate = None
    selected_training = None
    history = []
    for candidate_index, candidate in enumerate(candidate_specs):
        for seed_index, seed in enumerate(seeds):
            _seed_everything(seed)
            model = build_receiver(candidate).to(device)
            if candidate.architecture == "affine":
                row = _fit_affine_calibrator(
                    model,
                    train_data,
                    validation_data,
                    candidate=candidate,
                    seed=seed,
                    batch_size=batch_size,
                    workers=workers,
                    device=device,
                    max_iterations=max(
                        1, int(training.get("affine_max_iterations", 100))
                    ),
                    history_size=max(
                        1, int(training.get("affine_history_size", 20))
                    ),
                    weight_decay=max(
                        0.0, float(training.get("affine_weight_decay", 0.0))
                    ),
                )
                history.append(row)
                print(
                    "candidate=%s seed=%d fit=%s iterations=%d train_bce=%.7g "
                    "validation_bce=%.7g validation_ber=%.7g"
                    % (
                        candidate.id,
                        seed,
                        row["fit"],
                        row["optimizer_iterations"],
                        row["train_bce"],
                        row["validation_bce"],
                        row["validation_ber"],
                    )
                )
                rank = _checkpoint_rank(
                    validation_bce=row["validation_bce"],
                    validation_ber=row["validation_ber"],
                    candidate_index=candidate_index,
                    seed_index=seed_index,
                    epoch_index=0,
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_ber = row["validation_ber"]
                    best_loss = row["validation_bce"]
                    selected_seed = seed
                    selected_epoch = 1
                    selected_candidate = candidate
                    selected_training = dict(row)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                continue

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
            for epoch in range(epochs):
                model.train()
                train_losses = []
                for features, targets in loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    llr = model(features)
                    loss = bit_bce(llr, targets)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu()))
                validation_loss, validation_ber = _evaluate(
                    model,
                    validation_data,
                    batch_size=batch_size,
                    workers=workers,
                    seed=seed,
                    device=device,
                )
                row = {
                    "candidate": candidate.to_dict(),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "fit": "mini_batch_adamw",
                    "train_bce": float(np.mean(train_losses)),
                    "validation_bce": validation_loss,
                    "validation_ber": validation_ber,
                }
                history.append(row)
                print(
                    "candidate=%s seed=%d epoch=%d train_bce=%.7g "
                    "validation_bce=%.7g validation_ber=%.7g"
                    % (
                        candidate.id,
                        seed,
                        epoch + 1,
                        row["train_bce"],
                        validation_loss,
                        validation_ber,
                    )
                )
                rank = _checkpoint_rank(
                    validation_bce=validation_loss,
                    validation_ber=validation_ber,
                    candidate_index=candidate_index,
                    seed_index=seed_index,
                    epoch_index=epoch,
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_ber = validation_ber
                    best_loss = validation_loss
                    selected_seed = seed
                    selected_epoch = epoch + 1
                    selected_candidate = candidate
                    selected_training = dict(row)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

    if (
        best_state is None
        or selected_seed is None
        or selected_epoch is None
        or selected_candidate is None
        or selected_training is None
    ):
        raise RuntimeError("Training produced no valid neural-receiver model")
    selected = build_receiver(selected_candidate)
    selected.load_state_dict(best_state)
    component_path = Path(
        str(training.get("artifact_component_path") or "artifacts/neural_receiver.onnx")
    )
    component_sha = export_onnx_receiver(selected, component_path)
    provenance = {
        "selected_candidate": selected_candidate.to_dict(),
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "best_validation_ber": best_ber,
        "best_validation_bce": best_loss,
        "selected_fit": str(selected_training.get("fit") or ""),
        "selected_fit_details": {
            key: selected_training[key]
            for key in (
                "initialization",
                "optimizer_iterations",
                "optimizer_function_evaluations",
            )
            if key in selected_training
        },
        "candidate_search": [candidate.to_dict() for candidate in candidate_specs],
        "selection_order": list(CHECKPOINT_SELECTION_ORDER),
        "initialization_seeds": seeds,
        "train_capture_schema_sha256": list(train_data.capture_schema_sha256),
        "validation_capture_schema_sha256": list(
            validation_data.capture_schema_sha256
        ),
        "split_integrity": split_integrity,
    }
    Path("training_history.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "noema.neural_receiver_training_history",
                "candidate_search": [
                    candidate.to_dict() for candidate in candidate_specs
                ],
                "split_integrity": split_integrity,
                "selection": {
                    "candidate": selected_candidate.to_dict(),
                    "seed": selected_seed,
                    "epoch": selected_epoch,
                    "validation_ber": best_ber,
                    "validation_bce": best_loss,
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
        "validation_bce=%.7g"
        % (
            selected_candidate.id,
            selected_seed,
            selected_epoch,
            best_ber,
            best_loss,
        )
    )
    print("exported neural receiver ONNX: %s" % component_path)
    print("component sha256: %s" % component_sha)
    print("registered trained block artifact: %s" % manifest_path)
    print(
        "return to Noema and select this artifact on a compatible "
        "demodulation.neural_receiver_adapter block"
    )
    return 0


def _checkpoint_rank(
    *,
    validation_bce: float,
    validation_ber: float,
    candidate_index: int,
    seed_index: int,
    epoch_index: int,
) -> tuple[float, float, int, int, int]:
    """Rank soft-output checkpoints without chasing one-bit BER fluctuations.

    BCE is the optimized proper scoring rule and measures both the decision and
    confidence carried by the returned bit logits.  Empirical BER is discrete
    and has broad plateaus on this small validation split, so it is a
    tie-breaker rather than the primary checkpoint selector.
    """

    return (
        float(validation_bce),
        float(validation_ber),
        int(candidate_index),
        int(seed_index),
        int(epoch_index),
    )


def _fit_affine_calibrator(
    model: torch.nn.Module,
    train_data,
    validation_data,
    *,
    candidate,
    seed: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    max_iterations: int,
    history_size: int,
    weight_decay: float,
) -> dict:
    """Fit the physically matched affine bit logits to numerical convergence.

    The fixed receiver front end is an invertible affine transform, so the
    Bayes-optimal QPSK bit boundaries remain two affine lines.  A full-batch
    convex logistic fit estimates those lines directly from captured I/Q and
    bit pairs without receiving the simulator's hidden transform.
    """

    _initialize_affine_from_labels(model, train_data)
    features = torch.from_numpy(train_data.features).to(device)
    targets = torch.from_numpy(train_data.target_bits).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max(1, int(max_iterations)),
        history_size=max(1, int(history_size)),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = bit_bce(model(features), targets)
        if weight_decay:
            loss = loss + 0.5 * float(weight_decay) * sum(
                torch.sum(parameter * parameter)
                for parameter in model.parameters()
            )
        loss.backward()
        return loss

    model.train()
    optimizer.step(closure)
    train_bce, _train_ber = _evaluate(
        model,
        train_data,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        device=device,
    )
    validation_bce, validation_ber = _evaluate(
        model,
        validation_data,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        device=device,
    )
    optimizer_state = optimizer.state.get(next(iter(model.parameters())), {})
    return {
        "candidate": candidate.to_dict(),
        "seed": int(seed),
        "epoch": 1,
        "fit": "supervised_affine_logistic_lbfgs",
        "initialization": "supervised_least_squares",
        "optimizer_iterations": int(optimizer_state.get("n_iter") or 0),
        "optimizer_function_evaluations": int(
            optimizer_state.get("func_evals") or 0
        ),
        "train_bce": float(train_bce),
        "validation_bce": float(validation_bce),
        "validation_ber": float(validation_ber),
    }


def _initialize_affine_from_labels(
    model: torch.nn.Module,
    dataset,
) -> None:
    """Initialize the two affine boundaries from captured supervision only."""

    layer = getattr(model, "network", None)
    if not isinstance(layer, torch.nn.Linear) or layer.in_features != 2 or layer.out_features != 2:
        raise ValueError("Affine calibration requires one torch.nn.Linear(2, 2) layer")
    features = np.asarray(dataset.features, dtype=np.float64)
    targets = np.asarray(dataset.target_bits, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("Affine calibration features must have shape [symbol, 2]")
    if targets.shape != features.shape:
        raise ValueError("Affine calibration targets must have shape [symbol, 2]")
    design = np.concatenate(
        (features, np.ones((features.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    signed_bit_zero = 1.0 - 2.0 * targets
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        signed_bit_zero,
        rcond=None,
    )
    if int(rank) < 3 or not bool(np.all(np.isfinite(coefficients))):
        raise ValueError(
            "Captured I/Q data cannot identify two affine decision boundaries"
        )
    with torch.no_grad():
        layer.weight.copy_(
            torch.as_tensor(
                coefficients[:2, :].T,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
        )
        layer.bias.copy_(
            torch.as_tensor(
                coefficients[2, :],
                dtype=layer.bias.dtype,
                device=layer.bias.device,
            )
        )


def _evaluate(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    losses = []
    errors = 0
    bit_count = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            llr = model(features)
            losses.append(float(bit_bce(llr, targets).cpu()))
            errors += int(torch.sum((llr < 0.0) != targets.bool()).cpu())
            bit_count += int(targets.numel())
    return float(np.mean(losses)), float(errors / max(1, bit_count))


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
        str(training.get("artifact_manifest_path") or "trained_artifact.yaml")
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
            "Neural-receiver ONNX component must be inside the returned artifact package"
        ) from exc
    contract_sha = _canonical_sha256(contract)
    contract_file_sha = _sha256(contract_path)
    artifact_label = str(training.get("artifact_label") or "Learned neural receiver")
    operation = str(
        training.get("artifact_operation")
        or recipe.get("receiver_operation")
        or "demodulation.neural_receiver_adapter"
    )
    payload = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": str(training.get("artifact_id") or "external.neural_receiver"),
        "name": str(training.get("artifact_name") or "Learned neural receiver"),
        "label": artifact_label,
        "description": (
            "Portable per-symbol QPSK neural receiver selected on captured "
            "validation data and returned through the operation-owned ABI."
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
                "id": "receiver",
                "role": "neural_receiver",
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
                    "id": "neural_receiver",
                    "component": "receiver",
                    "inputs": [
                        {
                            "name": "rx_symbols_ri",
                            "dtype": "float32",
                            "shape": ["symbol", 2],
                            "layout": "symbol_real_imag",
                            "semantic": "received_qpsk_symbols_real_imag",
                        }
                    ],
                    "outputs": [
                        {
                            "name": "bit_llr",
                            "dtype": "float32",
                            "shape": ["symbol", 2],
                            "layout": "symbol_bit",
                            "semantic": "bit_log_likelihood_ratio_positive_bit_zero",
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
                "description": "Frozen QPSK neural receiver over received complex symbols.",
                "runtime_entrypoint": "neural_receiver",
                "required_inputs": ["rx_symbols"],
                "params": {
                    "mode": "learned_artifact",
                    "modulation": "qpsk",
                    "artifact_manifest_path": "trained_artifact.yaml",
                    "artifact_entrypoint": "neural_receiver",
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
                dict(provenance.get("selected_candidate") or {}).get("architecture")
                or (config.get("model") or {}).get("architecture")
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
