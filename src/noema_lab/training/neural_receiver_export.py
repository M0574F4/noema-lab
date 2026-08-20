from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from noema_lab.core.matrix import (
    canonicalize_recipe_matrix,
    matrix_values_for_step_param,
)
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    master_seed_from_recipe,
)
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.standalone_input import write_standalone_structured_input


JsonDict = Dict[str, Any]

NEURAL_RECEIVER_CAPTURE_SNR_DB = (-2.0, 2.0, 6.0, 10.0)


class NeuralReceiverDataContractError(ValueError):
    pass


def write_neural_receiver_data_contract(
    plan: Any,
    out_dir: Path,
    *,
    project_root: Path,
) -> JsonDict:
    """Write graph-derived, auditable receiver feature/label capture assets."""

    if str(plan.feature_kind) != "channel.rx_symbols.complex_numpy":
        raise NeuralReceiverDataContractError(
            "The QPSK neural-receiver capture contract requires received complex symbols; got %s"
            % plan.feature_kind
        )
    if "bits" not in str(plan.target_kind):
        raise NeuralReceiverDataContractError(
            "The neural-receiver capture target must be a canonical bit tensor; got %s"
            % plan.target_kind
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve()
    feature_tap = _tap_id(plan, plan.feature_reference)
    target_tap = _tap_id(plan, plan.target_reference)
    capture_recipes = []
    for split_spec in plan.capture_plan.splits:
        split = split_spec.id
        seed_offset = split_spec.seed_offset
        samples = split_spec.samples
        payload = _capture_recipe(
            plan,
            split=split,
            seed_offset=seed_offset,
            samples=samples,
        )
        path = out_dir / ("capture_%s_recipe.yaml" % split)
        _write_yaml(path, payload)
        capture_recipes.append(
            {
                "split": split,
                "path": path.name,
                "sha256": canonical_json_sha256(payload),
                "file_sha256": _file_sha256(path),
                "requested_samples": samples,
                "seed_offset": seed_offset,
                "taps": [
                    {"id": tap.id, "from": tap.reference}
                    for tap in plan.capture_plan.taps
                ],
            }
        )

    contract = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "captured_supervised_pairs",
        "source_recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
        },
        "ownership": {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        },
        "capture_plan": plan.capture_plan.to_dict(),
        "feature": {
            "tap_id": feature_tap,
            "reference": plan.feature_reference,
            "kind": plan.feature_kind,
            "stored_dtype": "complex64",
            "stored_shape": ["packet", "qpsk_symbol"],
            "starter_view": {
                "dtype": "float32",
                "shape": ["symbol", 2],
                "layout": "real_imag",
            },
        },
        "target": {
            "tap_id": target_tap,
            "reference": plan.target_reference,
            "kind": plan.target_kind,
            "dtype": "uint8",
            "stored_shape": ["packet", "coded_bit"],
            "pairing": "two consecutive target bits per QPSK symbol",
        },
        "runtime_prediction": {
            "name": "bit_llr",
            "dtype": "float32",
            "shape": ["symbol", 2],
            "sign_convention": "positive predicts bit 0; negative predicts bit 1",
        },
        "labels": {
            "required": True,
            "source": "transmitted bit boundary discovered from the recipe metric graph",
            "not_a_runtime_operation_input": True,
        },
        "conditioning": {
            "snr_db": {
                "source": "%s.params.snr_db" % _snr_step_id(plan),
                "distribution": list(_capture_snr_distribution(plan)),
                "sampling": (
                    "metadata.matrix"
                    if canonicalize_recipe_matrix(plan.recipe).enabled
                    else "dataset_capture.sweep_round_robin"
                ),
            }
        },
        "splits": [
            {
                "id": split,
                "requested_packet_records": samples,
                "seed_offset": seed_offset,
                "training_use": "held_out_evaluation" if split == "test" else split,
            }
            for split, seed_offset, samples in (
                (item.id, item.seed_offset, item.samples)
                for item in plan.capture_plan.splits
            )
        ],
        "capture_recipes": capture_recipes,
    }
    contract_path = out_dir / "data_contract.yaml"
    _write_yaml(contract_path, contract)
    jobs = _capture_jobs(plan, out_dir=out_dir, project_root=root)
    return {
        "data_contract": contract,
        "data_contract_sha256": canonical_json_sha256(contract),
        "data_contract_file_sha256": _file_sha256(contract_path),
        "capture_recipes": capture_recipes,
        "capture_jobs": jobs,
        "files": [
            "data_contract.yaml",
            "capture_train_recipe.yaml",
            "capture_validation_recipe.yaml",
            "capture_test_recipe.yaml",
        ],
    }


def write_neural_receiver_starter(
    plan: Any,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    """Write one non-normative trainer that returns the operation's portable ABI."""

    out_dir = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=NeuralReceiverDataContractError,
    )
    template = _template_dir()
    static_files = (
        "model.py",
        "frontend.py",
        "losses.py",
        "datamodule.py",
        "train.py",
        "evaluate.py",
        "build_benchmark.py",
        "README.md",
        "requirements.txt",
    )
    for filename in static_files:
        shutil.copy2(template / filename, out_dir / filename)
    write_standalone_structured_input(out_dir)
    shutil.copy2(template / "template.yaml", out_dir / "training_template.yaml")

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    config = _training_config(plan, project_root=project_root)
    _write_yaml(out_dir / "train_config.yaml", config)
    _write_yaml(out_dir / "noema_recipe.yaml", plan.recipe.to_dict())
    manifest = _starter_project_manifest(
        plan,
        out_dir=out_dir,
        project_root=project_root,
    )
    _write_yaml(out_dir / "project_manifest.yaml", manifest)
    files = [
        *static_files,
        "structured_input.py",
        "training_template.yaml",
        "train_config.yaml",
        "noema_recipe.yaml",
        "project_manifest.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": "neural-receiver",
        "training_template": "neural_receiver.supervised_qpsk",
        "framework": plan.framework,
        "loss": plan.loss,
        "out_dir": str(out_dir),
        "optimizable_steps": [plan.receiver_step.id],
        "features": {
            "step_id": plan.feature_step.id,
            "kind": plan.feature_kind,
            "from": plan.feature_reference,
        },
        "target": {
            "step_id": plan.target_step.id,
            "kind": plan.target_kind,
            "from": plan.target_reference,
        },
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _capture_recipe(
    plan: Any,
    *,
    split: str,
    seed_offset: int,
    samples: int,
) -> JsonDict:
    payload = plan.recipe.to_dict()
    metadata = dict(payload.get("metadata") or {})
    base_seed = master_seed_from_recipe(plan.recipe)
    metadata["seed"] = int(base_seed or 23) + int(seed_offset)
    metadata["training_performed"] = False
    metadata["capture_purpose"] = "supervised_neural_receiver_qpsk"
    metadata["capture_split"] = split
    metadata["target_discovery"] = "metric_reference_graph"
    payload["metadata"] = metadata

    # Explicit per-step seeds would defeat dataset_capture.seed_mode.  Generated
    # capture recipes instead let each stochastic operation derive an independent
    # stream from the split master seed and run index.
    for step in payload.get("steps") or []:
        params = dict(step.get("params") or {})
        params.pop("seed", None)
        step["params"] = params

    capture = dict(payload.get("dataset_capture") or {})
    capture.update(
        {
            "split": split,
            "samples": int(samples),
            "shard_size": min(
                int(capture.get("shard_size") or 8), int(samples)
            ),
            "max_runs": int(samples),
            "seed_mode": "increment_run_seed",
            "taps": [
                {"id": tap.id, "from": tap.reference}
                for tap in plan.capture_plan.taps
            ],
        }
    )
    if canonicalize_recipe_matrix(plan.recipe).enabled:
        _remove_redundant_matrix_sweep(plan, capture)
    else:
        capture["sweep"] = {
            "%s.snr_db" % _snr_step_id(plan): list(
                _capture_snr_distribution(plan)
            )
        }
    payload["dataset_capture"] = capture
    return payload


def _capture_jobs(plan: Any, *, out_dir: Path, project_root: Path) -> list[JsonDict]:
    capture_root = project_root / ".noema" / "dataset_captures"
    stem = _safe_stem(plan.recipe.name)
    jobs = []
    for split_spec in plan.capture_plan.splits:
        split = split_spec.id
        seed_offset = split_spec.seed_offset
        samples = split_spec.samples
        recipe_path = out_dir / ("capture_%s_recipe.yaml" % split)
        row: JsonDict = {
            "split": split,
            "label": "Held-out test" if split == "test" else split.title(),
            "recipe_path": _project_path(recipe_path, project_root),
            "bundle_recipe_path": recipe_path.name,
            "output_dir": _project_path(
                capture_root / ("%s_%s" % (stem, split)), project_root
            ),
            "requested_samples": samples,
            "expected_taps": [
                {"id": tap.id, "from": tap.reference}
                for tap in plan.capture_plan.taps
            ],
            "sample_unit": plan.capture_plan.sample_unit,
            "seed_offset": seed_offset,
            "seed_role": "held_out_evaluation" if split == "test" else split,
            "owner": "noema",
            "consumer": "external_researcher",
            "source_recipe_sha256": plan.recipe_sha256,
            "sweep": {
                "%s.snr_db" % _snr_step_id(plan): list(
                    _capture_snr_distribution(plan)
                )
            },
        }
        if recipe_path.is_file():
            recipe_payload = load_strict_yaml_or_json(recipe_path)
            if not isinstance(recipe_payload, Mapping):
                raise NeuralReceiverDataContractError(
                    "Capture recipe must contain a mapping: %s" % recipe_path
                )
            row["recipe_sha256"] = canonical_json_sha256(recipe_payload)
            row["recipe_file_sha256"] = _file_sha256(recipe_path)
        jobs.append(row)
    return jobs


def _training_config(plan: Any, *, project_root: Path) -> JsonDict:
    capture_root = project_root / ".noema" / "dataset_captures"
    stem = _safe_stem(plan.recipe.name)
    fixed_affine_calibration = (
        str(plan.feature_step.op) == "hardware.receiver_iq_imbalance"
    )
    if fixed_affine_calibration:
        model = {
            "class": "affine_iq_calibration_receiver",
            "architecture": "affine",
            "candidates": [
                {
                    "id": "affine_iq_calibrator",
                    "architecture": "affine",
                },
            ],
        }
        initialization_seeds = [23]
    else:
        model = {
            "class": "validation_selected_per_symbol_receiver",
            "architecture": "candidate_search",
            "candidates": [
                {
                    "id": "affine_qpsk",
                    "architecture": "affine",
                },
                {
                    "id": "symbol_mlp_32x32",
                    "architecture": "symbol_mlp",
                    "hidden_dims": [32, 32],
                },
            ],
        }
        initialization_seeds = [23, 41]
    return {
        "schema_version": 1,
        "training_template": "neural_receiver.supervised_qpsk",
        "project_root": str(Path(project_root).resolve()),
        "framework": "torch",
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "receiver_step": plan.receiver_step.id,
            "receiver_operation": plan.receiver_step.op,
            "feature_reference": plan.feature_reference,
            "target_reference": plan.target_reference,
        },
        "data": {
            "feature_tap": _tap_id(plan, plan.feature_reference),
            "target_tap": _tap_id(plan, plan.target_reference),
            "train_capture_dirs": [str(capture_root / ("%s_train" % stem))],
            "validation_capture_dirs": [
                str(capture_root / ("%s_validation" % stem))
            ],
            "test_capture_dirs": [str(capture_root / ("%s_test" % stem))],
            "modulation": "qpsk",
            "bits_per_symbol": 2,
            "snr_db_distribution": list(_capture_snr_distribution(plan)),
        },
        "model": model,
        "objective": {
            "loss": "binary_cross_entropy_with_logits",
            "llr_sign": "positive_bit_zero",
            "checkpoint_selection": [
                "minimum_validation_bce",
                "minimum_validation_ber",
                "configured_candidate_order",
                "configured_seed_order",
                "earliest_epoch",
            ],
        },
        "training": {
            "epochs": 24,
            "batch_size": 512,
            "learning_rate": 2e-3,
            "weight_decay": 1e-5,
            "affine_max_iterations": 100,
            "affine_history_size": 20,
            "affine_weight_decay": 0.0,
            "initialization_seeds": initialization_seeds,
            "num_workers": 0,
            "device": "cuda_if_available",
            # The demonstration trainer runs from reference_training/, while returned artifacts
            # live at the export-bundle root beside the neutral contracts.
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/neural_receiver.onnx",
            "artifact_entrypoint": "neural_receiver",
            "artifact_operation": plan.receiver_step.op,
            "artifact_id": "%s.%s.neural_receiver"
            % (_safe_stem(plan.recipe.name), _safe_stem(plan.receiver_step.id)),
            "artifact_name": "Learned neural receiver for %s" % plan.recipe.name,
            "artifact_label": "Learned receiver · %s" % plan.recipe.name,
        },
        "evaluation": {
            "primary_metric": "bit_error_rate",
            "test_split_exposed_to_training": False,
        },
    }


def _starter_project_manifest(
    plan: Any,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    root_bundle = out_dir.parent
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "neural-receiver",
        "training_template": "neural_receiver.supervised_qpsk",
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": _capture_jobs(
            plan,
            out_dir=root_bundle,
            project_root=project_root,
        ),
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(
                out_dir / "requirements.txt", project_root
            ),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _project_path(
                root_bundle / "trained_artifact.yaml", project_root
            ),
            "component_paths": [
                _project_path(
                    root_bundle / "artifacts" / "neural_receiver.onnx",
                    project_root,
                )
            ],
            "history_path": _project_path(
                out_dir / "training_history.json", project_root
            ),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(
                out_dir / "evaluation_metrics.json", project_root
            ),
            "split": "test",
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(
                out_dir / "build_benchmark.py", project_root
            ),
            "command": "python build_benchmark.py",
            "benchmark_recipe_path": _project_path(
                out_dir / "benchmark_recipe.yaml", project_root
            ),
            "benchmark_pack_path": _project_path(
                out_dir / "benchmark_pack.yaml", project_root
            ),
            "comparison": {
                "methods": [
                    "uncompensated_qpsk",
                    "calibrated_iq_oracle",
                    "learned_receiver",
                ],
                "paired_held_out_seeds": True,
                "sweep": "channel_snr_db",
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.receiver_step.op,
                "step_id": plan.receiver_step.id,
                "manifest_path": _project_path(
                    root_bundle / "trained_artifact.yaml", project_root
                ),
            }
        ],
    }


def _template_dir() -> Path:
    relative = Path("demo_trainings") / "neural_receiver_supervised_qpsk"
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise NeuralReceiverDataContractError(
        "Neural-receiver demonstration project was not found; expected %s"
        % relative
    )


def _capture_snr_distribution(plan: Any) -> tuple[float, ...]:
    """Return the declared receiver-capture SNR support in authored order."""

    matrix = canonicalize_recipe_matrix(plan.recipe)
    if matrix.enabled:
        configured = matrix_values_for_step_param(
            plan.recipe,
            _snr_step_id(plan),
            "snr_db",
        )
        if configured is not None:
            return _numeric_snr_values(
                configured,
                source="metadata.matrix receiver SNR values",
            )

    capture = dict(plan.recipe.dataset_capture or {})
    sweep = capture.get("sweep") or {}
    if isinstance(sweep, Mapping):
        configured = sweep.get("%s.snr_db" % _snr_step_id(plan))
        if isinstance(configured, (list, tuple)) and configured:
            return _numeric_snr_values(
                configured,
                source="dataset_capture.sweep receiver SNR values",
            )
    return NEURAL_RECEIVER_CAPTURE_SNR_DB


def _remove_redundant_matrix_sweep(
    plan: Any,
    capture: Dict[str, Any],
) -> None:
    """Keep metadata.matrix as the only capture-variant definition."""

    raw_sweep = capture.get("sweep")
    if raw_sweep is None:
        return
    snr_key = "%s.snr_db" % _snr_step_id(plan)
    equivalent = (
        isinstance(raw_sweep, Mapping)
        and set(str(key) for key in raw_sweep) == {snr_key}
        and _numeric_snr_values(
            raw_sweep.get(snr_key),
            source="dataset_capture.sweep receiver SNR values",
        )
        == _capture_snr_distribution(plan)
    )
    if not equivalent:
        raise NeuralReceiverDataContractError(
            "The neural-receiver capture contract cannot combine metadata.matrix "
            "with a different dataset_capture.sweep; keep metadata.matrix as the "
            "single variant definition"
        )
    capture.pop("sweep", None)


def _numeric_snr_values(
    configured: Any,
    *,
    source: str,
) -> tuple[float, ...]:
    if not isinstance(configured, (list, tuple)) or not configured:
        raise NeuralReceiverDataContractError("%s must be a non-empty list" % source)
    try:
        values = tuple(float(value) for value in configured)
    except (TypeError, ValueError) as exc:
        raise NeuralReceiverDataContractError("%s must be numeric" % source) from exc
    if not all(math.isfinite(value) for value in values):
        raise NeuralReceiverDataContractError("%s must be finite" % source)
    return values


def _snr_step_id(plan: Any) -> str:
    """Resolve channel SNR ownership independently of the selected feature tap."""

    wireless_steps = [
        step
        for step in plan.recipe.steps
        if str(step.op) == "wireless.channel"
    ]
    if len(wireless_steps) > 1:
        raise NeuralReceiverDataContractError(
            "The neural-receiver capture contract cannot infer SNR ownership "
            "from %d wireless.channel steps"
            % len(wireless_steps)
        )
    if wireless_steps:
        return str(wireless_steps[0].id)
    # Generic/custom receiver graphs may expose SNR directly on the feature
    # producer instead of containing Noema's canonical wireless.channel block.
    return str(plan.feature_step.id)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _safe_stem(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in str(value)
    ).strip("_")
    return safe or "neural_receiver"


def _tap_id(plan: Any, reference: str) -> str:
    for tap in plan.capture_plan.taps:
        if tap.reference == reference:
            return str(tap.id)
    raise NeuralReceiverDataContractError(
        "Required captured signal is missing from the resolved capture plan: %s"
        % reference
    )
