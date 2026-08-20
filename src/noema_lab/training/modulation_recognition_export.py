from __future__ import annotations

"""Training-contract data support and AMC demonstration-project generation."""

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.reproducibility import canonical_json_sha256, master_seed_from_recipe
from noema_lab.core.structured_input import load_strict_yaml_or_json
from noema_lab.core.training_plans import scenario_recipe_fingerprint
from noema_lab.training.capture_plan import (
    TrainingCapturePlan,
    TrainingCapturePlanError,
    resolve_training_capture_plan,
)
from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.standalone_input import write_standalone_structured_input


JsonDict = Dict[str, Any]
MODULATION_CLASSIFIER_OP = "model.modulation_classifier_adapter"
MODULATION_CLASSIFICATION_LOSS = "classification.cross_entropy"
MODULATION_CAPTURE_TOTAL = 10368


class ModulationRecognitionExportError(ValueError):
    pass


@dataclass(frozen=True)
class ModulationRecognitionExportPlan:
    recipe: Recipe
    recipe_sha256: str
    classifier_step: RecipeStep
    feature_step: RecipeStep
    target_step: RecipeStep
    feature_reference: str
    target_reference: str
    feature_kind: str
    target_kind: str
    framework: str
    loss: str
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None


def build_modulation_recognition_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> ModulationRecognitionExportPlan:
    validate_recipe_against_registry(recipe, registry)
    normalized_framework = str(framework or "torch").strip().lower().replace("_", "-")
    if normalized_framework != "torch":
        raise ModulationRecognitionExportError(
            "modulation-recognition export currently requires framework=torch"
        )
    normalized_loss = str(loss or MODULATION_CLASSIFICATION_LOSS).strip().lower()
    if normalized_loss not in {
        "classification.cross_entropy",
        "modulation.cross_entropy",
        "cross_entropy",
    }:
        raise ModulationRecognitionExportError(
            "modulation-recognition exporter supports classification.cross_entropy; got %s" % loss
        )
    selected_ids = [str(item) for item in optimizable_steps]
    candidates = [step for step in recipe.steps if step.op == MODULATION_CLASSIFIER_OP]
    if selected_ids:
        candidates = [step for step in candidates if step.id in selected_ids]
        unknown = sorted(set(selected_ids) - {step.id for step in candidates})
        if unknown:
            raise ModulationRecognitionExportError(
                "AMC training accepts only a modulation-classifier step; incompatible selection: %s"
                % ", ".join(unknown)
            )
    if len(candidates) != 1:
        raise ModulationRecognitionExportError(
            "modulation-recognition export requires exactly one %s step" % MODULATION_CLASSIFIER_OP
        )
    classifier = candidates[0]
    feature_reference = str(classifier.inputs.get("observation") or "")
    if not feature_reference:
        raise ModulationRecognitionExportError("AMC classifier is missing its observation input")
    feature_step = _producer(recipe, feature_reference)
    feature_kind = _output_kind(registry, feature_step, feature_reference)
    if feature_kind != "ai_phy.modulation_iq_frames.numpy":
        raise ModulationRecognitionExportError(
            "AMC classifier requires ai_phy.modulation_iq_frames.numpy, got %s" % feature_kind
        )
    evaluation = next(
        (
            step
            for step in recipe.steps
            if step.op == "metrics.modulation_classification"
            and str(step.inputs.get("prediction") or "").split(".", 1)[0] == classifier.id
        ),
        None,
    )
    if evaluation is None:
        raise ModulationRecognitionExportError(
            "AMC training target must be discovered from a downstream metrics.modulation_classification step"
        )
    target_reference = str(evaluation.inputs.get("truth") or "")
    target_step = _producer(recipe, target_reference)
    target_kind = _output_kind(registry, target_step, target_reference)
    if target_kind != "ai_phy.modulation_labels.numpy":
        raise ModulationRecognitionExportError(
            "AMC supervised target must be ai_phy.modulation_labels.numpy, got %s" % target_kind
        )
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=(
                {"id": "iq_frames", "from": feature_reference, "role": "classifier_input_features"},
                {"id": "modulation_labels", "from": target_reference, "role": "supervised_target"},
            ),
            sample_unit="modulation frames",
            suggested_total_samples=MODULATION_CAPTURE_TOTAL,
            suggested_percentages={
                "train": 66.6667,
                "validation": 16.6667,
                "test": 16.6666,
            },
        )
    except TrainingCapturePlanError as exc:
        raise ModulationRecognitionExportError(str(exc)) from exc
    return ModulationRecognitionExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        classifier_step=classifier,
        feature_step=feature_step,
        target_step=target_step,
        feature_reference=feature_reference,
        target_reference=target_reference,
        feature_kind=feature_kind,
        target_kind=target_kind,
        framework="torch",
        loss=MODULATION_CLASSIFICATION_LOSS,
        capture_plan=capture_plan,
    )


def write_modulation_recognition_data_contract(
    plan: ModulationRecognitionExportPlan,
    out_dir: Path,
    *,
    project_root: Path,
) -> JsonDict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_tap = _tap_id(plan, plan.feature_reference)
    target_tap = _tap_id(plan, plan.target_reference)
    capture_recipes = []
    for split_spec in plan.capture_plan.splits:
        payload = _capture_recipe(
            plan,
            split=split_spec.id,
            seed_offset=split_spec.seed_offset,
            samples=split_spec.samples,
        )
        path = out_dir / ("capture_%s_recipe.yaml" % split_spec.id)
        _write_yaml(path, payload)
        capture_recipes.append(
            {
                "split": split_spec.id,
                "path": path.name,
                "sha256": canonical_json_sha256(payload),
                "file_sha256": _file_sha256(path),
                "requested_samples": split_spec.samples,
                "seed_offset": split_spec.seed_offset,
                "taps": [{"id": tap.id, "from": tap.reference} for tap in plan.capture_plan.taps],
            }
        )
    ownership = {
        "capture_recipe_generation": "noema",
        "capture_execution": "noema",
        "capture_integrity_validation": "noema",
        "dataset_consumption": "external_researcher",
        "model_architecture": "external_researcher",
        "loss_definition": "external_researcher",
        "model_training": "external_researcher",
    }
    contract = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "captured_supervised_pairs",
        "source_recipe": {"name": plan.recipe.name, "sha256": plan.recipe_sha256},
        "ownership": ownership,
        "capture_plan": plan.capture_plan.to_dict(),
        "feature": {
            "tap_id": feature_tap,
            "reference": plan.feature_reference,
            "kind": plan.feature_kind,
            "stored_dtype": "float32",
            "stored_shape": ["frame", "symbol", 2],
            "layout": "frame_symbol_real_imag",
        },
        "target": {
            "tap_id": target_tap,
            "reference": plan.target_reference,
            "kind": plan.target_kind,
            "dtype": "int64",
            "stored_shape": ["frame"],
            "class_names": ["bpsk", "qpsk", "qam16"],
            "pairing": "one class ID per I/Q frame",
        },
        "runtime_prediction": {
            "name": "class_logits",
            "dtype": "float32",
            "shape": ["batch", 3],
            "class_names": ["bpsk", "qpsk", "qam16"],
        },
        "labels": {
            "required": True,
            "source": "separate modulation-label artifact discovered from the evaluation graph",
            "not_a_runtime_operation_input": True,
        },
        "conditioning": {
            "snr_db": {
                "source": "%s.params.snr_db_min/snr_db_max" % plan.feature_step.id,
                "distribution": list(_capture_snr_distribution(plan)),
                "sampling": "recipe_matrix_round_robin",
            }
        },
        "splits": [
            {
                "id": item.id,
                # The shared validator retains this historical field name for
                # all captured supervised records, including radio frames.
                "requested_packet_records": item.samples,
                "seed_offset": item.seed_offset,
                "training_use": "held_out_evaluation" if item.id == "test" else item.id,
            }
            for item in plan.capture_plan.splits
        ],
        "capture_recipes": capture_recipes,
    }
    contract_path = out_dir / "data_contract.yaml"
    _write_yaml(contract_path, contract)
    jobs = _capture_jobs(plan, out_dir=out_dir, project_root=Path(project_root).resolve())
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


def write_modulation_recognition_starter(
    plan: ModulationRecognitionExportPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    out_dir = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=ModulationRecognitionExportError,
    )
    template = _template_dir()
    static_files = (
        "model.py",
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
    manifest = _starter_manifest(plan, out_dir=out_dir, project_root=project_root)
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
        "exporter": "modulation-recognition",
        "training_template": "modulation_recognition.supervised_cnn1d",
        "framework": plan.framework,
        "loss": plan.loss,
        "out_dir": str(out_dir),
        "optimizable_steps": [plan.classifier_step.id],
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _capture_recipe(
    plan: ModulationRecognitionExportPlan,
    *,
    split: str,
    seed_offset: int,
    samples: int,
) -> JsonDict:
    payload = plan.recipe.to_dict()
    source_step = next(
        (
            step
            for step in payload.get("steps") or []
            if str(step.get("op") or "") == "source.modulation_frames"
        ),
        None,
    )
    frames_per_run = max(
        1,
        int(((source_step or {}).get("params") or {}).get("frame_count") or 1),
    )
    required_runs = max(1, int(math.ceil(float(samples) / float(frames_per_run))))
    metadata = dict(payload.get("metadata") or {})
    metadata["seed"] = int(master_seed_from_recipe(plan.recipe) or 23) + int(seed_offset)
    metadata["training_performed"] = False
    metadata["capture_purpose"] = "supervised_automatic_modulation_recognition"
    metadata["capture_split"] = split
    payload["metadata"] = metadata
    for step in payload.get("steps") or []:
        params = dict(step.get("params") or {})
        params.pop("seed", None)
        step["params"] = params
    capture = dict(payload.get("dataset_capture") or {})
    capture.update(
        {
            "split": split,
            "samples": int(samples),
            "shard_size": min(int(capture.get("shard_size") or 128), int(samples)),
            "max_runs": required_runs,
            "seed_mode": "increment_run_seed",
            "taps": [{"id": tap.id, "from": tap.reference} for tap in plan.capture_plan.taps],
        }
    )
    payload["dataset_capture"] = capture
    return payload


def _capture_jobs(
    plan: ModulationRecognitionExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> list[JsonDict]:
    stem = _safe_stem(plan.recipe.name)
    capture_root = project_root / ".noema" / "dataset_captures"
    jobs = []
    for item in plan.capture_plan.splits:
        recipe_path = out_dir / ("capture_%s_recipe.yaml" % item.id)
        row: JsonDict = {
            "split": item.id,
            "label": "Held-out test" if item.id == "test" else item.id.title(),
            "recipe_path": _project_path(recipe_path, project_root),
            "bundle_recipe_path": recipe_path.name,
            "output_dir": _project_path(capture_root / ("%s_%s" % (stem, item.id)), project_root),
            "requested_samples": item.samples,
            "expected_taps": [{"id": tap.id, "from": tap.reference} for tap in plan.capture_plan.taps],
            "sample_unit": plan.capture_plan.sample_unit,
            "seed_offset": item.seed_offset,
            "seed_role": "held_out_evaluation" if item.id == "test" else item.id,
            "owner": "noema",
            "consumer": "external_researcher",
            "source_recipe_sha256": plan.recipe_sha256,
        }
        if recipe_path.is_file():
            recipe_payload = load_strict_yaml_or_json(recipe_path)
            if not isinstance(recipe_payload, Mapping):
                raise ModulationRecognitionExportError(
                    "Capture recipe must contain a mapping: %s" % recipe_path
                )
            row["recipe_sha256"] = canonical_json_sha256(recipe_payload)
            row["recipe_file_sha256"] = _file_sha256(recipe_path)
        jobs.append(row)
    return jobs


def _training_config(plan: ModulationRecognitionExportPlan, *, project_root: Path) -> JsonDict:
    stem = _safe_stem(plan.recipe.name)
    capture_root = project_root / ".noema" / "dataset_captures"
    return {
        "schema_version": 1,
        "training_template": "modulation_recognition.supervised_cnn1d",
        "project_root": str(Path(project_root).resolve()),
        "framework": "torch",
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "classifier_step": plan.classifier_step.id,
            "classifier_operation": plan.classifier_step.op,
            "feature_reference": plan.feature_reference,
            "target_reference": plan.target_reference,
        },
        "data": {
            "feature_tap": _tap_id(plan, plan.feature_reference),
            "target_tap": _tap_id(plan, plan.target_reference),
            "train_capture_dirs": [str(capture_root / ("%s_train" % stem))],
            "validation_capture_dirs": [str(capture_root / ("%s_validation" % stem))],
            "test_capture_dirs": [str(capture_root / ("%s_test" % stem))],
            "class_names": ["bpsk", "qpsk", "qam16"],
            "input_layout": "batch_symbol_real_imag",
        },
        "model": {
            "class": "ReferenceModulationCNN1D",
            "architecture": "cumulant_prior_residual_temporal_cnn",
            "channels": [32, 64],
            "dropout": 0.25,
            "analytic_prior_weight": 0.35,
        },
        "objective": {
            "loss": "label_smoothed_cross_entropy_with_residual_regularization",
            "checkpoint_selection": "minimum_validation_cross_entropy_then_maximum_balanced_accuracy",
        },
        "training": {
            "epochs": 30,
            "batch_size": 64,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.03,
            "residual_l2_weight": 1e-3,
            "correct_prior_residual_weight": 1e-2,
            "early_stopping_patience": 8,
            "rotation_augmentation": True,
            "initialization_seeds": [23, 41],
            "num_workers": 0,
            "device": "cuda_if_available",
            # The demonstration trainer runs from reference_training/, while returned artifacts
            # live at the export-bundle root beside the neutral contracts.
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/modulation_classifier.onnx",
            "artifact_entrypoint": "modulation_classifier",
            "artifact_operation": plan.classifier_step.op,
            "artifact_id": "%s.%s.modulation_classifier" % (stem, _safe_stem(plan.classifier_step.id)),
            "artifact_name": "Learned modulation classifier for %s" % plan.recipe.name,
            "artifact_label": "Learned AMC · %s" % plan.recipe.name,
        },
        "evaluation": {
            "primary_metric": "balanced_accuracy",
            "secondary_metrics": ["accuracy", "macro_f1", "cross_entropy"],
            "test_split_exposed_to_training": False,
        },
    }


def _starter_manifest(
    plan: ModulationRecognitionExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    root_bundle = out_dir.parent
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "modulation-recognition",
        "training_template": "modulation_recognition.supervised_cnn1d",
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": _capture_jobs(plan, out_dir=root_bundle, project_root=project_root),
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(out_dir / "requirements.txt", project_root),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _project_path(root_bundle / "trained_artifact.yaml", project_root),
            "component_paths": [_project_path(root_bundle / "artifacts" / "modulation_classifier.onnx", project_root)],
            "history_path": _project_path(out_dir / "training_history.json", project_root),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(out_dir / "evaluation_metrics.json", project_root),
            "split": "test",
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(out_dir / "build_benchmark.py", project_root),
            "command": "python build_benchmark.py",
            "benchmark_pack_path": _project_path(out_dir / "benchmark_pack.yaml", project_root),
            "comparison": {
                "methods": [
                    "blind_cumulant",
                    "learned_classifier",
                    "oracle_likelihood",
                ],
                "paired_held_out_seeds": True,
                "sweep": "channel_snr_db",
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.classifier_step.op,
                "step_id": plan.classifier_step.id,
                "manifest_path": _project_path(root_bundle / "trained_artifact.yaml", project_root),
            }
        ],
    }


def _producer(recipe: Recipe, reference: str) -> RecipeStep:
    if "." not in reference:
        raise ModulationRecognitionExportError("Invalid recipe reference: %s" % reference)
    step_id = reference.split(".", 1)[0]
    try:
        return next(step for step in recipe.steps if step.id == step_id)
    except StopIteration as exc:
        raise ModulationRecognitionExportError("Unknown producer in recipe reference: %s" % reference) from exc


def _output_kind(registry: OperationRegistry, step: RecipeStep, reference: str) -> str:
    output = reference.split(".", 1)[1]
    try:
        return str(registry.get(step.op).describe()["output_kinds"][output])
    except KeyError as exc:
        raise ModulationRecognitionExportError("Unknown producer output: %s" % reference) from exc


def _tap_id(plan: ModulationRecognitionExportPlan, reference: str) -> str:
    for tap in plan.capture_plan.taps:
        if tap.reference == reference:
            return str(tap.id)
    raise ModulationRecognitionExportError("Required captured signal is missing: %s" % reference)


def _capture_snr_distribution(plan: ModulationRecognitionExportPlan) -> tuple[float, ...]:
    matrix = dict((plan.recipe.metadata or {}).get("matrix") or {})
    dimensions = dict(matrix.get("dimensions") or {})
    raw = dimensions.get("channel.snr_db")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ModulationRecognitionExportError(
            "AMC capture requires metadata.matrix.dimensions.channel.snr_db"
        )
    try:
        return tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ModulationRecognitionExportError(
            "AMC capture SNR matrix values must be numeric"
        ) from exc


def _template_dir() -> Path:
    relative = Path("demo_trainings") / "modulation_recognition_supervised_cnn"
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise ModulationRecognitionExportError(
        "AMC demonstration project was not found; expected %s" % relative
    )


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
    safe = "".join(character if character.isalnum() or character in "_-" else "_" for character in str(value)).strip("_")
    return safe or "modulation_recognition"
