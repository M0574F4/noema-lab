from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.core.training_plans import scenario_recipe_fingerprint


JsonDict = Dict[str, Any]


class PortableAiPhyAdapterExportError(ValueError):
    pass


@dataclass(frozen=True)
class PortableAiPhyAdapterSpec:
    exporter_id: str
    name: str
    operation_id: str
    task_directory: str
    template_id: str
    loss: str
    target_kind: Optional[str]
    artifact_filename: str
    artifact_entrypoint: str
    artifact_role: str
    comparison_methods: tuple[str, ...]
    benchmark_id: str


@dataclass(frozen=True)
class PortableAiPhyAdapterPlan:
    recipe: Recipe
    recipe_sha256: str
    selected_step: RecipeStep
    feature_reference: str
    target_reference: str
    spec: PortableAiPhyAdapterSpec
    project_root: Optional[Path] = None


SPECS: Mapping[str, PortableAiPhyAdapterSpec] = {
    "range-localization": PortableAiPhyAdapterSpec(
        exporter_id="range-localization",
        name="Two-dimensional range-localization demonstration project builder",
        operation_id="model.localization_adapter",
        task_directory="localization_supervised_mlp",
        template_id="localization.supervised_geometry_mlp",
        loss="position.mse",
        target_kind="ai_phy.localization_truth.numpy",
        artifact_filename="localization_estimator.onnx",
        artifact_entrypoint="localization_estimator",
        artifact_role="two_dimensional_range_localizer",
        comparison_methods=("trilateration", "regularized_trilateration", "learned_localizer"),
        benchmark_id="localization_sensing.learned_range_localizer_v1",
    ),
    "aoa-estimation": PortableAiPhyAdapterSpec(
        exporter_id="aoa-estimation",
        name="Narrowband ULA AoA-estimation demonstration project builder",
        operation_id="model.aoa_estimator_adapter",
        task_directory="aoa_estimation_covariance_mlp",
        template_id="aoa_estimation.covariance_mlp",
        loss="angle.mse",
        target_kind="ai_phy.aoa_truth.numpy",
        artifact_filename="aoa_estimator.onnx",
        artifact_entrypoint="aoa_estimator",
        artifact_role="single_source_ula_aoa_estimator",
        comparison_methods=("bartlett", "music", "learned_estimator"),
        benchmark_id="localization_sensing.learned_aoa_estimator_v1",
    ),
    "beam-selection": PortableAiPhyAdapterSpec(
        exporter_id="beam-selection",
        name="Single-user MISO beam-selection demonstration project builder",
        operation_id="model.beamforming_adapter",
        task_directory="beam_selection_supervised_mlp",
        template_id="beam_selection.codebook_classifier_mlp",
        loss="beam.codebook_cross_entropy",
        target_kind=None,
        artifact_filename="beam_policy.onnx",
        artifact_entrypoint="beam_policy",
        artifact_role="single_user_miso_beam_policy",
        comparison_methods=("mrt", "dft_codebook_sweep", "learned_beam_policy"),
        benchmark_id="beamforming_precoding.learned_beam_selection_v1",
    ),
}


def build_portable_ai_phy_adapter_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    exporter_id: str,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> PortableAiPhyAdapterPlan:
    spec = SPECS[str(exporter_id)]
    selected_ids = [str(item) for item in optimizable_steps]
    if len(selected_ids) != 1:
        raise PortableAiPhyAdapterExportError(
            "%s requires exactly one selected replacement step" % spec.name
        )
    steps = {step.id: step for step in recipe.steps}
    if selected_ids[0] not in steps:
        raise PortableAiPhyAdapterExportError(
            "Unknown selected replacement step `%s`" % selected_ids[0]
        )
    selected = steps[selected_ids[0]]
    if selected.op != spec.operation_id:
        raise PortableAiPhyAdapterExportError(
            "%s requires operation %s; step %s uses %s"
            % (spec.name, spec.operation_id, selected.id, selected.op)
        )
    normalized_framework = str(framework or "torch").strip().lower()
    if normalized_framework != "torch":
        raise PortableAiPhyAdapterExportError(
            "%s reference starter requires framework=torch" % spec.name
        )
    if str(loss or "").strip() != spec.loss:
        raise PortableAiPhyAdapterExportError(
            "%s requires objective %s; got %s"
            % (spec.name, spec.loss, loss or "missing")
        )
    operation = registry.get(selected.op).describe()
    if not bool((operation.get("training_capabilities") or {}).get("portable_replacement")):
        raise PortableAiPhyAdapterExportError(
            "%s does not declare a portable trained-artifact ABI" % selected.op
        )
    feature_reference = str(selected.inputs.get("problem") or "").strip()
    if not feature_reference:
        raise PortableAiPhyAdapterExportError(
            "%s requires a problem input" % selected.id
        )
    target_reference = _target_reference(recipe, registry, spec)
    return PortableAiPhyAdapterPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        selected_step=selected,
        feature_reference=feature_reference,
        target_reference=target_reference,
        spec=spec,
    )


def write_portable_ai_phy_adapter_starter(
    plan: PortableAiPhyAdapterPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    destination = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=PortableAiPhyAdapterExportError,
    )
    common = find_demo_training_dir(
        Path("demo_trainings") / "_portable_ai_phy_adapter_common",
        marker="datamodule.py",
    )
    task = find_demo_training_dir(
        Path("demo_trainings") / plan.spec.task_directory
    )
    if common is None or task is None:
        raise PortableAiPhyAdapterExportError(
            "installed demonstration training assets are incomplete"
        )
    common_files = (
        "datamodule.py",
        "train.py",
        "evaluate.py",
        "build_benchmark.py",
        "requirements.txt",
    )
    for filename in common_files:
        shutil.copy2(common / filename, destination / filename)
    for filename in ("task.py", "README.md", "template.yaml"):
        shutil.copy2(task / filename, destination / filename)

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    train_config = _training_config(plan, project_root=project_root)
    _write_yaml(destination / "train_config.yaml", train_config)
    _write_yaml(destination / "noema_recipe.yaml", plan.recipe.to_dict())
    project_manifest = _project_manifest(
        plan,
        out_dir=destination,
        project_root=project_root,
    )
    _write_yaml(destination / "project_manifest.yaml", project_manifest)
    files = [
        *common_files,
        "task.py",
        "README.md",
        "template.yaml",
        "train_config.yaml",
        "noema_recipe.yaml",
        "project_manifest.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": plan.spec.exporter_id,
        "framework": "torch",
        "loss": plan.spec.loss,
        "training_template": plan.spec.template_id,
        "out_dir": str(destination),
        "optimizable_steps": [plan.selected_step.id],
        "project_manifest": project_manifest,
        "capture_jobs": [],
        "trained_artifacts": list(project_manifest["trained_artifacts"]),
        "files": files,
    }


def _target_reference(
    recipe: Recipe,
    registry: OperationRegistry,
    spec: PortableAiPhyAdapterSpec,
) -> str:
    if spec.target_kind is None:
        return ""
    steps = {step.id: step for step in recipe.steps}
    matches = []
    for raw in list((recipe.dataset_capture or {}).get("taps") or []):
        if not isinstance(raw, Mapping):
            continue
        reference = str(raw.get("from") or "")
        if "." not in reference:
            continue
        step_id, output_name = reference.split(".", 1)
        step = steps.get(step_id)
        if step is None:
            continue
        kind = str(registry.get(step.op).output_kinds.get(output_name) or "")
        if kind == spec.target_kind and reference not in matches:
            matches.append(reference)
    if len(matches) != 1:
        raise PortableAiPhyAdapterExportError(
            "%s requires exactly one captured %s target; found %s"
            % (spec.name, spec.target_kind, matches or "none")
        )
    return matches[0]


def _training_config(
    plan: PortableAiPhyAdapterPlan,
    *,
    project_root: Path,
) -> JsonDict:
    return {
        "schema_version": 1,
        "training_template": plan.spec.template_id,
        "project_root": str(project_root),
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "replacement_step": plan.selected_step.id,
            "feature_reference": plan.feature_reference,
            "target_reference": plan.target_reference,
        },
        "framework": "torch",
        "objective": {"loss": plan.spec.loss, "direction": "minimize"},
        "data": {
            "feature_tap": _capture_tap_id(plan.recipe, plan.feature_reference),
            "target_tap": _capture_tap_id(plan.recipe, plan.target_reference),
            "train_capture_dirs": ["../data/train"],
            "validation_capture_dirs": ["../data/validation"],
            "test_capture_dirs": ["../data/test"],
            "test_split_exposed_to_training": False,
        },
        "training": {
            "epochs": 40,
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "early_stopping_patience": 8,
            "initialization_seeds": [23],
            "device": "cuda_if_available",
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/%s"
            % plan.spec.artifact_filename,
            "artifact_id": "%s.%s"
            % (plan.recipe.name, plan.spec.artifact_entrypoint),
            "artifact_name": "%s for %s"
            % (plan.spec.name, plan.recipe.name),
            "artifact_label": "Learned · %s" % plan.recipe.name,
            "artifact_operation": plan.spec.operation_id,
            "artifact_role": plan.spec.artifact_role,
            "artifact_entrypoint": plan.spec.artifact_entrypoint,
        },
        "evaluation": {"use_cuda": False},
        "post_training": {
            "benchmark_id": plan.spec.benchmark_id,
            "comparison_methods": list(plan.spec.comparison_methods),
        },
    }


def _project_manifest(
    plan: PortableAiPhyAdapterPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    artifact_manifest = _project_path(out_dir / "trained_artifact.yaml", project_root)
    component = _project_path(
        out_dir / "artifacts" / plan.spec.artifact_filename,
        project_root,
    )
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": plan.spec.exporter_id,
        "training_template": plan.spec.template_id,
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_jobs": [],
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(out_dir / "requirements.txt", project_root),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": artifact_manifest,
            "component_paths": [component],
            "history_path": _project_path(out_dir / "training_history.json", project_root),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(out_dir / "evaluation_metrics.json", project_root),
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(out_dir / "build_benchmark.py", project_root),
            "command": "python build_benchmark.py",
            "benchmark_pack_path": _project_path(out_dir / "benchmark_pack.yaml", project_root),
            "comparison": {
                "methods": list(plan.spec.comparison_methods),
                "paired_held_out_seeds": True,
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.spec.operation_id,
                "step_id": plan.selected_step.id,
                "manifest_path": artifact_manifest,
            }
        ],
    }


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _capture_tap_id(recipe: Recipe, reference: str) -> str:
    if not reference:
        return ""
    for raw in list((recipe.dataset_capture or {}).get("taps") or []):
        if isinstance(raw, Mapping) and str(raw.get("from") or "") == reference:
            value = str(raw.get("id") or "").strip()
            if value:
                return value
    raise PortableAiPhyAdapterExportError(
        "training capture does not declare a tap for %s" % reference
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )
