from __future__ import annotations

import math
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.reproducibility import canonical_json_sha256
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

RESOURCE_ALLOCATION_LOSS = "resource.negative_shannon_spectral_efficiency"
RESOURCE_ALLOCATION_TEMPLATE = "resource_allocation.unsupervised_shannon_deepsets"


class ResourceAllocationExportError(ValueError):
    pass


def _configured_capture_integer(
    value: Any,
    *,
    field: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceAllocationExportError(
            "%s must be an integer greater than or equal to %d"
            % (field, minimum)
        )
    if value < minimum:
        raise ResourceAllocationExportError(
            "%s must be greater than or equal to %d" % (field, minimum)
        )
    return value


def suggested_resource_capture_total(recipe: Recipe) -> int:
    """Choose a safe visible default that fits one generated CSI batch per split.

    Resource capture recipes intentionally use one fixed seeded realization for
    each split.  Suggesting more rows than one recipe run emits would either
    fail or tempt repeated CSI batches, so prefer the persisted recipe limit and
    otherwise the upstream synthetic-source batch size.
    """

    capture = dict(recipe.dataset_capture or {})
    split_plan = capture["split_plan"] if "split_plan" in capture else {}
    if not isinstance(split_plan, Mapping):
        raise ResourceAllocationExportError(
            "dataset_capture.split_plan must be a mapping"
        )
    configured_total = None
    if "total_samples" in split_plan:
        configured_total = _configured_capture_integer(
            split_plan["total_samples"],
            field="dataset_capture.split_plan.total_samples",
            minimum=3,
        )
    configured_samples = None
    if "samples" in capture:
        configured_samples = _configured_capture_integer(
            capture["samples"],
            field="dataset_capture.samples",
            minimum=3,
        )
    if configured_total is not None:
        return configured_total
    if configured_samples is not None:
        return configured_samples
    return max(3, min(192, resource_capture_single_run_capacity(recipe)))


def resource_capture_single_run_capacity(recipe: Recipe) -> int:
    """Conservative count of independent CSI rows emitted by one recipe run."""

    for step in recipe.steps:
        if step.op != "source.random_bits":
            continue
        if "batch_size" not in step.params:
            return 1
        return _configured_capture_integer(
            step.params["batch_size"],
            field="source.random_bits.batch_size",
            minimum=1,
        )
    return 1


@dataclass(frozen=True)
class ResourceAllocationExportPlan:
    recipe: Recipe
    recipe_sha256: str
    allocator_step: RecipeStep
    channel_state_step: RecipeStep
    framework: str
    loss: str
    feature_tap: str
    feature_reference: str
    subcarrier_count: int
    noise_variance: float
    average_power_budget: float
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None
    template_id: str = RESOURCE_ALLOCATION_TEMPLATE


def build_resource_allocation_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> ResourceAllocationExportPlan:
    validate_recipe_against_registry(recipe, registry)
    normalized_framework = str(framework or "torch").strip().lower().replace("_", "-")
    if normalized_framework != "torch":
        raise ResourceAllocationExportError(
            "resource-allocation demonstration project requires framework=torch"
        )
    normalized_loss = str(loss or "").strip().lower()
    accepted_losses = {
        RESOURCE_ALLOCATION_LOSS,
        "resource.negative.shannon.spectral.efficiency",
        "negative.shannon.spectral.efficiency",
        "negative_shannon_spectral_efficiency",
        "negative-shannon-spectral-efficiency",
    }
    if normalized_loss not in accepted_losses:
        raise ResourceAllocationExportError(
            "resource-allocation training requires loss=%s; got %s"
            % (RESOURCE_ALLOCATION_LOSS, loss)
        )
    step_ids = [str(value) for value in optimizable_steps]
    if len(step_ids) != 1:
        raise ResourceAllocationExportError(
            "resource-allocation training requires exactly one replacement allocator step"
        )
    allocator = _step(recipe, step_ids[0])
    if allocator.op != "model.symbol_power_allocator":
        raise ResourceAllocationExportError(
            "replacement step %s must use model.symbol_power_allocator" % allocator.id
        )
    policy = str(allocator.params.get("policy") or "")
    if policy == "learned_checkpoint":
        raise ResourceAllocationExportError(
            "Export training from a fixed or water_filling source recipe, not from an already frozen learned checkpoint"
        )
    if str(allocator.params.get("budget_mode") or "fixed_average") != "fixed_average":
        raise ResourceAllocationExportError("resource-allocation training requires budget_mode=fixed_average")
    if str(allocator.params.get("granularity") or "") != "per_subcarrier":
        raise ResourceAllocationExportError("resource-allocation training requires granularity=per_subcarrier")
    channel_reference = str(allocator.inputs.get("channel_state") or "")
    if not channel_reference or "." not in channel_reference:
        raise ResourceAllocationExportError("allocator must consume explicit channel_state CSI")
    channel_step_id, channel_output = channel_reference.split(".", 1)
    channel_state = _step(recipe, channel_step_id)
    if channel_state.op != "wireless.ofdm_channel_state" or channel_output != "state":
        raise ResourceAllocationExportError(
            "allocator channel_state must come from wireless.ofdm_channel_state.state"
        )
    subcarriers = int(channel_state.params.get("ofdm_fft_size") or allocator.params.get("subcarrier_count") or 0)
    if subcarriers < 1:
        raise ResourceAllocationExportError("Could not determine a positive OFDM subcarrier count")
    configured_subcarriers = int(allocator.params.get("subcarrier_count") or subcarriers)
    if configured_subcarriers != subcarriers:
        raise ResourceAllocationExportError(
            "allocator subcarrier_count=%d does not match channel-state FFT size %d"
            % (configured_subcarriers, subcarriers)
        )
    noise = float(channel_state.params.get("noise_variance") or 0.0)
    budget = float(allocator.params.get("target_power") or 0.0)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ResourceAllocationExportError("channel-state noise_variance must be finite and greater than zero")
    if not math.isfinite(budget) or budget <= 0.0:
        raise ResourceAllocationExportError("allocator target_power must be finite and greater than zero")
    feature_tap, feature_reference = _gain_tap(recipe, channel_state.id)
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=(
                {
                    "id": feature_tap,
                    "from": feature_reference,
                    "role": "allocator_input_csi",
                },
            ),
            sample_unit="CSI realizations",
            suggested_total_samples=suggested_resource_capture_total(recipe),
        )
    except TrainingCapturePlanError as exc:
        raise ResourceAllocationExportError(str(exc)) from exc
    single_run_capacity = resource_capture_single_run_capacity(recipe)
    oversized = [
        split for split in capture_plan.splits if split.samples > single_run_capacity
    ]
    if oversized:
        details = ", ".join(
            "%s=%d" % (split.id, split.samples) for split in oversized
        )
        raise ResourceAllocationExportError(
            "Resource capture uses one independently seeded CSI batch per split; "
            "requested split count(s) %s exceed the one-run capacity %d. "
            "Increase source.random_bits batch_size or reduce the capture total."
            % (details, single_run_capacity)
        )
    return ResourceAllocationExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        allocator_step=allocator,
        channel_state_step=channel_state,
        framework="torch",
        loss=RESOURCE_ALLOCATION_LOSS,
        feature_tap=feature_tap,
        feature_reference=feature_reference,
        subcarrier_count=subcarriers,
        noise_variance=noise,
        average_power_budget=budget,
        capture_plan=capture_plan,
    )


def write_resource_allocation_export(
    plan: ResourceAllocationExportPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    out_dir = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=ResourceAllocationExportError,
    )
    project_root = Path(plan.project_root or Path.cwd()).resolve()
    template_dir = _template_dir()
    static_files = [
        "model.py",
        "losses.py",
        "datamodule.py",
        "train.py",
        "evaluate.py",
        "build_benchmark.py",
        "README.md",
        "requirements.txt",
    ]
    for filename in static_files:
        shutil.copy2(template_dir / filename, out_dir / filename)
    write_standalone_structured_input(out_dir)
    shutil.copy2(template_dir / "template.yaml", out_dir / "training_template.yaml")
    _write_yaml(out_dir / "train_config.yaml", _training_config(plan, project_root=project_root))
    _write_yaml(out_dir / "noema_recipe.yaml", plan.recipe.to_dict())
    for split in plan.capture_plan.splits:
        _write_yaml(
            out_dir / ("capture_%s_recipe.yaml" % split.id),
            _capture_recipe(
                plan,
                split=split.id,
                seed_offset=split.seed_offset,
                samples=split.samples,
            ),
        )
    _write_yaml(out_dir / "power_allocation_contract.yaml", _contract(plan))
    project_manifest = _project_manifest(plan, out_dir=out_dir, project_root=project_root)
    _write_yaml(out_dir / "project_manifest.yaml", project_manifest)
    files = [
        *static_files,
        "structured_input.py",
        "training_template.yaml",
        "train_config.yaml",
        "noema_recipe.yaml",
        "capture_train_recipe.yaml",
        "capture_validation_recipe.yaml",
        "capture_test_recipe.yaml",
        "power_allocation_contract.yaml",
        "project_manifest.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": "resource-allocation",
        "training_template": plan.template_id,
        "framework": plan.framework,
        "loss": plan.loss,
        "objective": "maximize_parallel_channel_shannon_spectral_efficiency",
        "supervised_labels_used": False,
        "constraints": ["power_nonnegative", "exact_instantaneous_sum_power"],
        "out_dir": str(out_dir),
        "optimizable_steps": [plan.allocator_step.id],
        "features": {
            "tap_id": plan.feature_tap,
            "from": plan.feature_reference,
            "subcarrier_count": plan.subcarrier_count,
        },
        "project_manifest": project_manifest,
        "capture_jobs": list(project_manifest["capture_jobs"]),
        "trained_artifacts": list(project_manifest["trained_artifacts"]),
        "files": files,
    }


def write_resource_allocation_data_contract(
    plan: ResourceAllocationExportPlan,
    out_dir: Path,
    *,
    project_root: Path,
) -> JsonDict:
    """Write objective/model-neutral CSI capture assets for an allocator slot."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_recipes = []
    for split_spec in plan.capture_plan.splits:
        split = split_spec.id
        seed_offset = split_spec.seed_offset
        samples = split_spec.samples
        capture_recipe = _capture_recipe(
            plan,
            split=split,
            seed_offset=seed_offset,
            samples=samples,
        )
        capture_path = out_dir / ("capture_%s_recipe.yaml" % split)
        _write_yaml(
            capture_path,
            capture_recipe,
        )
        capture_recipes.append(
            {
                "split": split,
                "path": capture_path.name,
                "sha256": canonical_json_sha256(capture_recipe),
                "file_sha256": _file_sha256(capture_path),
                "requested_samples": samples,
                "seed_offset": seed_offset,
                "taps": [
                    {"id": tap.id, "from": tap.reference}
                    for tap in plan.capture_plan.taps
                ],
            }
        )
    jobs = list(
        _project_manifest(
            plan,
            out_dir=out_dir,
            project_root=Path(project_root).resolve(),
        )["capture_jobs"]
    )
    contract = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "captured_label_free",
        "source_recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
        },
        "ownership": {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_training": "external_researcher",
        },
        "capture_plan": plan.capture_plan.to_dict(),
        "feature": {
            "tap_id": plan.feature_tap,
            "reference": plan.feature_reference,
            "semantic": "per_subcarrier_channel_power_gain",
            "dtype": "float32",
            "shape": ["sample", plan.subcarrier_count],
        },
        "conditioning": {
            "noise_variance": "captured_or_recipe_controlled",
            "average_power_budget": "recipe_controlled",
        },
        "splits": [
            {
                "id": split,
                "requested_samples": samples,
                "seed_offset": seed_offset,
                "training_use": (
                    "held_out_evaluation" if split == "test" else split
                ),
            }
            for split, seed_offset, samples in (
                (item.id, item.seed_offset, item.samples)
                for item in plan.capture_plan.splits
            )
        ],
        "labels": {
            "required": False,
            "oracle_power_allocation_captured": False,
        },
        "capture_recipes": capture_recipes,
    }
    contract_path = out_dir / "data_contract.yaml"
    _write_yaml(contract_path, contract)
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


def _project_manifest(
    plan: ResourceAllocationExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    capture_root = Path(project_root) / ".noema" / "dataset_captures"
    stem = _safe_stem(plan.recipe.name)
    jobs = []
    for split_spec in plan.capture_plan.splits:
        split = split_spec.id
        samples = split_spec.samples
        seed_role = split_spec.training_use
        capture_path = Path(out_dir) / ("capture_%s_recipe.yaml" % split)
        row = {
            "split": split,
            "label": "Held-out test" if split == "test" else split.title(),
            "recipe_path": _project_path(capture_path, project_root),
            "bundle_recipe_path": capture_path.name,
            "output_dir": _project_path(
                capture_root / ("%s_%s" % (stem, split)), project_root
            ),
            "requested_samples": samples,
            "expected_taps": [
                {"id": tap.id, "from": tap.reference}
                for tap in plan.capture_plan.taps
            ],
            "sample_unit": plan.capture_plan.sample_unit,
            "seed_role": seed_role,
            "owner": "noema",
            "consumer": "external_researcher",
            "source_recipe_sha256": plan.recipe_sha256,
        }
        if capture_path.is_file():
            capture_recipe = load_strict_yaml_or_json(capture_path)
            if not isinstance(capture_recipe, Mapping):
                raise ResourceAllocationExportError(
                    "Capture recipe must contain a mapping: %s" % capture_path
                )
            row["recipe_sha256"] = canonical_json_sha256(capture_recipe)
            row["recipe_file_sha256"] = _file_sha256(capture_path)
        jobs.append(row)
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "resource-allocation",
        "training_template": plan.template_id,
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(Path(out_dir), project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": jobs,
        "training": {
            "owner": "external",
            "working_directory": _project_path(Path(out_dir), project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(Path(out_dir) / "requirements.txt", project_root),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _project_path(
                Path(out_dir) / "trained_artifact.yaml", project_root
            ),
            "component_paths": [
                _project_path(Path(out_dir) / "artifacts" / "power_policy.onnx", project_root)
            ],
            "history_path": _project_path(Path(out_dir) / "training_history.json", project_root),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(Path(out_dir) / "evaluation_metrics.json", project_root),
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(
                Path(out_dir) / "build_benchmark.py", project_root
            ),
            "command": "python build_benchmark.py",
            "benchmark_pack_path": _project_path(
                Path(out_dir) / "benchmark_pack.yaml", project_root
            ),
            "comparison": {
                "methods": [
                    "equal_power",
                    "learned_allocator",
                    "water_filling",
                ],
                "paired_held_out_seeds": True,
                "sweep": "average_transmit_power_budget",
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.allocator_step.op,
                "step_id": plan.allocator_step.id,
                "manifest_path": _project_path(Path(out_dir) / "trained_artifact.yaml", project_root),
            }
        ],
    }


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _training_config(plan: ResourceAllocationExportPlan, *, project_root: Path) -> JsonDict:
    stem = _safe_stem(plan.recipe.name)
    capture_root = Path(project_root) / ".noema" / "dataset_captures"
    return {
        "schema_version": 1,
        "training_template": plan.template_id,
        "project_root": str(Path(project_root)),
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "allocator_step": plan.allocator_step.id,
            "channel_state_step": plan.channel_state_step.id,
        },
        "framework": "torch",
        "objective": {
            "loss": plan.loss,
            "direction": "minimize",
            "uses_oracle_labels": False,
            "formula": "-mean(log2(1 + channel_gain * allocated_power / noise_variance))",
        },
        "constraints": {
            "nonnegative_power": "exact_euclidean_simplex_projection",
            "instantaneous_sum_power": "subcarrier_count * average_power_budget",
        },
        "data": {
            "feature_tap": plan.feature_tap,
            "train_capture_dirs": [str(capture_root / ("%s_train" % stem))],
            "validation_capture_dirs": [str(capture_root / ("%s_validation" % stem))],
            "test_capture_dirs": [str(capture_root / ("%s_test" % stem))],
            "subcarrier_count": plan.subcarrier_count,
            "oracle_allocation_tap": None,
        },
        "model": {
            "class": "DeepSetPowerAllocator",
            "architecture": "permutation_equivariant_deepsets_mean_pool",
            "hidden_dim": 64,
            "output_projection": "euclidean_simplex",
        },
        "training": {
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 20,
            "initialization_seeds": [23, 41, 67],
            "num_workers": 0,
            "device": "cuda_if_available",
            "average_power_budget_range": [
                max(plan.average_power_budget * 0.5, 1e-6),
                plan.average_power_budget * 2.0,
            ],
            "noise_variance_range": [plan.noise_variance, plan.noise_variance],
            "validation_average_power_budgets": [
                max(plan.average_power_budget * 0.5, 1e-6),
                plan.average_power_budget,
                plan.average_power_budget * 2.0,
            ],
            "validation_noise_variances": [plan.noise_variance],
            # The demonstration trainer runs from reference_training/, while returned artifacts
            # live at the export-bundle root beside the neutral contracts.
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/power_policy.onnx",
            "artifact_id": "%s.%s.csi_power_allocator" % (
                _safe_stem(plan.recipe.name),
                _safe_stem(plan.allocator_step.id),
            ),
            "artifact_name": "Learned CSI power allocator for %s" % plan.recipe.name,
            "artifact_label": "Learned · %s" % plan.recipe.name,
            "artifact_operation": plan.allocator_step.op,
        },
        "evaluation": {
            "average_power_budgets": [
                max(plan.average_power_budget * 0.5, 1e-6),
                plan.average_power_budget,
                plan.average_power_budget * 2.0,
            ],
            "noise_variances": [plan.noise_variance],
            "held_out_oracle": "water_filling",
            "use_cuda": False,
        },
    }


def _capture_recipe(
    plan: ResourceAllocationExportPlan,
    *,
    split: str,
    seed_offset: int,
    samples: int,
) -> JsonDict:
    payload = plan.recipe.to_dict()
    capture = dict(payload.get("dataset_capture") or {})
    capture["split"] = str(split)
    capture["samples"] = int(samples)
    configured_shard_size = int(capture.get("shard_size") or min(1024, int(samples)))
    capture["shard_size"] = min(configured_shard_size, int(samples))
    # Each split owns one explicit channel-state seed. Re-running it would
    # duplicate the same CSI batch rather than create independent examples.
    capture["max_runs"] = 1
    capture["seed_mode"] = "fixed_seed"
    capture["taps"] = [
        {"id": tap.id, "from": tap.reference}
        for tap in plan.capture_plan.taps
    ]
    payload["dataset_capture"] = capture
    for step in payload.get("steps") or []:
        if step.get("id") != plan.channel_state_step.id:
            continue
        params = step.setdefault("params", {})
        base_seed = int(params.get("seed") or (payload.get("metadata") or {}).get("seed") or 23)
        params["seed"] = base_seed + int(seed_offset)
    metadata = dict(payload.get("metadata") or {})
    metadata["training_performed"] = False
    metadata["capture_purpose"] = "label_free_csi_power_allocation"
    metadata["capture_split"] = str(split)
    metadata["channel_seed_role"] = "held_out_evaluation" if str(split) == "test" else "training_excluded_%s" % split
    metadata["oracle_allocation_captured"] = False
    payload["metadata"] = metadata
    return payload


def _contract(plan: ResourceAllocationExportPlan) -> JsonDict:
    return {
        "schema_version": 1,
        "kind": "noema.csi_power_allocation_training_contract",
        "training_template": plan.template_id,
        "input": {
            "tap_id": plan.feature_tap,
            "reference": plan.feature_reference,
            "value": "per_subcarrier_abs_h_squared",
            "shape": ["N", plan.subcarrier_count],
        },
        "runtime_conditioning": ["noise_variance", "average_power_budget"],
        "prediction": {
            "value": "per_subcarrier_power",
            "shape": ["N", plan.subcarrier_count],
        },
        "training": {
            "oracle_labels_used": False,
            "loss": plan.loss,
            "checkpoint_selection": "validation_shannon_spectral_efficiency",
        },
        "constraints": {
            "minimum_power": 0.0,
            "sum_power": "%d * average_power_budget" % plan.subcarrier_count,
        },
        "held_out_evaluation": {
            "oracle": "theoretical_water_filling",
            "common_channel_states_required": True,
        },
    }


def _gain_tap(recipe: Recipe, channel_state_id: str) -> tuple[str, str]:
    expected = "%s.state" % channel_state_id
    for item in list((recipe.dataset_capture or {}).get("taps") or []):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("from") or "") == expected:
            return str(item.get("id") or "channel_gains"), expected
    return "channel_gains", expected


def _step(recipe: Recipe, step_id: str) -> RecipeStep:
    for step in recipe.steps:
        if step.id == step_id:
            return step
    raise ResourceAllocationExportError("Recipe has no step %s" % step_id)


def _template_dir() -> Path:
    relative = Path("demo_trainings") / "resource_allocation_unsupervised_shannon"
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise ResourceAllocationExportError(
        "Resource-allocation demonstration project was not found; expected %s" % relative
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in str(value)).strip("_") or "resource_allocation"
