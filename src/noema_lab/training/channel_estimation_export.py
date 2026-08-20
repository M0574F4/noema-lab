from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.training_plans import scenario_recipe_fingerprint
from noema_lab.training.capture_plan import (
    TrainingCapturePlan,
    TrainingCapturePlanError,
    resolve_training_capture_plan,
)
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.standalone_input import write_standalone_structured_input
from noema_lab.training.starter_refresh import prepare_demo_starter_directory


JsonDict = Dict[str, Any]

CHANNEL_ESTIMATOR_OP = "model.channel_estimator_adapter"
CHANNEL_ESTIMATION_LOSS = "channel.normalized_mse"


class ChannelEstimationExportError(ValueError):
    pass


@dataclass(frozen=True)
class ChannelEstimationExportPlan:
    recipe: Recipe
    recipe_sha256: str
    estimator_step: RecipeStep
    observation_step: RecipeStep
    pilot_reference: str
    mask_reference: str
    feature_reference: str
    noise_reference: str
    target_reference: str
    feature_kind: str
    pilot_kind: str
    mask_kind: str
    noise_kind: str
    target_kind: str
    capture_plan: TrainingCapturePlan
    framework: str
    loss: str
    project_root: Optional[Path] = None


def build_channel_estimation_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> ChannelEstimationExportPlan:
    validate_recipe_against_registry(recipe, registry)
    if str(framework or "").strip().lower() != "torch":
        raise ChannelEstimationExportError(
            "The reference channel-estimation demonstration requires framework=torch"
        )
    if str(loss or "").strip().lower() != CHANNEL_ESTIMATION_LOSS:
        raise ChannelEstimationExportError(
            "The reference channel-estimation demonstration requires objective %s"
            % CHANNEL_ESTIMATION_LOSS
        )
    selected_ids = [str(value).strip() for value in optimizable_steps if str(value).strip()]
    estimators = [
        step
        for step in recipe.steps
        if step.op == CHANNEL_ESTIMATOR_OP
        and (not selected_ids or step.id in selected_ids)
    ]
    if len(estimators) != 1:
        raise ChannelEstimationExportError(
            "Select exactly one %s block" % CHANNEL_ESTIMATOR_OP
        )
    estimator = estimators[0]
    if selected_ids != [estimator.id]:
        if set(selected_ids) != {estimator.id}:
            raise ChannelEstimationExportError(
                "The channel-estimation starter supports only replacement block `%s`"
                % estimator.id
            )
    problem_reference = str(estimator.inputs.get("problem") or "")
    if "." not in problem_reference:
        raise ChannelEstimationExportError(
            "Estimator `%s` has no connected pilot-observation input" % estimator.id
        )
    observation_id = problem_reference.split(".", 1)[0]
    observation = next(
        (step for step in recipe.steps if step.id == observation_id),
        None,
    )
    if observation is None or observation.op != "wireless.pilot_observation":
        raise ChannelEstimationExportError(
            "The reference demonstration expects `%s` to consume wireless.pilot_observation"
            % estimator.id
        )
    pilot_reference = "%s.pilot_ls" % observation.id
    mask_reference = "%s.pilot_mask" % observation.id
    feature_reference = "%s.ls_estimate" % observation.id
    noise_reference = "%s.noise_variance" % observation.id
    target_reference = "%s.truth" % observation.id
    described = registry.get(observation.op).describe()
    output_kinds = dict(described.get("output_kinds") or {})
    references = {
        pilot_reference: str(output_kinds.get("pilot_ls") or ""),
        mask_reference: str(output_kinds.get("pilot_mask") or ""),
        feature_reference: str(output_kinds.get("ls_estimate") or ""),
        noise_reference: str(output_kinds.get("noise_variance") or ""),
        target_reference: str(output_kinds.get("truth") or ""),
    }
    if any(not value for value in references.values()):
        raise ChannelEstimationExportError(
            "wireless.pilot_observation does not expose sparse pilot LS, pilot "
            "mask, interpolated LS, noise variance, and truth capture tensors"
        )
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=[
                {
                    "id": "pilot_ls",
                    "from": pilot_reference,
                    "role": "model_input",
                },
                {
                    "id": "pilot_mask",
                    "from": mask_reference,
                    "role": "model_input_mask",
                },
                {
                    "id": "ls_estimate",
                    "from": feature_reference,
                    "role": "model_input",
                },
                {
                    "id": "noise_variance",
                    "from": noise_reference,
                    "role": "conditioning_input",
                },
                {
                    "id": "channel_truth",
                    "from": target_reference,
                    "role": "training_target",
                },
            ],
            sample_unit="MIMO-OFDM channel realizations",
            suggested_total_samples=18432,
        )
    except TrainingCapturePlanError as exc:
        raise ChannelEstimationExportError(str(exc)) from exc
    return ChannelEstimationExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        estimator_step=estimator,
        observation_step=observation,
        pilot_reference=pilot_reference,
        mask_reference=mask_reference,
        feature_reference=feature_reference,
        noise_reference=noise_reference,
        target_reference=target_reference,
        feature_kind=references[feature_reference],
        pilot_kind=references[pilot_reference],
        mask_kind=references[mask_reference],
        noise_kind=references[noise_reference],
        target_kind=references[target_reference],
        capture_plan=capture_plan,
        framework="torch",
        loss=CHANNEL_ESTIMATION_LOSS,
    )


def write_channel_estimation_starter(
    plan: ChannelEstimationExportPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    destination = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=ChannelEstimationExportError,
    )
    template_dir = find_demo_training_dir(
        Path("demo_trainings") / "mimo_ofdm_channel_estimation_cnn"
    )
    if template_dir is None:
        raise ChannelEstimationExportError(
            "The packaged MIMO-OFDM channel-estimation training scaffold is missing"
        )
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
        shutil.copy2(template_dir / filename, destination / filename)
    write_standalone_structured_input(destination)
    shutil.copy2(
        template_dir / "template.yaml",
        destination / "training_template.yaml",
    )

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    config = _training_config(plan, project_root=project_root)
    _write_yaml(destination / "train_config.yaml", config)
    _write_yaml(destination / "noema_recipe.yaml", plan.recipe.to_dict())
    manifest = _project_manifest(
        plan,
        out_dir=destination,
        project_root=project_root,
    )
    _write_yaml(destination / "project_manifest.yaml", manifest)
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
        "exporter": "mimo-ofdm-channel-estimation",
        "training_template": "mimo_ofdm.channel_estimation_sparse_pilot_dual_domain_v3",
        "framework": plan.framework,
        "loss": plan.loss,
        "out_dir": str(destination),
        "optimizable_steps": [plan.estimator_step.id],
        "features": [
            {"kind": plan.pilot_kind, "from": plan.pilot_reference},
            {"kind": plan.mask_kind, "from": plan.mask_reference},
            {"kind": plan.feature_kind, "from": plan.feature_reference},
            {"kind": plan.noise_kind, "from": plan.noise_reference},
        ],
        "target": {"kind": plan.target_kind, "from": plan.target_reference},
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _training_config(
    plan: ChannelEstimationExportPlan,
    *,
    project_root: Path,
) -> JsonDict:
    channel_step = _producer(plan.recipe, plan.observation_step, "channel")
    pilot_step = _producer(plan.recipe, plan.observation_step, "pilots")
    return {
        "schema_version": 1,
        "training_template": "mimo_ofdm.channel_estimation_sparse_pilot_dual_domain_v3",
        "project_root": str(project_root),
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "estimator_step": plan.estimator_step.id,
            "estimator_operation": plan.estimator_step.op,
            "pilot_reference": plan.pilot_reference,
            "mask_reference": plan.mask_reference,
            "feature_reference": plan.feature_reference,
            "noise_reference": plan.noise_reference,
            "target_reference": plan.target_reference,
        },
        "data": {
            "pilot_tap": _tap_id(plan, plan.pilot_reference),
            "mask_tap": _tap_id(plan, plan.mask_reference),
            "ls_tap": _tap_id(plan, plan.feature_reference),
            "noise_tap": _tap_id(plan, plan.noise_reference),
            "target_tap": _tap_id(plan, plan.target_reference),
            "train_capture_dirs": ["../data/train"],
            "validation_capture_dirs": ["../data/validation"],
            "test_capture_dirs": ["../data/test"],
        },
        "scenario": {
            "rx_antennas": int(channel_step.params.get("rx_antennas") or 2),
            "tx_antennas": int(channel_step.params.get("tx_antennas") or 2),
            "subcarriers": int(channel_step.params.get("subcarriers") or 64),
            "tdl_models": ["A", "C", "E"],
            "channel_statistics": "mixed_unknown_3gpp_tdl_profile",
            "pilot_spacing": int(pilot_step.params.get("pilot_spacing") or 4),
            "snr_db": [-5, 0, 5, 10, 15, 20],
        },
        "model": {
            "class": "sparse_pilot_dual_domain_residual_channel_estimator",
            "candidates": [
                {
                    "id": "sparse_pilot_dual_domain_48",
                    "hidden_channels": 48,
                    "depth": 5,
                },
                {
                    "id": "sparse_pilot_dual_domain_64",
                    "hidden_channels": 64,
                    "depth": 6,
                },
            ],
        },
        "objective": {
            "loss": "normalized_complex_mse",
            "checkpoint_selection": [
                "aggregate_win_against_strongest_classical_baseline",
                "no_material_snr_bin_regression_against_strongest_classical",
                "minimum_validation_nmse_db",
                "configured_candidate_order",
                "configured_seed_order",
            ],
        },
        "training": {
            "epochs": 40,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "initialization_seeds": [23, 41],
            "patience": 10,
            "snr_regression_tolerance_db": 0.15,
            "device": "cuda_if_available",
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/channel_estimator.onnx",
            "artifact_entrypoint": "channel_estimator",
            "artifact_operation": plan.estimator_step.op,
            "artifact_id": "%s.%s.channel_estimator"
            % (_safe_stem(plan.recipe.name), _safe_stem(plan.estimator_step.id)),
            "artifact_name": "Learned MIMO-OFDM channel estimator",
            "artifact_label": "Learned estimator · 2×2 mixed TDL",
        },
        "evaluation": {
            "primary_metric": "nmse_db",
            "secondary_metrics": [
                "zf_spectral_efficiency_bps_hz",
                "zf_rate_retention",
            ],
            "test_split_exposed_to_training": False,
        },
    }


def _project_manifest(
    plan: ChannelEstimationExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    bundle = out_dir.parent
    jobs = []
    for split in plan.capture_plan.splits:
        recipe_path = bundle / ("capture_%s_recipe.yaml" % split.id)
        jobs.append(
            {
                "split": split.id,
                "label": "Held-out test" if split.id == "test" else split.id.title(),
                "recipe_path": _project_path(recipe_path, project_root),
                "bundle_recipe_path": recipe_path.name,
                "output_dir": _project_path(bundle / "data" / split.id, project_root),
                "requested_samples": split.samples,
                "expected_taps": [
                    {"id": tap.id, "from": tap.reference}
                    for tap in plan.capture_plan.taps
                ],
                "sample_unit": plan.capture_plan.sample_unit,
                "seed_offset": split.seed_offset,
                "seed_role": split.training_use,
                "owner": "noema",
                "consumer": "external_researcher",
                "source_recipe_sha256": plan.recipe_sha256,
            }
        )
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "mimo-ofdm-channel-estimation",
        "training_template": "mimo_ofdm.channel_estimation_sparse_pilot_dual_domain_v3",
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": jobs,
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(out_dir / "requirements.txt", project_root),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "artifact_manifest_path": _project_path(
                bundle / "trained_artifact.yaml", project_root
            ),
            "component_paths": [
                _project_path(
                    bundle / "artifacts" / "channel_estimator.onnx",
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
            "benchmark_pack_path": _project_path(
                out_dir / "benchmark_pack.yaml", project_root
            ),
            "comparison": {
                "methods": [
                    "least_squares",
                    "fixed_prior_lmmse",
                    "learned_estimator",
                ],
                "diagnostic": "perfect_csi_zf_rate",
                "paired_held_out_seeds": True,
                "sweep": "channel.snr_db",
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.estimator_step.op,
                "step_id": plan.estimator_step.id,
                "manifest_path": _project_path(
                    bundle / "trained_artifact.yaml", project_root
                ),
            }
        ],
    }


def _producer(recipe: Recipe, consumer: RecipeStep, input_name: str) -> RecipeStep:
    reference = str(consumer.inputs.get(input_name) or "")
    producer_id = reference.split(".", 1)[0]
    producer = next((step for step in recipe.steps if step.id == producer_id), None)
    if producer is None:
        raise ChannelEstimationExportError(
            "Input %s.%s has no producer" % (consumer.id, input_name)
        )
    return producer


def _tap_id(plan: ChannelEstimationExportPlan, reference: str) -> str:
    for tap in plan.capture_plan.taps:
        if tap.reference == reference:
            return tap.id
    raise ChannelEstimationExportError("Capture plan omits %s" % reference)


def _project_path(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        return str(resolved)


def _safe_stem(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    ).strip("_") or "channel_estimator"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
