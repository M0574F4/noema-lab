from __future__ import annotations

"""Optional finite-blocklength allocation demonstration scaffold.

The neutral Noema training contract remains model-, loss-, and trainer-neutral.
This module is used only after a researcher explicitly attaches the checked-in
delayed-CSI demonstration.  The returned allocator receives delayed transmitter
CSI plus the public noise and power-budget scalars.  Current channel state is an
aligned, training-only outcome used to evaluate the reliability objective; it is
never part of the portable artifact ABI.
"""

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

DELAYED_CSI_RESOURCE_ALLOCATION_LOSS = (
    "resource.negative_expected_finite_blocklength_goodput"
)
DELAYED_CSI_RESOURCE_ALLOCATION_TEMPLATE = (
    "resource_allocation.delayed_csi_finite_blocklength_causal_history_cnn"
)


class DelayedCsiResourceAllocationExportError(ValueError):
    pass


@dataclass(frozen=True)
class DelayedCsiResourceAllocationExportPlan:
    recipe: Recipe
    recipe_sha256: str
    allocator_step: RecipeStep
    channel_state_step: RecipeStep
    csi_observation_step: RecipeStep
    reliability_metrics_step: RecipeStep
    framework: str
    loss: str
    delayed_csi_tap: str
    delayed_csi_reference: str
    current_csi_tap: str
    current_csi_reference: str
    subcarrier_count: int
    csi_history_length: int
    noise_variance: float
    average_power_budget: float
    benchmark_power_budgets: tuple[float, ...]
    blocklength_channel_uses: int
    target_rate_bps_hz: float
    include_third_order_term: bool
    robust_gain_shrinkage: float
    csi_prediction_horizon_ofdm_symbols: int
    csi_prediction_gain_confidence: float
    capture_plan: TrainingCapturePlan
    project_root: Optional[Path] = None
    template_id: str = DELAYED_CSI_RESOURCE_ALLOCATION_TEMPLATE


def build_delayed_csi_resource_allocation_export_plan(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    optimizable_steps: Sequence[str],
    loss: str,
    framework: str,
) -> DelayedCsiResourceAllocationExportPlan:
    validate_recipe_against_registry(recipe, registry)

    normalized_framework = (
        str(framework or "torch").strip().lower().replace("_", "-")
    )
    if normalized_framework != "torch":
        raise DelayedCsiResourceAllocationExportError(
            "delayed-csi-resource-allocation demonstration requires framework=torch"
        )

    normalized_loss = str(loss or "").strip().lower().replace("-", "_")
    accepted_losses = {
        DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
        "resource.negative.expected.finite.blocklength.goodput",
        "negative_expected_finite_blocklength_goodput",
    }
    if normalized_loss not in accepted_losses:
        raise DelayedCsiResourceAllocationExportError(
            "delayed-csi-resource-allocation training requires loss=%s; got %s"
            % (DELAYED_CSI_RESOURCE_ALLOCATION_LOSS, loss)
        )

    selected_ids = [str(value) for value in optimizable_steps]
    if len(selected_ids) != 1:
        raise DelayedCsiResourceAllocationExportError(
            "delayed-csi-resource-allocation training requires exactly one "
            "replacement allocator step"
        )
    allocator = _step(recipe, selected_ids[0])
    if allocator.op != "model.causal_csi_power_allocator":
        raise DelayedCsiResourceAllocationExportError(
            "replacement step %s must use model.causal_csi_power_allocator"
            % allocator.id
        )
    policy = str(allocator.params.get("policy") or "")
    if policy in {"learned_checkpoint", "learned_artifact"}:
        raise DelayedCsiResourceAllocationExportError(
            "Export training from a non-learned source policy, not from an "
            "already frozen learned allocator"
        )
    if str(allocator.params.get("budget_mode") or "fixed_average") != "fixed_average":
        raise DelayedCsiResourceAllocationExportError(
            "delayed-CSI allocation training requires budget_mode=fixed_average"
        )
    if str(allocator.params.get("granularity") or "") != "per_subcarrier":
        raise DelayedCsiResourceAllocationExportError(
            "delayed-CSI allocation training requires granularity=per_subcarrier"
        )

    delayed_reference = str(allocator.inputs.get("channel_state") or "").strip()
    if not delayed_reference or "." not in delayed_reference:
        raise DelayedCsiResourceAllocationExportError(
            "allocator must consume explicit delayed transmitter CSI"
        )
    observation_id, observation_output = delayed_reference.split(".", 1)
    observation = _step(recipe, observation_id)
    if (
        observation.op != "wireless.ofdm_delayed_csi"
        or observation_output != "transmitter_csi"
    ):
        raise DelayedCsiResourceAllocationExportError(
            "allocator channel_state must come from "
            "wireless.ofdm_delayed_csi.transmitter_csi"
        )

    source_reference = str(observation.inputs.get("state") or "").strip()
    if not source_reference or "." not in source_reference:
        raise DelayedCsiResourceAllocationExportError(
            "wireless.ofdm_delayed_csi must consume an explicit OFDM channel state"
        )
    channel_state_id, channel_state_output = source_reference.split(".", 1)
    channel_state = _step(recipe, channel_state_id)
    if (
        channel_state.op != "wireless.ofdm_channel_state"
        or channel_state_output != "state"
    ):
        raise DelayedCsiResourceAllocationExportError(
            "delayed-CSI observation state must come from "
            "wireless.ofdm_channel_state.state"
        )

    current_reference = "%s.actual_state" % observation.id
    _require_output_kind(
        registry,
        observation,
        "transmitter_csi",
        "channel.ofdm_channel_state.numpy",
    )
    _require_output_kind(
        registry,
        observation,
        "actual_state",
        "channel.ofdm_channel_state.numpy",
    )

    reliability_steps = [
        step
        for step in recipe.steps
        if step.op == "metrics.ofdm_finite_blocklength_allocation"
    ]
    if len(reliability_steps) != 1:
        raise DelayedCsiResourceAllocationExportError(
            "delayed-csi-resource-allocation requires exactly one "
            "metrics.ofdm_finite_blocklength_allocation step"
        )
    reliability = reliability_steps[0]
    expected_metric_inputs = {
        "actual_state": current_reference,
        "transmitter_csi": delayed_reference,
        "allocation": "%s.allocation" % allocator.id,
    }
    for input_name, expected_reference in expected_metric_inputs.items():
        configured = str(reliability.inputs.get(input_name) or "")
        if configured != expected_reference:
            raise DelayedCsiResourceAllocationExportError(
                "%s.%s must consume %s; got %s"
                % (
                    reliability.id,
                    input_name,
                    expected_reference,
                    configured or "missing",
                )
            )

    current_state_consumers = [
        step
        for step in recipe.steps
        if step.op == "wireless.channel"
        and str(step.inputs.get("channel_state") or "") == current_reference
    ]
    if not current_state_consumers:
        raise DelayedCsiResourceAllocationExportError(
            "the propagation channel must consume csi_observation.actual_state "
            "while the allocator consumes only transmitter_csi"
        )

    subcarriers = _positive_int(
        channel_state.params.get("ofdm_fft_size")
        or allocator.params.get("subcarrier_count"),
        "OFDM subcarrier count",
    )
    configured_subcarriers = _positive_int(
        allocator.params.get("subcarrier_count") or subcarriers,
        "allocator subcarrier_count",
    )
    if configured_subcarriers != subcarriers:
        raise DelayedCsiResourceAllocationExportError(
            "allocator subcarrier_count=%d does not match channel-state FFT size %d"
            % (configured_subcarriers, subcarriers)
        )

    noise = _positive_float(
        channel_state.params.get("noise_variance"),
        "channel-state noise_variance",
    )
    budget = _positive_float(
        allocator.params.get("target_power"),
        "allocator target_power",
    )
    benchmark_power_budgets = _configured_power_budgets(recipe, budget)
    blocklength = _positive_int(
        reliability.params.get("blocklength_channel_uses"),
        "finite-blocklength blocklength_channel_uses",
    )
    target_rate = _positive_float(
        reliability.params.get("target_rate_bps_hz"),
        "finite-blocklength target_rate_bps_hz",
    )
    shrinkage = _closed_unit_float(
        allocator.params.get("csi_gain_shrinkage", 0.6),
        "allocator csi_gain_shrinkage",
    )

    allocation_reference = "%s.allocation" % allocator.id
    for tap in list((recipe.dataset_capture or {}).get("taps") or []):
        if (
            isinstance(tap, Mapping)
            and str(tap.get("from") or "") == allocation_reference
        ):
            raise DelayedCsiResourceAllocationExportError(
                "The delayed-CSI demonstration is label-free: remove %s from "
                "captured signals. Capture delayed CSI and aligned current CSI, "
                "not the source allocator output."
                % allocation_reference
            )

    delayed_tap = _tap_id(recipe, delayed_reference, "csi_history")
    current_tap = _tap_id(recipe, current_reference, "current_csi")
    history_length = _positive_int(
        observation.params.get("csi_history_length"),
        "CSI history length",
    )
    try:
        capture_plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=(
                {
                    "id": delayed_tap,
                    "from": delayed_reference,
                    "role": "runtime_input_causal_complex_csi_history",
                },
                {
                    "id": current_tap,
                    "from": current_reference,
                    "role": "training_only_auxiliary_current_channel_outcome",
                },
            ),
            sample_unit="aligned delayed/current CSI states",
            suggested_total_samples=3072,
        )
    except TrainingCapturePlanError as exc:
        raise DelayedCsiResourceAllocationExportError(str(exc)) from exc

    return DelayedCsiResourceAllocationExportPlan(
        recipe=recipe,
        recipe_sha256=scenario_recipe_fingerprint(recipe),
        allocator_step=allocator,
        channel_state_step=channel_state,
        csi_observation_step=observation,
        reliability_metrics_step=reliability,
        framework="torch",
        loss=DELAYED_CSI_RESOURCE_ALLOCATION_LOSS,
        delayed_csi_tap=delayed_tap,
        delayed_csi_reference=delayed_reference,
        current_csi_tap=current_tap,
        current_csi_reference=current_reference,
        subcarrier_count=subcarriers,
        csi_history_length=history_length,
        noise_variance=noise,
        average_power_budget=budget,
        benchmark_power_budgets=benchmark_power_budgets,
        blocklength_channel_uses=blocklength,
        target_rate_bps_hz=target_rate,
        include_third_order_term=bool(
            reliability.params.get("include_third_order_term", True)
        ),
        robust_gain_shrinkage=shrinkage,
        csi_prediction_horizon_ofdm_symbols=int(
            observation.params.get("feedback_delay_ofdm_symbols") or 0
        ),
        csi_prediction_gain_confidence=0.4,
        capture_plan=capture_plan,
    )


def write_delayed_csi_resource_allocation_starter(
    plan: DelayedCsiResourceAllocationExportPlan,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    destination = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=DelayedCsiResourceAllocationExportError,
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
        source = template / filename
        if not source.is_file():
            raise DelayedCsiResourceAllocationExportError(
                "Delayed-CSI demonstration project is incomplete; missing %s"
                % source
            )
        shutil.copy2(source, destination / filename)
    write_standalone_structured_input(destination)

    template_path = template / "template.yaml"
    if not template_path.is_file():
        raise DelayedCsiResourceAllocationExportError(
            "Delayed-CSI demonstration project is incomplete; missing %s"
            % template_path
        )
    shutil.copy2(template_path, destination / "training_template.yaml")

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    _write_yaml(
        destination / "train_config.yaml",
        _training_config(plan, project_root=project_root),
    )
    _write_yaml(destination / "noema_recipe.yaml", plan.recipe.to_dict())
    _write_yaml(
        destination / "reliability_allocation_contract.yaml",
        _reliability_contract(plan),
    )
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
        "reliability_allocation_contract.yaml",
        "project_manifest.yaml",
    ]
    return {
        "status": "exported",
        "recipe": plan.recipe.name,
        "recipe_sha256": plan.recipe_sha256,
        "exporter": "delayed-csi-resource-allocation",
        "training_template": plan.template_id,
        "framework": plan.framework,
        "loss": plan.loss,
        "objective": "maximize_expected_finite_blocklength_goodput",
        "supervised_allocation_labels_used": False,
        "current_csi_used_for_training_objective": True,
        "current_csi_is_runtime_input": False,
        "out_dir": str(destination),
        "optimizable_steps": [plan.allocator_step.id],
        "features": [
            {
                "tap_id": plan.delayed_csi_tap,
                "from": plan.delayed_csi_reference,
                "semantic": "causal_delayed_noisy_complex_csi_history",
                "runtime_input": True,
            }
        ],
        "training_only_outcomes": [
            {
                "tap_id": plan.current_csi_tap,
                "from": plan.current_csi_reference,
                "semantic": "aligned_current_channel_state_for_loss",
                "runtime_input": False,
            }
        ],
        "runtime_conditioning": [
            "noise_variance",
            "average_power_budget",
        ],
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _training_config(
    plan: DelayedCsiResourceAllocationExportPlan,
    *,
    project_root: Path,
) -> JsonDict:
    root_bundle = Path("..")
    power_budgets = list(plan.benchmark_power_budgets)
    return {
        "schema_version": 1,
        "training_template": plan.template_id,
        "project_root": str(project_root),
        "framework": "torch",
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "allocator_step": plan.allocator_step.id,
            "channel_state_step": plan.channel_state_step.id,
            "csi_observation_step": plan.csi_observation_step.id,
            "reliability_metrics_step": plan.reliability_metrics_step.id,
        },
        "data": {
            "csi_history_tap": plan.delayed_csi_tap,
            "delayed_csi_tap": plan.delayed_csi_tap,
            "current_csi_tap": plan.current_csi_tap,
            "delayed_csi_reference": plan.delayed_csi_reference,
            "current_csi_reference": plan.current_csi_reference,
            "train_capture_dirs": [str(root_bundle / "data" / "train")],
            "validation_capture_dirs": [
                str(root_bundle / "data" / "validation")
            ],
            "test_capture_dirs": [str(root_bundle / "data" / "test")],
            "subcarrier_count": plan.subcarrier_count,
            "csi_history_length": plan.csi_history_length,
            "alignment": "same_capture_row_same_sionna_tdl_trajectory",
            "allocation_label_tap": None,
        },
        "objective": {
            "loss": plan.loss,
            "direction": "minimize",
            "formula": (
                "-mean(target_rate_bps_hz * "
                "(1 - normal_approximation_bler(current_gain, power)))"
            ),
            "blocklength_channel_uses": plan.blocklength_channel_uses,
            "target_rate_bps_hz": plan.target_rate_bps_hz,
            "include_third_order_term": plan.include_third_order_term,
            "uses_allocation_labels": False,
            "uses_current_csi_for_training_loss": True,
            "uses_current_csi_at_runtime": False,
        },
        "constraints": {
            "nonnegative_power": "exact_euclidean_simplex_projection",
            "instantaneous_sum_power": (
                "subcarrier_count * average_power_budget"
            ),
        },
        "model": {
            "class": "CausalCsiHistoryPowerAllocator",
            "architecture": "causal_csi_history_frequency_residual_cnn",
            "history_length": plan.csi_history_length,
            "hidden_dim": 48,
            "dilations": [1, 2, 4, 8, 16],
            "kernel_size": 3,
            "output_projection": "euclidean_simplex",
            "runtime_inputs": [
                "csi_history",
                "noise_variance",
                "average_power_budget",
            ],
        },
        "training": {
            "epochs": 120,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 24,
            "initialization_seeds": [23, 41],
            "num_workers": 0,
            "device": "cuda_if_available",
            "average_power_budget_range": [
                min(power_budgets),
                max(power_budgets),
            ],
            "noise_variance_range": [
                plan.noise_variance,
                plan.noise_variance,
            ],
            "validation_average_power_budgets": list(power_budgets),
            "validation_noise_variances": [plan.noise_variance],
            "minimum_relative_validation_goodput_improvement": 0.005,
            "maximum_relative_validation_point_regression": 0.002,
            "validation_confidence_level": 0.95,
            "minimum_validation_cluster_count": 30,
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/power_policy.onnx",
            "artifact_entrypoint": "power_policy",
            "artifact_operation": plan.allocator_step.op,
            "artifact_id": "%s.%s.delayed_csi_finite_blocklength_allocator"
            % (_safe_stem(plan.recipe.name), _safe_stem(plan.allocator_step.id)),
            "artifact_name": "Reliability-aware delayed-CSI allocator for %s"
            % plan.recipe.name,
            "artifact_label": "Learned reliability allocator · %s"
            % plan.recipe.name,
        },
        "evaluation": {
            "primary_metric": (
                "resource.finite_blocklength.expected_goodput_bps_hz"
            ),
            "average_power_budgets": list(power_budgets),
            "noise_variances": [plan.noise_variance],
            "robust_gain_shrinkage": plan.robust_gain_shrinkage,
            "csi_prediction_horizon_ofdm_symbols": (
                plan.csi_prediction_horizon_ofdm_symbols
            ),
            "csi_prediction_gain_confidence": (
                plan.csi_prediction_gain_confidence
            ),
            "comparison_methods": [
                "equal_power",
                "water_filling_on_delayed_csi",
                "uncertainty_shrunk_water_filling",
                "complex_ar_prediction_plus_water_filling",
                "learned_allocator",
                "current_csi_shannon_diagnostic",
                "current_csi_finite_blocklength_numerical_diagnostic",
            ],
            "current_csi_shannon_diagnostic_role": (
                "diagnostic_only_not_finite_blocklength_optimum"
            ),
            "current_csi_finite_blocklength_numerical_diagnostic_role": (
                "nondeployable_perfect_information_numerical_reference"
            ),
            "test_split_exposed_to_training": False,
        },
    }


def _reliability_contract(
    plan: DelayedCsiResourceAllocationExportPlan,
) -> JsonDict:
    return {
        "schema_version": 1,
        "kind": "noema.delayed_csi_reliability_allocation_training_contract",
        "training_template": plan.template_id,
        "runtime_abi": {
            "inputs": {
                "csi_history": {
                    "semantic": (
                        "causal_delayed_noisy_complex_transmitter_csi_history"
                    ),
                    "dtype": "float32",
                    "shape": [
                        "batch",
                        plan.csi_history_length,
                        plan.subcarrier_count,
                        2,
                    ],
                },
                "noise_variance": {
                    "dtype": "float32",
                    "shape": ["batch", 1],
                },
                "average_power_budget": {
                    "dtype": "float32",
                    "shape": ["batch", 1],
                },
            },
            "outputs": {
                "allocation_scores": {
                    "dtype": "float32",
                    "shape": ["batch", plan.subcarrier_count],
                }
            },
            "excludes": [
                "current_csi",
                "oracle_allocation",
                "source_allocator_output",
            ],
        },
        "training_only": {
            "current_csi": {
                "tap_id": plan.current_csi_tap,
                "reference": plan.current_csi_reference,
                "semantic": "aligned_current_channel_power_gain_for_loss",
                "runtime_input": False,
            }
        },
        "training": {
            "allocation_labels_used": False,
            "loss": plan.loss,
            "blocklength_channel_uses": plan.blocklength_channel_uses,
            "target_rate_bps_hz": plan.target_rate_bps_hz,
            "include_third_order_term": plan.include_third_order_term,
            "checkpoint_selection": (
                "validation_expected_finite_blocklength_goodput"
            ),
        },
        "constraints": {
            "minimum_power": 0.0,
            "sum_power": "%d * average_power_budget"
            % plan.subcarrier_count,
        },
    }


def _project_manifest(
    plan: DelayedCsiResourceAllocationExportPlan,
    *,
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    root_bundle = out_dir.parent
    return {
        "schema_version": 1,
        "kind": "noema.standalone_training_project",
        "exporter": "delayed-csi-resource-allocation",
        "training_template": plan.template_id,
        "source_recipe": plan.recipe.name,
        "source_recipe_sha256": plan.recipe_sha256,
        "out_dir": _project_path(out_dir, project_root),
        "capture_plan": plan.capture_plan.to_dict(),
        "capture_jobs": _capture_jobs(
            plan,
            root_bundle=root_bundle,
            project_root=project_root,
        ),
        "training": {
            "owner": "external",
            "working_directory": _project_path(out_dir, project_root),
            "requires_python": ">=3.11,<3.14",
            "dependency_file": _project_path(
                out_dir / "requirements.txt",
                project_root,
            ),
            "setup_command": "python -m pip install -r requirements.txt",
            "command": "python train.py",
            "objective": plan.loss,
            "allocation_labels_used": False,
            "current_csi_runtime_input": False,
            "artifact_manifest_path": _project_path(
                root_bundle / "trained_artifact.yaml",
                project_root,
            ),
            "component_paths": [
                _project_path(
                    root_bundle / "artifacts" / "power_policy.onnx",
                    project_root,
                )
            ],
            "history_path": _project_path(
                out_dir / "training_history.json",
                project_root,
            ),
        },
        "evaluation": {
            "owner": "external",
            "command": "python evaluate.py",
            "metrics_path": _project_path(
                out_dir / "evaluation_metrics.json",
                project_root,
            ),
            "split": "test",
        },
        "post_training": {
            "owner": "noema_generated",
            "helper_path": _project_path(
                out_dir / "build_benchmark.py",
                project_root,
            ),
            "command": "python build_benchmark.py",
            "benchmark_pack_path": _project_path(
                out_dir / "benchmark_pack.yaml",
                project_root,
            ),
            "comparison": {
                "methods": [
                    "equal_power",
                    "water_filling_on_delayed_csi",
                    "uncertainty_shrunk_water_filling",
                    "learned_allocator",
                    "current_csi_shannon_diagnostic",
                ],
                "paired_held_out_seeds": True,
                "sweep": "average_transmit_power_budget",
                "current_csi_shannon_diagnostic_role": (
                    "diagnostic_only_not_finite_blocklength_optimum"
                ),
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.allocator_step.op,
                "step_id": plan.allocator_step.id,
                "manifest_path": _project_path(
                    root_bundle / "trained_artifact.yaml",
                    project_root,
                ),
            }
        ],
    }


def _capture_jobs(
    plan: DelayedCsiResourceAllocationExportPlan,
    *,
    root_bundle: Path,
    project_root: Path,
) -> list[JsonDict]:
    jobs = []
    for split in plan.capture_plan.splits:
        recipe_path = root_bundle / ("capture_%s_recipe.yaml" % split.id)
        jobs.append(
            {
                "split": split.id,
                "label": (
                    "Held-out test" if split.id == "test" else split.id.title()
                ),
                "recipe_path": _project_path(recipe_path, project_root),
                "bundle_recipe_path": recipe_path.name,
                "output_dir": _project_path(
                    root_bundle / "data" / split.id,
                    project_root,
                ),
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
                **(
                    {
                        "recipe_file_sha256": _file_sha256(recipe_path),
                    }
                    if recipe_path.is_file()
                    else {}
                ),
            }
        )
    return jobs


def _tap_id(recipe: Recipe, reference: str, fallback: str) -> str:
    for item in list((recipe.dataset_capture or {}).get("taps") or []):
        if (
            isinstance(item, Mapping)
            and str(item.get("from") or "") == reference
        ):
            return str(item.get("id") or fallback)
    return fallback


def _require_output_kind(
    registry: OperationRegistry,
    step: RecipeStep,
    output_name: str,
    expected_kind: str,
) -> None:
    description = registry.get(step.op).describe()
    output_kinds = dict(description.get("output_kinds") or {})
    actual_kind = str(output_kinds.get(output_name) or "")
    if actual_kind != expected_kind:
        raise DelayedCsiResourceAllocationExportError(
            "%s.%s must produce %s; got %s"
            % (step.id, output_name, expected_kind, actual_kind or "missing")
        )


def _step(recipe: Recipe, step_id: str) -> RecipeStep:
    for step in recipe.steps:
        if step.id == step_id:
            return step
    raise DelayedCsiResourceAllocationExportError(
        "Recipe has no step %s" % step_id
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DelayedCsiResourceAllocationExportError(
            "%s must be a positive integer" % label
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DelayedCsiResourceAllocationExportError(
            "%s must be a positive integer" % label
        ) from exc
    if parsed < 1 or (
        isinstance(value, float) and not float(value).is_integer()
    ):
        raise DelayedCsiResourceAllocationExportError(
            "%s must be a positive integer" % label
        )
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DelayedCsiResourceAllocationExportError(
            "%s must be finite and greater than zero" % label
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise DelayedCsiResourceAllocationExportError(
            "%s must be finite and greater than zero" % label
        )
    return parsed


def _closed_unit_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DelayedCsiResourceAllocationExportError(
            "%s must be finite and in [0, 1]" % label
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise DelayedCsiResourceAllocationExportError(
            "%s must be finite and in [0, 1]" % label
        )
    return parsed


def _configured_power_budgets(
    recipe: Recipe,
    fallback: float,
) -> tuple[float, ...]:
    matrix = recipe.metadata.get("matrix")
    dimensions = (
        matrix.get("dimensions")
        if isinstance(matrix, Mapping)
        else None
    )
    raw = (
        dimensions.get("resource.average_transmit_power_budget")
        if isinstance(dimensions, Mapping)
        else None
    )
    if not isinstance(raw, list) or not raw:
        return (
            max(0.5 * float(fallback), 1e-6),
            float(fallback),
            2.0 * float(fallback),
        )
    parsed = tuple(
        sorted(
            {
                _positive_float(
                    value,
                    "resource.average_transmit_power_budget matrix value",
                )
                for value in raw
            }
        )
    )
    if not parsed:
        raise DelayedCsiResourceAllocationExportError(
            "resource.average_transmit_power_budget matrix must not be empty"
        )
    return parsed


def _template_dir() -> Path:
    relative = (
        Path("demo_trainings")
        / "resource_allocation_delayed_csi_finite_blocklength"
    )
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise DelayedCsiResourceAllocationExportError(
        "Delayed-CSI finite-blocklength demonstration project was not found; "
        "expected %s" % relative
    )


def _project_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str) -> str:
    return (
        "".join(
            character
            if character.isalnum() or character in "_-"
            else "_"
            for character in str(value)
        ).strip("_")
        or "delayed_csi_resource_allocation"
    )
