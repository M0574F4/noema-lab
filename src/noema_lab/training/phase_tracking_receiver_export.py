from __future__ import annotations

"""Optional packet-context receiver demonstration scaffold.

The neutral Workbench contract remains architecture and trainer independent.
This module only copies the checked-in example after the researcher explicitly
attaches the phase-tracking demonstration.
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from noema_lab.training.starter_refresh import prepare_demo_starter_directory
from noema_lab.training.package_layout import find_demo_training_dir
from noema_lab.training.standalone_input import write_standalone_structured_input


JsonDict = Dict[str, Any]

PHASE_TRACKING_CAPTURE_SNR_DB = (-2.0, 2.0, 6.0, 10.0)


class PhaseTrackingReceiverExportError(ValueError):
    pass


def write_phase_tracking_receiver_starter(
    plan: Any,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    """Write the non-normative temporal receiver example beside a contract."""

    destination = prepare_demo_starter_directory(
        Path(out_dir),
        force=force,
        error_type=PhaseTrackingReceiverExportError,
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
        shutil.copy2(template / filename, destination / filename)
    write_standalone_structured_input(destination)
    _write_yaml(destination / "training_template.yaml", _training_template())

    project_root = Path(plan.project_root or Path.cwd()).resolve()
    config = _training_config(plan, project_root=project_root)
    _write_yaml(destination / "train_config.yaml", config)
    _write_yaml(destination / "noema_recipe.yaml", plan.recipe.to_dict())
    manifest = _starter_project_manifest(
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
        "exporter": "phase-tracking-receiver",
        "training_template": "neural_receiver.phase_tracking_qpsk",
        "framework": plan.framework,
        "loss": plan.loss,
        "out_dir": str(destination),
        "optimizable_steps": [plan.receiver_step.id],
        "features": [
            {
                "step_id": plan.feature_step.id,
                "kind": plan.feature_kind,
                "from": plan.feature_reference,
                "runtime_input": True,
            },
            {
                "step_id": plan.pilot_context_step.id,
                "kind": plan.pilot_context_kind,
                "from": plan.pilot_context_reference,
                "runtime_input": True,
            },
        ],
        "target": {
            "step_id": plan.target_step.id,
            "kind": plan.target_kind,
            "from": plan.target_reference,
            "runtime_input": False,
        },
        "oracle_phase_is_runtime_input": False,
        "project_manifest": manifest,
        "capture_jobs": list(manifest["capture_jobs"]),
        "trained_artifacts": list(manifest["trained_artifacts"]),
        "files": files,
    }


def _training_config(plan: Any, *, project_root: Path) -> JsonDict:
    root_bundle = Path("..")
    phase_truth_reference = str(
        plan.receiver_step.inputs.get("phase_truth") or ""
    ).strip()
    phase_truth_tap = _optional_tap_id(plan, phase_truth_reference)
    return {
        "schema_version": 1,
        "training_template": "neural_receiver.phase_tracking_qpsk",
        "project_root": str(project_root),
        "framework": "torch",
        "recipe": {
            "name": plan.recipe.name,
            "sha256": plan.recipe_sha256,
            "receiver_step": plan.receiver_step.id,
            "receiver_operation": plan.receiver_step.op,
            "feature_reference": plan.feature_reference,
            "pilot_context_reference": plan.pilot_context_reference,
            "target_reference": plan.target_reference,
        },
        "data": {
            "feature_tap": _tap_id(plan, plan.feature_reference),
            "pilot_context_tap": _tap_id(
                plan,
                plan.pilot_context_reference,
            ),
            "target_tap": _tap_id(plan, plan.target_reference),
            "phase_truth_tap": phase_truth_tap,
            "feature_reference": plan.feature_reference,
            "pilot_context_reference": plan.pilot_context_reference,
            "target_reference": plan.target_reference,
            "phase_truth_reference": phase_truth_reference,
            "pilot_smoothing_neighbors": int(
                plan.receiver_step.params.get(
                    "pilot_smoothing_nearest_pilots",
                    5,
                )
            ),
            "train_capture_dirs": [str(root_bundle / "data" / "train")],
            "validation_capture_dirs": [
                str(root_bundle / "data" / "validation")
            ],
            "test_capture_dirs": [str(root_bundle / "data" / "test")],
            "modulation": "qpsk",
            "bits_per_data_symbol": 2,
            "receiver_feature_channels": [
                "pilot_smoothing_corrected_real",
                "pilot_smoothing_corrected_imag",
                "received_real",
                "received_imag",
                "pilot_mask",
                "pilot_innovation_relative_to_smoothing_real",
                "pilot_innovation_relative_to_smoothing_imag",
                "pilot_smoothing_phasor_real",
                "pilot_smoothing_phasor_imag",
                "normalized_negative_qpsk_fourth_power_real",
                "normalized_negative_qpsk_fourth_power_imag",
            ],
            "phase_truth_used_as_training_target": bool(phase_truth_tap),
            "phase_truth_used_at_runtime": False,
        },
        "model": {
            "class": "validation_selected_packet_context_receiver",
            "architecture": "candidate_search",
            "candidates": [
                {
                    "id": "pilot_smoother_residual_tcn_32",
                    "architecture": "pilot_smoother_residual_tcn",
                    "hidden_dim": 32,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                    "kernel_size": 5,
                },
                {
                    "id": "pilot_smoother_residual_tcn_48",
                    "architecture": "pilot_smoother_residual_tcn",
                    "hidden_dim": 48,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                    "kernel_size": 5,
                },
            ],
        },
        "objective": {
            "loss": (
                "phase_only_pretraining_then_masked_bit_bce_plus_"
                "training_only_circular_phase_loss"
            ),
            "llr_sign": "positive_bit_zero",
            "mask": "nonpilot_data_symbols",
            "checkpoint_selection": [
                "pilot_smoothing_noninferiority_guard",
                "minimum_validation_ber_improvement",
                "minimum_worst_per_snr_ber_regression",
                "configured_candidate_order",
                "configured_seed_order",
                "earliest_epoch",
            ],
        },
        "training": {
            "epochs": 48,
            "batch_size": 16,
            "learning_rate": 2e-3,
            "weight_decay": 1e-5,
            "initialization_seeds": [23, 41],
            "num_workers": 0,
            "phase_pretraining_epochs": 16,
            "phase_loss_weight": 1.0,
            "minimum_epochs": 8,
            "early_stopping_patience": 8,
            "minimum_validation_ber_improvement": 0.001,
            "per_snr_noninferiority_margin": 0.001,
            "high_snr_curriculum": {
                "enabled": True,
                "threshold_db": 6.0,
                "start_weight": 2.0,
                "end_weight": 1.0,
                "anneal_epochs": 16,
            },
            "minimum_learning_rate": 1e-5,
            "device": "cuda_if_available",
            "artifact_manifest_path": "../trained_artifact.yaml",
            "artifact_component_path": "../artifacts/phase_tracking_receiver.onnx",
            "artifact_entrypoint": "phase_tracking_receiver",
            "artifact_operation": plan.receiver_step.op,
            "artifact_id": "%s.%s.phase_tracking_receiver"
            % (_safe_stem(plan.recipe.name), _safe_stem(plan.receiver_step.id)),
            "artifact_name": "Learned phase-tracking receiver for %s"
            % plan.recipe.name,
            "artifact_label": "Learned phase tracker · %s" % plan.recipe.name,
        },
        "evaluation": {
            "primary_metric": "bit_error_rate",
            "test_split_exposed_to_training": False,
            "pll_alpha": float(plan.receiver_step.params.get("pll_alpha", 0.12)),
            "pll_beta": float(plan.receiver_step.params.get("pll_beta", 0.005)),
            "pilot_smoothing_nearest_pilots": int(
                plan.receiver_step.params.get(
                    "pilot_smoothing_nearest_pilots",
                    5,
                )
            ),
            "comparison_methods": [
                "uncompensated_qpsk",
                "pilot_interpolation",
                "pilot_smoothing",
                "decision_directed_pll",
                "learned_receiver",
                "oracle_phase",
            ],
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
        "exporter": "phase-tracking-receiver",
        "training_template": "neural_receiver.phase_tracking_qpsk",
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
            "artifact_manifest_path": _project_path(
                root_bundle / "trained_artifact.yaml",
                project_root,
            ),
            "component_paths": [
                _project_path(
                    root_bundle / "artifacts" / "phase_tracking_receiver.onnx",
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
                    "uncompensated_qpsk",
                    "pilot_interpolation",
                    "pilot_smoothing",
                    "decision_directed_pll",
                    "learned_receiver",
                    "oracle_phase",
                ],
                "paired_held_out_seeds": True,
                "sweep": "channel_snr_db",
                "oracle_role": "upper_bound",
            },
        },
        "trained_artifacts": [
            {
                "role": "trained_model",
                "operation": plan.receiver_step.op,
                "step_id": plan.receiver_step.id,
                "manifest_path": _project_path(
                    root_bundle / "trained_artifact.yaml",
                    project_root,
                ),
            }
        ],
    }


def _capture_jobs(
    plan: Any,
    *,
    root_bundle: Path,
    project_root: Path,
) -> list[JsonDict]:
    jobs = []
    for split_spec in plan.capture_plan.splits:
        recipe_path = root_bundle / ("capture_%s_recipe.yaml" % split_spec.id)
        jobs.append(
            {
                "split": split_spec.id,
                "label": (
                    "Held-out test"
                    if split_spec.id == "test"
                    else split_spec.id.title()
                ),
                "recipe_path": _project_path(recipe_path, project_root),
                "bundle_recipe_path": recipe_path.name,
                "output_dir": _project_path(
                    root_bundle / "data" / split_spec.id,
                    project_root,
                ),
                "requested_samples": split_spec.samples,
                "expected_taps": [
                    {"id": tap.id, "from": tap.reference}
                    for tap in plan.capture_plan.taps
                ],
                "sample_unit": plan.capture_plan.sample_unit,
                "seed_offset": split_spec.seed_offset,
                "seed_role": split_spec.training_use,
                "owner": "noema",
                "consumer": "external_researcher",
                "source_recipe_sha256": plan.recipe_sha256,
            }
        )
    return jobs


def _template_dir() -> Path:
    relative = Path("demo_trainings") / "neural_receiver_phase_tracking_qpsk"
    candidate = find_demo_training_dir(relative)
    if candidate is not None:
        return candidate
    raise PhaseTrackingReceiverExportError(
        "Phase-tracking receiver demonstration project was not found; expected %s"
        % relative
    )


def _training_template() -> JsonDict:
    """Return the versioned contract summary shipped with a generated starter."""

    return {
        "schema_version": 1,
        "kind": "noema.demo_training_template",
        "id": "neural_receiver.phase_tracking_qpsk",
        "name": "Neural receiver · packet phase tracking",
        "status": "reference_only",
        "operation": "demodulation.phase_tracking_receiver_adapter",
        "framework": "torch",
        "data": {
            "owner": "noema_capture_contract",
            "runtime_features": [
                "pilot-smoothing-corrected and raw received QPSK frame",
                "public pilot innovations and smoothing phasor",
                "normalized fourth-power QPSK phase cue",
            ],
            "target": "transmitted data-bit boundary",
            "training_only_auxiliary_target": "oracle carrier phase trace",
        },
        "model": {
            "owner": "external_researcher",
            "reference_candidates": [
                "depthwise-separable smoothing-residual temporal receiver, width 32",
                "depthwise-separable smoothing-residual temporal receiver, width 48",
            ],
            "selection": (
                "pilot-smoothing noninferiority, minimum validation BER "
                "improvement, then deterministic tie breakers"
            ),
        },
        "loss": {
            "owner": "external_researcher",
            "reference_objective": (
                "phase-only pretraining followed by masked bit BCE plus a "
                "training-only circular residual-phase objective"
            ),
        },
        "artifact_return": {
            "schema": "noema.trained_block_artifact/v2",
            "runtime": "onnxruntime",
            "application": "single_binding",
            "entrypoint": "phase_tracking_receiver",
            "inputs": ["receiver_features_v3 [packet, frame_symbol, 11]"],
            "outputs": ["residual_phase_rad [packet, frame_symbol]"],
            "excludes": ["phase_truth"],
        },
        "post_training": {
            "helper": "build_benchmark.py",
            "output": "benchmark_pack.yaml",
            "comparison": [
                "uncompensated_qpsk",
                "pilot_interpolation",
                "pilot_smoothing",
                "decision_directed_pll",
                "learned_receiver",
                "oracle_phase",
            ],
            "sweep": "wireless_channel.snr_db",
            "pairing": (
                "common payload, AWGN, and carrier-impairment seeds at each SNR"
            ),
            "presentation": "metadata.demo@1",
        },
    }


def _tap_id(plan: Any, reference: str) -> str:
    for tap in plan.capture_plan.taps:
        if tap.reference == reference:
            return str(tap.id)
    raise PhaseTrackingReceiverExportError(
        "Required captured signal is missing from the resolved capture plan: %s"
        % reference
    )


def _optional_tap_id(plan: Any, reference: str) -> str:
    for tap in plan.capture_plan.taps:
        if tap.reference == reference:
            return str(tap.id)
    return ""


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


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
    return safe or "phase_tracking_receiver"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
