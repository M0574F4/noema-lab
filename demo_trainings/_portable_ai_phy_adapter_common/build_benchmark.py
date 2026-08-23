from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

import task
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SEEDS = (81001, 82001, 83001)
DEFAULT_SNR_DB = (0.0, 15.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a paired post-training sensing/beamforming benchmark."
    )
    parser.add_argument("--artifact", help="Returned trained_artifact.yaml")
    parser.add_argument("--recipe", default="noema_recipe.yaml")
    parser.add_argument("--output", default="benchmark_pack.yaml")
    parser.add_argument("--project-root")
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--snr-db", default=",".join(str(value) for value in DEFAULT_SNR_DB))
    args = parser.parse_args()
    config = _mapping(Path("train_config.yaml"))
    training = dict(config.get("training") or {})
    project_root = Path(
        args.project_root or config.get("project_root") or "."
    ).expanduser().resolve()
    artifact_path = Path(
        args.artifact or training.get("artifact_manifest_path") or "../trained_artifact.yaml"
    ).expanduser().resolve()
    recipe_path = Path(args.recipe).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    seeds = _integers(args.seeds)
    snr_values = _floats(args.snr_db)
    inspected = inspect_trained_artifact(
        artifact_path,
        project_root=project_root,
        registry=build_registry(),
    )
    if not bool(inspected.get("ready")):
        raise ValueError(
            "trained artifact is not runnable: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    bindings = [
        item
        for item in list(inspected.get("compatible_operations") or [])
        if str(item.get("operation") or "") == task.OPERATION_ID
    ]
    if len(bindings) != 1 or not bool(bindings[0].get("available")):
        raise ValueError("trained artifact has no available %s binding" % task.OPERATION_ID)
    try:
        recipe_reference = str(recipe_path.relative_to(output_path.parent))
    except ValueError:
        recipe_reference = str(recipe_path)
    try:
        artifact_reference = str(artifact_path.relative_to(project_root))
    except ValueError as exc:
        raise ValueError("trained artifact must be inside the Noema project root") from exc
    package_sha = str(inspected.get("package_sha256") or "")
    pack = _build_pack(
        recipe_reference=recipe_reference,
        artifact_reference=artifact_reference,
        package_sha256=package_sha,
        replacement_step=str((config.get("recipe") or {}).get("replacement_step") or ""),
        seeds=seeds,
        snr_values=snr_values,
        artifact_path=artifact_path,
        evidence_base=output_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    print("wrote paired benchmark: %s" % output_path)
    print("run: noema benchmark run %s --project-root %s" % (output_path, project_root))
    return 0


def _build_pack(
    *,
    recipe_reference: str,
    artifact_reference: str,
    package_sha256: str,
    replacement_step: str,
    seeds: list[int],
    snr_values: list[float],
    artifact_path: Path,
    evidence_base: Path,
) -> dict[str, Any]:
    settings = _settings()
    methods = task.benchmark_methods(artifact_reference, package_sha256)
    recipes = []
    for snr_db in snr_values:
        for seed in seeds:
            for method_id, label, role, method_params in methods:
                step_params: dict[str, dict[str, Any]] = {
                    replacement_step: dict(method_params),
                    settings["snr_step"]: {"snr_db": float(snr_db)},
                }
                for step_id, offset in settings["seed_steps"].items():
                    step_params.setdefault(step_id, {})["seed"] = int(seed + offset)
                recipes.append(
                    {
                        "id": "%s_snr%s_seed%d"
                        % (method_id, _number_id(snr_db), seed),
                        "label": "%s · %g dB · seed %d" % (label, snr_db, seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method_id,
                            "matrix_selection": {"channel.snr_db": float(snr_db)},
                            "step_params": step_params,
                        },
                    }
                )
    return {
        "schema_version": 1,
        "id": settings["benchmark_id"],
        "version": "1.0.0",
        "name": settings["name"],
        "description": settings["description"],
        "suite": settings["suite"],
        "dataset": settings["dataset"],
        "task": settings["task"],
        "metrics": settings["metrics"],
        "baselines": [method[0] for method in methods if method[2] != "candidate"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "paired_held_out_seeds": True,
            "default_plot": {
                "plot": "graceful-degradation",
                "x": "channel.snr_db",
                "y": settings["primary_metric"],
            },
            "demo": _demo_metadata(
                settings,
                methods,
                trained_artifact=_file_evidence(artifact_path, evidence_base),
                training_history=_file_evidence(Path("training_history.json"), evidence_base),
                evaluation_metrics=_file_evidence(Path("evaluation_metrics.json"), evidence_base),
            ),
            "notes": settings["notes"],
        },
    }


def _demo_metadata(
    settings: Mapping[str, Any],
    methods,
    *,
    trained_artifact: Mapping[str, str],
    training_history: Mapping[str, str],
    evaluation_metrics: Mapping[str, str],
) -> dict[str, Any]:
    candidate = next(method for method in methods if method[2] == "candidate")
    return {
        "schema_version": 1,
        "slug": settings["benchmark_id"].replace(".", "-"),
        "title": settings["name"],
        "summary": settings["description"],
        "question": settings["question"],
        "tutorial": settings["tutorial"],
        "held_constant": settings["held_constant"],
        "changed": ["replacement method", "channel SNR", "held-out paired seed"],
        "primary_metric": settings["primary_metric"],
        "comparison_axis": "channel.snr_db",
        "series": [
            {"id": method_id, "label": label, "role": role}
            for method_id, label, role, _ in methods
        ],
        "plots": [
            {
                "id": "primary-metric-vs-snr",
                "title": "%s vs SNR" % settings["primary_metric"],
                "kind": "line",
                "x": "channel.snr_db",
                "y": settings["primary_metric"],
                "group": "benchmark_method",
                "method_order": [method[0] for method in methods],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            }
        ],
        "table_metrics": ["channel.snr_db", settings["primary_metric"], "task.score"],
        "training_evidence": [
            {
                "series": candidate[0],
                "trained_artifact_manifest": dict(trained_artifact),
                "training_history": dict(training_history),
                "evaluation_metrics": dict(evaluation_metrics),
            }
        ],
    }


def _settings() -> dict[str, Any]:
    channel_metric = {
        "id": "channel.snr_db",
        "family": "channel",
        "unit": "dB",
        "direction": "neutral",
        "definition_version": 1,
    }
    if task.TASK_ID == "isac_joint_allocation":
        channel_metric.update(
            {"source_step": "data", "source_operation": "source.isac_ofdm_scenario"}
        )
        return {
            "benchmark_id": "resource_allocation.learned_isac_joint_allocation_v1",
            "name": "Learned Joint ISAC OFDM Allocation",
            "description": "Equal power, communication-only water filling, per-scene scalarized optimization, and a learned policy on paired synthetic communication/sensing gains.",
            "suite": {"id": "resource_allocation", "name": "Resource Allocation", "status": "experimental", "version": "v1-draft"},
            "dataset": {"id": "synthetic_isac_frequency_response", "modality": "wireless", "version": "synthetic-isac-frequency-response-v1", "split": "held_out_seeded"},
            "task": {"id": "joint_isac_resource_allocation", "kind": "link_and_sensing_optimization", "modality": "wireless"},
            "metrics": [channel_metric, {"id": "isac.scalarized_utility", "family": "isac", "unit": "utility", "direction": "higher_is_better"}, {"id": "isac.communication_rate_bps_hz", "family": "link", "unit": "bit/s/Hz", "direction": "higher_is_better"}, {"id": "isac.sensing_snr_db", "family": "sensing", "unit": "dB", "direction": "higher_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
            "primary_metric": "isac.scalarized_utility",
            "snr_step": "data",
            "seed_steps": {"data": 0},
            "notes": "All methods receive identical held-out per-subcarrier gains, noise, sensing weight, and power budget; the iterative reference is an explicitly labeled numerical oracle for this synthetic objective.",
            "question": "Can a portable policy recover the declared communication/sensing tradeoff without oracle allocation labels?",
            "tutorial": "../../../tutorials/learned_isac_ofdm_allocation_demo.html",
            "held_constant": ["twelve-subcarrier synthetic response", "unit total-power simplex", "paired communication and sensing gains", "fixed scalarized utility"],
        }
    if task.TASK_ID == "near_field_range_angle":
        channel_metric.update(
            {"source_step": "data", "source_operation": "source.near_field_xl_mimo_scenario"}
        )
        return {
            "benchmark_id": "localization_sensing.learned_near_field_focusing_v1",
            "name": "Learned Near-Field Range-Angle Focusing",
            "description": "Far-field steering, polar-codebook search, simulation-truth focusing, and a learned range-angle estimator on paired spherical-wave observations.",
            "suite": {"id": "localization_sensing", "name": "Localization / Sensing", "status": "experimental", "version": "v1-draft"},
            "dataset": {"id": "synthetic_near_field_array", "modality": "wireless", "version": "synthetic-near-field-array-v1", "split": "held_out_seeded"},
            "task": {"id": "near_field_range_angle_focusing", "kind": "estimation_and_beamforming", "modality": "wireless"},
            "metrics": [channel_metric, {"id": "near_field.normalized_focusing_gain", "family": "beamforming", "unit": "ratio", "direction": "higher_is_better"}, {"id": "near_field.range_rmse_m", "family": "sensing", "unit": "m", "direction": "lower_is_better"}, {"id": "near_field.angle_rmse_deg", "family": "sensing", "unit": "degree", "direction": "lower_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
            "primary_metric": "near_field.normalized_focusing_gain",
            "snr_step": "data",
            "seed_steps": {"data": 0},
            "notes": "All methods receive the same coherent 28 GHz spherical-wave observations; true-position focusing is a simulation-only upper bound.",
            "question": "Can a learned estimator convert near-field phase curvature into useful range-angle focusing under the same observations as classical searches?",
            "tutorial": "../../../tutorials/learned_near_field_xl_mimo_demo.html",
            "held_constant": ["32-element half-wavelength array", "28 GHz carrier", "paired target states and noise", "common physical output bounds"],
        }
    if task.TASK_ID == "leo_ntn_tracking":
        channel_metric.update(
            {"source_step": "data", "source_operation": "source.leo_ntn_tracking_scenario"}
        )
        return {
            "benchmark_id": "beamforming_precoding.learned_leo_ntn_tracking_v1",
            "name": "Learned LEO-NTN Doppler and Beam Tracking",
            "description": "Hold-last, linear extrapolation, future-state oracle, and a learned causal tracker on paired synthetic LEO observation histories.",
            "suite": {"id": "beamforming_precoding", "name": "Beamforming / Precoding", "status": "experimental", "version": "v1-draft"},
            "dataset": {"id": "synthetic_leo_ntn_track", "modality": "wireless", "version": "synthetic-leo-ntn-track-v1", "split": "held_out_seeded"},
            "task": {"id": "leo_ntn_doppler_beam_tracking", "kind": "causal_prediction_and_handover", "modality": "wireless"},
            "metrics": [channel_metric, {"id": "ntn.beam_handover_accuracy", "family": "mobility", "unit": "ratio", "direction": "higher_is_better"}, {"id": "ntn.doppler_mae_hz", "family": "channel", "unit": "Hz", "direction": "lower_is_better"}, {"id": "ntn.beam_outage_rate", "family": "mobility", "unit": "ratio", "direction": "lower_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
            "primary_metric": "ntn.beam_handover_accuracy",
            "snr_step": "data",
            "seed_steps": {"data": 0},
            "notes": "Every method receives the same causal observation history and must predict at the same one-second horizon; the future-state oracle is simulation-only.",
            "question": "Can a learned causal history model improve the joint Doppler and next-beam decision under noisy observations?",
            "tutorial": "../../../tutorials/learned_leo_ntn_tracking_demo.html",
            "held_constant": ["six-sample causal history", "one-second prediction horizon", "nine fixed beam sectors", "paired synthetic tracks and measurement noise"],
        }
    if task.TASK_ID == "range_localization":
        channel_metric.update(
            {"source_step": "range_observation", "source_operation": "wireless.range_observation"}
        )
        return {
            "benchmark_id": "localization_sensing.learned_range_localizer_v1",
            "name": "Learned Range Localization",
            "description": "Linear, regularized, and learned range localizers on paired noisy four-anchor scenes.",
            "suite": {"id": "localization_sensing", "name": "Localization / Sensing", "status": "experimental", "version": "v1-draft"},
            "dataset": {"id": "synthetic_localization_geometry", "modality": "wireless", "version": "synthetic-localization-geometry-v1", "split": "held_out_seeded"},
            "task": {"id": "wireless_localization", "kind": "estimation", "modality": "wireless"},
            "metrics": [channel_metric, {"id": "localization.rmse_m", "family": "sensing", "unit": "m", "direction": "lower_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
            "primary_metric": "localization.rmse_m",
            "snr_step": "range_observation",
            "seed_steps": {"data": 0, "range_observation": 100000},
            "notes": "All methods receive the same anchors and noisy ranges for every held-out seed; only the localizer changes.",
            "question": "Can a learned geometry-aware residual reduce held-out localization error without changing anchors, ranges, or noise?",
            "tutorial": "../../../tutorials/learned_range_localization_demo.html",
            "held_constant": ["four-anchor geometry protocol", "paired target positions", "paired range-noise seeds"],
        }
    if task.TASK_ID == "aoa_estimation":
        channel_metric.update(
            {"source_step": "array_observation", "source_operation": "wireless.ula_array_observation"}
        )
        return {
            "benchmark_id": "localization_sensing.learned_aoa_estimator_v1",
            "name": "Learned ULA AoA Estimation",
            "description": "Bartlett, MUSIC, and a learned covariance estimator on paired narrowband ULA snapshots.",
            "suite": {"id": "localization_sensing", "name": "Localization / Sensing", "status": "experimental", "version": "v1-draft"},
            "dataset": {"id": "synthetic_ula_aoa", "modality": "wireless", "version": "synthetic-ula-aoa-v1", "split": "held_out_seeded"},
            "task": {"id": "aoa_estimation", "kind": "estimation", "modality": "wireless"},
            "metrics": [channel_metric, {"id": "aoa.rmse_deg", "family": "sensing", "unit": "degree", "direction": "lower_is_better"}, {"id": "aoa.mae_deg", "family": "sensing", "unit": "degree", "direction": "lower_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
            "primary_metric": "aoa.rmse_deg",
            "snr_step": "array_observation",
            "seed_steps": {"data": 0, "array_observation": 100000},
            "notes": "All methods receive the same single-source half-wavelength ULA snapshots for every held-out seed; only the estimator changes.",
            "question": "Can a covariance-domain learned estimator improve held-out angle error under the same narrowband snapshots used by Bartlett and MUSIC?",
            "tutorial": "../../../tutorials/learned_aoa_estimation_demo.html",
            "held_constant": ["single-source half-wavelength ULA protocol", "paired source angles", "paired snapshot and AWGN seeds"],
        }
    channel_metric.update(
        {"source_step": "data", "source_operation": "source.beamforming_scenario"}
    )
    return {
        "benchmark_id": "beamforming_precoding.learned_beam_selection_v1",
        "name": "Learned MISO Beam Selection",
        "description": "A learned finite-codebook policy compared with exhaustive codebook selection and perfect-CSIT MRT on paired channels.",
        "suite": {"id": "beamforming_precoding", "name": "Beamforming / Precoding", "status": "experimental", "version": "v1-draft"},
        "dataset": {"id": "synthetic_beamforming", "modality": "wireless", "version": "synthetic-beamforming-v1", "split": "held_out_seeded"},
        "task": {"id": "beamforming_precoding", "kind": "link_optimization", "modality": "wireless"},
        "metrics": [channel_metric, {"id": "beamforming.spectral_efficiency_bps_hz", "family": "link", "unit": "bit/s/Hz", "direction": "higher_is_better"}, {"id": "task.score", "family": "task", "unit": "score", "direction": "higher_is_better"}],
        "primary_metric": "beamforming.spectral_efficiency_bps_hz",
        "snr_step": "data",
        "seed_steps": {"data": 0},
        "notes": "All three methods receive the same held-out flat-fading channels; MRT and exhaustive codebook search are explicitly labeled oracles.",
        "question": "Can a learned policy recover the exhaustive finite-codebook beam choice on unseen channels?",
        "tutorial": "../../../tutorials/learned_beam_selection_demo.html",
        "held_constant": ["single-user eight-antenna MISO protocol", "common DFT codebook", "paired held-out channel vectors"],
    }


def _mapping(path: Path) -> dict[str, Any]:
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping: %s" % path)
    return dict(payload)


def _integers(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seeds must be a non-empty unique integer list")
    return values


def _floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("snr-db must be a non-empty numeric list")
    return values


def _number_id(value: float) -> str:
    return ("%g" % value).replace("-", "m").replace(".", "p")


def _file_evidence(path: Path, relative_to: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {
        "path": os.path.relpath(str(resolved), str(relative_to.resolve())),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
