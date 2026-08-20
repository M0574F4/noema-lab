from __future__ import annotations

"""Build the paired post-training delayed-CSI allocation campaign."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_BUDGETS = (0.4, 0.6, 0.8, 1.0, 1.4)
DEFAULT_HELD_OUT_SEEDS = (95101, 95201, 95301)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paired finite-blocklength OFDM allocation benchmark."
        )
    )
    parser.add_argument("--artifact", help="Returned trained_artifact.yaml")
    parser.add_argument("--recipe", default="noema_recipe.yaml")
    parser.add_argument("--output", default="benchmark_pack.yaml")
    parser.add_argument("--project-root")
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_HELD_OUT_SEEDS),
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = Path(args.recipe).expanduser().resolve()
    artifact_path = _artifact_path(args.artifact)
    project_root = _project_root(args.project_root)
    budgets = _positive_floats(args.budgets, "budgets")
    seeds = _nonnegative_integers(args.seeds, "seeds")

    recipe = _load_mapping(recipe_path, "source recipe")
    _require_step(recipe, "channel_state", "wireless.ofdm_channel_state")
    _require_step(
        recipe,
        "csi_observation",
        "wireless.ofdm_delayed_csi",
    )
    _require_step(recipe, "tx_power", "model.causal_csi_power_allocator")
    _require_step(recipe, "wireless_channel", "wireless.channel")
    _require_step(
        recipe,
        "allocation_evaluation",
        "metrics.ofdm_finite_blocklength_allocation",
    )
    _assert_seed_namespace_is_held_out(recipe_path, seeds)
    artifact = _validate_artifact(artifact_path, project_root)
    history_path = Path("training_history.json").resolve()
    evaluation_path = Path("evaluation_metrics.json").resolve()
    _load_json_evidence(history_path, "training history")
    evaluation = _load_json_evidence(
        evaluation_path,
        "held-out evaluation",
    )
    _validate_evaluation_binding(
        evaluation,
        artifact_path=artifact_path,
        component_sha256=artifact["component_sha256"],
    )

    pack = _build_pack(
        recipe_reference=_relative_path(recipe_path, output_path.parent),
        artifact_reference=_relative_path(artifact_path, project_root),
        artifact_runtime_identity=artifact["runtime_identity_sha256"],
        artifact_evidence=_evidence_file_spec(
            artifact_path,
            output_path.parent,
        ),
        training_history=_evidence_file_spec(
            history_path,
            output_path.parent,
        ),
        evaluation_metrics=_evidence_file_spec(
            evaluation_path,
            output_path.parent,
        ),
        budgets=budgets,
        seeds=seeds,
    )
    output_path.write_text(
        yaml.safe_dump(pack, sort_keys=False),
        encoding="utf-8",
    )
    _print_commands(output_path, project_root)
    return 0


def _build_pack(
    *,
    recipe_reference: str,
    artifact_reference: str,
    artifact_runtime_identity: str,
    artifact_evidence: Mapping[str, str],
    training_history: Mapping[str, str],
    evaluation_metrics: Mapping[str, str],
    budgets: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    methods = (
        (
            "equal_power",
            "Equal power",
            "baseline",
            {"policy": "fixed"},
        ),
        (
            "observed_csi_water_filling",
            "Water filling on delayed/noisy CSI",
            "baseline",
            {"policy": "observed_csi_water_filling"},
        ),
        (
            "robust_csi_water_filling",
            "Uncertainty-shrunk water filling",
            "baseline",
            {
                "policy": "robust_csi_water_filling",
                "csi_gain_shrinkage": 0.6,
            },
        ),
        (
            "causal_ar_water_filling",
            "Complex-AR prediction + water filling",
            "baseline",
            {
                "policy": "causal_ar_water_filling",
                "csi_prediction_horizon_ofdm_symbols": 5,
                "csi_prediction_gain_confidence": 0.4,
            },
        ),
        (
            "learned_allocator",
            "Learned reliability-aware allocator",
            "candidate",
            {
                "policy": "learned_artifact",
                "artifact_manifest_path": artifact_reference,
                "artifact_entrypoint": "power_policy",
                "artifact_package_sha256": artifact_runtime_identity,
            },
        ),
    )
    recipes = []
    for budget in budgets:
        for paired_seed in seeds:
            for method_id, label, role, allocator_params in methods:
                tx_power = dict(allocator_params)
                tx_power["target_power"] = float(budget)
                recipes.append(
                    {
                        "id": "%s_p%s_seed%d"
                        % (method_id, _number_id(budget), paired_seed),
                        "label": "%s · P=%g · seed %d"
                        % (label, budget, paired_seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method_id,
                            "matrix_selection": {
                                "resource.average_transmit_power_budget": float(
                                    budget
                                ),
                                "benchmark.paired_seed": int(paired_seed),
                            },
                            "metadata": {
                                "seed": int(paired_seed) + 500_000,
                                "benchmark_method": method_id,
                                "benchmark_method_label": label,
                                "benchmark_paired_seed": int(paired_seed),
                                "pairing_id": str(paired_seed),
                                "aggregation_cell_id": (
                                    "average_power_budget=%s"
                                    % _number_id(budget)
                                ),
                                "statistical_unit": (
                                    "paired held-out TDL trajectory seed"
                                ),
                                "benchmark_held_out": True,
                            },
                            "step_params": {
                                "data": {"seed": int(paired_seed)},
                                "channel_state": {
                                    "seed": int(paired_seed) + 100_000,
                                    "average_power_budget": float(budget),
                                },
                                "csi_observation": {
                                    "seed": int(paired_seed) + 200_000,
                                },
                                "tx_power": tx_power,
                                "wireless_channel": {
                                    "seed": int(paired_seed) + 300_000,
                                },
                            },
                        },
                    }
                )

    method_order = [item[0] for item in methods]
    demo = {
        "schema_version": 1,
        "slug": "reliability-aware-delayed-csi-ofdm-allocation",
        "title": "Reliability-aware OFDM allocation with delayed CSI",
        "summary": (
            "A frequency-aware learned policy is compared with equal power and "
            "three deployable delayed-CSI baselines, including a causal "
            "complex-AR predictor, under paired temporally correlated TDL "
            "trajectories."
        ),
        "question": (
            "Can direct finite-blocklength goodput training exploit correlation "
            "between delayed transmitter CSI and the later channel better than "
            "Shannon-oriented observed-CSI heuristics?"
        ),
        "tutorial": (
            "../../../tutorials/"
            "reliability_aware_ofdm_allocation_demo.html"
        ),
        "held_constant": [
            "canonical recipe topology and finite-blocklength model",
            "TDL profile, mobility, CSI age, and CSI-estimation quality",
            "paired payload, channel, estimation-error, and AWGN seeds",
            "runtime information available to every deployable policy",
        ],
        "changed": [
            "allocator policy",
            "average transmit-power budget",
            "held-out paired seed",
        ],
        "primary_metric": (
            "resource.finite_blocklength.expected_goodput_bps_hz"
        ),
        "comparison_axis": "resource.average_transmit_power_budget",
        "series": [
            {"id": item[0], "label": item[1], "role": item[2]}
            for item in methods
        ],
        "plots": [
            {
                "id": "finite-blocklength-goodput-vs-power",
                "title": "Expected short-packet goodput vs power budget",
                "kind": "line",
                "x": "resource.average_transmit_power_budget",
                "y": (
                    "resource.finite_blocklength.expected_goodput_bps_hz"
                ),
                "group": "benchmark_method",
                "method_order": method_order,
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "predicted-bler-vs-power",
                "title": "Predicted short-packet BLER vs power budget",
                "kind": "line",
                "x": "resource.average_transmit_power_budget",
                "y": "resource.finite_blocklength.predicted_bler",
                "group": "benchmark_method",
                "method_order": method_order,
                "style": {"aggregation": "mean_ci", "y_scale": "log"},
            },
        ],
        "table_metrics": [
            "resource.finite_blocklength.expected_goodput_bps_hz",
            "resource.finite_blocklength.predicted_bler",
            "resource.finite_blocklength.mean_capacity_bps_hz",
            "resource.finite_blocklength.p05_goodput_bps_hz",
            "resource.csi.observed_actual_gain_correlation",
            "resource.power_constraint.max_abs_error",
            "resource.power_constraint.max_negative_violation",
        ],
        "training_evidence": [
            {
                "series": "learned_allocator",
                "trained_artifact_manifest": artifact_evidence,
                "training_history": training_history,
                "evaluation_metrics": evaluation_metrics,
            }
        ],
    }
    return {
        "schema_version": 1,
        "id": (
            "resource_allocation."
            "delayed_csi_finite_blocklength_post_training_v3"
        ),
        "version": "3.0.0",
        "name": "Delayed-CSI Finite-Blocklength OFDM Allocation",
        "description": (
            "Five deployable allocation policies compared with paired held-out "
            "TDL channel trajectories and a short-packet reliability objective."
        ),
        "suite": {
            "id": "resource_allocation",
            "name": "Resource Allocation",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "synthetic_random_bits_sionna_tdl_delayed_csi",
            "modality": "wireless",
            "version": "synthetic-random-bits-sionna-tdl-delayed-csi-v2",
            "split": "held_out_seeded_trajectory",
        },
        "task": {
            "id": "resource_allocation",
            "kind": "policy_optimization",
            "modality": "wireless",
        },
        "metrics": [
            {
                "id": "resource.average_transmit_power_budget",
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "system",
                "unit": "normalized_power",
                "direction": "neutral",
            },
            {
                "id": (
                    "resource.finite_blocklength."
                    "expected_goodput_bps_hz"
                ),
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "reliability",
                "unit": "bit/s/Hz",
                "direction": "higher_is_better",
            },
            {
                "id": "resource.finite_blocklength.predicted_bler",
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "reliability",
                "unit": "ratio",
                "direction": "lower_is_better",
            },
            {
                "id": (
                    "resource.finite_blocklength.mean_capacity_bps_hz"
                ),
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "system",
                "unit": "bit/s/Hz",
                "direction": "higher_is_better",
            },
            {
                "id": (
                    "resource.finite_blocklength.p05_goodput_bps_hz"
                ),
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "reliability",
                "unit": "bit/s/Hz",
                "direction": "higher_is_better",
            },
            {
                "id": "resource.csi.observed_actual_gain_correlation",
                "definition_version": 1,
                "source_step": "csi_observation",
                "source_operation": "wireless.ofdm_delayed_csi",
                "family": "channel",
                "unit": "correlation",
                "direction": "higher_is_better",
            },
            {
                "id": "resource.power_constraint.max_abs_error",
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "constraint",
                "unit": "normalized_power",
                "direction": "lower_is_better",
            },
            {
                "id": "resource.power_constraint.max_negative_violation",
                "definition_version": 1,
                "source_step": "allocation_evaluation",
                "source_operation": (
                    "metrics.ofdm_finite_blocklength_allocation"
                ),
                "family": "constraint",
                "unit": "normalized_power",
                "direction": "lower_is_better",
            },
        ],
        "baselines": [
            "equal_power",
            "observed_csi_water_filling",
            "robust_csi_water_filling",
            "causal_ar_water_filling",
        ],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "delayed-csi-fbl-allocation-v3",
            "paired_held_out_seeds": seeds,
            "average_power_budgets": budgets,
            "statistical_unit": "paired held-out TDL trajectory seed",
            "perfect_current_csi_available_to_deployable_policies": False,
            "demo": demo,
        },
    }


def _validate_artifact(
    path: Path,
    project_root: Path,
) -> dict[str, str]:
    manifest = _load_mapping(path, "trained artifact manifest")
    if int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("trained artifact requires training provenance")
    if bool(training.get("uses_power_allocation_labels", True)):
        raise ValueError("trained artifact unexpectedly used allocation labels")
    if bool(training.get("current_csi_available_at_runtime", True)):
        raise ValueError(
            "trained artifact illegally exposes current CSI at runtime"
        )
    validation_evidence = training.get("demo_evidence_status")
    if (
        not isinstance(validation_evidence, Mapping)
        or str(validation_evidence.get("status") or "")
        != "passed"
    ):
        raise ValueError(
            "trained artifact is runtime-valid but lacks sufficient validation "
            "evidence to be presented as a learned benchmark candidate"
        )
    if bool(training.get("equal_power_fallback", False)):
        raise ValueError(
            "training retained its epoch-zero equal-power safeguard; the "
            "artifact is valid but cannot be presented as a learned benchmark "
            "candidate"
        )
    bindings = manifest.get("compatible_operations") or []
    compatible = next(
        (
            item
            for item in bindings
            if isinstance(item, Mapping)
            and str(item.get("operation") or "")
            == "model.causal_csi_power_allocator"
            and str(item.get("runtime_entrypoint") or "") == "power_policy"
        ),
        None,
    )
    if compatible is None:
        raise ValueError(
            "artifact is not compatible with model.causal_csi_power_allocator"
        )
    components = manifest.get("components") or []
    component = next(
        (
            item
            for item in components
            if isinstance(item, Mapping)
            and str(item.get("id") or "") == "policy"
        ),
        None,
    )
    if not isinstance(component, Mapping):
        raise ValueError("artifact is missing policy component")
    component_path = path.parent / str(component.get("path") or "")
    expected = _required_sha256(
        component.get("sha256"),
        "policy component sha256",
    )
    if not component_path.is_file() or _sha256(component_path) != expected:
        raise ValueError("policy component failed SHA-256 verification")
    try:
        inspected = inspect_trained_artifact(
            path,
            project_root=project_root,
            registry=build_registry(),
        )
    except Exception as exc:
        raise ValueError(
            "trained artifact runtime validation failed: %s" % exc
        ) from exc
    if inspected.get("issues") or not inspected.get("ready"):
        raise ValueError(
            "trained artifact is not runtime-ready: %s"
            % "; ".join(str(item) for item in inspected.get("issues") or [])
        )
    runtime_identity = _required_sha256(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256"),
        "trained artifact runtime identity",
    )
    return {
        "manifest_sha256": _sha256(path),
        "component_sha256": expected,
        "runtime_identity_sha256": runtime_identity,
    }


def _validate_evaluation_binding(
    payload: Any,
    *,
    artifact_path: Path,
    component_sha256: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation metrics must be a JSON object")
    artifact = payload.get("trained_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("evaluation lacks trained-artifact binding")
    if str(artifact.get("manifest_sha256") or "") != _sha256(artifact_path):
        raise ValueError(
            "evaluation was produced for a different trained artifact"
        )
    components = artifact.get("components") or []
    selected = next(
        (
            item
            for item in components
            if isinstance(item, Mapping)
            and str(item.get("id") or "") == "policy"
        ),
        None,
    )
    if (
        not isinstance(selected, Mapping)
        or str(selected.get("sha256") or "") != component_sha256
    ):
        raise ValueError(
            "evaluation policy component does not match trained artifact"
        )
    integrity = payload.get("split_integrity")
    if not isinstance(integrity, Mapping) or integrity.get("status") != "passed":
        raise ValueError("evaluation lacks passing held-out split evidence")
    demo_evidence = payload.get("demo_evidence_status")
    if (
        not isinstance(demo_evidence, Mapping)
        or str(demo_evidence.get("status") or "") != "passed"
    ):
        raise ValueError(
            "held-out evaluation lacks sufficient evidence to present the "
            "learned policy as a benchmark candidate"
        )


def _assert_seed_namespace_is_held_out(
    recipe_path: Path,
    benchmark_seeds: list[int],
) -> None:
    reserved = set()
    for directory in (recipe_path.parent, recipe_path.parent.parent):
        for split in ("train", "validation", "test"):
            path = directory / ("capture_%s_recipe.yaml" % split)
            if not path.is_file():
                continue
            payload = _load_mapping(path, "%s capture recipe" % split)
            reserved.add(int((payload.get("metadata") or {}).get("seed") or 0))
            for step in payload.get("steps") or []:
                if not isinstance(step, Mapping):
                    continue
                seed = (step.get("params") or {}).get("seed")
                if seed is not None:
                    reserved.add(int(seed))
    proposed = set()
    for seed in benchmark_seeds:
        proposed.update(
            {
                int(seed),
                int(seed) + 100_000,
                int(seed) + 200_000,
                int(seed) + 300_000,
                int(seed) + 500_000,
            }
        )
    overlap = sorted(reserved.intersection(proposed))
    if overlap:
        raise ValueError(
            "benchmark operation seeds overlap generated capture seeds: %s"
            % ", ".join(str(value) for value in overlap)
        )


def _artifact_path(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    value = str(
        (config.get("training") or {}).get("artifact_manifest_path") or ""
    ).strip()
    if not value:
        raise ValueError(
            "train_config.yaml does not declare artifact_manifest_path"
        )
    return Path(value).expanduser().resolve()


def _project_root(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    value = str(config.get("project_root") or "").strip()
    if not value:
        raise ValueError("train_config.yaml does not declare project_root")
    return Path(value).expanduser().resolve()


def _require_step(
    recipe: Mapping[str, Any],
    step_id: str,
    operation: str,
) -> None:
    step = next(
        (
            item
            for item in recipe.get("steps") or []
            if isinstance(item, Mapping)
            and str(item.get("id") or "") == step_id
        ),
        None,
    )
    if step is None or str(step.get("op") or "") != operation:
        raise ValueError(
            "canonical recipe requires step %s using %s"
            % (step_id, operation)
        )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a mapping" % label)
    return dict(payload)


def _load_json_evidence(path: Path, label: str) -> Any:
    payload = _load_mapping(path, label)
    if not payload:
        raise ValueError("%s must not be empty" % label)
    return payload


def _evidence_file_spec(
    path: Path,
    relative_to: Path,
) -> dict[str, str]:
    return {
        "path": _relative_path(path, relative_to),
        "sha256": _sha256(path),
    }


def _positive_floats(value: str, label: str) -> list[float]:
    numbers = [
        float(item.strip())
        for item in str(value).split(",")
        if item.strip()
    ]
    if not numbers or any(
        not math.isfinite(item) or item <= 0.0 for item in numbers
    ):
        raise ValueError("%s must contain positive finite numbers" % label)
    return list(dict.fromkeys(numbers))


def _nonnegative_integers(value: str, label: str) -> list[int]:
    numbers = [
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    ]
    if not numbers or any(item < 0 for item in numbers):
        raise ValueError("%s must contain nonnegative integers" % label)
    return list(dict.fromkeys(numbers))


def _required_sha256(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError("%s must be a lowercase SHA-256 digest" % label)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _number_id(value: float) -> str:
    return ("%.9g" % float(value)).replace("-", "m").replace(".", "p")


def _print_commands(output_path: Path, project_root: Path) -> None:
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate and run:")
    print("  cd %s" % project_root)
    print("  noema benchmark validate %s" % output_path)
    print("  noema benchmark run %s" % output_path)
    print("after the run prints RESULT_ID:")
    print("  noema benchmark verify RESULT_ID")
    print(
        "  noema benchmark publish RESULT_ID "
        "--slug reliability-aware-delayed-csi-ofdm-allocation "
        "--out docs/demo/experiments/"
        "reliability-aware-delayed-csi-ofdm-allocation"
    )


if __name__ == "__main__":
    raise SystemExit(main())
