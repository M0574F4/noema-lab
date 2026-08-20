from __future__ import annotations

"""Build a reproducible equal/learned/oracle campaign after training.

The generated pack intentionally points every run at the same canonical source
recipe.  Method, power-budget, and held-out seed changes are expressed only as
benchmark entry overrides, so topology and all other protocol choices remain
paired.
"""

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
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_BUDGETS = (0.5, 1.0, 2.0)
DEFAULT_HELD_OUT_SEEDS = (71001, 72001, 73001)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the post-training OFDM allocator benchmark pack."
    )
    parser.add_argument(
        "--artifact",
        help="Returned trained_artifact.yaml (defaults to train_config.yaml).",
    )
    parser.add_argument(
        "--recipe",
        default="noema_recipe.yaml",
        help="Canonical scenario recipe copied into this training bundle.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_pack.yaml",
        help="Generated benchmark-pack path.",
    )
    parser.add_argument(
        "--project-root",
        help="Noema project root used to resolve runtime artifact paths (defaults to train_config.yaml).",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="Comma-separated average per-subcarrier power budgets.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_HELD_OUT_SEEDS),
        help="Comma-separated held-out master seeds paired across methods.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = Path(args.recipe).expanduser().resolve()
    artifact_path = _artifact_path(args.artifact)
    project_root = _project_root(args.project_root)
    budgets = _positive_floats(args.budgets, "budgets")
    seeds = _integer_list(args.seeds, "seeds")

    recipe = _load_mapping(recipe_path, "source recipe")
    _require_step(recipe, "tx_power", "model.symbol_power_allocator")
    _require_step(recipe, "channel_state", "wireless.ofdm_channel_state")
    _require_step(recipe, "wireless_channel", "wireless.channel")
    _assert_held_out_operation_seeds(
        recipe_path,
        seeds,
        {"data": 0, "channel_state": 100_000, "wireless_channel": 200_000},
    )
    artifact_evidence = _validate_artifact(
        artifact_path,
        "model.symbol_power_allocator",
        "power_policy",
        project_root=project_root,
    )

    history_path = Path("training_history.json").resolve()
    evaluation_path = Path("evaluation_metrics.json").resolve()
    _load_json_evidence(history_path, "training history")
    evaluation = _load_json_evidence(evaluation_path, "evaluation metrics")
    _validate_evaluation_binding(
        evaluation,
        artifact_evidence=artifact_evidence,
        config_path=Path("train_config.yaml").resolve(),
    )

    recipe_reference = _relative_path(recipe_path, output_path.parent)
    artifact_reference = _relative_path(artifact_path, project_root)
    training_artifact_reference = _evidence_file_spec(
        artifact_path, output_path.parent
    )
    history_reference = _evidence_file_spec(history_path, output_path.parent)
    evaluation_reference = _evidence_file_spec(evaluation_path, output_path.parent)
    pack = _build_pack(
        recipe_reference=recipe_reference,
        artifact_reference=artifact_reference,
        artifact_package_sha256=artifact_evidence[
            "runtime_identity_sha256"
        ],
        training_artifact_reference=training_artifact_reference,
        training_history=history_reference,
        evaluation_metrics=evaluation_reference,
        budgets=budgets,
        seeds=seeds,
    )
    output_path.write_text(
        yaml.safe_dump(pack, sort_keys=False), encoding="utf-8"
    )
    _print_commands(output_path, project_root)
    return 0


def _build_pack(
    *,
    recipe_reference: str,
    artifact_reference: str,
    artifact_package_sha256: str,
    training_artifact_reference: Mapping[str, str],
    training_history: Mapping[str, str],
    evaluation_metrics: Mapping[str, str],
    budgets: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    methods = (
        ("equal_power", "Equal power", "baseline", {"policy": "fixed"}),
        (
            "learned_allocator",
            "Learned allocator",
            "candidate",
            {
                "policy": "learned_artifact",
                "artifact_manifest_path": artifact_reference,
                "artifact_entrypoint": "power_policy",
                "artifact_package_sha256": artifact_package_sha256,
            },
        ),
        (
            "water_filling",
            "Water filling (Shannon oracle)",
            "oracle",
            {"policy": "water_filling"},
        ),
    )
    for budget in budgets:
        for paired_seed in seeds:
            for method, label, role, allocator_params in methods:
                identifier = "%s_p%s_seed%d" % (
                    method,
                    _number_id(budget),
                    paired_seed,
                )
                tx_power = dict(allocator_params)
                tx_power["target_power"] = float(budget)
                recipes.append(
                    {
                        "id": identifier,
                        "label": "%s · P=%g · seed %d"
                        % (label, budget, paired_seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method,
                            "matrix_selection": {
                                "resource.average_transmit_power_budget": float(
                                    budget
                                ),
                                "benchmark.paired_seed": int(paired_seed),
                            },
                            "metadata": {
                                "benchmark_method": method,
                                "benchmark_paired_seed": int(paired_seed),
                                "benchmark_held_out": True,
                            },
                            "step_params": {
                                "data": {"seed": int(paired_seed)},
                                "channel_state": {
                                    "seed": int(paired_seed) + 100_000,
                                    "average_power_budget": float(budget),
                                },
                                "tx_power": tx_power,
                                "wireless_channel": {
                                    "seed": int(paired_seed) + 200_000,
                                    "channel": "ofdm_tdl",
                                    "wireless_backend": "sionna",
                                    "channel_state_mode": "explicit",
                                    "receiver_processing": "matched",
                                },
                            },
                        },
                    }
                )

    demo = {
        "schema_version": 1,
        "slug": "learned-ofdm-resource-allocation",
        "title": "Learning OFDM power allocation without oracle labels",
        "summary": (
            "A permutation-equivariant learned allocator is compared with equal "
            "power and the exact water-filling reference on paired held-out TDL-A "
            "channel realizations."
        ),
        "question": (
            "Can a label-free learned policy recover water-filling spectral "
            "efficiency while satisfying the power constraint exactly?"
        ),
        "tutorial": "../../../tutorials/ofdm_resource_allocation_demo.html",
        "held_constant": [
            "canonical OFDM recipe topology",
            "seeded random payload bits",
            "Sionna TDL-A channel and fixed noise variance",
            "paired payload, channel, and noise seeds across methods",
        ],
        "changed": [
            "allocator policy",
            "average transmit-power budget",
            "held-out paired seed",
        ],
        "primary_metric": (
            "resource.theoretical_shannon_spectral_efficiency_bps_hz"
        ),
        "comparison_axis": "resource.average_transmit_power_budget",
        "series": [
            {"id": "equal_power", "label": "Equal power", "role": "baseline"},
            {
                "id": "learned_allocator",
                "label": "Learned allocator",
                "role": "candidate",
            },
            {
                "id": "water_filling",
                "label": "Water filling (Shannon oracle)",
                "role": "oracle",
            },
        ],
        "plots": [
            {
                "id": "spectral-efficiency-vs-power",
                "title": "Spectral efficiency vs power budget",
                "kind": "line",
                "x": "resource.average_transmit_power_budget",
                "y": (
                    "resource.theoretical_shannon_spectral_efficiency_bps_hz"
                ),
                "group": "benchmark_method",
                "method_order": [
                    "equal_power",
                    "learned_allocator",
                    "water_filling",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "shannon-gap-vs-power",
                "title": "Relative gap to the water-filling Shannon optimum",
                "kind": "line",
                "x": "resource.average_transmit_power_budget",
                "y": "resource.water_filling_relative_optimality_gap",
                "group": "benchmark_method",
                "method_order": [
                    "equal_power",
                    "learned_allocator",
                    "water_filling",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
        ],
        "table_metrics": [
            "resource.theoretical_shannon_spectral_efficiency_bps_hz",
            "resource.water_filling_relative_optimality_gap",
            "resource.water_filling_kkt_normalized_residual",
            "resource.power_constraint.max_abs_error",
            "resource.power_constraint.max_negative_violation",
        ],
        "training_evidence": [
            {
                "series": "learned_allocator",
                "trained_artifact_manifest": training_artifact_reference,
                "training_history": training_history,
                "evaluation_metrics": evaluation_metrics,
            }
        ],
    }
    return {
        "schema_version": 1,
        "id": "resource_allocation.learned_allocator_post_training_v2",
        "version": "2.0.0",
        "name": "Learned OFDM Power Allocation",
        "description": (
            "Equal-power, learned, and exact water-filling policies derived from "
            "one canonical OFDM recipe across paired held-out seeds and budgets."
        ),
        "suite": {
            "id": "resource_allocation",
            "name": "Resource Allocation",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "synthetic_random_bits_sionna_tdl",
            "modality": "wireless",
            "version": "synthetic-random-bits-sionna-tdl-v1",
            "split": "held_out_seeded",
        },
        "task": {
            "id": "resource_allocation",
            "kind": "policy_optimization",
            "modality": "wireless",
        },
        "metrics": [
            {
                "id": "resource.average_transmit_power_budget",
                "family": "system",
                "unit": "normalized_power",
                "direction": "neutral",
            },
            {
                "id": "resource.theoretical_shannon_spectral_efficiency_bps_hz",
                "family": "system",
                "unit": "bit/s/Hz",
                "direction": "higher_is_better",
            },
            {
                "id": "resource.water_filling_relative_optimality_gap",
                "family": "system",
                "unit": "ratio",
                "direction": "lower_is_better",
            },
            {
                "id": "resource.water_filling_kkt_normalized_residual",
                "family": "system",
                "unit": "ratio",
                "direction": "lower_is_better",
            },
            {
                "id": "resource.power_constraint.max_abs_error",
                "family": "system",
                "unit": "normalized_power",
                "direction": "lower_is_better",
            },
            {
                "id": "resource.power_constraint.max_negative_violation",
                "family": "system",
                "unit": "normalized_power",
                "direction": "lower_is_better",
            },
            {
                "id": "channel.achieved_payload_goodput_bits_per_resource_element",
                "family": "channel",
                "unit": "bit/resource-element",
                "direction": "higher_is_better",
            },
            {
                "id": "channel.payload.ber",
                "family": "transport",
                "unit": "ratio",
                "direction": "lower_is_better",
            },
            {
                "id": "channel.payload.bler",
                "family": "transport",
                "unit": "ratio",
                "direction": "lower_is_better",
            },
        ],
        "baselines": ["equal_power", "water_filling"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "learned-ofdm-allocation-post-training-v2",
            "paired_held_out_seeds": seeds,
            "average_power_budgets": budgets,
            "demo": demo,
        },
    }


def _artifact_path(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    configured = str(
        (config.get("training") or {}).get("artifact_manifest_path") or ""
    ).strip()
    if not configured:
        raise ValueError(
            "train_config.yaml does not declare training.artifact_manifest_path"
        )
    return Path(configured).expanduser().resolve()


def _project_root(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    configured = str(config.get("project_root") or "").strip()
    if not configured:
        raise ValueError(
            "train_config.yaml does not declare project_root; pass --project-root"
        )
    return Path(configured).expanduser().resolve()


def _validate_artifact(
    path: Path,
    operation: str,
    entrypoint: str,
    *,
    project_root: Path | None = None,
) -> dict[str, str]:
    manifest = _load_mapping(path, "trained artifact manifest")
    if int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    if str(manifest.get("kind") or "") != "noema.trained_block_artifact":
        raise ValueError("artifact kind must be noema.trained_block_artifact")
    _required_text(manifest, "id", "artifact id")
    _required_text(manifest, "name", "artifact name")

    package_root = path.resolve().parent
    contract = _required_mapping(manifest, "contract", "artifact contract")
    contract_id = _required_text(contract, "id", "artifact contract id")
    contract_version = _positive_integer(
        contract.get("version"), "artifact contract version"
    )
    contract_path = _confined_package_file(
        package_root,
        _required_text(contract, "path", "artifact contract path"),
        "artifact contract",
    )
    expected_contract_file_sha = _required_sha256(
        contract.get("file_sha256"), "artifact contract file_sha256"
    )
    if _file_sha256(contract_path) != expected_contract_file_sha:
        raise ValueError(
            "artifact contract file SHA-256 does not match contract.file_sha256"
        )
    contract_payload = _load_mapping(contract_path, "artifact contract")
    if str(contract_payload.get("kind") or "") != "noema.trainable_slot_contract@1":
        raise ValueError(
            "artifact contract kind must be noema.trainable_slot_contract@1"
        )
    if str(contract_payload.get("id") or "") != contract_id:
        raise ValueError("artifact contract id does not match contract.id")
    if _positive_integer(
        contract_payload.get("version"), "artifact contract file version"
    ) != contract_version:
        raise ValueError("artifact contract version does not match contract.version")
    expected_contract_sha = _required_sha256(
        contract.get("sha256"), "artifact contract sha256"
    )
    if _canonical_sha256(contract_payload) != expected_contract_sha:
        raise ValueError(
            "artifact contract semantic SHA-256 does not match contract.sha256"
        )
    _require_contract_operation(contract_payload, operation)

    raw_components = manifest.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("artifact components must be a non-empty list")
    components: dict[str, dict[str, Any]] = {}
    component_paths: set[Path] = set()
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, Mapping):
            raise ValueError("artifact components[%d] must be a mapping" % index)
        component = dict(raw_component)
        component_id = _required_text(
            component, "id", "artifact components[%d].id" % index
        )
        if component_id in components:
            raise ValueError("artifact component ids must be unique")
        component_format = _required_text(
            component, "format", "artifact components[%d].format" % index
        ).lower()
        component_path = _confined_package_file(
            package_root,
            _required_text(
                component, "path", "artifact components[%d].path" % index
            ),
            "artifact component %s" % component_id,
        )
        if component_path == contract_path or component_path in component_paths:
            raise ValueError("artifact component paths must be unique")
        component_paths.add(component_path)
        expected_sha = _required_sha256(
            component.get("sha256"),
            "artifact components[%d].sha256" % index,
        )
        if _file_sha256(component_path) != expected_sha:
            raise ValueError(
                "artifact component %s SHA-256 does not match the manifest"
                % component_id
            )
        components[component_id] = {
            **component,
            "format": component_format,
            "resolved_path": component_path,
        }

    runtime = _required_mapping(manifest, "runtime", "artifact runtime")
    if str(runtime.get("backend") or "").strip().lower() != "onnxruntime":
        raise ValueError("artifact runtime backend must be onnxruntime")
    abi_version = _positive_integer(
        runtime.get("abi_version"), "artifact runtime abi_version"
    )
    if abi_version != 1:
        raise ValueError(
            "artifact runtime abi_version %d is unsupported by this campaign; "
            "expected 1" % abi_version
        )
    raw_entrypoints = runtime.get("entrypoints")
    if not isinstance(raw_entrypoints, list) or not raw_entrypoints:
        raise ValueError("artifact runtime entrypoints must be a non-empty list")
    entrypoints: dict[str, dict[str, Any]] = {}
    for index, raw_runtime_entrypoint in enumerate(raw_entrypoints):
        if not isinstance(raw_runtime_entrypoint, Mapping):
            raise ValueError(
                "artifact runtime entrypoints[%d] must be a mapping" % index
            )
        runtime_entrypoint = dict(raw_runtime_entrypoint)
        runtime_id = _required_text(
            runtime_entrypoint,
            "id",
            "artifact runtime entrypoints[%d].id" % index,
        )
        if runtime_id in entrypoints:
            raise ValueError("artifact runtime entrypoint ids must be unique")
        component_id = _required_text(
            runtime_entrypoint,
            "component",
            "artifact runtime entrypoints[%d].component" % index,
        )
        component = components.get(component_id)
        if component is None:
            raise ValueError(
                "artifact runtime entrypoint %s references an unknown component"
                % runtime_id
            )
        if str(component.get("format") or "") != "onnx":
            raise ValueError(
                "artifact runtime entrypoint %s requires an ONNX component"
                % runtime_id
            )
        _validate_tensor_list(
            runtime_entrypoint.get("inputs"),
            "artifact runtime entrypoint %s inputs" % runtime_id,
        )
        _validate_tensor_list(
            runtime_entrypoint.get("outputs"),
            "artifact runtime entrypoint %s outputs" % runtime_id,
        )
        entrypoints[runtime_id] = runtime_entrypoint
    if entrypoint not in entrypoints:
        raise ValueError(
            "artifact runtime does not declare expected entrypoint %s" % entrypoint
        )

    raw_bindings = manifest.get("compatible_operations")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("artifact compatible_operations must be a non-empty list")
    if any(not isinstance(item, Mapping) for item in raw_bindings):
        raise ValueError("artifact compatible_operations entries must be mappings")
    bindings = [dict(item) for item in raw_bindings]
    for index, binding in enumerate(bindings):
        _required_text(
            binding,
            "operation",
            "artifact compatible_operations[%d].operation" % index,
        )
        binding_entrypoint = _required_text(
            binding,
            "runtime_entrypoint",
            "artifact compatible_operations[%d].runtime_entrypoint" % index,
        )
        if binding_entrypoint not in entrypoints:
            raise ValueError(
                "artifact compatible_operations[%d] references an unknown runtime entrypoint"
                % index
            )
    compatible = next(
        (item for item in bindings if str(item.get("operation") or "") == operation),
        None,
    )
    if compatible is None:
        raise ValueError("trained artifact is not compatible with %s" % operation)
    actual_entrypoint = str(compatible.get("runtime_entrypoint") or "")
    if actual_entrypoint != entrypoint:
        raise ValueError(
            "trained artifact binding must use runtime entrypoint %s" % entrypoint
        )
    inspected = inspect_trained_artifact(
        path,
        project_root=(project_root or path.parent).resolve(),
        registry=build_registry(),
    )
    runtime_identity = str(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256")
        or ""
    )
    if re.fullmatch(r"[0-9a-f]{64}", runtime_identity) is None:
        raise ValueError("trained artifact runtime identity is missing")
    selected_component_id = str(entrypoints[entrypoint]["component"])
    selected_component = components[selected_component_id]
    return {
        "manifest_sha256": _file_sha256(path),
        "component_id": selected_component_id,
        "component_sha256": _required_sha256(
            selected_component.get("sha256"), "selected artifact component sha256"
        ),
        "runtime_identity_sha256": runtime_identity,
    }


def _required_mapping(
    payload: Mapping[str, Any], key: str, label: str
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % label)
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError("%s must be non-empty" % label)
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be a positive integer" % label)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a positive integer" % label) from exc
    if result <= 0 or result != value:
        raise ValueError("%s must be a positive integer" % label)
    return result


def _required_sha256(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError("%s must be a lowercase SHA-256 digest" % label)
    return result


def _confined_package_file(package_root: Path, value: str, label: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError("%s path must be relative to the artifact package" % label)
    resolved = (package_root / raw_path).resolve()
    try:
        resolved.relative_to(package_root.resolve())
    except ValueError as exc:
        raise ValueError("%s path escapes the artifact package" % label) from exc
    if not resolved.is_file():
        raise FileNotFoundError("%s file does not exist: %s" % (label, resolved))
    if resolved.stat().st_size <= 0:
        raise ValueError("%s file must be non-empty" % label)
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("artifact contract must be finite and JSON-compatible") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_tensor_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("%s must be a non-empty list" % label)
    for index, tensor in enumerate(value):
        if not isinstance(tensor, Mapping):
            raise ValueError("%s[%d] must be a mapping" % (label, index))
        for key in ("name", "dtype", "shape"):
            if key == "shape":
                if not isinstance(tensor.get(key), list) or not tensor.get(key):
                    raise ValueError("%s[%d].shape must be a non-empty list" % (label, index))
            elif not str(tensor.get(key) or "").strip():
                raise ValueError("%s[%d].%s must be non-empty" % (label, index, key))


def _require_contract_operation(
    contract_payload: Mapping[str, Any], operation: str
) -> None:
    artifact_return = contract_payload.get("artifact_return")
    bindings = (
        artifact_return.get("bindings")
        if isinstance(artifact_return, Mapping)
        else None
    )
    if not isinstance(bindings, list) or not any(
        isinstance(binding, Mapping)
        and str(binding.get("operation") or "") == operation
        for binding in bindings
    ):
        raise ValueError(
            "artifact contract does not declare an artifact-return binding for %s"
            % operation
        )


def _load_json_evidence(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    try:
        payload = load_strict_yaml_or_json(path)
    except (OSError, ValueError) as exc:
        raise ValueError("%s must contain valid JSON: %s" % (label, path)) from exc
    if not isinstance(payload, (list, dict)) or not payload:
        raise ValueError("%s must contain a non-empty JSON list or object" % label)
    return payload


def _validate_evaluation_binding(
    payload: Any,
    *,
    artifact_evidence: Mapping[str, str],
    config_path: Path,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "evaluation metrics must be a JSON object with artifact and test-capture binding fields"
        )
    trained_artifact = payload.get("trained_artifact")
    if not isinstance(trained_artifact, Mapping):
        raise ValueError(
            "evaluation metrics is missing trained_artifact binding evidence"
        )
    manifest_sha = _required_sha256(
        trained_artifact.get("manifest_sha256"),
        "evaluation metrics trained_artifact.manifest_sha256",
    )
    if manifest_sha != artifact_evidence["manifest_sha256"]:
        raise ValueError(
            "evaluation metrics was produced for a different trained artifact manifest"
        )
    components = trained_artifact.get("components")
    if not isinstance(components, list):
        raise ValueError(
            "evaluation metrics trained_artifact.components must be a list"
        )
    selected = next(
        (
            component
            for component in components
            if isinstance(component, Mapping)
            and str(component.get("id") or "") == artifact_evidence["component_id"]
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        raise ValueError(
            "evaluation metrics is missing the selected artifact component binding"
        )
    component_sha = _required_sha256(
        selected.get("sha256"),
        "evaluation metrics selected component sha256",
    )
    if component_sha != artifact_evidence["component_sha256"]:
        raise ValueError(
            "evaluation metrics component SHA-256 does not match the selected artifact component"
        )
    _validate_test_capture_binding(payload, config_path)


def _validate_test_capture_binding(
    payload: Mapping[str, Any], config_path: Path
) -> None:
    raw_hashes = payload.get("test_capture_schema_sha256")
    if not isinstance(raw_hashes, list) or not raw_hashes:
        raise ValueError(
            "evaluation metrics requires non-empty test_capture_schema_sha256 binding evidence"
        )
    observed = [
        _required_sha256(value, "evaluation metrics test capture schema SHA-256")
        for value in raw_hashes
    ]
    expected = _configured_test_capture_schema_sha256(config_path)
    if expected is not None and observed != expected:
        raise ValueError(
            "evaluation metrics test capture schema SHA-256 does not match the configured held-out capture"
        )


def _configured_test_capture_schema_sha256(
    config_path: Path,
) -> list[str] | None:
    if not config_path.is_file():
        return None
    config = _load_mapping(config_path, "train_config.yaml")
    raw_directories = (config.get("data") or {}).get("test_capture_dirs") or []
    if not isinstance(raw_directories, list) or not raw_directories:
        return None
    schema_paths = []
    for value in raw_directories:
        directory = Path(str(value)).expanduser()
        if not directory.is_absolute():
            directory = config_path.parent / directory
        schema_paths.append(directory.resolve() / "schema.json")
    present = [path.is_file() for path in schema_paths]
    if not any(present):
        return None
    if not all(present):
        missing = [str(path) for path, exists in zip(schema_paths, present) if not exists]
        raise FileNotFoundError(
            "cannot verify held-out capture binding; missing configured schema(s): %s"
            % ", ".join(missing)
        )
    return [_file_sha256(path) for path in schema_paths]


def _evidence_file_spec(path: Path, relative_to: Path) -> dict[str, str]:
    return {
        "path": _relative_path(path, relative_to),
        "sha256": _file_sha256(path),
    }


def _require_step(recipe: Mapping[str, Any], step_id: str, operation: str) -> None:
    step = next(
        (
            item
            for item in recipe.get("steps") or []
            if isinstance(item, Mapping) and str(item.get("id") or "") == step_id
        ),
        None,
    )
    if step is None or str(step.get("op") or "") != operation:
        raise ValueError(
            "canonical recipe requires step %s using %s" % (step_id, operation)
        )


def _assert_held_out_operation_seeds(
    recipe_path: Path,
    paired_seeds: list[int],
    step_offsets: Mapping[str, int],
) -> None:
    """Reject benchmark RNG streams already used by generated capture splits."""

    captures: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in (recipe_path.parent, recipe_path.parent.parent):
        for split in ("train", "validation", "test"):
            path = directory / ("capture_%s_recipe.yaml" % split)
            if path.is_file() and split not in captures:
                captures[split] = (path, _load_mapping(path, "%s capture recipe" % split))
    missing = [split for split in ("train", "validation", "test") if split not in captures]
    if missing:
        raise ValueError(
            "cannot prove benchmark seeds are held out; missing generated capture recipe(s): %s"
            % ", ".join(missing)
        )

    modulus = 2**31 - 1
    reserved: dict[str, dict[int, str]] = {step_id: {} for step_id in step_offsets}
    for split, (path, capture_recipe) in captures.items():
        capture = dict(capture_recipe.get("dataset_capture") or {})
        mode = str(capture.get("seed_mode") or "fixed_seed")
        run_count = int(capture.get("max_runs") or 1) if mode == "increment_run_seed" else 1
        base_seed = int((capture_recipe.get("metadata") or {}).get("seed") or 0)
        steps = {
            str(item.get("id") or ""): item
            for item in capture_recipe.get("steps") or []
            if isinstance(item, Mapping)
        }
        for step_id in step_offsets:
            step = steps.get(step_id)
            if step is None:
                raise ValueError("%s is missing required step %s" % (path, step_id))
            params = dict(step.get("params") or {})
            for run_index in range(max(1, run_count)):
                master_seed = (base_seed + run_index) % modulus
                master_seed = int(master_seed or modulus)
                if params.get("seed") is not None:
                    operation_seed = int(params["seed"])
                else:
                    payload = "%d|%s|%s|default" % (
                        master_seed,
                        str(capture_recipe.get("name") or ""),
                        step_id,
                    )
                    operation_seed = int(
                        int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
                        % modulus
                    )
                reserved[step_id][operation_seed] = split

    overlaps = []
    for paired_seed in paired_seeds:
        for step_id, offset in step_offsets.items():
            operation_seed = int(paired_seed) + int(offset)
            if operation_seed in reserved[step_id]:
                overlaps.append(
                    "%d (%s, %s capture)"
                    % (paired_seed, step_id, reserved[step_id][operation_seed])
                )
    if overlaps:
        raise ValueError(
            "benchmark seeds are not held out from dataset capture: %s"
            % ", ".join(overlaps)
        )


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    payload = load_strict_yaml_or_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("%s must contain a YAML mapping" % label)
    return dict(payload)


def _positive_floats(value: str, label: str) -> list[float]:
    numbers = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not numbers or any(not math.isfinite(item) or item <= 0 for item in numbers):
        raise ValueError("%s must be a non-empty list of positive finite numbers" % label)
    return list(dict.fromkeys(numbers))


def _integer_list(value: str, label: str) -> list[int]:
    numbers = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not numbers or any(item < 0 for item in numbers):
        raise ValueError("%s must be a non-empty list of nonnegative integers" % label)
    return list(dict.fromkeys(numbers))


def _number_id(value: float) -> str:
    return ("%.9g" % float(value)).replace("-", "m").replace(".", "p")


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _print_commands(output_path: Path, project_root: Path) -> None:
    print("wrote benchmark pack: %s" % output_path)
    print("from the Noema project root, validate:")
    print("  cd %s" % project_root)
    print("  noema benchmark validate %s" % output_path)
    print("from the Noema project root, run:")
    print("  cd %s" % project_root)
    print("  noema benchmark run %s" % output_path)
    print("after the run prints RESULT_ID, verify and publish:")
    print("  noema benchmark verify RESULT_ID")
    print(
        "  noema benchmark publish RESULT_ID --slug learned-ofdm-resource-allocation "
        "--out docs/demo/experiments/learned-ofdm-resource-allocation"
    )


if __name__ == "__main__":
    raise SystemExit(main())
