from __future__ import annotations

"""Build a paired blind-baseline/learned/oracle modulation-recognition campaign."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from noema_lab.core.reproducibility import derive_seed
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


DEFAULT_SNR_DB = (-2.0, 2.0, 6.0, 10.0, 14.0, 18.0)
DEFAULT_HELD_OUT_SEEDS = (71001, 72001, 73001)
DEFAULT_FRAMES_PER_RUN = 1536


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the post-training automatic modulation-recognition benchmark pack."
    )
    parser.add_argument("--artifact", help="Returned trained_artifact.yaml (defaults to train_config.yaml).")
    parser.add_argument("--recipe", default="noema_recipe.yaml", help="Canonical AMC scenario copied into this bundle.")
    parser.add_argument("--output", default="benchmark_pack.yaml", help="Generated benchmark-pack path.")
    parser.add_argument(
        "--project-root",
        help="Noema project root used to resolve runtime artifact paths (defaults to train_config.yaml).",
    )
    parser.add_argument(
        "--snr-db",
        default=",".join(str(value) for value in DEFAULT_SNR_DB),
        help="Comma-separated evaluation SNR grid in dB; values must belong to the canonical recipe matrix.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_HELD_OUT_SEEDS),
        help="Comma-separated held-out master seeds paired across classifiers.",
    )
    parser.add_argument(
        "--frames-per-run",
        type=int,
        default=DEFAULT_FRAMES_PER_RUN,
        help="Balanced evaluation frames per run (must be divisible by three).",
    )
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = Path(args.recipe).expanduser().resolve()
    artifact_path = _artifact_path(args.artifact)
    project_root = _project_root(args.project_root)
    snr_grid = _finite_floats(args.snr_db, "snr-db")
    seeds = _integer_list(args.seeds, "seeds")
    frames_per_run = int(args.frames_per_run)
    if frames_per_run < 3 or frames_per_run % 3:
        raise ValueError("frames-per-run must be a positive multiple of three")
    recipe = _load_mapping(recipe_path, "source recipe")
    _require_step(recipe, "data", "source.modulation_frames")
    _require_step(recipe, "observation", "wireless.modulation_awgn_observation")
    _require_step(recipe, "receiver", "model.modulation_classifier_adapter")
    _require_step(recipe, "evaluation", "metrics.modulation_classification")
    _assert_held_out_operation_seeds(
        recipe_path,
        seeds,
        {"data": 0, "observation": 100_000},
        {"data": "modulation_frames", "observation": "modulation_awgn"},
    )
    artifact_evidence = _validate_artifact(
        artifact_path,
        "model.modulation_classifier_adapter",
        "modulation_classifier",
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
    pack = _build_pack(
        recipe_reference=_relative_path(recipe_path, output_path.parent),
        artifact_reference=_relative_path(artifact_path, project_root),
        artifact_package_sha256=artifact_evidence["runtime_identity_sha256"],
        training_artifact_reference=_evidence_file_spec(
            artifact_path, output_path.parent
        ),
        training_history=_evidence_file_spec(history_path, output_path.parent),
        evaluation_metrics=_evidence_file_spec(evaluation_path, output_path.parent),
        snr_grid=snr_grid,
        seeds=seeds,
        frames_per_run=frames_per_run,
    )
    output_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
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
    snr_grid: list[float],
    seeds: list[int],
    frames_per_run: int,
) -> dict[str, Any]:
    recipes: list[dict[str, Any]] = []
    methods = (
        (
            "blind_cumulant",
            "Blind differential-cumulant classifier",
            "baseline",
            {"mode": "classical_cumulant"},
        ),
        (
            "learned_classifier",
            "Learned blind-carrier classifier",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_reference,
                "artifact_entrypoint": "modulation_classifier",
                "artifact_package_sha256": artifact_package_sha256,
            },
        ),
        (
            "oracle_likelihood",
            "Oracle-synchronized likelihood reference",
            "reference",
            {"mode": "classical_likelihood"},
        ),
    )
    for snr_db in snr_grid:
        for paired_seed in seeds:
            for method, label, role, classifier_params in methods:
                recipes.append(
                    {
                        "id": "%s_snr%s_seed%d" % (method, _number_id(snr_db), paired_seed),
                        "label": "%s · %g dB · seed %d" % (label, snr_db, paired_seed),
                        "role": role,
                        "path": recipe_reference,
                        "params": {
                            "method_id": method,
                            "matrix_selection": {
                                "channel.snr_db": float(snr_db),
                                "benchmark.paired_seed": int(paired_seed),
                            },
                            "metadata": {
                                "benchmark_method": method,
                                "benchmark_paired_seed": int(paired_seed),
                                "benchmark_held_out": True,
                                "pairing_id": str(paired_seed),
                                "aggregation_cell_id": "snr_db=%s"
                                % _number_id(snr_db),
                                "statistical_unit": (
                                    "paired held-out symbol/carrier/noise seed"
                                ),
                            },
                            "step_params": {
                                "data": {
                                    "seed": int(paired_seed),
                                    "frame_count": int(frames_per_run),
                                },
                                "observation": {"seed": int(paired_seed) + 100_000},
                                "receiver": dict(classifier_params),
                            },
                        },
                    }
                )
    demo = {
        "schema_version": 1,
        "slug": "learned-automatic-modulation-recognition",
        "title": "Blind automatic modulation recognition under carrier uncertainty",
        "summary": (
            "A blind differential-cumulant baseline and a returned learned "
            "I/Q-only classifier are compared under unknown per-frame carrier "
            "phase and residual frequency offset. An oracle-synchronized "
            "likelihood rule is retained as a diagnostic reference."
        ),
        "question": (
            "Can a researcher replace only the modulation-classifier block while preserving "
            "the waveform generator, carrier-impairment protocol, class vocabulary, "
            "and paired evaluation?"
        ),
        "tutorial": "../../../tutorials/automatic_modulation_recognition_demo.html",
        "held_constant": [
            "balanced fixed-length BPSK/QPSK/16-QAM source",
            "symbol-timed complex I/Q observation contract",
            "per-frame phase and frequency-offset distributions",
            "paired symbol, carrier-impairment, and noise seeds across methods",
            "class order and confusion-matrix evaluation",
        ],
        "changed": ["classifier implementation", "channel SNR", "held-out paired seed"],
        "primary_metric": "modulation_recognition.balanced_accuracy",
        "comparison_axis": "channel.snr_db",
        "series": [
            {
                "id": "blind_cumulant",
                "label": "Blind differential cumulant",
                "role": "baseline",
            },
            {
                "id": "learned_classifier",
                "label": "Learned blind-carrier classifier",
                "role": "candidate",
            },
            {
                "id": "oracle_likelihood",
                "label": "Oracle-synchronized likelihood",
                "role": "reference",
            },
        ],
        "plots": [
            {
                "id": "accuracy-vs-snr",
                "title": "Recognition accuracy vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "modulation_recognition.accuracy",
                "group": "benchmark_method",
                "method_order": [
                    "blind_cumulant",
                    "learned_classifier",
                    "oracle_likelihood",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
            {
                "id": "macro-f1-vs-snr",
                "title": "Macro F1 vs SNR",
                "kind": "line",
                "x": "channel.snr_db",
                "y": "modulation_recognition.macro_f1",
                "group": "benchmark_method",
                "method_order": [
                    "blind_cumulant",
                    "learned_classifier",
                    "oracle_likelihood",
                ],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            },
        ],
        "table_metrics": [
            "channel.snr_db",
            "modulation_recognition.accuracy",
            "modulation_recognition.balanced_accuracy",
            "modulation_recognition.macro_f1",
            "modulation_recognition.frame_count",
        ],
        "training_evidence": [
            {
                "series": "learned_classifier",
                "trained_artifact_manifest": training_artifact_reference,
                "training_history": training_history,
                "evaluation_metrics": evaluation_metrics,
            }
        ],
    }
    return {
        "schema_version": 1,
        "id": (
            "neural_receiver_ai_phy."
            "learned_modulation_recognition_blind_carrier_post_training_v1"
        ),
        "version": "1.0.0",
        "name": "Learned Automatic Modulation Recognition",
        "description": (
            "A blind cumulant baseline, learned I/Q-only classifier, and "
            "oracle-synchronized likelihood reference evaluated over a paired "
            "held-out carrier-uncertain AWGN SNR campaign."
        ),
        "suite": {
            "id": "neural_receiver",
            "name": "Neural Receiver / AI-PHY",
            "status": "experimental",
            "version": "v1-draft",
        },
        "dataset": {
            "id": "synthetic_modulation_iq",
            "modality": "wireless",
            "version": "synthetic-modulation-iq-blind-carrier-v1",
            "split": "held_out_seeded",
        },
        "task": {
            "id": "automatic_modulation_recognition",
            "kind": "signal_classification",
            "modality": "wireless",
        },
        "metrics": [
            {
                "id": "channel.snr_db",
                "definition_version": 1,
                "source_step": "observation",
                "source_operation": "wireless.modulation_awgn_observation",
                "family": "channel",
                "unit": "dB",
                "direction": "neutral",
            },
            {
                "id": "modulation_recognition.accuracy",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.modulation_classification",
                "family": "classification",
                "unit": "fraction",
                "direction": "higher_is_better",
            },
            {
                "id": "modulation_recognition.balanced_accuracy",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.modulation_classification",
                "family": "classification",
                "unit": "fraction",
                "direction": "higher_is_better",
            },
            {
                "id": "modulation_recognition.macro_f1",
                "definition_version": 1,
                "source_step": "evaluation",
                "source_operation": "metrics.modulation_classification",
                "family": "classification",
                "unit": "fraction",
                "direction": "higher_is_better",
            },
            {
                "id": "modulation_recognition.frame_count",
                "definition_version": 1,
                "source_step": "data",
                "source_operation": "source.modulation_frames",
                "family": "classification",
                "unit": "frames",
                "direction": "neutral",
            },
        ],
        "baselines": ["blind_cumulant"],
        "recipes": recipes,
        "metadata": {
            "benchmark_tier": "experimental",
            "protocol_revision": "learned-amc-blind-carrier-post-training-v1",
            "paired_held_out_seeds": seeds,
            "snr_db_grid": snr_grid,
            "evaluation_frames_per_run": int(frames_per_run),
            "statistical_unit": (
                "paired held-out symbol/carrier/noise seed"
            ),
            "demo": demo,
        },
    }


def _artifact_path(argument: str | None) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    config = _load_mapping(Path("train_config.yaml"), "train_config.yaml")
    configured = str((config.get("training") or {}).get("artifact_manifest_path") or "").strip()
    if not configured:
        raise ValueError("train_config.yaml does not declare training.artifact_manifest_path")
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
    project_root: Path,
) -> dict[str, Any]:
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
    if str(compatible.get("runtime_entrypoint") or "") != entrypoint:
        raise ValueError("trained artifact binding must use runtime entrypoint %s" % entrypoint)
    selected_component_id = str(entrypoints[entrypoint]["component"])
    selected_component = components[selected_component_id]
    training = _required_mapping(
        manifest, "training", "artifact training provenance"
    )
    training_record_fingerprints = _validate_split_record_fingerprints(
        training.get("split_record_fingerprints"),
        required_splits=("train", "validation"),
        label="artifact training split_record_fingerprints",
    )
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
    runtime_identity = _required_sha256(
        inspected.get("runtime_identity_sha256")
        or inspected.get("package_sha256"),
        "trained artifact runtime identity",
    )
    return {
        "manifest_sha256": _file_sha256(path),
        "component_id": selected_component_id,
        "component_sha256": _required_sha256(
            selected_component.get("sha256"), "selected artifact component sha256"
        ),
        "training_record_fingerprints": training_record_fingerprints,
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
                    raise ValueError(
                        "%s[%d].shape must be a non-empty list" % (label, index)
                    )
            elif not str(tensor.get(key) or "").strip():
                raise ValueError(
                    "%s[%d].%s must be non-empty" % (label, index, key)
                )


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
    artifact_evidence: Mapping[str, Any],
    config_path: Path,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "evaluation metrics must be a JSON object with artifact and test-capture binding fields"
        )
    if not payload.get("component_sha256"):
        raise ValueError(
            "evaluation metrics is missing component_sha256 binding evidence"
        )
    component_sha = _required_sha256(
        payload.get("component_sha256"),
        "evaluation metrics component_sha256",
    )
    if component_sha != artifact_evidence["component_sha256"]:
        raise ValueError(
            "evaluation metrics component SHA-256 does not match the selected artifact component"
        )
    evaluation_records = _validate_split_record_fingerprints(
        payload.get("split_record_fingerprints"),
        required_splits=("train", "validation", "test"),
        label="evaluation metrics split_record_fingerprints",
    )
    artifact_records = artifact_evidence.get("training_record_fingerprints") or {}
    for name in ("train", "validation"):
        if tuple(evaluation_records[name]) != tuple(artifact_records.get(name) or ()):
            raise ValueError(
                "evaluation metrics %s record fingerprints do not match the "
                "selected artifact training provenance" % name
            )
    _validate_test_capture_binding(payload, config_path)


def _validate_split_record_fingerprints(
    raw_evidence: Any,
    *,
    required_splits: tuple[str, ...],
    label: str,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("%s must be a mapping" % label)
    if (
        str(raw_evidence.get("algorithm") or "")
        != "sha256:noema.amc.capture-record@1"
    ):
        raise ValueError("%s uses an unsupported fingerprint algorithm" % label)
    if raw_evidence.get("disjoint") is not True:
        raise ValueError("%s must declare disjoint=true" % label)
    raw_splits = raw_evidence.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("%s.splits must be a mapping" % label)
    records: dict[str, tuple[str, ...]] = {}
    for name in required_splits:
        row = raw_splits.get(name)
        if not isinstance(row, Mapping):
            raise ValueError("%s is missing split %s" % (label, name))
        raw_records = row.get("record_sha256")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(
                "%s split %s requires record_sha256 evidence" % (label, name)
            )
        fingerprints = tuple(
            _required_sha256(value, "%s split %s record SHA-256" % (label, name))
            for value in raw_records
        )
        unique = sorted(set(fingerprints))
        expected_fields = {
            "record_count": len(fingerprints),
            "unique_record_count": len(unique),
            "fingerprint_set_sha256": _fingerprint_set_sha256(unique),
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    "%s split %s has inconsistent %s" % (label, name, field)
                )
        records[name] = fingerprints

    declared_pairs = raw_evidence.get("pairwise_overlap")
    if not isinstance(declared_pairs, list):
        raise ValueError("%s.pairwise_overlap must be a list" % label)
    declared = {
        frozenset(
            (
                str(item.get("left_split") or ""),
                str(item.get("right_split") or ""),
            )
        ): item.get("overlap_count")
        for item in declared_pairs
        if isinstance(item, Mapping)
    }
    for left_index, left in enumerate(required_splits):
        for right in required_splits[left_index + 1 :]:
            overlap_count = len(set(records[left]).intersection(records[right]))
            if overlap_count:
                raise ValueError(
                    "%s contains %d byte-identical record%s across %s and %s"
                    % (
                        label,
                        overlap_count,
                        "" if overlap_count == 1 else "s",
                        left,
                        right,
                    )
                )
            if declared.get(frozenset((left, right))) != 0:
                raise ValueError(
                    "%s lacks zero-overlap evidence for %s and %s"
                    % (label, left, right)
                )
    return records


def _fingerprint_set_sha256(fingerprints: list[str]) -> str:
    digest = hashlib.sha256(b"noema.amc.capture-record-set@1\0")
    for fingerprint in fingerprints:
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


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
        raise ValueError("canonical recipe requires step %s using %s" % (step_id, operation))


def _assert_held_out_operation_seeds(
    recipe_path: Path,
    paired_seeds: list[int],
    step_offsets: Mapping[str, int],
    step_streams: Mapping[str, str],
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
                    stream = str(step_streams.get(step_id) or "default")
                    operation_seed = derive_seed(
                        master_seed,
                        str(capture_recipe.get("name") or ""),
                        step_id,
                        stream,
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


def _finite_floats(value: str, label: str) -> list[float]:
    numbers = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not numbers or any(not math.isfinite(item) for item in numbers):
        raise ValueError("%s must be a non-empty list of finite numbers" % label)
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
        "  noema benchmark publish RESULT_ID --slug learned-automatic-modulation-recognition "
        "--out docs/demo/experiments/learned-automatic-modulation-recognition"
    )


if __name__ == "__main__":
    raise SystemExit(main())
