from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from noema_lab.core.matrix import canonicalize_recipe_matrix
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.recipes import Recipe
from noema_lab.core.reproducibility import canonical_json_sha256, master_seed_from_recipe
from noema_lab.core.training_plans import scenario_recipe_fingerprint
from noema_lab.training.capture_plan import (
    TrainingCapturePlanError,
    replacement_boundary_capture_requirements,
    resolve_training_capture_plan,
)


JsonDict = Dict[str, Any]


class GenericCaptureDataContractError(ValueError):
    pass


def write_generic_capture_data_contract(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    selected_step_ids: Sequence[str],
    out_dir: Path,
    project_root: Path,
) -> JsonDict:
    """Write an operation-agnostic tensor-capture handoff for custom slots."""

    _capture_matrix_mode(recipe)
    requirements = replacement_boundary_capture_requirements(
        recipe, registry, selected_step_ids
    )
    unsupported = list(requirements.get("unsupported_inputs") or [])
    if unsupported:
        details = ", ".join(
            "%s.%s <- %s"
            % (item.get("step_id"), item.get("input"), item.get("from"))
            for item in unsupported
        )
        raise GenericCaptureDataContractError(
            "Generic capture cannot materialize non-array replacement input(s): %s"
            % details
        )
    try:
        plan = resolve_training_capture_plan(
            recipe,
            registry,
            required_taps=list(requirements.get("required_taps") or []),
            sample_unit="recipe records",
            suggested_total_samples=1000,
        )
    except TrainingCapturePlanError as exc:
        raise GenericCaptureDataContractError(str(exc)) from exc
    if not plan.taps:
        raise GenericCaptureDataContractError(
            "Select at least one capturable recipe output for the external training dataset"
        )

    required_steps = _capture_ancestor_steps(
        recipe, [tap.reference for tap in plan.taps]
    )
    _require_retained_capture_sweep_targets(recipe, required_steps)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data_root = destination / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve()
    source_sha = scenario_recipe_fingerprint(recipe)
    candidate_by_reference = {
        str(item.get("from") or ""): item for item in plan.candidates
    }
    capture_recipes = []
    jobs = []
    for split_spec in plan.splits:
        payload = _capture_recipe(
            recipe,
            plan=plan,
            split=split_spec.id,
            seed_offset=split_spec.seed_offset,
            samples=split_spec.samples,
        )
        path = destination / ("capture_%s_recipe.yaml" % split_spec.id)
        _write_yaml(path, payload)
        taps = [
            {"id": tap.id, "from": tap.reference}
            for tap in plan.taps
        ]
        semantic_sha = canonical_json_sha256(payload)
        file_sha = _file_sha256(path)
        capture_recipes.append(
            {
                "split": split_spec.id,
                "path": path.name,
                "sha256": semantic_sha,
                "file_sha256": file_sha,
                "requested_samples": split_spec.samples,
                "seed_offset": split_spec.seed_offset,
                "taps": taps,
            }
        )
        jobs.append(
            {
                "split": split_spec.id,
                "label": (
                    "Held-out test"
                    if split_spec.id == "test"
                    else split_spec.id.title()
                ),
                "recipe_path": _project_path(path, root),
                "bundle_recipe_path": path.name,
                "output_dir": _project_path(
                    data_root / split_spec.id, root
                ),
                "requested_samples": split_spec.samples,
                "expected_taps": taps,
                "sample_unit": plan.sample_unit,
                "seed_offset": split_spec.seed_offset,
                "seed_role": split_spec.training_use,
                "owner": "noema",
                "consumer": "external_researcher",
                "source_recipe_sha256": source_sha,
                "recipe_sha256": semantic_sha,
                "recipe_file_sha256": file_sha,
            }
        )

    signals = []
    for tap in plan.taps:
        candidate = candidate_by_reference.get(tap.reference, {})
        signals.append(
            {
                "tap_id": tap.id,
                "reference": tap.reference,
                "kind": str(candidate.get("kind") or "operation_defined_array"),
                "role": tap.role,
                "required": bool(tap.required),
            }
        )
    contract = {
        "schema_version": 1,
        "kind": "noema.training_data_contract@1",
        "mode": "captured_generic_tensors",
        "source_recipe": {"name": recipe.name, "sha256": source_sha},
        "selected_replacement_steps": [str(item) for item in selected_step_ids],
        "ownership": {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        },
        "capture_plan": plan.to_dict(),
        "signals": signals,
        "splits": [
            {
                "id": item.id,
                "requested_samples": item.samples,
                "seed_offset": item.seed_offset,
                "training_use": item.training_use,
            }
            for item in plan.splits
        ],
        "capture_recipes": capture_recipes,
    }
    contract_path = destination / "data_contract.yaml"
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


def _capture_recipe(
    recipe: Recipe,
    *,
    plan: Any,
    split: str,
    seed_offset: int,
    samples: int,
) -> JsonDict:
    payload = recipe.to_dict()
    required_steps = _capture_ancestor_steps(
        recipe,
        [tap.reference for tap in plan.taps],
    )
    # The generated recipe is an execution projection, not the complete
    # research scenario.  Preserve the source topology as provenance while
    # declaring the deliberately pruned capture graph as custom; otherwise a
    # standard profile validator correctly rejects the missing downstream
    # stages when the capture job starts.
    source_profile = dict(payload.get("execution_profile") or {})
    if str(source_profile.get("id") or "") != "custom":
        payload["execution_profile"] = {
            "id": "custom",
            "version": 1,
            "based_on": {
                "id": str(source_profile.get("id") or "custom"),
                "version": int(source_profile.get("version") or 1),
            },
        }
    metadata = dict(payload.get("metadata") or {})
    metadata["seed"] = int(master_seed_from_recipe(recipe) or 23) + int(seed_offset)
    metadata["training_performed"] = False
    metadata["capture_purpose"] = "generic_portable_replacement"
    metadata["capture_split"] = str(split)
    _project_capture_matrix(
        recipe,
        metadata,
        retained_step_ids=required_steps,
    )
    payload["metadata"] = metadata
    # A capture job executes only the ancestors needed to produce the selected
    # taps. Replacement slots and unrelated downstream evaluation blocks are
    # not part of input capture unless the researcher explicitly selects one of
    # their outputs.
    payload["steps"] = [
        step
        for step in payload.get("steps") or []
        if str(step.get("id") or "") in required_steps
    ]
    # Let the capture runner derive independent per-run streams from the split
    # master seed instead of preserving a hard-coded seed on one operation.
    for step in payload.get("steps") or []:
        params = dict(step.get("params") or {})
        params.pop("seed", None)
        step["params"] = params
    capture = dict(payload.get("dataset_capture") or {})
    capture.update(
        {
            "split": str(split),
            "samples": int(samples),
            "shard_size": min(int(capture.get("shard_size") or 64), int(samples)),
            "max_runs": int(samples),
            "seed_mode": "increment_run_seed",
            "taps": [
                {"id": tap.id, "from": tap.reference}
                for tap in plan.taps
            ],
        }
    )
    capture.pop("split_plan", None)
    payload["dataset_capture"] = capture
    return payload


def _project_capture_matrix(
    recipe: Recipe,
    metadata: JsonDict,
    *,
    retained_step_ids: set[str],
) -> None:
    """Project recipe-wide matrix bindings onto the pruned capture graph.

    Capture recipes intentionally omit replacement and downstream steps.  A
    copied matrix may therefore no longer reference those steps: matrix
    validation happens before execution and correctly rejects such dangling
    bindings.  The capture projection keeps only bindings on retained ancestor
    steps and only dimensions still referenced by those bindings.  The source
    ``Recipe`` and its authored matrix remain unchanged.
    """

    if _capture_matrix_mode(recipe) == "exclude":
        metadata.pop("matrix", None)
        metadata.pop("sweeps", None)
        metadata.pop("ui_sweeps", None)
        return

    compiled = canonicalize_recipe_matrix(recipe)
    if not compiled.enabled:
        return
    definition = dict(compiled.definition or {})
    step_params = dict(definition.get("step_params") or {})
    retained_bindings = {
        str(step_id): copy.deepcopy(template)
        for step_id, template in step_params.items()
        if str(step_id) in retained_step_ids and isinstance(template, Mapping)
    }
    referenced_dimensions: set[str] = set()
    _collect_matrix_dimensions(retained_bindings, referenced_dimensions)

    # Generated execution projections use the canonical field even when the
    # source entered through a legacy compatibility alias.  The authored recipe
    # is not mutated.
    metadata.pop("sweeps", None)
    metadata.pop("ui_sweeps", None)
    if not retained_bindings or not referenced_dimensions:
        metadata.pop("matrix", None)
        return

    dimensions = {
        str(name): copy.deepcopy(values)
        for name, values in dict(definition.get("dimensions") or {}).items()
        if str(name) in referenced_dimensions
    }
    metadata["matrix"] = {
        "dimensions": dimensions,
        "step_params": retained_bindings,
    }


def _capture_matrix_mode(recipe: Recipe) -> str:
    raw_mode = (recipe.dataset_capture or {}).get("matrix_mode", "inherit")
    mode = str(raw_mode or "inherit").strip().lower()
    if mode not in {"inherit", "exclude"}:
        raise GenericCaptureDataContractError(
            "dataset_capture.matrix_mode must be one of inherit or exclude"
        )
    return mode


def _collect_matrix_dimensions(value: Any, dimensions: set[str]) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"matrix"} and isinstance(value.get("matrix"), str):
            dimensions.add(str(value["matrix"]))
            return
        for item in value.values():
            _collect_matrix_dimensions(item, dimensions)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_matrix_dimensions(item, dimensions)


def _capture_ancestor_steps(recipe: Recipe, references: Sequence[str]) -> set[str]:
    step_by_id = {step.id: step for step in recipe.steps}
    required = {
        str(reference).split(".", 1)[0]
        for reference in references
        if isinstance(reference, str) and "." in reference
    }
    pending = list(required)
    while pending:
        step_id = pending.pop()
        step = step_by_id.get(step_id)
        if step is None:
            continue
        for reference in step.inputs.values():
            producer = str(reference).split(".", 1)[0]
            if producer and producer not in required:
                required.add(producer)
                pending.append(producer)
    return required


def _require_retained_capture_sweep_targets(
    recipe: Recipe,
    retained_step_ids: set[str],
) -> None:
    """Reject capture conditions that disappear from the pruned capture graph."""

    capture = dict(recipe.dataset_capture or {})
    raw_sweep = capture.get("sweep")
    if isinstance(raw_sweep, Mapping):
        assignments = [raw_sweep]
    elif isinstance(raw_sweep, list):
        assignments = [
            assignment
            for assignment in raw_sweep
            if isinstance(assignment, Mapping)
        ]
    else:
        return

    pruned_paths = []
    seen = set()
    for assignment in assignments:
        for raw_path in assignment:
            path = str(raw_path)
            step_id = _capture_sweep_target_step_id(recipe, path)
            if step_id and step_id not in retained_step_ids and path not in seen:
                pruned_paths.append(path)
                seen.add(path)
    if not pruned_paths:
        return
    raise GenericCaptureDataContractError(
        "dataset_capture.sweep target(s) are outside the selected-tap ancestor "
        "graph and would be pruned from the generic capture recipe: %s. "
        "Capture an output downstream of each target or remove the target from "
        "dataset_capture.sweep."
        % ", ".join(pruned_paths)
    )


def _capture_sweep_target_step_id(recipe: Recipe, path: str) -> str:
    """Resolve the target block using the dataset-capture runner's aliases."""

    step_name, separator, param_name = str(path).partition(".")
    if not separator or not step_name or not param_name:
        return ""
    step_by_id = {step.id: step for step in recipe.steps}
    if step_name in step_by_id:
        return step_name
    if step_name != "channel":
        return ""
    for step in recipe.steps:
        if (
            str(step.op).startswith("wireless.")
            or step.id == "wireless_channel"
            or param_name in step.params
        ):
            return step.id
    return ""


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
