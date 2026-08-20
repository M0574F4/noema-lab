from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence

from noema_lab.core.reproducibility import SEED_MODULUS, canonical_json_sha256


JsonDict = Dict[str, Any]


class CommonConditionError(ValueError):
    """A publication comparison did not materialize its declared conditions."""


def bind_common_conditions(
    payload: JsonDict,
    *,
    benchmark_id: str,
    benchmark_version: str,
    entry_id: str,
    conditions: Mapping[str, Any],
    publication_ready: bool,
) -> None:
    """Freeze the declaration and pair stochastic operations across methods."""

    if not conditions:
        return
    declaration = copy.deepcopy(dict(conditions))
    metadata = dict(payload.get("metadata") or {})
    pairing_id = next(
        (
            metadata.get(key)
            for key in ("pairing_id", "paired_seed", "benchmark_paired_seed")
            if metadata.get(key) not in (None, "")
        ),
        None,
    )
    cell_id = metadata.get("aggregation_cell_id")
    randomness = declaration.get("randomness")
    randomness = randomness if isinstance(randomness, Mapping) else {}
    raw_operations = randomness.get("paired_operation_ids") or []
    paired_operations = (
        [str(item).strip() for item in raw_operations if str(item).strip()]
        if isinstance(raw_operations, list)
        else []
    )
    if publication_ready and (pairing_id in (None, "") or cell_id in (None, "")):
        raise CommonConditionError(
            "publication-ready benchmark recipe %s requires pairing_id and aggregation_cell_id"
            % entry_id
        )
    seed_material = {
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "aggregation_cell_id": None if cell_id in (None, "") else str(cell_id),
        "pairing_id": None if pairing_id in (None, "") else str(pairing_id),
    }
    paired_seed = _paired_seed(seed_material)
    bound_steps: List[JsonDict] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        operation = str(step.get("op") or "")
        if operation not in paired_operations:
            continue
        params = dict(step.get("params") or {})
        params["seed"] = paired_seed
        step["params"] = params
        bound_steps.append(
            {
                "step_id": str(step.get("id") or ""),
                "operation": operation,
                "seed": paired_seed,
            }
        )
    if publication_ready and (not paired_operations or not bound_steps):
        raise CommonConditionError(
            "publication-ready benchmark recipe %s must bind at least one common_conditions.randomness.paired_operation_ids step"
            % entry_id
        )
    contract = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "declaration": declaration,
        "declaration_sha256": canonical_json_sha256(declaration),
        "pairing_id": seed_material["pairing_id"],
        "aggregation_cell_id": seed_material["aggregation_cell_id"],
        "paired_seed": paired_seed,
        "implemented_seed_derivation": (
            "sha256(benchmark_id,benchmark_version,aggregation_cell_id,pairing_id) "
            "mod 2147483647; operation-owned per-item derivation binds source identity"
        ),
        "bound_random_steps": bound_steps,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    metadata["benchmark_common_condition_contract"] = contract
    payload["metadata"] = metadata


def materialize_common_condition_evidence(
    recipe: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> JsonDict:
    """Project condition evidence from immutable recipe and run semantics."""

    metadata = recipe.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    contract = metadata.get("benchmark_common_condition_contract")
    if not isinstance(contract, Mapping):
        return {}
    declaration = contract.get("declaration")
    if not isinstance(declaration, Mapping):
        raise CommonConditionError("common-condition contract has no declaration")
    contract_without_hash = {
        key: copy.deepcopy(value)
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    if contract.get("contract_sha256") != canonical_json_sha256(
        contract_without_hash
    ):
        raise CommonConditionError("common-condition contract SHA-256 is invalid")
    if contract.get("declaration_sha256") != canonical_json_sha256(
        dict(declaration)
    ):
        raise CommonConditionError("common-condition declaration SHA-256 is invalid")
    pairing_id = next(
        (
            metadata.get(key)
            for key in ("pairing_id", "paired_seed", "benchmark_paired_seed")
            if metadata.get(key) not in (None, "")
        ),
        None,
    )
    cell_id = metadata.get("aggregation_cell_id")
    expected_seed_material = {
        "benchmark_id": contract.get("benchmark_id"),
        "benchmark_version": contract.get("benchmark_version"),
        "aggregation_cell_id": None if cell_id in (None, "") else str(cell_id),
        "pairing_id": None if pairing_id in (None, "") else str(pairing_id),
    }
    if (
        contract.get("pairing_id") != expected_seed_material["pairing_id"]
        or contract.get("aggregation_cell_id")
        != expected_seed_material["aggregation_cell_id"]
        or contract.get("paired_seed") != _paired_seed(expected_seed_material)
    ):
        raise CommonConditionError(
            "common-condition pairing identity or derived seed is invalid"
        )
    steps = [step for step in summary.get("steps") or [] if isinstance(step, Mapping)]
    recipe_steps = {
        str(step.get("id") or ""): step
        for step in recipe.get("steps") or []
        if isinstance(step, Mapping)
    }
    source_evidence = _source_evidence(steps)
    randomness_evidence = _randomness_evidence(
        steps,
        recipe_steps,
        declaration,
        paired_seed=contract.get("paired_seed"),
    )
    power_evidence = _power_evidence(steps, declaration)
    receiver_evidence = _receiver_evidence(steps, declaration)
    failure_evidence = _failure_evidence(
        steps,
        declaration,
        source_count=source_evidence.get("item_count"),
    )
    sections = {
        "source": source_evidence,
        "randomness": randomness_evidence,
        "power": power_evidence,
        "receiver": receiver_evidence,
        "failure": failure_evidence,
    }
    complete = all(bool(section.get("complete")) for section in sections.values())
    comparison_identity = {
        "source": {
            key: source_evidence.get(key)
            for key in (
                "ordered_post_transform_sha256",
                "batch_tensor_sha256",
                "dataset_manifest_sha256",
                "source_operation_contract_sha256",
                "item_count",
                "ordered_item_ids",
            )
        },
        "randomness": {
            key: randomness_evidence.get(key)
            for key in (
                "paired_seed",
                "source_item_channel_seeds",
                "source_item_channel_identity_keys",
                "channel_realization_contract_sha256",
                "materializations_sha256",
            )
        },
        "power": {
            **{
                key: power_evidence.get(key)
                for key in (
                    "coordinate",
                    "normalization_scope",
                    "target",
                    "power_unit",
                )
            },
            **_canonical_power_comparison(power_evidence),
        },
        "receiver": {
            key: receiver_evidence.get(key)
            for key in (
                "channel_state_information",
                "receiver_processing",
                "channel_state_mode",
                "channel_state_shared",
                "transmitter_csi_assumption",
                "channel",
                "wireless_backend",
                "equalizer",
            )
        },
        "failure": {
            "denominator_policy": failure_evidence.get("denominator_policy"),
            "denominator": failure_evidence.get("denominator"),
        },
    }
    evidence: JsonDict = {
        "schema_version": 1,
        "contract_sha256": contract.get("contract_sha256"),
        "declaration_sha256": contract.get("declaration_sha256"),
        "pairing_id": contract.get("pairing_id"),
        "aggregation_cell_id": contract.get("aggregation_cell_id"),
        "complete": complete,
        "sections": sections,
        "comparison_identity": comparison_identity,
        "comparison_identity_sha256": canonical_json_sha256(comparison_identity),
    }
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def _canonical_power_comparison(power_evidence: Mapping[str, Any]) -> JsonDict:
    """Collapse values already accepted as target-equivalent to the target.

    The completeness predicate uses a scale-aware floating tolerance.  The
    cross-method identity must use the same equivalence relation; otherwise
    two accepted normalizations such as 1.0 and float32(1.0) can fail only at
    the subsequent digest comparison.
    """

    target = power_evidence.get("target")
    try:
        target_value = float(target)
        tolerance = max(1e-9, abs(target_value) * 1e-7)
    except (TypeError, ValueError):
        return {
            "actual_coordinate_value": power_evidence.get(
                "actual_coordinate_value"
            ),
            "source_item_power_after": power_evidence.get(
                "source_item_power_after"
            ),
        }

    def canonical(value: Any) -> Any:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if abs(numeric - target_value) <= tolerance:
            return target
        return value

    per_item = power_evidence.get("source_item_power_after")
    return {
        "actual_coordinate_value": canonical(
            power_evidence.get("actual_coordinate_value")
        ),
        "source_item_power_after": (
            [canonical(value) for value in per_item]
            if isinstance(per_item, list)
            else per_item
        ),
    }


def validate_common_condition_evidence_set(
    entries: Sequence[Mapping[str, Any]],
    *,
    publication_ready: bool,
) -> List[str]:
    """Return fail-closed cross-method condition errors, grouped by paired unit."""

    errors: List[str] = []
    groups: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}
    for index, entry in enumerate(entries):
        status = str(entry.get("status") or "").lower()
        if status not in {"completed", "rejected_resource_budget"}:
            continue
        evidence = entry.get("common_condition_evidence")
        if not isinstance(evidence, Mapping):
            if publication_ready:
                errors.append("recipe %d has no common-condition evidence" % index)
            continue
        if evidence.get("complete") is not True:
            errors.append(
                "recipe %s has incomplete common-condition evidence"
                % (entry.get("id") or index)
            )
        if evidence.get("evidence_sha256") != canonical_json_sha256(
            {key: copy.deepcopy(value) for key, value in evidence.items() if key != "evidence_sha256"}
        ):
            errors.append(
                "recipe %s common-condition evidence hash is invalid"
                % (entry.get("id") or index)
            )
        cell = str(evidence.get("aggregation_cell_id") or "")
        pair = str(evidence.get("pairing_id") or "")
        if publication_ready and (not cell or not pair):
            errors.append(
                "recipe %s common-condition evidence lacks a paired cell identity"
                % (entry.get("id") or index)
            )
        groups.setdefault((cell, pair), []).append(entry)
    for (cell, pair), rows in groups.items():
        if len(rows) < 2:
            if publication_ready:
                errors.append(
                    "paired cell %s/%s has no second method with the same pairing identity"
                    % (cell, pair)
                )
            continue
        baseline = rows[0].get("common_condition_evidence") or {}
        expected = baseline.get("comparison_identity_sha256")
        for row in rows[1:]:
            evidence = row.get("common_condition_evidence") or {}
            if evidence.get("comparison_identity_sha256") != expected:
                errors.append(
                    "paired cell %s/%s materialized different source/randomness/power/receiver/failure conditions"
                    % (cell, pair)
                )
    return errors


def _paired_seed(seed_material: Mapping[str, Any]) -> int:
    value = int(canonical_json_sha256(dict(seed_material))[:16], 16) % SEED_MODULUS
    return value or 1


def _step_output_metadata(step: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    output: List[Mapping[str, Any]] = []
    metadata = step.get("metadata")
    if isinstance(metadata, Mapping):
        output.append(metadata)
    for artifact in (step.get("outputs") or {}).values() if isinstance(step.get("outputs"), Mapping) else []:
        if isinstance(artifact, Mapping) and isinstance(artifact.get("metadata"), Mapping):
            output.append(artifact["metadata"])
    return output


def _source_evidence(steps: Sequence[Mapping[str, Any]]) -> JsonDict:
    candidates: List[Mapping[str, Any]] = []
    for step in steps:
        operation = str(step.get("op") or "")
        if not operation.startswith("source."):
            continue
        for metadata in _step_output_metadata(step):
            if metadata.get("ordered_post_transform_sha256"):
                candidates.append(metadata)
    if len(candidates) != 1:
        return {"complete": False, "reason": "requires exactly one ordered post-transform source"}
    metadata = candidates[0]
    items = metadata.get("ordered_post_transform_items")
    if not isinstance(items, list) or not items:
        return {"complete": False, "reason": "source has no ordered item inventory"}
    expected = canonical_json_sha256(items)
    actual = str(metadata.get("ordered_post_transform_sha256") or "")
    item_count = int(metadata.get("source_item_count") or len(items))
    batch_tensor_sha256 = str(metadata.get("batch_tensor_sha256") or "")
    source_contract_sha256 = str(
        metadata.get("source_operation_contract_sha256") or ""
    )
    ordered_ids = [str(item.get("item_id") or "") for item in items if isinstance(item, Mapping)]
    complete = bool(
        actual == expected
        and _is_sha256(batch_tensor_sha256)
        and _is_sha256(source_contract_sha256)
        and item_count == len(items) == len(ordered_ids)
        and all(ordered_ids)
    )
    return {
        "complete": complete,
        "ordered_post_transform_sha256": actual,
        "batch_tensor_sha256": batch_tensor_sha256,
        "source_operation_contract_sha256": source_contract_sha256,
        "item_count": item_count,
        "ordered_item_ids": ordered_ids,
        "dataset_manifest_sha256": metadata.get("dataset_manifest_sha256"),
    }


def _randomness_evidence(
    steps: Sequence[Mapping[str, Any]],
    recipe_steps: Mapping[str, Mapping[str, Any]],
    declaration: Mapping[str, Any],
    *,
    paired_seed: Any,
) -> JsonDict:
    randomness = declaration.get("randomness")
    randomness = randomness if isinstance(randomness, Mapping) else {}
    operation_ids = randomness.get("paired_operation_ids") or []
    operation_ids = set(operation_ids) if isinstance(operation_ids, list) else set()
    rows: List[JsonDict] = []
    for step in steps:
        if str(step.get("op") or "") not in operation_ids:
            continue
        metadata_candidates = _step_output_metadata(step)
        metadata = next(
            (
                value
                for value in reversed(metadata_candidates)
                if value.get("seed") is not None
                or value.get("source_item_channel_seeds") is not None
                or value.get("channel_state_seed") is not None
            ),
            {},
        )
        recipe_step = recipe_steps.get(str(step.get("id") or "")) or {}
        params = dict(recipe_step.get("params") or {})
        rows.append(
            {
                "step_id": str(step.get("id") or ""),
                "operation": str(step.get("op") or ""),
                "seed": metadata.get("seed", params.get("seed")),
                "channel_state_seed": metadata.get("channel_state_seed"),
                "source_item_channel_seeds": metadata.get("source_item_channel_seeds"),
                "source_item_channel_identity_keys": metadata.get("source_item_channel_identity_keys"),
                "realization_artifact_sha256": sorted(
                    str(artifact.get("sha256"))
                    for artifact in (step.get("outputs") or {}).values()
                    if isinstance(artifact, Mapping)
                    and artifact.get("sha256")
                    and (
                        "channel_state" in str(artifact.get("kind") or "")
                        or "channel_state" in str(artifact.get("path") or "")
                    )
                )
                if isinstance(step.get("outputs"), Mapping)
                else [],
                "channel_contract": {
                    "operation": str(step.get("op") or ""),
                    "params": params,
                    "channel": metadata.get("channel"),
                    "wireless_backend": metadata.get("wireless_backend"),
                    "receiver_processing": metadata.get("receiver_processing"),
                    "channel_state_mode": metadata.get("channel_state_mode"),
                },
            }
        )
    if not rows:
        return {"complete": False, "reason": "requires paired stochastic channel operation evidence"}
    rows.sort(key=lambda row: (str(row["operation"]), str(row["step_id"])))
    per_item_rows = [
        row
        for row in rows
        if isinstance(row.get("source_item_channel_seeds"), list)
        and row.get("source_item_channel_seeds")
    ]
    state_rows = [
        row
        for row in rows
        if row.get("channel_state_seed") is not None
        and row.get("realization_artifact_sha256")
    ]
    seed_matches = all(
        isinstance(row.get("seed"), int) and row.get("seed") == paired_seed
        for row in rows
    )
    per_item_valid = all(
        isinstance(row.get("source_item_channel_identity_keys"), list)
        and len(row["source_item_channel_identity_keys"])
        == len(row["source_item_channel_seeds"])
        for row in per_item_rows
    )
    realization_present = bool(per_item_rows or state_rows)
    complete = bool(seed_matches and per_item_valid and realization_present)
    primary = per_item_rows[-1] if per_item_rows else rows[-1]
    materializations = [
        {
            "operation": row["operation"],
            "seed": row["seed"],
            "channel_state_seed": row["channel_state_seed"],
            "source_item_channel_seeds": row["source_item_channel_seeds"],
            "source_item_channel_identity_keys": row[
                "source_item_channel_identity_keys"
            ],
            "realization_artifact_sha256": row["realization_artifact_sha256"],
            "channel_contract_sha256": canonical_json_sha256(
                row["channel_contract"]
            ),
        }
        for row in rows
    ]
    return {
        "complete": complete,
        "paired_seed": paired_seed,
        "source_item_channel_seeds": primary.get("source_item_channel_seeds"),
        "source_item_channel_identity_keys": primary.get(
            "source_item_channel_identity_keys"
        ),
        "channel_realization_contract_sha256": canonical_json_sha256(
            primary["channel_contract"]
        ),
        "materializations": materializations,
        "materializations_sha256": canonical_json_sha256(materializations),
    }


def _power_evidence(
    steps: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any]
) -> JsonDict:
    power = declaration.get("power")
    power = power if isinstance(power, Mapping) else {}
    coordinate = str(power.get("coordinate") or "").strip()
    target = power.get("target")
    scope = str(power.get("normalization_scope") or "").strip()
    metadata_rows = [
        metadata
        for step in steps
        for metadata in _step_output_metadata(step)
        if metadata.get("tx_power_per_executed_use") is not None
        or metadata.get("power_normalization_target") is not None
    ]
    if not metadata_rows:
        return {"complete": False, "reason": "no measured power evidence"}
    downstream = metadata_rows[-1]
    normalized = next(
        (
            row
            for row in reversed(metadata_rows)
            if row.get("power_normalization_target") is not None
        ),
        downstream,
    )
    if "energy" in coordinate.lower():
        actual = downstream.get("tx_total_energy", normalized.get("power_total_energy"))
    else:
        actual = downstream.get("tx_power_per_executed_use", normalized.get("power_after"))
    actual_scope = str(
        normalized.get("power_normalization_scope") or ""
    ).strip()
    try:
        matches = abs(float(actual) - float(target)) <= max(1e-9, abs(float(target)) * 1e-7)
    except (TypeError, ValueError):
        matches = False
    source_item_power_after = normalized.get("source_item_power_after")
    if scope == "source_item":
        per_item_matches = bool(
            isinstance(source_item_power_after, list)
            and source_item_power_after
            and all(
                abs(float(value) - float(target))
                <= max(1e-9, abs(float(target)) * 1e-7)
                for value in source_item_power_after
            )
        )
    else:
        per_item_matches = True
    return {
        "complete": bool(matches and actual_scope == scope and per_item_matches),
        "coordinate": coordinate,
        "normalization_scope": actual_scope,
        "target": target,
        "actual_coordinate_value": actual,
        "power_unit": downstream.get("power_unit", normalized.get("power_unit")),
        "tx_total_energy": downstream.get("tx_total_energy", normalized.get("power_total_energy")),
        "source_item_power_after": source_item_power_after,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _receiver_evidence(
    steps: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any]
) -> JsonDict:
    receiver = declaration.get("receiver")
    receiver = receiver if isinstance(receiver, Mapping) else {}
    rows = [
        metadata
        for step in steps
        for metadata in _step_output_metadata(step)
        if metadata.get("receiver_processing") is not None
        and metadata.get("channel_state_mode") is not None
    ]
    if not rows:
        return {"complete": False, "reason": "no receiver/CSI evidence"}
    actual = rows[-1]
    declared_csi = str(receiver.get("channel_state_information") or "").strip()
    declared_processing = str(receiver.get("receiver_processing") or "").strip()
    state_mode = str(actual.get("channel_state_mode") or "").strip()
    actual_csi = str(actual.get("channel_state_information") or state_mode).strip()
    processing = str(actual.get("receiver_processing") or "").strip()
    return {
        "complete": bool(actual_csi == declared_csi and processing == declared_processing),
        "channel_state_information": actual_csi,
        "channel_state_mode": state_mode,
        "receiver_processing": processing,
        "channel_state_shared": actual.get("channel_state_shared"),
        "transmitter_csi_assumption": actual.get("transmitter_csi_assumption"),
        "channel": actual.get("channel"),
        "wireless_backend": actual.get("wireless_backend"),
        "equalizer": actual.get("equalizer"),
    }


def _failure_evidence(
    steps: Sequence[Mapping[str, Any]],
    declaration: Mapping[str, Any],
    *,
    source_count: Any,
) -> JsonDict:
    failure = declaration.get("failure")
    failure = failure if isinstance(failure, Mapping) else {}
    rows = [
        metadata
        for step in steps
        for metadata in _step_output_metadata(step)
        if isinstance(metadata.get("source_item_outage"), list)
    ]
    if not rows:
        return {"complete": False, "reason": "no per-item failure/outage evidence"}
    actual = rows[-1]
    outcomes = list(actual.get("source_item_outage") or [])
    policy = str(actual.get("on_decode_failure") or "").strip()
    declared_policy = str(failure.get("decode_failure_policy") or "").strip()
    denominator_policy = str(failure.get("denominator_policy") or "").strip()
    try:
        denominator = int(source_count)
    except (TypeError, ValueError):
        denominator = -1
    complete = bool(
        denominator >= 1
        and len(outcomes) == denominator
        and all(item in {0, 1, False, True} for item in outcomes)
        and policy == declared_policy
        and denominator_policy
    )
    return {
        "complete": complete,
        "outage_definition": failure.get("outage_definition"),
        "decode_failure_policy": policy,
        "denominator_policy": denominator_policy,
        "denominator": denominator,
        "failed_count": sum(int(bool(item)) for item in outcomes),
        "source_item_outage": [int(bool(item)) for item in outcomes],
    }
