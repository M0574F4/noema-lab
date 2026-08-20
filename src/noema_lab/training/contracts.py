from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from noema_lab.core.execution_profiles import inspect_execution_profile
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import Recipe, RecipeStep
from noema_lab.core.reproducibility import canonical_json_sha256, seed_policy
from noema_lab.core.training_plans import neutral_recipe, scenario_recipe_fingerprint
from noema_lab.training.standalone_input import write_standalone_structured_input


JsonDict = Dict[str, Any]

TRAINABLE_SLOT_CONTRACT_KIND = "noema.trainable_slot_contract@1"
TYPED_SCENARIO_GRAPH_KIND = "noema.typed_training_scenario_graph@1"
TRAINING_INTERFACE_PROJECT_KIND = "noema.training_interface_bundle@1"


class TrainingContractError(ValueError):
    """Raised when a recipe cannot be represented by the neutral training contract."""


@dataclass(frozen=True)
class TensorSpec:
    """Framework-neutral tensor boundary used by a trainable slot or graph edge."""

    kind: str
    dtype: str
    shape: Tuple[str | int, ...] = ("...",)
    layout: str = "operation_defined"
    units: str = "unitless"
    domain: str = "operation_defined"
    description: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "units": self.units,
            "domain": self.domain,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class SlotGroupSpec:
    id: str
    slots: Tuple[str, ...]
    joint_training: bool = True
    atomic_artifact_return: bool = True
    description: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "slots": list(self.slots),
            "joint_training": bool(self.joint_training),
            "artifact_application": (
                "all_group_bindings" if self.atomic_artifact_return else "independent_bindings"
            ),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class NamedValueSpec:
    id: str
    source: str
    dtype: str = "float32"
    shape: Tuple[str | int, ...] = ()
    units: str = "unitless"
    description: str = ""
    required: bool = True

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "source": self.source,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "units": self.units,
            "required": bool(self.required),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class SignalSpec:
    id: str
    source: str
    tensor: TensorSpec
    purpose: str = "loss_or_diagnostics"
    description: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "source": self.source,
            "purpose": self.purpose,
            "tensor": self.tensor.to_dict(),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    expression: str
    enforcement: str
    scope: str = "per_sample"
    description: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "expression": self.expression,
            "enforcement": self.enforcement,
            "scope": self.scope,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class TrainingContractOptions:
    trainable_steps: Tuple[str, ...]
    framework: str = "torch"
    # Recipe evaluation/loss steps whose typed input operands define the live
    # downstream route. The metric operation itself remains external.
    loss_steps: Tuple[str, ...] = ()
    slot_roles: Mapping[str, str] = field(default_factory=dict)
    slot_groups: Tuple[SlotGroupSpec | Mapping[str, Any], ...] = ()
    tensor_overrides: Mapping[str, Mapping[str, Any] | TensorSpec] = field(default_factory=dict)
    conditioning: Tuple[NamedValueSpec | Mapping[str, Any], ...] = ()
    signals: Tuple[SignalSpec | Mapping[str, Any], ...] = ()
    constraints: Tuple[ConstraintSpec | Mapping[str, Any], ...] = ()
    scenario_steps: Tuple[str, ...] = ()
    contract_id: str = ""


@dataclass(frozen=True)
class CompiledTrainingContract:
    contract: JsonDict
    scenario_graph: JsonDict
    source_recipe: JsonDict

    @property
    def contract_sha256(self) -> str:
        return canonical_json_sha256(self.contract)

    @property
    def scenario_graph_sha256(self) -> str:
        return canonical_json_sha256(self.scenario_graph)

    def to_dict(self) -> JsonDict:
        return {
            "contract": dict(self.contract),
            "scenario_graph": dict(self.scenario_graph),
            "source_recipe": dict(self.source_recipe),
            "contract_sha256": self.contract_sha256,
            "scenario_graph_sha256": self.scenario_graph_sha256,
        }


_TENSOR_KIND_DEFAULTS: Mapping[str, TensorSpec] = {
    "image.batch.numpy": TensorSpec(
        kind="image.batch.numpy",
        dtype="uint8",
        shape=("batch", "height", "width", "channels"),
        layout="NHWC",
        domain="integer_[0,255]",
    ),
    "channel.symbols.complex_numpy": TensorSpec(
        kind="channel.symbols.complex_numpy",
        dtype="complex64",
        shape=("...",),
        units="normalized_complex_amplitude",
    ),
    "channel.rx_symbols.complex_numpy": TensorSpec(
        kind="channel.rx_symbols.complex_numpy",
        dtype="complex64",
        shape=("...",),
        units="normalized_complex_amplitude",
    ),
    "channel.ofdm_channel_state.numpy": TensorSpec(
        kind="channel.ofdm_channel_state.numpy",
        dtype="float32",
        shape=("channel_state", "subcarrier"),
        layout="state_subcarrier",
        domain="nonnegative_power_gain_and_channel_metadata",
    ),
    "channel.power_allocation.numpy": TensorSpec(
        kind="channel.power_allocation.numpy",
        dtype="float64",
        shape=("channel_state", "subcarrier"),
        layout="state_subcarrier",
        units="normalized_power",
        domain="nonnegative",
    ),
    "channel.llr.numpy": TensorSpec(
        kind="channel.llr.numpy",
        dtype="float64",
        shape=("...",),
        units="log_likelihood_ratio",
    ),
}


def tensor_spec_for_kind(
    kind: str,
    override: Optional[Mapping[str, Any] | TensorSpec] = None,
) -> TensorSpec:
    """Return an honest default tensor spec, with caller overrides for task-specific layouts."""

    base = _TENSOR_KIND_DEFAULTS.get(str(kind))
    if base is None:
        normalized = str(kind)
        if "bits" in normalized:
            base = TensorSpec(normalized, "uint8", ("...",), domain="binary_{0,1}")
        elif "indices" in normalized or "token" in normalized:
            base = TensorSpec(normalized, "int64", ("...",), domain="nonnegative_integer")
        elif "complex" in normalized or normalized.endswith("symbols.tensor"):
            base = TensorSpec(normalized, "complex64", ("...",))
        elif normalized.endswith(".numpy") or normalized.endswith(".tensor"):
            base = TensorSpec(normalized, "float32", ("...",))
        elif normalized.endswith(".json") or normalized.endswith(".report"):
            base = TensorSpec(normalized, "structured", (), layout="mapping")
        else:
            base = TensorSpec(normalized, "operation_defined", ("...",))
    if override is None:
        return base
    if isinstance(override, TensorSpec):
        if override.kind != kind:
            raise TrainingContractError(
                "Tensor override kind %s does not match operation kind %s" % (override.kind, kind)
            )
        return override
    if not isinstance(override, Mapping):
        raise TrainingContractError("Tensor overrides must be mappings or TensorSpec values")
    unknown = set(override) - {
        "kind",
        "dtype",
        "shape",
        "layout",
        "units",
        "domain",
        "description",
    }
    if unknown:
        raise TrainingContractError(
            "Tensor override has unknown field(s): %s" % ", ".join(sorted(str(item) for item in unknown))
        )
    override_kind = str(override.get("kind") or base.kind)
    if override_kind != kind:
        raise TrainingContractError(
            "Tensor override kind %s does not match operation kind %s" % (override_kind, kind)
        )
    raw_shape = override.get("shape", base.shape)
    if not isinstance(raw_shape, (list, tuple)):
        raise TrainingContractError("Tensor override shape must be a list or tuple")
    return TensorSpec(
        kind=override_kind,
        dtype=str(override.get("dtype") or base.dtype),
        shape=tuple(raw_shape),
        layout=str(override.get("layout") or base.layout),
        units=str(override.get("units") or base.units),
        domain=str(override.get("domain") or base.domain),
        description=str(override.get("description") or base.description),
    )


def compile_training_contract(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    options: TrainingContractOptions,
) -> CompiledTrainingContract:
    """Compile a recipe into an architecture-, loss-, and trainer-neutral interface contract."""

    validate_recipe_against_registry(recipe, registry)
    trainable_steps = _unique_nonempty(options.trainable_steps, "trainable_steps")
    steps_by_id = {step.id: step for step in recipe.steps}
    unknown = [step_id for step_id in trainable_steps if step_id not in steps_by_id]
    if unknown:
        raise TrainingContractError(
            "Trainable slot step(s) are not in the recipe: %s" % ", ".join(unknown)
        )

    operation_descriptions = {
        step.id: registry.get(step.op).describe()
        for step in recipe.steps
    }
    for step_id in trainable_steps:
        artifact_abi = operation_descriptions[step_id].get("trained_artifact_abi") or {}
        if not bool(artifact_abi):
            raise TrainingContractError(
                "Step %s is not a replaceable block with a trained-artifact ABI; operation %s "
                "may only be used as built-in or frozen support"
                % (step_id, steps_by_id[step_id].op)
            )

    recipe_sha = scenario_recipe_fingerprint(recipe)
    profile_inspection = inspect_execution_profile(recipe)
    source_recipe = neutral_recipe(recipe).to_dict()
    port_tensor_specs = _resolve_port_tensor_specs(
        recipe,
        operation_descriptions=operation_descriptions,
        tensor_overrides=options.tensor_overrides,
    )
    loss_steps = (
        _unique_nonempty(options.loss_steps, "loss_steps")
        if options.loss_steps
        else ()
    )
    unknown_loss_steps = [
        step_id for step_id in loss_steps if step_id not in steps_by_id
    ]
    if unknown_loss_steps:
        raise TrainingContractError(
            "Recipe loss step(s) are not in the recipe: %s"
            % ", ".join(unknown_loss_steps)
        )

    slots = _compile_slots(
        recipe,
        trainable_steps=trainable_steps,
        steps_by_id=steps_by_id,
        operation_descriptions=operation_descriptions,
        tensor_overrides=options.tensor_overrides,
        port_tensor_specs=port_tensor_specs,
        slot_roles=options.slot_roles,
    )
    groups = _compile_groups(trainable_steps, options.slot_groups)
    conditioning = _compile_conditioning(options.conditioning)
    loss_signals = _recipe_loss_input_signals(
        recipe,
        loss_steps,
        port_tensor_specs=port_tensor_specs,
        existing_signals=options.signals,
    )
    raw_signals = tuple(options.signals) + tuple(loss_signals)
    signals = _compile_signals(slots, raw_signals)
    constraints = _compile_constraints(options.constraints)
    declared_boundaries = _declared_training_boundaries(
        options.conditioning,
        raw_signals,
    )

    scenario_step_ids = (
        _unique_nonempty(options.scenario_steps, "scenario_steps")
        if options.scenario_steps
        else _minimal_replacement_scenario_steps(
            recipe,
            trainable_steps,
            signal_sources=_training_source_values(raw_signals),
            slot_groups=groups,
        )
    )
    scenario_unknown = [step_id for step_id in scenario_step_ids if step_id not in steps_by_id]
    if scenario_unknown:
        raise TrainingContractError(
            "Scenario step(s) are not in the recipe: %s" % ", ".join(scenario_unknown)
        )
    missing_slots = [step_id for step_id in trainable_steps if step_id not in scenario_step_ids]
    if missing_slots:
        raise TrainingContractError(
            "Scenario steps omit trainable slot(s): %s" % ", ".join(missing_slots)
        )
    _validate_scenario_topological_order(recipe, scenario_step_ids)

    graph = _compile_typed_scenario_graph(
        recipe,
        steps_by_id=steps_by_id,
        operation_descriptions=operation_descriptions,
        scenario_step_ids=scenario_step_ids,
        trainable_steps=trainable_steps,
        tensor_overrides=options.tensor_overrides,
        port_tensor_specs=port_tensor_specs,
        framework=str(options.framework or "torch"),
        recipe_sha=recipe_sha,
        execution_profile=profile_inspection.to_dict(),
        declared_boundaries=declared_boundaries,
    )
    contract_identity = canonical_json_sha256(
        {
            "source_recipe_sha256": recipe_sha,
            "framework": str(options.framework or "torch"),
            "trainable_steps": list(trainable_steps),
            "loss_steps": list(loss_steps),
            "slot_groups": groups,
            "scenario_steps": list(scenario_step_ids),
        }
    )
    contract_id = str(options.contract_id or "").strip() or "%s.%s" % (
        _safe_identifier(recipe.name),
        contract_identity[:12],
    )
    contract: JsonDict = {
        "schema_version": 1,
        "version": 1,
        "kind": TRAINABLE_SLOT_CONTRACT_KIND,
        "id": contract_id,
        "identity_sha256": contract_identity,
        "source_recipe": {
            "name": recipe.name,
            "schema_version": int(recipe.schema_version),
            "sha256": recipe_sha,
            "execution_profile": recipe.execution_profile.to_dict(),
            "execution_profile_status": profile_inspection.status,
        },
        "framework": str(options.framework or "torch"),
        "ownership": {
            "noema": [
                "recipe_and_execution_profile",
                "typed_slot_interfaces",
                "frozen_context_graph",
                "conditioning_and_signal_semantics",
                "constraints_and_artifact_return_boundary",
            ],
            "external_researcher": [
                "model_architecture",
                "training_loss",
                "optimizer_and_schedule",
                "trainer_and_checkpoint_selection",
            ],
        },
        "trainable_slots": slots,
        "slot_groups": groups,
        "conditioning": conditioning,
        "signals": signals,
        "recipe_loss_steps": list(loss_steps),
        "constraints": constraints,
        "randomness": seed_policy(recipe),
        "scenario_graph": {
            "path": "scenario_graph.json",
            "kind": TYPED_SCENARIO_GRAPH_KIND,
            "sha256": canonical_json_sha256(graph),
        },
        "artifact_return": {
            "manifest_kind": "noema.trained_block_artifact",
            "interface_contract_required": True,
            "bindings": _artifact_return_bindings(slots, groups),
            "groups": groups,
            "runtime_policy": (
                "A built-in safe format may be architecture-specific; arbitrary architectures require "
                "a declared portable runtime or explicit trusted adapter/plugin."
            ),
        },
        "training_policy": {
            "architecture": "external",
            "loss": "external",
            "trainer": "external",
            "benchmark_evaluation": "ordinary_noema_recipe_or_benchmark",
            "gradient_route": "selected_by_external_loss_over_declared_signals",
            "frozen_differentiable_materialization": (
                "only_nodes_on_the_selected_gradient_route_are_required_during_training"
            ),
        },
    }
    validate_compiled_training_contract(contract, graph, recipe=recipe)
    return CompiledTrainingContract(contract, graph, source_recipe)


def validate_compiled_training_contract(
    contract: Mapping[str, Any],
    scenario_graph: Mapping[str, Any],
    *,
    recipe: Optional[Recipe] = None,
) -> None:
    """Validate the normative structure and cross-references of a compiled contract."""

    if str(contract.get("kind") or "") != TRAINABLE_SLOT_CONTRACT_KIND:
        raise TrainingContractError("Unsupported training contract kind: %s" % contract.get("kind"))
    if int(contract.get("schema_version") or 0) != 1:
        raise TrainingContractError("Training contract requires schema_version=1")
    source = contract.get("source_recipe")
    if not isinstance(source, Mapping):
        raise TrainingContractError("Training contract requires source_recipe")
    recipe_sha = str(source.get("sha256") or "")
    if not _is_sha256(recipe_sha):
        raise TrainingContractError("Training contract source_recipe.sha256 is invalid")
    profile = source.get("execution_profile")
    if not isinstance(profile, Mapping) or not str(profile.get("id") or ""):
        raise TrainingContractError("Training contract requires a source execution profile")
    slots = contract.get("trainable_slots")
    if not isinstance(slots, list) or not slots:
        raise TrainingContractError("Training contract requires at least one trainable slot")
    slot_ids: set[str] = set()
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            raise TrainingContractError("trainable_slots[%d] must be a mapping" % index)
        step_id = str(slot.get("step_id") or "")
        if not step_id or step_id in slot_ids:
            raise TrainingContractError("Trainable slot step IDs must be non-empty and unique")
        slot_ids.add(step_id)
        for direction in ("inputs", "outputs"):
            ports = slot.get(direction)
            if not isinstance(ports, Mapping) or not ports:
                raise TrainingContractError("Trainable slot %s requires typed %s" % (step_id, direction))
            for port, tensor in ports.items():
                if not str(port) or not isinstance(tensor, Mapping):
                    raise TrainingContractError("Trainable slot %s has an invalid %s port" % (step_id, direction))
                _validate_tensor_mapping(tensor, "%s.%s.%s" % (step_id, direction, port))
    group_members: set[str] = set()
    for group in list(contract.get("slot_groups") or []):
        if not isinstance(group, Mapping):
            raise TrainingContractError("slot_groups entries must be mappings")
        members = set(str(item) for item in group.get("slots") or [])
        if not members or not members.issubset(slot_ids):
            raise TrainingContractError("Slot group contains missing or unknown trainable slots")
        if group_members.intersection(members):
            raise TrainingContractError("Trainable slots may not appear in multiple slot groups")
        group_members.update(members)
    if group_members != slot_ids:
        raise TrainingContractError("Slot groups must cover every trainable slot")
    for section in ("conditioning", "signals", "constraints"):
        rows = contract.get(section)
        if not isinstance(rows, list):
            raise TrainingContractError("Training contract %s must be a list" % section)
        row_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise TrainingContractError("Training contract %s entries must be mappings" % section)
            row_id = str(row.get("id") or "")
            if not row_id or row_id in row_ids:
                raise TrainingContractError("Training contract %s IDs must be non-empty and unique" % section)
            row_ids.add(row_id)
            if section in {"conditioning", "signals"} and not str(row.get("source") or ""):
                raise TrainingContractError("Training contract %s entry %s requires source" % (section, row_id))
            if section == "signals":
                tensor = row.get("tensor")
                if not isinstance(tensor, Mapping):
                    raise TrainingContractError("Training signal %s requires tensor" % row_id)
                _validate_tensor_mapping(tensor, "signal.%s" % row_id)
            if section == "constraints":
                if not str(row.get("expression") or "") or not str(row.get("enforcement") or ""):
                    raise TrainingContractError("Training constraint %s is incomplete" % row_id)

    if str(scenario_graph.get("kind") or "") != TYPED_SCENARIO_GRAPH_KIND:
        raise TrainingContractError("Unsupported typed scenario graph kind")
    graph_reference = contract.get("scenario_graph")
    if not isinstance(graph_reference, Mapping):
        raise TrainingContractError("Training contract requires scenario_graph reference")
    expected_graph_sha = str(graph_reference.get("sha256") or "")
    if expected_graph_sha != canonical_json_sha256(scenario_graph):
        raise TrainingContractError("Scenario graph SHA-256 does not match the training contract")
    if str((scenario_graph.get("source_recipe") or {}).get("sha256") or "") != recipe_sha:
        raise TrainingContractError("Scenario graph and training contract use different recipe hashes")
    if dict((scenario_graph.get("source_recipe") or {}).get("execution_profile") or {}) != dict(
        source.get("execution_profile") or {}
    ):
        raise TrainingContractError(
            "Scenario graph and training contract use different execution profiles"
        )
    expected_identity = canonical_json_sha256(
        {
            "source_recipe_sha256": recipe_sha,
            "framework": contract.get("framework"),
            "trainable_steps": [
                str(item.get("step_id") or "") for item in slots
            ],
            "loss_steps": list(contract.get("recipe_loss_steps") or []),
            "slot_groups": list(contract.get("slot_groups") or []),
            "scenario_steps": list(scenario_graph.get("topological_order") or []),
        }
    )
    if str(contract.get("identity_sha256") or "") != expected_identity:
        raise TrainingContractError("Training contract identity SHA-256 is invalid")
    nodes = scenario_graph.get("nodes")
    edges = scenario_graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TrainingContractError("Scenario graph requires node and edge lists")
    node_by_id = {str(node.get("id") or ""): node for node in nodes if isinstance(node, Mapping)}
    if "" in node_by_id or len(node_by_id) != len(nodes):
        raise TrainingContractError("Scenario graph node IDs must be non-empty and unique")
    placeholder_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if str(node.get("role") or "") == "trainable_placeholder"
    }
    if placeholder_ids != slot_ids:
        raise TrainingContractError("Scenario graph placeholders do not match trainable slots")
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise TrainingContractError("Scenario graph edges must be mappings")
        source_ref = edge.get("source") or {}
        target_ref = edge.get("target") or {}
        source_id = str(source_ref.get("step_id") or "")
        target_id = str(target_ref.get("step_id") or "")
        source_port = str(source_ref.get("port") or "")
        target_port = str(target_ref.get("port") or "")
        if source_id not in node_by_id or target_id not in node_by_id:
            raise TrainingContractError("Scenario graph edge references an unknown node")
        if source_port not in (node_by_id[source_id].get("outputs") or {}):
            raise TrainingContractError("Scenario graph edge references an unknown source port")
        if target_port not in (node_by_id[target_id].get("inputs") or {}):
            raise TrainingContractError("Scenario graph edge references an unknown target port")
        _validate_tensor_mapping(edge.get("tensor") or {}, "scenario edge")
    _validate_declared_sources(contract, scenario_graph)
    _validate_artifact_return_bindings(contract)
    if recipe is not None and scenario_recipe_fingerprint(recipe) != recipe_sha:
        raise TrainingContractError("Training contract recipe hash does not match the supplied recipe")


def _validate_declared_sources(
    contract: Mapping[str, Any],
    scenario_graph: Mapping[str, Any],
) -> None:
    nodes = {
        str(node.get("id") or ""): node
        for node in list(scenario_graph.get("nodes") or [])
        if isinstance(node, Mapping)
    }
    boundary_sources = {
        str(item.get("recipe_reference") or "")
        for item in list(scenario_graph.get("external_inputs") or [])
        if isinstance(item, Mapping)
    }
    # External-input IDs name the local target port (for example
    # sender.images); only recipe_reference names a readable boundary source.
    # Treating the target ID as a source would let an invalid sender.images
    # output declaration pass merely because sender also has an images input.
    boundary_sources.update(
        str(item.get("id") or "")
        for item in list(scenario_graph.get("external_outputs") or [])
        if isinstance(item, Mapping)
    )
    boundary_sources.discard("")
    for section in ("conditioning", "signals"):
        for row in list(contract.get(section) or []):
            source = str((row or {}).get("source") or "")
            if _declared_source_resolves(source, nodes, boundary_sources):
                continue
            raise TrainingContractError(
                "Training contract %s source %s does not resolve to a scenario output, "
                "boundary, or recipe parameter"
                % (section, source)
            )


def _declared_source_resolves(
    source: str,
    nodes: Mapping[str, Mapping[str, Any]],
    boundary_sources: set[str],
) -> bool:
    if source in boundary_sources:
        return True
    parts = str(source or "").split(".")
    if len(parts) < 2 or parts[0] not in nodes:
        return False
    node = nodes[parts[0]]
    if parts[1] != "params":
        return len(parts) == 2 and parts[1] in dict(node.get("outputs") or {})
    value: Any = node.get("params") or {}
    for key in parts[2:]:
        if not isinstance(value, Mapping) or key not in value:
            return False
        value = value[key]
    return True


def _validate_artifact_return_bindings(contract: Mapping[str, Any]) -> None:
    slots = {
        str(slot.get("step_id") or ""): slot
        for slot in list(contract.get("trainable_slots") or [])
        if isinstance(slot, Mapping)
    }
    bindings = list((contract.get("artifact_return") or {}).get("bindings") or [])
    bound_steps: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        if not isinstance(raw_binding, Mapping):
            raise TrainingContractError(
                "artifact_return.bindings[%d] must be a mapping" % index
            )
        step_id = str(raw_binding.get("step_id") or "")
        slot = slots.get(step_id)
        if slot is None or step_id in bound_steps:
            raise TrainingContractError(
                "Artifact return bindings must reference each trainable slot exactly once"
            )
        bound_steps.add(step_id)
        required_inputs = sorted(str(item) for item in raw_binding.get("required_inputs") or [])
        unknown = sorted(set(required_inputs) - set((slot.get("inputs") or {}).keys()))
        if unknown:
            raise TrainingContractError(
                "Artifact binding %s requires unknown operation input(s): %s"
                % (step_id, ", ".join(unknown))
            )
        runtime_abi = dict(slot.get("runtime_artifact_abi") or {})
        expected = sorted(
            str(item) for item in runtime_abi.get("required_operation_inputs") or []
        )
        if not expected:
            expected = sorted(
                str(name)
                for name, tensor in dict(slot.get("inputs") or {}).items()
                if bool((tensor or {}).get("required", False))
            )
        if required_inputs != expected:
            raise TrainingContractError(
                "Artifact binding %s required inputs disagree with its runtime ABI" % step_id
            )
    if bound_steps != set(slots):
        raise TrainingContractError(
            "Artifact return bindings must cover every trainable slot"
        )


def write_training_contract_bundle(
    compiled: CompiledTrainingContract,
    out_dir: Path,
    *,
    force: bool = False,
) -> JsonDict:
    """Write a neutral interface bundle without supplying a model, loss, optimizer, or trainer."""

    validate_compiled_training_contract(compiled.contract, compiled.scenario_graph)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise TrainingContractError(
            "Training contract output directory is not empty: %s. Use force to overwrite contract-owned files."
            % out_dir
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = out_dir / "test_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    contract_path = out_dir / "training_contract.yaml"
    graph_path = out_dir / "scenario_graph.json"
    recipe_path = out_dir / "noema_recipe.yaml"
    _write_yaml(contract_path, compiled.contract)
    _write_json(graph_path, compiled.scenario_graph)
    _write_yaml(recipe_path, compiled.source_recipe)
    contract_file_sha256 = _file_sha256(contract_path)
    scenario_graph_file_sha256 = _file_sha256(graph_path)

    vector_payload = _contract_test_vectors(
        compiled.contract,
        compiled.source_recipe,
    )
    vector_path = vectors_dir / "contract_vectors.json"
    _write_json(vector_path, vector_payload)
    artifact_template_path = out_dir / "trained_artifact.template.yaml"
    _write_yaml(
        artifact_template_path,
        _trained_artifact_template(
            compiled.contract,
            contract_sha256=compiled.contract_sha256,
            contract_file_sha256=contract_file_sha256,
        ),
    )
    _write_text(
        out_dir / "interfaces.py",
        _interfaces_py(
            compiled.contract,
            compiled.source_recipe,
        ),
    )
    write_standalone_structured_input(out_dir)
    _write_text(out_dir / "validate_contract.py", _validator_py())
    _write_text(out_dir / "package_artifact.py", _package_artifact_py())
    _write_text(
        out_dir / "README.md",
        _bundle_readme(
            compiled.contract,
            compiled.source_recipe,
        ),
    )

    manifest: JsonDict = {
        "schema_version": 1,
        "kind": TRAINING_INTERFACE_PROJECT_KIND,
        "id": str(compiled.contract.get("id") or "training_interface_bundle"),
        "out_dir": str(out_dir.resolve()),
        "source_recipe": dict(compiled.contract["source_recipe"]),
        "contracts": {
            "trainable_slots": {
                "path": "training_contract.yaml",
                "kind": TRAINABLE_SLOT_CONTRACT_KIND,
                "sha256": compiled.contract_sha256,
                "file_sha256": contract_file_sha256,
            },
            "scenario_graph": {
                "path": "scenario_graph.json",
                "kind": TYPED_SCENARIO_GRAPH_KIND,
                "sha256": compiled.scenario_graph_sha256,
                "file_sha256": scenario_graph_file_sha256,
            },
        },
        "external_training": {
            "owner": "researcher",
            "architecture": "not_supplied",
            "loss": "not_supplied",
            "trainer": "not_supplied",
            "optional_demo_scaffold": None,
        },
        "capture_jobs": [],
        "training": {
            "owner": "external_researcher",
            "working_directory": str(out_dir.resolve()),
            "trainer_included": False,
            "command_status": "not_supplied",
            "instructions": "README.md",
        },
        "evaluation": {
            "owner": "noema_ordinary_recipe_or_benchmark",
        },
        "trained_artifacts": [
            {
                "role": "returned_slot_implementation",
                "operations": [
                    str(item.get("operation") or "")
                    for item in list(
                        (compiled.contract.get("artifact_return") or {}).get(
                            "bindings"
                        )
                        or []
                    )
                ],
                "step_ids": [
                    str(item.get("step_id") or "")
                    for item in list(
                        (compiled.contract.get("artifact_return") or {}).get(
                            "bindings"
                        )
                        or []
                    )
                ],
                "manifest_path": str((out_dir / "trained_artifact.yaml").resolve()),
            }
        ],
        "helpers": {
            "interfaces": "interfaces.py",
            "validate": "python validate_contract.py",
            "artifact_package_request": "python package_artifact.py <artifact> --format <format> --runtime <runtime>",
            "onnx_artifact_template": "trained_artifact.template.yaml",
            "test_vectors": "test_vectors/contract_vectors.json",
        },
        "artifact_return": dict(compiled.contract.get("artifact_return") or {}),
    }
    manifest_path = out_dir / "project_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    files = [
        "training_contract.yaml",
        "scenario_graph.json",
        "noema_recipe.yaml",
        "project_manifest.yaml",
        "interfaces.py",
        "structured_input.py",
        "validate_contract.py",
        "package_artifact.py",
        "test_vectors/contract_vectors.json",
        "trained_artifact.template.yaml",
        "README.md",
    ]
    return {
        "status": "exported",
        "kind": TRAINING_INTERFACE_PROJECT_KIND,
        "out_dir": str(out_dir),
        "contract_id": str(compiled.contract.get("id") or ""),
        "contract_sha256": compiled.contract_sha256,
        "contract_file_sha256": contract_file_sha256,
        "scenario_graph_sha256": compiled.scenario_graph_sha256,
        "scenario_graph_file_sha256": scenario_graph_file_sha256,
        "project_manifest": manifest,
        "files": files,
    }


def _resolve_port_tensor_specs(
    recipe: Recipe,
    *,
    operation_descriptions: Mapping[str, JsonDict],
    tensor_overrides: Mapping[str, Mapping[str, Any] | TensorSpec],
) -> Dict[Tuple[str, str], TensorSpec]:
    """Resolve one canonical tensor spec for every producer port and all of its edges."""

    consumers = _consumer_references(recipe.steps)
    resolved: Dict[Tuple[str, str], TensorSpec] = {}
    for step in recipe.steps:
        output_kinds = dict(operation_descriptions[step.id].get("output_kinds") or {})
        for output_name, output_kind in output_kinds.items():
            reference = "%s.%s" % (step.id, output_name)
            override_items: List[Tuple[str, Mapping[str, Any] | TensorSpec]] = []
            source_key = "%s.outputs.%s" % (step.id, output_name)
            if source_key in tensor_overrides:
                override_items.append((source_key, tensor_overrides[source_key]))
            for consumer_reference in consumers.get(reference, []):
                consumer_id, input_name = consumer_reference.split(".", 1)
                target_key = "%s.inputs.%s" % (consumer_id, input_name)
                if target_key in tensor_overrides:
                    override_items.append((target_key, tensor_overrides[target_key]))
            candidates = [
                (key, tensor_spec_for_kind(str(output_kind), override))
                for key, override in override_items
            ]
            if candidates:
                selected_key, selected_spec = candidates[0]
                for candidate_key, candidate_spec in candidates[1:]:
                    if candidate_spec.to_dict() != selected_spec.to_dict():
                        raise TrainingContractError(
                            "Connected tensor overrides %s and %s disagree for %s"
                            % (selected_key, candidate_key, reference)
                        )
                resolved[(step.id, output_name)] = selected_spec
            else:
                resolved[(step.id, output_name)] = tensor_spec_for_kind(str(output_kind))
    return resolved


def _compile_typed_scenario_graph(
    recipe: Recipe,
    *,
    steps_by_id: Mapping[str, RecipeStep],
    operation_descriptions: Mapping[str, JsonDict],
    scenario_step_ids: Sequence[str],
    trainable_steps: Sequence[str],
    tensor_overrides: Mapping[str, Mapping[str, Any] | TensorSpec],
    port_tensor_specs: Mapping[Tuple[str, str], TensorSpec],
    framework: str,
    recipe_sha: str,
    execution_profile: JsonDict,
    declared_boundaries: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    included = set(scenario_step_ids)
    selected = set(trainable_steps)
    nodes = []
    edges = []
    external_inputs = []
    external_outputs = []
    consumer_references = _consumer_references(recipe.steps)
    for step_id in scenario_step_ids:
        step = steps_by_id[step_id]
        operation = operation_descriptions[step_id]
        differentiability = dict(operation.get("differentiability") or {})
        role = _graph_node_role(step_id, selected, differentiability)
        inputs: JsonDict = {}
        required_inputs = dict(operation.get("input_kinds") or {})
        optional_inputs = dict(operation.get("optional_input_kinds") or {})
        for input_name, accepted in {**required_inputs, **optional_inputs}.items():
            reference = step.inputs.get(input_name)
            actual_kind = _input_actual_kind(
                reference,
                operation_descriptions=operation_descriptions,
                accepted=accepted,
            )
            if reference:
                producer_id, output_name = reference.split(".", 1)
                spec = port_tensor_specs[(producer_id, output_name)]
            else:
                spec = tensor_spec_for_kind(
                    actual_kind,
                    tensor_overrides.get("%s.inputs.%s" % (step_id, input_name)),
                )
            inputs[input_name] = {
                "required": input_name in required_inputs,
                "accepted_kinds": list(accepted or []),
                "source": reference,
                "tensor": spec.to_dict(),
            }
            if reference:
                producer_id, output_name = reference.split(".", 1)
                if producer_id in included:
                    edges.append(
                        {
                            "id": "%s.%s->%s.%s" % (producer_id, output_name, step_id, input_name),
                            "source": {"step_id": producer_id, "port": output_name},
                            "target": {"step_id": step_id, "port": input_name},
                            "tensor": spec.to_dict(),
                        }
                    )
                else:
                    external_inputs.append(
                        {
                            "id": "%s.%s" % (step_id, input_name),
                            "recipe_reference": reference,
                            "target": {"step_id": step_id, "port": input_name},
                            "tensor": spec.to_dict(),
                        }
                    )
        outputs = {
            output_name: port_tensor_specs[(step_id, output_name)].to_dict()
            for output_name in dict(operation.get("output_kinds") or {})
        }
        for output_name, tensor in outputs.items():
            reference = "%s.%s" % (step_id, output_name)
            all_consumers = list(consumer_references.get(reference) or [])
            outside_consumers = [
                consumer
                for consumer in all_consumers
                if consumer.split(".", 1)[0] not in included
            ]
            if outside_consumers or not all_consumers:
                external_outputs.append(
                    {
                        "id": reference,
                        "source": {"step_id": step_id, "port": output_name},
                        "recipe_consumers": outside_consumers,
                        "tensor": dict(tensor),
                    }
                )
        placeholder = role == "trainable_placeholder"
        node_differentiability = (
            {
                "applicable": False,
                "reason": (
                    "The recipe's current implementation is not materialized at a selected "
                    "replacement boundary; the researcher-supplied module defines autograd behavior."
                ),
            }
            if placeholder
            else differentiability
        )
        nodes.append(
            {
                "id": step.id,
                "operation": step.op,
                "role": role,
                "boundary_kind": "portable_replacement" if placeholder else "frozen_support",
                "implementation_owner": "external" if placeholder else "noema",
                "parameters_trainable": placeholder,
                # Runtime parameters belong to the replaced implementation and
                # remain in noema_recipe.yaml for provenance. They must not be
                # presented as parameters of the researcher-supplied module.
                "params": {} if placeholder else dict(step.params),
                "inputs": inputs,
                "outputs": outputs,
                "differentiability": node_differentiability,
                "materialization": _selected_materialization(operation, framework, role),
            }
        )
    existing_boundaries = {
        str(item.get("recipe_reference") or item.get("id") or "")
        for item in external_inputs
    }
    for declaration in declared_boundaries:
        source = str(declaration.get("source") or "").strip()
        source_step = source.split(".", 1)[0]
        source_suffix = source.split(".", 1)[1] if "." in source else ""
        source_operation = operation_descriptions.get(source_step, {})
        source_recipe_step = steps_by_id.get(source_step)
        source_is_recipe_output = source_suffix in dict(
            source_operation.get("output_kinds") or {}
        )
        source_is_recipe_parameter = (
            source_recipe_step is not None
            and (
                source_suffix == "params"
                or (
                    source_suffix.startswith("params.")
                    and source_suffix.split(".", 1)[1]
                    in dict(source_recipe_step.params)
                )
            )
        )
        # Do not turn a typo or an operation input (for example
        # sender.images) into an apparently valid external source. The
        # cross-reference validator below will report it precisely.
        if not source_is_recipe_output and not source_is_recipe_parameter:
            continue
        source_is_included_output = (
            source_step in included
            and source_is_recipe_output
        )
        source_is_frozen_parameter = (
            source_step in included
            and source_step not in selected
            and (
                source_suffix == "params"
                or source_suffix.startswith("params.")
            )
        )
        if (
            not source
            or source in existing_boundaries
            or source_is_included_output
            or source_is_frozen_parameter
        ):
            continue
        external_inputs.append(
            {
                "id": source,
                "recipe_reference": source,
                "purpose": str(declaration.get("purpose") or "declared_training_source"),
                "declaration_id": str(declaration.get("id") or ""),
                "tensor": dict(
                    declaration.get("tensor")
                    or _tensor_for_recipe_reference(source, port_tensor_specs)
                    or tensor_spec_for_kind("operation_defined").to_dict()
                ),
            }
        )
        existing_boundaries.add(source)
    return {
        "schema_version": 1,
        "kind": TYPED_SCENARIO_GRAPH_KIND,
        "source_recipe": {
            "name": recipe.name,
            "sha256": recipe_sha,
            "execution_profile": recipe.execution_profile.to_dict(),
            "execution_profile_status": execution_profile.get("status"),
        },
        "framework": framework,
        "topological_order": list(scenario_step_ids),
        "nodes": nodes,
        "edges": edges,
        "external_inputs": external_inputs,
        "external_outputs": external_outputs,
        "semantics": {
            "trainable_placeholder": (
                "Portable replacement boundary. The source recipe operation is omitted; architecture "
                "and autograd behavior are supplied by the external researcher."
            ),
            "frozen_differentiable": "Noema-owned differentiable materialization; parameters are fixed.",
            "frozen_context": "Noema-owned context or observation node; no gradient promise is made.",
        },
    }


def _compile_slots(
    recipe: Recipe,
    *,
    trainable_steps: Sequence[str],
    steps_by_id: Mapping[str, RecipeStep],
    operation_descriptions: Mapping[str, JsonDict],
    tensor_overrides: Mapping[str, Mapping[str, Any] | TensorSpec],
    port_tensor_specs: Mapping[Tuple[str, str], TensorSpec],
    slot_roles: Mapping[str, str],
) -> List[JsonDict]:
    consumers = _consumer_references(recipe.steps)
    slots: List[JsonDict] = []
    for step_id in trainable_steps:
        step = steps_by_id[step_id]
        operation = operation_descriptions[step_id]
        required_inputs = dict(operation.get("input_kinds") or {})
        optional_inputs = dict(operation.get("optional_input_kinds") or {})
        inputs: JsonDict = {}
        for input_name, accepted in {**required_inputs, **optional_inputs}.items():
            reference = step.inputs.get(input_name)
            actual_kind = _input_actual_kind(
                reference,
                operation_descriptions=operation_descriptions,
                accepted=accepted,
            )
            if reference:
                producer_id, output_name = reference.split(".", 1)
                spec = port_tensor_specs[(producer_id, output_name)].to_dict()
            else:
                spec = tensor_spec_for_kind(
                    actual_kind,
                    tensor_overrides.get("%s.inputs.%s" % (step_id, input_name)),
                ).to_dict()
            spec["required"] = input_name in required_inputs
            if reference:
                spec["recipe_reference"] = reference
            spec["accepted_kinds"] = list(accepted or [])
            inputs[input_name] = spec
        outputs: JsonDict = {}
        for output_name in dict(operation.get("output_kinds") or {}):
            spec = port_tensor_specs[(step_id, output_name)].to_dict()
            spec["recipe_consumers"] = consumers.get("%s.%s" % (step_id, output_name), [])
            outputs[output_name] = spec
        slots.append(
            {
                "id": step_id,
                "step_id": step_id,
                "operation": step.op,
                "role": str(slot_roles.get(step_id) or _infer_slot_role(step)),
                "implementation_owner": "external_researcher",
                "architecture": "unspecified",
                "inputs": inputs,
                "outputs": outputs,
                "runtime_artifact_abi": dict(
                    operation.get("trained_artifact_abi") or {}
                ),
            }
        )
    return slots


def _compile_groups(
    trainable_steps: Sequence[str],
    raw_groups: Sequence[SlotGroupSpec | Mapping[str, Any]],
) -> List[JsonDict]:
    if raw_groups:
        groups = [_group_from_value(item) for item in raw_groups]
    else:
        groups = [
            SlotGroupSpec(
                id="joint_trainable_group" if len(trainable_steps) > 1 else "%s_group" % trainable_steps[0],
                slots=tuple(trainable_steps),
                joint_training=len(trainable_steps) > 1,
                atomic_artifact_return=len(trainable_steps) > 1,
            )
        ]
    known = set(trainable_steps)
    seen: set[str] = set()
    covered: set[str] = set()
    payload = []
    for group in groups:
        if not group.id or group.id in seen:
            raise TrainingContractError("Slot group IDs must be non-empty and unique")
        seen.add(group.id)
        members = set(group.slots)
        if not members or not members.issubset(known):
            raise TrainingContractError("Slot group %s contains unknown slots" % group.id)
        if covered.intersection(members):
            raise TrainingContractError("Trainable slots may not appear in more than one slot group")
        covered.update(members)
        payload.append(group.to_dict())
    if covered != known:
        raise TrainingContractError(
            "Slot groups do not cover trainable slots: %s" % ", ".join(sorted(known - covered))
        )
    return payload


def _artifact_return_bindings(
    slots: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> List[JsonDict]:
    group_by_slot = {
        str(step_id): group
        for group in groups
        for step_id in list(group.get("slots") or [])
    }
    bindings = []
    for slot in slots:
        step_id = str(slot["step_id"])
        group = group_by_slot[step_id]
        runtime_abi = dict(slot.get("runtime_artifact_abi") or {})
        required_inputs = list(runtime_abi.get("required_operation_inputs") or [])
        if not required_inputs:
            required_inputs = [
                str(name)
                for name, tensor in dict(slot.get("inputs") or {}).items()
                if bool((tensor or {}).get("required", False))
            ]
        bindings.append(
            {
                "step_id": step_id,
                "operation": slot["operation"],
                "role": slot["role"],
                "required_inputs": sorted(required_inputs),
                "binding_group": str(group.get("id") or ""),
                "application": str(group.get("artifact_application") or "independent_bindings"),
            }
        )
    return bindings


def _compile_signals(
    slots: Sequence[Mapping[str, Any]],
    raw_signals: Sequence[SignalSpec | Mapping[str, Any]],
) -> List[JsonDict]:
    signals: List[SignalSpec] = []
    for slot in slots:
        step_id = str(slot["step_id"])
        for port, tensor in dict(slot.get("inputs") or {}).items():
            signals.append(
                SignalSpec(
                    id="%s.input.%s" % (step_id, port),
                    source=str(tensor.get("recipe_reference") or "%s.%s" % (step_id, port)),
                    tensor=_tensor_from_mapping(tensor),
                    purpose="model_input",
                )
            )
        for port, tensor in dict(slot.get("outputs") or {}).items():
            signals.append(
                SignalSpec(
                    id="%s.output.%s" % (step_id, port),
                    source="%s.%s" % (step_id, port),
                    tensor=_tensor_from_mapping(tensor),
                    purpose="model_output_or_loss_signal",
                )
            )
    signals.extend(_signal_from_value(item) for item in raw_signals)
    seen: set[str] = set()
    payload = []
    for signal in signals:
        if not signal.id or not signal.source or signal.id in seen:
            raise TrainingContractError("Signal IDs must be non-empty and unique: %s" % signal.id)
        seen.add(signal.id)
        payload.append(signal.to_dict())
    return payload


def _compile_conditioning(
    raw_values: Sequence[NamedValueSpec | Mapping[str, Any]],
) -> List[JsonDict]:
    values = [_named_value_from_value(item) for item in raw_values]
    seen: set[str] = set()
    payload = []
    for value in values:
        if not value.id or not value.source or value.id in seen:
            raise TrainingContractError(
                "Conditioning IDs and sources must be non-empty, and IDs must be unique: %s"
                % value.id
            )
        seen.add(value.id)
        payload.append(value.to_dict())
    return payload


def _compile_constraints(
    raw_values: Sequence[ConstraintSpec | Mapping[str, Any]],
) -> List[JsonDict]:
    values = [_constraint_from_value(item) for item in raw_values]
    seen: set[str] = set()
    payload = []
    for value in values:
        if (
            not value.id
            or not value.expression
            or not value.enforcement
            or value.id in seen
        ):
            raise TrainingContractError(
                "Constraint IDs, expressions, and enforcement must be non-empty, and IDs must be unique: %s"
                % value.id
            )
        seen.add(value.id)
        payload.append(value.to_dict())
    return payload


def _input_actual_kind(
    reference: Optional[str],
    *,
    operation_descriptions: Mapping[str, JsonDict],
    accepted: Sequence[str],
) -> str:
    if reference:
        producer_id, output_name = reference.split(".", 1)
        return str((operation_descriptions[producer_id].get("output_kinds") or {})[output_name])
    if len(accepted) == 1:
        return str(accepted[0])
    return str(accepted[0]) if accepted else "operation_defined"


def _graph_node_role(step_id: str, selected: set[str], differentiability: Mapping[str, Any]) -> str:
    if step_id in selected:
        return "trainable_placeholder"
    if (
        bool(differentiability.get("exportable", False))
        and str(differentiability.get("gradient") or "") in {"full", "surrogate"}
    ):
        return "frozen_differentiable"
    return "frozen_context"


def _selected_materialization(operation: Mapping[str, Any], framework: str, role: str) -> JsonDict:
    if role == "trainable_placeholder":
        return {
            "runner": "external_training",
            "backend": framework,
            "implementation": "researcher_supplied_slot",
            "status": "required",
        }
    candidates = [
        dict(item)
        for item in list(operation.get("materializations") or [])
        if isinstance(item, Mapping)
        and str(item.get("runner") or "") == "differentiable_export"
    ]
    normalized_framework = str(framework or "").strip().lower().replace("_", "-")
    # ``torch-sionna`` names the researcher's combined training stack. The
    # operation materialization registry uses the concrete implementation
    # backend name ``sionna`` for its PyTorch-native PHY blocks. Keep the full
    # framework label on the graph and external placeholders, but resolve
    # frozen Noema support through the Sionna materialization rather than
    # silently falling back to the first native-Torch candidate.
    compatible_backends = (
        ("sionna", "torch")
        if normalized_framework == "torch-sionna"
        else (normalized_framework,)
    )
    exact = next(
        (
            item
            for backend in compatible_backends
            for item in candidates
            if str(item.get("backend") or "") == backend
        ),
        None,
    )
    if exact is not None:
        return exact
    if candidates:
        return {**candidates[0], "selected_for_framework": False}
    return {
        "runner": "differentiable_export",
        "backend": framework,
        "implementation": "not_materialized",
        "status": "context_only",
    }


def _consumer_references(steps: Iterable[RecipeStep]) -> Dict[str, List[str]]:
    consumers: Dict[str, List[str]] = {}
    for step in steps:
        for input_name, reference in step.inputs.items():
            consumers.setdefault(reference, []).append("%s.%s" % (step.id, input_name))
    return consumers


def _infer_slot_role(step: RecipeStep) -> str:
    value = "%s %s" % (step.id.lower(), step.op.lower())
    if "encode" in value or "sender" in value:
        return "encoder"
    if "decode" in value or "receiver" in value:
        return "decoder"
    if "allocat" in value or "tx_power" in value:
        return "allocator"
    if "estimat" in value:
        return "estimator"
    return "trainable_module"


def _group_from_value(value: SlotGroupSpec | Mapping[str, Any]) -> SlotGroupSpec:
    if isinstance(value, SlotGroupSpec):
        return value
    if not isinstance(value, Mapping):
        raise TrainingContractError("Slot groups must be mappings or SlotGroupSpec values")
    slots = value.get("slots") or []
    if not isinstance(slots, (list, tuple)):
        raise TrainingContractError("Slot group slots must be a list")
    application = str(value.get("artifact_application") or "")
    return SlotGroupSpec(
        id=str(value.get("id") or ""),
        slots=tuple(str(item) for item in slots),
        joint_training=bool(value.get("joint_training", True)),
        atomic_artifact_return=bool(
            value.get("atomic_artifact_return", application == "all_group_bindings")
        ),
        description=str(value.get("description") or ""),
    )


def _named_value_from_value(value: NamedValueSpec | Mapping[str, Any]) -> NamedValueSpec:
    if isinstance(value, NamedValueSpec):
        return value
    if not isinstance(value, Mapping):
        raise TrainingContractError("Conditioning values must be mappings or NamedValueSpec values")
    shape = value.get("shape") or []
    if not isinstance(shape, (list, tuple)):
        raise TrainingContractError("Conditioning shape must be a list")
    return NamedValueSpec(
        id=str(value.get("id") or ""),
        source=str(value.get("source") or ""),
        dtype=str(value.get("dtype") or "float32"),
        shape=tuple(shape),
        units=str(value.get("units") or "unitless"),
        description=str(value.get("description") or ""),
        required=bool(value.get("required", True)),
    )


def _signal_from_value(value: SignalSpec | Mapping[str, Any]) -> SignalSpec:
    if isinstance(value, SignalSpec):
        return value
    if not isinstance(value, Mapping):
        raise TrainingContractError("Signals must be mappings or SignalSpec values")
    tensor_value = value.get("tensor")
    if isinstance(tensor_value, TensorSpec):
        tensor = tensor_value
    elif isinstance(tensor_value, Mapping):
        tensor = _tensor_from_mapping(tensor_value)
    else:
        kind = str(value.get("kind") or "operation_defined")
        tensor = tensor_spec_for_kind(kind)
    return SignalSpec(
        id=str(value.get("id") or ""),
        source=str(value.get("source") or ""),
        tensor=tensor,
        purpose=str(value.get("purpose") or "loss_or_diagnostics"),
        description=str(value.get("description") or ""),
    )


def _constraint_from_value(value: ConstraintSpec | Mapping[str, Any]) -> ConstraintSpec:
    if isinstance(value, ConstraintSpec):
        return value
    if not isinstance(value, Mapping):
        raise TrainingContractError("Constraints must be mappings or ConstraintSpec values")
    return ConstraintSpec(
        id=str(value.get("id") or ""),
        expression=str(value.get("expression") or ""),
        enforcement=str(value.get("enforcement") or "external_loss_or_model"),
        scope=str(value.get("scope") or "per_sample"),
        description=str(value.get("description") or ""),
    )


def _tensor_from_mapping(value: Mapping[str, Any]) -> TensorSpec:
    shape = value.get("shape") or []
    return TensorSpec(
        kind=str(value.get("kind") or "operation_defined"),
        dtype=str(value.get("dtype") or "operation_defined"),
        shape=tuple(shape) if isinstance(shape, (list, tuple)) else ("...",),
        layout=str(value.get("layout") or "operation_defined"),
        units=str(value.get("units") or "unitless"),
        domain=str(value.get("domain") or "operation_defined"),
        description=str(value.get("description") or ""),
    )


def _validate_tensor_mapping(value: Mapping[str, Any], label: str) -> None:
    for key in ("kind", "dtype", "shape", "layout", "units", "domain"):
        if key not in value:
            raise TrainingContractError("Tensor spec %s requires %s" % (label, key))
    if not isinstance(value.get("shape"), list):
        raise TrainingContractError("Tensor spec %s shape must be a list" % label)


def _unique_nonempty(values: Sequence[str], label: str) -> Tuple[str, ...]:
    result = []
    seen = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise TrainingContractError("%s must not contain empty step IDs" % label)
        if value not in seen:
            result.append(value)
            seen.add(value)
    if not result:
        raise TrainingContractError("%s must name at least one step" % label)
    return tuple(result)


def _training_source_values(values: Sequence[Any]) -> List[str]:
    sources: List[str] = []
    for value in values:
        if isinstance(value, Mapping):
            source = value.get("source")
        else:
            source = getattr(value, "source", "")
        normalized = str(source or "").strip()
        if normalized and normalized not in sources:
            sources.append(normalized)
    return sources


def _recipe_loss_input_signals(
    recipe: Recipe,
    loss_steps: Sequence[str],
    *,
    port_tensor_specs: Mapping[Tuple[str, str], TensorSpec],
    existing_signals: Sequence[SignalSpec | Mapping[str, Any]],
) -> List[SignalSpec]:
    """Turn recipe metric operands into typed, external-loss boundaries.

    The metric/loss operation is intentionally not materialized. Its inputs
    identify both the downstream model signal and any reference/target tensor
    needed by a researcher-defined loss.
    """

    steps_by_id = {step.id: step for step in recipe.steps}
    existing_sources = {
        str(_signal_from_value(value).source or "").strip()
        for value in existing_signals
    }
    result: List[SignalSpec] = []
    for loss_step_id in loss_steps:
        loss_step = steps_by_id[loss_step_id]
        if not loss_step.inputs:
            raise TrainingContractError(
                "Recipe loss step %s has no typed input operands" % loss_step_id
            )
        for input_name, raw_reference in loss_step.inputs.items():
            reference = str(raw_reference or "").strip()
            if not reference or reference in existing_sources:
                continue
            producer_id, output_name = reference.split(".", 1)
            tensor = port_tensor_specs.get((producer_id, output_name))
            if tensor is None:
                raise TrainingContractError(
                    "Recipe loss input %s.%s references an unknown output %s"
                    % (loss_step_id, input_name, reference)
                )
            result.append(
                SignalSpec(
                    id="recipe_loss.%s.%s" % (loss_step_id, input_name),
                    source=reference,
                    tensor=tensor,
                    purpose="external_loss_input",
                    description=(
                        "Typed operand consumed by recipe evaluation step %s; "
                        "the external researcher defines the training loss."
                        % loss_step_id
                    ),
                )
            )
            existing_sources.add(reference)
    return result


def _declared_training_boundaries(
    conditioning: Sequence[NamedValueSpec | Mapping[str, Any]],
    signals: Sequence[SignalSpec | Mapping[str, Any]],
) -> List[JsonDict]:
    """Preserve explicit training-source types when a producer is omitted.

    Signals take precedence over scalar conditioning declarations when both
    happen to name the same recipe reference.
    """

    boundaries: List[JsonDict] = []
    seen: set[str] = set()
    for raw_signal in signals:
        signal = _signal_from_value(raw_signal)
        source = str(signal.source or "").strip()
        if not source or source in seen:
            continue
        boundaries.append(
            {
                "id": signal.id,
                "source": source,
                "purpose": signal.purpose,
                "tensor": signal.tensor.to_dict(),
            }
        )
        seen.add(source)
    for raw_value in conditioning:
        value = _named_value_from_value(raw_value)
        source = str(value.source or "").strip()
        if not source or source in seen:
            continue
        boundaries.append(
            {
                "id": value.id,
                "source": source,
                "purpose": "conditioning",
                "tensor": TensorSpec(
                    kind="training.conditioning.value",
                    dtype=value.dtype,
                    shape=value.shape,
                    layout="scalar" if not value.shape else "declared",
                    units=value.units,
                    domain="declared_conditioning",
                    description=value.description,
                ).to_dict(),
            }
        )
        seen.add(source)
    return boundaries


def _tensor_for_recipe_reference(
    source: str,
    port_tensor_specs: Mapping[Tuple[str, str], TensorSpec],
) -> Optional[JsonDict]:
    if "." not in source:
        return None
    step_id, output_name = source.split(".", 1)
    spec = port_tensor_specs.get((step_id, output_name))
    return spec.to_dict() if spec is not None else None


def _minimal_replacement_scenario_steps(
    recipe: Recipe,
    replacement_steps: Sequence[str],
    *,
    signal_sources: Sequence[str],
    slot_groups: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    """Keep placeholders and only the frozen downstream routes they require.

    Producers upstream of a replacement become external/captured inputs. The
    selected block's original implementation is a placeholder, and only nodes
    between a replacement and a declared downstream signal are materialized.
    """

    ordered_ids = [step.id for step in recipe.steps]
    known = set(ordered_ids)
    selected = [step_id for step_id in replacement_steps if step_id in known]
    included = set(selected)
    consumers: Dict[str, List[str]] = {step_id: [] for step_id in ordered_ids}
    predecessors: Dict[str, List[str]] = {step_id: [] for step_id in ordered_ids}
    for step in recipe.steps:
        for reference in step.inputs.values():
            producer_id = str(reference).split(".", 1)[0]
            if producer_id in consumers and step.id not in consumers[producer_id]:
                consumers[producer_id].append(step.id)
                predecessors[step.id].append(producer_id)

    downstream_targets = {
        str(source).split(".", 1)[0]
        for source in signal_sources
        if str(source).split(".", 1)[0] in known
    }
    for start in selected:
        for target in downstream_targets:
            included.update(
                _nodes_on_any_scenario_path(start, target, consumers, predecessors)
            )
    # Only explicitly joint groups require paths between replacement slots.
    # Independent selected blocks otherwise remain independent boundaries.
    for group in slot_groups:
        if not bool(group.get("joint_training", False)):
            continue
        members = [str(item) for item in list(group.get("slots") or []) if str(item) in known]
        for start in members:
            for target in members:
                if start == target:
                    continue
                included.update(
                    _nodes_on_any_scenario_path(start, target, consumers, predecessors)
                )
    return tuple(step_id for step_id in ordered_ids if step_id in included)


def _nodes_on_any_scenario_path(
    start: str,
    target: str,
    consumers: Mapping[str, Sequence[str]],
    predecessors: Mapping[str, Sequence[str]],
) -> set[str]:
    forward = _scenario_reachable(start, consumers)
    if target not in forward:
        return set()
    reverse = _scenario_reachable(target, predecessors)
    return forward.intersection(reverse)


def _scenario_reachable(
    start: str,
    adjacency: Mapping[str, Sequence[str]],
) -> set[str]:
    pending = [start]
    visited: set[str] = set()
    while pending:
        step_id = pending.pop()
        if step_id in visited:
            continue
        visited.add(step_id)
        pending.extend(
            child for child in adjacency.get(step_id, ()) if child not in visited
        )
    return visited


def _validate_scenario_topological_order(
    recipe: Recipe,
    scenario_step_ids: Sequence[str],
) -> None:
    positions = {step_id: index for index, step_id in enumerate(scenario_step_ids)}
    for step in recipe.steps:
        if step.id not in positions:
            continue
        for reference in step.inputs.values():
            producer_id = str(reference).split(".", 1)[0]
            if producer_id not in positions:
                continue
            if positions[producer_id] >= positions[step.id]:
                raise TrainingContractError(
                    "Scenario steps are not topologically ordered: %s must precede %s"
                    % (producer_id, step.id)
                )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "training_interface"


def _normalized_runtime_tensors(values: Mapping[str, Any]) -> JsonDict:
    """Return the deployed entrypoint tensor ABI keyed by its actual tensor names."""

    normalized: JsonDict = {}
    for name, raw_spec in values.items():
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {}
        normalized[str(name)] = {
            "dtype": str(spec.get("dtype") or "float32"),
            "shape": list(spec.get("shape") or ["..."]),
            "semantic": str(spec.get("semantic") or str(name)),
            **({"layout": str(spec["layout"])} if spec.get("layout") else {}),
        }
    return normalized


def _identity_tensor_binding(
    operation_tensor: Mapping[str, Any],
    runtime_tensor: Mapping[str, Any],
) -> bool:
    """Return true only when an identity operation-to-runtime binding is provable."""

    if str(operation_tensor.get("dtype") or "") != str(runtime_tensor.get("dtype") or ""):
        return False
    if list(operation_tensor.get("shape") or []) != list(runtime_tensor.get("shape") or []):
        return False
    runtime_layout = str(runtime_tensor.get("layout") or "").strip()
    if runtime_layout and runtime_layout != str(operation_tensor.get("layout") or "").strip():
        return False
    return True


def _runtime_input_bindings(
    slot: Mapping[str, Any],
    *,
    recipe_params: Mapping[str, Any],
) -> JsonDict:
    """Describe only operation-to-runtime mappings justified by the operation ABI.

    ``required_operation_inputs`` says which block inputs the canonical Noema
    adapter consumes.  It does not, by itself, define how structured artifacts,
    metadata, or recipe parameters become runtime tensors.  Exact-name,
    tensor-compatible inputs are safe identity mappings; every other mapping is
    deliberately marked as adapter-owned rather than guessed here.
    """

    operation_inputs = dict(slot.get("inputs") or {})
    runtime_abi = dict(slot.get("runtime_artifact_abi") or {})
    runtime_inputs = _normalized_runtime_tensors(runtime_abi.get("inputs") or {})
    required_names = [
        str(item) for item in runtime_abi.get("required_operation_inputs") or []
    ]
    dependencies = []
    for name in required_names:
        operation_tensor = dict(operation_inputs.get(name) or {})
        dependency: JsonDict = {
            "kind": "operation_input",
            "name": name,
            "tensor": operation_tensor,
        }
        reference = str(operation_tensor.get("recipe_reference") or "")
        if reference:
            dependency["recipe_reference"] = reference
        dependencies.append(dependency)

    bindings: JsonDict = {}
    for runtime_name, runtime_tensor in runtime_inputs.items():
        operation_tensor = dict(operation_inputs.get(runtime_name) or {})
        if runtime_name in required_names and _identity_tensor_binding(
            operation_tensor,
            runtime_tensor,
        ):
            source: JsonDict = {
                "kind": "operation_input",
                "name": runtime_name,
            }
            reference = str(operation_tensor.get("recipe_reference") or "")
            if reference:
                source["recipe_reference"] = reference
            bindings[runtime_name] = {
                "resolution": "identity",
                "source": source,
                "runtime_tensor": runtime_tensor,
            }
            continue

        exact_param = runtime_name if runtime_name in recipe_params else ""
        bindings[runtime_name] = {
            "resolution": "operation_adapter_required",
            "source": None,
            "runtime_tensor": runtime_tensor,
            "operation_input_dependencies": dependencies,
            "recipe_parameter_candidate": exact_param or None,
            "reason": (
                "The operation ABI does not declare an identity-safe source for this runtime tensor. "
                "Use the canonical operation preprocessing or supply an explicitly prepared value."
            ),
        }
    return bindings


def _bundle_interface_descriptor(
    contract: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
) -> JsonDict:
    source_steps = {
        str(step.get("id") or ""): step
        for step in list(source_recipe.get("steps") or [])
        if isinstance(step, Mapping)
    }
    slots: JsonDict = {}
    for raw_slot in list(contract.get("trainable_slots") or []):
        slot = dict(raw_slot) if isinstance(raw_slot, Mapping) else {}
        step_id = str(slot.get("step_id") or "")
        recipe_params = dict((source_steps.get(step_id) or {}).get("params") or {})
        runtime_abi = dict(slot.get("runtime_artifact_abi") or {})
        operation_inputs = dict(slot.get("inputs") or {})
        slots[step_id] = {
            "step_id": step_id,
            "role": str(slot.get("role") or ""),
            "operation": str(slot.get("operation") or ""),
            "operation_boundary": {
                "required_inputs": [
                    str(name)
                    for name, tensor in operation_inputs.items()
                    if bool((tensor or {}).get("required", False))
                ],
                "artifact_adapter_inputs": [
                    str(item)
                    for item in runtime_abi.get("required_operation_inputs") or []
                ],
                "inputs": operation_inputs,
                "outputs": dict(slot.get("outputs") or {}),
            },
            "recipe_params": recipe_params,
            "runtime_artifact": {
                "component_id": str(runtime_abi.get("component_id") or step_id),
                "component_role": str(runtime_abi.get("component_role") or slot.get("role") or "model"),
                "entrypoint_id": str(runtime_abi.get("entrypoint_id") or ""),
                "required_operation_inputs": [
                    str(item)
                    for item in runtime_abi.get("required_operation_inputs") or []
                ],
                "inputs": _normalized_runtime_tensors(runtime_abi.get("inputs") or {}),
                "outputs": _normalized_runtime_tensors(runtime_abi.get("outputs") or {}),
                "input_bindings": _runtime_input_bindings(
                    slot,
                    recipe_params=recipe_params,
                ),
                "binding_params": dict(runtime_abi.get("binding_params") or {}),
            },
        }
    return {
        "schema_version": 1,
        "kind": "noema.generated_training_interfaces@1",
        "contract_id": str(contract.get("id") or ""),
        "slots": slots,
    }


def _contract_test_vectors(
    contract: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
) -> JsonDict:
    interfaces = _bundle_interface_descriptor(
        contract,
        source_recipe,
    )
    return {
        "schema_version": 1,
        "kind": "noema.training_contract_symbolic_test_vectors@1",
        "contract_id": str(contract.get("id") or ""),
        "source_recipe_sha256": str((contract.get("source_recipe") or {}).get("sha256") or ""),
        "note": (
            "Operation-boundary tensors describe recipe connections and captured values. "
            "Runtime-artifact tensors describe the exact returned-model entrypoint. Shapes are symbolic."
        ),
        "slots": [
            {
                "step_id": slot.get("step_id"),
                "role": slot.get("role"),
                # Compatibility aliases. These are operation-boundary tensors,
                # not necessarily the deployed model's tensor names or shapes.
                "inputs": dict(slot.get("inputs") or {}),
                "expected_outputs": dict(slot.get("outputs") or {}),
                "operation_boundary": dict(
                    interfaces["slots"][str(slot.get("step_id") or "")][
                        "operation_boundary"
                    ]
                ),
                "runtime_artifact": dict(
                    interfaces["slots"][str(slot.get("step_id") or "")][
                        "runtime_artifact"
                    ]
                ),
            }
            for slot in list(contract.get("trainable_slots") or [])
        ],
        "conditioning": list(contract.get("conditioning") or []),
        "constraints": list(contract.get("constraints") or []),
    }


def _trained_artifact_template(
    contract: Mapping[str, Any],
    *,
    contract_sha256: str,
    contract_file_sha256: str,
) -> JsonDict:
    """Create an ONNX-first return template from operation-owned runtime ABIs."""

    slots = {
        str(slot.get("step_id") or ""): slot
        for slot in list(contract.get("trainable_slots") or [])
    }
    contract_bindings = list((contract.get("artifact_return") or {}).get("bindings") or [])
    components: List[JsonDict] = []
    entrypoints: List[JsonDict] = []
    bindings: List[JsonDict] = []
    component_ids: set[str] = set()
    has_atomic_group = False
    for raw_binding in contract_bindings:
        binding = dict(raw_binding)
        step_id = str(binding.get("step_id") or "")
        slot = dict(slots.get(step_id) or {})
        abi = dict(slot.get("runtime_artifact_abi") or {})
        if not abi:
            continue
        component_id = str(abi.get("component_id") or step_id or "model")
        entrypoint_id = str(abi.get("entrypoint_id") or component_id)
        component_role = str(abi.get("component_role") or slot.get("role") or "model")
        if component_id not in component_ids:
            component_ids.add(component_id)
            components.append(
                {
                    "id": component_id,
                    "role": component_role,
                    "path": "artifacts/%s.onnx" % component_id,
                    "format": "onnx",
                    "sha256": "0" * 64,
                }
            )
        entrypoints.append(
            {
                "id": entrypoint_id,
                "component": component_id,
                "inputs": _runtime_abi_tensors(abi.get("inputs") or {}),
                "outputs": _runtime_abi_tensors(abi.get("outputs") or {}),
            }
        )
        application = str(binding.get("application") or "independent_bindings")
        has_atomic_group = has_atomic_group or application == "all_group_bindings"
        runtime_binding: JsonDict = {
            "operation": str(binding.get("operation") or slot.get("operation") or ""),
            "runtime_entrypoint": entrypoint_id,
            "preferred_step_id": step_id,
            "role": str(binding.get("role") or slot.get("role") or component_role),
            "required_inputs": list(abi.get("required_operation_inputs") or []),
            "params": dict(abi.get("binding_params") or {}),
        }
        group = str(binding.get("binding_group") or "")
        if group:
            runtime_binding["binding_group"] = group
        bindings.append(runtime_binding)
    return {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "replace_with_artifact_id",
        "name": "Replace with trained artifact name",
        "description": (
            "Template generated from the Noema slot contract. Replace component files, "
            "component SHA-256 values, id, and name. Training provenance is optional "
            "for runtime use; publication readiness separately requires a hash-bound, "
            "validation-only model-selection history."
        ),
        "contract": {
            "id": str(contract.get("id") or ""),
            "version": int(contract.get("version") or contract.get("schema_version") or 1),
            "path": "training_contract.yaml",
            "sha256": contract_sha256,
            "file_sha256": contract_file_sha256,
        },
        "components": components,
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": entrypoints,
        },
        "application": {
            "mode": "all_group_bindings" if has_atomic_group else "single_binding"
        },
        "compatible_operations": bindings,
        "source": {
            "origin": "external_training_against_noema_contract",
            "source_recipe_sha256": str(
                (contract.get("source_recipe") or {}).get("sha256") or ""
            ),
        },
    }


def _runtime_abi_tensors(values: Mapping[str, Any]) -> List[JsonDict]:
    tensors: List[JsonDict] = []
    for name, raw_spec in values.items():
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {}
        tensors.append(
            {
                "name": str(name),
                "dtype": str(spec.get("dtype") or "float32"),
                "shape": list(spec.get("shape") or ["..."]),
                "semantic": str(spec.get("semantic") or str(name)),
                **({"layout": str(spec["layout"])} if spec.get("layout") else {}),
            }
        )
    return tensors


def _interfaces_py(
    contract: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
) -> str:
    compact_contract = json.dumps(
        {
            "kind": contract.get("kind"),
            "id": contract.get("id"),
            "trainable_slots": contract.get("trainable_slots"),
            "slot_groups": contract.get("slot_groups"),
        },
        indent=2,
        sort_keys=True,
    )
    compact_interfaces = json.dumps(
        _bundle_interface_descriptor(contract, source_recipe),
        indent=2,
        sort_keys=True,
    )
    return '''
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol


class TrainableSlot(Protocol):
    """Architecture-neutral callable: keyword input ports -> mapping of output ports."""

    def __call__(self, **inputs: Any) -> Mapping[str, Any]:
        ...


# Generated from training_contract.yaml. These are the recipe/block boundaries.
# Captured datasets use these operation-side names and tensor semantics.
SLOT_CONTRACT = %s


# Generated from the operation-owned trained-artifact ABI. Returned model files
# must expose these runtime entrypoint names and tensor names/dtypes/shapes.
# They may differ from the operation boundary because Noema's operation adapter
# can unpack artifacts, convert layouts/dtypes, or derive runtime values from
# recipe parameters.
INTERFACE_CONTRACT = %s
OPERATION_BOUNDARIES = {
    step_id: slot["operation_boundary"]
    for step_id, slot in INTERFACE_CONTRACT["slots"].items()
}
RUNTIME_ARTIFACT_INTERFACES = {
    step_id: slot["runtime_artifact"]
    for step_id, slot in INTERFACE_CONTRACT["slots"].items()
}
RUNTIME_INPUT_BINDINGS = {
    step_id: slot["runtime_artifact"]["input_bindings"]
    for step_id, slot in INTERFACE_CONTRACT["slots"].items()
}


def _interface_slot(step_id: str) -> Mapping[str, Any]:
    try:
        return INTERFACE_CONTRACT["slots"][step_id]
    except KeyError as exc:
        raise ValueError("Unknown trainable step: %%s" %% step_id) from exc


def assert_operation_input_ports(step_id: str, inputs: Mapping[str, Any]) -> None:
    """Check the required recipe/block input ports, before runtime preprocessing."""

    boundary = _interface_slot(step_id)["operation_boundary"]
    missing = sorted(set(boundary["required_inputs"]) - set(inputs))
    if missing:
        raise ValueError(
            "Operation boundary %%s omitted input port(s): %%s"
            %% (step_id, ", ".join(missing))
        )


def assert_artifact_adapter_inputs(step_id: str, inputs: Mapping[str, Any]) -> None:
    """Check captured operation inputs needed by the returned-artifact adapter."""

    boundary = _interface_slot(step_id)["operation_boundary"]
    missing = sorted(set(boundary["artifact_adapter_inputs"]) - set(inputs))
    if missing:
        raise ValueError(
            "Artifact adapter %%s omitted captured operation input(s): %%s"
            %% (step_id, ", ".join(missing))
        )


def assert_output_ports(step_id: str, outputs: Mapping[str, Any]) -> None:
    """Check operation-boundary outputs (kept for compatibility)."""

    slot = next(item for item in SLOT_CONTRACT["trainable_slots"] if item["step_id"] == step_id)
    missing = sorted(set(slot["outputs"]) - set(outputs))
    if missing:
        raise ValueError("Slot %%s omitted output port(s): %%s" %% (step_id, ", ".join(missing)))


def assert_runtime_output_ports(step_id: str, outputs: Mapping[str, Any]) -> None:
    """Check the exact output names required from the returned model entrypoint."""

    runtime = _interface_slot(step_id)["runtime_artifact"]
    missing = sorted(set(runtime["outputs"]) - set(outputs))
    if missing:
        raise ValueError(
            "Runtime entrypoint %%s omitted output tensor(s): %%s"
            %% (runtime["entrypoint_id"], ", ".join(missing))
        )


def prepare_runtime_inputs(
    step_id: str,
    operation_inputs: Optional[Mapping[str, Any]] = None,
    recipe_params: Optional[Mapping[str, Any]] = None,
    *,
    prepared_values: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Resolve model-entrypoint values without assuming a framework or architecture.

    Identity-safe bindings are copied from operation_inputs. For a binding marked
    operation_adapter_required, preprocess the captured operation value exactly as
    the operation adapter does and pass the result in prepared_values under the
    runtime tensor name. The generated metadata intentionally never guesses such a
    conversion. recipe_params defaults to the selected step's exported parameters.
    """

    slot = _interface_slot(step_id)
    operation_values = dict(operation_inputs or {})
    parameter_values = dict(slot.get("recipe_params") or {})
    parameter_values.update(dict(recipe_params or {}))
    explicit_values = dict(prepared_values or {})
    runtime = slot["runtime_artifact"]
    expected = set(runtime["inputs"])
    unexpected = sorted(set(explicit_values) - expected)
    if unexpected:
        raise ValueError(
            "Prepared values contain unknown runtime input(s) for %%s: %%s"
            %% (runtime["entrypoint_id"], ", ".join(unexpected))
        )

    resolved = {}
    unresolved = []
    for runtime_name, binding in runtime["input_bindings"].items():
        if runtime_name in explicit_values:
            resolved[runtime_name] = explicit_values[runtime_name]
            continue
        source = binding.get("source") or {}
        if binding.get("resolution") == "identity" and source.get("kind") == "operation_input":
            source_name = str(source.get("name") or "")
            if source_name in operation_values:
                resolved[runtime_name] = operation_values[source_name]
                continue
        if binding.get("resolution") == "identity" and source.get("kind") == "recipe_param":
            source_name = str(source.get("name") or "")
            if source_name in parameter_values:
                resolved[runtime_name] = parameter_values[source_name]
                continue
        unresolved.append(runtime_name)

    if unresolved:
        raise ValueError(
            "Runtime input(s) require explicit operation preprocessing for %%s: %%s. "
            "Supply already prepared values by runtime tensor name in prepared_values; "
            "inspect RUNTIME_INPUT_BINDINGS for declared dependencies."
            %% (runtime["entrypoint_id"], ", ".join(sorted(unresolved)))
        )
    return resolved
''' % (
        repr(json.loads(compact_contract)),
        repr(json.loads(compact_interfaces)),
    )


def _validator_py() -> str:
    return r'''
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree one-file inspection.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _sha(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path):
    payload = load_strict_yaml_or_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("expected a mapping in %s" % path)
    return payload


def _load_json(path):
    payload = load_strict_yaml_or_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object in %s" % path)
    return payload


def _capture_dataset_directory(root, job):
    split = str(job.get("split") or "")
    raw = Path(str(job.get("output_dir") or ""))
    candidates = [root / "data" / split]
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((Path.cwd() / raw, root / raw))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "schema.json").is_file():
            return resolved
    return candidates[0].expanduser().resolve()


def _capture_record_sha256(arrays, index):
    digest = hashlib.sha256()
    digest.update(b"noema.capture.record@1\0")
    for tap_id, array in arrays:
        record = array[index]
        if not record.flags.c_contiguous:
            import numpy as np
            record = np.ascontiguousarray(record)
        digest.update(tap_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(record.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _capture_split_fingerprints(directory, expected_split, expected_taps):
    import numpy as np

    schema_path = directory / "schema.json"
    schema = _load_json(schema_path)
    if schema.get("kind") != "noema.capture_dataset":
        raise ValueError("unsupported Noema capture schema: %s" % schema_path)
    if str(schema.get("split") or "") != expected_split:
        raise ValueError(
            "capture %s declares split %r, expected %r"
            % (directory, schema.get("split"), expected_split)
        )
    tap_ids = tuple(sorted(str(item) for item in expected_taps if str(item)))
    if not tap_ids:
        tap_ids = tuple(
            sorted(str(item) for item in (schema.get("tap_schemas") or {}) if str(item))
        )
    if not tap_ids or len(set(tap_ids)) != len(tap_ids):
        raise ValueError("capture tensor taps must be non-empty and unique: %s" % directory)
    fingerprints = []
    shards = list(schema.get("shards") or [])
    if not shards:
        raise ValueError("capture has no shards: %s" % directory)
    for raw_shard in shards:
        relative = raw_shard.get("path") if isinstance(raw_shard, dict) else raw_shard
        shard_path = (directory / str(relative or "")).resolve()
        try:
            shard_path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("capture shard escapes its dataset directory: %s" % shard_path) from exc
        if not shard_path.is_file():
            raise ValueError("capture shard is missing: %s" % shard_path)
        with np.load(str(shard_path), allow_pickle=False) as payload:
            missing = [tap for tap in tap_ids if tap not in payload.files]
            if missing:
                raise ValueError(
                    "capture shard %s omits selected tap(s): %s"
                    % (shard_path, ", ".join(missing))
                )
            arrays = [(tap, np.asarray(payload[tap])) for tap in tap_ids]
            counts = {int(array.shape[0]) for _, array in arrays if array.ndim >= 1}
            if len(counts) != 1 or any(array.ndim < 1 for _, array in arrays):
                raise ValueError(
                    "capture shard %s does not align selected taps on the record axis"
                    % shard_path
                )
            for index in range(counts.pop()):
                fingerprints.append(_capture_record_sha256(arrays, index))
    if not fingerprints:
        raise ValueError("capture contains no records: %s" % directory)
    declared = int(schema.get("captured_samples") or 0)
    if declared and declared != len(fingerprints):
        raise ValueError(
            "capture %s declares %d records but its shards contain %d"
            % (directory, declared, len(fingerprints))
        )
    return tap_ids, tuple(fingerprints)


def _validate_capture_split_integrity(root, manifest):
    from collections import Counter

    jobs = [item for item in manifest.get("capture_jobs") or [] if isinstance(item, dict)]
    if not jobs:
        return
    directories = {
        str(job.get("split") or ""): _capture_dataset_directory(root, job)
        for job in jobs
    }
    # Contract validation is useful before capture. Perform the data-level
    # guard once every declared split has actually been materialized.
    if not all((directory / "schema.json").is_file() for directory in directories.values()):
        return
    split_records = []
    for job in jobs:
        split = str(job.get("split") or "")
        taps = [
            str(item.get("id") or "")
            for item in job.get("expected_taps") or []
            if isinstance(item, dict) and str(item.get("id") or "")
        ]
        tap_ids, fingerprints = _capture_split_fingerprints(
            directories[split], split, taps
        )
        split_records.append((split, tap_ids, fingerprints))
    reference_taps = split_records[0][1]
    mismatched = [
        split for split, tap_ids, _ in split_records[1:] if tap_ids != reference_taps
    ]
    if mismatched:
        raise ValueError(
            "captured dataset splits do not contain the same selected taps: %s"
            % ", ".join(mismatched)
        )
    overlaps = []
    for left_index, (left, tap_ids, left_records) in enumerate(split_records):
        for right, _, right_records in split_records[left_index + 1:]:
            shared = Counter(left_records) & Counter(right_records)
            shared_records = sum(shared.values())
            if shared_records:
                overlaps.append(
                    "%s and %s share %d byte-identical complete record%s"
                    % (
                        left,
                        right,
                        shared_records,
                        "" if shared_records == 1 else "s",
                    )
                )
    if overlaps:
        raise ValueError(
            "captured dataset leakage detected: %s. The comparison fingerprints all "
            "selected taps together, so repeated labels alone are allowed. Do not "
            "train, validate, or report metrics from these captures. Overwrite all "
            "affected splits after correcting the random-seed or partitioning configuration."
            % "; ".join(overlaps)
        )


def _source_resolves(source, graph):
    nodes = {
        str(item.get("id") or ""): item
        for item in graph.get("nodes") or []
        if isinstance(item, dict)
    }
    boundary = {
        str(item.get("recipe_reference") or "")
        for item in graph.get("external_inputs") or []
        if isinstance(item, dict)
    }
    boundary.update(
        str(item.get("id") or "")
        for item in graph.get("external_outputs") or []
        if isinstance(item, dict)
    )
    if source in boundary:
        return True
    parts = str(source or "").split(".")
    if len(parts) < 2 or parts[0] not in nodes:
        return False
    node = nodes[parts[0]]
    if parts[1] != "params":
        return len(parts) == 2 and parts[1] in (node.get("outputs") or {})
    value = node.get("params") or {}
    for key in parts[2:]:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def _validate_contract_sources(contract, graph):
    for section in ("conditioning", "signals"):
        for row in contract.get(section) or []:
            source = str(row.get("source") or "")
            if not _source_resolves(source, graph):
                raise ValueError("%s source does not resolve: %s" % (section, source))


def _validate_artifact_bindings(contract):
    slots = {str(item.get("step_id") or ""): item for item in contract.get("trainable_slots") or []}
    seen = set()
    for binding in (contract.get("artifact_return") or {}).get("bindings") or []:
        step_id = str(binding.get("step_id") or "")
        if step_id not in slots or step_id in seen:
            raise ValueError("artifact bindings must reference each slot exactly once")
        seen.add(step_id)
        slot = slots[step_id]
        abi = slot.get("runtime_artifact_abi") or {}
        expected = sorted(str(item) for item in abi.get("required_operation_inputs") or [])
        if not expected:
            expected = sorted(
                str(name)
                for name, spec in (slot.get("inputs") or {}).items()
                if bool((spec or {}).get("required", False))
            )
        actual = sorted(str(item) for item in binding.get("required_inputs") or [])
        if actual != expected:
            raise ValueError("artifact binding required inputs disagree with runtime ABI: %s" % step_id)
    if seen != set(slots):
        raise ValueError("artifact bindings do not cover all trainable slots")


def _validate_data_contract(root, manifest, training_contract):
    reference = manifest.get("data_contract")
    jobs = manifest.get("capture_jobs") or []
    if not reference:
        if jobs:
            raise ValueError("capture jobs require a bound data contract")
        return
    path = root / str(reference.get("path") or "")
    data_contract = _load_yaml(path)
    if data_contract.get("kind") != "noema.training_data_contract@1":
        raise ValueError("unsupported training data contract kind")
    if _sha(data_contract) != reference.get("sha256"):
        raise ValueError("training data contract hash mismatch")
    if _file_sha(path) != reference.get("file_sha256"):
        raise ValueError("training data contract file hash mismatch")
    if (data_contract.get("source_recipe") or {}).get("sha256") != training_contract["source_recipe"]["sha256"]:
        raise ValueError("training data and slot contracts use different source recipes")
    mode = str(data_contract.get("mode") or "")
    ownership = data_contract.get("ownership") or {}
    if mode == "file_backed_live_differentiable":
        expected_ownership = {
            "source_selection": "noema_recipe",
            "partition_materialization": "noema_export",
            "file_integrity_validation": "noema_contract_and_external_trainer",
            "dataset_consumption": "external_researcher",
            "model_training": "external_researcher",
            "test_evaluation": "noema_ordinary_recipe_or_benchmark",
        }
        if ownership != expected_ownership or (reference.get("ownership") or {}) != expected_ownership:
            raise ValueError("file-backed training data ownership boundary is incomplete")
        if jobs or (data_contract.get("capture_recipes") or []):
            raise ValueError("file-backed live data must not declare capture jobs")
        split_rows = [
            item for item in data_contract.get("splits") or []
            if isinstance(item, dict)
        ]
        splits = {str(item.get("id") or ""): item for item in split_rows}
        if len(split_rows) != 2 or set(splits) != {"train", "validation"}:
            raise ValueError("file-backed live data requires exact train and validation splits")
        materialized_ids = []
        seen_ids = set()
        seen_paths = set()
        for split in ("train", "validation"):
            files = splits[split].get("files") or []
            if not files or int(splits[split].get("image_count") or 0) != len(files):
                raise ValueError("file-backed split is empty or has the wrong image count: %s" % split)
            for row in files:
                image_id = str((row or {}).get("image_id") or "")
                resolved_path = str((row or {}).get("resolved_path") or "")
                expected_sha = str((row or {}).get("sha256") or "")
                if not image_id or image_id in seen_ids:
                    raise ValueError("file-backed image IDs must be non-empty and unique")
                path = Path(resolved_path).expanduser()
                normalized_path = str(path.resolve())
                if not path.is_file() or normalized_path in seen_paths:
                    raise ValueError("file-backed source path is missing or duplicated: %s" % resolved_path)
                if len(expected_sha) != 64 or _file_sha(path) != expected_sha:
                    raise ValueError("file-backed source image hash mismatch: %s" % image_id)
                seen_ids.add(image_id)
                seen_paths.add(normalized_path)
                materialized_ids.append(image_id)
        selected_ids = [
            str(item) for item in (data_contract.get("dataset") or {}).get("selected_image_ids") or []
        ]
        if materialized_ids != selected_ids:
            raise ValueError("file-backed splits do not match the ordered recipe image selection")
        test = data_contract.get("test_evaluation") or {}
        if bool(test.get("included_in_training_bundle")) or (test.get("image_ids") or []):
            raise ValueError("file-backed training data must not expose held-out test images")
        return

    if mode == "captured_supervised_pairs":
        expected_ownership = {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        }
        if ownership != expected_ownership or (reference.get("ownership") or {}) != expected_ownership:
            raise ValueError("supervised capture ownership boundary is incomplete")
        labels = data_contract.get("labels") or {}
        if labels.get("required") is not True or labels.get("not_a_runtime_operation_input") is not True:
            raise ValueError("supervised capture must declare training-only target labels")
        feature = data_contract.get("feature") or {}
        target = data_contract.get("target") or {}
        if not str(feature.get("tap_id") or "") or not str(feature.get("reference") or ""):
            raise ValueError("supervised capture feature contract is incomplete")
        if not str(target.get("tap_id") or "") or not str(target.get("reference") or ""):
            raise ValueError("supervised capture target contract is incomplete")
        if feature.get("tap_id") == target.get("tap_id") or feature.get("reference") == target.get("reference"):
            raise ValueError("supervised capture feature and target must be distinct")
        split_rows = [
            item for item in data_contract.get("splits") or []
            if isinstance(item, dict)
        ]
        splits = {str(item.get("id") or ""): item for item in split_rows}
        if len(split_rows) != 3 or set(splits) != {"train", "validation", "test"}:
            raise ValueError("supervised capture requires exact train, validation, and test splits")
        if str(splits["test"].get("training_use") or "") != "held_out_evaluation":
            raise ValueError("supervised capture test split must remain held out")
        target_signals = [
            item for item in training_contract.get("signals") or []
            if isinstance(item, dict)
            and str(item.get("purpose") or "") == "supervised_target"
            and str(item.get("source") or "") == str(target.get("reference") or "")
        ]
        if len(target_signals) != 1:
            raise ValueError("slot contract must expose exactly one matching supervised target signal")
    elif mode == "captured_self_supervised_csi":
        expected_ownership = {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        }
        if ownership != expected_ownership or (reference.get("ownership") or {}) != expected_ownership:
            raise ValueError("self-supervised CSI capture ownership boundary is incomplete")
        labels = data_contract.get("labels") or {}
        if labels.get("required") is not False or labels.get("separate_target_captured") is not False:
            raise ValueError("self-supervised CSI capture must not require a separate target")
        feature = data_contract.get("feature_and_target") or {}
        if not str(feature.get("tap_id") or "") or not str(feature.get("reference") or ""):
            raise ValueError("self-supervised CSI feature-and-target contract is incomplete")
        if str(feature.get("pairing") or "") != "autoencoder_input_equals_reconstruction_target":
            raise ValueError("self-supervised CSI must pair one true-CSI tensor as input and target")
        feedback = data_contract.get("feedback_constraint") or {}
        dimension = int(feedback.get("feedback_dimension") or 0)
        bits = int(feedback.get("bits_per_latent") or 0)
        if dimension <= 0 or bits <= 0 or int(feedback.get("feedback_bits_per_sample") or 0) != dimension * bits:
            raise ValueError("CSI feedback bit budget is incomplete or inconsistent")
        if str(feedback.get("transport") or "") not in {"ideal_noiseless", "uniform_quantized"}:
            raise ValueError("CSI feedback transport is unsupported")
        split_rows = [
            item for item in data_contract.get("splits") or []
            if isinstance(item, dict)
        ]
        splits = {str(item.get("id") or ""): item for item in split_rows}
        if len(split_rows) != 3 or set(splits) != {"train", "validation", "test"}:
            raise ValueError("self-supervised CSI capture requires exact train, validation, and test splits")
        if str(splits["test"].get("training_use") or "") != "held_out_evaluation":
            raise ValueError("self-supervised CSI test split must remain held out")
        matching_true_csi = [
            item for item in training_contract.get("signals") or []
            if isinstance(item, dict)
            and str(item.get("purpose") or "") == "self_supervised_input_and_reconstruction_target"
            and str(item.get("source") or "") == str(feature.get("reference") or "")
        ]
        if len(matching_true_csi) != 1:
            raise ValueError("slot contract must expose exactly one matching true-CSI signal")
    elif mode == "captured_generic_tensors":
        expected_ownership = {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_architecture": "external_researcher",
            "loss_definition": "external_researcher",
            "model_training": "external_researcher",
        }
        if ownership != expected_ownership or (reference.get("ownership") or {}) != expected_ownership:
            raise ValueError("generic tensor-capture ownership boundary is incomplete")
        signal_rows = [
            item for item in data_contract.get("signals") or []
            if isinstance(item, dict)
        ]
        signal_pairs = {
            (str(item.get("tap_id") or ""), str(item.get("reference") or ""))
            for item in signal_rows
        }
        if not signal_rows or len(signal_pairs) != len(signal_rows) or any(not left or not right for left, right in signal_pairs):
            raise ValueError("generic tensor-capture signals must be non-empty and unique")
        split_rows = [
            item for item in data_contract.get("splits") or []
            if isinstance(item, dict)
        ]
        splits = {str(item.get("id") or ""): item for item in split_rows}
        if len(split_rows) != 3 or set(splits) != {"train", "validation", "test"}:
            raise ValueError("generic tensor capture requires exact train, validation, and test splits")
        if str(splits["test"].get("training_use") or "") != "held_out_evaluation":
            raise ValueError("generic tensor-capture test split must remain held out")
    elif mode in {"", "captured_label_free"}:
        expected_ownership = {
            "capture_recipe_generation": "noema",
            "capture_execution": "noema",
            "capture_integrity_validation": "noema",
            "dataset_consumption": "external_researcher",
            "model_training": "external_researcher",
        }
        if ownership != expected_ownership or (reference.get("ownership") or {}) != expected_ownership:
            raise ValueError("training data ownership boundary is incomplete")
        labels = data_contract.get("labels") or {}
        if labels.get("required") is not False or labels.get("oracle_power_allocation_captured") is not False:
            raise ValueError("label-free data contract must not require or capture oracle allocation")
    else:
        raise ValueError("unsupported training data contract mode: %s" % mode)

    capture_plan = data_contract.get("capture_plan") or {}
    if capture_plan:
        if str(capture_plan.get("mode") or "") != "captured_tensors":
            raise ValueError("capture-backed data contract has an invalid capture-plan mode")
        total = int(capture_plan.get("total_samples") or 0)
        counts = (capture_plan.get("split_plan") or {}).get("counts") or {}
        if total <= 0 or set(counts) != {"train", "validation", "test"}:
            raise ValueError("capture plan requires an exact train/validation/test split")
        if sum(int(value) for value in counts.values()) != total:
            raise ValueError("capture-plan split counts do not sum to total_samples")

    assets = {
        str(item.get("split") or ""): item
        for item in data_contract.get("capture_recipes") or []
        if isinstance(item, dict)
    }
    job_splits = {str(item.get("split") or "") for item in jobs if isinstance(item, dict)}
    if not assets or set(assets) != job_splits:
        raise ValueError("data contract capture recipes and manifest jobs differ")
    if mode in {"captured_supervised_pairs", "captured_self_supervised_csi", "captured_generic_tensors"} and job_splits != {"train", "validation", "test"}:
        raise ValueError("capture jobs must cover train, validation, and test")
    csi_capture_seeds = set()
    for job in jobs:
        split = str(job.get("split") or "")
        if job.get("owner") != "noema" or job.get("consumer") != "external_researcher":
            raise ValueError("capture job ownership is invalid: %s" % split)
        capture_path = root / str(job.get("bundle_recipe_path") or "")
        capture_recipe = _load_yaml(capture_path)
        semantic_sha = _sha(capture_recipe)
        file_sha = _file_sha(capture_path)
        asset = assets[split]
        if semantic_sha != job.get("recipe_sha256") or semantic_sha != asset.get("sha256"):
            raise ValueError("capture recipe hash mismatch: %s" % split)
        if file_sha != job.get("recipe_file_sha256") or file_sha != asset.get("file_sha256"):
            raise ValueError("capture recipe file hash mismatch: %s" % split)
        capture = capture_recipe.get("dataset_capture") or {}
        if str(capture.get("split") or "") != split:
            raise ValueError("capture recipe split mismatch: %s" % split)
        if int(capture.get("samples") or 0) != int(job.get("requested_samples") or 0):
            raise ValueError("capture recipe sample count mismatch: %s" % split)
        taps = capture.get("taps") or []
        if taps != (job.get("expected_taps") or []) or taps != (asset.get("taps") or []):
            raise ValueError("capture recipe taps mismatch: %s" % split)
        tap_ids = [str((tap or {}).get("id") or "") for tap in taps]
        tap_refs = [str((tap or {}).get("from") or "") for tap in taps]
        if not tap_ids or len(set(tap_ids)) != len(tap_ids) or len(set(tap_refs)) != len(tap_refs):
            raise ValueError("capture recipe tap IDs and references must be non-empty and unique: %s" % split)
        for tap in taps:
            tap_id = str(tap.get("id") or "").lower()
            source = str(tap.get("from") or "")
            if mode in {"", "captured_label_free"} and (
                "oracle" in tap_id or tap_id == "power_allocation" or source.endswith(".allocation")
            ):
                raise ValueError("capture recipe contains an oracle allocation tap: %s" % split)
        if mode == "captured_supervised_pairs":
            expected_pair = {
                (
                    str((data_contract.get("feature") or {}).get("tap_id") or ""),
                    str((data_contract.get("feature") or {}).get("reference") or ""),
                ),
                (
                    str((data_contract.get("target") or {}).get("tap_id") or ""),
                    str((data_contract.get("target") or {}).get("reference") or ""),
                ),
            }
            captured = {(str(tap.get("id") or ""), str(tap.get("from") or "")) for tap in taps}
            if not expected_pair.issubset(captured):
                raise ValueError("supervised capture recipe omits its feature or target contract: %s" % split)
            split_contract = next(
                item for item in data_contract.get("splits") or []
                if str((item or {}).get("id") or "") == split
            )
            if int(split_contract.get("requested_packet_records") or 0) != int(job.get("requested_samples") or 0):
                raise ValueError("supervised split and capture job sample counts differ: %s" % split)
        elif mode == "captured_self_supervised_csi":
            expected = data_contract.get("feature_and_target") or {}
            required_csi = (
                str(expected.get("tap_id") or ""),
                str(expected.get("reference") or ""),
            )
            captured = {(str(tap.get("id") or ""), str(tap.get("from") or "")) for tap in taps}
            if captured != {required_csi}:
                raise ValueError("CSI capture recipe must contain only its realized true-CSI tap: %s" % split)
            steps = capture_recipe.get("steps") or []
            if len(steps) != 1 or str((steps[0] or {}).get("op") or "") != "wireless.miso_ofdm_csi":
                raise ValueError("CSI capture recipe must execute only wireless.miso_ofdm_csi: %s" % split)
            seed = int(((steps[0] or {}).get("params") or {}).get("seed"))
            if seed in csi_capture_seeds:
                raise ValueError("CSI train/validation/test capture seeds must be disjoint")
            csi_capture_seeds.add(seed)
            split_contract = next(
                item for item in data_contract.get("splits") or []
                if str((item or {}).get("id") or "") == split
            )
            if int(split_contract.get("requested_samples") or 0) != int(job.get("requested_samples") or 0):
                raise ValueError("CSI split and capture job sample counts differ: %s" % split)
        elif mode == "captured_generic_tensors":
            expected_signals = {
                (
                    str((item or {}).get("tap_id") or ""),
                    str((item or {}).get("reference") or ""),
                )
                for item in data_contract.get("signals") or []
            }
            captured = {
                (str(tap.get("id") or ""), str(tap.get("from") or ""))
                for tap in taps
            }
            if captured != expected_signals:
                raise ValueError("generic capture recipe signals differ from its data contract: %s" % split)
            split_contract = next(
                item for item in data_contract.get("splits") or []
                if str((item or {}).get("id") or "") == split
            )
            if int(split_contract.get("requested_samples") or 0) != int(job.get("requested_samples") or 0):
                raise ValueError("generic split and capture job sample counts differ: %s" % split)
        else:
            feature = data_contract.get("feature") or {}
            required_feature = (
                str(feature.get("tap_id") or ""),
                str(feature.get("reference") or ""),
            )
            captured = {(str(tap.get("id") or ""), str(tap.get("from") or "")) for tap in taps}
            if required_feature not in captured:
                raise ValueError("label-free capture recipe omits its required feature: %s" % split)


def validate_bundle(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    contract = _load_yaml(root / "training_contract.yaml")
    graph = _load_json(root / "scenario_graph.json")
    recipe = _load_yaml(root / "noema_recipe.yaml")
    manifest = _load_yaml(root / "project_manifest.yaml")
    if contract.get("kind") != "noema.trainable_slot_contract@1":
        raise ValueError("unsupported training contract kind")
    if graph.get("kind") != "noema.typed_training_scenario_graph@1":
        raise ValueError("unsupported scenario graph kind")
    if _sha(recipe) != contract["source_recipe"]["sha256"]:
        raise ValueError("source recipe hash mismatch")
    if _sha(contract) != manifest["contracts"]["trainable_slots"]["sha256"]:
        raise ValueError("training contract hash mismatch")
    if _file_sha(root / "training_contract.yaml") != manifest["contracts"]["trainable_slots"]["file_sha256"]:
        raise ValueError("training contract file hash mismatch")
    if _sha(graph) != manifest["contracts"]["scenario_graph"]["sha256"]:
        raise ValueError("scenario graph hash mismatch")
    if _file_sha(root / "scenario_graph.json") != manifest["contracts"]["scenario_graph"]["file_sha256"]:
        raise ValueError("scenario graph file hash mismatch")
    if _sha(graph) != (contract.get("scenario_graph") or {}).get("sha256"):
        raise ValueError("training contract scenario graph reference mismatch")
    if (graph.get("source_recipe") or {}).get("sha256") != contract["source_recipe"]["sha256"]:
        raise ValueError("scenario graph source recipe mismatch")
    if (graph.get("source_recipe") or {}).get("execution_profile") != contract["source_recipe"].get("execution_profile"):
        raise ValueError("scenario graph execution profile mismatch")
    identity = _sha(
        {
            "source_recipe_sha256": contract["source_recipe"]["sha256"],
            "framework": contract.get("framework"),
            "trainable_steps": [item["step_id"] for item in contract["trainable_slots"]],
            "loss_steps": contract.get("recipe_loss_steps") or [],
            "slot_groups": contract.get("slot_groups") or [],
            "scenario_steps": graph.get("topological_order") or [],
        }
    )
    if identity != contract.get("identity_sha256"):
        raise ValueError("training contract identity mismatch")
    slot_ids = {item["step_id"] for item in contract["trainable_slots"]}
    placeholders = {item["id"] for item in graph["nodes"] if item["role"] == "trainable_placeholder"}
    if slot_ids != placeholders:
        raise ValueError("trainable slots and graph placeholders differ")
    _validate_contract_sources(contract, graph)
    _validate_artifact_bindings(contract)
    _validate_data_contract(root, manifest, contract)
    _validate_capture_split_integrity(root, manifest)
    return contract


def main() -> int:
    contract = validate_bundle()
    print("contract valid: %s" % contract["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _package_artifact_py() -> str:
    return r'''
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from validate_contract import validate_bundle


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a runtime-adapter package request for a trained artifact.")
    parser.add_argument("artifact")
    parser.add_argument("--format", required=True)
    parser.add_argument("--runtime", required=True, choices=("builtin", "onnx", "torchscript", "trusted_plugin"))
    parser.add_argument("--out", default="artifact_package_request.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    artifact = Path(args.artifact).expanduser().resolve()
    if not artifact.is_file():
        raise ValueError("artifact is not a readable file: %s" % artifact)
    contract = validate_bundle(root)
    contract_sha = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "noema.trained_artifact_package_request@1",
        "status": "requires_runtime_adapter_validation",
        "artifact": {"path": str(artifact), "sha256": _file_sha256(artifact), "format": args.format},
        "runtime": {"kind": args.runtime},
        "training_contract": {
            "id": contract["id"],
            "version": contract.get("version", contract.get("schema_version")),
            "sha256": contract_sha,
            "file_sha256": _file_sha256(root / "training_contract.yaml"),
        },
        "bindings": contract["artifact_return"]["bindings"],
        "next_step": (
            "Validate this request with the runtime adapter for the declared format. Only that adapter may emit "
            "a ready noema.trained_block_artifact manifest."
        ),
    }
    destination = Path(args.out)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("wrote package request: %s" % destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _readme_tensor_signature(values: Mapping[str, Any]) -> str:
    if not values:
        return "none"
    rows = []
    for name, raw_spec in values.items():
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {}
        shape = "[{}]".format(", ".join(str(item) for item in spec.get("shape") or []))
        rows.append(
            "`{}` (`{}` `{}`)".format(
                str(name),
                str(spec.get("dtype") or "operation_defined"),
                shape,
            )
        )
    return ", ".join(rows)


def _bundle_readme(
    contract: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
) -> str:
    interfaces = _bundle_interface_descriptor(
        contract,
        source_recipe,
    )
    slot_sections = []
    for step_id, raw_slot in interfaces["slots"].items():
        slot = dict(raw_slot)
        boundary = dict(slot["operation_boundary"])
        runtime = dict(slot["runtime_artifact"])
        bindings = dict(runtime.get("input_bindings") or {})
        adapter_sources = []
        for name in runtime.get("required_operation_inputs") or []:
            spec = dict((boundary.get("inputs") or {}).get(str(name)) or {})
            reference = str(spec.get("recipe_reference") or "")
            adapter_sources.append(
                "`{}`{}".format(
                    str(name),
                    " from `{}`".format(reference) if reference else "",
                )
            )
        identity = sorted(
            name
            for name, binding in bindings.items()
            if str((binding or {}).get("resolution") or "") == "identity"
        )
        adapted = sorted(set(bindings) - set(identity))
        preparation = []
        if identity:
            preparation.append(
                "identity-safe: " + ", ".join("`{}`".format(name) for name in identity)
            )
        if adapted:
            preparation.append(
                "operation preprocessing required: "
                + ", ".join("`{}`".format(name) for name in adapted)
            )
        slot_sections.append(
            "\n".join(
                [
                    "### `{}` — {}".format(step_id, str(slot.get("role") or "trainable block")),
                    "",
                    "- Operation input boundary (recipe/capture names): {}.".format(
                        _readme_tensor_signature(boundary.get("inputs") or {})
                    ),
                    "- Operation output boundary (recipe graph names): {}.".format(
                        _readme_tensor_signature(boundary.get("outputs") or {})
                    ),
                    "- Captured artifact-adapter input(s): {}.".format(
                        ", ".join(adapter_sources) or "none"
                    ),
                    "- Returned-artifact entrypoint: `{}`.".format(
                        str(runtime.get("entrypoint_id") or "")
                    ),
                    "- Runtime model inputs (exact exported-model names): {}.".format(
                        _readme_tensor_signature(runtime.get("inputs") or {})
                    ),
                    "- Runtime model outputs (exact exported-model names): {}.".format(
                        _readme_tensor_signature(runtime.get("outputs") or {})
                    ),
                    "- Input preparation: {}.".format("; ".join(preparation) or "none"),
                ]
            )
        )
    return """
# Noema training interface bundle: {contract_id}

This is an architecture-, loss-, optimizer-, and trainer-neutral contract bundle. It does not contain
a model or claim that external training results are benchmark results.

## Commands included in this bundle

From this directory, validate the exported recipe, contracts, scenario graph, and integrity hashes with:

```bash
python validate_contract.py
```

**No training program is included.** In particular, this directory does not contain `train.py`, so
`python train.py` is not a valid command unless you add that file yourself or explicitly attach a
demonstration training project. Supply your own model, loss, optimizer, and trainer, and run the command
defined by that external project. Noema intentionally cannot invent one architecture-neutral training
command from the tensor contract alone.

After external training, complete `trained_artifact.yaml` from
`trained_artifact.template.yaml` and place the declared model components at its paths. In Workbench,
select **Return and validate model** > **Validate returned model**. That check validates the returned
artifact's runtime ABI, bindings, hashes, and integrity; it does not evaluate research performance.

For an artifact format that requires a separate runtime adapter, create an adapter-validation request
with an explicit model path, format, and runtime, for example:

```bash
python package_artifact.py path/to/model.onnx --format onnx --runtime onnx
```

This packaging helper does not train a model and does not by itself create a ready Noema artifact.

## Operation boundary versus returned-model runtime ABI

These are two related but different interfaces:

- The **operation boundary** is the complete recipe graph interface. Its port names, storage dtype,
  and layout describe values exchanged between Noema blocks.
- The **captured artifact-adapter inputs** are the subset of operation inputs required to run the
  returned artifact. Capture records these using their recipe references, plus any optional signals
  selected separately for an external objective or diagnostics.
- The **runtime artifact ABI** is the callable interface that the returned model file must expose.
  The entrypoint name and every runtime tensor name, dtype, and shape are exact deployment requirements.

Do not make a model expose operation-boundary output names unless they are also listed in the runtime
ABI. A Noema operation adapter may unpack a captured artifact, convert dtype/layout, derive tensors from
metadata or recipe parameters, and convert runtime outputs back to operation outputs. Existing ABI
metadata only proves an identity mapping when both the name and tensor specification match. Every other
mapping is marked `operation_adapter_required`; the bundle does not guess preprocessing.
`runtime_artifact.binding_params` configures the Noema block to load the artifact; those values are
deployment settings, not additional model input tensors.

`interfaces.py` exposes `OPERATION_BOUNDARIES`, `RUNTIME_ARTIFACT_INTERFACES`, and
`RUNTIME_INPUT_BINDINGS`. `prepare_runtime_inputs(...)` copies proven identity mappings and accepts
explicitly preprocessed runtime tensors through `prepared_values`. It is framework-neutral and validates
names only; the runtime adapter performs final dtype and shape validation.

{slot_sections}

## Normative files

- `training_contract.yaml`: typed trainable slots, conditioning, signals, constraints, and artifact-return boundary.
- `scenario_graph.json`: port-wired DAG with external trainable placeholders and Noema-owned frozen context.
- `noema_recipe.yaml`: training-neutral source scenario snapshot.
- `training_plan.yaml` (export workflow): separately hashed replacement steps, recipe loss boundaries used for route analysis, capture policy, and support framework. Capture-backed plans retain the analyzed boundary even though `scenario_graph.json` has no live recipe backward route. An externally authored objective may be retained as provenance, but Noema does not choose it.
- `project_manifest.yaml`: hashes and bundle ownership.
- `interfaces.py`: separate operation-boundary and returned-model runtime-ABI declarations plus framework-neutral port helpers.
- `structured_input.py`: standalone strict YAML/JSON decoding for the generated validator; duplicate keys, non-finite values, and non-JSON YAML types are rejected.
- `test_vectors/contract_vectors.json`: symbolic operation and runtime interfaces for independent tooling.
- `trained_artifact.template.yaml`: ONNX-first runtime ABI and ordinary-block bindings generated from the selected slots.

Run `python validate_contract.py` again after moving or modifying the bundle. Train the external model
to expose the runtime artifact ABI above. A package request is not a ready Noema trained artifact until
a compatible runtime adapter validates it.
""".strip().format(
        contract_id=str(contract.get("id") or "training_interface"),
        slot_sections="\n\n".join(slot_sections),
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CompiledTrainingContract",
    "ConstraintSpec",
    "NamedValueSpec",
    "SignalSpec",
    "SlotGroupSpec",
    "TensorSpec",
    "TRAINABLE_SLOT_CONTRACT_KIND",
    "TRAINING_INTERFACE_PROJECT_KIND",
    "TYPED_SCENARIO_GRAPH_KIND",
    "TrainingContractError",
    "TrainingContractOptions",
    "compile_training_contract",
    "tensor_spec_for_kind",
    "validate_compiled_training_contract",
    "write_training_contract_bundle",
]
