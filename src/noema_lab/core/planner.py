from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from noema_lab.core.execution_profiles import (
    CUSTOM_EXECUTION_PROFILE_ID,
    inspect_execution_profile,
)
from noema_lab.core.artifacts import file_sha256
from noema_lab.core.materialization import (
    MaterializationRegistry,
    ResolvedMaterialization,
    build_materialization_registry,
    normalize_materialization_backend,
    normalize_materialization_runner,
)
from noema_lab.core.operations import (
    Operation,
    OperationError,
    OperationRegistry,
    normalize_input_metadata_requirements,
    normalize_output_metadata_guarantees,
)
from noema_lab.core.params import validate_params
from noema_lab.core.recipes import (
    Recipe,
    RecipeStep,
    RecipeValidationError,
    validate_recipe_step_id,
)
from noema_lab.core.reproducibility import canonical_json_sha256, recipe_fingerprint
from noema_lab.core.research import validate_recipe_research_metadata

JsonDict = Dict[str, Any]

EXECUTION_PLAN_SCHEMA_VERSION = 1
PLANNED_STEP_SCHEMA_VERSION = 1
EXECUTION_PLAN_KIND = "noema.execution_plan"
DEFAULT_EXECUTION_RUNNER = "benchmark_run"

_BACKEND_PARAM_KEYS = (
    "materialization_backend",
    "execution_backend",
    "wireless_backend",
    "runtime_backend",
    "backend",
)
_AUTOMATIC_BACKEND_VALUES = {"", "auto", "automatic", "default"}
_MATERIALIZATION_SELECTOR_SCHEMA_KEY = "x-noema-materialization-selector"


class RecipePlanningError(RecipeValidationError):
    pass


@dataclass(frozen=True)
class PlannedStep:
    """Immutable binding between one recipe step and one implementation.

    ``operation`` is the already-resolved executable object. It is intentionally
    excluded from serialized evidence; its stable class identity and the full
    operation contract are captured by the other fields and the parent plan.
    """

    step_id: str
    operation_id: str
    runner: str
    backend: str
    implementation: str
    materialization_id: str
    implementation_identity: str
    implementation_metadata: Mapping[str, Any]
    operation_contract_sha256: str
    binding_sha256: str
    operation: Operation = field(repr=False, compare=False)
    schema_version: int = PLANNED_STEP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation_metadata",
            _freeze_mapping(self.implementation_metadata),
        )

    def to_dict(self) -> JsonDict:
        return {
            **self._binding_payload(),
            "binding_sha256": self.binding_sha256,
        }

    def _binding_payload(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "step_id": self.step_id,
            "operation_id": self.operation_id,
            "runner": self.runner,
            "backend": self.backend,
            "implementation": self.implementation,
            "materialization_id": self.materialization_id,
            "implementation_identity": self.implementation_identity,
            "implementation_metadata": _thaw_json(self.implementation_metadata),
            "operation_contract_sha256": self.operation_contract_sha256,
        }

    def params_for_execution(self, params: Mapping[str, Any]) -> JsonDict:
        """Apply planner-owned backend choices to a fresh parameter mapping."""

        payload = copy.deepcopy(dict(params))
        overrides = self.implementation_metadata.get("parameter_overrides") or {}
        if isinstance(overrides, Mapping):
            payload.update(_thaw_json(overrides))
        return payload

    def assert_implementation_unchanged(self) -> None:
        """Reject source or contract mutation between planning and dispatch."""

        current = _operation_source_identity(self.operation)
        expected_module = str(
            self.implementation_metadata.get("source_module") or ""
        )
        expected_sha = str(
            self.implementation_metadata.get("source_sha256") or ""
        )
        if (
            current["source_module"] != expected_module
            or current["source_sha256"] != expected_sha
        ):
            raise RecipePlanningError(
                "implementation bytes changed after planning for step %s" % self.step_id
            )
        current_identity = "%s.%s" % (
            self.operation.__class__.__module__,
            self.operation.__class__.__qualname__,
        )
        if current_identity != self.implementation_identity:
            raise RecipePlanningError(
                "implementation class changed after planning for step %s" % self.step_id
            )
        try:
            contract_sha256 = canonical_json_sha256(self.operation.describe())
        except (OperationError, TypeError, ValueError) as exc:
            raise RecipePlanningError(
                "operation contract became invalid after planning for step %s: %s"
                % (self.step_id, exc)
            ) from exc
        if contract_sha256 != self.operation_contract_sha256:
            raise RecipePlanningError(
                "operation contract changed after planning for step %s" % self.step_id
            )


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable, serializable execution decision made before run side effects."""

    runner: str
    recipe_name: str
    recipe_sha256: str
    steps: Tuple[PlannedStep, ...]
    operation_contracts: Mapping[str, Mapping[str, Any]]
    operation_contracts_sha256: str
    sha256: str
    schema_version: int = EXECUTION_PLAN_SCHEMA_VERSION
    kind: str = EXECUTION_PLAN_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(
            self,
            "operation_contracts",
            _freeze_mapping(self.operation_contracts),
        )

    def to_dict(self) -> JsonDict:
        return {
            **self._plan_payload(),
            "sha256": self.sha256,
        }

    def operation_contracts_to_dict(self) -> JsonDict:
        return {
            "schema_version": 1,
            "sha256": self.operation_contracts_sha256,
            "operations": _thaw_json(self.operation_contracts),
        }

    def _plan_payload(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "runner": self.runner,
            "recipe": {
                "name": self.recipe_name,
                "sha256": self.recipe_sha256,
                "step_count": len(self.steps),
            },
            "operation_contracts": self.operation_contracts_to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


def plan_recipe(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    runner: str = DEFAULT_EXECUTION_RUNNER,
    backend: Optional[str] = None,
    implementation: Optional[str] = None,
    enforce_execution_profile: bool = True,
) -> ExecutionPlan:
    """Validate and bind every recipe step to a runner materialization.

    Resolution is deterministic. An explicit planner ``backend`` has highest
    precedence, followed by a backend-selecting step parameter (notably
    ``wireless_backend``). ``auto`` and unspecified backends select the first
    compatible implemented materialization in the operation's declared
    contract order. Selectors with a declared ``automatic_value`` are pinned
    to that concrete runtime value before the plan is hashed.
    """

    try:
        runner_id = normalize_materialization_runner(runner, "runner")
        planner_backend = (
            normalize_materialization_backend(backend, "backend")
            if backend is not None
            else None
        )
        operations = _validate_and_resolve_operations(
            recipe,
            registry,
            enforce_execution_profile=enforce_execution_profile,
        )
        materializations = build_materialization_registry(registry)
    except RecipePlanningError:
        raise
    except OperationError as exc:
        raise RecipePlanningError("Cannot build execution plan: %s" % exc) from exc

    operation_contracts: JsonDict = {}
    operation_contract_digests: Dict[str, str] = {}
    for operation_id in sorted({step.op for step in recipe.steps}):
        try:
            contract = operations[operation_id].describe()
        except OperationError as exc:
            raise RecipePlanningError(
                "Operation %s exposes an invalid contract: %s" % (operation_id, exc)
            ) from exc
        operation_contracts[operation_id] = copy.deepcopy(contract)
        operation_contract_digests[operation_id] = canonical_json_sha256(contract)

    planned_steps = []
    for step in recipe.steps:
        resolved, selection, parameter_overrides = _resolve_step_materialization(
            step,
            operations[step.op],
            materializations,
            runner=runner_id,
            planner_backend=planner_backend,
            implementation=implementation,
        )
        operation_identity = "%s.%s" % (
            resolved.operation.__class__.__module__,
            resolved.operation.__class__.__qualname__,
        )
        materialization_id = "%s@%s/%s/%s" % (
            resolved.operation_id,
            resolved.runner,
            resolved.backend,
            resolved.implementation,
        )
        implementation_metadata: JsonDict = {
            "operation_name": resolved.operation.name,
            "operation_class": operation_identity,
            "status": resolved.status,
            "selection": selection,
            **_operation_source_identity(resolved.operation),
        }
        selector_key = _backend_selector_key(
            step,
            operations[step.op],
            {item.backend for item in materializations.list(step.op, runner=runner_id)},
        )
        if selector_key is not None and selector_key not in parameter_overrides:
            parameter_overrides[selector_key] = _backend_selector_execution_value(
                step,
                operations[step.op],
                selector_key,
                resolved.backend,
                {
                    item.backend
                    for item in materializations.list(step.op, runner=runner_id)
                },
            )
        subordinate_selectors = _subordinate_backend_selector_evidence(
            step,
            operations[step.op],
            {
                item.backend
                for item in materializations.list(step.op, runner=runner_id)
            },
        )
        for subordinate_key, subordinate in subordinate_selectors.items():
            if not subordinate.get("automatic") or "effective_value" not in subordinate:
                continue
            effective_value = copy.deepcopy(subordinate["effective_value"])
            if (
                subordinate_key in parameter_overrides
                and not _json_values_equal(
                    parameter_overrides[subordinate_key],
                    effective_value,
                )
            ):
                raise RecipePlanningError(
                    "Step %s has conflicting planner bindings for params.%s"
                    % (step.id, subordinate_key)
                )
            parameter_overrides[subordinate_key] = effective_value
        execution_params = {
            **copy.deepcopy(dict(step.params)),
            **copy.deepcopy(parameter_overrides),
        }
        if parameter_overrides:
            try:
                validate_params(
                    step.op,
                    execution_params,
                    operations[step.op].params_schema,
                )
                operations[step.op].validate_preflight(
                    execution_params,
                    step.inputs,
                )
            except OperationError as exc:
                raise RecipePlanningError(
                    "Step %s params are invalid after binding %s/%s: %s"
                    % (
                        step.id,
                        resolved.backend,
                        resolved.implementation,
                        exc,
                    )
                ) from exc
            implementation_metadata["parameter_overrides"] = parameter_overrides
        availability = _validate_step_runtime_availability(
            step,
            operations[step.op],
            execution_params,
            runner=runner_id,
            backend=resolved.backend,
            implementation=resolved.implementation,
        )
        if availability is not None:
            implementation_metadata["runtime_availability"] = availability
        if subordinate_selectors:
            implementation_metadata["subordinate_runtime_selectors"] = (
                subordinate_selectors
            )
        if resolved.spec.notes:
            implementation_metadata["notes"] = resolved.spec.notes
        binding_payload = {
            "schema_version": PLANNED_STEP_SCHEMA_VERSION,
            "step_id": step.id,
            "operation_id": step.op,
            "runner": resolved.runner,
            "backend": resolved.backend,
            "implementation": resolved.implementation,
            "materialization_id": materialization_id,
            "implementation_identity": operation_identity,
            "implementation_metadata": implementation_metadata,
            "operation_contract_sha256": operation_contract_digests[step.op],
        }
        planned_steps.append(
            PlannedStep(
                step_id=step.id,
                operation_id=step.op,
                runner=resolved.runner,
                backend=resolved.backend,
                implementation=resolved.implementation,
                materialization_id=materialization_id,
                implementation_identity=operation_identity,
                implementation_metadata=implementation_metadata,
                operation_contract_sha256=operation_contract_digests[step.op],
                binding_sha256=canonical_json_sha256(binding_payload),
                operation=resolved.operation,
            )
        )

    contracts_sha256 = canonical_json_sha256(operation_contracts)
    plan_without_digest = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "kind": EXECUTION_PLAN_KIND,
        "runner": runner_id,
        "recipe": {
            "name": recipe.name,
            "sha256": recipe_fingerprint(recipe),
            "step_count": len(planned_steps),
        },
        "operation_contracts": {
            "schema_version": 1,
            "sha256": contracts_sha256,
            "operations": operation_contracts,
        },
        "steps": [step.to_dict() for step in planned_steps],
    }
    return ExecutionPlan(
        runner=runner_id,
        recipe_name=recipe.name,
        recipe_sha256=recipe_fingerprint(recipe),
        steps=tuple(planned_steps),
        operation_contracts=operation_contracts,
        operation_contracts_sha256=contracts_sha256,
        sha256=canonical_json_sha256(plan_without_digest),
    )


def _operation_source_identity(operation: Operation) -> JsonDict:
    source = inspect.getsourcefile(operation.__class__)
    if not source:
        raise RecipePlanningError(
            "Operation %s has no content-identifiable Python source" % operation.id
        )
    path = Path(source).resolve()
    if not path.is_file():
        raise RecipePlanningError(
            "Operation %s source file is missing: %s" % (operation.id, path)
        )
    return {
        "source_module": operation.__class__.__module__,
        "source_sha256": file_sha256(path),
    }


def validate_recipe_against_registry(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    enforce_execution_profile: bool = True,
) -> None:
    """Preserve the validation-only API without imposing a runner choice."""

    _validate_and_resolve_operations(
        recipe,
        registry,
        enforce_execution_profile=enforce_execution_profile,
    )


def validate_execution_plan_runtime_availability(
    recipe: Recipe,
    plan: ExecutionPlan,
) -> None:
    """Recheck volatile runtime dependencies for an immutable cached plan."""

    recipe_steps = {step.id: step for step in recipe.steps}
    for planned_step in plan.steps:
        step = recipe_steps.get(planned_step.step_id)
        if step is None:
            raise RecipePlanningError(
                "Cached execution plan references missing recipe step %s"
                % planned_step.step_id
            )
        _validate_step_runtime_availability(
            step,
            planned_step.operation,
            planned_step.params_for_execution(step.params),
            runner=planned_step.runner,
            backend=planned_step.backend,
            implementation=planned_step.implementation,
        )


def validate_recipe_execution_profile(recipe: Recipe) -> None:
    """Reject a non-conformant declared standard execution profile."""

    profile = inspect_execution_profile(recipe)
    if profile.reference.id == CUSTOM_EXECUTION_PROFILE_ID or not profile.issues:
        return
    details = "; ".join(
        "%s%s: %s"
        % (
            issue.code,
            " [%s]" % issue.step_id if issue.step_id else "",
            issue.message,
        )
        for issue in profile.issues
    )
    raise RecipePlanningError(
        "Recipe does not conform to execution profile `%s` version %s: %s"
        % (profile.reference.id, profile.reference.version, details)
    )


def _validate_and_resolve_operations(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    enforce_execution_profile: bool,
) -> Dict[str, Operation]:
    for step in recipe.steps:
        try:
            validate_recipe_step_id(step.id)
        except RecipeValidationError as exc:
            raise RecipePlanningError(str(exc)) from exc
    try:
        validate_recipe_research_metadata(recipe)
    except RecipeValidationError:
        raise
    except Exception as exc:
        raise RecipePlanningError("Recipe metadata is invalid: %s" % exc) from exc

    produced_kinds: Dict[str, Dict[str, str]] = {}
    produced_metadata: Dict[str, Dict[str, set[str]]] = {}
    operations: Dict[str, Operation] = {}

    for step in recipe.steps:
        try:
            operation = registry.get(step.op)
        except OperationError as exc:
            raise RecipePlanningError(
                "Step %s references unknown operation '%s'" % (step.id, step.op)
            ) from exc
        operations[step.op] = operation
        if operation.status != "implemented":
            raise RecipePlanningError(
                "Step %s uses operation %s with status %s"
                % (step.id, step.op, operation.status)
            )
        try:
            validate_params(step.op, step.params, operation.params_schema)
            operation.validate_preflight(step.params, step.inputs)
        except OperationError as exc:
            raise RecipePlanningError("Step %s params are invalid: %s" % (step.id, exc)) from exc

        required_inputs = dict(operation.input_kinds)
        optional_inputs = dict(getattr(operation, "optional_input_kinds", {}) or {})
        overlap = set(required_inputs).intersection(optional_inputs)
        if overlap:
            raise RecipePlanningError(
                "Operation %s declares inputs as both required and optional: %s"
                % (step.op, ", ".join(sorted(overlap)))
            )
        expected_inputs = {**required_inputs, **optional_inputs}
        for input_name in required_inputs:
            if input_name not in step.inputs:
                raise RecipePlanningError(
                    "Step %s requires input '%s' for operation %s"
                    % (step.id, input_name, step.op)
                )
        for input_name, reference in step.inputs.items():
            if input_name not in expected_inputs:
                raise RecipePlanningError(
                    "Step %s passes unexpected input '%s' to operation %s"
                    % (step.id, input_name, step.op)
                )
            producer_step_id, output_name = reference.split(".", 1)
            producer_outputs = produced_kinds.get(producer_step_id)
            if producer_outputs is None or output_name not in producer_outputs:
                raise RecipePlanningError(
                    "Step %s input '%s' references unavailable output %s"
                    % (step.id, input_name, reference)
                )
            actual_kind = producer_outputs[output_name]
            accepted_kinds = expected_inputs[input_name]
            if accepted_kinds and actual_kind not in accepted_kinds:
                raise RecipePlanningError(
                    "Step %s input '%s' expects one of %s but %s produces %s"
                    % (step.id, input_name, accepted_kinds, reference, actual_kind)
                )
            requirements = normalize_input_metadata_requirements(
                getattr(operation, "input_metadata_requirements", None),
                operation.id,
                input_names=set(expected_inputs),
            ).get(input_name)
            if requirements:
                guarantees = produced_metadata.get(producer_step_id, {}).get(
                    output_name,
                    set(),
                )
                _validate_input_metadata_contract(
                    step_id=step.id,
                    input_name=input_name,
                    reference=reference,
                    requirements=requirements,
                    guarantees=guarantees,
                )
        produced_kinds[step.id] = dict(operation.output_kinds)
        output_guarantees = normalize_output_metadata_guarantees(
            getattr(operation, "output_metadata_guarantees", None),
            operation.id,
            output_names=set(operation.output_kinds),
        )
        produced_metadata[step.id] = {
            output_name: set(output_guarantees.get(output_name) or [])
            for output_name in operation.output_kinds
        }

    if enforce_execution_profile:
        validate_recipe_execution_profile(recipe)
    return operations


def _validate_input_metadata_contract(
    *,
    step_id: str,
    input_name: str,
    reference: str,
    requirements: Mapping[str, Any],
    guarantees: set[str],
) -> None:
    all_of = set(requirements.get("all_of") or [])
    missing = sorted(all_of - guarantees)
    if missing:
        raise RecipePlanningError(
            "Step %s input '%s' requires producer metadata %s, but %s does not guarantee it"
            % (
                step_id,
                input_name,
                ", ".join("metadata.%s" % item for item in missing),
                reference,
            )
        )
    any_of = set(requirements.get("any_of") or [])
    if any_of and not any_of.intersection(guarantees):
        raise RecipePlanningError(
            "Step %s input '%s' requires one of producer metadata %s, but %s guarantees none"
            % (
                step_id,
                input_name,
                ", ".join("metadata.%s" % item for item in sorted(any_of)),
                reference,
            )
        )


def _validate_step_runtime_availability(
    step: RecipeStep,
    operation: Operation,
    params: Mapping[str, Any],
    *,
    runner: str,
    backend: str,
    implementation: str,
) -> Optional[JsonDict]:
    if runner not in {"benchmark_run", "dataset_capture"}:
        return None
    try:
        availability = operation.runtime_availability(params)
    except Exception as exc:
        raise RecipePlanningError(
            "Step %s could not inspect runtime availability for %s/%s: %s"
            % (step.id, backend, implementation, exc)
        ) from exc
    if availability is None:
        return None
    if not isinstance(availability, Mapping):
        raise RecipePlanningError(
            "Step %s operation %s returned an invalid runtime availability contract"
            % (step.id, step.op)
        )
    payload = copy.deepcopy(dict(availability))
    available = payload.get("available")
    if available is not None and not isinstance(available, bool):
        raise RecipePlanningError(
            "Step %s operation %s runtime availability `available` must be boolean"
            % (step.id, step.op)
        )
    if available is False:
        reason = str(
            payload.get("reason")
            or "required runtime dependencies are unavailable"
        )
        raise RecipePlanningError(
            "Step %s runtime is unavailable for %s/%s: %s"
            % (step.id, backend, implementation, reason)
        )
    return payload


def _resolve_step_materialization(
    step: RecipeStep,
    operation: Operation,
    materializations: MaterializationRegistry,
    *,
    runner: str,
    planner_backend: Optional[str],
    implementation: Optional[str],
) -> Tuple[ResolvedMaterialization, JsonDict, JsonDict]:
    try:
        candidates = materializations.list(
            operation_id=step.op,
            runner=runner,
            include_unimplemented=False,
        )
    except OperationError as exc:
        raise RecipePlanningError(
            "Step %s cannot inspect %s materializations: %s" % (step.id, runner, exc)
        ) from exc

    requested_backend = planner_backend
    source = "planner.backend" if planner_backend is not None else "declared_contract_order"
    raw_requested: Optional[str] = planner_backend
    if requested_backend is None:
        requested_backend, source, raw_requested = _step_backend_preference(
            step,
            operation,
            candidates,
        )

    backend_matching = [
        item
        for item in candidates
        if (requested_backend is None or item.backend == requested_backend)
        and (implementation is None or item.implementation == implementation)
    ]
    if not backend_matching:
        detail = ["runner=%s" % runner]
        if requested_backend is not None:
            detail.append("backend=%s" % requested_backend)
        if implementation is not None:
            detail.append("implementation=%s" % implementation)
        available = [
            "%s/%s" % (item.backend, item.implementation) for item in candidates
        ]
        raise RecipePlanningError(
            "Step %s (%s) has no implemented materialization for %s; available: %s"
            % (step.id, step.op, ", ".join(detail), ", ".join(available) or "none")
        )

    matching, selector_key = _matching_parameter_bound_materializations(
        step,
        operation,
        backend_matching,
    )
    explicit_materialization_override = planner_backend is not None or implementation is not None
    if not matching and explicit_materialization_override and len(backend_matching) == 1:
        # A unique explicit backend/implementation is authoritative. Its
        # declared parameter bindings are injected below so runtime dispatch
        # cannot disagree with the recorded plan.
        matching = list(backend_matching)
    if not matching:
        other_runner_candidates = [
            item
            for item in materializations.list(
                operation_id=step.op,
                include_unimplemented=False,
            )
            if (
                item.runner != runner
                and item.spec.parameter_bindings
                and (
                    requested_backend is None
                    or item.backend == requested_backend
                )
                and (
                    implementation is None
                    or item.implementation == implementation
                )
            )
        ]
        other_runner_matches, _ = _matching_parameter_bound_materializations(
            step,
            operation,
            other_runner_candidates,
        )
        if other_runner_matches:
            runner_names = sorted({item.runner for item in other_runner_matches})
            selected_bindings = []
            for item in other_runner_matches:
                binding = dict(item.spec.parameter_bindings)
                if binding not in selected_bindings:
                    selected_bindings.append(binding)
            raise RecipePlanningError(
                "Step %s (%s) params select %s, implemented only for runner=%s; "
                "runner=%s cannot execute that materialization"
                % (
                    step.id,
                    step.op,
                    ", ".join(str(item) for item in selected_bindings),
                    ", ".join(runner_names),
                    runner,
                )
            )
        selectors = [
            item.spec.parameter_bindings
            for item in backend_matching
            if item.spec.parameter_bindings
        ]
        if selectors:
            raise RecipePlanningError(
                "Step %s (%s) params do not select an implemented materialization for "
                "runner=%s; declared parameter bindings: %s"
                % (
                    step.id,
                    step.op,
                    runner,
                    ", ".join(str(dict(item)) for item in selectors),
                )
            )
        matching = list(backend_matching)

    # Registry order is operation-contract order and therefore stable for a
    # fixed operation contract. Do not silently apply lexical backend policy.
    resolved = matching[0]
    parameter_overrides = _thaw_json(resolved.spec.parameter_bindings)
    if (
        selector_key is not None
        and planner_backend is None
        and requested_backend is None
    ):
        source = "step.params.%s" % selector_key
    selection: JsonDict = {
        "source": source,
        "requested_backend": raw_requested,
        "resolved_backend": resolved.backend,
        "candidate_index": candidates.index(resolved),
    }
    if implementation is not None:
        selection["requested_implementation"] = implementation
    if selector_key is not None:
        selection["requested_parameter"] = {
            "name": selector_key,
            "value": copy.deepcopy(_effective_step_param(step, operation, selector_key)),
        }
    if parameter_overrides:
        selection["parameter_bindings"] = copy.deepcopy(parameter_overrides)
    return resolved, selection, parameter_overrides


def _matching_parameter_bound_materializations(
    step: RecipeStep,
    operation: Operation,
    candidates: Tuple[ResolvedMaterialization, ...] | list[ResolvedMaterialization],
) -> Tuple[list[ResolvedMaterialization], Optional[str]]:
    """Return candidates whose declared runtime-dispatch bindings match.

    A binding is an exact, type-sensitive JSON value. More-specific matching
    candidates win over unbound compatibility entries. Backend-style ``auto``
    remains handled by the backend selector contract rather than being guessed
    for arbitrary operation parameters.
    """

    matches: list[Tuple[ResolvedMaterialization, int, Optional[str]]] = []
    for candidate in candidates:
        specificity = 0
        first_selector: Optional[str] = None
        compatible = True
        for key, expected in candidate.spec.parameter_bindings.items():
            actual = _effective_step_param(step, operation, key)
            if not _json_values_equal(actual, expected):
                compatible = False
                break
            specificity += 1
            if first_selector is None:
                first_selector = key
        if compatible:
            matches.append((candidate, specificity, first_selector))
    if not matches:
        return [], None
    maximum_specificity = max(item[1] for item in matches)
    selected = [item for item in matches if item[1] == maximum_specificity]
    selector_key = next((item[2] for item in selected if item[2] is not None), None)
    return [item[0] for item in selected], selector_key


def _effective_step_param(step: RecipeStep, operation: Operation, key: str) -> Any:
    if key in step.params:
        return step.params[key]
    properties = (operation.params_schema or {}).get("properties") or {}
    schema = properties.get(key) if isinstance(properties, Mapping) else None
    if isinstance(schema, Mapping) and "default" in schema:
        return schema.get("default")
    return None


def _json_values_equal(left: Any, right: Any) -> bool:
    return canonical_json_sha256({"value": left}) == canonical_json_sha256({"value": right})


def _step_backend_preference(
    step: RecipeStep,
    operation: Operation,
    candidates: Tuple[ResolvedMaterialization, ...] | list[ResolvedMaterialization],
) -> Tuple[Optional[str], str, Optional[str]]:
    candidate_backends = {item.backend for item in candidates}
    properties = dict((operation.params_schema or {}).get("properties") or {})
    key = _backend_selector_key(step, operation, candidate_backends)
    if key is not None:
        schema = properties.get(key) if isinstance(properties.get(key), Mapping) else {}
        raw_value = step.params.get(key, schema.get("default"))
        raw = str(raw_value or "").strip()
        selector_contract = _backend_selector_contract(schema, candidate_backends)
        if selector_contract is not None:
            automatic_values = selector_contract["automatic_values"]
            if raw in automatic_values:
                automatic_value = selector_contract.get("automatic_value")
                if automatic_value is not None:
                    return (
                        selector_contract["mapping"][automatic_value],
                        "step.params.%s" % key,
                        raw,
                    )
                return None, "step.params.%s" % key, raw
            mapping = selector_contract["mapping"]
            if raw not in mapping:
                raise RecipePlanningError(
                    "Step %s has materialization selector params.%s=%r, which is not declared in %s"
                    % (step.id, key, raw_value, _MATERIALIZATION_SELECTOR_SCHEMA_KEY)
                )
            return mapping[raw], "step.params.%s" % key, raw
        normalized_raw = raw.lower().replace("-", "_")
        if normalized_raw in _AUTOMATIC_BACKEND_VALUES:
            return None, "step.params.%s" % key, raw
        try:
            selected = normalize_materialization_backend(raw, "step.params.%s" % key)
        except OperationError as exc:
            raise RecipePlanningError(
                "Step %s has invalid materialization backend in params.%s: %s"
                % (step.id, key, exc)
            ) from exc
        return selected, "step.params.%s" % key, raw
    return None, "declared_contract_order", None


def _backend_selector_key(
    step: RecipeStep,
    operation: Operation,
    candidate_backends: set[str],
) -> Optional[str]:
    properties = dict((operation.params_schema or {}).get("properties") or {})
    for key, raw_schema in properties.items():
        schema = raw_schema if isinstance(raw_schema, Mapping) else {}
        if _backend_selector_contract(schema, candidate_backends) is not None:
            return str(key)
    for key in _BACKEND_PARAM_KEYS:
        if key not in step.params and key not in properties:
            continue
        if key == "backend" and not _generic_backend_param_selects_materialization(
            properties.get(key),
            candidate_backends,
        ):
            continue
        return key
    return None


def _backend_selector_contract(
    schema: Mapping[str, Any],
    candidate_backends: set[str],
) -> Optional[JsonDict]:
    contract = _declared_backend_selector_contract(schema)
    if contract is None:
        return None
    # A subordinate selector (for example portable data-plane kernels inside
    # a NumPy-wrapped source) must not take ownership of a broader or narrower
    # top-level materialization choice. It is authoritative only when the two
    # backend domains correspond exactly.
    if candidate_backends != set(contract["mapping"].values()):
        return None
    return contract


def _declared_backend_selector_contract(
    schema: Mapping[str, Any],
) -> Optional[JsonDict]:
    raw_contract = schema.get(_MATERIALIZATION_SELECTOR_SCHEMA_KEY)
    if not isinstance(raw_contract, Mapping):
        return None
    if str(raw_contract.get("target") or "").strip() != "backend":
        return None
    raw_mapping = raw_contract.get("mapping")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise RecipePlanningError(
            "%s.mapping must be a non-empty object"
            % _MATERIALIZATION_SELECTOR_SCHEMA_KEY
        )
    mapping: JsonDict = {}
    for raw_value, raw_backend in raw_mapping.items():
        value = str(raw_value)
        try:
            mapping[value] = normalize_materialization_backend(
                raw_backend,
                "%s.mapping.%s" % (_MATERIALIZATION_SELECTOR_SCHEMA_KEY, value),
            )
        except OperationError as exc:
            raise RecipePlanningError(str(exc)) from exc
    raw_automatic = raw_contract.get("automatic_values") or []
    if not isinstance(raw_automatic, (list, tuple)):
        raise RecipePlanningError(
            "%s.automatic_values must be a list"
            % _MATERIALIZATION_SELECTOR_SCHEMA_KEY
        )
    automatic_value = raw_contract.get("automatic_value")
    if automatic_value is not None:
        automatic_value = str(automatic_value)
        if automatic_value not in mapping:
            raise RecipePlanningError(
                "%s.automatic_value must name a value declared in mapping"
                % _MATERIALIZATION_SELECTOR_SCHEMA_KEY
            )
    return {
        "mapping": mapping,
        "automatic_values": [str(value) for value in raw_automatic],
        "automatic_value": automatic_value,
    }


def _subordinate_backend_selector_evidence(
    step: RecipeStep,
    operation: Operation,
    candidate_backends: set[str],
) -> JsonDict:
    properties = dict((operation.params_schema or {}).get("properties") or {})
    evidence: JsonDict = {}
    for key, raw_schema in properties.items():
        schema = raw_schema if isinstance(raw_schema, Mapping) else {}
        contract = _declared_backend_selector_contract(schema)
        if contract is None or candidate_backends == set(contract["mapping"].values()):
            continue
        value = _effective_step_param(step, operation, str(key))
        normalized = str(value)
        automatic = normalized in contract["automatic_values"]
        effective_value = (
            contract.get("automatic_value") if automatic else normalized
        )
        target_backend = (
            contract["mapping"].get(effective_value)
            if effective_value is not None
            else None
        )
        if not automatic and target_backend is None:
            raise RecipePlanningError(
                "Step %s has runtime selector params.%s=%r, which is not declared in %s"
                % (step.id, key, value, _MATERIALIZATION_SELECTOR_SCHEMA_KEY)
            )
        evidence[str(key)] = {
            "scope": "subordinate",
            "requested_value": copy.deepcopy(value),
            "automatic": automatic,
            "target_backend": target_backend,
        }
        if automatic and effective_value is not None:
            evidence[str(key)]["effective_value"] = effective_value
    return evidence


def _backend_selector_execution_value(
    step: RecipeStep,
    operation: Operation,
    key: str,
    resolved_backend: str,
    candidate_backends: set[str],
) -> Any:
    properties = dict((operation.params_schema or {}).get("properties") or {})
    schema = properties.get(key) if isinstance(properties.get(key), Mapping) else {}
    selector_contract = _backend_selector_contract(schema, candidate_backends)
    if selector_contract is None:
        return resolved_backend
    mapping = selector_contract["mapping"]
    current = _effective_step_param(step, operation, key)
    current_key = str(current)
    automatic_value = selector_contract.get("automatic_value")
    if (
        current_key in selector_contract["automatic_values"]
        and automatic_value is not None
        and mapping.get(automatic_value) == resolved_backend
    ):
        return automatic_value
    if (
        current_key not in selector_contract["automatic_values"]
        and mapping.get(current_key) == resolved_backend
    ):
        return copy.deepcopy(current)
    choices = [
        value
        for value, backend in mapping.items()
        if backend == resolved_backend
    ]
    if not choices:
        raise RecipePlanningError(
            "Step %s selector params.%s cannot activate planned backend %s"
            % (step.id, key, resolved_backend)
        )
    return choices[0]


def _generic_backend_param_selects_materialization(
    schema: Any,
    candidate_backends: set[str],
) -> bool:
    if not isinstance(schema, Mapping):
        return False
    enum = schema.get("enum")
    if not isinstance(enum, (list, tuple)):
        return False
    normalized = {
        str(value).strip().lower().replace("-", "_") for value in enum
    }
    return bool(normalized.intersection(candidate_backends))


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_json(item) for item in value), key=repr))
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)
