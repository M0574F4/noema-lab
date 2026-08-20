from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from noema_lab.core.execution_profiles import (
    ExecutionProfileDeclarationError,
    ExecutionProfileRef,
    custom_execution_profile_ref,
    execution_profile_ref_from_value,
)
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]

RECIPE_SCHEMA_VERSION = 1
RECIPE_FIELDS = {
    "schema_version",
    "name",
    "description",
    "execution_profile",
    "metadata",
    "dataset_capture",
    "suite",
    "steps",
}
RECIPE_STEP_FIELDS = {"id", "op", "params", "inputs", "description"}
RECIPE_COMPILE_MODES = {"compat", "strict"}
RECIPE_STEP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_OFDM_CHANNEL_SCENARIO_PARAMS = (
    "tdl_model",
    "ofdm_fft_size",
    "num_ofdm_symbols",
    "subcarrier_spacing_khz",
    "carrier_frequency_ghz",
    "delay_spread_ns",
    "mobility_kmh",
    "normalize_channel",
)


class RecipeValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RecipeDiagnostic:
    severity: str
    code: str
    message: str
    path: str = "$"

    def to_dict(self) -> JsonDict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class RecipeStep:
    id: str
    op: str
    params: JsonDict = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    extra_fields: JsonDict = field(default_factory=dict, repr=False)

    def to_dict(self) -> JsonDict:
        payload = copy.deepcopy(self.extra_fields)
        payload.update(
            {
                "id": self.id,
                "op": self.op,
                "params": copy.deepcopy(self.params),
                "inputs": copy.deepcopy(self.inputs),
            }
        )
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class Recipe:
    name: str
    steps: List[RecipeStep]
    description: Optional[str] = None
    schema_version: int = 1
    metadata: JsonDict = field(default_factory=dict)
    # Compatibility/materialization field only. Canonical scenario recipes keep
    # capture intent in noema.training_plan; generated capture jobs may embed it.
    dataset_capture: JsonDict = field(default_factory=dict)
    suite: JsonDict = field(default_factory=dict)
    execution_profile: ExecutionProfileRef = field(
        default_factory=custom_execution_profile_ref
    )
    extra_fields: JsonDict = field(default_factory=dict, repr=False)
    diagnostics: List[RecipeDiagnostic] = field(
        default_factory=list, repr=False, compare=False
    )

    def to_dict(self) -> JsonDict:
        payload = copy.deepcopy(self.extra_fields)
        payload.update(
            {
                "schema_version": self.schema_version,
                "name": self.name,
                "execution_profile": self.execution_profile.to_dict(),
                "metadata": copy.deepcopy(self.metadata),
                "steps": [step.to_dict() for step in self.steps],
            }
        )
        if self.description:
            payload["description"] = self.description
        if self.suite:
            payload["suite"] = copy.deepcopy(self.suite)
        if self.dataset_capture:
            payload["dataset_capture"] = copy.deepcopy(self.dataset_capture)
        return payload


@dataclass
class RecipeCompilation:
    raw: JsonDict
    mode: str
    recipe: Optional[Recipe]
    effective_recipe: Optional[Recipe]
    diagnostics: List[RecipeDiagnostic] = field(default_factory=list)
    defaults_materialized: bool = False

    @property
    def errors(self) -> List[RecipeDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self) -> List[RecipeDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return self.recipe is not None and not self.errors

    def raise_for_errors(self) -> "RecipeCompilation":
        if self.errors:
            raise RecipeValidationError("; ".join(item.message for item in self.errors))
        return self

    def require_recipe(self, *, effective: bool = False) -> Recipe:
        self.raise_for_errors()
        recipe = self.effective_recipe if effective else self.recipe
        if recipe is None:
            raise RecipeValidationError("Recipe compilation did not produce a recipe")
        return recipe

    def to_dict(self) -> JsonDict:
        return {
            "mode": self.mode,
            "status": "valid" if self.is_valid else "invalid",
            "defaults_materialized": self.defaults_materialized,
            "raw": copy.deepcopy(self.raw),
            "normalized": self.recipe.to_dict() if self.recipe is not None else None,
            "effective": self.effective_recipe.to_dict()
            if self.effective_recipe is not None
            else None,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def load_recipe(
    path: Path,
    *,
    mode: str = "compat",
    registry: Optional[Any] = None,
    effective: bool = False,
) -> Recipe:
    try:
        data = load_strict_yaml_or_json(path)
    except StructuredInputError as exc:
        raise RecipeValidationError(str(exc)) from exc
    return compile_recipe(
        data,
        mode=mode,
        registry=registry,
    ).require_recipe(effective=effective)


def compile_recipe(
    data: Any,
    *,
    mode: str = "compat",
    registry: Optional[Any] = None,
) -> RecipeCompilation:
    """Compile a raw v1 recipe into normalized and effective representations.

    ``data`` may be a raw mapping or an already parsed :class:`Recipe`.
    ``compat`` preserves v1 conveniences while reporting unknown fields and
    implicit step IDs as warnings. ``strict`` reports those conditions as
    errors. Unknown fields are retained in both modes so diagnostics never
    cause a lossy round-trip.

    When an operation registry is supplied, defaults declared in each
    operation's parameter schema are recursively materialized into the
    effective recipe. The normalized recipe always retains the author's
    original parameter choices.
    """

    if mode not in RECIPE_COMPILE_MODES:
        raise ValueError(
            "Recipe compile mode must be one of %s" % sorted(RECIPE_COMPILE_MODES)
        )
    if isinstance(data, Recipe):
        data = data.to_dict()
    if not isinstance(data, Mapping):
        diagnostic = RecipeDiagnostic(
            "error",
            "recipe_top_level_type",
            "Recipe must contain a mapping at the top level",
        )
        return RecipeCompilation({}, mode, None, None, [diagnostic])

    candidate = dict(data)
    try:
        _validate_json_value(candidate)
    except RecipeValidationError as exc:
        diagnostic = RecipeDiagnostic(
            "error",
            "recipe_json_value_invalid",
            str(exc),
        )
        return RecipeCompilation({}, mode, None, None, [diagnostic])

    raw = copy.deepcopy(candidate)
    diagnostics = _compile_shape_diagnostics(raw, mode)
    try:
        recipe = _recipe_from_dict(raw)
    except RecipeValidationError as exc:
        diagnostics.append(
            RecipeDiagnostic("error", "recipe_validation_error", str(exc))
        )
        return RecipeCompilation(raw, mode, None, None, diagnostics)

    effective_recipe = copy.deepcopy(recipe)
    defaults_materialized = False
    if registry is not None:
        _inherit_ofdm_channel_authority(effective_recipe)
        defaults_materialized = _materialize_operation_defaults(
            effective_recipe,
            registry,
            diagnostics,
        )
        _synchronize_ofdm_channel_contract(effective_recipe)

    recipe.diagnostics = list(diagnostics)
    effective_recipe.diagnostics = list(diagnostics)
    return RecipeCompilation(
        raw=raw,
        mode=mode,
        recipe=recipe,
        effective_recipe=effective_recipe,
        diagnostics=diagnostics,
        defaults_materialized=defaults_materialized,
    )


def _linked_ofdm_channel_pairs(recipe: Recipe):
    states = {step.id: step for step in recipe.steps if step.op == "wireless.ofdm_channel_state"}
    for channel in recipe.steps:
        if channel.op != "wireless.channel" or str(channel.params.get("channel") or "awgn") != "ofdm_tdl":
            continue
        reference = str(channel.inputs.get("channel_state") or "")
        state_id, separator, output_name = reference.partition(".")
        state = states.get(state_id)
        if state is not None and separator and output_name == "state":
            yield state, channel


def _inherit_ofdm_channel_authority(recipe: Recipe) -> None:
    """Migrate legacy CSI-owned scenario fields into the physical channel.

    This happens only in the effective recipe. Authored recipes therefore
    remain losslessly round-trippable while all execution paths use the
    physical channel as their single scenario authority.
    """

    for state, channel in _linked_ofdm_channel_pairs(recipe):
        for name in _OFDM_CHANNEL_SCENARIO_PARAMS:
            if name not in channel.params and name in state.params:
                channel.params[name] = copy.deepcopy(state.params[name])


def _synchronize_ofdm_channel_contract(recipe: Recipe) -> None:
    """Materialize one resolved OFDM scenario for CSI, channel, and allocator."""

    for state, channel in _linked_ofdm_channel_pairs(recipe):
        for name in _OFDM_CHANNEL_SCENARIO_PARAMS:
            if name in channel.params:
                state.params[name] = copy.deepcopy(channel.params[name])
        noise_mode = str(channel.params.get("noise_mode") or "snr_at_unit_power")
        if noise_mode == "fixed_variance":
            noise_variance = float(channel.params["noise_variance"])
        else:
            snr_db = float(channel.params.get("snr_db", 12.0))
            noise_variance = math.pow(10.0, -snr_db / 10.0)
        state.params["noise_variance"] = noise_variance
        recipe.metadata["resolved_channel_scenario"] = {
            "authority": channel.id,
            "channel_state": state.id,
            "noise_mode": noise_mode,
            "noise_variance": noise_variance,
            "reference_snr_db": -10.0 * math.log10(noise_variance),
        }


def recipe_from_dict(data: JsonDict) -> Recipe:
    """Parse a v1 recipe using compatibility-mode compiler semantics."""

    return compile_recipe(data, mode="compat").require_recipe()


def require_strict_recipe(recipe: Recipe) -> None:
    """Reject compatibility conveniences before an executable inspection."""

    strict_codes = {
        "unknown_recipe_field",
        "unknown_recipe_step_field",
        "implicit_recipe_step_id",
    }
    messages = [
        diagnostic.message
        for diagnostic in recipe.diagnostics
        if diagnostic.code in strict_codes
    ]
    strict_compilation = compile_recipe(recipe.to_dict(), mode="strict")
    messages.extend(item.message for item in strict_compilation.errors)
    unique = list(dict.fromkeys(messages))
    if unique:
        raise RecipeValidationError(
            "Recipe is not strict-compatible: %s" % "; ".join(unique)
        )


def validate_recipe_step_id(step_id: Any) -> str:
    """Return a step ID that is safe as one artifact-directory component."""

    if not isinstance(step_id, str) or not RECIPE_STEP_ID_RE.fullmatch(step_id):
        raise RecipeValidationError(
            "Recipe step id %r is unsafe for artifact storage; use an "
            "alphanumeric id containing only letters, digits, `_`, `-`, or `.`"
            % step_id
        )
    return step_id


def _recipe_from_dict(data: JsonDict) -> Recipe:
    schema_version = _schema_version(data)
    if schema_version != RECIPE_SCHEMA_VERSION:
        raise RecipeValidationError("Unsupported recipe schema_version: %s" % schema_version)
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise RecipeValidationError("Recipe requires a string 'name'")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise RecipeValidationError("Recipe requires a non-empty 'steps' list")

    steps: List[RecipeStep] = []
    seen = set()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise RecipeValidationError("Step %d must be a mapping" % index)
        step = _step_from_dict(dict(raw_step), index)
        if step.id in seen:
            raise RecipeValidationError("Duplicate step id: %s" % step.id)
        seen.add(step.id)
        steps.append(step)
    _validate_backward_references(steps)
    raw_metadata = data.get("metadata")
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise RecipeValidationError("Recipe metadata must be a mapping")
    raw_dataset_capture = data.get("dataset_capture")
    if raw_dataset_capture is None:
        raw_dataset_capture = {}
    # Continue accepting v1 authored recipes and executable capture-job recipes.
    # `neutral_recipe()` migrates this value into a separate training plan.
    if not isinstance(raw_dataset_capture, Mapping):
        raise RecipeValidationError("Recipe dataset_capture must be a mapping")
    raw_suite = data.get("suite")
    if raw_suite is None:
        raw_suite = {}
    if not isinstance(raw_suite, Mapping):
        raise RecipeValidationError("Recipe suite must be a mapping")
    try:
        execution_profile = execution_profile_ref_from_value(data.get("execution_profile"))
    except ExecutionProfileDeclarationError as exc:
        raise RecipeValidationError(str(exc)) from exc

    return Recipe(
        name=name,
        description=data.get("description"),
        schema_version=schema_version,
        metadata=dict(raw_metadata),
        dataset_capture=dict(raw_dataset_capture),
        suite=dict(raw_suite),
        execution_profile=execution_profile,
        steps=steps,
        extra_fields=_extra_fields(data, RECIPE_FIELDS),
    )


def _step_from_dict(data: JsonDict, index: int) -> RecipeStep:
    operation_id = data.get("op")
    if not isinstance(operation_id, str) or not operation_id:
        raise RecipeValidationError("Step %d requires a string 'op'" % index)
    step_id = data.get("id") if "id" in data else None
    if step_id is None or step_id == "":
        step_id = "step_%d" % (index + 1)
    if not isinstance(step_id, str):
        raise RecipeValidationError("Step %d has a non-string 'id'" % index)
    validate_recipe_step_id(step_id)
    inputs = data.get("inputs")
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise RecipeValidationError("Step %s inputs must be a mapping" % step_id)
    params = data.get("params")
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise RecipeValidationError("Step %s params must be a mapping" % step_id)
    return RecipeStep(
        id=step_id,
        op=operation_id,
        params=dict(params),
        inputs={str(key): str(value) for key, value in dict(inputs).items()},
        description=data.get("description"),
        extra_fields=_extra_fields(data, RECIPE_STEP_FIELDS),
    )


def _schema_version(data: Mapping[str, Any]) -> int:
    if "schema_version" not in data:
        return RECIPE_SCHEMA_VERSION
    value = data["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeValidationError("Recipe schema_version must be the integer 1")
    return value


def _compile_shape_diagnostics(data: JsonDict, mode: str) -> List[RecipeDiagnostic]:
    severity = "error" if mode == "strict" else "warning"
    diagnostics = [
        RecipeDiagnostic(
            severity,
            "unknown_recipe_field",
            "Unknown recipe field `%s`" % key,
            "$.%s" % key,
        )
        for key in data
        if key not in RECIPE_FIELDS
    ]
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        return diagnostics
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            continue
        diagnostics.extend(
            RecipeDiagnostic(
                severity,
                "unknown_recipe_step_field",
                "Unknown field `%s` on recipe step %d" % (key, index),
                "$.steps[%d].%s" % (index, key),
            )
            for key in raw_step
            if key not in RECIPE_STEP_FIELDS
        )
        if (
            "id" not in raw_step
            or raw_step.get("id") is None
            or raw_step.get("id") == ""
        ):
            diagnostics.append(
                RecipeDiagnostic(
                    severity,
                    "implicit_recipe_step_id",
                    "Recipe step %d requires an explicit non-empty `id` in strict mode"
                    % index,
                    "$.steps[%d].id" % index,
                )
            )
    return diagnostics


def _extra_fields(data: Mapping[str, Any], known_fields: set) -> JsonDict:
    return {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key not in known_fields
    }


def _validate_json_value(value: Any, path: str = "$") -> None:
    """Reject values that cannot be represented by strict JSON.

    This validation also covers programmatic recipe construction, where the
    duplicate-aware file decoders are not involved.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecipeValidationError(
                "Recipe value at %s must not be NaN or infinity" % path
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, "%s[%d]" % (path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecipeValidationError(
                    "Recipe object key at %s must be a string" % path
                )
            child_path = "%s.%s" % (path, key) if key else "%s['']" % path
            _validate_json_value(item, child_path)
        return
    raise RecipeValidationError(
        "Recipe value at %s contains unsupported %s"
        % (path, type(value).__name__)
    )


def _materialize_operation_defaults(
    recipe: Recipe,
    registry: Any,
    diagnostics: List[RecipeDiagnostic],
) -> bool:
    complete = True
    for index, step in enumerate(recipe.steps):
        try:
            operation = registry.get(step.op)
        except Exception as exc:
            complete = False
            diagnostics.append(
                RecipeDiagnostic(
                    "error",
                    "operation_defaults_unavailable",
                    "Cannot resolve defaults for step %s (%s): %s"
                    % (step.id, step.op, exc),
                    "$.steps[%d].op" % index,
                )
            )
            continue
        schema = getattr(operation, "params_schema", None)
        if not isinstance(schema, Mapping):
            complete = False
            diagnostics.append(
                RecipeDiagnostic(
                    "error",
                    "operation_params_schema_invalid",
                    "Operation %s does not expose a mapping params_schema" % step.op,
                    "$.steps[%d].params" % index,
                )
            )
            continue
        step.params = _value_with_schema_defaults(step.params, schema)
    return complete


def _value_with_schema_defaults(value: Any, schema: Mapping[str, Any]) -> Any:
    type_name = schema.get("type")
    if (type_name == "object" or "properties" in schema) and isinstance(value, Mapping):
        resolved = copy.deepcopy(dict(value))
        properties = dict(schema.get("properties") or {})
        for name, raw_property_schema in properties.items():
            if not isinstance(raw_property_schema, Mapping):
                continue
            property_schema = dict(raw_property_schema)
            if (
                name not in resolved
                and "default" in property_schema
                and _schema_parameter_is_effective(property_schema, resolved, properties)
            ):
                resolved[name] = copy.deepcopy(property_schema["default"])
            if name in resolved:
                resolved[name] = _value_with_schema_defaults(
                    resolved[name], property_schema
                )
        return resolved
    if type_name == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [
                _value_with_schema_defaults(item, item_schema) for item in value
            ]
    return copy.deepcopy(value)


def _schema_parameter_is_effective(
    property_schema: Mapping[str, Any],
    values: Mapping[str, Any],
    sibling_schemas: Mapping[str, Any],
) -> bool:
    conditions = property_schema.get("x-noema-effective-when")
    if not isinstance(conditions, Mapping) or not conditions:
        return True
    for controller, expected in conditions.items():
        sibling_schema = sibling_schemas.get(controller)
        default = (
            sibling_schema.get("default")
            if isinstance(sibling_schema, Mapping)
            else None
        )
        if values.get(controller, default) != expected:
            return False
    return True


def _validate_backward_references(steps: List[RecipeStep]) -> None:
    known = set()
    for step in steps:
        for input_name, reference in step.inputs.items():
            parts = reference.split(".")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise RecipeValidationError(
                    "Input %s on step %s must reference '<step_id>.<output_name>'"
                    % (input_name, step.id)
                )
            if parts[0] not in known:
                raise RecipeValidationError(
                    "Input %s on step %s references unknown or future step '%s'"
                    % (input_name, step.id, parts[0])
                )
        known.add(step.id)
