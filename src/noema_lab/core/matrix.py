from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from noema_lab.core.recipes import Recipe

JsonDict = Dict[str, Any]

MAX_MATRIX_VARIANTS = 256
MATRIX_VARIANT_ID_CONTRACT = "noema.recipe-matrix-selection.v1"
_MATRIX_FIELDS = {"dimensions", "step_params"}
_LEGACY_SWEEP_FIELDS = ("sweeps", "ui_sweeps")
_LEGACY_CATEGORICAL_PATHS = {
    "codecParams.encoder.model",
    "codecParams.encoder.checkpoint_preset",
}


class RecipeMatrixError(ValueError):
    """Raised when a recipe matrix cannot be normalized or expanded."""


@dataclass(frozen=True)
class MatrixDiagnostic:
    severity: str
    code: str
    message: str
    path: str

    def to_dict(self) -> JsonDict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class CanonicalRecipeMatrix:
    definition: JsonDict
    source: str
    diagnostics: Tuple[MatrixDiagnostic, ...] = field(default_factory=tuple)

    @property
    def enabled(self) -> bool:
        return bool(self.definition.get("dimensions"))

    @property
    def variant_count(self) -> int:
        dimensions = dict(self.definition.get("dimensions") or {})
        count = 1
        for values in dimensions.values():
            count *= len(values)
        return count if dimensions else 0

    def to_dict(self) -> JsonDict:
        return {
            "source": self.source,
            "enabled": self.enabled,
            "variant_count": self.variant_count,
            "matrix": copy.deepcopy(self.definition),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def matrix_variant_id(selection: Mapping[str, Any]) -> str:
    """Return the stable, filesystem-safe identity of one typed selection.

    The ID deliberately excludes the recipe display name and matrix index. It
    therefore survives harmless recipe renames and dimension reordering. JSON
    values are represented with explicit type tags before hashing so Python's
    equality aliases (notably ``True == 1``) cannot collapse distinct points.
    """

    if not isinstance(selection, Mapping):
        raise RecipeMatrixError("matrix selection must be a mapping")
    raw_names = list(selection)
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise RecipeMatrixError("matrix selection names must be non-empty strings")
    normalized = []
    for raw_name in sorted(raw_names):
        value = selection[raw_name]
        _validate_json_value(value, "metadata.matrix_selection.%s" % raw_name)
        normalized.append([raw_name, _typed_json_value(value)])
    canonical = json.dumps(
        {
            "contract": MATRIX_VARIANT_ID_CONTRACT,
            "selection": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "mxv1-%s" % hashlib.sha256(canonical).hexdigest()


def canonicalize_recipe_matrix(
    recipe: Recipe,
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> CanonicalRecipeMatrix:
    """Return one validated canonical matrix without mutating ``recipe``.

    ``metadata.matrix`` is authoritative. The v1 ``metadata.sweeps`` and
    ``metadata.ui_sweeps`` maps are accepted in compatibility mode and are
    translated into typed dimensions bound to concrete step parameters.
    Callers that persist public recipes can set ``strict_legacy=True`` to
    reject those compatibility fields.
    """

    if (
        isinstance(max_variants, bool)
        or not isinstance(max_variants, int)
        or max_variants <= 0
    ):
        raise ValueError("max_variants must be a positive integer")

    metadata = recipe.metadata if isinstance(recipe.metadata, Mapping) else {}
    diagnostics: List[MatrixDiagnostic] = []
    canonical_present = "matrix" in metadata and metadata.get("matrix") is not None
    legacy_present = [key for key in _LEGACY_SWEEP_FIELDS if key in metadata]

    if canonical_present:
        if legacy_present and strict_legacy:
            raise RecipeMatrixError(
                "%s is a compatibility field and cannot accompany metadata.matrix"
                % ", ".join("metadata.%s" % key for key in legacy_present)
            )
        if legacy_present:
            diagnostics.append(
                MatrixDiagnostic(
                    "warning",
                    "legacy_sweep_ignored",
                    "metadata.matrix is authoritative; ignored %s"
                    % ", ".join("metadata.%s" % key for key in legacy_present),
                    "$.metadata.matrix",
                )
            )
        definition = _validated_matrix_definition(
            metadata.get("matrix"), recipe, max_variants=max_variants
        )
        return CanonicalRecipeMatrix(
            definition=definition,
            source="matrix",
            diagnostics=tuple(diagnostics),
        )

    if not legacy_present:
        return CanonicalRecipeMatrix(definition={}, source="none")

    if strict_legacy:
        raise RecipeMatrixError(
            "%s is a compatibility field; use metadata.matrix"
            % ", ".join("metadata.%s" % key for key in legacy_present)
        )

    selected_key = legacy_present[0]
    if len(legacy_present) > 1:
        diagnostics.append(
            MatrixDiagnostic(
                "warning",
                "legacy_sweep_precedence",
                "Both metadata.sweeps and metadata.ui_sweeps are present; metadata.sweeps is authoritative",
                "$.metadata.sweeps",
            )
        )
        selected_key = "sweeps"
    diagnostics.append(
        MatrixDiagnostic(
            "warning",
            "legacy_sweep_normalized",
            "metadata.%s is deprecated and was normalized to metadata.matrix"
            % selected_key,
            "$.metadata.%s" % selected_key,
        )
    )
    definition = _legacy_sweeps_to_matrix(metadata.get(selected_key), recipe, selected_key)
    definition = _validated_matrix_definition(
        definition, recipe, max_variants=max_variants
    )
    return CanonicalRecipeMatrix(
        definition=definition,
        source=selected_key,
        diagnostics=tuple(diagnostics),
    )


def expand_recipe_matrix(
    recipe: Recipe,
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> JsonDict:
    """Expand a recipe matrix into deterministic concrete recipe payloads."""

    compiled = canonicalize_recipe_matrix(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )
    if not compiled.enabled:
        return {"recipe": recipe.name, "expanded_count": 0, "recipes": []}

    dimensions = dict(compiled.definition["dimensions"])
    step_param_templates = dict(compiled.definition["step_params"])
    names = sorted(dimensions)
    expanded = []
    for index, values in enumerate(product(*(dimensions[name] for name in names))):
        selected = dict(zip(names, values))
        expanded.append(
            _expanded_recipe_payload(
                recipe,
                selected,
                step_param_templates,
                matrix_index=index,
            )
        )
    return {
        "recipe": recipe.name,
        "expanded_count": len(expanded),
        "recipes": expanded,
    }


def materialize_recipe_matrix_selection(
    recipe: Recipe,
    selection: Mapping[str, Any],
    *,
    strict_legacy: bool = False,
    max_variants: int = MAX_MATRIX_VARIANTS,
) -> JsonDict:
    """Materialize one complete, typed matrix selection without changing ``recipe``.

    Additional coordinates are retained as provenance, but every dimension
    declared by the recipe must be selected from its authored value list.
    """

    if not isinstance(selection, Mapping):
        raise RecipeMatrixError("matrix selection must be a mapping")
    selected = copy.deepcopy(dict(selection))
    for raw_name, value in selected.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise RecipeMatrixError("matrix selection names must be non-empty strings")
        _validate_json_value(value, "metadata.matrix_selection.%s" % raw_name)

    compiled = canonicalize_recipe_matrix(
        recipe,
        strict_legacy=strict_legacy,
        max_variants=max_variants,
    )
    if not compiled.enabled:
        raise RecipeMatrixError("recipe has no matrix definition to materialize")

    dimensions = dict(compiled.definition["dimensions"])
    missing = sorted(set(dimensions) - set(selected))
    if missing:
        raise RecipeMatrixError(
            "matrix selection is missing dimension(s): %s" % ", ".join(missing)
        )

    known_selection: JsonDict = {}
    for name, allowed_values in dimensions.items():
        value = selected[name]
        if not any(_json_values_equal(value, allowed) for allowed in allowed_values):
            raise RecipeMatrixError(
                "matrix selection value for %s is not in the authored dimension" % name
            )
        known_selection[name] = copy.deepcopy(value)

    names = sorted(dimensions)
    matrix_index = next(
        index
        for index, values in enumerate(product(*(dimensions[name] for name in names)))
        if all(
            _json_values_equal(known_selection[name], value)
            for name, value in zip(names, values)
        )
    )
    return _materialized_recipe_payload(
        recipe,
        known_selection,
        dict(compiled.definition["step_params"]),
        matrix_index=matrix_index,
        metadata_selection=selected,
    )


def matrix_values_for_step_param(
    recipe: Recipe,
    step_id: str,
    param_name: str,
    *,
    strict_legacy: bool = False,
) -> Optional[List[Any]]:
    """Return dimension values directly bound to one step parameter, if any."""

    compiled = canonicalize_recipe_matrix(recipe, strict_legacy=strict_legacy)
    if not compiled.enabled:
        return None
    step_templates = dict(compiled.definition.get("step_params") or {}).get(str(step_id))
    if not isinstance(step_templates, Mapping) or param_name not in step_templates:
        return None
    marker = step_templates[param_name]
    if not _is_matrix_marker(marker):
        return None
    dimension = str(marker["matrix"])
    return copy.deepcopy(list(compiled.definition["dimensions"][dimension]))


def _validated_matrix_definition(
    raw: Any,
    recipe: Recipe,
    *,
    max_variants: int,
) -> JsonDict:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise RecipeMatrixError("metadata.matrix must be a mapping")
    unknown_fields = sorted(str(key) for key in raw if str(key) not in _MATRIX_FIELDS)
    if unknown_fields:
        raise RecipeMatrixError(
            "metadata.matrix has unknown field(s): %s" % ", ".join(unknown_fields)
        )

    raw_dimensions = raw.get("dimensions")
    if not isinstance(raw_dimensions, Mapping) or not raw_dimensions:
        raise RecipeMatrixError("metadata.matrix.dimensions must be a non-empty mapping")
    dimensions: JsonDict = {}
    variant_count = 1
    for raw_name, raw_values in raw_dimensions.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise RecipeMatrixError("metadata.matrix dimension names must be non-empty strings")
        name = raw_name.strip()
        if name in dimensions:
            raise RecipeMatrixError("metadata.matrix has duplicate dimension %s" % name)
        if not isinstance(raw_values, list) or not raw_values:
            raise RecipeMatrixError(
                "metadata.matrix.dimensions.%s must be a non-empty typed list" % name
            )
        typed_values: Dict[str, int] = {}
        for index, value in enumerate(raw_values):
            _validate_json_value(
                value,
                "metadata.matrix.dimensions.%s[%d]" % (name, index),
            )
            typed_key = _typed_json_key(value)
            if typed_key in typed_values:
                raise RecipeMatrixError(
                    "metadata.matrix.dimensions.%s contains duplicate typed value "
                    "at indices %d and %d"
                    % (name, typed_values[typed_key], index)
                )
            typed_values[typed_key] = index
        dimensions[name] = copy.deepcopy(raw_values)
        variant_count *= len(raw_values)
        if variant_count > max_variants:
            raise RecipeMatrixError(
                "metadata.matrix expands to %d variants; maximum is %d"
                % (variant_count, max_variants)
            )

    raw_step_params = raw.get("step_params")
    if not isinstance(raw_step_params, Mapping) or not raw_step_params:
        raise RecipeMatrixError("metadata.matrix.step_params must be a non-empty mapping")
    known_step_ids = {step.id for step in recipe.steps}
    step_params: JsonDict = {}
    referenced_dimensions: Set[str] = set()
    for raw_step_id, raw_templates in raw_step_params.items():
        step_id = str(raw_step_id)
        if step_id not in known_step_ids:
            raise RecipeMatrixError(
                "metadata.matrix.step_params references unknown step %s" % step_id
            )
        if not isinstance(raw_templates, Mapping) or not raw_templates:
            raise RecipeMatrixError(
                "metadata.matrix.step_params.%s must be a non-empty mapping" % step_id
            )
        normalized_templates = copy.deepcopy(dict(raw_templates))
        _collect_matrix_references(
            normalized_templates,
            dimensions=set(dimensions),
            referenced=referenced_dimensions,
            path="metadata.matrix.step_params.%s" % step_id,
        )
        step_params[step_id] = normalized_templates

    unused = sorted(set(dimensions) - referenced_dimensions)
    if unused:
        raise RecipeMatrixError(
            "metadata.matrix has unused dimension(s): %s" % ", ".join(unused)
        )
    return {"dimensions": dimensions, "step_params": step_params}


def _collect_matrix_references(
    value: Any,
    *,
    dimensions: Set[str],
    referenced: Set[str],
    path: str,
) -> None:
    if _is_matrix_marker(value):
        raw_name = value["matrix"]
        if not isinstance(raw_name, str) or not raw_name:
            raise RecipeMatrixError("%s.matrix must name a non-empty string dimension" % path)
        if raw_name not in dimensions:
            raise RecipeMatrixError(
                "%s references unknown matrix dimension %s" % (path, raw_name)
            )
        referenced.add(raw_name)
        return
    if isinstance(value, Mapping):
        if "matrix" in value:
            raise RecipeMatrixError(
                "%s uses an invalid matrix marker; {matrix: <dimension>} cannot have sibling fields"
                % path
            )
        for key, item in value.items():
            _collect_matrix_references(
                item,
                dimensions=dimensions,
                referenced=referenced,
                path="%s.%s" % (path, key),
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_matrix_references(
                item,
                dimensions=dimensions,
                referenced=referenced,
                path="%s[%d]" % (path, index),
            )


def _legacy_sweeps_to_matrix(raw: Any, recipe: Recipe, field_name: str) -> JsonDict:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise RecipeMatrixError("metadata.%s must be a mapping" % field_name)

    dimensions: JsonDict = {}
    step_params: JsonDict = {}
    targets: Dict[Tuple[str, str], str] = {}
    for raw_path, raw_spec in raw.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RecipeMatrixError(
                "metadata.%s paths must be non-empty strings" % field_name
            )
        path = raw_path.strip()
        step_id, param_name = _resolve_legacy_sweep_target(recipe, path)
        target = (step_id, param_name)
        if target in targets:
            raise RecipeMatrixError(
                "metadata.%s paths %s and %s target the same parameter %s.%s"
                % (field_name, targets[target], path, step_id, param_name)
            )
        targets[target] = path
        values = _legacy_sweep_values(path, raw_spec)
        if not values:
            raise RecipeMatrixError("metadata.%s.%s has no values" % (field_name, path))
        dimensions[path] = values
        step_params.setdefault(step_id, {})[param_name] = {"matrix": path}
    return {"dimensions": dimensions, "step_params": step_params}


def _resolve_legacy_sweep_target(recipe: Recipe, path: str) -> Tuple[str, str]:
    steps = {step.id: step for step in recipe.steps}
    aliases = {
        "codecParams.encoder.quality": ("codec", "quality"),
        "codecParams.encoder.vbr_scale_index": ("codec", "vbr_scale_index"),
        "codecParams.encoder.force_ind": ("codec", "force_ind"),
        "codecParams.encoder.model": ("codec", "model"),
        "codecParams.encoder.checkpoint_preset": ("codec", "checkpoint_preset"),
        "channel.snr_db": ("channel", "snr_db"),
        "channel.txPowerTarget": ("power", "target_power"),
    }
    alias = aliases.get(path)
    if alias is not None:
        kind, param_name = alias
        if kind == "codec":
            candidates = [step_id for step_id in ("codec_export", "sender") if step_id in steps]
            if candidates:
                return candidates[0], param_name
        elif kind == "channel":
            if "wireless_channel" in steps:
                return "wireless_channel", param_name
            candidates = [
                step.id
                for step in recipe.steps
                if step.op.startswith("wireless.")
                or step.op in {"channel.capacity_oracle_digital_link"}
            ]
            if len(candidates) == 1:
                return candidates[0], param_name
            if len(candidates) > 1:
                raise RecipeMatrixError(
                    "legacy sweep path %s is ambiguous across steps %s"
                    % (path, ", ".join(candidates))
                )
        elif kind == "power":
            if "tx_power" in steps:
                return "tx_power", param_name
            candidates = [
                step.id
                for step in recipe.steps
                if step.op
                in {
                    "channel.symbol_power_normalize",
                    "model.symbol_power_allocator",
                    "model.causal_csi_power_allocator",
                }
            ]
            if len(candidates) == 1:
                return candidates[0], param_name
            if len(candidates) > 1:
                raise RecipeMatrixError(
                    "legacy sweep path %s is ambiguous across steps %s"
                    % (path, ", ".join(candidates))
                )
        raise RecipeMatrixError("legacy sweep path %s has no compatible target step" % path)

    separator = path.find(".")
    if separator <= 0 or separator >= len(path) - 1:
        raise RecipeMatrixError(
            "legacy sweep path must use <step_id>.<param>, got %s" % path
        )
    step_id = path[:separator]
    param_name = path[separator + 1 :]
    if step_id not in steps:
        raise RecipeMatrixError("legacy sweep path references unknown step %s" % step_id)
    return step_id, param_name


def _legacy_sweep_values(path: str, value: Any) -> List[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return copy.deepcopy(list(value))
    if path in _LEGACY_CATEGORICAL_PATHS:
        if not isinstance(value, str):
            return [copy.deepcopy(value)]
        return _dedupe([item.strip() for item in value.split(",") if item.strip()])
    if not isinstance(value, str):
        return [copy.deepcopy(value)]

    text = value.strip()
    if not text:
        return []
    if "," in text:
        if ":" in text:
            raise RecipeMatrixError(
                "legacy numeric sweep must use either a:b:c or a,b,c syntax: %s" % text
            )
        parts = [item.strip() for item in text.split(",")]
        if any(not item for item in parts):
            raise RecipeMatrixError("legacy numeric sweep contains an empty value: %s" % text)
        try:
            return _dedupe([_clean_number(float(item)) for item in parts])
        except ValueError as exc:
            raise RecipeMatrixError("legacy numeric sweep contains a non-number: %s" % text) from exc
    if ":" not in text:
        try:
            return [_clean_number(float(text))]
        except ValueError as exc:
            raise RecipeMatrixError("legacy numeric sweep contains a non-number: %s" % text) from exc

    raw_parts = [item.strip() for item in text.split(":")]
    if len(raw_parts) not in {2, 3} or any(not item for item in raw_parts):
        raise RecipeMatrixError("legacy numeric range must use a:c or a:b:c syntax: %s" % text)
    try:
        parts = [float(item) for item in raw_parts]
    except ValueError as exc:
        raise RecipeMatrixError("legacy numeric range contains a non-number: %s" % text) from exc
    start = parts[0]
    step = 1.0 if len(parts) == 2 else parts[1]
    end = parts[1] if len(parts) == 2 else parts[2]
    if step == 0:
        raise RecipeMatrixError("legacy numeric range step must not be zero: %s" % text)
    values: List[Any] = []
    epsilon = abs(step) / 1e9
    current = start
    if step > 0:
        while current <= end + epsilon and len(values) <= MAX_MATRIX_VARIANTS:
            values.append(_clean_number(current))
            current += step
    else:
        while current >= end - epsilon and len(values) <= MAX_MATRIX_VARIANTS:
            values.append(_clean_number(current))
            current += step
    return _dedupe(values)


def _expanded_recipe_payload(
    recipe: Recipe,
    selected: JsonDict,
    step_param_templates: JsonDict,
    *,
    matrix_index: int,
) -> JsonDict:
    payload = _materialized_recipe_payload(
        recipe,
        selected,
        step_param_templates,
        matrix_index=matrix_index,
    )
    readable_suffix = _safe_selection_name_suffix(selected)
    payload["name"] = "%s__%s" % (
        recipe.name,
        readable_suffix or payload["metadata"]["matrix_variant_id"],
    )
    return payload


def _materialized_recipe_payload(
    recipe: Recipe,
    selected: JsonDict,
    step_param_templates: JsonDict,
    *,
    matrix_index: int,
    metadata_selection: Optional[JsonDict] = None,
) -> JsonDict:
    payload = recipe.to_dict()
    payload["metadata"] = dict(payload.get("metadata") or {})
    provenance_selection = selected if metadata_selection is None else metadata_selection
    payload["metadata"]["matrix_selection"] = copy.deepcopy(provenance_selection)
    payload["metadata"]["matrix_index"] = int(matrix_index)
    payload["metadata"]["matrix_variant_id"] = matrix_variant_id(provenance_selection)
    payload["metadata"].pop("matrix", None)
    payload["metadata"].pop("sweeps", None)
    payload["metadata"].pop("ui_sweeps", None)
    for step in payload["steps"]:
        templates = dict(step_param_templates.get(step["id"]) or {})
        for key, value in templates.items():
            step.setdefault("params", {})[key] = _resolve_matrix_value(value, selected)
    return payload


def _resolve_matrix_value(value: Any, selected: JsonDict) -> Any:
    if _is_matrix_marker(value):
        name = str(value["matrix"])
        if name not in selected:
            raise RecipeMatrixError("matrix template references unknown dimension %s" % name)
        return copy.deepcopy(selected[name])
    if isinstance(value, list):
        return [_resolve_matrix_value(item, selected) for item in value]
    if isinstance(value, Mapping):
        return {key: _resolve_matrix_value(item, selected) for key, item in value.items()}
    return copy.deepcopy(value)


def _is_matrix_marker(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {"matrix"}


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecipeMatrixError("%s must contain finite JSON values" % path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, "%s[%d]" % (path, index))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecipeMatrixError("%s object keys must be strings" % path)
            _validate_json_value(item, "%s.%s" % (path, key))
        return
    raise RecipeMatrixError("%s contains a non-JSON value of type %s" % (path, type(value).__name__))


def _json_values_equal(left: Any, right: Any) -> bool:
    return _typed_json_key(left) == _typed_json_key(right)


def _typed_json_key(value: Any) -> str:
    return json.dumps(
        _typed_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _typed_json_value(value: Any) -> Any:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["boolean", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return [
            "float",
            json.dumps(value, ensure_ascii=False, allow_nan=False),
        ]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, list):
        return ["array", [_typed_json_value(item) for item in value]]
    if isinstance(value, Mapping):
        return [
            "object",
            [
                [key, _typed_json_value(value[key])]
                for key in sorted(value)
            ],
        ]
    # All callers validate first; retain an explicit defensive error for direct
    # use by future consumers.
    raise RecipeMatrixError(
        "matrix value contains a non-JSON value of type %s"
        % type(value).__name__
    )


def _safe_selection_name_suffix(selection: Mapping[str, Any]) -> Optional[str]:
    rows: List[str] = []
    safe_name = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")
    safe_value = re.compile(r"^-?[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")
    for name in sorted(selection):
        value = selection[name]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (int, float)):
            rendered = json.dumps(value, allow_nan=False)
        elif isinstance(value, str):
            rendered = value
        else:
            return None
        if not safe_name.fullmatch(name) or not safe_value.fullmatch(rendered):
            return None
        rows.append("%s_%s" % (name, rendered))
    suffix = "__".join(rows)
    return suffix if suffix and len(suffix) <= 160 else None


def _clean_number(value: float) -> float | int:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return int(rounded)
    return float(round(value, 8))


def _dedupe(values: Sequence[Any]) -> List[Any]:
    rows: List[Any] = []
    for value in values:
        if value not in rows:
            rows.append(value)
    return rows
