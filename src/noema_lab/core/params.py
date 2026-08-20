from __future__ import annotations

import re
import math
from typing import Any, Dict, Mapping

from noema_lab.core.operations import OperationError

JsonDict = Dict[str, Any]

_JSON_SCHEMA_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}


def validate_params_schema(op_id: str, schema: Mapping[str, Any]) -> None:
    """Validate the supported parameter-schema contract at registration.

    Noema intentionally implements a compact JSON-Schema subset.  Rejecting a
    malformed declaration here is safer than letting different surfaces make
    different guesses about it later.
    """

    if not isinstance(schema, Mapping):
        raise OperationError("%s params_schema must be an object" % op_id)
    _validate_schema_node(op_id, "params_schema", schema, require_object=True)


def validate_params(op_id: str, params: JsonDict, schema: Mapping[str, Any]) -> None:
    if not isinstance(params, dict):
        raise OperationError("%s params must be an object" % op_id)
    _validate_json_scalar_tree(op_id, "params", params)
    if schema.get("type") != "object":
        raise OperationError("%s params_schema.type must be object" % op_id)
    properties = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])
    for name in required:
        if name not in params:
            raise OperationError("%s requires parameter '%s'" % (op_id, name))
    if schema.get("additionalProperties") is False:
        for name in params:
            if name not in properties:
                raise OperationError("%s got unexpected parameter '%s'" % (op_id, name))
    for name, value in params.items():
        if name in properties:
            _validate_effective_parameter(op_id, name, params, properties)
            _validate_value(op_id, name, value, dict(properties[name]))


def _validate_effective_parameter(
    op_id: str,
    name: str,
    params: JsonDict,
    properties: Mapping[str, Any],
) -> None:
    property_schema = properties.get(name)
    if not isinstance(property_schema, Mapping):
        return
    conditions = property_schema.get("x-noema-effective-when")
    if not isinstance(conditions, Mapping) or not conditions:
        return
    mismatches = []
    for controller, expected in conditions.items():
        controller_schema = properties.get(controller)
        default = (
            controller_schema.get("default")
            if isinstance(controller_schema, Mapping)
            else None
        )
        actual = params.get(controller, default)
        if actual != expected:
            mismatches.append("%s=%r (requires %r)" % (controller, actual, expected))
    if mismatches:
        raise OperationError(
            "%s parameter '%s' is inactive under %s"
            % (op_id, name, ", ".join(mismatches))
        )


def _validate_value(op_id: str, name: str, value: Any, schema: JsonDict) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise OperationError(
            "%s parameter '%s' must be one of %s" % (op_id, name, schema["enum"])
        )
    type_name = schema.get("type")
    if type_name is None:
        return
    if isinstance(type_name, list):
        matching_types = [item for item in type_name if _matches_type(value, item)]
        if matching_types:
            _validate_typed_value(op_id, name, value, schema, matching_types[0])
            return
        raise OperationError("%s parameter '%s' has invalid type" % (op_id, name))
    if not _matches_type(value, type_name):
        raise OperationError("%s parameter '%s' must be %s" % (op_id, name, type_name))
    _validate_typed_value(op_id, name, value, schema, type_name)


def _validate_typed_value(
    op_id: str,
    name: str,
    value: Any,
    schema: JsonDict,
    type_name: str,
) -> None:
    if type_name in ("integer", "number"):
        if isinstance(value, float) and not math.isfinite(value):
            raise OperationError(
                "%s parameter '%s' must not be NaN or infinity" % (op_id, name)
            )
        if "minimum" in schema and value < schema["minimum"]:
            raise OperationError(
                "%s parameter '%s' must be >= %s" % (op_id, name, schema["minimum"])
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise OperationError(
                "%s parameter '%s' must be <= %s" % (op_id, name, schema["maximum"])
            )
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise OperationError(
                "%s parameter '%s' must be > %s" % (op_id, name, schema["exclusiveMinimum"])
            )
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise OperationError(
                "%s parameter '%s' must be < %s" % (op_id, name, schema["exclusiveMaximum"])
            )
    elif type_name == "string":
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise OperationError(
                "%s parameter '%s' must contain at least %s character(s)"
                % (op_id, name, schema["minLength"])
            )
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise OperationError(
                "%s parameter '%s' must contain at most %s character(s)"
                % (op_id, name, schema["maxLength"])
            )
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise OperationError(
                "%s parameter '%s' must match pattern %s" % (op_id, name, schema["pattern"])
            )
    elif type_name == "object":
        properties = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        for child_name in required:
            if child_name not in value:
                raise OperationError("%s requires parameter '%s.%s'" % (op_id, name, child_name))
        if schema.get("additionalProperties") is False:
            for child_name in value:
                if child_name not in properties:
                    raise OperationError(
                        "%s got unexpected parameter '%s.%s'" % (op_id, name, child_name)
                    )
        for child_name, child_value in value.items():
            if child_name in properties:
                _validate_value(
                    op_id,
                    "%s.%s" % (name, child_name),
                    child_value,
                    dict(properties[child_name]),
                )
    elif type_name == "array":
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise OperationError(
                "%s parameter '%s' must contain at least %s item(s)"
                % (op_id, name, schema["minItems"])
            )
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise OperationError(
                "%s parameter '%s' must contain at most %s item(s)"
                % (op_id, name, schema["maxItems"])
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_value(op_id, "%s[%d]" % (name, index), item, dict(item_schema))


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "null":
        return value is None
    return False


def _validate_schema_node(
    op_id: str,
    label: str,
    schema: Mapping[str, Any],
    *,
    require_object: bool = False,
) -> None:
    if not isinstance(schema, Mapping):
        raise OperationError("%s %s must be an object" % (op_id, label))
    type_value = schema.get("type")
    if require_object and type_value != "object":
        raise OperationError("%s %s.type must be object" % (op_id, label))
    type_names = type_value if isinstance(type_value, list) else [type_value]
    if type_value is not None:
        if (
            not type_names
            or any(not isinstance(item, str) or item not in _JSON_SCHEMA_TYPES for item in type_names)
            or len(set(type_names)) != len(type_names)
        ):
            raise OperationError("%s %s.type is invalid" % (op_id, label))

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise OperationError("%s %s.properties must be an object" % (op_id, label))
    for raw_name, child_schema in properties.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise OperationError("%s %s.properties has an empty/non-string name" % (op_id, label))
        if not isinstance(child_schema, Mapping):
            raise OperationError(
                "%s %s.properties.%s must be an object" % (op_id, label, raw_name)
            )
        _validate_schema_node(
            op_id,
            "%s.properties.%s" % (label, raw_name),
            child_schema,
        )

    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(item, str) or not item.strip() for item in required
    ):
        raise OperationError("%s %s.required must be a list of names" % (op_id, label))
    if len(set(required)) != len(required):
        raise OperationError("%s %s.required contains duplicates" % (op_id, label))
    unknown_required = sorted(set(required) - set(properties))
    if unknown_required:
        raise OperationError(
            "%s %s.required names undeclared properties: %s"
            % (op_id, label, ", ".join(unknown_required))
        )
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise OperationError("%s %s.additionalProperties must be boolean" % (op_id, label))

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise OperationError("%s %s.enum must be a non-empty list" % (op_id, label))
        for index, value in enumerate(enum):
            _validate_json_scalar_tree(op_id, "%s.enum[%d]" % (label, index), value)

    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise OperationError("%s %s.%s must be finite numeric" % (op_id, label, keyword))
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OperationError("%s %s.%s must be a non-negative integer" % (op_id, label, keyword))
    if "pattern" in schema:
        try:
            re.compile(str(schema["pattern"]))
        except re.error as exc:
            raise OperationError("%s %s.pattern is invalid: %s" % (op_id, label, exc)) from exc
    if "items" in schema:
        item_schema = schema["items"]
        if not isinstance(item_schema, Mapping):
            raise OperationError("%s %s.items must be an object" % (op_id, label))
        _validate_schema_node(op_id, "%s.items" % label, item_schema)
    if "default" in schema:
        default = schema["default"]
        _validate_json_scalar_tree(op_id, "%s.default" % label, default)
        # Reuse runtime semantics for defaults, except at the root where a
        # default object is not part of Noema's contract.
        if not require_object:
            _validate_value(op_id, "%s.default" % label, default, dict(schema))


def _validate_json_scalar_tree(op_id: str, label: str, value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OperationError("%s %s must not be NaN or infinity" % (op_id, label))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_scalar_tree(op_id, "%s[%d]" % (label, index), item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OperationError("%s %s object keys must be strings" % (op_id, label))
            _validate_json_scalar_tree(op_id, "%s.%s" % (label, key), item)
        return
    raise OperationError(
        "%s %s contains unsupported %s" % (op_id, label, type(value).__name__)
    )
