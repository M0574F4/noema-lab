from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml
from yaml.resolver import BaseResolver


class StructuredInputError(ValueError):
    """Raised when an evidence-bearing YAML/JSON document is ambiguous."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> Dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: Dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise StructuredInputError(
                "YAML mapping keys must be strings at line %d"
                % (key_node.start_mark.line + 1)
            )
        if key in mapping:
            raise StructuredInputError(
                "Duplicate YAML mapping key `%s` at line %d"
                % (key, key_node.start_mark.line + 1)
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _reject_json_constant(value: str) -> Any:
    raise StructuredInputError(
        "JSON contains non-finite numeric constant `%s`" % value
    )


def _construct_unique_json_object(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise StructuredInputError("Duplicate JSON object key `%s`" % key)
        payload[key] = value
    return payload


def decode_strict_yaml_or_json(
    raw_text: str,
    *,
    input_format: str,
) -> Any:
    """Decode YAML/JSON text without erasing ambiguity or non-finite values."""

    normalized_format = str(input_format or "").strip().lower().lstrip(".")
    if normalized_format not in {"json", "yaml", "yml"}:
        raise StructuredInputError(
            "Structured input format must be json, yaml, or yml"
        )
    try:
        if normalized_format == "json":
            value = json.loads(
                raw_text,
                object_pairs_hook=_construct_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        else:
            value = yaml.load(raw_text, Loader=_UniqueKeySafeLoader)
    except StructuredInputError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError, RecursionError) as exc:
        message = str(exc)
        if "recursive node" in message.lower():
            message = "Recursive YAML aliases are not supported"
        raise StructuredInputError(message) from exc
    _reject_nonfinite_values(value, path="$", ancestors=set())
    return value


def load_strict_yaml_or_json(path: Path) -> Any:
    """Decode one evidence input without erasing duplicate keys or NaN/Inf."""

    return decode_strict_yaml_or_json(
        path.read_text(encoding="utf-8"),
        input_format="json" if path.suffix.lower() == ".json" else "yaml",
    )

def decode_strict_json(raw_text: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite values."""

    return decode_strict_yaml_or_json(raw_text, input_format="json")


def decode_strict_json_object(
    raw_text: str,
    *,
    label: str = "JSON input",
) -> Dict[str, Any]:
    """Decode one unambiguous JSON object.

    Metadata callers must not coerce arrays of pairs or other JSON values into
    mappings.  Keeping the object check beside the strict decoder gives every
    artifact loader the same fail-closed contract.
    """

    value = decode_strict_json(raw_text)
    if not isinstance(value, dict):
        raise StructuredInputError(
            "%s must contain a JSON object; found %s"
            % (label, _json_value_type(value))
        )
    return value


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _reject_nonfinite_values(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuredInputError("NaN or infinity is not allowed at %s" % path)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise StructuredInputError("Recursive YAML aliases are not supported at %s" % path)
        ancestors.add(identity)
        try:
            for key, child in value.items():
                _reject_nonfinite_values(
                    child,
                    path="%s.%s" % (path, key),
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise StructuredInputError("Recursive YAML aliases are not supported at %s" % path)
        ancestors.add(identity)
        try:
            for index, child in enumerate(value):
                _reject_nonfinite_values(
                    child,
                    path="%s[%d]" % (path, index),
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    raise StructuredInputError(
        "Unsupported YAML value type `%s` at %s; structured inputs must be JSON-compatible"
        % (type(value).__name__, path)
    )
