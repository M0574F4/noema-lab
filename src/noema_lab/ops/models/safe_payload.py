"""Bounded, data-only serialization for transmitted codec payloads.

This module deliberately supports only a small tree of Python built-ins.  Its
decoder never imports modules, resolves names, or constructs application
classes.  In particular, pickle is not a legacy fallback: a payload without
the Noema magic marker is rejected before JSON parsing.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


WIRE_MAGIC = b"NOEMA_SAFE_DATA\x00"
WIRE_FORMAT = "noema.safe_data_json_base64"
WIRE_VERSION = 1
PAYLOAD_FORMAT = "%s.v%d" % (WIRE_FORMAT, WIRE_VERSION)


class SafePayloadError(ValueError):
    """The payload is unsupported, malformed, or exceeds a safety bound."""


@dataclass(frozen=True)
class PayloadLimits:
    """Hard resource bounds applied to both serialization and parsing."""

    max_wire_bytes: int = 256 * 1024 * 1024
    max_binary_bytes: int = 128 * 1024 * 1024
    max_string_bytes: int = 16 * 1024 * 1024
    max_items: int = 1_000_000
    max_depth: int = 64
    max_integer_digits: int = 128

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("%s must be a positive integer" % name)


DEFAULT_LIMITS = PayloadLimits()
_INTEGER_RE = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


def dumps(value: Any, *, limits: PayloadLimits = DEFAULT_LIMITS) -> bytes:
    """Serialize a bounded tree of dict/list/tuple/bytes/scalar values."""

    state = _Budget(limits)
    encoded = _encode_node(value, state, depth=0)
    envelope = {
        "format": WIRE_FORMAT,
        "version": WIRE_VERSION,
        "value": encoded,
    }
    try:
        document = json.dumps(
            envelope,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SafePayloadError("payload could not be encoded as canonical JSON: %s" % exc) from exc
    raw = WIRE_MAGIC + document
    if len(raw) > limits.max_wire_bytes:
        raise SafePayloadError(
            "encoded payload is %d bytes; limit is %d"
            % (len(raw), limits.max_wire_bytes)
        )
    return raw


def loads(raw: bytes, *, limits: PayloadLimits = DEFAULT_LIMITS) -> Any:
    """Parse a Noema safe payload without any executable-object fallback."""

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise SafePayloadError("payload must be bytes-like")
    wire = bytes(raw)
    if len(wire) > limits.max_wire_bytes:
        raise SafePayloadError(
            "encoded payload is %d bytes; limit is %d"
            % (len(wire), limits.max_wire_bytes)
        )
    if not wire.startswith(WIRE_MAGIC):
        raise SafePayloadError(
            "payload does not have the Noema safe-data magic marker; legacy pickle payloads are not accepted"
        )
    document = wire[len(WIRE_MAGIC) :]
    if not document:
        raise SafePayloadError("safe payload JSON document is empty")
    try:
        text = document.decode("utf-8", errors="strict")
        envelope = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except SafePayloadError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SafePayloadError("safe payload JSON is invalid: %s" % exc) from exc
    if not isinstance(envelope, dict) or set(envelope) != {"format", "version", "value"}:
        raise SafePayloadError("safe payload envelope must contain exactly format, version, and value")
    if envelope["format"] != WIRE_FORMAT:
        raise SafePayloadError("unsupported safe payload format: %r" % envelope["format"])
    if type(envelope["version"]) is not int or envelope["version"] != WIRE_VERSION:
        raise SafePayloadError("unsupported safe payload version: %r" % envelope["version"])
    state = _Budget(limits)
    return _decode_node(envelope["value"], state, depth=0)


class _Budget:
    def __init__(self, limits: PayloadLimits) -> None:
        self.limits = limits
        self.items = 0
        self.binary_bytes = 0

    def node(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise SafePayloadError(
                "payload nesting depth exceeds %d" % self.limits.max_depth
            )
        self.items += 1
        if self.items > self.limits.max_items:
            raise SafePayloadError(
                "payload item count exceeds %d" % self.limits.max_items
            )

    def binary(self, size: int) -> None:
        self.binary_bytes += size
        if self.binary_bytes > self.limits.max_binary_bytes:
            raise SafePayloadError(
                "decoded binary data exceeds %d bytes"
                % self.limits.max_binary_bytes
            )


def _utf8_size(value: str, limits: PayloadLimits) -> int:
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise SafePayloadError("payload strings must contain valid Unicode") from exc
    if size > limits.max_string_bytes:
        raise SafePayloadError(
            "payload string is %d bytes; limit is %d"
            % (size, limits.max_string_bytes)
        )
    return size


def _encode_node(value: Any, state: _Budget, depth: int) -> Dict[str, Any]:
    state.node(depth)
    if value is None:
        return {"t": "null"}
    if type(value) is bool:
        return {"t": "bool", "v": value}
    if type(value) is int:
        digits = str(value)
        if len(digits.lstrip("-")) > state.limits.max_integer_digits:
            raise SafePayloadError("integer exceeds digit limit")
        return {"t": "int", "v": digits}
    if type(value) is float:
        if not math.isfinite(value):
            raise SafePayloadError("non-finite floats are not permitted")
        return {"t": "float", "v": value.hex()}
    if type(value) is str:
        _utf8_size(value, state.limits)
        return {"t": "str", "v": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        binary = bytes(value)
        state.binary(len(binary))
        return {"t": "bytes", "v": base64.b64encode(binary).decode("ascii")}
    if isinstance(value, list):
        return {
            "t": "list",
            "v": [_encode_node(item, state, depth + 1) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "t": "tuple",
            "v": [_encode_node(item, state, depth + 1) for item in value],
        }
    if isinstance(value, dict):
        keys = list(value.keys())
        if not all(type(key) is str for key in keys):
            raise SafePayloadError("payload dictionaries require string keys")
        for key in keys:
            _utf8_size(key, state.limits)
        keys.sort()
        return {
            "t": "dict",
            "v": [
                [key, _encode_node(value[key], state, depth + 1)]
                for key in keys
            ],
        }
    raise SafePayloadError(
        "unsupported payload value type: %s" % type(value).__name__
    )


def _decode_node(node: Any, state: _Budget, depth: int) -> Any:
    state.node(depth)
    if not isinstance(node, dict):
        raise SafePayloadError("every encoded value must be a tagged object")
    tag = node.get("t")
    if type(tag) is not str:
        raise SafePayloadError("encoded value is missing a string type tag")
    if tag == "null":
        _require_keys(node, {"t"}, tag)
        return None
    _require_keys(node, {"t", "v"}, tag)
    value = node["v"]
    if tag == "bool":
        if type(value) is not bool:
            raise SafePayloadError("bool payload value must be a JSON boolean")
        return value
    if tag == "int":
        if type(value) is not str or not _INTEGER_RE.fullmatch(value):
            raise SafePayloadError("integer payload value is not canonical decimal")
        if len(value.lstrip("-")) > state.limits.max_integer_digits:
            raise SafePayloadError("integer exceeds digit limit")
        return int(value)
    if tag == "float":
        if type(value) is not str:
            raise SafePayloadError("float payload value must be a hexadecimal string")
        try:
            number = float.fromhex(value)
        except ValueError as exc:
            raise SafePayloadError("float payload value is invalid") from exc
        if not math.isfinite(number) or number.hex() != value:
            raise SafePayloadError("float payload value is non-finite or non-canonical")
        return number
    if tag == "str":
        if type(value) is not str:
            raise SafePayloadError("string payload value must be a JSON string")
        _utf8_size(value, state.limits)
        return value
    if tag == "bytes":
        if type(value) is not str:
            raise SafePayloadError("bytes payload value must be base64 text")
        try:
            encoded = value.encode("ascii")
        except UnicodeError as exc:
            raise SafePayloadError("bytes payload value is not ASCII base64") from exc
        if len(encoded) % 4:
            raise SafePayloadError("bytes payload value is not padded base64")
        padding = len(encoded) - len(encoded.rstrip(b"="))
        if padding > 2:
            raise SafePayloadError("bytes payload value has invalid base64 padding")
        decoded_size = (len(encoded) // 4) * 3 - padding
        state.binary(decoded_size)
        try:
            binary = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SafePayloadError("bytes payload value is not valid base64") from exc
        if len(binary) != decoded_size:
            raise SafePayloadError("bytes payload length does not match its base64 encoding")
        if base64.b64encode(binary).decode("ascii") != value:
            raise SafePayloadError("bytes payload value is not canonical base64")
        return binary
    if tag in {"list", "tuple"}:
        if not isinstance(value, list):
            raise SafePayloadError("%s payload value must be a JSON array" % tag)
        decoded = [_decode_node(item, state, depth + 1) for item in value]
        return decoded if tag == "list" else tuple(decoded)
    if tag == "dict":
        if not isinstance(value, list):
            raise SafePayloadError("dict payload value must be an array of entries")
        result: Dict[str, Any] = {}
        previous: str | None = None
        for entry in value:
            if not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) is not str:
                raise SafePayloadError("dict payload entries must be [string, value] pairs")
            key = entry[0]
            _utf8_size(key, state.limits)
            if previous is not None and key <= previous:
                raise SafePayloadError("dict payload keys must be unique and sorted")
            previous = key
            result[key] = _decode_node(entry[1], state, depth + 1)
        return result
    raise SafePayloadError("unknown payload type tag: %r" % tag)


def _require_keys(node: Dict[str, Any], expected: set[str], tag: str) -> None:
    if set(node) != expected:
        raise SafePayloadError(
            "%s payload value must contain exactly %s"
            % (tag, ", ".join(sorted(expected)))
        )


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafePayloadError("duplicate JSON object key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SafePayloadError("non-finite JSON number is not permitted: %s" % value)
