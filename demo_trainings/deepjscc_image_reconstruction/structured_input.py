"""Strict structured-input decoding used when this template is inspected in-tree."""

from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_yaml_or_json,
    load_strict_yaml_or_json,
)

__all__ = [
    "StructuredInputError",
    "decode_strict_yaml_or_json",
    "load_strict_yaml_or_json",
]
