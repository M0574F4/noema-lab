from __future__ import annotations

import unittest

from noema_lab.core.operations import OperationError
from noema_lab.core.params import validate_params


class ParameterValidationTests(unittest.TestCase):
    def test_nested_object_rejects_unknown_and_missing_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {"width": {"type": "integer", "minimum": 1}},
                    "required": ["width"],
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }

        with self.assertRaisesRegex(OperationError, "model.width"):
            validate_params("test.nested", {"model": {}}, schema)
        with self.assertRaisesRegex(OperationError, "model.depth"):
            validate_params("test.nested", {"model": {"width": 8, "depth": 2}}, schema)

    def test_array_items_are_validated_recursively(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "levels": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                }
            },
            "additionalProperties": False,
        }

        validate_params("test.array", {"levels": [0.5, 1]}, schema)
        with self.assertRaisesRegex(OperationError, r"levels\[1\].*must be > 0"):
            validate_params("test.array", {"levels": [0.5, 0]}, schema)

    def test_string_constraints_are_enforced(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 3, "pattern": "^[a-z]+$"}
            },
            "additionalProperties": False,
        }

        validate_params("test.string", {"id": "codec"}, schema)
        with self.assertRaisesRegex(OperationError, "at least 3"):
            validate_params("test.string", {"id": "ab"}, schema)
        with self.assertRaisesRegex(OperationError, "must match pattern"):
            validate_params("test.string", {"id": "ABC"}, schema)

    def test_inactive_conditional_parameter_is_rejected(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "default": "fixed", "enum": ["fixed", "reference"]},
                "variance": {
                    "type": "number",
                    "x-noema-effective-when": {"mode": "fixed"},
                },
                "reference_db": {
                    "type": "number",
                    "x-noema-effective-when": {"mode": "reference"},
                },
            },
            "additionalProperties": False,
        }

        validate_params("test.conditional", {"mode": "fixed", "variance": 0.2}, schema)
        with self.assertRaisesRegex(OperationError, "reference_db.*inactive.*mode='fixed'"):
            validate_params(
                "test.conditional",
                {"mode": "fixed", "variance": 0.2, "reference_db": 10.0},
                schema,
            )


if __name__ == "__main__":
    unittest.main()
