from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone
from pathlib import Path

from noema_lab.core.reproducibility import canonical_json_sha256


class CanonicalContentIdentityTests(unittest.TestCase):
    """Checked-in vectors for Noema's exact representation identity."""

    def test_fixed_utf8_json_vector_is_independent_of_map_insertion_order(
        self,
    ) -> None:
        expected = (
            "89dc83258fab57bac7978de59831eb8f64273d5ef2a43684f462dc58e8afef0e"
        )
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": [True, None, "μ"]}),
            expected,
        )
        self.assertEqual(
            canonical_json_sha256({"b": [True, None, "μ"], "a": 1}),
            expected,
        )

    def test_sequences_remain_order_sensitive(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"steps": ["a", "b"]}),
            "64f58a2ced5051f0ab9e7ec9d61a3726d5135c409d95ae77452cb50fd0e4cc59",
        )
        self.assertEqual(
            canonical_json_sha256({"steps": ["b", "a"]}),
            "423713705a4e9c663f7ef57ae7a820bffe60f1eead18ac83cf64aad8dbbc11a5",
        )

    def test_no_undeclared_semantic_equivalence_is_applied(self) -> None:
        distinct_pairs = (
            ({"n": 1}, {"n": 1.0}),
            ({"n": -0.0}, {"n": 0.0}),
            ({}, {"x": None}),
            ({"text": "é"}, {"text": "e\u0301"}),
        )
        for left, right in distinct_pairs:
            with self.subTest(left=left, right=right):
                self.assertNotEqual(
                    canonical_json_sha256(left),
                    canonical_json_sha256(right),
                )

        self.assertEqual(
            canonical_json_sha256({"n": -0.0}),
            "a8a313cade05001e69f7ddb5db01e1e2d06fb8f6913ab492cc4506d4e65d465a",
        )
        self.assertEqual(
            canonical_json_sha256({"text": "é"}),
            "42d3cbf59fdccced04e5dff14433fb52d34d58e385e9770ffd896ff517d63b92",
        )

    def test_non_json_and_nonfinite_values_fail_closed(self) -> None:
        invalid = (
            {"value": math.nan},
            {"value": math.inf},
            {"value": -math.inf},
            {"value": Path("artifact.bin")},
            {"value": datetime(2026, 7, 29, tzinfo=timezone.utc)},
            {"value": (1, 2)},
            {1: "non-string-key"},
        )
        for payload in invalid:
            with self.subTest(payload=repr(payload)):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_json_sha256(payload)


if __name__ == "__main__":
    unittest.main()
