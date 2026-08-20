from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.capture import DatasetCaptureError, _load_npz_tap
from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_json_object,
)
from noema_lab.ops.metrics.bits import _load_bits
from noema_lab.ui.server import _artifact_preview


class MetadataJsonObjectContractTests(unittest.TestCase):
    INVALID_NON_OBJECTS = (
        ('[["role", "reference"]]', "array"),
        ('"reference"', "string"),
        ("42", "number"),
        ("null", "null"),
    )

    def test_shared_decoder_rejects_every_non_object_json_shape(self):
        for raw, value_type in self.INVALID_NON_OBJECTS:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    StructuredInputError,
                    r"Fixture metadata_json must contain a JSON object; found %s"
                    % value_type,
                ):
                    decode_strict_json_object(
                        raw,
                        label="Fixture metadata_json",
                    )

    def test_shared_decoder_still_rejects_duplicate_object_keys(self):
        with self.assertRaisesRegex(
            StructuredInputError,
            "Duplicate JSON object key `role`",
        ):
            decode_strict_json_object(
                '{"role":"reference","role":"candidate"}',
                label="Fixture metadata_json",
            )

    def test_core_artifact_and_capture_reject_array_of_pairs_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bits.npz"
            np.savez_compressed(
                path,
                bits=np.asarray([0, 1], dtype=np.uint8),
                metadata_json='[["bit_role", "reference"]]',
            )
            with self.assertRaisesRegex(
                ValueError,
                "NPZ metadata_json.*must contain a JSON object; found array",
            ):
                artifact("channel.bits.numpy", path)
            with self.assertRaisesRegex(
                DatasetCaptureError,
                "contains malformed metadata_json",
            ):
                _load_npz_tap(
                    {
                        "path": str(path),
                        "metadata": {"array": "bits"},
                    },
                    "source.bits",
                )

    def test_runtime_bit_loader_rejects_scalar_and_duplicate_metadata(self):
        cases = (
            ('"reference"', "must contain a JSON object; found string"),
            (
                '{"bit_role":"reference","bit_role":"candidate"}',
                "Duplicate JSON object key `bit_role`",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (raw, message) in enumerate(cases):
                with self.subTest(raw=raw):
                    path = Path(tmp) / ("bits_%d.npz" % index)
                    np.savez_compressed(
                        path,
                        bits=np.asarray([0, 1], dtype=np.uint8),
                        metadata_json=raw,
                    )
                    with self.assertRaisesRegex(StructuredInputError, message):
                        _load_bits(path, {})

    def test_ui_artifact_preview_rejects_non_object_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.npz"
            np.savez_compressed(
                path,
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                metadata_json="false",
            )
            with self.assertRaisesRegex(
                StructuredInputError,
                "Artifact preview metadata_json must contain a JSON object; "
                "found boolean",
            ):
                _artifact_preview(path)


if __name__ == "__main__":
    unittest.main()
