from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import artifact


class ArtifactStrictMetadataTests(unittest.TestCase):
    def test_npz_metadata_rejects_duplicate_keys_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambiguous.npz"
            np.savez_compressed(
                path,
                values=np.asarray([1.0], dtype=np.float32),
                metadata_json='{"bit_count":1,"bit_count":2}',
            )

            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                artifact("test.array", path)

    def test_npz_metadata_requires_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scalar-metadata.npz"
            np.savez_compressed(
                path,
                values=np.asarray([1.0], dtype=np.float32),
                metadata_json='"not-an-object"',
            )

            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                artifact("test.array", path)


if __name__ == "__main__":
    unittest.main()
