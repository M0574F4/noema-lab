import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.source.local_npz import LocalNpzImagesOperation


class LocalNpzSourceContractTests(unittest.TestCase):
    def _run(self, root: Path, source: Path, **params):
        return LocalNpzImagesOperation().run(
            OperationContext(
                recipe_name="local_npz_contract",
                step_id="source",
                params={"path": str(source), **params},
                inputs={},
                run_dir=root,
                step_dir=root / "source",
            )
        )

    def test_non_uint8_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "images.npz"
            np.savez_compressed(
                source,
                images=np.asarray([[[[-1.2, 0.5, 300.0]]]], dtype=np.float32),
            )
            with self.assertRaisesRegex(OperationError, "must use uint8"):
                self._run(root, source)

    def test_explicit_lossy_conversion_is_recorded(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "images.npz"
            values = np.asarray([[[[-1.2, 1.5, 300.0]]]], dtype=np.float32)
            np.savez_compressed(source, images=values)
            result = self._run(root, source, dtype_policy="clip_round_uint8")
            artifact = result.outputs["images"]
            self.assertEqual(artifact.metadata["source_dtype"], "float32")
            self.assertEqual(artifact.metadata["dtype"], "uint8")
            self.assertTrue(artifact.metadata["dtype_coerced"])
            self.assertEqual(artifact.metadata["dtype_policy"], "clip_round_uint8")
            with np.load(artifact.path) as payload:
                self.assertEqual(payload["images"].tolist(), [[[[0, 2, 255]]]])
                embedded = json.loads(str(payload["metadata_json"]))
            self.assertTrue(embedded["dtype_coerced"])


if __name__ == "__main__":
    unittest.main()
