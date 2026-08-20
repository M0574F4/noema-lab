from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.operations import OperationError
from noema_lab.ops.portable_onnx import (
    infer_power_scores_onnx,
    load_portable_onnx_component,
    project_power_scores,
)


class PortableOnnxTests(unittest.TestCase):
    def test_power_policy_has_architecture_neutral_scores_and_noema_owned_constraint(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception as exc:  # pragma: no cover - optional dependency gate
            self.skipTest("onnx is unavailable: %s" % exc)

        graph = helper.make_graph(
            [helper.make_node("Identity", ["channel_gain"], ["allocation_scores"])],
            "identity_power_policy",
            [
                helper.make_tensor_value_info(
                    "channel_gain", TensorProto.FLOAT, ["batch", "subcarrier"]
                ),
                helper.make_tensor_value_info(
                    "noise_variance", TensorProto.FLOAT, ["batch", 1]
                ),
                helper.make_tensor_value_info(
                    "average_power_budget", TensorProto.FLOAT, ["batch", 1]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "allocation_scores", TensorProto.FLOAT, ["batch", "subcarrier"]
                )
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=9,
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.onnx"
            onnx.save_model(model, str(path))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            component = load_portable_onnx_component(
                str(path),
                sha,
                expected_inputs=(
                    "channel_gain",
                    "noise_variance",
                    "average_power_budget",
                ),
                expected_outputs=("allocation_scores",),
            )
            gains = np.asarray([[0.1, 1.0, 3.0], [4.0, 0.5, 0.2]], dtype=np.float32)
            scores = infer_power_scores_onnx(component, gains, 0.2, 1.5)
            allocation = project_power_scores(scores, 1.5)
            self.assertEqual(allocation.shape, gains.shape)
            self.assertTrue(np.all(allocation >= 0.0))
            np.testing.assert_allclose(allocation.sum(axis=1), [4.5, 4.5], atol=1e-7)

    def test_hash_mismatch_is_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "invalid.onnx"
            path.write_bytes(b"not an onnx graph")
            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                load_portable_onnx_component(str(path), "0" * 64)


if __name__ == "__main__":
    unittest.main()
