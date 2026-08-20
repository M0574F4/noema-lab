from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import OperationContext
from noema_lab.ops.metrics.image import ImageDeliveryStatusOperation


class ImageDeliveryStatusTests(unittest.TestCase):
    def test_finite_continuous_batch_owns_all_attempted_item_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images_path = root / "images.npz"
            images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
            np.savez_compressed(
                images_path,
                images=images,
                metadata_json=json.dumps(
                    {"source_item_ids": ["item-1", "item-2"]}
                ),
            )
            context = OperationContext(
                recipe_name="delivery_status_test",
                step_id="delivery_status",
                params={"on_decode_failure": "report_outage"},
                inputs={
                    "images": artifact(
                        "image.batch.numpy",
                        images_path,
                        {"source_item_ids": ["item-1", "item-2"]},
                    )
                },
                run_dir=root,
                step_dir=root / "delivery_status",
            )
            context.step_dir.mkdir()
            result = ImageDeliveryStatusOperation().run(context)
            report = result.outputs["report"].metadata
            self.assertEqual(report["attempted_source_item_count"], 2)
            self.assertEqual(report["source_item_outage"], [0, 0])
            self.assertEqual(
                report["denominator_policy"],
                "all attempted source items",
            )
            self.assertEqual(report["on_decode_failure"], "report_outage")
            self.assertEqual(result.metrics["channel.outage_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
