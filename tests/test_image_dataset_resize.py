from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from noema_lab.core.operations import OperationContext
from noema_lab.ops import build_registry
from noema_lab.ops.source.image_dataset import (
    ImageDatasetOperation,
    _center_crop,
    _load_image_manifest,
    _parse_kodak_image_ids,
    _resize_shorter_side,
)


class ImageDatasetResizeTests(unittest.TestCase):
    def test_external_image_manifest_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.yaml"
            path.write_text(
                "schema_version: 999\nschema_version: 1\n"
                "kind: noema.image_dataset_manifest\nid: custom\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate YAML mapping key"):
                _load_image_manifest(path, expected_dataset="custom")

    def test_kodak_selection_rejects_duplicate_source_identities(self):
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _parse_kodak_image_ids("kodim01,kodim01")

    def test_center_crop_rejects_oversized_request_instead_of_clamping(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        with self.assertRaisesRegex(
            RuntimeError,
            "crop_size=4 exceeds the transformed image dimensions 3x5",
        ):
            _center_crop(image, 4)

    def test_schema_preserves_native_resolution_by_default(self):
        properties = build_registry().get("source.image_dataset").describe()[
            "params_schema"
        ]["properties"]
        self.assertEqual(properties["resize_shorter_side"]["default"], 0)
        self.assertEqual(properties["resize_shorter_side"]["minimum"], 0)

    def test_aspect_preserving_resize_is_deterministic_bilinear(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[:, :, 0] = np.asarray(
            [[0, 64, 128, 255], [255, 128, 64, 0]], dtype=np.uint8
        )
        image[:, :, 1] = 50
        image[:, :, 2] = 200

        first = _resize_shorter_side(image, 4)
        second = _resize_shorter_side(image, 4)

        self.assertEqual(first.shape, (4, 8, 3))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first[:, :, 0],
            np.asarray(
                [
                    [0, 16, 48, 80, 112, 160, 223, 255],
                    [64, 68, 76, 88, 104, 132, 171, 191],
                    [191, 171, 132, 104, 88, 76, 68, 64],
                    [255, 223, 160, 112, 80, 48, 16, 0],
                ],
                dtype=np.uint8,
            ),
        )

    def test_operation_resizes_before_center_crop_and_records_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "kodak"
            dataset_dir.mkdir()
            source = np.zeros((2, 4, 3), dtype=np.uint8)
            source[:, :2] = [10, 20, 30]
            source[:, 2:] = [210, 220, 230]
            Image.fromarray(source, mode="RGB").save(dataset_dir / "kodim01.png")
            step_dir = root / "artifacts" / "data"
            step_dir.mkdir(parents=True)

            with mock.patch(
                "noema_lab.ops.source.image_dataset.download_kodak_dataset",
                lambda _path: None,
            ):
                result = ImageDatasetOperation().run(
                    OperationContext(
                        recipe_name="resize_then_crop",
                        step_id="data",
                        params={
                            "dataset": "kodak",
                            "dataset_dir": str(dataset_dir),
                            "image_ids": "kodim01",
                            "resize_shorter_side": 4,
                            "crop_size": 4,
                        },
                        inputs={},
                        run_dir=root,
                        step_dir=step_dir,
                    )
                )

            output = result.outputs["images"]
            self.assertEqual(output.metadata["resize_shorter_side"], 4)
            self.assertEqual(output.metadata["crop_size"], 4)
            self.assertEqual(output.metadata["shape"], [1, 4, 4, 3])
            with np.load(output.path, allow_pickle=False) as payload:
                self.assertEqual(payload["images"].shape, (1, 4, 4, 3))


if __name__ == "__main__":
    unittest.main()
