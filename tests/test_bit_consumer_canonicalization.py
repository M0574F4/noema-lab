from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.tensor import _load_bits as load_tensor_bits
from noema_lab.ops.foundation import _load_bits as load_foundation_bits
from noema_lab.ops.models.eflic import _load_bits as load_eflic_bits
from noema_lab.ops.models.external import (
    ExternalBitsDecodeOperation,
    ExternalBitsEncodeOperation,
    _example_count_from_metadata,
)
from noema_lab.ops.models.learned_codecs import (
    _load_bits as load_learned_codec_bits,
)
from noema_lab.ops.models.text_codec import _load_bits as load_text_codec_bits
from noema_lab.ops.models.upstream_lic import (
    _load_bits as load_upstream_lic_bits,
)
from noema_lab.ops.vqa_goal import _load_bits as load_vqa_bits


class BitConsumerCanonicalizationTests(unittest.TestCase):
    def test_all_public_bit_consumers_reject_nonbinary_uint8_values(self):
        loaders = (
            load_tensor_bits,
            load_foundation_bits,
            load_text_codec_bits,
            load_learned_codec_bits,
            load_upstream_lic_bits,
            load_eflic_bits,
            load_vqa_bits,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-bits.npz"
            np.savez_compressed(path, bits=np.asarray([0, 2, 1], dtype=np.uint8))
            for loader in loaders:
                with self.subTest(loader=loader.__module__):
                    with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                        loader(path, {})

    def test_external_decoder_rejects_noncanonical_input_before_adapter_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "invalid-bits.npz"
            np.savez_compressed(path, bits=np.asarray([0, 255, 1], dtype=np.uint8))
            context = OperationContext(
                recipe_name="strict_external_bits",
                step_id="decode",
                params={},
                inputs={"bits": Artifact("channel.payload_bits.numpy", path, {})},
                run_dir=root,
                step_dir=root / "decode",
            )
            with mock.patch(
                "noema_lab.ops.models.external._call_external"
            ) as adapter:
                with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                    ExternalBitsDecodeOperation().run(context)
            adapter.assert_not_called()

    def test_external_encoder_does_not_threshold_invalid_adapter_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images.npz"
            np.savez_compressed(
                image_path,
                images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            )
            context = OperationContext(
                recipe_name="strict_external_bits",
                step_id="encode",
                params={"bit_storage": "unpacked_bits"},
                inputs={"images": Artifact("image.batch.numpy", image_path, {})},
                run_dir=root,
                step_dir=root / "encode",
            )
            with mock.patch(
                "noema_lab.ops.models.external._call_external",
                return_value=np.asarray([0, 2, 1], dtype=np.uint8),
            ):
                with self.assertRaisesRegex(OperationError, "values 0 or 1"):
                    ExternalBitsEncodeOperation().run(context)

    def test_external_adapter_rejects_invalid_declared_record_count(self):
        array = np.zeros((2, 4), dtype=np.float32)
        for metadata, message in (
            ({"source_item_count": 0}, "greater than zero"),
            ({"source_item_count": "two"}, "must be an integer"),
            ({"original_shape": []}, "non-empty shape array"),
            ({"image_shape": ["two", 4]}, r"image_shape\[0\] must be an integer"),
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(OperationError, message):
                    _example_count_from_metadata(metadata, array)


if __name__ == "__main__":
    unittest.main()
