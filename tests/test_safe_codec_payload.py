import json
import math
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.artifacts import artifact
from noema_lab.core.operations import OperationContext
from noema_lab.ops.models.learned_codecs import (
    JpegDecodeOperation,
    JpegEncodeOperation,
    _independent_item_payload_bits,
    _independent_item_payload_bytes,
    _torch_module_state_sha256,
)
from noema_lab.ops.models.codec_wire import (
    PAYLOAD_FORMAT as COMPACT_PAYLOAD_FORMAT,
    CodecWireError,
    dumps as compact_dumps,
    loads as compact_loads,
)
from noema_lab.ops.models.safe_payload import (
    PAYLOAD_FORMAT,
    WIRE_FORMAT,
    WIRE_MAGIC,
    PayloadLimits,
    SafePayloadError,
    dumps,
    loads,
)


class _PickleCommand:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, ("touch %s" % self.marker,)


class SafeCodecPayloadTests(unittest.TestCase):
    def test_torch_module_state_digest_is_stable_and_weight_sensitive(self):
        import torch

        first = torch.nn.Linear(3, 2)
        second = torch.nn.Linear(3, 2)
        second.load_state_dict(first.state_dict())
        first_digest = _torch_module_state_sha256(torch, first)
        self.assertEqual(first_digest, _torch_module_state_sha256(torch, second))
        self.assertRegex(first_digest, r"^[0-9a-f]{64}$")

        with torch.no_grad():
            second.weight[0, 0] += 1.0
        self.assertNotEqual(
            first_digest,
            _torch_module_state_sha256(torch, second),
        )

    def test_compact_codec_wire_round_trips_jpeg_and_compressai_payloads(self):
        jpeg_payload = {
            "codec": "jpeg",
            "payload_version": 2,
            "entry": {"data": b"\xff\xd8jpeg-entropy\xff\xd9"},
            "quality": 12,
        }
        jpeg_wire = compact_dumps(jpeg_payload)
        self.assertEqual(
            compact_loads(jpeg_wire)["entry"]["data"],
            jpeg_payload["entry"]["data"],
        )
        self.assertLess(len(jpeg_wire), len(dumps(jpeg_payload)))

        compressai_payload = {
            "codec": "compressai",
            "payload_version": 3,
            "model": "bmshj2018_factorized",
            "quality": 1,
            "metric": "mse",
            "pretrained": True,
            "vbr": {"enabled": False},
            "entry": {
                "strings": [[b"first"], [b"second", b""]],
                "shape": [8, 8],
                "original_shape": [1, 128, 128, 3],
            },
        }
        compressai_wire = compact_dumps(compressai_payload)
        self.assertEqual(compact_loads(compressai_wire), compressai_payload)
        self.assertLess(len(compressai_wire), len(dumps(compressai_payload)))

        with self.assertRaisesRegex(CodecWireError, "truncated"):
            compact_loads(compressai_wire[:-1])
        with self.assertRaisesRegex(CodecWireError, "unsupported"):
            compact_loads(b"not-a-codec-wire")

    def test_round_trip_preserves_supported_data_types(self):
        payload = {
            "none": None,
            "bools": [True, False],
            "integer": -(2**63),
            "floats": [0.0, -0.0, 1.25],
            "text": "Noema λ",
            "bytes": b"\x00\xffentropy-stream",
            "tuple": ("shape", 3, 5),
            "nested": {"items": [b"a", (b"b",)]},
        }

        wire = dumps(payload)

        self.assertTrue(wire.startswith(WIRE_MAGIC))
        self.assertEqual(loads(wire), payload)
        self.assertEqual(dumps(payload), wire)
        self.assertEqual(PAYLOAD_FORMAT, "%s.v1" % WIRE_FORMAT)

    def test_independent_item_framing_uses_safe_payload_bytes(self):
        payloads = [
            {"entry": {"data": b"first"}, "shape": (1, 2, 3)},
            {"entry": {"data": b"second"}, "shape": [1, 4, 5]},
        ]
        bits, bit_counts, byte_counts, backend = _independent_item_payload_bits(
            payloads, "python_numpy"
        )
        metadata = {"source_item_payload_bit_counts": bit_counts}

        recovered = _independent_item_payload_bytes(bits, metadata, "python_numpy")

        self.assertIsNotNone(recovered)
        rows, decoded_backend = recovered
        self.assertEqual(backend, "python_numpy")
        self.assertEqual(decoded_backend, "python_numpy")
        self.assertEqual(byte_counts, [len(item) for item in rows])
        self.assertTrue(all(item.startswith(WIRE_MAGIC) for item in rows))
        self.assertEqual([loads(item) for item in rows], payloads)

    def test_jpeg_codec_emits_and_consumes_safe_payload_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
            images[:, 4:12, 4:12, :] = [40, 180, 230]
            image_metadata = {
                "image_ids": ["image-0"],
                "original_shape": list(images.shape),
                "original_shapes": [[1, 16, 16, 3]],
                "storage_shape": list(images.shape),
            }
            image_path = root / "images.npz"
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )
            source = artifact("image.batch.numpy", image_path, image_metadata)

            encode_dir = root / "encode"
            encode_dir.mkdir()
            encoded = JpegEncodeOperation().run(
                OperationContext(
                    recipe_name="safe_payload_jpeg_smoke",
                    step_id="encode",
                    params={"quality": 75},
                    inputs={"images": source},
                    run_dir=root,
                    step_dir=encode_dir,
                )
            )
            self.assertEqual(encoded.outputs["bits"].metadata["payload_format"], PAYLOAD_FORMAT)

            decode_dir = root / "decode"
            decode_dir.mkdir()
            decoded = JpegDecodeOperation().run(
                OperationContext(
                    recipe_name="safe_payload_jpeg_smoke",
                    step_id="decode",
                    params={"on_error": "fail"},
                    inputs={"bits": encoded.outputs["bits"]},
                    run_dir=root,
                    step_dir=decode_dir,
                )
            )
            with np.load(decoded.outputs["images"].path, allow_pickle=False) as result:
                self.assertEqual(result["images"].shape, images.shape)
            self.assertEqual(
                decoded.outputs["images"].metadata["source_item_decode_success"],
                [True],
            )

    def test_jpeg_codec_emits_and_consumes_compact_wire_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.zeros((1, 16, 16, 3), dtype=np.uint8)
            images[:, 3:13, 3:13, :] = [210, 80, 25]
            image_metadata = {
                "image_ids": ["image-0"],
                "original_shape": list(images.shape),
                "original_shapes": [[1, 16, 16, 3]],
                "storage_shape": list(images.shape),
            }
            image_path = root / "images.npz"
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )
            source = artifact("image.batch.numpy", image_path, image_metadata)

            encode_dir = root / "encode"
            encode_dir.mkdir()
            encoded = JpegEncodeOperation().run(
                OperationContext(
                    recipe_name="compact_payload_jpeg_smoke",
                    step_id="encode",
                    params={"quality": 75, "wire_format": "compact_binary_v1"},
                    inputs={"images": source},
                    run_dir=root,
                    step_dir=encode_dir,
                )
            )
            self.assertEqual(
                encoded.outputs["bits"].metadata["payload_format"],
                COMPACT_PAYLOAD_FORMAT,
            )
            self.assertEqual(
                encoded.outputs["bits"].metadata[
                    "serialized_payload_rate_boundary"
                ],
                "noema_compact_codec_wire_bytes",
            )

            decode_dir = root / "decode"
            decode_dir.mkdir()
            decoded = JpegDecodeOperation().run(
                OperationContext(
                    recipe_name="compact_payload_jpeg_smoke",
                    step_id="decode",
                    params={"on_error": "fail"},
                    inputs={"bits": encoded.outputs["bits"]},
                    run_dir=root,
                    step_dir=decode_dir,
                )
            )
            with np.load(decoded.outputs["images"].path, allow_pickle=False) as result:
                self.assertEqual(result["images"].shape, images.shape)
            self.assertEqual(
                decoded.outputs["images"].metadata["source_item_decode_success"],
                [True],
            )

    def test_legacy_malicious_pickle_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "pickle_executed"
            malicious = pickle.dumps(_PickleCommand(str(marker)), protocol=4)

            with self.assertRaisesRegex(SafePayloadError, "legacy pickle"):
                loads(malicious)

            self.assertFalse(marker.exists())

    def test_unknown_tag_and_nonfinite_values_are_rejected(self):
        unknown = WIRE_MAGIC + json.dumps(
            {
                "format": WIRE_FORMAT,
                "version": 1,
                "value": {"t": "python-object", "v": "module.Class"},
            },
            separators=(",", ":"),
        ).encode("ascii")
        with self.assertRaisesRegex(SafePayloadError, "unknown payload type tag"):
            loads(unknown)

        with self.assertRaisesRegex(SafePayloadError, "unsupported payload value type"):
            dumps(_PickleCommand("unused"))

        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SafePayloadError, "non-finite"):
                    dumps(value)

    def test_depth_item_binary_and_wire_limits_are_enforced(self):
        with self.assertRaisesRegex(SafePayloadError, "nesting depth"):
            dumps([[[0]]], limits=PayloadLimits(max_depth=1))

        with self.assertRaisesRegex(SafePayloadError, "item count"):
            dumps([1, 2, 3], limits=PayloadLimits(max_items=3))

        with self.assertRaisesRegex(SafePayloadError, "binary data"):
            dumps(b"12345", limits=PayloadLimits(max_binary_bytes=4))

        wire = dumps(b"12345")
        with self.assertRaisesRegex(SafePayloadError, "binary data"):
            loads(wire, limits=PayloadLimits(max_binary_bytes=4))
        with self.assertRaisesRegex(SafePayloadError, "encoded payload"):
            loads(wire, limits=PayloadLimits(max_wire_bytes=len(wire) - 1))

    def test_duplicate_keys_and_noncanonical_base64_are_rejected(self):
        duplicate_envelope = (
            WIRE_MAGIC
            + b'{"format":"noema.safe_data_json_base64","format":"other",'
            b'"version":1,"value":{"t":"null"}}'
        )
        with self.assertRaisesRegex(SafePayloadError, "duplicate JSON object key"):
            loads(duplicate_envelope)

        noncanonical_bytes = WIRE_MAGIC + json.dumps(
            {
                "format": WIRE_FORMAT,
                "version": 1,
                "value": {"t": "bytes", "v": "YQ"},
            },
            separators=(",", ":"),
        ).encode("ascii")
        with self.assertRaisesRegex(SafePayloadError, "base64"):
            loads(noncanonical_bytes)


if __name__ == "__main__":
    unittest.main()
