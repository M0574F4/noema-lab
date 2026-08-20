from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core import dataplane
from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.tensor import BitsToLatentsOperation
from noema_lab.ops.models import eflic


def _index_meta(width: int = 1) -> list[dict]:
    return [
        {"shape": [1], "bits_per_index": width if index == 0 else 1}
        for index in range(5)
    ]


def _entry(offset: int, count: int, *, meta: list[dict] | None = None) -> dict:
    return {
        "bit_offset": offset,
        "bit_count": count,
        "index_meta": list(meta or _index_meta()),
        "original_shape": [1, 1, 1, 3],
        "padded_shape": [1, 3, 1, 1],
    }


class ExactPayloadLengthTests(unittest.TestCase):
    def test_dataplane_bits_to_indices_rejects_short_and_trailing_bits(self):
        for count in (5, 7):
            with self.subTest(bit_count=count):
                with self.assertRaisesRegex(
                    OperationError,
                    "requires exactly 6 bits.*got %d" % count,
                ):
                    dataplane.bits_to_indices(
                        np.zeros(count, dtype=np.uint8),
                        3,
                        (2,),
                        8,
                        "mod",
                        "python_numpy",
                    )

    def test_bits_to_latents_rejects_short_and_trailing_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for count in (31, 33):
                with self.subTest(bit_count=count):
                    path = root / ("bits_%d.npz" % count)
                    np.savez_compressed(
                        path,
                        bits=np.zeros(count, dtype=np.uint8),
                        metadata_json=json.dumps(
                            {
                                "byte_count": 4,
                                "tensor_dtype": "float32",
                                "tensor_shape": [1],
                            }
                        ),
                    )
                    context = OperationContext(
                        recipe_name="length_contract",
                        step_id="decode_%d" % count,
                        params={"data_plane_backend": "python_numpy"},
                        inputs={
                            "bits": Artifact(
                                "channel.payload_bits.numpy",
                                path,
                                {},
                            )
                        },
                        run_dir=root,
                        step_dir=root / ("decode_%d" % count),
                    )
                    with self.assertRaisesRegex(
                        OperationError,
                        "requires exactly 32 bits; got %d" % count,
                    ):
                        BitsToLatentsOperation().run(context)

    def test_eflic_entry_table_rejects_out_of_range_gap_and_overlap(self):
        cases = (
            (
                "out_of_range",
                np.zeros(4, dtype=np.uint8),
                [_entry(0, 5)],
                "range \\[0, 5\\) exceeds the 4-bit payload",
            ),
            (
                "gap",
                np.zeros(10, dtype=np.uint8),
                [_entry(0, 5), _entry(6, 5)],
                "starts at bit 6, expected 5; it leaves a gap",
            ),
            (
                "overlap",
                np.zeros(10, dtype=np.uint8),
                [_entry(0, 5), _entry(4, 5)],
                "starts at bit 4, expected 5; it overlaps a prior entry",
            ),
        )
        for label, bits, entries, message in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(RuntimeError, message):
                    eflic._validated_eflic_payload_entries(
                        bits,
                        {"entries": entries},
                        "EF-LIC test payload",
                    )

    def test_eflic_entry_table_rejects_index_metadata_length_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "declares 5 bits but index_meta requires 6",
        ):
            eflic._validated_eflic_payload_entries(
                np.zeros(5, dtype=np.uint8),
                {"entries": [_entry(0, 5, meta=_index_meta(width=2))]},
                "EF-LIC test payload",
            )

    def test_eflic_unpacker_rejects_short_and_trailing_bits(self):
        for count in (4, 6):
            with self.subTest(bit_count=count):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires exactly 5 bits; got %d" % count,
                ):
                    eflic._unpack_inds_numpy(
                        np.zeros(count, dtype=np.uint8),
                        _index_meta(),
                        "python_numpy",
                    )

    def test_eflic_malformed_entry_table_cannot_trigger_zeros_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bits.npz"
            metadata = {
                "bit_count": 4,
                "payload_bit_count": 4,
                "entries": [_entry(0, 5)],
            }
            np.savez_compressed(
                path,
                bits=np.zeros(4, dtype=np.uint8),
                metadata_json=json.dumps(metadata),
            )
            context = OperationContext(
                recipe_name="eflic_length_contract",
                step_id="decode",
                params={"on_error": "zeros"},
                inputs={
                    "bits": Artifact(
                        "channel.payload_bits.numpy",
                        path,
                        {},
                    )
                },
                run_dir=root,
                step_dir=root / "decode",
            )
            with mock.patch.object(eflic, "_require_torch", return_value=object()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "range \\[0, 5\\) exceeds the 4-bit payload",
                ):
                    eflic.EfLicDecodeOperation().run(context)


if __name__ == "__main__":
    unittest.main()
