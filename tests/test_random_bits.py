from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.operations import OperationContext
from noema_lab.ops.source.random_bits import (
    MAX_RANDOM_BITS_PER_OUTPUT,
    RandomBitsOperation,
)


class RandomBitsOperationTests(unittest.TestCase):
    def _run(self, root: Path, seed: int):
        step_dir = root / f"bits_{seed}"
        return RandomBitsOperation().run(
            OperationContext(
                recipe_name="seeded_random_bits_test",
                step_id="data",
                params={"bit_count": 32, "batch_size": 4, "seed": seed},
                inputs={},
                run_dir=root,
                step_dir=step_dir,
            )
        )

    def test_seeded_source_is_exactly_reproducible_and_records_block_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._run(root / "first", 23)
            second = self._run(root / "second", 23)
            different = self._run(root / "different", 24)
            with np.load(first.outputs["bits"].path, allow_pickle=False) as payload:
                first_bits = payload["bits"].copy()
            with np.load(second.outputs["bits"].path, allow_pickle=False) as payload:
                second_bits = payload["bits"].copy()
            with np.load(different.outputs["bits"].path, allow_pickle=False) as payload:
                different_bits = payload["bits"].copy()

        np.testing.assert_array_equal(first_bits, second_bits)
        self.assertFalse(np.array_equal(first_bits, different_bits))
        self.assertEqual(first_bits.dtype, np.uint8)
        self.assertEqual(first_bits.size, 128)
        self.assertTrue(np.all((first_bits == 0) | (first_bits == 1)))
        self.assertEqual(first.outputs["bits"].metadata["seed"], 23)
        self.assertEqual(first.outputs["bits"].metadata["bit_count_per_example"], 32)
        self.assertEqual(first.outputs["bits"].metadata["example_count"], 4)
        self.assertEqual(first.outputs["bits"].metadata["transport_block_size_bits"], 32)
        self.assertEqual(first.outputs["bits"].metadata["transport_block_count"], 4)
        self.assertEqual(first.outputs["bits"].metadata["payload_bit_count"], 128)

    def test_rejects_allocation_product_above_hard_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "must be at most"):
                RandomBitsOperation().run(
                    OperationContext(
                        recipe_name="oversized_random_bits",
                        step_id="data",
                        params={
                            "bit_count": MAX_RANDOM_BITS_PER_OUTPUT,
                            "batch_size": 2,
                        },
                        inputs={},
                        run_dir=root,
                        step_dir=root / "data",
                    )
                )


if __name__ == "__main__":
    unittest.main()
