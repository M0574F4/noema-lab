import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


def _qpsk_capture_recipe(
    split: str,
    *,
    samples: int,
    explicit_seeds: bool = False,
    seed_mode: str = "increment_run_seed",
    shard_size: Optional[int] = None,
):
    source_params = {"bit_count": 256, "batch_size": 1}
    channel_params = {
        "channel": "awgn",
        "snr_db": 4,
        "wireless_backend": "numpy",
    }
    if explicit_seeds:
        source_params["seed"] = 17
        channel_params["seed"] = 19
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "capture_seed_contract",
            # Deliberately use the same master seed for every split. Split
            # isolation is a responsibility of Dataset Capture itself.
            "metadata": {"seed": 123},
            "dataset_capture": {
                "split": split,
                "samples": samples,
                "shard_size": shard_size or samples,
                "max_runs": samples,
                "seed_mode": seed_mode,
                "taps": [
                    {"id": "bits", "from": "source.bits"},
                    {"id": "rx_symbols", "from": "channel.rx_symbols"},
                ],
            },
            "steps": [
                {
                    "id": "source",
                    "op": "source.random_bits",
                    "params": source_params,
                },
                {
                    "id": "modulator",
                    "op": "modulation.digital_modulate",
                    "params": {"modulation": "qpsk"},
                    "inputs": {"bits": "source.bits"},
                },
                {
                    "id": "channel",
                    "op": "wireless.channel",
                    "params": channel_params,
                    "inputs": {"symbols": "modulator.symbols"},
                },
            ],
        }
    )


def _captured_arrays(path: Path):
    with np.load(str(path / "shards" / "shard_0000.npz"), allow_pickle=False) as shard:
        return np.array(shard["bits"], copy=True), np.array(
            shard["rx_symbols"], copy=True
        )


def _row_keys(values: np.ndarray):
    return {np.ascontiguousarray(row).tobytes() for row in values}


class CaptureSeedSemanticsTests(unittest.TestCase):
    def test_omitted_operation_seeds_vary_within_and_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(root / ".noema")
            registry = build_registry()
            captured = {}
            for split, samples in (("train", 3), ("validation", 2), ("test", 2)):
                out_dir = root / split
                run_dataset_capture_recipe(
                    _qpsk_capture_recipe(split, samples=samples),
                    registry,
                    store,
                    out_dir,
                )
                captured[split] = _captured_arrays(out_dir)
                schema = json.loads(
                    (out_dir / "schema.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    schema["seed_policy"]["capture_seed_namespace"],
                    "capture_seed_contract|dataset_capture_split=%s" % split,
                )
                for run in schema["runs"]:
                    effective = json.loads(
                        (Path(run["run_dir"]) / "recipe.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    params_by_step = {
                        step["id"]: step["params"] for step in effective["steps"]
                    }
                    self.assertNotIn("seed", params_by_step["source"])
                    self.assertNotIn("seed", params_by_step["channel"])

            for bits, rx_symbols in captured.values():
                self.assertEqual(len(_row_keys(bits)), int(bits.shape[0]))
                self.assertEqual(
                    len(_row_keys(rx_symbols)), int(rx_symbols.shape[0])
                )

            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                self.assertTrue(
                    _row_keys(captured[left][0]).isdisjoint(
                        _row_keys(captured[right][0])
                    )
                )
                self.assertTrue(
                    _row_keys(captured[left][1]).isdisjoint(
                        _row_keys(captured[right][1])
                    )
                )

    def test_explicit_operation_seeds_remain_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "train"
            run_dataset_capture_recipe(
                _qpsk_capture_recipe(
                    "train",
                    samples=2,
                    explicit_seeds=True,
                ),
                build_registry(),
                LocalStore(root / ".noema"),
                out_dir,
            )
            bits, rx_symbols = _captured_arrays(out_dir)

            self.assertTrue(np.array_equal(bits[0], bits[1]))
            self.assertTrue(np.array_equal(rx_symbols[0], rx_symbols[1]))

    def test_capture_seed_modes_are_not_affected_by_internal_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = build_registry()
            store = LocalStore(root / ".noema")

            fixed_dir = root / "fixed"
            run_dataset_capture_recipe(
                _qpsk_capture_recipe(
                    "train",
                    samples=2,
                    seed_mode="fixed_seed",
                ),
                registry,
                store,
                fixed_dir,
            )
            fixed_bits, fixed_rx = _captured_arrays(fixed_dir)
            self.assertTrue(np.array_equal(fixed_bits[0], fixed_bits[1]))
            self.assertTrue(np.array_equal(fixed_rx[0], fixed_rx[1]))

            per_shard_dir = root / "per_shard"
            run_dataset_capture_recipe(
                _qpsk_capture_recipe(
                    "train",
                    samples=4,
                    seed_mode="recipe_seed_plus_shard",
                    shard_size=2,
                ),
                registry,
                store,
                per_shard_dir,
            )
            first_bits, first_rx = _captured_arrays(per_shard_dir)
            with np.load(
                str(per_shard_dir / "shards" / "shard_0001.npz"),
                allow_pickle=False,
            ) as shard:
                second_bits = np.array(shard["bits"], copy=True)
                second_rx = np.array(shard["rx_symbols"], copy=True)
            self.assertTrue(np.array_equal(first_bits[0], first_bits[1]))
            self.assertTrue(np.array_equal(first_rx[0], first_rx[1]))
            self.assertTrue(np.array_equal(second_bits[0], second_bits[1]))
            self.assertTrue(np.array_equal(second_rx[0], second_rx[1]))
            self.assertFalse(np.array_equal(first_bits[0], second_bits[0]))
            self.assertFalse(np.array_equal(first_rx[0], second_rx[0]))


if __name__ == "__main__":
    unittest.main()
