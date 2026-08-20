from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext
from noema_lab.core.recipes import load_recipe
from noema_lab.ops.channel.digital import WirelessChannelOperation
from noema_lab.training.differentiability import sionna_available


ROOT = Path(__file__).resolve().parents[1]
SIONNA_AVAILABLE = sionna_available()


class DeepJsccSionnaChannelTests(unittest.TestCase):
    def test_template_uses_explicit_sionna_awgn_benchmark_path(self):
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml")
        channel = next(step for step in recipe.steps if step.id == "wireless_channel")
        self.assertEqual(channel.op, "wireless.channel")
        self.assertEqual(channel.params["channel"], "awgn")
        self.assertEqual(channel.params["wireless_backend"], "sionna")
        self.assertEqual(channel.params["seed"], 23001)

        default = load_recipe(ROOT / "recipes" / "compressai_kodak_default.yaml")
        default_channel = next(step for step in default.steps if step.id == "wireless_channel")
        self.assertEqual(default_channel.op, "channel.identity_symbol_link")
        self.assertFalse(default.metadata["channel_enabled"])

    @unittest.skipUnless(SIONNA_AVAILABLE, "Sionna wireless extra is not installed")
    def test_sionna_awgn_honors_recipe_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            metadata = {"power_unit": "normalized"}
            np.savez_compressed(
                symbols_path,
                symbols=np.ones(128, dtype=np.complex64),
                metadata_json=json.dumps(metadata),
            )
            source = Artifact("channel.symbols.complex_numpy", symbols_path, metadata)

            def realize(seed: int, label: str) -> tuple[np.ndarray, str]:
                step_dir = root / label
                step_dir.mkdir()
                result = WirelessChannelOperation().run(
                    OperationContext(
                        recipe_name="deepjscc_sionna_seed_test",
                        step_id="wireless_channel",
                        params={
                            "channel": "awgn",
                            "snr_db": 10.0,
                            "wireless_backend": "sionna",
                            "seed": seed,
                        },
                        inputs={"symbols": source},
                        run_dir=root,
                        step_dir=step_dir,
                    )
                )
                with np.load(result.outputs["rx_symbols"].path, allow_pickle=False) as payload:
                    symbols = np.asarray(payload["symbols"], dtype=np.complex64)
                return symbols, str(result.outputs["rx_symbols"].metadata["wireless_backend_detail"])

            first, backend = realize(23001, "first")
            repeated, repeated_backend = realize(23001, "repeated")
            different, _ = realize(23002, "different")

            self.assertEqual(backend, "sionna.awgn.pytorch")
            self.assertEqual(repeated_backend, backend)
            np.testing.assert_array_equal(first, repeated)
            self.assertFalse(np.array_equal(first, different))


if __name__ == "__main__":
    unittest.main()
