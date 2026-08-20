from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import OfdmChannelStateOperation, SymbolPowerAllocatorOperation


class OfdmPowerOwnershipTests(unittest.TestCase):
    def test_channel_state_owns_scenario_budget_and_allocator_declares_decision(self):
        state_params = OfdmChannelStateOperation.params_schema["properties"]
        allocator_params = SymbolPowerAllocatorOperation.params_schema["properties"]

        self.assertNotIn("target_power", state_params)
        self.assertIn("average_power_budget", state_params)
        self.assertIn("target_power", allocator_params)

    def test_allocator_target_changes_average_power_with_explicit_channel_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            state_path = root / "channel_state.npz"
            symbols = np.ones(8, dtype=np.complex64)
            gains = np.asarray(
                [
                    [
                        [0.1, 0.5, 2.0, 4.0],
                        [0.2, 1.0, 1.5, 3.0],
                    ]
                ],
                dtype=np.float32,
            )
            h_freq = np.sqrt(gains).astype(np.complex64)
            symbol_metadata = {
                "symbol_count": int(symbols.size),
                "channel_use_count": int(symbols.size),
            }
            state_metadata = {
                "noise_variance": 0.2,
                "reference_snr_db": float(10.0 * np.log10(1.0 / 0.2)),
                "average_power_budget": 0.5,
                "total_power_budget": 2.0,
                # A stale legacy value in an artifact must not override the allocator.
                "target_power": 9999.0,
            }
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(symbol_metadata),
            )
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=gains.reshape(-1, gains.shape[-1]),
                metadata_json=json.dumps(state_metadata),
            )
            inputs = {
                "symbols": Artifact("channel.symbols.complex_numpy", symbols_path, symbol_metadata),
                "channel_state": Artifact("channel.ofdm_channel_state.numpy", state_path, state_metadata),
            }

            def allocate(target_power: float, step_id: str):
                return SymbolPowerAllocatorOperation().run(
                    OperationContext(
                        recipe_name="ofdm_power_ownership_test",
                        step_id=step_id,
                        params={
                            "policy": "water_filling",
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": target_power,
                            "subcarrier_count": 4,
                        },
                        inputs=inputs,
                        run_dir=root,
                        step_dir=root / step_id,
                    )
                )

            low = allocate(0.5, "allocate_low")

            self.assertAlmostEqual(low.metrics["channel.tx_power.selected"], 0.5, places=6)
            self.assertAlmostEqual(low.metrics["channel.tx_power.average"], 0.5, places=6)
            with self.assertRaisesRegex(
                OperationError, "contradicts.*average_power_budget"
            ):
                allocate(2.0, "allocate_high")


if __name__ == "__main__":
    unittest.main()
