import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext
from noema_lab.ops.channel.digital import WirelessChannelOperation
from noema_lab.training.differentiability import sionna_available


@unittest.skipUnless(
    sionna_available(),
    "Sionna 2.x/PyTorch is required for realized OFDM accounting",
)
class OfdmPowerAccountingTests(unittest.TestCase):
    def test_realized_ofdm_reports_antenna_and_equalizer_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols = np.ones((8,), dtype=np.complex64)
            h_freq = np.full((1, 1, 8), 0.5 + 0.0j, dtype=np.complex64)
            gains = np.abs(h_freq).astype(np.float32) ** 2
            symbols_metadata = {"power_unit": "normalized"}
            state_metadata = {
                "noise_variance": 0.2,
                "reference_snr_db": float(10.0 * np.log10(5.0)),
                "channel_state_seed": 17,
            }
            symbols_path = root / "symbols.npz"
            state_path = root / "state.npz"
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(symbols_metadata),
            )
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=gains,
                metadata_json=json.dumps(state_metadata),
            )
            result = WirelessChannelOperation().run(
                OperationContext(
                    recipe_name="power_accounting",
                    step_id="wireless_channel",
                    params={
                        "channel": "ofdm_tdl",
                        "noise_mode": "fixed_variance",
                        "noise_variance": 0.2,
                        "wireless_backend": "sionna",
                        "channel_state_mode": "explicit",
                        "ofdm_fft_size": 8,
                        "num_ofdm_symbols": 1,
                        "seed": 19,
                    },
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            symbols_metadata,
                        ),
                        "channel_state": Artifact(
                            "channel.ofdm_channel_state.numpy",
                            state_path,
                            state_metadata,
                        ),
                    },
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )

            metrics = result.metrics
            antenna_signal = metrics["channel.rx_antenna_signal_power.average"]
            antenna_noise = metrics["channel.rx_antenna_noise_power.average"]
            antenna_components = metrics["channel.rx_antenna_component_power.average"]
            antenna_measured = metrics["channel.rx_antenna_power.average"]
            antenna_cross = metrics["channel.rx_antenna_signal_noise_cross_power.average"]
            equalized_signal = metrics["channel.post_equalizer_signal_power.average"]
            equalized_noise = metrics["channel.post_equalizer_noise_power.average"]
            equalized_components = metrics["channel.post_equalizer_component_power.average"]
            equalized_measured = metrics["channel.post_equalizer_output_power.average"]
            equalized_cross = metrics["channel.post_equalizer_signal_noise_cross_power.average"]

            self.assertAlmostEqual(antenna_signal, 0.25, places=6)
            self.assertAlmostEqual(equalized_signal, 1.0, places=6)
            self.assertAlmostEqual(equalized_noise, 4.0 * antenna_noise, places=5)
            self.assertAlmostEqual(antenna_components, antenna_signal + antenna_noise, places=7)
            self.assertAlmostEqual(equalized_components, equalized_signal + equalized_noise, places=7)
            self.assertAlmostEqual(antenna_measured, antenna_components + antenna_cross, places=6)
            self.assertAlmostEqual(equalized_measured, equalized_components + equalized_cross, places=6)
            self.assertEqual(
                result.outputs["rx_symbols"].metadata["equalizer"],
                "perfect_csi_ofdm_zero_forcing_one_tap",
            )


if __name__ == "__main__":
    unittest.main()
