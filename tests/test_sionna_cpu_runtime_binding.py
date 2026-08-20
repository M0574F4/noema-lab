from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from noema_lab.ops.channel import digital, nr_ldpc


class _ArrayTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class SionnaCpuRuntimeBindingTests(unittest.TestCase):
    def test_ofdm_generation_explicitly_binds_cpu_single_precision(self):
        calls: dict[str, object] = {}
        config = types.SimpleNamespace(seed=None)

        class ResourceGrid:
            def __init__(self, **kwargs) -> None:
                calls["resource_grid"] = dict(kwargs)

        class TDL:
            def __init__(self, **kwargs) -> None:
                calls["tdl"] = dict(kwargs)

        class GenerateOFDMChannel:
            def __init__(self, channel_model, resource_grid, **kwargs) -> None:
                calls["generator"] = dict(kwargs)

            def __call__(self, block_count: int) -> _ArrayTensor:
                return _ArrayTensor(
                    np.ones(
                        (block_count, 1, 1, 1, 1, 2, 8),
                        dtype=np.complex64,
                    )
                )

        modules = self._fake_sionna_modules(
            config=config,
            ResourceGrid=ResourceGrid,
            TDL=TDL,
            GenerateOFDMChannel=GenerateOFDMChannel,
        )
        with (
            mock.patch.object(digital, "_sionna_available", return_value=True),
            mock.patch.dict(sys.modules, modules),
        ):
            result = digital._generate_sionna_ofdm_channel_state(
                1,
                {
                    "ofdm_fft_size": 8,
                    "num_ofdm_symbols": 2,
                    "subcarrier_spacing_khz": 15.0,
                    "carrier_frequency_ghz": 3.5,
                    "delay_spread_ns": 100.0,
                    "mobility_kmh": 3.0,
                    "tdl_model": "A",
                },
                123,
            )

        self.assertEqual(result["h_freq"].shape, (1, 2, 8))
        self.assertEqual(config.seed, 123)
        for key in ("resource_grid", "tdl", "generator"):
            self.assertEqual(calls[key]["device"], "cpu")
            self.assertEqual(calls[key]["precision"], "single")

    def test_ofdm_application_explicitly_binds_cpu_single_precision(self):
        calls: dict[str, object] = {}
        config = types.SimpleNamespace(seed=None)

        class ApplyOFDMChannel:
            def __init__(self, **kwargs) -> None:
                calls["apply"] = dict(kwargs)

            def __call__(self, x, h, _noise) -> _ArrayTensor:
                return _ArrayTensor(np.asarray(x) * np.asarray(h)[:, 0, 0])

        fake_torch = types.ModuleType("torch")
        fake_torch.complex64 = np.complex64
        fake_torch.float32 = np.float32
        fake_torch.as_tensor = lambda value, dtype=None: np.asarray(
            value,
            dtype=dtype,
        )
        modules = self._fake_sionna_modules(
            config=config,
            ApplyOFDMChannel=ApplyOFDMChannel,
        )
        modules["torch"] = fake_torch
        with (
            mock.patch.object(digital, "_sionna_available", return_value=True),
            mock.patch.dict(sys.modules, modules),
        ):
            output, report = digital._apply_sionna_realized_ofdm_channel(
                np.ones((8,), dtype=np.complex64),
                np.ones((1, 1, 8), dtype=np.complex64),
                0.0,
                {"subcarrier_spacing_khz": 15.0},
                456,
            )

        self.assertEqual(output.shape, (8,))
        self.assertEqual(report["data_plane_backend"], "torch")
        self.assertEqual(config.seed, 456)
        self.assertEqual(calls["apply"]["device"], "cpu")
        self.assertEqual(calls["apply"]["precision"], "single")

    def test_nr_codec_explicitly_binds_cpu_single_precision(self):
        calls: dict[str, object] = {}

        class TBEncoder:
            def __init__(self, **kwargs) -> None:
                calls["encoder"] = dict(kwargs)

        class TBDecoder:
            def __init__(self, encoder, **kwargs) -> None:
                calls["decoder_encoder"] = encoder
                calls["decoder"] = dict(kwargs)

        nr_ldpc._nr_codec.cache_clear()
        try:
            with mock.patch.object(
                nr_ldpc,
                "_require_sionna",
                return_value=(object(), TBEncoder, TBDecoder, object()),
            ):
                encoder, decoder = nr_ldpc._nr_codec(
                    1024,
                    2048,
                    0.5,
                    2,
                    1,
                    1,
                    1,
                    "PUSCH",
                    0,
                    20,
                )
        finally:
            nr_ldpc._nr_codec.cache_clear()

        self.assertIs(calls["decoder_encoder"], encoder)
        self.assertIsInstance(decoder, TBDecoder)
        for key in ("encoder", "decoder"):
            self.assertEqual(calls[key]["device"], "cpu")
            self.assertEqual(calls[key]["precision"], "single")

    @staticmethod
    def _fake_sionna_modules(*, config, **symbols):
        sionna = types.ModuleType("sionna")
        phy = types.ModuleType("sionna.phy")
        channel = types.ModuleType("sionna.phy.channel")
        tr38901 = types.ModuleType("sionna.phy.channel.tr38901")
        ofdm = types.ModuleType("sionna.phy.ofdm")
        phy.config = config
        sionna.phy = phy
        phy.channel = channel
        phy.ofdm = ofdm
        channel.tr38901 = tr38901
        for name, value in symbols.items():
            if name == "ResourceGrid":
                setattr(ofdm, name, value)
            elif name == "TDL":
                setattr(tr38901, name, value)
            else:
                setattr(channel, name, value)
        return {
            "sionna": sionna,
            "sionna.phy": phy,
            "sionna.phy.channel": channel,
            "sionna.phy.channel.tr38901": tr38901,
            "sionna.phy.ofdm": ofdm,
        }


if __name__ == "__main__":
    unittest.main()
