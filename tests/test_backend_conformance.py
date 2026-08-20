import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core import dataplane
from noema_lab.core.artifacts import Artifact
from noema_lab.core.boundaries import validate_bits_per_pixel, validate_rate_accounting_point
from noema_lab.core.operations import OperationContext
from noema_lab.ops.channel.digital import (
    DigitalModulateOperation,
    IdentitySymbolLinkOperation,
    SymbolPowerNormalizeOperation,
    SymbolBoundaryCheckpointOperation,
    WirelessChannelOperation,
    _apply_channel,
    _approximate_llr,
    _demodulate,
    _modulate,
    _noise_variance_from_snr,
    _sionna_rng_serialized,
)
from noema_lab.training.differentiability import torch_available

if torch_available():
    import torch

    from noema_lab.training.sionna_blocks import (
        AwgnChannelBlock,
        FlatRayleighChannelBlock,
        IdentityExportBlock,
        PowerNormalizationBlock,
        QamPamMapperBlock,
        SoftDemapperBlock,
    )
else:  # pragma: no cover - optional dependency state
    torch = None
    AwgnChannelBlock = None
    FlatRayleighChannelBlock = None
    IdentityExportBlock = None
    PowerNormalizationBlock = None
    QamPamMapperBlock = None
    SoftDemapperBlock = None


class BackendConformanceTests(unittest.TestCase):
    def test_seeded_channel_is_schedule_invariant_and_sionna_section_is_serial(self):
        self.assertTrue(
            WirelessChannelOperation().describe()["execution_safety"]["thread_safe"]
        )
        symbols = np.ones(2048, dtype=np.complex64)

        def sample(seed):
            values, _backend, _report = _apply_channel(
                symbols,
                "flat_rayleigh",
                12.0,
                np.random.RandomState(seed),
                receiver_processing="matched",
            )
            return values

        expected = [sample(seed) for seed in (7, 11, 13, 17)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(sample, (7, 11, 13, 17)))
        for serial, parallel in zip(expected, actual):
            np.testing.assert_array_equal(serial, parallel)

        state = {"active": 0, "maximum": 0}
        state_lock = threading.Lock()

        @_sionna_rng_serialized
        def simulated_sionna_call(value):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.01)
            with state_lock:
                state["active"] -= 1
            return value

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(simulated_sionna_call, range(8))), list(range(8)))
        self.assertEqual(state["maximum"], 1)

    def test_ofdm_charges_the_executed_padded_grid(self):
        symbols = np.ones(130, dtype=np.complex64)
        metadata = {"symbol_count": 130, "channel_use_count": 130}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(metadata),
            )
            result = WirelessChannelOperation().run(
                OperationContext(
                    recipe_name="ofdm_grid_accounting",
                    step_id="wireless_channel",
                    params={
                        "channel": "ofdm_tdl",
                        "wireless_backend": "numpy",
                        "receiver_processing": "matched",
                        "channel_state_mode": "none",
                        "ofdm_fft_size": 32,
                        "num_ofdm_symbols": 4,
                        "snr_db": 30.0,
                        "seed": 17,
                    },
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            metadata,
                        )
                    },
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )

        self.assertEqual(result.metrics["channel.payload_symbol_count"], 130)
        self.assertEqual(result.metrics["channel.channel_use_count"], 256)
        self.assertEqual(result.metrics["channel.grid_padding_symbol_count"], 126)
        self.assertAlmostEqual(
            result.metrics["channel.tx_power_per_executed_use.average"],
            130.0 / 256.0,
            places=7,
        )

    def test_continuous_symbol_channel_uses_image_shape_for_bandwidth_ratio(self):
        symbols = np.ones(120, dtype=np.complex64)
        metadata = {
            "symbol_count": int(symbols.size),
            "channel_use_count": int(symbols.size),
            "image_shape": [2, 4, 5, 3],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(metadata),
            )
            result = WirelessChannelOperation().run(
                OperationContext(
                    recipe_name="deepjscc_bandwidth_ratio",
                    step_id="wireless_channel",
                    params={"channel": "awgn", "snr_db": 30.0, "seed": 7},
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            metadata,
                        )
                    },
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )

        # One complex tensor element is one channel use. The RGB channel count
        # is not part of the source-pixel denominator: 120 / (2*4*5) = 3.
        self.assertEqual(result.metrics["channel.channel_use_count"], 120)
        self.assertAlmostEqual(result.metrics["channel.uses_per_pixel"], 3.0)

    def test_continuous_symbol_bandwidth_ratio_prefers_original_image_extents(self):
        symbols = np.ones(70, dtype=np.complex64)
        metadata = {
            "symbol_count": int(symbols.size),
            "channel_use_count": int(symbols.size),
            "image_shape": [2, 4, 5, 3],
            "original_shapes": [[1, 4, 5, 3], [1, 3, 5, 3]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(metadata),
            )
            result = WirelessChannelOperation().run(
                OperationContext(
                    recipe_name="deepjscc_unpadded_bandwidth_ratio",
                    step_id="wireless_channel",
                    params={"channel": "awgn", "snr_db": 30.0, "seed": 7},
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            metadata,
                        )
                    },
                    run_dir=root,
                    step_dir=root / "wireless_channel",
                )
            )

        # The two source images contain 4*5 + 3*5 = 35 pixels. Their common
        # storage tensor is padded to 2*4*5, which must not enter the ratio.
        self.assertAlmostEqual(result.metrics["channel.uses_per_pixel"], 2.0)

    def test_bit_pack_unpack_is_exact_across_available_dataplane_backends(self):
        indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        bits_numpy, backend = dataplane.indices_to_bits(indices, bits_per_index=3, backend="python_numpy")
        self.assertEqual(backend, "python_numpy")
        roundtrip_numpy, invalid_fraction, backend = dataplane.bits_to_indices(
            bits_numpy,
            bits_per_index=3,
            shape=indices.shape,
            codebook_size=8,
            invalid_policy="mod",
            backend="python_numpy",
        )
        self.assertEqual(backend, "python_numpy")
        self.assertEqual(invalid_fraction, 0.0)
        np.testing.assert_array_equal(roundtrip_numpy, indices)

        payload = b"Noema exact bit packing"
        payload_bits, _ = dataplane.bytes_to_bits(payload, backend="python_numpy")
        recovered, _ = dataplane.bits_to_bytes(payload_bits, len(payload), backend="python_numpy")
        self.assertEqual(recovered, payload)

        if dataplane.native_available():
            bits_cpp, backend = dataplane.indices_to_bits(indices, bits_per_index=3, backend="cpp_native")
            self.assertEqual(backend, "cpp_native")
            np.testing.assert_array_equal(bits_cpp, bits_numpy)
            roundtrip_cpp, invalid_fraction_cpp, backend = dataplane.bits_to_indices(
                bits_cpp,
                bits_per_index=3,
                shape=indices.shape,
                codebook_size=8,
                invalid_policy="mod",
                backend="cpp_native",
            )
            self.assertEqual(backend, "cpp_native")
            self.assertEqual(invalid_fraction_cpp, 0.0)
            np.testing.assert_array_equal(roundtrip_cpp, indices)

    def test_symbol_identity_artifact_and_torch_materializations_are_exact(self):
        symbols = np.array([1 + 0j, -0.5 + 0.25j, 0.125 - 0.75j], dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            boundary = SymbolBoundaryCheckpointOperation().run(
                OperationContext(
                    recipe_name="conformance",
                    step_id="tx_symbol_boundary",
                    params={"label": "modulator_input", "role": "symbol_channel_input"},
                    inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                    run_dir=root,
                    step_dir=root / "tx_symbol_boundary",
                )
            )
            linked = IdentitySymbolLinkOperation().run(
                OperationContext(
                    recipe_name="conformance",
                    step_id="wireless_channel",
                    params={"label": "identity_symbol_channel"},
                    inputs={"symbols": boundary.outputs["symbols"]},
                    run_dir=root,
                    step_dir=root / "identity_symbol_channel",
                )
            )
            with np.load(linked.outputs["symbols"].path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["symbols"], symbols)

        if torch_available():
            tensor = torch.tensor(symbols)
            out = IdentityExportBlock()(tensor)
            self.assertTrue(torch.equal(out, tensor))

    def test_power_normalization_torch_matches_numpy_reference(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        symbols = np.array([1 + 1j, -2 + 0.5j, 0.25 - 3j, -0.75 - 0.5j], dtype=np.complex64)
        target_power = 2.5
        expected = symbols * np.sqrt(target_power / float(np.mean(np.abs(symbols) ** 2)))
        actual = PowerNormalizationBlock(target_power=target_power)(torch.tensor(symbols)).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected.astype(np.complex64), rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(np.mean(np.abs(actual) ** 2)), target_power, places=5)

    def test_source_item_power_normalization_matches_numpy_and_torch(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        rows = np.array(
            [
                [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
                [2 + 0j, 4 + 0j, 2 + 0j, 4 + 0j],
            ],
            dtype=np.complex64,
        )
        metadata = {
            "source_item_symbol_counts": [4, 4],
            "source_item_ids": ["a", "b"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=rows.reshape(-1),
                metadata_json=json.dumps(metadata),
            )
            result = SymbolPowerNormalizeOperation().run(
                OperationContext(
                    recipe_name="normalization_conformance",
                    step_id="tx_power",
                    params={
                        "target_power": 1.0,
                        "normalization_scope": "source_item",
                    },
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            metadata,
                        )
                    },
                    run_dir=root,
                    step_dir=root / "tx_power",
                )
            )
            with np.load(result.outputs["symbols"].path, allow_pickle=False) as payload:
                numpy_values = np.array(payload["symbols"], copy=True).reshape(rows.shape)

        torch_values = (
            PowerNormalizationBlock(
                target_power=1.0,
                normalization_scope="source_item",
            )(torch.tensor(rows))
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(torch_values, numpy_values, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            np.mean(np.abs(torch_values) ** 2, axis=1),
            np.ones(2),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_flat_rayleigh_receiver_processing_is_bound_and_differentiable(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        symbols = torch.ones(256, dtype=torch.complex64, requires_grad=True)
        matched = FlatRayleighChannelBlock(
            snr_db=30.0,
            receiver_processing="matched",
            seed=91,
        )
        raw = FlatRayleighChannelBlock(
            snr_db=30.0,
            receiver_processing="none",
            seed=91,
        )
        matched_output = matched(symbols)
        raw_output = raw(symbols)
        self.assertEqual(
            matched.export_description()["materialization_contract"][
                "receiver_processing"
            ],
            "matched",
        )
        self.assertEqual(
            raw.export_description()["materialization_contract"][
                "receiver_processing"
            ],
            "none",
        )
        self.assertFalse(bool(torch.equal(matched_output, raw_output)))
        torch.mean(torch.abs(matched_output) ** 2).backward()
        self.assertIsNotNone(symbols.grad)
        self.assertTrue(bool(torch.isfinite(symbols.grad.real).all()))
        self.assertTrue(bool(torch.isfinite(symbols.grad.imag).all()))

    def test_numpy_flat_rayleigh_matches_explicit_receiver_selector(self):
        symbols = np.ones(512, dtype=np.complex64)
        matched, _backend, matched_report = _apply_channel(
            symbols,
            "flat_rayleigh",
            20.0,
            np.random.RandomState(37),
            receiver_processing="matched",
        )
        raw, _backend, raw_report = _apply_channel(
            symbols,
            "flat_rayleigh",
            20.0,
            np.random.RandomState(37),
            receiver_processing="none",
        )

        self.assertTrue(matched_report["channel_equalized"])
        self.assertEqual(matched_report["equalizer"], "perfect_csi_one_tap")
        self.assertFalse(raw_report["channel_equalized"])
        self.assertIsNone(raw_report["equalizer"])
        self.assertNotIn("rx_equalized_power_average", raw_report)
        self.assertFalse(bool(np.array_equal(matched, raw)))

    def test_awgn_artifact_and_torch_materializations_match_noise_statistics(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        snr_db = 11.0
        expected_noise_variance = _noise_variance_from_snr(snr_db)
        symbol_count = 60000
        symbols_np = np.ones(symbol_count, dtype=np.complex64)
        rx_np, backend, _report = _apply_channel(
            symbols_np,
            "awgn",
            snr_db,
            np.random.RandomState(1234),
            backend="python_numpy",
        )
        self.assertEqual(backend, "python_numpy")
        noise_np = rx_np - symbols_np

        symbols_torch = torch.ones(symbol_count, dtype=torch.complex64)
        rx_torch = AwgnChannelBlock(snr_db=snr_db, backend="torch", seed=1234)(symbols_torch).detach().cpu().numpy()
        noise_torch = rx_torch - symbols_np

        for noise in [noise_np, noise_torch]:
            measured_variance = float(np.mean(np.abs(noise) ** 2))
            self.assertLess(abs(measured_variance - expected_noise_variance) / expected_noise_variance, 0.06)
            self.assertLess(abs(float(np.mean(noise.real))), 0.004)
            self.assertLess(abs(float(np.mean(noise.imag))), 0.004)
        variance_np = float(np.mean(np.abs(noise_np) ** 2))
        variance_torch = float(np.mean(np.abs(noise_torch) ** 2))
        self.assertLess(abs(variance_np - variance_torch) / expected_noise_variance, 0.08)

    def test_qpsk_mapper_torch_matches_artifact_numpy(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
        expected_symbols, padded_bits, bits_per_symbol, backend = _modulate(bits, "qpsk", backend="python_numpy")
        self.assertEqual(backend, "python_numpy")
        self.assertEqual(bits_per_symbol, 2)
        self.assertEqual(int(padded_bits.size), 10)
        actual_symbols = QamPamMapperBlock("qpsk", normalize_power=False)(torch.tensor(bits)).detach().cpu().numpy()
        np.testing.assert_allclose(actual_symbols, expected_symbols, rtol=1e-7, atol=1e-7)

    def test_qpsk_demapper_torch_soft_llr_matches_artifact_maxlog_llr(self):
        if not torch_available():
            self.skipTest("torch is not installed")
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols, _padded_bits, _bits_per_symbol, _backend = _modulate(bits, "qpsk", backend="python_numpy")
        perturbation = np.array([0.04 - 0.02j, -0.03 + 0.01j, 0.02 + 0.03j, -0.01 - 0.04j], dtype=np.complex64)
        rx_symbols = (symbols + perturbation).astype(np.complex64)
        hard_bits, backend = _demodulate(rx_symbols, "qpsk", backend="python_numpy")
        self.assertEqual(backend, "python_numpy")
        np.testing.assert_array_equal(hard_bits, bits)
        noise_variance = 0.2
        llr_numpy = _approximate_llr(rx_symbols, "qpsk", hard_bits, noise_variance)
        llr_torch = SoftDemapperBlock("qpsk", noise_variance=noise_variance)(torch.tensor(rx_symbols)).detach().cpu().numpy()
        np.testing.assert_allclose(llr_torch, llr_numpy, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal((llr_torch < 0).astype(np.uint8), bits)

    def test_rate_accounting_fixed_points_are_exact(self):
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        metadata = {"bit_count": int(bits.size), "original_shape": [1, 1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits_path = root / "bits.npz"
            np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(metadata))
            result = DigitalModulateOperation().run(
                OperationContext(
                    recipe_name="conformance",
                    step_id="modulator",
                    params={"modulation": "qpsk", "data_plane_backend": "python_numpy"},
                    inputs={"bits": Artifact("channel.bits.numpy", bits_path, metadata)},
                    run_dir=root,
                    step_dir=root / "modulator",
                )
            )
        self.assertEqual(result.metrics["channel.coded_bit_count"], 5)
        self.assertEqual(result.metrics["channel.transmitted_bit_count"], 6)
        self.assertEqual(result.metrics["channel.symbol_count"], 3)
        self.assertEqual(result.metrics["channel.channel_use_count"], 3)
        self.assertEqual(result.metrics["channel.bits_per_symbol"], 2)
        self.assertEqual(validate_rate_accounting_point("tx_bits", result.metrics["channel.transmitted_bit_count"], "bits"), 6)
        self.assertAlmostEqual(validate_bits_per_pixel(6, 2, 3.0), 3.0)
        self.assertAlmostEqual(result.metrics["channel.uses_per_pixel"], 1.5)


if __name__ == "__main__":
    unittest.main()
