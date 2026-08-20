from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import (
    DigitalDemodulateOperation,
    SymbolPowerAllocatorOperation,
    SymbolPowerIdentityOperation,
    SymbolPowerNormalizeOperation,
)


class SymbolPowerAllocatorTests(unittest.TestCase):

    def test_learned_checkpoint_rejects_duplicate_abi_contract_metadata(self):
        from noema_lab.ops.channel.power_allocator_checkpoint import (
            load_csi_power_allocator_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "duplicate-contract.npz"
            metadata_json = (
                '{"schema_version":1,'
                '"kind":"noema.csi_power_allocator_checkpoint",'
                '"format":"noema_csi_power_deepset_npz_v1",'
                '"input_contract":"attacker-selected-contract",'
                '"input_contract":'
                '"log(max(gain*average_power/noise_variance,eps))",'
                '"output_contract":'
                '"euclidean_simplex_projection_times_fixed_sum_power",'
                '"activation":"relu","hidden_dim":2}'
            )
            checkpoint_sha = _write_test_deepset_checkpoint(
                checkpoint_path, metadata_json=metadata_json
            )
            with self.assertRaisesRegex(
                OperationError, "Duplicate JSON object key `input_contract`"
            ):
                load_csi_power_allocator_checkpoint(
                    str(checkpoint_path), checkpoint_sha
                )

    def test_learned_checkpoint_is_permutation_equivariant_and_exactly_feasible(self):
        from noema_lab.ops.channel.power_allocator_checkpoint import (
            infer_csi_power_allocation,
            load_csi_power_allocator_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "allocator.npz"
            checkpoint_sha = _write_test_deepset_checkpoint(checkpoint_path)
            checkpoint = load_csi_power_allocator_checkpoint(str(checkpoint_path), checkpoint_sha)
            gains = np.asarray([[0.1, 0.5, 2.0, 4.0]], dtype=np.float64)
            power = infer_csi_power_allocation(checkpoint, gains, 0.2, 1.0)
            permutation = np.asarray([2, 0, 3, 1])
            permuted = infer_csi_power_allocation(checkpoint, gains[:, permutation], 0.2, 1.0)
            np.testing.assert_allclose(permuted, power[:, permutation], atol=1e-10)
            np.testing.assert_allclose(np.sum(power, axis=1), np.asarray([4.0]), atol=1e-10)
            self.assertTrue(np.all(power >= 0.0))
            self.assertGreater(float(power[0, 3]), float(power[0, 0]))

    def test_symbol_allocator_runs_hash_pinned_learned_checkpoint_on_explicit_csi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "allocator.npz"
            checkpoint_sha = _write_test_deepset_checkpoint(checkpoint_path)
            symbols_path = root / "symbols.npz"
            state_path = root / "channel_state.npz"
            symbols = np.ones(20, dtype=np.complex64)
            gain_row = np.asarray([0.1, 0.5, 2.0, 4.0], dtype=np.float32)
            gains = np.tile(gain_row.reshape(1, 4), (5, 1))
            h_freq = np.sqrt(gains).astype(np.complex64).reshape(1, 5, 4)
            metadata = {
                "symbol_count": 20,
                "channel_use_count": 20,
                "noise_variance": 0.2,
                "reference_snr_db": float(10.0 * np.log10(1.0 / 0.2)),
            }
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=gains,
                metadata_json=json.dumps(metadata),
            )
            result = SymbolPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="learned_allocator_test",
                    step_id="tx_power",
                    params={
                        "policy": "learned_checkpoint",
                        "granularity": "per_subcarrier",
                        "budget_mode": "fixed_average",
                        "target_power": 1.0,
                        "model_batch_size": 2,
                        "subcarrier_count": 4,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha,
                    },
                    inputs={
                        "symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata),
                        "channel_state": Artifact("channel.ofdm_channel_state.numpy", state_path, metadata),
                    },
                    run_dir=root,
                    step_dir=root / "tx_power",
                )
            )
            with np.load(result.outputs["allocation"].path) as payload:
                power = np.asarray(payload["power"], dtype=np.float64)
            np.testing.assert_allclose(np.sum(power, axis=1), np.full((5,), 4.0), atol=1e-6)
            self.assertTrue(np.all(power >= 0.0))
            self.assertEqual(result.outputs["allocation"].metadata["policy"], "learned_csi_power_allocator")
            self.assertEqual(result.outputs["allocation"].metadata["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(result.outputs["allocation"].metadata["model_batch_size"], 2)
            self.assertEqual(result.outputs["allocation"].metadata["model_batch_count"], 3)
            self.assertEqual(result.outputs["allocation"].metadata["model_input_example_count"], 5)
            self.assertEqual(result.metrics["channel.tx_power.model_batch_size"], 2)
            self.assertFalse(
                result.outputs["allocation"].metadata["checkpoint_training"]["supervised_labels_used"]
            )

            with self.assertRaisesRegex(OperationError, "SHA-256 mismatch"):
                SymbolPowerAllocatorOperation().run(
                    OperationContext(
                        recipe_name="learned_allocator_test",
                        step_id="tx_power_bad_hash",
                        params={
                            "policy": "learned_checkpoint",
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": 1.0,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": "0" * 64,
                        },
                        inputs={
                            "symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata),
                            "channel_state": Artifact("channel.ofdm_channel_state.numpy", state_path, metadata),
                        },
                        run_dir=root,
                        step_dir=root / "tx_power_bad_hash",
                    )
                )

    def test_disabled_power_block_passes_symbols_through_and_records_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="power_test",
                step_id="tx_power",
                params={"label": "tx_power_off"},
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_power",
            )
            result = SymbolPowerIdentityOperation().run(ctx)
            loaded = np.load(result.outputs["symbols"].path)
            np.testing.assert_array_equal(loaded["symbols"], symbols)
            self.assertFalse(result.outputs["symbols"].metadata["tx_power_enabled"])
            self.assertAlmostEqual(result.metrics["channel.tx_power.average"], float(np.mean(np.abs(symbols) ** 2)), places=6)
            self.assertIn("tx_power_preview", result.outputs["symbols"].metadata)

    def test_snr_power_allocator_scales_symbols_and_records_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.array([1 + 0j, 1j, -1 + 0j, -1j], dtype=np.complex64)
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="power_test",
                step_id="tx_power_allocate",
                params={
                    "policy": "snr_sigmoid",
                    "snr_db": 12.0,
                    "min_power": 0.5,
                    "max_power": 1.5,
                    "midpoint_snr_db": 12.0,
                    "slope_db": 3.0,
                },
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_power_allocate",
            )
            result = SymbolPowerAllocatorOperation().run(ctx)
            self.assertEqual(result.outputs["symbols"].kind, "channel.symbols.complex_numpy")
            self.assertAlmostEqual(result.metrics["channel.tx_power.selected"], 1.0, places=6)
            self.assertAlmostEqual(result.metrics["channel.tx_power.average"], 1.0, places=6)
            self.assertAlmostEqual(result.metrics["channel.tx_power.total_energy"], 4.0, places=6)
            preview = result.outputs["symbols"].metadata.get("tx_power_preview")
            self.assertIsInstance(preview, dict)
            self.assertEqual(preview["kind"], "tx_power_trace")
            self.assertEqual(preview["source_count"], 4)
            self.assertEqual(len(preview["values"]), 4)


    def test_per_subcarrier_allocator_redistributes_fixed_average_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.array([1 + 0j, 2 + 0j, 1 + 0j, 2 + 0j, 1 + 0j, 2 + 0j, 1 + 0j, 2 + 0j], dtype=np.complex64)
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="power_test",
                step_id="tx_power_allocate",
                params={
                    "policy": "snr_sigmoid",
                    "granularity": "per_subcarrier",
                    "budget_mode": "fixed_average",
                    "target_power": 1.0,
                    "allocation_contrast": 1.0,
                    "subcarrier_count": 4,
                    "snr_db": 20.0,
                    "midpoint_snr_db": 12.0,
                    "slope_db": 3.0,
                },
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_power_allocate",
            )
            result = SymbolPowerAllocatorOperation().run(ctx)
            loaded = np.load(result.outputs["symbols"].path)
            allocated = loaded["symbols"]
            self.assertAlmostEqual(float(np.mean(np.abs(allocated) ** 2)), 1.0, places=6)
            group_powers = []
            for group in range(4):
                group_powers.append(float(np.mean(np.abs(allocated[group::4]) ** 2)))
            self.assertGreater(max(group_powers) - min(group_powers), 0.1)
            self.assertEqual(result.metrics["channel.tx_power.group_count"], 4)
            metadata = result.outputs["symbols"].metadata
            self.assertEqual(metadata["power_allocator_granularity"], "per_subcarrier")
            self.assertEqual(metadata["power_allocator_budget_mode"], "fixed_average")
            preview = metadata.get("tx_power_preview")
            self.assertEqual(preview["allocation_group_count"], 4)
            self.assertEqual(len(preview["allocation_target_power_preview"]), 4)

    def test_water_filling_uses_explicit_ofdm_state_and_emits_capture_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            state_path = root / "channel_state.npz"
            symbols = np.ones(4, dtype=np.complex64)
            h_freq = np.sqrt(np.asarray([[[0.1, 0.5, 2.0, 4.0]]], dtype=np.float32)).astype(np.complex64)
            gains = (np.abs(h_freq) ** 2).astype(np.float32)
            metadata = {
                "symbol_count": 4,
                "channel_use_count": 4,
                "noise_variance": 0.2,
                "reference_snr_db": float(10.0 * np.log10(1.0 / 0.2)),
            }
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(metadata),
            )
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=gains.reshape(1, 4),
                metadata_json=json.dumps(metadata),
            )
            result = SymbolPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="water_filling_test",
                    step_id="tx_power",
                    params={
                        "policy": "water_filling",
                        "granularity": "per_subcarrier",
                        "budget_mode": "fixed_average",
                        "target_power": 1.0,
                        "subcarrier_count": 4,
                    },
                    inputs={
                        "symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata),
                        "channel_state": Artifact("channel.ofdm_channel_state.numpy", state_path, metadata),
                    },
                    run_dir=root,
                    step_dir=root / "tx_power",
                )
            )
            self.assertIn("allocation", result.outputs)
            with np.load(result.outputs["allocation"].path) as payload:
                power = payload["power"].astype(np.float64)
                water_level = payload["water_level"].astype(np.float64)
                np.testing.assert_allclose(np.sum(power, axis=1), np.asarray([4.0]), atol=1e-6)
                np.testing.assert_allclose(
                    power,
                    np.maximum(water_level[:, None] - 0.2 / gains.reshape(1, 4), 0.0),
                    atol=1e-6,
                )
            self.assertAlmostEqual(result.metrics["channel.tx_power.average"], 1.0, places=6)
            preview = result.outputs["allocation"].metadata["resource_allocation_preview"]
            self.assertEqual(preview["allocation_axis"], "subcarrier")
            self.assertEqual(preview["policy"], "theoretical_water_filling")

    def test_variable_average_budget_is_distinct_from_fixed_redistribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.ones(8, dtype=np.complex64)
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            result = SymbolPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="power_test",
                    step_id="tx_power_allocate",
                    params={
                        "policy": "snr_sigmoid",
                        "budget_mode": "variable_average",
                        "snr_db": 12.0,
                        "midpoint_snr_db": 12.0,
                        "min_power": 0.5,
                        "max_power": 1.5,
                    },
                    inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                    run_dir=root,
                    step_dir=root / "tx_power_allocate",
                )
            )
            self.assertAlmostEqual(result.metrics["channel.tx_power.selected"], 1.0, places=6)
            self.assertEqual(result.outputs["symbols"].metadata["power_allocator_budget_mode"], "variable_average")
            low_snr = SymbolPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="power_test",
                    step_id="tx_power_allocate_low_snr",
                    params={
                        "policy": "snr_sigmoid",
                        "budget_mode": "variable_average",
                        "snr_db": -30.0,
                        "midpoint_snr_db": 12.0,
                        "min_power": 0.5,
                        "max_power": 1.5,
                    },
                    inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                    run_dir=root,
                    step_dir=root / "tx_power_allocate_low_snr",
                )
            )
            self.assertGreater(low_snr.metrics["channel.tx_power.selected"], result.metrics["channel.tx_power.selected"])

    def test_allocation_aware_bit_loading_round_trips_bits_and_shortens_resource_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            bits_path = root / "coded_bits.npz"
            state_path = root / "channel_state.npz"
            bits = np.asarray(
                [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                dtype=np.uint8,
            )
            symbols = np.ones((10,), dtype=np.complex64)
            h_freq = np.ones((2, 2, 4), dtype=np.complex64)
            metadata = {
                "bit_count": int(bits.size),
                "symbol_count": int(symbols.size),
                "channel_use_count": int(symbols.size),
                "noise_variance": 0.01,
                "reference_snr_db": 20.0,
                "source_item_symbol_counts": [5, 5],
                "source_item_modulator_input_bit_counts": [10, 10],
                "source_item_padded_bit_counts": [10, 10],
            }
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            np.savez_compressed(bits_path, bits=bits, metadata_json=json.dumps(metadata))
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=np.ones((4, 4), dtype=np.float32),
                metadata_json=json.dumps(metadata),
            )
            allocated = SymbolPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="adaptive_transport_test",
                    step_id="tx_power",
                    params={
                        "policy": "fixed",
                        "granularity": "per_subcarrier",
                        "budget_mode": "fixed_average",
                        "target_power": 1.0,
                        "transport_mode": "allocation_aware_bit_loading",
                        "bit_loading_bpsk_min_snr_db": -4.0,
                        "bit_loading_qpsk_min_snr_db": -3.0,
                        "bit_loading_qam16_min_snr_db": -2.0,
                        "bit_loading_qam64_min_snr_db": -1.0,
                        "bit_loading_max_bits_per_symbol": 4,
                    },
                    inputs={
                        "symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata),
                        "bits": Artifact("channel.coded_bits.numpy", bits_path, metadata),
                        "channel_state": Artifact("channel.ofdm_channel_state.numpy", state_path, metadata),
                    },
                    run_dir=root,
                    step_dir=root / "tx_power",
                )
            )
            self.assertEqual(allocated.outputs["symbols"].metadata["adaptive_scheduled_resource_element_count"], 8)
            self.assertLess(
                allocated.outputs["symbols"].metadata["adaptive_scheduled_resource_element_count"],
                h_freq.size,
            )
            self.assertNotIn(
                "source_item_symbol_counts",
                allocated.outputs["symbols"].metadata,
            )
            self.assertFalse(
                allocated.outputs["symbols"].metadata[
                    "source_item_partition_preserved"
                ]
            )
            demodulated = DigitalDemodulateOperation().run(
                OperationContext(
                    recipe_name="adaptive_transport_test",
                    step_id="demodulator",
                    params={"modulation": "auto"},
                    inputs={
                        "rx_symbols": Artifact(
                            "channel.rx_symbols.complex_numpy",
                            allocated.outputs["symbols"].path,
                            allocated.outputs["symbols"].metadata,
                        ),
                        "allocation": allocated.outputs["allocation"],
                    },
                    run_dir=root,
                    step_dir=root / "demodulator",
                )
            )
            with np.load(demodulated.outputs["bits"].path) as payload:
                np.testing.assert_array_equal(payload["bits"], bits)

    def test_fixed_power_normalizer_records_total_energy_for_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            symbols = np.array([2 + 0j, 0 + 2j], dtype=np.complex64)
            metadata = {"symbol_count": int(symbols.size), "channel_use_count": int(symbols.size)}
            np.savez_compressed(symbols_path, symbols=symbols, metadata_json=json.dumps(metadata))
            ctx = OperationContext(
                recipe_name="power_test",
                step_id="tx_power_normalize",
                params={"target_power": 2.0},
                inputs={"symbols": Artifact("channel.symbols.complex_numpy", symbols_path, metadata)},
                run_dir=root,
                step_dir=root / "tx_power_normalize",
            )
            result = SymbolPowerNormalizeOperation().run(ctx)
            self.assertAlmostEqual(result.metrics["channel.tx_power.average"], 2.0, places=6)
            self.assertAlmostEqual(result.metrics["channel.tx_power.total_energy"], 4.0, places=6)
            self.assertIn("tx_power_preview", result.outputs["symbols"].metadata)


def _write_test_deepset_checkpoint(
    path: Path,
    *,
    metadata_json: str | None = None,
) -> str:
    metadata = {
        "schema_version": 1,
        "kind": "noema.csi_power_allocator_checkpoint",
        "format": "noema_csi_power_deepset_npz_v1",
        "input_contract": "log(max(gain*average_power/noise_variance,eps))",
        "output_contract": "euclidean_simplex_projection_times_fixed_sum_power",
        "activation": "relu",
        "hidden_dim": 2,
        "training": {
            "objective": "maximize_parallel_channel_shannon_spectral_efficiency",
            "supervised_labels_used": False,
            "water_filling_used_during_training": False,
        },
    }
    np.savez_compressed(
        path,
        feature_mean=np.asarray([0.0], dtype=np.float32),
        feature_scale=np.asarray([1.0], dtype=np.float32),
        phi_weight_0=np.asarray([[1.0], [-1.0]], dtype=np.float32),
        phi_bias_0=np.zeros(2, dtype=np.float32),
        phi_weight_1=np.eye(2, dtype=np.float32),
        phi_bias_1=np.zeros(2, dtype=np.float32),
        rho_weight_0=np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0]],
            dtype=np.float32,
        ),
        rho_bias_0=np.zeros(2, dtype=np.float32),
        rho_weight_out=np.asarray([[1.0, -1.0]], dtype=np.float32),
        rho_bias_out=np.zeros(1, dtype=np.float32),
        metadata_json=metadata_json or json.dumps(metadata, sort_keys=True),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
