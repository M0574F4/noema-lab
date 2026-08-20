from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry
from noema_lab.ops.channel.digital import SymbolPowerAllocatorOperation
from noema_lab.ops.channel.digital import CausalCsiPowerAllocatorOperation
from noema_lab.ops.resource_reliability import (
    OfdmDelayedCsiOperation,
    OfdmFiniteBlocklengthReliabilityMetricsOperation,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes" / "resource_delayed_csi_finite_blocklength.yaml"
PACKAGED_RECIPE_PATH = (
    ROOT
    / "src"
    / "noema_lab"
    / "recipe_starters"
    / "resource_delayed_csi_finite_blocklength.yaml"
)


def _state_artifact(path: Path, h_freq: np.ndarray, metadata: dict) -> Artifact:
    gains = np.maximum(np.abs(h_freq) ** 2, 1e-12).astype(np.float32)
    np.savez_compressed(
        path,
        h_freq=np.asarray(h_freq, dtype=np.complex64),
        gains=gains.reshape(-1, gains.shape[-1]),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return Artifact("channel.ofdm_channel_state.numpy", path, metadata)


class DelayedCsiReliabilityTests(unittest.TestCase):
    def test_small_recipe_executes_the_aligned_delayed_csi_path(self) -> None:
        payload = deepcopy(load_recipe(RECIPE_PATH).to_dict())
        payload["name"] = "small_delayed_csi_reliability"
        payload["metadata"].pop("matrix", None)
        for step in payload["steps"]:
            if step["id"] == "data":
                step["params"].update({"bit_count": 64, "batch_size": 2})
            elif step["id"] == "channel_state":
                step["params"].update(
                    {
                        "ofdm_fft_size": 8,
                        "num_ofdm_symbols": 4,
                        "capacity_multiplier": 2,
                    }
                )
            elif step["id"] == "csi_observation":
                step["params"].update(
                    {
                        "feedback_delay_ofdm_symbols": 1,
                        "csi_history_length": 1,
                        "allocation_ofdm_symbols": 3,
                        "capture_temporal_stride": 1,
                    }
                )
            elif step["id"] == "tx_power":
                step["params"]["subcarrier_count"] = 8
            elif step["id"] == "wireless_channel":
                step["params"].update(
                    {"ofdm_fft_size": 8, "num_ofdm_symbols": 3}
                )
            elif step["id"] in {"coded_bler", "payload_bler"}:
                step["params"]["block_size"] = 64
            elif step["id"] == "allocation_evaluation":
                step["params"].update(
                    {
                        "blocklength_channel_uses": 32,
                        "target_rate_bps_hz": 1.0,
                    }
                )

        def fake_state(block_count: int, params: dict, seed: int) -> dict:
            del seed
            fft_size = int(params["ofdm_fft_size"])
            symbol_count = int(params["num_ofdm_symbols"])
            frequency = np.linspace(
                0.6, 1.4, fft_size, dtype=np.float32
            )[None, None, :]
            time = np.arange(symbol_count, dtype=np.float32)[None, :, None]
            blocks = np.arange(block_count, dtype=np.float32)[:, None, None]
            h_freq = (
                frequency
                * (1.0 + 0.05 * np.cos(0.3 * time + blocks))
                * np.exp(1j * (0.12 * time + 0.02 * blocks))
            )
            return {
                "h_freq": h_freq.astype(np.complex64),
                "tdl_model": "A",
                "subcarrier_spacing_khz": float(
                    params["subcarrier_spacing_khz"]
                ),
                "carrier_frequency_ghz": float(
                    params["carrier_frequency_ghz"]
                ),
                "delay_spread_ns": float(params["delay_spread_ns"]),
                "mobility_kmh": float(params["mobility_kmh"]),
            }

        def fake_apply(
            symbols: np.ndarray,
            h_freq: np.ndarray,
            noise_variance: float,
            params: dict,
            seed: int,
        ):
            del h_freq, params, seed
            values = np.asarray(symbols, dtype=np.complex64)
            return values, {
                "backend": "sionna",
                "backend_detail": "deterministic_test_double",
                "preset": "ofdm_tdl",
                "noise_variance": float(noise_variance),
                "channel_equalized": True,
                "equalizer": "deterministic_test_double",
                "executed_channel_use_count": int(values.size),
                "payload_symbol_count": int(values.size),
                "grid_padding_symbol_count": 0,
            }

        available = {
            "available": True,
            "extra": "wireless",
            "missing": [],
            "reason": "",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch(
                "noema_lab.ops.channel.digital._generate_sionna_ofdm_channel_state",
                side_effect=fake_state,
            ),
            mock.patch(
                "noema_lab.ops.channel.digital._apply_sionna_realized_ofdm_channel",
                side_effect=fake_apply,
            ),
            mock.patch(
                "noema_lab.ops.channel.digital._sionna_availability",
                return_value=available,
            ),
        ):
            root = Path(tmp)
            run_dir = LocalExecutor(
                build_registry(), LocalStore(root / ".noema")
            ).run(recipe_from_dict(payload))
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["status"], "completed")
        evaluation = next(
            row
            for row in summary["steps"]
            if row["id"] == "allocation_evaluation"
        )
        self.assertIn(
            "resource.finite_blocklength.expected_goodput_bps_hz",
            evaluation["metrics"],
        )
        self.assertGreater(
            evaluation["metrics"][
                "resource.csi.observed_actual_complex_correlation"
            ],
            0.0,
        )

    def test_delayed_csi_slices_each_block_from_one_causal_trajectory(self) -> None:
        h_freq = (
            np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
            + 1j
            * np.arange(100, 100 + 2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
        ).astype(np.complex64)
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "subcarrier_spacing_khz": 15.0,
            "channel_state_seed": 71,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = OfdmDelayedCsiOperation().run(
                OperationContext(
                    recipe_name="same_trajectory_delayed_csi",
                    step_id="csi_observation",
                    params={
                        "feedback_delay_ofdm_symbols": 2,
                        "allocation_ofdm_symbols": 4,
                        "add_estimation_noise": False,
                        "seed": 72,
                    },
                    inputs={
                        "state": _state_artifact(root / "state.npz", h_freq, metadata)
                    },
                    run_dir=root,
                    step_dir=root / "csi_observation",
                )
            )
            with np.load(
                result.outputs["transmitter_csi"].path, allow_pickle=False
            ) as payload:
                observed = np.asarray(payload["h_freq"])
                observed_capture = np.asarray(payload["capture_gains"])
            with np.load(
                result.outputs["actual_state"].path, allow_pickle=False
            ) as payload:
                actual = np.asarray(payload["h_freq"])
                actual_capture = np.asarray(payload["capture_gains"])

        np.testing.assert_array_equal(observed, h_freq[:, :4, :])
        np.testing.assert_array_equal(actual, h_freq[:, 2:6, :])
        np.testing.assert_allclose(
            observed_capture,
            (np.abs(observed) ** 2).transpose(1, 0, 2).reshape(-1, 3),
        )
        np.testing.assert_allclose(
            actual_capture,
            (np.abs(actual) ** 2).transpose(1, 0, 2).reshape(-1, 3),
        )
        # In particular, the last old state of block 0 is paired inside block 0,
        # never with the start of independently generated block 1.
        np.testing.assert_array_equal(actual[0, -1], h_freq[0, 5])
        self.assertEqual(
            result.outputs["actual_state"].metadata["trajectory_pairing"],
            "same_sionna_tdl_block_causal_slice",
        )
        self.assertFalse(
            result.outputs["actual_state"].metadata[
                "trajectory_pairing_leaks_future_csi"
            ]
        )
        self.assertEqual(
            result.outputs["actual_state"].metadata["array"],
            "capture_gains",
        )
        self.assertEqual(
            result.outputs["actual_state"].metadata["capture_record_order"],
            "ofdm_symbol_then_independent_tdl_block",
        )

    def test_causal_complex_history_and_current_capture_share_stride(self) -> None:
        h_freq = (
            np.arange(12 * 3, dtype=np.float32).reshape(1, 12, 3)
            + 1j
            * np.arange(100, 100 + 12 * 3, dtype=np.float32).reshape(
                1, 12, 3
            )
        ).astype(np.complex64)
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "subcarrier_spacing_khz": 15.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = OfdmDelayedCsiOperation().run(
                OperationContext(
                    recipe_name="causal_history",
                    step_id="csi_observation",
                    params={
                        "feedback_delay_ofdm_symbols": 2,
                        "csi_history_length": 3,
                        "allocation_ofdm_symbols": 6,
                        "capture_temporal_stride": 2,
                        "add_estimation_noise": False,
                    },
                    inputs={
                        "state": _state_artifact(
                            root / "state.npz", h_freq, metadata
                        )
                    },
                    run_dir=root,
                    step_dir=root / "csi_observation",
                )
            )
            with np.load(
                result.outputs["transmitter_csi"].path,
                allow_pickle=False,
            ) as payload:
                observed = np.asarray(payload["h_freq"])
                history = np.asarray(payload["csi_history"])
                captured_history = np.asarray(
                    payload["capture_csi_history"]
                )
            with np.load(
                result.outputs["actual_state"].path,
                allow_pickle=False,
            ) as payload:
                captured_actual = np.asarray(payload["capture_gains"])

        expected_history_complex = np.stack(
            [h_freq[:, offset : offset + 6, :] for offset in range(3)],
            axis=2,
        )
        expected_history_iq = np.stack(
            [
                expected_history_complex.real,
                expected_history_complex.imag,
            ],
            axis=-1,
        ).astype(np.float32)
        np.testing.assert_array_equal(observed, h_freq[:, 2:8, :])
        np.testing.assert_array_equal(history, expected_history_iq)
        np.testing.assert_array_equal(
            captured_history,
            expected_history_iq[:, [0, 2, 4], :, :, :]
            .transpose(1, 0, 2, 3, 4)
            .reshape(3, 3, 3, 2),
        )
        np.testing.assert_allclose(
            captured_actual,
            (np.abs(h_freq[:, [4, 6, 8], :]) ** 2)
            .transpose(1, 0, 2)
            .reshape(3, 3),
        )
        output_metadata = result.outputs["transmitter_csi"].metadata
        self.assertEqual(output_metadata["array"], "capture_csi_history")
        self.assertEqual(output_metadata["capture_temporal_stride"], 2)
        self.assertEqual(
            output_metadata["capture_selected_allocation_symbol_indices"],
            [0, 2, 4],
        )
        self.assertEqual(output_metadata["capture_record_count"], 3)
        self.assertFalse(
            output_metadata[
                "csi_history_contains_current_or_future_state"
            ]
        )

    def test_causal_allocator_dispatches_history_without_changing_legacy_abi(
        self,
    ) -> None:
        registry = build_registry()
        legacy_abi = registry.get(
            "model.symbol_power_allocator"
        ).describe()["trained_artifact_abi"]
        history_abi = registry.get(
            "model.causal_csi_power_allocator"
        ).describe()["trained_artifact_abi"]
        self.assertEqual(
            set(legacy_abi["inputs"]),
            {"channel_gain", "noise_variance", "average_power_budget"},
        )
        self.assertEqual(
            set(history_abi["inputs"]),
            {"csi_history", "noise_variance", "average_power_budget"},
        )
        self.assertEqual(
            history_abi["inputs"]["csi_history"]["shape"],
            ["batch", "history", "subcarrier", 2],
        )

        h_freq = np.asarray(
            [[[1 + 0j, 0.8 + 0.2j, 0.3 - 0.1j],
              [0.9 + 0.1j, 0.7 + 0.3j, 0.4 - 0.2j]]],
            dtype=np.complex64,
        )
        csi_history_complex = np.stack(
            [
                0.8 * h_freq,
                0.9 * h_freq,
                h_freq,
            ],
            axis=2,
        )
        csi_history = np.stack(
            [csi_history_complex.real, csi_history_complex.imag],
            axis=-1,
        ).astype(np.float32)
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "csi_history_length": 3,
            "csi_role": "delayed_noisy_transmitter_observation",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            state_path = root / "state.npz"
            manifest_path = root / "trained_artifact.yaml"
            manifest_path.write_text("schema_version: 2\n", encoding="utf-8")
            np.savez_compressed(
                symbols_path,
                symbols=np.ones((6,), dtype=np.complex64),
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            np.savez_compressed(
                state_path,
                h_freq=h_freq,
                gains=(np.abs(h_freq) ** 2).reshape(2, 3),
                csi_history=csi_history,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            captured_inputs = {}

            def fake_artifact_runtime(
                _manifest,
                _entrypoint,
                inputs,
                **_kwargs,
            ):
                captured_inputs.update(inputs)
                return {
                    "allocation_scores": np.zeros(
                        (2, 3), dtype=np.float32
                    )
                }

            with mock.patch(
                "noema_lab.core.trained_artifact_runtime."
                "run_trained_artifact_entrypoint",
                side_effect=fake_artifact_runtime,
            ):
                result = CausalCsiPowerAllocatorOperation().run(
                    OperationContext(
                        recipe_name="causal_history_allocator",
                        step_id="tx_power",
                        params={
                            "policy": "learned_artifact",
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": 1.0,
                            "subcarrier_count": 3,
                            "artifact_manifest_path": str(manifest_path),
                            "artifact_package_sha256": "0" * 64,
                        },
                        inputs={
                            "symbols": Artifact(
                                "channel.symbols.complex_numpy",
                                symbols_path,
                                metadata,
                            ),
                            "channel_state": Artifact(
                                "channel.ofdm_channel_state.numpy",
                                state_path,
                                metadata,
                            ),
                        },
                        run_dir=root,
                        step_dir=root / "tx_power",
                    )
                )

        self.assertIn("csi_history", captured_inputs)
        self.assertNotIn("channel_gain", captured_inputs)
        np.testing.assert_array_equal(
            captured_inputs["csi_history"],
            csi_history.reshape(2, 3, 3, 2),
        )
        self.assertEqual(
            result.outputs["allocation"].metadata[
                "model_input_representation"
            ],
            "causal_complex_csi_history_iq",
        )

    def test_zero_delay_without_estimation_noise_is_an_exact_control(self) -> None:
        rng = np.random.RandomState(73)
        h_freq = (
            rng.standard_normal((2, 4, 5))
            + 1j * rng.standard_normal((2, 4, 5))
        ).astype(np.complex64)
        metadata = {
            "noise_variance": 0.5,
            "average_power_budget": 1.0,
            "subcarrier_spacing_khz": 30.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = OfdmDelayedCsiOperation().run(
                OperationContext(
                    recipe_name="zero_delay_control",
                    step_id="csi_observation",
                    params={
                        "feedback_delay_ofdm_symbols": 0,
                        "allocation_ofdm_symbols": 4,
                        "add_estimation_noise": False,
                    },
                    inputs={
                        "state": _state_artifact(root / "state.npz", h_freq, metadata)
                    },
                    run_dir=root,
                    step_dir=root / "csi_observation",
                )
            )

        self.assertAlmostEqual(
            result.metrics["resource.csi.observed_actual_complex_correlation"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result.metrics["resource.csi.observed_actual_complex_nmse"],
            0.0,
            places=12,
        )

    def test_finite_blocklength_metric_uses_current_not_observed_gain(self) -> None:
        actual_h = np.sqrt(
            np.asarray([[[0.1, 0.1], [10.0, 10.0]]], dtype=np.float32)
        ).astype(np.complex64)
        # Give both rows the same transmitter observation. Their reliability
        # must still differ because scoring is against the later actual state.
        observed_h = np.ones_like(actual_h, dtype=np.complex64)
        metadata = {
            "noise_variance": 1.0,
            "average_power_budget": 1.0,
            "feedback_delay_ofdm_symbols": 2,
            "feedback_delay_seconds_nominal_without_cp": 1e-4,
        }
        power = np.ones((2, 2), dtype=np.float32)
        allocation_metadata = {
            **metadata,
            "total_power": 2.0,
            "target_power": 1.0,
            "policy": "fixed",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allocation_path = root / "allocation.npz"
            np.savez_compressed(
                allocation_path,
                power=power,
                metadata_json=json.dumps(allocation_metadata, sort_keys=True),
            )
            result = OfdmFiniteBlocklengthReliabilityMetricsOperation().run(
                OperationContext(
                    recipe_name="finite_blocklength_metric",
                    step_id="allocation_evaluation",
                    params={
                        "blocklength_channel_uses": 128,
                        "target_rate_bps_hz": 2.0,
                    },
                    inputs={
                        "actual_state": _state_artifact(
                            root / "actual.npz", actual_h, metadata
                        ),
                        "transmitter_csi": _state_artifact(
                            root / "observed.npz", observed_h, metadata
                        ),
                        "allocation": Artifact(
                            "channel.power_allocation.numpy",
                            allocation_path,
                            allocation_metadata,
                        ),
                    },
                    run_dir=root,
                    step_dir=root / "allocation_evaluation",
                )
            )

        rows = result.metadata["rows"]
        self.assertGreater(rows[0]["predicted_bler"], rows[1]["predicted_bler"])
        self.assertLess(
            rows[0]["expected_goodput_bps_hz"],
            rows[1]["expected_goodput_bps_hz"],
        )
        self.assertEqual(
            result.metrics["task.score"],
            result.metrics[
                "resource.finite_blocklength.expected_goodput_bps_hz"
            ],
        )
        self.assertFalse(
            result.metadata["metadata"]["current_csi_available_to_allocator"]
        )

    def test_observed_and_uncertainty_shrunk_water_filling_are_named_baselines(
        self,
    ) -> None:
        gains = np.asarray([[[0.01, 0.1, 1.0, 10.0]]], dtype=np.float32)
        h_freq = np.sqrt(gains).astype(np.complex64)
        symbols = np.ones((4,), dtype=np.complex64)
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "reference_snr_db": float(10.0 * np.log10(1.0 / 0.2)),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=symbols,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            state = _state_artifact(root / "observed.npz", h_freq, metadata)
            outputs = {}
            for policy in (
                "observed_csi_water_filling",
                "robust_csi_water_filling",
            ):
                outputs[policy] = SymbolPowerAllocatorOperation().run(
                    OperationContext(
                        recipe_name="delayed_csi_baselines",
                        step_id=policy,
                        params={
                            "policy": policy,
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": 1.0,
                            "csi_gain_shrinkage": 0.4,
                        },
                        inputs={
                            "symbols": Artifact(
                                "channel.symbols.complex_numpy",
                                symbols_path,
                                metadata,
                            ),
                            "channel_state": state,
                        },
                        run_dir=root,
                        step_dir=root / policy,
                    )
                )

            with np.load(
                outputs["observed_csi_water_filling"].outputs["allocation"].path,
                allow_pickle=False,
            ) as payload:
                observed_power = np.asarray(payload["power"])
            with np.load(
                outputs["robust_csi_water_filling"].outputs["allocation"].path,
                allow_pickle=False,
            ) as payload:
                robust_power = np.asarray(payload["power"])

        np.testing.assert_allclose(np.sum(observed_power, axis=1), [4.0], atol=1e-6)
        np.testing.assert_allclose(np.sum(robust_power, axis=1), [4.0], atol=1e-6)
        self.assertFalse(np.allclose(observed_power, robust_power))
        self.assertEqual(
            outputs["observed_csi_water_filling"]
            .outputs["allocation"]
            .metadata["policy"],
            "water_filling_on_observed_csi",
        )
        self.assertEqual(
            outputs["robust_csi_water_filling"]
            .outputs["allocation"]
            .metadata["policy"],
            "uncertainty_shrunk_water_filling",
        )

    def test_causal_ar_predictive_water_filling_uses_history_only(self) -> None:
        history_complex = np.asarray(
            [
                [
                    [
                        [1.0 + 0.0j, 0.4 + 0.0j, 0.2 + 0.0j, 0.1 + 0.0j],
                        [0.9 + 0.1j, 0.45 + 0.0j, 0.2 + 0.05j, 0.1 + 0.0j],
                        [0.8 + 0.2j, 0.5 + 0.0j, 0.2 + 0.1j, 0.1 + 0.0j],
                        [0.7 + 0.3j, 0.55 + 0.0j, 0.2 + 0.15j, 0.1 + 0.0j],
                    ]
                ]
            ],
            dtype=np.complex64,
        )
        history_iq = np.stack(
            [history_complex.real, history_complex.imag],
            axis=-1,
        ).astype(np.float32)
        newest = history_complex[:, :, -1, :]
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "feedback_delay_ofdm_symbols": 2,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            state_path = root / "history.npz"
            np.savez_compressed(
                symbols_path,
                symbols=np.ones((4,), dtype=np.complex64),
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            np.savez_compressed(
                state_path,
                h_freq=newest,
                gains=np.maximum(np.abs(newest) ** 2, 1e-12).reshape(1, 4),
                csi_history=history_iq,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            result = CausalCsiPowerAllocatorOperation().run(
                OperationContext(
                    recipe_name="causal_ar_baseline",
                    step_id="tx_power",
                    params={
                        "policy": "causal_ar_water_filling",
                        "granularity": "per_subcarrier",
                        "budget_mode": "fixed_average",
                        "target_power": 1.0,
                        "subcarrier_count": 4,
                        "csi_prediction_horizon_ofdm_symbols": 0,
                        "csi_prediction_gain_confidence": 0.4,
                    },
                    inputs={
                        "symbols": Artifact(
                            "channel.symbols.complex_numpy",
                            symbols_path,
                            metadata,
                        ),
                        "channel_state": Artifact(
                            "channel.ofdm_channel_state.numpy",
                            state_path,
                            metadata,
                        ),
                    },
                    run_dir=root,
                    step_dir=root / "tx_power",
                )
            )
            with np.load(
                result.outputs["allocation"].path,
                allow_pickle=False,
            ) as payload:
                power = np.asarray(payload["power"])

        np.testing.assert_allclose(np.sum(power, axis=1), [4.0], atol=1e-6)
        allocation_metadata = result.outputs["allocation"].metadata
        self.assertEqual(
            allocation_metadata["policy"],
            "causal_complex_ar_prediction_water_filling",
        )
        self.assertEqual(
            allocation_metadata["csi_prediction_horizon_ofdm_symbols"],
            2,
        )

    def test_recipe_is_registered_lint_clean_and_packaged_byte_exact(self) -> None:
        self.assertEqual(RECIPE_PATH.read_bytes(), PACKAGED_RECIPE_PATH.read_bytes())
        registry = build_registry()
        recipe = load_recipe(RECIPE_PATH)
        report = lint_recipe_invariants(recipe, registry, strict=True)
        self.assertEqual(report["status"], "passed", report["issues"])
        self.assertEqual(report["error_count"], 0, report["issues"])

        steps = {step.id: step for step in recipe.steps}
        self.assertEqual(
            steps["channel_state"].params["tdl_model"], "C"
        )
        self.assertEqual(
            steps["channel_state"].params["ofdm_fft_size"], 128
        )
        self.assertEqual(
            steps["channel_state"].params["delay_spread_ns"], 1000.0
        )
        self.assertFalse(
            steps["channel_state"].params["normalize_channel"]
        )
        self.assertEqual(
            steps["channel_state"].params["capacity_multiplier"], 16
        )
        self.assertEqual(
            steps["csi_observation"].params["csi_history_length"], 4
        )
        self.assertEqual(
            steps["csi_observation"].params["capture_temporal_stride"], 12
        )
        self.assertEqual(
            steps["tx_power"].op, "model.causal_csi_power_allocator"
        )
        self.assertEqual(
            steps["tx_power"].inputs["channel_state"],
            "csi_observation.transmitter_csi",
        )
        self.assertEqual(
            steps["wireless_channel"].inputs["channel_state"],
            "csi_observation.actual_state",
        )
        self.assertEqual(
            steps["allocation_evaluation"].inputs["actual_state"],
            "csi_observation.actual_state",
        )

    def test_linter_rejects_current_csi_leakage_into_allocator(self) -> None:
        payload = load_recipe(RECIPE_PATH).to_dict()
        allocator = next(
            step for step in payload["steps"] if step["id"] == "tx_power"
        )
        allocator["inputs"]["channel_state"] = "csi_observation.actual_state"

        report = lint_recipe_invariants(
            recipe_from_dict(payload), build_registry(), strict=True
        )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(
                issue["code"] == "input_reference_invalid"
                and issue.get("step_id") == "tx_power"
                for issue in report["issues"]
            ),
            report["issues"],
        )

    def test_perfect_csi_water_filling_label_is_rejected_on_delayed_csi(
        self,
    ) -> None:
        payload = load_recipe(RECIPE_PATH).to_dict()
        allocator = next(
            step for step in payload["steps"] if step["id"] == "tx_power"
        )
        allocator["params"]["policy"] = "water_filling"
        report = lint_recipe_invariants(
            recipe_from_dict(payload), build_registry(), strict=True
        )
        self.assertTrue(
            any(
                issue["code"] == "delayed_csi_water_filling_policy_invalid"
                for issue in report["issues"]
            ),
            report["issues"],
        )

        gains = np.asarray([[[0.2, 0.5, 1.0, 2.0]]], dtype=np.float32)
        h_freq = np.sqrt(gains).astype(np.complex64)
        metadata = {
            "noise_variance": 0.2,
            "average_power_budget": 1.0,
            "csi_role": "delayed_noisy_transmitter_observation",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols_path = root / "symbols.npz"
            np.savez_compressed(
                symbols_path,
                symbols=np.ones((4,), dtype=np.complex64),
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
            with self.assertRaisesRegex(
                OperationError, "observed_csi_water_filling"
            ):
                SymbolPowerAllocatorOperation().run(
                    OperationContext(
                        recipe_name="mislabelled_delayed_water_filling",
                        step_id="tx_power",
                        params={
                            "policy": "water_filling",
                            "granularity": "per_subcarrier",
                            "budget_mode": "fixed_average",
                            "target_power": 1.0,
                        },
                        inputs={
                            "symbols": Artifact(
                                "channel.symbols.complex_numpy",
                                symbols_path,
                                metadata,
                            ),
                            "channel_state": _state_artifact(
                                root / "state.npz", h_freq, metadata
                            ),
                        },
                        run_dir=root,
                        step_dir=root / "tx_power",
                    )
                )


if __name__ == "__main__":
    unittest.main()
