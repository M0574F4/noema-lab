from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.ops.channel.digital import OfdmPowerAllocationMetricsOperation


class OfdmPowerAllocationMetricsTests(unittest.TestCase):
    def _run_metrics(
        self,
        root: Path,
        *,
        gains: np.ndarray,
        power: np.ndarray,
        noise_variance: float,
        target_power: float,
        total_power: float,
        reference_bits: np.ndarray | None = None,
        candidate_bits: np.ndarray | None = None,
        tx_symbols: np.ndarray | None = None,
        metric_params: dict | None = None,
        allocation_metadata_overrides: dict | None = None,
    ):
        state_path = root / "state.npz"
        allocation_path = root / "allocation.npz"
        state_metadata = {
            "noise_variance": noise_variance,
            "reference_snr_db": float(-10.0 * np.log10(noise_variance)),
            "average_power_budget": target_power,
            "total_power_budget": total_power,
        }
        allocation_metadata = {
            **state_metadata,
            "target_power": target_power,
            "total_power": total_power,
            **dict(allocation_metadata_overrides or {}),
        }
        np.savez_compressed(
            state_path,
            gains=np.asarray(gains),
            metadata_json=json.dumps(state_metadata),
        )
        np.savez_compressed(
            allocation_path,
            power=np.asarray(power),
            metadata_json=json.dumps(allocation_metadata),
        )
        inputs = {
            "state": Artifact("channel.ofdm_channel_state.numpy", state_path, state_metadata),
            "allocation": Artifact(
                "channel.power_allocation.numpy",
                allocation_path,
                allocation_metadata,
            ),
        }
        if reference_bits is not None and candidate_bits is not None and tx_symbols is not None:
            reference_path = root / "reference_bits.npz"
            candidate_path = root / "candidate_bits.npz"
            symbols_path = root / "tx_symbols.npz"
            delivery_metadata = {
                "adaptive_scheduled_resource_element_count": int(np.asarray(tx_symbols).size),
                "adaptive_data_resource_element_count": int(np.count_nonzero(np.asarray(tx_symbols))),
            }
            np.savez_compressed(reference_path, bits=np.asarray(reference_bits, dtype=np.uint8))
            np.savez_compressed(candidate_path, bits=np.asarray(candidate_bits, dtype=np.uint8))
            np.savez_compressed(
                symbols_path,
                symbols=np.asarray(tx_symbols, dtype=np.complex64),
                metadata_json=json.dumps(delivery_metadata),
            )
            inputs.update(
                {
                    "reference": Artifact("channel.payload_bits.numpy", reference_path, {}),
                    "candidate": Artifact("channel.payload_bits.numpy", candidate_path, {}),
                    "symbols": Artifact(
                        "channel.symbols.complex_numpy", symbols_path, delivery_metadata
                    ),
                }
            )
        return OfdmPowerAllocationMetricsOperation().run(
            OperationContext(
                recipe_name="ofdm_allocation_metrics_test",
                step_id="allocation_evaluation",
                params=dict(metric_params or {}),
                inputs=inputs,
                run_dir=root,
                step_dir=root / "allocation_evaluation",
            )
        )

    def test_theoretical_sum_is_subcarrier_count_times_normalized_spectral_efficiency(self):
        gains = np.asarray(
            [
                [0.5, 1.0, 2.0, 4.0],
                [1.0, 1.5, 2.5, 3.0],
            ],
            dtype=np.float64,
        )
        power = np.asarray(
            [
                [0.25, 0.75, 1.25, 1.75],
                [1.75, 1.25, 0.75, 0.25],
            ],
            dtype=np.float64,
        )
        noise_variance = 0.5

        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_metrics(
                Path(tmp),
                gains=gains,
                power=power,
                noise_variance=noise_variance,
                target_power=1.0,
                total_power=4.0,
            )

        rates = np.log2(1.0 + gains * power / noise_variance)
        expected_sum = float(np.mean(np.sum(rates, axis=1)))
        expected_spectral_efficiency = float(np.mean(rates))
        metrics = result.metrics
        self.assertAlmostEqual(
            metrics["resource.theoretical_shannon_sum_bits_per_ofdm_symbol"],
            expected_sum,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["resource.theoretical_shannon_spectral_efficiency_bps_hz"],
            expected_spectral_efficiency,
            places=12,
        )
        self.assertAlmostEqual(expected_sum, gains.shape[1] * expected_spectral_efficiency, places=12)
        self.assertAlmostEqual(metrics["task.score"], expected_spectral_efficiency, places=12)

    def test_active_resource_element_fraction_uses_scale_aware_strict_threshold(self):
        target_power = 2.0
        threshold = target_power * 1e-9
        power = np.asarray(
            [[0.0, 1e-12, threshold, threshold * 1.05, 10.0 - threshold * 2.05 - 1e-12]],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_metrics(
                Path(tmp),
                gains=np.ones_like(power),
                power=power,
                noise_variance=1.0,
                target_power=target_power,
                total_power=10.0,
            )

        self.assertEqual(result.metadata["metadata"]["active_power_threshold"], threshold)
        self.assertAlmostEqual(result.metrics["resource.active_resource_element_fraction"], 2.0 / 5.0)
        self.assertEqual(result.metadata["rows"][0]["active_subcarrier_count"], 2)

    def test_sum_budget_infeasible_allocation_is_rejected_before_scoring(self):
        total_power = 4_000_000_000.0
        power = np.asarray(
            [
                [1_000_000_000.0, 1_000_000_000.0, 1_000_000_000.0, 1_000_000_800.0],
                [1_000_000_000.0, 1_000_000_000.0, 1_000_000_000.0, 999_999_600.0],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OperationError, "sum-power budget"):
                self._run_metrics(
                    Path(tmp),
                    gains=np.ones_like(power),
                    power=power,
                    noise_variance=0.25,
                    target_power=1_000_000_000.0,
                    total_power=total_power,
                )

    def test_negative_power_or_forged_noise_metadata_cannot_raise_score(self):
        gains = np.ones((1, 4), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "negative").mkdir()
            with self.assertRaisesRegex(OperationError, "negative power"):
                self._run_metrics(
                    root / "negative",
                    gains=gains,
                    power=np.asarray([[-100.0, 0.0, 0.0, 104.0]]),
                    noise_variance=1.0,
                    target_power=1.0,
                    total_power=4.0,
                )

            (root / "forged_noise").mkdir()
            with self.assertRaisesRegex(
                OperationError, "noise_variance.*contradicts"
            ):
                self._run_metrics(
                    root / "forged_noise",
                    gains=gains,
                    power=np.ones_like(gains),
                    noise_variance=1.0,
                    target_power=1.0,
                    total_power=4.0,
                    allocation_metadata_overrides={"noise_variance": 1e-9},
                )

    def test_nonfinite_and_shape_invalid_allocations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nonfinite").mkdir()
            with self.assertRaisesRegex(OperationError, "finite values"):
                self._run_metrics(
                    root / "nonfinite",
                    gains=np.ones((1, 4), dtype=np.float64),
                    power=np.asarray([[1.0, 1.0, np.nan, 1.0]]),
                    noise_variance=1.0,
                    target_power=1.0,
                    total_power=4.0,
                )

            (root / "shape").mkdir()
            with self.assertRaisesRegex(OperationError, "non-empty 2-D"):
                self._run_metrics(
                    root / "shape",
                    gains=np.ones((1, 4), dtype=np.float64),
                    power=np.ones((4,), dtype=np.float64),
                    noise_variance=1.0,
                    target_power=1.0,
                    total_power=4.0,
                )

    def test_measured_allocation_diagnostics_distinguish_equal_and_gain_aligned_power(self):
        gains = np.asarray([[1.0, 2.0, 4.0, 8.0]], dtype=np.float64)
        equal_power = np.ones_like(gains)
        gain_aligned_power = np.asarray([[0.25, 0.5, 1.0, 2.25]], dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "equal").mkdir()
            (root / "gain_aligned").mkdir()
            equal = self._run_metrics(
                root / "equal",
                gains=gains,
                power=equal_power,
                noise_variance=0.5,
                target_power=1.0,
                total_power=4.0,
            )
            gain_aligned = self._run_metrics(
                root / "gain_aligned",
                gains=gains,
                power=gain_aligned_power,
                noise_variance=0.5,
                target_power=1.0,
                total_power=4.0,
            )

        self.assertEqual(equal.metrics["resource.allocation_power_coefficient_of_variation"], 0.0)
        self.assertEqual(equal.metrics["resource.allocated_power_weighted_channel_gain_lift"], 0.0)
        self.assertGreater(gain_aligned.metrics["resource.allocation_power_coefficient_of_variation"], 0.0)
        self.assertGreater(gain_aligned.metrics["resource.allocated_power_weighted_channel_gain_lift"], 0.0)

    def test_measured_delivery_metrics_report_actual_goodput_and_energy(self):
        reference = np.asarray([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_metrics(
                Path(tmp),
                gains=np.ones((1, 4), dtype=np.float64),
                power=np.ones((1, 4), dtype=np.float64),
                noise_variance=0.25,
                target_power=1.0,
                total_power=4.0,
                reference_bits=reference,
                candidate_bits=reference.copy(),
                tx_symbols=np.ones((4,), dtype=np.complex64),
            )

        self.assertEqual(result.metrics["channel.payload_delivery_success"], 1)
        self.assertEqual(result.metrics["channel.payload_delivery_block_error_rate"], 0.0)
        self.assertEqual(
            result.metrics["channel.achieved_payload_goodput_bits_per_resource_element"], 2.0
        )
        self.assertEqual(result.metrics["channel.tx_energy_per_delivered_payload_bit"], 0.5)
        self.assertAlmostEqual(
            result.metrics["task.score"],
            math.log2(5.0),
            places=12,
        )

    def test_delivery_goodput_and_bler_are_scored_per_transport_block(self):
        reference = np.asarray([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        candidate = reference.copy()
        candidate[5] ^= np.uint8(1)
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_metrics(
                Path(tmp),
                gains=np.ones((2, 4), dtype=np.float64),
                power=np.ones((2, 4), dtype=np.float64),
                noise_variance=0.25,
                target_power=1.0,
                total_power=4.0,
                reference_bits=reference,
                candidate_bits=candidate,
                tx_symbols=np.ones((6,), dtype=np.complex64),
                metric_params={"transport_block_size_bits": 4},
            )

        self.assertEqual(result.metrics["channel.payload_delivery_block_count"], 3)
        self.assertEqual(result.metrics["channel.payload_delivery_block_error_count"], 1)
        self.assertAlmostEqual(result.metrics["channel.payload_delivery_block_error_rate"], 1.0 / 3.0)
        self.assertEqual(result.metrics["channel.delivered_payload_bit_count"], 8)
        self.assertEqual(result.metrics["channel.payload_delivery_success"], 0)
        self.assertEqual(result.metrics["channel.offered_payload_bits_per_resource_element"], 2.0)
        self.assertAlmostEqual(
            result.metrics["channel.achieved_payload_goodput_bits_per_resource_element"],
            8.0 / 6.0,
        )
        self.assertAlmostEqual(
            result.metrics["channel.payload_energy_efficiency_bits_per_normalized_energy"],
            8.0 / 6.0,
        )

    def test_water_filling_oracle_gap_and_outage_are_explicit(self):
        gains = np.asarray(
            [[0.1, 0.5, 2.0, 4.0], [0.2, 0.8, 1.5, 3.0]], dtype=np.float64
        )
        equal_power = np.ones_like(gains)
        from noema_lab.ops.ai_phy import _water_filling_allocation

        oracle_power = np.stack(
            [_water_filling_allocation(row, 0.2, 4.0)[0] for row in gains], axis=0
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "equal").mkdir()
            (root / "oracle").mkdir()
            equal = self._run_metrics(
                root / "equal",
                gains=gains,
                power=equal_power,
                noise_variance=0.2,
                target_power=1.0,
                total_power=4.0,
                metric_params={"outage_target_spectral_efficiency_bps_hz": 2.0},
            )
            oracle = self._run_metrics(
                root / "oracle",
                gains=gains,
                power=oracle_power,
                noise_variance=0.2,
                target_power=1.0,
                total_power=4.0,
                metric_params={"outage_target_spectral_efficiency_bps_hz": 2.0},
            )

        self.assertGreater(equal.metrics["resource.water_filling_optimality_gap_bps_hz"], 0.0)
        self.assertGreater(equal.metrics["resource.water_filling_relative_optimality_gap"], 0.0)
        self.assertGreater(equal.metrics["resource.water_filling_power_normalized_rmse"], 0.0)
        self.assertGreater(equal.metrics["resource.water_filling_kkt_normalized_residual"], 0.0)
        self.assertAlmostEqual(oracle.metrics["resource.water_filling_optimality_gap_bps_hz"], 0.0, places=12)
        self.assertAlmostEqual(oracle.metrics["resource.water_filling_relative_optimality_gap"], 0.0, places=12)
        self.assertAlmostEqual(oracle.metrics["resource.water_filling_power_normalized_rmse"], 0.0, places=7)
        self.assertLess(oracle.metrics["resource.water_filling_kkt_normalized_residual"], 1e-10)
        self.assertEqual(oracle.metrics["resource.power_constraint.max_negative_violation"], 0.0)
        expected_state_rates = np.mean(np.log2(1.0 + gains * equal_power / 0.2), axis=1)
        self.assertAlmostEqual(
            equal.metrics["resource.theoretical_outage_probability"],
            float(np.mean(expected_state_rates < 2.0)),
        )
        self.assertAlmostEqual(
            equal.metrics["resource.theoretical_p05_state_spectral_efficiency_bps_hz"],
            float(np.percentile(expected_state_rates, 5.0)),
        )


if __name__ == "__main__":
    unittest.main()
