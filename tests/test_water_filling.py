from __future__ import annotations

from copy import deepcopy
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.benchmarks import (
    _load_benchmark_recipe,
    load_benchmark_pack,
    validate_benchmark_pack,
)
from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry
from noema_lab.ops.ai_phy import _resource_allocation_preview, _water_filling_allocation


ROOT = Path(__file__).resolve().parents[1]


def _step_by_id(recipe, step_id):
    return next(step for step in recipe.steps if step.id == step_id)


def _topology_for(recipe, step_ids=None):
    selected = recipe.steps if step_ids is None else [_step_by_id(recipe, step_id) for step_id in step_ids]
    return [(step.id, step.op, step.inputs) for step in selected]


def _resource_recipe_without_permitted_policy_differences(recipe):
    payload = deepcopy(recipe.to_dict())
    payload.pop("name", None)
    payload.pop("description", None)
    metadata = payload.get("metadata", {})
    metadata.pop("research_stage", None)
    metadata.pop("power_allocator_policy", None)
    allocator = next(step for step in payload["steps"] if step["id"] == "tx_power")
    allocator["params"].pop("policy", None)
    return payload


class WaterFillingTests(unittest.TestCase):
    def test_templates_share_the_seeded_bit_sionna_graph(self):
        equal = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        water = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")

        self.assertEqual(_topology_for(equal), _topology_for(water))

        ops = {step.op for step in water.steps}
        self.assertIn("source.random_bits", ops)
        self.assertIn("channel.identity_encoder", ops)
        self.assertIn("channel.identity_decoder", ops)
        self.assertIn("wireless.ofdm_channel_state", ops)
        self.assertIn("model.symbol_power_allocator", ops)
        self.assertIn("wireless.channel", ops)
        self.assertIn("demodulation.digital_demodulate", ops)
        self.assertIn("metrics.bit_error_rate", ops)
        self.assertIn("metrics.block_error_rate", ops)
        self.assertIn("metrics.ofdm_power_allocation", ops)
        self.assertNotIn("source.image_dataset", ops)
        self.assertNotIn("model.compressai_analysis_encode", ops)
        self.assertNotIn("model.compressai_entropy_encode", ops)
        self.assertNotIn("model.compressai_entropy_decode", ops)
        self.assertNotIn("model.compressai_synthesis_decode", ops)
        self.assertNotIn("metrics.image_reconstruction", ops)
        self.assertNotIn("source.resource_allocation_scenario", ops)
        data = _step_by_id(water, "data")
        allocator = _step_by_id(water, "tx_power")
        channel = _step_by_id(water, "wireless_channel")
        state = _step_by_id(water, "channel_state")
        self.assertEqual(data.params["seed"], 23)
        self.assertEqual(data.params["bit_count"], 1024)
        self.assertEqual(data.params["batch_size"], 256)
        self.assertEqual(allocator.params["policy"], "water_filling")
        self.assertEqual(allocator.inputs["channel_state"], "channel_state.state")
        self.assertEqual(channel.inputs["channel_state"], "channel_state.state")
        self.assertEqual(state.inputs["symbols"], "modulator.symbols")
        self.assertEqual(channel.params["wireless_backend"], "sionna")
        self.assertEqual(channel.params["noise_mode"], "fixed_variance")
        self.assertFalse(
            water.dataset_capture,
            "training capture belongs to the selected training plan, not the neutral benchmark recipe",
        )

    def test_templates_differ_only_by_the_declared_allocation_policy(self):
        equal = load_recipe(ROOT / "recipes" / "resource_equal_power_baseline.yaml")
        water = load_recipe(ROOT / "recipes" / "resource_water_filling_baseline.yaml")

        self.assertEqual(_step_by_id(equal, "tx_power").params["policy"], "fixed")
        self.assertEqual(_step_by_id(water, "tx_power").params["policy"], "water_filling")
        self.assertEqual(equal.metadata["power_allocator_policy"], "equal_power")
        self.assertEqual(water.metadata["power_allocator_policy"], "water_filling")
        self.assertEqual(
            _resource_recipe_without_permitted_policy_differences(equal),
            _resource_recipe_without_permitted_policy_differences(water),
        )

    def test_templates_bind_state_truth_and_allocator_to_one_power_budget(self):
        for recipe_name in ("resource_equal_power_baseline", "resource_water_filling_baseline"):
            recipe = load_recipe(ROOT / "recipes" / f"{recipe_name}.yaml")
            channel_state = _step_by_id(recipe, "channel_state")
            allocator = _step_by_id(recipe, "tx_power")
            channel_encoder = _step_by_id(recipe, "channel_encoder")
            channel_decoder = _step_by_id(recipe, "channel_decoder")

            self.assertNotIn("target_power", channel_state.params)
            self.assertEqual(channel_state.params["average_power_budget"], 1.0)
            self.assertEqual(allocator.params["target_power"], 1.0)
            self.assertEqual(
                allocator.params["target_power"],
                recipe.metadata["average_tx_power_budget"],
            )
            self.assertEqual(
                channel_state.params["average_power_budget"],
                allocator.params["target_power"],
            )
            self.assertEqual(channel_encoder.op, "channel.identity_encoder")
            self.assertEqual(channel_decoder.op, "channel.identity_decoder")
            self.assertEqual(recipe.metadata["channel_code"], "identity")
            self.assertEqual(_step_by_id(recipe, "coded_bler").params["block_size"], 1024)
            self.assertEqual(_step_by_id(recipe, "payload_bler").params["block_size"], 1024)

    def test_templates_sweep_the_effective_power_budget_not_inactive_snr(self):
        for recipe_name in ("resource_equal_power_baseline", "resource_water_filling_baseline"):
            recipe = load_recipe(ROOT / "recipes" / f"{recipe_name}.yaml")
            matrix = recipe.metadata["matrix"]
            self.assertEqual(
                matrix["dimensions"]["resource.average_transmit_power_budget"],
                [0.5, 1.0, 2.0],
            )
            self.assertEqual(
                matrix["step_params"]["tx_power"]["target_power"],
                {"matrix": "resource.average_transmit_power_budget"},
            )
            self.assertEqual(
                matrix["step_params"]["channel_state"]["average_power_budget"],
                {"matrix": "resource.average_transmit_power_budget"},
            )
            self.assertNotIn("snr_db", matrix["step_params"].get("wireless_channel", {}))

    def test_resource_pack_binds_state_truth_and_decision_at_every_budget(self):
        pack = load_benchmark_pack(
            ROOT / "benchmarks" / "resource_allocation" / "power_allocation_v1.yaml"
        )
        validation = validate_benchmark_pack(pack, build_registry(), ROOT)
        self.assertEqual(validation["recipe_count"], 6)
        observed = set()
        for entry in pack.recipes:
            recipe = _load_benchmark_recipe(pack, entry, ROOT)
            state_budget = _step_by_id(
                recipe, "channel_state"
            ).params["average_power_budget"]
            decision_budget = _step_by_id(recipe, "tx_power").params[
                "target_power"
            ]
            selected_budget = entry.params["matrix_selection"][
                "resource.average_transmit_power_budget"
            ]
            self.assertEqual(state_budget, selected_budget)
            self.assertEqual(decision_budget, selected_budget)
            observed.add(selected_budget)
        self.assertEqual(observed, {0.5, 1.0, 2.0})

    def test_allocation_satisfies_sum_power_and_kkt_water_level(self):
        gains = np.asarray([0.1, 0.5, 2.0, 4.0], dtype=np.float64)
        noise_variance = 0.2
        power, water_level = _water_filling_allocation(gains, noise_variance, 1.0)

        self.assertAlmostEqual(float(np.sum(power)), 1.0, places=12)
        self.assertTrue(np.all(power >= 0.0))
        expected = np.maximum(water_level - noise_variance / gains, 0.0)
        np.testing.assert_allclose(power, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(float(power[0]), 0.0)

    def test_water_filling_rate_is_not_below_equal_power(self):
        gains = np.asarray([0.08, 0.3, 1.5, 5.0], dtype=np.float64)
        noise_variance = 0.1
        water_filling, _ = _water_filling_allocation(gains, noise_variance, 1.0)
        equal = np.full_like(gains, 0.25)

        oracle_rate = float(np.sum(np.log2(1.0 + gains * water_filling / noise_variance)))
        equal_rate = float(np.sum(np.log2(1.0 + gains * equal / noise_variance)))
        self.assertGreater(oracle_rate, equal_rate)

    def test_preview_exposes_subcarrier_snr_inverse_floor_and_power(self):
        gains = np.asarray([[0.1, 0.5, 2.0, 4.0]], dtype=np.float64)
        noise_variance = 0.2
        power, water_level = _water_filling_allocation(gains[0], noise_variance, 1.0)
        preview = _resource_allocation_preview(
            gains,
            power[None, :],
            noise_variance,
            {
                "scenario_kind": "ofdm_frequency_selective_parallel_channels",
                "allocation_axis": "subcarrier",
                "allocation_granularity": "per_subcarrier_per_channel_snapshot",
                "snapshot_axis": "ofdm_channel_state",
                "snr_db": 10.0,
                "total_power": 1.0,
            },
            {"policy": "theoretical_water_filling"},
            np.asarray([water_level]),
        )

        self.assertEqual(preview["allocation_axis"], "subcarrier")
        self.assertEqual(preview["channel_count"], 4)
        snapshot = preview["snapshots"][0]
        np.testing.assert_allclose(snapshot["inverse_unit_snr"], noise_variance / gains[0])
        np.testing.assert_allclose(snapshot["allocated_power"], power)
        self.assertGreater(snapshot["unit_power_snr_db"][3], snapshot["unit_power_snr_db"][1])
        self.assertGreater(snapshot["allocated_power"][3], snapshot["allocated_power"][1])


if __name__ == "__main__":
    unittest.main()
