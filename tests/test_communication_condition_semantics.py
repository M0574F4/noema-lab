from __future__ import annotations

import unittest

import numpy as np

from noema_lab.core.common_conditions import _power_evidence
from noema_lab.ops.channel.digital import _apply_wireless_channel_per_item


class CommunicationConditionSemanticsTests(unittest.TestCase):
    def test_unequal_dimensions_share_item_seed_namespace_not_shape(
        self,
    ) -> None:
        params = {
            "channel": "awgn",
            "noise_mode": "fixed_variance",
            "noise_variance": 0.25,
            "wireless_backend": "numpy",
            "data_plane_backend": "numpy",
        }

        def apply(count: int, item_id: str):
            symbols = np.ones((count,), dtype=np.complex64)
            return _apply_wireless_channel_per_item(
                symbols,
                {"source_item_ids": [item_id]},
                [count],
                "awgn",
                0.0,
                params,
                "numpy",
                "numpy",
                77,
            )

        short_rx, short_report, short_seeds = apply(3, "item-a")
        long_rx, long_report, long_seeds = apply(5, "item-a")
        _other_rx, _other_report, other_seeds = apply(3, "item-b")

        self.assertEqual(short_seeds, long_seeds)
        self.assertNotEqual(short_seeds, other_seeds)
        self.assertEqual(short_rx.shape, (3,))
        self.assertEqual(long_rx.shape, (5,))
        self.assertEqual(
            short_report["source_item_identity_keys"],
            ["item-a"],
        )
        self.assertEqual(
            long_report["source_item_identity_keys"],
            ["item-a"],
        )
        self.assertTrue(short_report["channel_realization_order_invariant"])
        self.assertTrue(long_report["channel_realization_order_invariant"])

    def test_active_use_power_retains_but_does_not_equalize_total_energy(
        self,
    ) -> None:
        declaration = {
            "power": {
                "coordinate": "average transmitted symbol power",
                "normalization_scope": "source_item",
                "target": 1.0,
            }
        }

        def steps(total_energy: float):
            return [
                {
                    "metadata": {
                        "power_normalization_target": 1.0,
                        "power_normalization_scope": "source_item",
                        "power_after": 1.0,
                        "power_unit": "normalized",
                        "source_item_power_after": [1.0],
                    }
                },
                {
                    "metadata": {
                        "tx_power_per_executed_use": 1.0,
                        "tx_total_energy": total_energy,
                        "power_unit": "normalized",
                    }
                },
            ]

        ten_uses = _power_evidence(steps(10.0), declaration)
        twenty_uses = _power_evidence(steps(20.0), declaration)

        self.assertTrue(ten_uses["complete"])
        self.assertTrue(twenty_uses["complete"])
        self.assertEqual(ten_uses["actual_coordinate_value"], 1.0)
        self.assertEqual(twenty_uses["actual_coordinate_value"], 1.0)
        self.assertEqual(ten_uses["tx_total_energy"], 10.0)
        self.assertEqual(twenty_uses["tx_total_energy"], 20.0)


if __name__ == "__main__":
    unittest.main()
