from __future__ import annotations

import unittest
from pathlib import Path

from noema_lab.core.benchmarks import (
    BenchmarkError,
    BenchmarkRecipe,
    _apply_benchmark_recipe_params,
)
from noema_lab.core.matrix import matrix_variant_id


class BenchmarkMatrixSelectionTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict:
        return {
            "schema_version": 1,
            "name": "base",
            "metadata": {
                "preserved": "value",
                "sweep_values": {"stale.coordinate": -1},
            },
            "steps": [],
        }

    @staticmethod
    def _entry(params: dict) -> BenchmarkRecipe:
        return BenchmarkRecipe(
            id="point",
            path=Path("recipes/base.yaml"),
            params=params,
        )

    @staticmethod
    def _matrix_payload() -> dict:
        return {
            "schema_version": 1,
            "name": "matrix_source",
            "metadata": {
                "matrix": {
                    "dimensions": {
                        "snr": [0, 8],
                        "seed": [1, 2],
                    },
                    "step_params": {
                        "channel": {
                            "snr_db": {"matrix": "snr"},
                            "seed": {"matrix": "seed"},
                        }
                    },
                },
                "sweeps": {"channel.snr_db": "100,200"},
                "preserved": "value",
            },
            "steps": [
                {
                    "id": "channel",
                    "op": "wireless.channel",
                    "params": {"snr_db": -1, "seed": 0},
                    "inputs": {},
                }
            ],
        }

    def test_canonical_matrix_selection_is_the_only_concrete_coordinate_field(self):
        coordinates = {"channel.snr_db": 8, "source.quality": 75}
        payload = self._payload()

        _apply_benchmark_recipe_params(
            payload,
            self._entry({"matrix_selection": coordinates}),
        )

        self.assertEqual(payload["metadata"]["matrix_selection"], coordinates)
        self.assertEqual(
            payload["metadata"]["matrix_variant_id"],
            matrix_variant_id(coordinates),
        )
        self.assertNotIn("sweep_values", payload["metadata"])
        self.assertEqual(payload["metadata"]["preserved"], "value")
        coordinates["channel.snr_db"] = 99
        self.assertEqual(payload["metadata"]["matrix_selection"]["channel.snr_db"], 8)

    def test_legacy_sweep_values_are_read_and_normalized_to_matrix_selection(self):
        payload = self._payload()

        _apply_benchmark_recipe_params(
            payload,
            self._entry({"sweep_values": {"channel.snr_db": 6}}),
        )

        self.assertEqual(
            payload["metadata"]["matrix_selection"],
            {"channel.snr_db": 6},
        )
        self.assertEqual(
            payload["metadata"]["matrix_variant_id"],
            matrix_variant_id({"channel.snr_db": 6}),
        )
        self.assertNotIn("sweep_values", payload["metadata"])

    def test_matching_canonical_and_legacy_coordinates_are_accepted(self):
        payload = self._payload()
        coordinates = {"channel.snr_db": 14}

        _apply_benchmark_recipe_params(
            payload,
            self._entry(
                {
                    "matrix_selection": coordinates,
                    "sweep_values": dict(coordinates),
                }
            ),
        )

        self.assertEqual(payload["metadata"]["matrix_selection"], coordinates)
        self.assertNotIn("sweep_values", payload["metadata"])

    def test_conflicting_canonical_and_legacy_coordinates_are_rejected(self):
        payload = self._payload()

        with self.assertRaisesRegex(
            BenchmarkError,
            "conflicting params.matrix_selection and deprecated params.sweep_values",
        ):
            _apply_benchmark_recipe_params(
                payload,
                self._entry(
                    {
                        "matrix_selection": {"channel.snr_db": 8},
                        "sweep_values": {"channel.snr_db": 12},
                    }
                ),
            )

        self.assertEqual(payload["name"], "base")
        self.assertNotIn("matrix_selection", payload["metadata"])

    def test_coordinate_comparison_preserves_json_value_types(self):
        with self.assertRaisesRegex(BenchmarkError, "conflicting"):
            _apply_benchmark_recipe_params(
                self._payload(),
                self._entry(
                    {
                        "matrix_selection": {"enabled": True},
                        "sweep_values": {"enabled": 1},
                    }
                ),
            )

    def test_coordinate_fields_must_be_mappings(self):
        for field in ("matrix_selection", "sweep_values"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    BenchmarkError,
                    "params.%s must be a mapping" % field,
                ):
                    _apply_benchmark_recipe_params(
                        self._payload(),
                        self._entry({field: ["channel.snr_db", 8]}),
                    )

    def test_source_matrix_bindings_are_materialized_and_extra_coordinates_survive(self):
        payload = self._matrix_payload()

        _apply_benchmark_recipe_params(
            payload,
            self._entry(
                {
                    "matrix_selection": {
                        "snr": 8,
                        "seed": 2,
                        "benchmark.fold": "validation",
                    }
                }
            ),
        )

        self.assertEqual(payload["steps"][0]["params"]["snr_db"], 8)
        self.assertEqual(payload["steps"][0]["params"]["seed"], 2)
        self.assertEqual(
            payload["metadata"]["matrix_selection"],
            {"snr": 8, "seed": 2, "benchmark.fold": "validation"},
        )
        self.assertEqual(payload["metadata"]["matrix_index"], 3)
        self.assertEqual(payload["metadata"]["preserved"], "value")
        for field in ("matrix", "sweeps", "ui_sweeps", "sweep_values"):
            self.assertNotIn(field, payload["metadata"])

    def test_source_matrix_requires_every_known_coordinate_and_an_authored_value(self):
        cases = [
            ({"snr": 8}, r"missing dimension\(s\): seed"),
            ({"snr": 99, "seed": 2}, "snr is not in the authored dimension"),
        ]
        for selection, expected in cases:
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(BenchmarkError, expected):
                    _apply_benchmark_recipe_params(
                        self._matrix_payload(),
                        self._entry({"matrix_selection": selection}),
                    )

    def test_step_param_override_must_match_materialized_matrix_binding(self):
        with self.assertRaisesRegex(
            BenchmarkError,
            r"step_params\.channel\.snr_db conflicts with params\.matrix_selection",
        ):
            _apply_benchmark_recipe_params(
                self._matrix_payload(),
                self._entry(
                    {
                        "matrix_selection": {"snr": 8, "seed": 2},
                        "step_params": {"channel": {"snr_db": 0}},
                    }
                ),
            )

        payload = self._matrix_payload()
        _apply_benchmark_recipe_params(
            payload,
            self._entry(
                {
                    "matrix_selection": {"snr": 8, "seed": 2},
                    "step_params": {
                        "channel": {"snr_db": 8, "unbound": "kept"}
                    },
                }
            ),
        )
        self.assertEqual(payload["steps"][0]["params"]["snr_db"], 8)
        self.assertEqual(payload["steps"][0]["params"]["unbound"], "kept")


if __name__ == "__main__":
    unittest.main()
