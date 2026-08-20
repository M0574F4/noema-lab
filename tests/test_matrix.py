import copy
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noema_lab.core.matrix import (
    MAX_MATRIX_VARIANTS,
    RecipeMatrixError,
    canonicalize_recipe_matrix,
    expand_recipe_matrix,
    matrix_values_for_step_param,
)
from noema_lab.core.recipes import RecipeValidationError, recipe_from_dict
from noema_lab.training.exporter import DifferentiableExportError, _snr_values


def _recipe(metadata=None):
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "matrix_contract",
            "metadata": copy.deepcopy(metadata or {}),
            "steps": [
                {
                    "id": "sender",
                    "op": "model.jpeg_encode",
                    "params": {"quality": 75},
                },
                {
                    "id": "tx_power",
                    "op": "channel.symbol_power_normalize",
                    "params": {"target_power": 1.0},
                },
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "params": {"snr_db": 12.0, "seed": 0},
                },
            ],
        }
    )


class CanonicalMatrixTests(unittest.TestCase):
    def test_multidimensional_expansion_is_deterministic_recursive_and_non_mutating(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {
                        "snr": [0.0, 10.0],
                        "seed": [2, 1],
                    },
                    "step_params": {
                        "wireless_channel": {
                            "snr_db": {"matrix": "snr"},
                            "nested": {
                                "seed": {"matrix": "seed"},
                                "rows": [{"matrix": "snr"}],
                            },
                        }
                    },
                },
                "untouched": {"owner": "author"},
            }
        )
        before = recipe.to_dict()

        expanded = expand_recipe_matrix(recipe)

        self.assertEqual(expanded["expanded_count"], 4)
        self.assertEqual(
            [row["metadata"]["matrix_selection"] for row in expanded["recipes"]],
            [
                {"seed": 2, "snr": 0.0},
                {"seed": 2, "snr": 10.0},
                {"seed": 1, "snr": 0.0},
                {"seed": 1, "snr": 10.0},
            ],
        )
        for index, row in enumerate(expanded["recipes"]):
            self.assertEqual(row["metadata"]["matrix_index"], index)
            self.assertNotIn("matrix", row["metadata"])
            self.assertEqual(row["metadata"]["untouched"], {"owner": "author"})
            params = next(
                step["params"]
                for step in row["steps"]
                if step["id"] == "wireless_channel"
            )
            self.assertEqual(params["snr_db"], row["metadata"]["matrix_selection"]["snr"])
            self.assertEqual(params["nested"]["seed"], row["metadata"]["matrix_selection"]["seed"])
            self.assertEqual(params["nested"]["rows"], [row["metadata"]["matrix_selection"]["snr"]])
        self.assertEqual(recipe.to_dict(), before)

    def test_canonical_validation_rejects_non_list_dimension(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": "0:5:10"},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, "non-empty typed list"):
            canonicalize_recipe_matrix(recipe)

    def test_canonical_validation_rejects_unknown_step(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [0, 5]},
                    "step_params": {
                        "missing": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, "unknown step missing"):
            canonicalize_recipe_matrix(recipe)

    def test_canonical_validation_rejects_unknown_dimension(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [0, 5]},
                    "step_params": {
                        "wireless_channel": {
                            "snr_db": {"matrix": "not_snr"}
                        }
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, "unknown matrix dimension not_snr"):
            canonicalize_recipe_matrix(recipe)

    def test_canonical_validation_rejects_unused_dimension(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [0, 5], "seed": [1, 2]},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, r"unused dimension\(s\): seed"):
            canonicalize_recipe_matrix(recipe)

    def test_canonical_validation_rejects_invalid_marker_with_siblings(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [0, 5]},
                    "step_params": {
                        "wireless_channel": {
                            "snr_db": {"matrix": "snr", "unit": "dB"}
                        }
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, "cannot have sibling fields"):
            canonicalize_recipe_matrix(recipe)

    def test_canonical_validation_limits_cartesian_product(self):
        allowed = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": list(range(MAX_MATRIX_VARIANTS))},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        self.assertEqual(
            canonicalize_recipe_matrix(allowed).variant_count,
            MAX_MATRIX_VARIANTS,
        )

        rejected = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": list(range(MAX_MATRIX_VARIANTS + 1))},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        with self.assertRaisesRegex(RecipeMatrixError, "maximum is 256"):
            canonicalize_recipe_matrix(rejected)

    def test_canonical_validation_rejects_non_finite_values(self):
        with self.assertRaisesRegex(
            RecipeValidationError,
            "NaN or infinity|finite",
        ):
            _recipe(
                {
                    "matrix": {
                        "dimensions": {"snr": [0, math.inf]},
                        "step_params": {
                            "wireless_channel": {"snr_db": {"matrix": "snr"}}
                        },
                    }
                }
            )


class LegacySweepCompatibilityTests(unittest.TestCase):
    def test_legacy_sweeps_are_typed_and_bound_to_real_steps(self):
        recipe = _recipe(
            {
                "sweeps": {
                    "channel.snr_db": "-4:2:0",
                    "codecParams.encoder.quality": "50,75",
                    "channel.txPowerTarget": "0.5,1",
                }
            }
        )

        compiled = canonicalize_recipe_matrix(recipe)

        self.assertEqual(compiled.source, "sweeps")
        self.assertEqual(
            compiled.definition["dimensions"],
            {
                "channel.snr_db": [-4, -2, 0],
                "codecParams.encoder.quality": [50, 75],
                "channel.txPowerTarget": [0.5, 1],
            },
        )
        self.assertEqual(
            compiled.definition["step_params"]["wireless_channel"]["snr_db"],
            {"matrix": "channel.snr_db"},
        )
        self.assertEqual(
            compiled.definition["step_params"]["sender"]["quality"],
            {"matrix": "codecParams.encoder.quality"},
        )
        self.assertEqual(
            compiled.definition["step_params"]["tx_power"]["target_power"],
            {"matrix": "channel.txPowerTarget"},
        )
        self.assertEqual(compiled.diagnostics[-1].code, "legacy_sweep_normalized")

    def test_sweeps_precede_ui_sweeps_with_diagnostic(self):
        recipe = _recipe(
            {
                "sweeps": {"wireless_channel.snr_db": "1,2"},
                "ui_sweeps": {"wireless_channel.snr_db": "8,9"},
            }
        )
        compiled = canonicalize_recipe_matrix(recipe)
        self.assertEqual(
            compiled.definition["dimensions"]["wireless_channel.snr_db"],
            [1, 2],
        )
        self.assertIn(
            "legacy_sweep_precedence",
            [diagnostic.code for diagnostic in compiled.diagnostics],
        )

    def test_canonical_matrix_precedes_legacy_aliases(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [3, 4]},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                },
                "sweeps": {"channel.snr_db": "9,10"},
            }
        )
        compiled = canonicalize_recipe_matrix(recipe)
        self.assertEqual(compiled.source, "matrix")
        self.assertEqual(compiled.definition["dimensions"], {"snr": [3, 4]})
        self.assertEqual(compiled.diagnostics[0].code, "legacy_sweep_ignored")
        with self.assertRaisesRegex(RecipeMatrixError, "cannot accompany metadata.matrix"):
            canonicalize_recipe_matrix(recipe, strict_legacy=True)

    def test_strict_mode_rejects_legacy_alias(self):
        recipe = _recipe({"ui_sweeps": {"wireless_channel.snr_db": "1,2"}})
        with self.assertRaisesRegex(RecipeMatrixError, "compatibility field"):
            canonicalize_recipe_matrix(recipe, strict_legacy=True)

    def test_legacy_expansion_removes_definition_aliases(self):
        recipe = _recipe(
            {
                "sweeps": {"channel.snr_db": "0,5"},
                "ui_sweeps": {"channel.snr_db": "8,9"},
            }
        )
        before = recipe.to_dict()

        rows = expand_recipe_matrix(recipe)["recipes"]

        self.assertEqual(len(rows), 2)
        for index, row in enumerate(rows):
            self.assertNotIn("matrix", row["metadata"])
            self.assertNotIn("sweeps", row["metadata"])
            self.assertNotIn("ui_sweeps", row["metadata"])
            self.assertEqual(row["metadata"]["matrix_index"], index)
            channel = next(step for step in row["steps"] if step["id"] == "wireless_channel")
            self.assertEqual(channel["params"]["snr_db"], [0, 5][index])
        self.assertEqual(recipe.to_dict(), before)

    def test_legacy_unknown_step_is_an_error(self):
        recipe = _recipe({"sweeps": {"missing.snr_db": "0,5"}})
        with self.assertRaisesRegex(RecipeMatrixError, "unknown step missing"):
            canonicalize_recipe_matrix(recipe)


class MatrixConsumerTests(unittest.TestCase):
    def test_step_parameter_lookup_follows_canonical_binding(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [-2, 4]},
                    "step_params": {
                        "wireless_channel": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        self.assertEqual(
            matrix_values_for_step_param(recipe, "wireless_channel", "snr_db"),
            [-2, 4],
        )
        self.assertIsNone(
            matrix_values_for_step_param(recipe, "wireless_channel", "seed")
        )

    def test_differentiable_exporter_reads_canonical_snr_binding(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"train_snr": [2, 6, 10]},
                    "step_params": {
                        "wireless_channel": {
                            "snr_db": {"matrix": "train_snr"}
                        }
                    },
                }
            }
        )
        channel = next(step for step in recipe.steps if step.id == "wireless_channel")
        self.assertEqual(_snr_values(recipe, channel), [2.0, 6.0, 10.0])

    def test_differentiable_exporter_keeps_legacy_snr_compatibility(self):
        recipe = _recipe({"sweeps": {"channel.snr_db": "8:2:12"}})
        channel = next(step for step in recipe.steps if step.id == "wireless_channel")
        self.assertEqual(_snr_values(recipe, channel), [8.0, 10.0, 12.0])

    def test_differentiable_exporter_reports_invalid_matrix(self):
        recipe = _recipe(
            {
                "matrix": {
                    "dimensions": {"snr": [2, 6]},
                    "step_params": {
                        "missing": {"snr_db": {"matrix": "snr"}}
                    },
                }
            }
        )
        channel = next(step for step in recipe.steps if step.id == "wireless_channel")
        with self.assertRaisesRegex(DifferentiableExportError, "Invalid recipe SNR matrix"):
            _snr_values(recipe, channel)


if __name__ == "__main__":
    unittest.main()
