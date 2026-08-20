from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.recipes import (
    RecipeValidationError,
    compile_recipe,
    load_recipe,
    recipe_from_dict,
)
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


def _recipe_payload():
    return {
        "schema_version": 1,
        "name": "compiler_test",
        "steps": [
            {
                "id": "source",
                "op": "test.source",
                "params": {},
            }
        ],
    }


class _FakeOperation:
    params_schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "default": 4},
            "nested": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "default": "default-label"},
                    },
                },
            },
        },
        "additionalProperties": False,
    }


class _FakeRegistry:
    def get(self, operation_id):
        if operation_id != "test.source":
            raise KeyError(operation_id)
        return _FakeOperation()


class RecipeCompilerTests(unittest.TestCase):
    def test_schema_version_is_exact(self):
        for value in (0, 2, False, None, "1", 1.0):
            with self.subTest(value=value):
                payload = _recipe_payload()
                payload["schema_version"] = value
                with self.assertRaises(RecipeValidationError):
                    recipe_from_dict(payload)

        payload = _recipe_payload()
        payload.pop("schema_version")
        self.assertEqual(recipe_from_dict(payload).schema_version, 1)

    def test_compat_reports_and_preserves_unknown_fields(self):
        payload = _recipe_payload()
        payload["future_recipe_field"] = {"enabled": True}
        payload["steps"][0]["future_step_field"] = ["retained"]

        compilation = compile_recipe(payload, mode="compat")

        self.assertTrue(compilation.is_valid)
        self.assertEqual(
            {item.code for item in compilation.warnings},
            {"unknown_recipe_field", "unknown_recipe_step_field"},
        )
        normalized = compilation.require_recipe().to_dict()
        self.assertEqual(normalized["future_recipe_field"], {"enabled": True})
        self.assertEqual(normalized["steps"][0]["future_step_field"], ["retained"])

        legacy_recipe = recipe_from_dict(payload)
        self.assertEqual(len(legacy_recipe.diagnostics), 2)
        self.assertEqual(legacy_recipe.to_dict()["future_recipe_field"], {"enabled": True})

    def test_strict_rejects_unknown_fields_without_discarding_them(self):
        payload = _recipe_payload()
        payload["misspelled_metadata"] = {"seed": 7}
        payload["steps"][0]["parameter"] = {"count": 2}

        compilation = compile_recipe(payload, mode="strict")

        self.assertFalse(compilation.is_valid)
        self.assertEqual(
            {item.code for item in compilation.errors},
            {"unknown_recipe_field", "unknown_recipe_step_field"},
        )
        self.assertIsNotNone(compilation.recipe)
        normalized = compilation.recipe.to_dict()
        self.assertEqual(normalized["misspelled_metadata"], {"seed": 7})
        self.assertEqual(normalized["steps"][0]["parameter"], {"count": 2})
        with self.assertRaises(RecipeValidationError):
            compilation.require_recipe()

    def test_file_loader_can_enforce_strict_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_text(
                "schema_version: 1\n"
                "name: strict_file\n"
                "metdata: {}\n"
                "steps:\n"
                "  - id: source\n"
                "    op: test.source\n",
                encoding="utf-8",
            )

            compat = load_recipe(path)
            self.assertEqual(compat.to_dict()["metdata"], {})
            with self.assertRaisesRegex(
                RecipeValidationError,
                "Unknown recipe field `metdata`",
            ):
                load_recipe(path, mode="strict")

    def test_implicit_step_ids_warn_in_compat_and_fail_in_strict(self):
        payload = _recipe_payload()
        payload["steps"][0].pop("id")

        compat = compile_recipe(payload, mode="compat")
        self.assertTrue(compat.is_valid)
        self.assertEqual(compat.require_recipe().steps[0].id, "step_1")
        self.assertIn(
            "implicit_recipe_step_id",
            {item.code for item in compat.warnings},
        )

        strict = compile_recipe(payload, mode="strict")
        self.assertFalse(strict.is_valid)
        self.assertIsNotNone(strict.recipe)
        self.assertEqual(strict.recipe.steps[0].id, "step_1")
        self.assertIn(
            "implicit_recipe_step_id",
            {item.code for item in strict.errors},
        )

    def test_effective_recipe_materializes_schema_defaults_without_mutating_source(self):
        payload = _recipe_payload()
        payload["steps"][0]["params"] = {
            "nested": {},
            "items": [{}, {"label": "explicit"}],
        }
        original = copy.deepcopy(payload)

        compilation = compile_recipe(
            payload,
            mode="strict",
            registry=_FakeRegistry(),
        )

        self.assertTrue(compilation.is_valid)
        self.assertTrue(compilation.defaults_materialized)
        self.assertEqual(payload, original)
        self.assertEqual(
            compilation.require_recipe().steps[0].params,
            {"nested": {}, "items": [{}, {"label": "explicit"}]},
        )
        self.assertEqual(
            compilation.require_recipe(effective=True).steps[0].params,
            {
                "count": 4,
                "nested": {"enabled": True},
                "items": [
                    {"label": "default-label"},
                    {"label": "explicit"},
                ],
            },
        )

    def test_ofdm_channel_is_single_authority_for_csi_noise_and_scenario(self):
        payload = {
            "schema_version": 1,
            "name": "ofdm_authority",
            "steps": [
                {
                    "id": "modulator",
                    "op": "channel.symbol_power_identity",
                    "inputs": {},
                    "params": {},
                },
                {
                    "id": "channel_state",
                    "op": "wireless.ofdm_channel_state",
                    "inputs": {"symbols": "modulator.symbols"},
                    "params": {
                        "tdl_model": "A",
                        "delay_spread_ns": 300,
                        "noise_variance": 0.2,
                    },
                },
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "inputs": {
                        "symbols": "channel_state.symbols",
                        "channel_state": "channel_state.state",
                    },
                    "params": {
                        "channel": "ofdm_tdl",
                        "noise_mode": "snr_at_unit_power",
                        "snr_db": 20,
                    },
                },
            ],
        }

        compilation = compile_recipe(payload, registry=build_registry())
        authored = compilation.require_recipe()
        effective = compilation.require_recipe(effective=True)
        authored_state = next(step for step in authored.steps if step.id == "channel_state")
        effective_state = next(step for step in effective.steps if step.id == "channel_state")
        effective_channel = next(step for step in effective.steps if step.id == "wireless_channel")

        self.assertEqual(authored_state.params["noise_variance"], 0.2)
        self.assertAlmostEqual(effective_state.params["noise_variance"], 0.01)
        self.assertEqual(effective_channel.params["delay_spread_ns"], 300)
        self.assertEqual(effective_state.params["delay_spread_ns"], 300)
        self.assertEqual(effective.metadata["resolved_channel_scenario"]["authority"], "wireless_channel")
        self.assertAlmostEqual(
            effective.metadata["resolved_channel_scenario"]["noise_variance"],
            0.01,
        )

    def test_conditional_defaults_do_not_materialize_inactive_noise_controls(self):
        payload = {
            "schema_version": 1,
            "name": "fixed_noise_defaults",
            "steps": [
                {
                    "id": "wireless_channel",
                    "op": "wireless.channel",
                    "inputs": {},
                    "params": {
                        "channel": "awgn",
                        "noise_mode": "fixed_variance",
                        "noise_variance": 0.2,
                    },
                }
            ],
        }

        effective = compile_recipe(
            payload,
            registry=build_registry(),
        ).require_recipe(effective=True)
        params = effective.steps[0].params
        self.assertEqual(params["noise_variance"], 0.2)
        self.assertNotIn("snr_db", params)

    def test_executor_records_authored_and_runs_default_expanded_recipe(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "effective_execution",
                "steps": [
                    {
                        "id": "source",
                        "op": "source.random_bits",
                        "params": {"bit_count": 8},
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(
                build_registry(),
                LocalStore(Path(tmp)),
            ).run(recipe)
            authored = json.loads(
                (run_dir / "recipe.authored.json").read_text(encoding="utf-8")
            )
            effective = json.loads(
                (run_dir / "recipe.json").read_text(encoding="utf-8")
            )

        self.assertEqual(recipe.steps[0].params, {"bit_count": 8})
        self.assertEqual(authored["steps"][0]["params"], {"bit_count": 8})
        self.assertEqual(effective["steps"][0]["params"]["bit_count"], 8)
        self.assertEqual(effective["steps"][0]["params"]["batch_size"], 1)
        self.assertEqual(effective["steps"][0]["params"]["seed"], 0)


if __name__ == "__main__":
    unittest.main()
