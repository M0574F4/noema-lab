from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from noema_lab.core.executor import LocalExecutor
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.planner import RecipePlanningError, plan_recipe
from noema_lab.core.recipes import load_recipe, recipe_from_dict
from noema_lab.core.storage import LocalStore
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


def _step(summary, step_id):
    return next(step for step in summary["steps"] if step["id"] == step_id)


class SemanticAccountingKindTests(unittest.TestCase):
    def test_legacy_and_strict_contracts_have_distinct_identities(self):
        registry = build_registry()
        legacy_packetizer = registry.get("channel.packetize_crc32").describe()
        strict_packetizer = registry.get("channel.packetize_crc32.v2").describe()
        legacy_accounting = registry.get(
            "channel.communication_resource_accounting"
        ).describe()
        strict_accounting = registry.get(
            "channel.communication_resource_accounting.v2"
        ).describe()

        self.assertEqual(
            registry.get(
                "channel.communication_resource_accounting"
            ).accounting_profile,
            "noema.communication_resources.v1",
        )
        self.assertEqual(
            registry.get(
                "channel.communication_resource_accounting.v2"
            ).accounting_profile,
            "noema.communication_resources.v2",
        )
        self.assertEqual(
            legacy_packetizer["output_kinds"]["bits"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            strict_packetizer["output_kinds"]["bits"],
            "channel.framed_bits.numpy",
        )
        self.assertEqual(
            strict_packetizer["input_kinds"]["bits"],
            ["channel.payload_bits.numpy"],
        )
        self.assertEqual(
            legacy_accounting["input_kinds"]["payload_bits"],
            ["channel.payload_bits.numpy", "channel.bits.numpy"],
        )
        self.assertEqual(
            legacy_accounting["input_kinds"]["framed_bits"],
            ["channel.payload_bits.numpy", "channel.bits.numpy"],
        )
        self.assertEqual(
            legacy_accounting["input_kinds"]["coded_bits"],
            ["channel.coded_bits.numpy", "channel.bits.numpy"],
        )
        self.assertEqual(
            strict_accounting["input_kinds"],
            {
                "payload_bits": ["channel.payload_bits.numpy"],
                "framed_bits": ["channel.framed_bits.numpy"],
                "coded_bits": ["channel.coded_bits.numpy"],
                "symbols": ["channel.symbols.complex_numpy"],
            },
        )

    def test_versioned_packetizer_is_valid_in_layered_digital_lint(self):
        recipe = load_recipe(
            ROOT / "recipes" / "jpeg_q75_kodak_repetition_awgn.yaml"
        )
        packetizer = next(step for step in recipe.steps if step.id == "packetizer")
        packetizer.op = "channel.packetize_crc32.v2"
        packetizer.inputs["bits"] = "sender.bits"

        report = lint_recipe_invariants(recipe, build_registry(), strict=True)

        self.assertEqual(report["status"], "passed", report)

    def test_generic_bits_cannot_plan_into_strict_accounting(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "generic_bits_strict_accounting",
                "metadata": {"seed": 1},
                "steps": [
                    {
                        "id": "data",
                        "op": "source.random_bits",
                        "params": {"bit_count": 16},
                    },
                    {
                        "id": "generic",
                        "op": "channel.bit_boundary",
                        "inputs": {"bits": "data.bits"},
                    },
                    {
                        "id": "modulator",
                        "op": "modulation.digital_modulate",
                        "inputs": {"bits": "generic.bits"},
                        "params": {"modulation": "qpsk"},
                    },
                    {
                        "id": "accounting",
                        "op": "channel.communication_resource_accounting.v2",
                        "inputs": {
                            "payload_bits": "generic.bits",
                            "framed_bits": "generic.bits",
                            "coded_bits": "generic.bits",
                            "symbols": "modulator.symbols",
                        },
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            RecipePlanningError,
            "input 'payload_bits' expects one of.*channel.payload_bits.numpy",
        ):
            plan_recipe(recipe, build_registry())

    def test_generic_bits_cannot_plan_into_strict_packetizer(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "generic_bits_strict_packetizer",
                "metadata": {"seed": 1},
                "steps": [
                    {
                        "id": "data",
                        "op": "source.random_bits",
                        "params": {"bit_count": 16},
                    },
                    {
                        "id": "generic",
                        "op": "channel.bit_boundary",
                        "inputs": {"bits": "data.bits"},
                    },
                    {
                        "id": "packetizer",
                        "op": "channel.packetize_crc32.v2",
                        "inputs": {"bits": "generic.bits"},
                    },
                ],
            }
        )

        with self.assertRaisesRegex(
            RecipePlanningError,
            "input 'bits' expects one of.*channel.payload_bits.numpy",
        ):
            plan_recipe(recipe, build_registry())

    def test_valid_strict_chain_plans_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images.npz"
            images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
            images[0, 4:12, 4:12, :] = [20, 160, 230]
            images[1, 2:14, 6:10, :] = [210, 80, 25]
            np.savez_compressed(image_path, images=images)
            recipe = recipe_from_dict(
                {
                    "schema_version": 1,
                    "name": "strict_resource_accounting",
                    "metadata": {"seed": 7},
                    "steps": [
                        {
                            "id": "data",
                            "op": "source.local_npz_images",
                            "params": {
                                "path": str(image_path),
                                "array": "images",
                            },
                        },
                        {
                            "id": "sender",
                            "op": "model.jpeg_encode",
                            "inputs": {"images": "data.images"},
                            "params": {"quality": 75},
                        },
                        {
                            "id": "packetizer",
                            "op": "channel.packetize_crc32.v2",
                            "inputs": {"bits": "sender.bits"},
                            "params": {"packet_payload_bits": 512},
                        },
                        {
                            "id": "channel_encoder",
                            "op": "channel.repetition_encoder",
                            "inputs": {"bits": "packetizer.bits"},
                            "params": {"factor": 3},
                        },
                        {
                            "id": "modulator",
                            "op": "modulation.digital_modulate",
                            "inputs": {
                                "bits": "channel_encoder.coded_bits"
                            },
                            "params": {"modulation": "qpsk"},
                        },
                        {
                            "id": "accounting",
                            "op": "channel.communication_resource_accounting.v2",
                            "inputs": {
                                "payload_bits": "sender.bits",
                                "framed_bits": "packetizer.bits",
                                "coded_bits": "channel_encoder.coded_bits",
                                "symbols": "modulator.symbols",
                            },
                        },
                    ],
                }
            )
            registry = build_registry()
            plan_recipe(recipe, registry)
            store = LocalStore(root / "workspace")
            run_dir = LocalExecutor(registry, store).run(recipe)
            summary = store.get_run(run_dir.name)

        self.assertEqual(
            _step(summary, "sender")["outputs"]["bits"]["kind"],
            "channel.payload_bits.numpy",
        )
        self.assertEqual(
            _step(summary, "packetizer")["outputs"]["bits"]["kind"],
            "channel.framed_bits.numpy",
        )
        self.assertEqual(
            _step(summary, "channel_encoder")["outputs"]["coded_bits"]["kind"],
            "channel.coded_bits.numpy",
        )
        report = _step(summary, "accounting")["outputs"]["report"]["metadata"]
        self.assertEqual(
            report["accounting_profile"],
            "noema.communication_resources.v2",
        )
        self.assertLess(
            report["serialized_payload_bit_count"],
            report["framed_bit_count"],
        )
        self.assertLess(
            report["framed_bit_count"],
            report["coded_bit_count"],
        )


if __name__ == "__main__":
    unittest.main()
