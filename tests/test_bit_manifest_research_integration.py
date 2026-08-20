from __future__ import annotations

import unittest

from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import load_research_catalog


class BitManifestResearchIntegrationTests(unittest.TestCase):
    def test_delayed_csi_dataset_accepts_frozen_bit_manifest_source(self) -> None:
        dataset = load_research_catalog().dataset(
            "synthetic_random_bits_sionna_tdl_delayed_csi"
        )

        self.assertIsNotNone(dataset)
        self.assertIn("source.random_bits", dataset.source_ops)
        self.assertIn("source.bit_manifest", dataset.source_ops)

    def test_explicit_current_profile_identity_survives_research_resolution(self) -> None:
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "frozen_delayed_csi_payload_contract",
                "metadata": {
                    "research": {
                        "dataset": {
                            "id": "synthetic_random_bits_sionna_tdl_delayed_csi",
                            "modality": "wireless",
                            "version": (
                                "synthetic-random-bits-sionna-tdl-delayed-csi-v2"
                            ),
                            "split": "development_phy_pilot",
                            "source": "source.bit_manifest",
                        },
                        "task": {"id": "resource_allocation"},
                    }
                },
                "steps": [
                    {
                        "id": "payloads",
                        "op": "source.bit_manifest",
                        "params": {
                            "manifest_path": "payloads.json",
                            "manifest_sha256": "a" * 64,
                            "selection": "development_phy_pilot",
                        },
                    }
                ],
            }
        )

        specs = research_specs_from_recipe(recipe)

        self.assertEqual(specs["catalog_validation"]["status"], "valid")
        self.assertEqual(
            specs["dataset"]["id"],
            "synthetic_random_bits_sionna_tdl_delayed_csi",
        )
        self.assertEqual(specs["dataset"]["source"], "source.bit_manifest")
        self.assertEqual(specs["dataset"]["split"], "development_phy_pilot")
        self.assertEqual(specs["task"]["id"], "resource_allocation")


if __name__ == "__main__":
    unittest.main()
