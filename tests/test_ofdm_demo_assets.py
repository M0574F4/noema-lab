from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_ofdm_resource_allocation_demo_assets.py"
SOURCE_CSV = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "ofdm_resource_allocation"
    / "current_results_all_metrics.csv"
)
SNAPSHOT_MANIFEST = SOURCE_CSV.parent / "snapshot_manifest.json"
SOURCE_SHA256 = "e37f400827d9bc2a53b50e3090a1f2dfd3c99cdc5834b9947ee5f666bd008611"
COMMITTED_SVG_MATPLOTLIB_VERSION = "3.11.0"


def _load_generator():
    spec = importlib.util.spec_from_file_location("ofdm_demo_assets", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load OFDM demo asset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfdmDemoAssetTests(unittest.TestCase):
    def test_snapshot_projection_is_complete_and_deterministic(self):
        generator = _load_generator()
        self.assertEqual(
            hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest(), SOURCE_SHA256
        )
        manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["sha256"], SOURCE_SHA256)
        self.assertEqual(manifest["source"]["data_row_count"], 2505)
        records = generator.load_records(SOURCE_CSV)
        self.assertEqual(len(manifest["runs"]), len(records))
        self.assertEqual(
            {run["run_id"] for run in manifest["runs"]},
            {record["run_id"] for record in records},
        )
        for run in manifest["runs"]:
            self.assertRegex(run["authored_recipe_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(run["effective_recipe_sha256"], r"^[0-9a-f]{64}$")

        self.assertEqual(len(records), 15)
        self.assertEqual(
            {record["policy"] for record in records},
            {"equal_power", "learned_allocator", "water_filling"},
        )
        self.assertEqual(
            {record["average_transmit_power_budget"] for record in records},
            {0.5, 1.0, 2.0, 3.0, 4.0},
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = generator.generate(records, root / "first-data", root / "first-assets")
            second = generator.generate(
                records, root / "second-data", root / "second-assets"
            )
            self.assertEqual(set(first), set(second))
            for name in first:
                self.assertEqual(
                    first[name].read_bytes(),
                    second[name].read_bytes(),
                    name,
                )
                committed = generator._output_paths(
                    generator.DEFAULT_DATA_DIR, generator.DEFAULT_ASSET_DIR
                )[name]
                if (
                    first[name].suffix.lower() != ".svg"
                    or matplotlib.__version__ == COMMITTED_SVG_MATPLOTLIB_VERSION
                ):
                    self.assertEqual(
                        first[name].read_bytes(),
                        committed.read_bytes(),
                        name,
                    )

            with first["spectral_efficiency_table"].open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 5)
            self.assertEqual(table[0]["Nominal TX SNR (dB)"], "3.98")
            self.assertEqual(table[-1]["Nominal TX SNR (dB)"], "13.01")
            self.assertEqual(
                table[-1]["Learned (bit/s/Hz)"],
                "4.097255",
            )
            self.assertAlmostEqual(
                float(
                    records[-2]["shannon_optimum_achievement_percent"]
                ),
                99.999989467258,
                places=9,
            )

            for name in (
                "spectral_efficiency_plot",
                "shannon_optimum_achievement_plot",
            ):
                svg = first[name].read_text(encoding="utf-8")
                self.assertNotIn("<dc:date>", svg)
                self.assertIn("Noema OFDM demo asset generator", svg)


if __name__ == "__main__":
    unittest.main()
