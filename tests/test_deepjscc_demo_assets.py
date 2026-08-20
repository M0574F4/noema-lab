from __future__ import annotations

import csv
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_deepjscc_demo_assets.py"
DATA_DIR = ROOT / "docs" / "demo" / "data" / "digital_vs_deepjscc"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "summary_table.csv"
QUALITY_PATH = DATA_DIR / "quality_summary_table.csv"
OPERATING_PATH = DATA_DIR / "jpeg_operating_points_table.csv"
CHART_DATA_PATH = DATA_DIR / "chart_data.json"
RESOURCE_PATH = DATA_DIR / "resource_audit.csv"
TUTORIAL_PATH = ROOT / "docs" / "tutorials" / "digital_vs_deepjscc_sionna.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "deepjscc_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DeepJSCC docs generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepJsccDemoAssetTests(unittest.TestCase):
    def test_snapshot_is_complete_and_resource_admissible(self):
        generator = _load_generator()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with PROJECTION_PATH.open("r", encoding="utf-8", newline="") as handle:
            projection = list(csv.DictReader(handle))
        with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
            summary = list(csv.DictReader(handle))
        with RESOURCE_PATH.open("r", encoding="utf-8", newline="") as handle:
            resources = list(csv.DictReader(handle))
        with QUALITY_PATH.open("r", encoding="utf-8", newline="") as handle:
            quality = list(csv.DictReader(handle))
        with OPERATING_PATH.open("r", encoding="utf-8", newline="") as handle:
            operating = list(csv.DictReader(handle))
        charts = json.loads(CHART_DATA_PATH.read_text(encoding="utf-8"))

        generator.verify_snapshot_assets(MANIFEST_PATH)
        self.assertEqual(manifest["source"]["result_id"], generator.DEFAULT_RESULT_ID)
        self.assertEqual(len(projection), 32)
        self.assertEqual(len(manifest["runs"]), 32)
        self.assertEqual(
            {row["method_id"] for row in projection},
            set(generator.EXPECTED_METHODS),
        )
        self.assertEqual(
            {float(row["snr_db"]) for row in projection},
            set(generator.EXPECTED_SNRS),
        )
        self.assertEqual(
            {row["channel_seed"] for row in projection},
            {"", *(str(seed) for seed in generator.EXPECTED_SEEDS)},
        )
        self.assertTrue(
            all(
                float(row["resource_guard_observed"])
                <= float(row["resource_guard_maximum"]) + 1e-9
                for row in projection
            )
        )
        self.assertEqual(generator._summary_rows(projection), summary)
        self.assertEqual(len(summary), 8)
        self.assertEqual(generator._quality_display_rows(summary), quality)
        self.assertEqual(generator._jpeg_operating_point_rows(summary), operating)
        self.assertEqual(len(resources), 2)
        self.assertTrue(all(row["Admitted"] == "yes" for row in resources))
        self.assertFalse(manifest["publication"]["publication_ready"])
        self.assertNotIn(
            "representative_reconstructions",
            manifest["projection"],
        )
        self.assertGreater(
            float(summary[0]["DeepJSCC PSNR (dB)"]),
            float(summary[0]["Capacity-matched JPEG PSNR (dB)"]),
        )
        self.assertLess(
            float(summary[-1]["DeepJSCC PSNR (dB)"]),
            float(summary[-1]["Capacity-matched JPEG PSNR (dB)"]),
        )
        self.assertEqual(
            set(charts),
            {
                "deepjscc-psnr",
                "deepjscc-ms-ssim",
            },
        )
        for chart in charts.values():
            self.assertTrue(
                all(series["color"].lower() != "#000000" for series in chart["series"])
            )
            self.assertTrue(chart["accessibleSummary"])
            self.assertEqual(
                {series["sampleCount"] for series in chart["series"]},
                {1, 3},
            )

    def test_tutorial_fails_closed_for_publication_and_fetches_its_dataset(self):
        tutorial = TUTORIAL_PATH.read_text(encoding="utf-8")
        configuration = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
        self.assertIn("noema data fetch kodak", tutorial)
        self.assertIn("Choose one workflow", tutorial)
        self.assertIn("Experimental tutorial evidence", tutorial)
        self.assertIn("not a paper-grade population claim", tutorial)
        self.assertIsNone(
            re.search(r"(?m)^\s*--allow-warnings(?:\s|$)", tutorial),
        )
        self.assertNotIn("reconstructions/source.png", tutorial)
        self.assertIn(
            'Path("data") / "digital_vs_deepjscc" / "reconstructions"',
            configuration,
        )
        self.assertIn(
            'Path("data") / "deepjscc_slow_rayleigh" / "reconstructions"',
            configuration,
        )


if __name__ == "__main__":
    unittest.main()
