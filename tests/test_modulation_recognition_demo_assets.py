from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_modulation_recognition_demo_assets.py"
DATA_DIR = ROOT / "docs" / "demo" / "data" / "modulation_recognition"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "summary_table.csv"
CHART_DATA_PATH = DATA_DIR / "chart_data.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "modulation_recognition_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load modulation-recognition generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModulationRecognitionDemoAssetTests(unittest.TestCase):
    def test_snapshot_is_complete_paired_and_reproducible(self):
        generator = _load_generator()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with PROJECTION_PATH.open("r", encoding="utf-8", newline="") as handle:
            projection = list(csv.DictReader(handle))
        with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
            summary = list(csv.DictReader(handle))
        charts = json.loads(CHART_DATA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["source"]["result_id"],
            generator.DEFAULT_RESULT_ID,
        )
        self.assertEqual(len(projection), 54)
        self.assertEqual(len(manifest["runs"]), 54)
        self.assertEqual(
            {row["method_id"] for row in projection},
            set(generator.EXPECTED_METHODS),
        )
        self.assertEqual(
            {float(row["snr_db"]) for row in projection},
            set(generator.EXPECTED_SNRS),
        )
        self.assertEqual(
            {int(row["paired_seed"]) for row in projection},
            set(generator.EXPECTED_SEEDS),
        )
        self.assertEqual(generator._summary_rows(projection), summary)
        self.assertEqual(len(summary), 6)
        self.assertGreater(
            float(summary[0]["Learned accuracy"]),
            float(summary[0]["Blind accuracy"]),
        )
        self.assertLess(
            float(summary[0]["Learned accuracy"]),
            float(summary[0]["Oracle accuracy"]),
        )
        self.assertGreaterEqual(float(summary[-2]["Learned accuracy"]), 0.99)
        self.assertGreaterEqual(float(summary[-1]["Learned accuracy"]), 0.99)
        self.assertEqual(
            set(charts),
            {
                "modulation-recognition-accuracy",
                "modulation-recognition-macro-f1",
            },
        )
        for chart in charts.values():
            self.assertEqual(len(chart["series"]), 3)
            self.assertTrue(
                all(series["color"].lower() != "#000000" for series in chart["series"])
            )


if __name__ == "__main__":
    unittest.main()
