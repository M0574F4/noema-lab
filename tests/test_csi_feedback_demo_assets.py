from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_csi_feedback_demo_assets.py"
DATA_DIR = ROOT / "docs" / "demo" / "data" / "csi_feedback"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "summary_table.csv"
CHART_DATA_PATH = DATA_DIR / "chart_data.json"
CHART_JS_PATH = ROOT / "docs" / "_static" / "noema-csi-feedback-chart-data.js"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "csi_feedback_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load CSI-feedback generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CsiFeedbackDemoAssetTests(unittest.TestCase):
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
        self.assertEqual(len(projection), 60)
        self.assertEqual(len(manifest["runs"]), 60)
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
        self.assertEqual(len(summary), 5)
        self.assertGreater(
            float(summary[0]["Learned rate (bit/s/Hz)"]),
            float(summary[0]["Truncated rate (bit/s/Hz)"]),
        )
        self.assertLess(
            abs(
                float(summary[0]["Learned rate (bit/s/Hz)"])
                - float(summary[0]["KLT rate (bit/s/Hz)"])
            ),
            1e-3,
        )
        self.assertLess(
            float(summary[0]["Learned rate (bit/s/Hz)"]),
            float(summary[0]["Perfect-CSIT rate (bit/s/Hz)"]),
        )
        self.assertEqual(
            set(charts),
            {
                "csi-feedback-rate",
                "csi-feedback-retention",
                "csi-feedback-nmse",
                "csi-feedback-response-preview",
                "csi-feedback-error-preview",
            },
        )
        self.assertEqual(
            len(charts["csi-feedback-response-preview"]["series"]),
            4,
        )
        self.assertEqual(
            len(charts["csi-feedback-error-preview"]["series"]),
            3,
        )
        for spec in charts.values():
            for series in spec["series"]:
                self.assertNotIn(
                    str(series["color"]).lower(),
                    {"black", "#000", "#000000"},
                )
        javascript = CHART_JS_PATH.read_text(encoding="utf-8")
        for chart_id in charts:
            self.assertIn('"%s"' % chart_id, javascript)

    def test_tutorial_binds_the_verified_result_and_benchmark_commands(self):
        source = (ROOT / "docs" / "tutorials" / "learned_csi_feedback.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(generator_result_id(), source)
        self.assertIn("build_benchmark.py", source)
        self.assertIn("noema benchmark run", source)
        self.assertIn("60 completed runs", source)


def generator_result_id() -> str:
    return _load_generator().DEFAULT_RESULT_ID


if __name__ == "__main__":
    unittest.main()
