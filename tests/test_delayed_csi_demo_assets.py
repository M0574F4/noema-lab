from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_delayed_csi_ofdm_demo_assets.py"
DATA_DIR = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "reliability_aware_delayed_csi_ofdm_allocation"
)
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "summary_table.csv"
CHART_DATA_PATH = DATA_DIR / "chart_data.json"
CHART_JS_PATH = (
    ROOT / "docs" / "_static" / "noema-delayed-csi-demo-chart-data.js"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "delayed_csi_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load delayed-CSI demo asset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DelayedCsiDemoAssetTests(unittest.TestCase):
    def test_snapshot_is_complete_objective_aligned_and_reproducible(self):
        generator = _load_generator()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with PROJECTION_PATH.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            projection = list(csv.DictReader(handle))
        with SUMMARY_PATH.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            summary = list(csv.DictReader(handle))
        charts = json.loads(CHART_DATA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["source"]["result_id"],
            generator.DEFAULT_RESULT_ID,
        )
        self.assertEqual(
            manifest["source"]["files"]["result.json"]["sha256"],
            "9208bea28be10b8e90329ed4a2b620c07460089de7329d0cb26a7417a3eb77c8",
        )
        self.assertEqual(
            manifest["source"]["files"]["metrics.csv"]["sha256"],
            "b5b39b81d4374a5b6bb3226a17c64a14e310f79f53654f6a229dd5c6db97793f",
        )
        self.assertEqual(len(projection), 60)
        self.assertEqual(len(manifest["runs"]), 60)
        self.assertEqual(
            {row["method_id"] for row in projection},
            set(generator.EXPECTED_METHODS),
        )
        self.assertEqual(
            {float(row["average_power_budget"]) for row in projection},
            set(generator.EXPECTED_BUDGETS),
        )
        self.assertEqual(
            {int(row["paired_seed"]) for row in projection},
            set(generator.EXPECTED_SEEDS),
        )
        self.assertFalse(
            any(
                "payload" in column.lower()
                for column in projection[0]
            )
        )

        self.assertEqual(
            generator._summary_rows(projection),
            summary,
        )
        self.assertEqual(len(summary), 5)
        self.assertTrue(
            all(
                row["Strongest baseline"]
                == "Uncertainty-shrunk water filling"
                for row in summary
            )
        )
        self.assertTrue(
            all(float(row["Paired goodput gain (bit/s/Hz)"]) > 0 for row in summary)
        )
        self.assertAlmostEqual(
            float(summary[0]["Relative gain (%)"]),
            16.53708333,
            places=7,
        )
        self.assertAlmostEqual(
            float(summary[-1]["Relative gain (%)"]),
            3.457379393,
            places=7,
        )

        self.assertEqual(
            set(charts),
            {
                "delayed-csi-goodput",
                "delayed-csi-predicted-bler",
                "delayed-csi-representative-state",
            },
        )
        state_chart = charts["delayed-csi-representative-state"]
        self.assertEqual(state_chart["rightYLabel"], "Allocated power")
        self.assertEqual(
            {
                series["id"]
                for series in state_chart["series"]
                if series.get("yAxis") == "right"
            },
            set(generator.EXPECTED_METHODS),
        )
        self.assertEqual(
            {
                series["id"]
                for series in state_chart["series"]
                if series.get("yAxis") != "right"
            },
            {"current_channel", "delayed_csi"},
        )
        self.assertEqual(
            manifest["representative_preview"]["subcarrier_count"],
            128,
        )
        self.assertEqual(
            manifest["representative_preview"]["display_stride"],
            2,
        )
        javascript = CHART_JS_PATH.read_text(encoding="utf-8")
        for chart_id in charts:
            self.assertIn('"%s"' % chart_id, javascript)

    def test_future_builder_declares_safe_aggregation_semantics(self):
        builder = (
            ROOT
            / "demo_trainings"
            / "resource_allocation_delayed_csi_finite_blocklength"
            / "build_benchmark.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"pairing_id": str(paired_seed)', builder)
        self.assertIn('"aggregation_cell_id": (', builder)
        self.assertIn(
            '"statistical_unit": ('
            '\n                                    "paired held-out TDL trajectory seed"',
            builder,
        )


if __name__ == "__main__":
    unittest.main()
