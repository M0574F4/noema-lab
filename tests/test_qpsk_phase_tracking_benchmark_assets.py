from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "tools" / "generate_qpsk_phase_tracking_benchmark_assets.py"
)
DATA_DIR = ROOT / "docs" / "demo" / "data" / "qpsk_phase_tracking"
MANIFEST_PATH = DATA_DIR / "paired_snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "paired_benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "paired_summary_table.csv"
CHART_DATA_PATH = DATA_DIR / "paired_chart_data.json"
TRACE_PATH = DATA_DIR / "representative_phase_trace.csv"
TRACE_MANIFEST_PATH = DATA_DIR / "representative_phase_manifest.json"
CHART_JS_PATH = (
    ROOT / "docs" / "_static" / "noema-qpsk-phase-tracking-chart-data.js"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "qpsk_phase_tracking_benchmark_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load QPSK phase-tracking generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QpskPhaseTrackingBenchmarkAssetTests(unittest.TestCase):
    def test_completed_benchmark_projection_is_paired_and_reproducible(self):
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
        self.assertEqual(len(projection), 126)
        self.assertEqual(len(manifest["runs"]), 126)
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
        self.assertEqual(
            {int(row["compared_bits"]) for row in projection},
            {262144},
        )
        self.assertEqual(generator._summary_rows(projection), summary)
        self.assertEqual(len(summary), 7)
        for row in summary:
            self.assertEqual(
                row["Best deployable classical tracker"],
                "Pilot smoothing",
            )
            self.assertGreater(
                float(row["Learned BER reduction vs best classical (%)"]),
                0,
            )
            self.assertLess(float(row["Oracle BER"]), float(row["Learned BER"]))

        self.assertEqual(
            set(charts),
            {
                "qpsk-phase-tracking-ber",
                "qpsk-phase-tracking-phase-estimates",
                "qpsk-phase-tracking-phase-error",
            },
        )
        self.assertEqual(
            len(charts["qpsk-phase-tracking-ber"]["series"]),
            6,
        )
        self.assertTrue(
            all(
                len(series["values"]) == 7
                for series in charts["qpsk-phase-tracking-ber"]["series"]
            )
        )
        self.assertEqual(
            len(charts["qpsk-phase-tracking-phase-estimates"]["series"]),
            5,
        )
        self.assertEqual(
            len(charts["qpsk-phase-tracking-phase-error"]["series"]),
            4,
        )
        javascript = CHART_JS_PATH.read_text(encoding="utf-8")
        for chart_id in charts:
            self.assertIn('"%s"' % chart_id, javascript)

    def test_representative_trace_is_bound_and_has_zero_oracle_error(self):
        generator = _load_generator()
        trace = generator._load_trace(TRACE_PATH, TRACE_MANIFEST_PATH)
        manifest = json.loads(
            TRACE_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(len(trace), 160)
        self.assertEqual(manifest["snr_db"], 6)
        self.assertEqual(manifest["packet_index"], 0)
        self.assertEqual(set(manifest["source_runs"]), set(generator.TRACE_METHODS))
        self.assertTrue(
            all(
                abs(float(row["oracle_phase_error_rad"])) < 1e-12
                for row in trace
            )
        )


if __name__ == "__main__":
    unittest.main()
