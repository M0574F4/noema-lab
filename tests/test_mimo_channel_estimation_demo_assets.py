from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "tools" / "generate_mimo_channel_estimation_demo_assets.py"
)
DATA_DIR = (
    ROOT / "docs" / "demo" / "data" / "mimo_ofdm_channel_estimation"
)
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
PROJECTION_PATH = DATA_DIR / "benchmark_projection.csv"
SUMMARY_PATH = DATA_DIR / "summary_table.csv"
CHART_DATA_PATH = DATA_DIR / "chart_data.json"
CHART_JS_PATH = (
    ROOT / "docs" / "_static" / "noema-mimo-channel-estimation-chart-data.js"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "mimo_channel_estimation_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load MIMO channel-estimation generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MimoChannelEstimationDemoAssetTests(unittest.TestCase):
    def test_snapshot_is_complete_paired_and_reproducible(self):
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
            "98c0ab5f50e63de88dc83b3e0e47a076941b13ce493032df9420adb2dbd62181",
        )
        self.assertEqual(
            manifest["source"]["files"]["metrics.csv"]["sha256"],
            "f5d147f077ff6f57d2824efe632b2c610fee5c9ed6b836b1db6335cafeaecfc0",
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
        self.assertEqual(
            {
                (row["tdl_profile"], int(row["paired_seed"]))
                for row in projection
            },
            {
                (profile, seed)
                for seed, profile in generator.EXPECTED_PROFILE_BY_SEED.items()
            },
        )

        self.assertEqual(generator._summary_rows(projection), summary)
        self.assertEqual(len(summary), 6)
        self.assertLess(
            float(summary[0]["Learned − strongest baseline (dB)"]),
            0,
        )
        self.assertLess(
            float(summary[-1]["Learned − strongest baseline (dB)"]),
            0,
        )
        self.assertEqual(
            set(charts),
            {
                "mimo-channel-estimation-nmse",
                "mimo-channel-estimation-profile-gain",
                "mimo-channel-estimation-zf-rate",
                "mimo-channel-estimation-zf-retention",
                "mimo-channel-estimation-response-preview",
            },
        )
        preview = charts["mimo-channel-estimation-response-preview"]
        self.assertEqual(len(preview["series"]), 3)
        self.assertTrue(
            all(len(series["values"]) == 64 for series in preview["series"])
        )
        javascript = CHART_JS_PATH.read_text(encoding="utf-8")
        for chart_id in charts:
            self.assertIn('"%s"' % chart_id, javascript)

    def test_completed_v2_result_matches_the_current_trainer(self):
        tutorial = (
            ROOT
            / "docs"
            / "tutorials"
            / "learned_mimo_ofdm_channel_estimation_demo.md"
        ).read_text(encoding="utf-8")
        model = (
            ROOT
            / "demo_trainings"
            / "mimo_ofdm_channel_estimation_cnn"
            / "model.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "20260726T235356Z_mimo_ofdm.learned_channel_estimation_post_training_v2",
            tutorial,
        )
        self.assertIn("18 paired", tutorial)
        self.assertIn("learned dual-domain estimator", tutorial)
        self.assertIn("pilot_mask", model)
        self.assertIn("self.idft_cos", model)
        self.assertIn("self.delay_head", model)
        self.assertIn("self.shrinkage", model)


if __name__ == "__main__":
    unittest.main()
