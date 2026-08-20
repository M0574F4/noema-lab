from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "tools" / "generate_deepjscc_slow_rayleigh_demo_assets.py"
)
DATA_DIR = ROOT / "docs" / "demo" / "data" / "deepjscc_slow_rayleigh"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "deepjscc_slow_rayleigh_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load slow-Rayleigh docs generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepJsccSlowRayleighDemoAssetTests(unittest.TestCase):
    def test_snapshot_is_complete_paired_and_non_black(self):
        generator = _load_generator()
        manifest = json.loads(
            (DATA_DIR / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        charts = json.loads(
            (DATA_DIR / "chart_data.json").read_text(encoding="utf-8")
        )
        with (DATA_DIR / "benchmark_projection.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            projection = list(csv.DictReader(handle))
        with (DATA_DIR / "pairing_audit.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            pairing = list(csv.DictReader(handle))
        with (DATA_DIR / "snr_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            snr_summary = list(csv.DictReader(handle))
        with (DATA_DIR / "rate_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rate_summary = list(csv.DictReader(handle))

        self.assertEqual(
            manifest["source"]["result_id"], generator.DEFAULT_RESULT_ID
        )
        self.assertEqual(len(projection), 48)
        self.assertEqual(len(manifest["runs"]), 48)
        self.assertEqual(len(pairing), 24)
        self.assertTrue(all(row["Matched"] == "yes" for row in pairing))
        self.assertEqual(len(snr_summary), 5)
        self.assertEqual(len(rate_summary), 3)
        self.assertTrue(
            all(
                float(row["DeepJSCC PSNR advantage (dB)"]) > 0
                for row in snr_summary
            )
        )
        self.assertEqual(
            set(charts),
            {
                "deepjscc-slow-psnr-snr",
                "deepjscc-slow-ms-ssim-snr",
                "deepjscc-slow-rate-psnr",
            },
        )
        for chart in charts.values():
            self.assertTrue(
                all(
                    series["color"].lower() != "#000000"
                    for series in chart["series"]
                )
            )
        self.assertNotIn(
            "representative_reconstructions",
            manifest["projection"],
        )
        self.assertFalse(manifest["publication"]["publication_ready"])
        self.assertIn(
            "representative_reconstructions",
            manifest["publication"]["excluded_public_assets"],
        )
        self.assertFalse((DATA_DIR / "reconstructions").exists())


if __name__ == "__main__":
    unittest.main()
