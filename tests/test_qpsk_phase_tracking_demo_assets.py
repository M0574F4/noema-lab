from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_qpsk_phase_tracking_demo_assets.py"
SOURCE_CSV = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "qpsk_phase_tracking"
    / "snapshot_metrics.csv"
)
MANIFEST_PATH = SOURCE_CSV.parent / "snapshot_manifest.json"
SOURCE_SHA256 = "ee19842bdff48cbb0ed2a2ac4db684b8c83892dfede99caa62fd7dddfb33cd4d"
COMMITTED_SVG_MATPLOTLIB_VERSION = "3.11.0"


def _load_generator():
    spec = importlib.util.spec_from_file_location("qpsk_phase_demo_assets", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load QPSK phase-tracking asset generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QpskPhaseTrackingDemoAssetTests(unittest.TestCase):
    def test_snapshot_projection_is_complete_and_deterministic(self):
        generator = _load_generator()
        self.assertEqual(
            hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest(), SOURCE_SHA256
        )
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["sha256"], SOURCE_SHA256)
        self.assertEqual(manifest["source"]["data_row_count"], 1780)
        self.assertEqual(manifest["evidence_level"], "illustrative_paired_snapshot")
        self.assertEqual(manifest["scope"]["bits_per_run"], 2048)
        self.assertEqual(manifest["scope"]["bits_per_packet"], 1024)
        self.assertEqual(manifest["scope"]["packets_per_run"], 2)
        self.assertIn(
            "ordinary recipe-matrix outputs, not a noema benchmark result bundle",
            manifest["limitations"],
        )

        records = generator.load_records(SOURCE_CSV)
        generator.validate_manifest(MANIFEST_PATH, SOURCE_CSV, records)
        self.assertEqual(len(records), 20)
        self.assertEqual(
            {record["method"] for record in records}, set(generator.METHOD_ORDER)
        )
        self.assertEqual(
            {record["snr_db"] for record in records}, set(generator.EXPECTED_SNRS)
        )
        self.assertEqual({record["compared_bits"] for record in records}, {2048})
        self.assertEqual({record["compared_blocks"] for record in records}, {2})

        pll_10 = next(
            record
            for record in records
            if record["method"] == "decision_directed_pll" and record["snr_db"] == 10
        )
        self.assertEqual(pll_10["bit_errors"], 4)
        learned_6 = next(
            record
            for record in records
            if record["method"] == "learned_artifact" and record["snr_db"] == 6
        )
        self.assertEqual(learned_6["bit_errors"], 54)

        for run in manifest["runs"]:
            self.assertRegex(run["authored_recipe_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(run["effective_recipe_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(run["seeds"]["data"], 23)
            self.assertEqual(run["seeds"]["wireless_channel"], 23001)
            self.assertEqual(run["seeds"]["carrier_impairment"], 23002)
            self.assertEqual(
                set(run["upstream_artifact_sha256"]),
                {
                    "payload_bits",
                    "transmitted_symbols",
                    "pilot_context",
                    "awgn_rx_symbols",
                    "carrier_impaired_rx_symbols",
                    "carrier_phase_truth",
                },
            )
            for digest in run["upstream_artifact_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        for snr_db in generator.EXPECTED_SNRS:
            paired = {
                json.dumps(
                    run["upstream_artifact_sha256"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for run in manifest["runs"]
                if run["snr_db"] == snr_db
            }
            self.assertEqual(len(paired), 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = generator.generate(
                records,
                root / "first-data",
                root / "first-assets",
                manifest_path=MANIFEST_PATH,
            )
            second = generator.generate(
                records,
                root / "second-data",
                root / "second-assets",
                manifest_path=MANIFEST_PATH,
            )
            self.assertEqual(set(first), set(second))
            committed = generator.output_paths(
                generator.DEFAULT_DATA_DIR, generator.DEFAULT_ASSET_DIR
            )
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes(), name)
                if (
                    first[name].suffix.lower() != ".svg"
                    or matplotlib.__version__ == COMMITTED_SVG_MATPLOTLIB_VERSION
                ):
                    self.assertEqual(
                        first[name].read_bytes(),
                        committed[name].read_bytes(),
                        name,
                    )

            with first["ber_table"].open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 4)
            self.assertEqual(table[0]["SNR (dB)"], "-2")
            self.assertEqual(table[-1]["Decision-directed PLL BER"], "0.001953")
            self.assertEqual(table[-1]["True-phase oracle BER"], "0.000977")

            svg = first["ber_plot"].read_text(encoding="utf-8")
            self.assertNotIn("<dc:date>", svg)
            self.assertIn("Noema QPSK phase-tracking demo asset generator", svg)
            self.assertNotIn("0/2048", svg)


if __name__ == "__main__":
    unittest.main()
