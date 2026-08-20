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
GENERATOR_PATH = (
    ROOT / "tools" / "generate_qpsk_iq_calibration_demo_assets.py"
)
SOURCE_CSV = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "qpsk_iq_calibration"
    / "benchmark_projection.csv"
)
MANIFEST_PATH = SOURCE_CSV.parent / "benchmark_manifest.json"
SOURCE_SHA256 = "1a1211d8aa2f45293111c135d4d669f8496610dfc41ded446ebbb3ed4ed6985a"
COMMITTED_SVG_MATPLOTLIB_VERSION = "3.11.0"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "qpsk_iq_calibration_demo_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load QPSK I/Q-calibration asset generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QpskIqCalibrationDemoAssetTests(unittest.TestCase):
    def test_completed_projection_is_complete_and_deterministic(self):
        generator = _load_generator()
        self.assertEqual(
            hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest(),
            SOURCE_SHA256,
        )

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["result"]["id"],
            generator.RESULT_ID,
        )
        self.assertEqual(manifest["result"]["status"], "completed")
        self.assertEqual(
            manifest["result"]["benchmark_semantic_sha256"],
            "8dbfe6298ccee25b5490781ed8fa312026de1b598914055a396b60e816a960eb",
        )
        self.assertEqual(manifest["projection"]["sha256"], SOURCE_SHA256)
        self.assertEqual(manifest["projection"]["row_count"], 63)
        self.assertEqual(manifest["verification"]["integrity_errors"], [])
        self.assertEqual(
            manifest["evidence_level"],
            "completed_experimental_benchmark",
        )
        self.assertRegex(
            manifest["scope"]["trained_artifact_runtime_identity_sha256"],
            r"^[0-9a-f]{64}$",
        )

        records = generator.load_records(SOURCE_CSV)
        generator.validate_manifest(MANIFEST_PATH, SOURCE_CSV, records)
        self.assertEqual(len(records), 63)
        self.assertEqual(
            {record["method"] for record in records},
            set(generator.METHOD_ORDER),
        )
        self.assertEqual(
            {record["snr_db"] for record in records},
            set(generator.EXPECTED_SNRS),
        )
        self.assertEqual(
            {record["paired_seed"] for record in records},
            set(generator.EXPECTED_SEEDS),
        )
        self.assertEqual(
            {record["compared_bits"] for record in records},
            {1_048_576},
        )

        grouped = generator._grouped_ber(records)
        for snr_db in generator.EXPECTED_SNRS:
            uncompensated = sum(
                grouped[snr_db]["uncompensated_qpsk"]
            ) / len(generator.EXPECTED_SEEDS)
            oracle = sum(grouped[snr_db]["calibrated_iq_oracle"]) / len(
                generator.EXPECTED_SEEDS
            )
            learned = sum(grouped[snr_db]["learned_receiver"]) / len(
                generator.EXPECTED_SEEDS
            )
            self.assertLess(learned, uncompensated)
            self.assertLess((learned - oracle) / oracle, 0.08)

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
            committed = generator.output_paths(
                generator.DEFAULT_DATA_DIR,
                generator.DEFAULT_ASSET_DIR,
            )
            self.assertEqual(set(first), set(second))
            for name in first:
                self.assertEqual(
                    first[name].read_bytes(),
                    second[name].read_bytes(),
                    name,
                )
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
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 7)
            self.assertEqual(table[0]["SNR (dB)"], "-2")
            self.assertEqual(
                table[0]["Learned reduction vs uncompensated"],
                "16.18%",
            )
            self.assertEqual(
                table[-1]["Learned relative gap to oracle"],
                "0.25%",
            )

            svg = first["ber_plot"].read_text(encoding="utf-8")
            self.assertNotIn("<dc:date>", svg)
            self.assertIn(
                "Noema QPSK I/Q-calibration completed-benchmark asset generator",
                svg,
            )
            self.assertIn(generator.RESULT_ID, svg)


if __name__ == "__main__":
    unittest.main()
