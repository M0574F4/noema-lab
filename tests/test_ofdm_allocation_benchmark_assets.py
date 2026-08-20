from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "tools" / "generate_ofdm_resource_allocation_benchmark_assets.py"
)
COMPLETED_RESULT = (
    ROOT
    / ".noema"
    / "benchmarks"
    / "20260727T002419Z_resource_allocation.learned_allocator_post_training_v2"
    / "result.json"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "ofdm_allocation_benchmark_assets",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load OFDM benchmark asset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfdmAllocationBenchmarkAssetTests(unittest.TestCase):
    @unittest.skipUnless(
        COMPLETED_RESULT.is_file(),
        "the completed local OFDM benchmark result is not shipped",
    )
    def test_completed_paired_result_projects_deterministically(self):
        generator = _load_generator()
        result = json.loads(
            (generator.DEFAULT_RESULT_DIR / "result.json").read_text(
                encoding="utf-8"
            )
        )
        generator._validate_result(result)
        records = generator._project_records(result)
        self.assertEqual(len(records), 27)
        self.assertEqual(
            {row["method_id"] for row in records},
            set(generator.EXPECTED_METHODS),
        )
        self.assertEqual(
            {float(row["average_power_budget"]) for row in records},
            set(generator.EXPECTED_BUDGETS),
        )
        self.assertEqual(
            {int(row["paired_seed"]) for row in records},
            set(generator.EXPECTED_SEEDS),
        )

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            generated = generator.generate(
                generator.DEFAULT_RESULT_DIR,
                temporary_root / "data",
                temporary_root / generator.DEFAULT_CHART_JS.name,
            )
            committed = {
                "projection": generator.DEFAULT_DATA_DIR
                / "paired_benchmark_projection.csv",
                "summary": generator.DEFAULT_DATA_DIR
                / "paired_summary_table.csv",
                "chart_data": generator.DEFAULT_DATA_DIR
                / "paired_chart_data.json",
                "chart_javascript": generator.DEFAULT_CHART_JS,
                "manifest": generator.DEFAULT_DATA_DIR
                / "paired_snapshot_manifest.json",
            }
            for name, path in generated.items():
                self.assertEqual(
                    path.read_bytes(),
                    committed[name].read_bytes(),
                    name,
                )

            with generated["summary"].open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 3)
            self.assertGreater(
                min(float(row["Learned / oracle (%)"]) for row in summary),
                99.99999,
            )
            self.assertTrue(
                all(
                    float(row["Learned (bit/s/Hz)"])
                    < float(row["Water filling (bit/s/Hz)"])
                    for row in summary
                )
            )

            chart_data = json.loads(
                generated["chart_data"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(chart_data),
                {
                    "ofdm-allocation-spectral-efficiency",
                    "ofdm-allocation-shannon-optimum-achievement",
                    "ofdm-allocation-representative-power",
                },
            )
            representative = chart_data[
                "ofdm-allocation-representative-power"
            ]
            self.assertEqual(representative["rightYLabel"], "Allocated power")
            self.assertEqual(len(representative["series"]), 4)

            manifest = json.loads(
                generated["manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["source"]["result_id"],
                generator.DEFAULT_RESULT_ID,
            )
            self.assertEqual(len(manifest["runs"]), 27)


if __name__ == "__main__":
    unittest.main()
