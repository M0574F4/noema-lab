from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools" / "generate_trainable_system_demo_assets.py"
EXPECTED_METHOD_COUNTS = {
    "range-localization": 3,
    "aoa-estimation": 3,
    "beam-selection": 3,
    "isac-allocation": 4,
    "near-field": 4,
    "leo-ntn": 4,
}


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "trainable_system_demo_assets", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load trainable-system asset generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainableSystemDemoAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = _load_generator()

    def test_committed_snapshots_cover_all_six_trainable_systems(self):
        self.assertEqual(set(self.generator.CONFIGS), set(EXPECTED_METHOD_COUNTS))
        for name, method_count in EXPECTED_METHOD_COUNTS.items():
            with self.subTest(name=name):
                config = self.generator.CONFIGS[name]
                data_dir = ROOT / "docs" / "demo" / "data" / config["data_slug"]
                chart_js = ROOT / "docs" / "_static" / config["chart_js"]
                paths = {
                    "projection": data_dir / "benchmark_projection.csv",
                    "summary": data_dir / "summary_table.csv",
                    "chart_data": data_dir / "chart_data.json",
                    "manifest": data_dir / "snapshot_manifest.json",
                    "chart_javascript": chart_js,
                }
                self.assertTrue(all(path.is_file() for path in paths.values()))

                with paths["projection"].open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    projection = list(csv.DictReader(handle))
                self.assertEqual(len(projection), method_count * 2 * 3)
                self.assertEqual(
                    {float(row["snr_db"]) for row in projection}, {0.0, 15.0}
                )
                self.assertEqual(
                    {int(row["paired_seed"]) for row in projection},
                    {81001, 82001, 83001},
                )

                with paths["summary"].open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    summary = list(csv.DictReader(handle))
                self.assertEqual(len(summary), method_count * 2)

                chart_data = json.loads(paths["chart_data"].read_text())
                self.assertEqual(set(chart_data), {config["chart_id"]})
                chart = chart_data[config["chart_id"]]
                self.assertEqual(len(chart["series"]), method_count)
                for series in chart["series"]:
                    self.assertEqual([point[0] for point in series["values"]], [0.0, 15.0])
                    self.assertEqual(len(series["range"]), 2)

                manifest = json.loads(paths["manifest"].read_text())
                self.assertEqual(manifest["projection"]["rows"], len(projection))
                for key, path in paths.items():
                    manifest_key = {
                        "projection": "benchmark_projection.csv",
                        "summary": "summary_table.csv",
                        "chart_data": "chart_data.json",
                        "chart_javascript": "chart_javascript",
                    }.get(key)
                    if manifest_key is None:
                        continue
                    evidence = manifest["projection"][manifest_key]
                    self.assertEqual(
                        evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                    )

    def test_local_completed_results_reproduce_committed_assets(self):
        for name, config in self.generator.CONFIGS.items():
            result_dir = ROOT / ".noema" / "benchmarks" / config["result_id"]
            if not (result_dir / "result.json").is_file():
                continue
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                temporary_root = Path(temporary)
                generated = self.generator.generate(
                    name,
                    result_dir,
                    temporary_root / "data",
                    temporary_root / config["chart_js"],
                )
                committed = {
                    "projection": ROOT
                    / "docs"
                    / "demo"
                    / "data"
                    / config["data_slug"]
                    / "benchmark_projection.csv",
                    "summary": ROOT
                    / "docs"
                    / "demo"
                    / "data"
                    / config["data_slug"]
                    / "summary_table.csv",
                    "chart_data": ROOT
                    / "docs"
                    / "demo"
                    / "data"
                    / config["data_slug"]
                    / "chart_data.json",
                    "chart_javascript": ROOT
                    / "docs"
                    / "_static"
                    / config["chart_js"],
                    "manifest": ROOT
                    / "docs"
                    / "demo"
                    / "data"
                    / config["data_slug"]
                    / "snapshot_manifest.json",
                }
                for key, generated_path in generated.items():
                    self.assertEqual(
                        generated_path.read_bytes(), committed[key].read_bytes(), key
                    )

    def test_reworked_estimators_clear_the_documented_quality_regressions(self):
        def means(slug: str, metric: str) -> dict[tuple[str, float], float]:
            path = ROOT / "docs" / "demo" / "data" / slug / "benchmark_projection.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            groups: dict[tuple[str, float], list[float]] = {}
            for row in rows:
                key = (row["method_id"], float(row["snr_db"]))
                groups.setdefault(key, []).append(float(row[metric]))
            return {key: sum(values) / len(values) for key, values in groups.items()}

        aoa = means("aoa_estimation", "aoa.rmse_deg")
        self.assertLess(aoa[("learned_estimator", 0.0)], aoa[("music", 0.0)])
        for snr in (0.0, 15.0):
            strongest_classical = min(aoa[("bartlett", snr)], aoa[("music", snr)])
            self.assertLessEqual(
                aoa[("learned_estimator", snr)], 1.02 * strongest_classical
            )

        beam = means(
            "miso_beam_selection", "beamforming.spectral_efficiency_bps_hz"
        )
        for snr in (0.0, 15.0):
            self.assertGreater(
                beam[("learned_beam_policy", snr)],
                beam[("dft_codebook_sweep", snr)],
            )
            self.assertLess(beam[("learned_beam_policy", snr)], beam[("mrt", snr)])

        near_field = means(
            "near_field_xl_mimo", "near_field.normalized_focusing_gain"
        )
        for snr in (0.0, 15.0):
            strongest_classical = max(
                near_field[("far_field_steering", snr)],
                near_field[("polar_codebook", snr)],
            )
            self.assertGreater(
                near_field[("learned_near_field_estimator", snr)],
                strongest_classical,
            )
            self.assertLess(
                near_field[("learned_near_field_estimator", snr)],
                near_field[("oracle_focus", snr)],
            )


if __name__ == "__main__":
    unittest.main()
