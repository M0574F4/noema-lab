from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "generate_launch_assets.py"
ASSET_ROOT = ROOT / "docs" / "_static" / "launch"


def _generator():
    spec = importlib.util.spec_from_file_location("generate_launch_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _generator()


class LaunchAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(
            (ROOT / "launch_evidence.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (ASSET_ROOT / "manifest.json").read_text(encoding="utf-8")
        )

    def test_assets_are_the_exact_deterministic_render(self) -> None:
        self.assertEqual(GENERATOR.main(["--check"]), 0)

    def test_manifest_schema_and_hashes_close_every_output(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "launch_asset_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(self.manifest)), [])
        self.assertEqual(len(self.manifest["figures"]), 6)
        self.assertEqual(len(self.manifest["tables"]), 6)
        for entry in self.manifest["figures"] + self.manifest["tables"]:
            path = ROOT / entry["path"]
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertEqual(entry["sha256"], GENERATOR._file_digest(path))
                self.assertEqual(entry["size_bytes"], path.stat().st_size)

    def test_f0_to_f5_are_accessible_svg_figures(self) -> None:
        for identifier, filename, title in GENERATOR.FIGURES:
            payload = (ASSET_ROOT / filename).read_text(encoding="utf-8")
            with self.subTest(identifier=identifier):
                self.assertIn('role="img"', payload)
                self.assertIn("<title", payload)
                self.assertIn("<desc", payload)
                self.assertIn(title.split()[0], payload)
                self.assertNotIn("paper/", payload)

    def test_f3_and_t0_are_derived_from_every_comparison_cell(self) -> None:
        figure = (ASSET_ROOT / "f3-receiver-ber-vs-snr.svg").read_text(
            encoding="utf-8"
        )
        self.assertIn("observed min/max, not confidence intervals", figure)
        self.assertNotIn("Selected by highest predeclared SNR", figure)
        self.assertNotIn('<rect x="1040" y="382"', figure)
        self.assertGreaterEqual(figure.count(f'stroke="{GENERATOR.ORANGE}"'), 9)
        self.assertIn(
            "Across seven predeclared SNR cells",
            figure,
        )
        table_path = ASSET_ROOT / "tables" / "t0-result-summary.csv"
        with table_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(self.evidence["comparisons"]))
        for rendered, source in zip(rows, self.evidence["comparisons"]):
            self.assertEqual(float(rendered["SNR (dB)"]), source["snr_db"])
            self.assertEqual(
                rendered["Learned mean BER"],
                f'{source["learned_mean_ber"]:.6e}',
            )
            self.assertEqual(
                rendered["Reduction vs uncompensated"],
                f'{source["learned_reduction_percent"]:.2f}%',
            )

    def test_presentation_contract_points_to_generated_f3_and_t0(self) -> None:
        presentation = self.evidence["presentation"]
        self.assertEqual(presentation["figure"]["status"], "canonical_asset_generated")
        self.assertEqual(
            presentation["figure"]["path"],
            "docs/_static/launch/f3-receiver-ber-vs-snr.svg",
        )
        self.assertEqual(presentation["table"]["status"], "canonical_asset_generated")
        self.assertEqual(
            presentation["table"]["paths"],
            [
                "docs/_static/launch/tables/t0-result-summary.csv",
                "docs/_static/launch/tables/t0-result-summary.md",
            ],
        )
        self.assertEqual(
            presentation["video_overlay"]["status"],
            "recorded_external_video",
        )
        self.assertEqual(
            presentation["video_overlay"]["watch_url"],
            "https://www.youtube.com/watch?v=bKNXS_vHLHc",
        )

    def test_gallery_and_release_surfaces_reference_the_asset_set(self) -> None:
        gallery = (ROOT / "docs" / "launch_assets.md").read_text(encoding="utf-8")
        for _, filename, _ in GENERATOR.FIGURES:
            self.assertIn(filename, gallery)
        for _, filename, _ in GENERATOR.TABLES:
            self.assertIn(filename, gallery)
        self.assertIn("launch_assets", (ROOT / "docs" / "index.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
