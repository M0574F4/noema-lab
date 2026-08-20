from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryTests(unittest.TestCase):
    def test_private_research_material_is_absent(self) -> None:
        for relative in (
            "paper",
            "research",
            "literature_matches",
            "research_profile.json",
            "research_profile.md",
            "publication_handoff.yaml",
            "release_identity.yaml",
            "public_source_policy.json",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_flagship_demo_and_video_are_public(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://www.youtube.com/watch?v=bKNXS_vHLHc", readme)
        self.assertTrue((ROOT / "launch_evidence.json").is_file())
        self.assertTrue((ROOT / "docs" / "break_the_comparison.md").is_file())
        self.assertTrue((ROOT / "docs" / "demo" / "index.html").is_file())
        self.assertTrue(
            (ROOT / "docs" / "demo" / "data" / "qpsk_iq_calibration"
             / "benchmark_projection.csv").is_file()
        )

    def test_package_manifest_contains_public_demo_builders(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        for relative in (
            "launch_evidence.json",
            "tools/generate_docs_reference.py",
            "tools/generate_hosted_demo_catalog.py",
            "tools/generate_launch_assets.py",
            "tools/probe_break_the_comparison_demo.py",
        ):
            self.assertIn(relative, manifest)


if __name__ == "__main__":
    unittest.main()
