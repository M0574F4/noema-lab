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
        self.assertIn(
            "docs/_static/noema-workflow-video-card.svg",
            readme,
        )
        self.assertIn("launch_video.html", readme)
        self.assertLess(
            readme.index("## Watch the complete workflow"),
            readme.index("## Start here"),
        )
        self.assertIn(
            "| **choose a system to train** | **the [ready-to-train matrix]",
            readme,
        )
        for removed_heading in (
            "## See why the contract matters",
            "## Suites and maturity",
            "## Documentation map",
            "## Release status",
        ):
            self.assertNotIn(removed_heading, readme)
        self.assertIn("docs/_static/noema-logo-dark.svg", readme)
        self.assertNotIn("docs/_static/noema-training-loop.svg", readme)
        self.assertLess(
            readme.index("docs/_static/launch/f0-launch-hero.svg"),
            readme.index("img.shields.io/github/actions/workflow/status"),
        )
        self.assertLess(
            readme.index("img.shields.io/github/actions/workflow/status"),
            readme.index("docs/_static/noema-logo-dark.svg"),
        )
        self.assertLess(
            readme.index("docs/_static/noema-logo-dark.svg"),
            readme.index("# Noema"),
        )
        self.assertEqual(
            readme.count("docs/_static/launch/f2-contract-to-evidence.svg"),
            1,
        )
        self.assertTrue((ROOT / "docs" / "_static" / "noema-logo.svg").is_file())
        self.assertTrue(
            (ROOT / "docs" / "_static" / "noema-logo-dark.svg").is_file()
        )
        self.assertTrue(
            (ROOT / "docs" / "_static" / "noema-training-loop.svg").is_file()
        )
        self.assertTrue((ROOT / "launch_evidence.json").is_file())
        self.assertTrue((ROOT / "docs" / "break_the_comparison.md").is_file())
        self.assertTrue((ROOT / "docs" / "demo" / "index.html").is_file())
        self.assertTrue(
            (ROOT / "docs" / "demo" / "data" / "qpsk_iq_calibration"
             / "benchmark_projection.csv").is_file()
        )

    def test_readiness_matrix_distinguishes_trainable_and_benchmark_only_systems(
        self,
    ) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Ready-to-train systems", readme)
        self.assertEqual(readme.count("✅ [Train + compare]"), 12)
        self.assertNotIn("🟡 [Benchmark now]", readme)
        for relative in (
            "tutorials/learned_qpsk_demapper_demo.html",
            "tutorials/learned_qpsk_phase_tracking_demo.html",
            "tutorials/automatic_modulation_recognition_demo.html",
            "tutorials/learned_mimo_ofdm_channel_estimation_demo.html",
            "tutorials/learned_csi_feedback.html",
            "tutorials/ofdm_resource_allocation_demo.html",
            "tutorials/reliability_aware_ofdm_allocation_demo.html",
            "tutorials/digital_vs_deepjscc_sionna.html",
            "tutorials/deepjscc_slow_rayleigh.html",
            "tutorials/learned_range_localization_demo.html",
            "tutorials/learned_aoa_estimation_demo.html",
            "tutorials/learned_beam_selection_demo.html",
        ):
            self.assertIn(
                f"https://M0574F4.github.io/noema-lab/{relative}",
                readme,
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

    def test_quickstart_and_mobile_demo_leave_a_usable_public_checkout(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        script = (ROOT / "docs" / "demo" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "docs" / "demo" / "styles.css").read_text(
            encoding="utf-8"
        )
        workflow = (
            ROOT / "docs" / "_static" / "noema-training-loop.svg"
        ).read_text(encoding="utf-8")

        self.assertIn("/noema-quickstart.yaml", ignore)
        self.assertIn("uv sync --frozen", readme)
        self.assertIn("Ctrl+C", readme)
        self.assertIn("Math.max(0.15, value)", script)
        self.assertIn(".hosted-action.primary", styles)
        self.assertNotIn(".hosted-action:first-of-type", styles)
        self.assertIn("Segoe UI, Arial, sans-serif", workflow)


if __name__ == "__main__":
    unittest.main()
