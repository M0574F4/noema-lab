from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA_JS = DOCS / "_static" / "noema-demo-chart-data.js"
RUNTIME_JS = DOCS / "_static" / "noema-demo-charts-v2.js"


class DocumentationInteractiveChartTests(unittest.TestCase):
    def test_current_demo_result_figures_use_the_shared_interactive_runtime(self):
        pages = {
            "tutorials/learned_qpsk_demapper_demo.md": {
                "qpsk-iq-calibration-ber",
                "qpsk-iq-calibration-boundaries",
            },
            "tutorials/learned_qpsk_phase_tracking_demo.md": {
                "qpsk-phase-tracking-ber",
                "qpsk-phase-tracking-phase-estimates",
                "qpsk-phase-tracking-phase-error",
            },
            "tutorials/ofdm_resource_allocation_demo.md": {
                "ofdm-allocation-spectral-efficiency",
                "ofdm-allocation-shannon-optimum-achievement",
                "ofdm-allocation-representative-power",
            },
            "tutorials/reliability_aware_ofdm_allocation_demo.md": {
                "delayed-csi-goodput",
                "delayed-csi-predicted-bler",
                "delayed-csi-representative-state",
            },
            "tutorials/learned_mimo_ofdm_channel_estimation_demo.md": {
                "mimo-channel-estimation-nmse",
                "mimo-channel-estimation-profile-gain",
                "mimo-channel-estimation-zf-rate",
                "mimo-channel-estimation-zf-retention",
                "mimo-channel-estimation-response-preview",
            },
            "tutorials/learned_csi_feedback.md": {
                "csi-feedback-rate",
                "csi-feedback-retention",
                "csi-feedback-nmse",
                "csi-feedback-response-preview",
                "csi-feedback-error-preview",
            },
            "tutorials/digital_vs_deepjscc_sionna.md": {
                "deepjscc-psnr",
                "deepjscc-ms-ssim",
            },
            "tutorials/deepjscc_slow_rayleigh.md": {
                "deepjscc-slow-psnr-snr",
                "deepjscc-slow-ms-ssim-snr",
                "deepjscc-slow-rate-psnr",
            },
        }
        for relative_path, expected_ids in pages.items():
            source = (DOCS / relative_path).read_text(encoding="utf-8")
            actual_ids = set(re.findall(r'data-noema-chart="([^"]+)"', source))
            self.assertEqual(actual_ids, expected_ids, relative_path)
            self.assertNotRegex(
                source,
                r"demo/assets/.+\.svg",
                relative_path,
            )

        configuration = (DOCS / "conf.py").read_text(encoding="utf-8")
        self.assertIn('"noema-demo-chart-data.js"', configuration)
        self.assertIn(
            '"noema-ofdm-resource-allocation-chart-data.js"',
            configuration,
        )
        self.assertIn(
            '"noema-delayed-csi-demo-chart-data.js"',
            configuration,
        )
        self.assertIn(
            '"noema-mimo-channel-estimation-chart-data.js"',
            configuration,
        )
        self.assertIn(
            '"noema-csi-feedback-chart-data.js"',
            configuration,
        )
        self.assertIn(
            '"noema-qpsk-phase-tracking-chart-data.js"',
            configuration,
        )
        self.assertIn('"noema-deepjscc-chart-data.js"', configuration)
        self.assertIn(
            '"noema-deepjscc-slow-rayleigh-chart-data-v2.js"',
            configuration,
        )
        self.assertIn('"noema-demo-charts-v2.js"', configuration)
        runtime = RUNTIME_JS.read_text(encoding="utf-8")
        self.assertIn("Array.isArray(this.spec.yDomain)", runtime)
        self.assertIn('this.root.setAttribute("role", "figure")', runtime)
        self.assertIn('aria-labelledby="${titleId}"', runtime)
        self.assertNotIn('role="listitem"', runtime)
        self.assertIn('header.setAttribute("scope", "col")', runtime)
        self.assertIn('header.setAttribute("scope", "row")', runtime)
        self.assertIn("enhanceCodeBlocks()", runtime)

    def test_qpsk_ber_and_boundary_figures_share_canonical_method_styles(self):
        source = DATA_JS.read_text(encoding="utf-8")
        for method_id, label, color in (
            ("uncompensated_qpsk", "Uncompensated QPSK", "#2563eb"),
            ("calibrated_iq_oracle", "Calibrated I/Q oracle", "#dc2626"),
            ("learned_receiver", "Learned I/Q receiver", "#16a34a"),
        ):
            self.assertIn(f"{method_id}: Object.freeze({{", source)
            self.assertIn(f'label: "{label}"', source)
            self.assertIn(f'color: "{color}"', source)
        self.assertIn(
            '{ id: "uncompensated_qpsk", ...receiverStyles.uncompensated_qpsk',
            source,
        )
        self.assertIn(
            '{ id: "calibrated_iq_oracle", ...receiverStyles.calibrated_iq_oracle',
            source,
        )
        self.assertIn(
            '{ id: "learned_receiver", ...receiverStyles.learned_receiver',
            source,
        )

    def test_runtime_exposes_the_expected_chart_interactions(self):
        source = RUNTIME_JS.read_text(encoding="utf-8")
        self.assertIn('addEventListener("wheel"', source)
        self.assertIn('addEventListener("pointerdown"', source)
        self.assertIn('addEventListener("dblclick"', source)
        self.assertIn('addEventListener("mouseenter"', source)
        self.assertIn("new ResizeObserver", source)
        self.assertIn("this.resetView()", source)
        self.assertIn('this.yScale === "log"', source)
        self.assertIn("95% interval", source)

    def test_documentation_data_never_uses_black_or_near_black_series(self):
        forbidden = {
            "#000",
            "#000000",
            "#0f172a",
            "#111827",
            "#1f2937",
            "black",
        }
        chart_files = tuple((DOCS / "_static").glob("*chart-data.js"))
        self.assertTrue(chart_files)
        for path in chart_files:
            source = path.read_text(encoding="utf-8").lower()
            for color in forbidden:
                self.assertNotIn(
                    f'"color": "{color}"',
                    source,
                    str(path.relative_to(ROOT)),
                )

    def test_include_zero_keeps_negative_only_series_in_view(self):
        source = RUNTIME_JS.read_text(encoding="utf-8")
        self.assertIn(
            "includeZero ? Math.min(0, minimum - pad) : minimum - pad",
            source,
        )
        self.assertIn(
            "includeZero ? Math.max(0, maximum + pad) : maximum + pad",
            source,
        )
        self.assertNotIn("min: includeZero ? 0 : minimum - pad", source)

    def test_boundary_rows_are_mapped_from_q_min_without_double_inversion(self):
        source = RUNTIME_JS.read_text(encoding="utf-8")
        segment_start = source.index("segmentsFor(series)")
        segment_source = source[
            segment_start : source.index("drawBoundaries()", segment_start)
        ]
        self.assertIn(
            "const y = yMin + sourceRow * cellHeight;",
            segment_source,
        )
        self.assertNotIn("height - sourceRow", segment_source)
        self.assertIn(
            "yToPixel performs the only screen-axis inversion",
            segment_source,
        )

        data = DATA_JS.read_text(encoding="utf-8")
        self.assertIn(
            "Labeled markers are the noiseless received centroids",
            data,
        )


if __name__ == "__main__":
    unittest.main()
