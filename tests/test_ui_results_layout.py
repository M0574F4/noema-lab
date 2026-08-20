from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiResultsLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.styles = STYLES_CSS.read_text(encoding="utf-8")
        cls.app = APP_JS.read_text(encoding="utf-8")

    def test_result_figure_rows_use_a_strict_two_column_grid(self):
        start = self.styles.index(".results-chart-row {")
        block = self.styles[start : self.styles.index("}", start)]
        self.assertIn("display: grid", block)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", block)

        group_start = self.styles.index(".result-performance-group {")
        group_block = self.styles[group_start : self.styles.index("}", group_start)]
        self.assertIn("display: contents", group_block)

        self.assertNotIn("flex: 1 1 50%;", self.styles)
        self.assertNotIn(".results-chart-row-single", self.styles)
        self.assertNotIn("results-chart-row-single", self.app)

    def test_only_explicit_wide_figures_can_span_both_columns(self):
        self.assertIn(".results-chart-row > .result-figure-full-width,", self.styles)
        self.assertIn("grid-column: 1 / -1;", self.styles)
        for selector in (
            ".resource-allocation-chart-wrap {",
            ".csi-feedback-preview-chart-wrap {",
            ".receiver-decision-chart-wrap {",
            ".phase-tracking-chart-wrap {",
        ):
            start = self.styles.index(selector)
            block = self.styles[start : self.styles.index("}", start)]
            self.assertNotIn("grid-column", block)

    def test_figures_fill_their_half_width_without_internal_caps(self):

        chart_start = self.styles.index(".rd-chart {")
        chart_block = self.styles[chart_start : self.styles.index("}", chart_start)]
        self.assertIn("width: 100%", chart_block)
        self.assertIn("max-width: none", chart_block)

        for selector in (".receiver-decision-chart {", ".phase-tracking-chart {"):
            start = self.styles.index(selector)
            block = self.styles[start : self.styles.index("}", start)]
            self.assertIn("width: 100%", block)
            self.assertIn("max-width: none", block)

    def test_figure_rows_stack_at_the_existing_narrow_breakpoint(self):
        media_start = self.styles.index("@media (max-width: 1040px)")
        responsive_styles = self.styles[media_start:]
        responsive_row = responsive_styles.index(".results-chart-row {")
        responsive_block = responsive_styles[responsive_row : responsive_styles.index("}", responsive_row)]
        self.assertIn("grid-template-columns: 1fr", responsive_block)


if __name__ == "__main__":
    unittest.main()
