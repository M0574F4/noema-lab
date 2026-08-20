import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


class UiGlobalVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_js = APP_JS.read_text(encoding="utf-8")
        cls.styles = STYLES_CSS.read_text(encoding="utf-8")

    def test_svg_figures_and_graph_receive_theme_preserving_exports(self):
        self.assertIn("function bindVisualizationControls()", self.app_js)
        self.assertIn(".rd-chart-wrap svg.rd-chart", self.app_js)
        self.assertIn("function bindGraphVisualizationSaveButton()", self.app_js)
        self.assertIn("function downloadVisualizationSvg(sourceSvg, filename)", self.app_js)
        self.assertIn("inlineSvgComputedStyles(sourceSvg, clone)", self.app_js)
        self.assertIn('"image/svg+xml;charset=utf-8"', self.app_js)
        self.assertIn("visualization-save-button", self.styles)
        self.assertIn("positionVisualizationSaveButton", self.app_js)

    def test_external_html_legends_are_embedded_in_standalone_svg(self):
        self.assertIn("function embedExternalVisualizationLegends(sourceSvg, clone, viewBox)", self.app_js)
        self.assertIn("function externalVisualizationLegendItems(sourceSvg)", self.app_js)
        self.assertIn('".resource-allocation-series-legend"', self.app_js)
        self.assertIn('".resource-performance-legend"', self.app_js)
        self.assertIn('".channel-response-legend"', self.app_js)
        self.assertIn('".csi-feedback-legend"', self.app_js)
        self.assertIn('".visualization-legend"', self.app_js)
        self.assertIn('"data-svg-export-legend": "true"', self.app_js)
        self.assertIn("height: viewBox.height + extensionHeight", self.app_js)
        export_start = self.app_js.index("function embedExternalVisualizationLegends")
        export_end = self.app_js.index("function inlineSvgComputedStyles", export_start)
        legend_export_source = self.app_js[export_start:export_end]
        self.assertNotIn("resource-allocation-toolbar", legend_export_source)
        self.assertNotIn('querySelectorAll("select")', legend_export_source)

    def test_native_html_titles_are_migrated_while_svg_titles_stay_accessible(self):
        self.assertIn("function initializeUiTooltipSystem()", self.app_js)
        self.assertIn('root.querySelectorAll("[title]")', self.app_js)
        self.assertIn('element.removeAttribute("title")', self.app_js)
        self.assertIn('root.querySelectorAll("svg title")', self.app_js)
        self.assertIn(
            'if (!parent.hasAttribute("aria-label")) parent.setAttribute("aria-label", text);',
            self.app_js,
        )
        self.assertIn("titleElement.remove();", self.app_js)
        self.assertIn('tooltip.className = "ui-value-tooltip result-value-tooltip"', self.app_js)
        self.assertIn(
            "<title>${escapeHtml(entry.methodLabel)}</title>",
            self.app_js,
        )
        self.assertIn(".ui-value-tooltip", self.styles)

    def test_graph_nodes_and_ports_use_accessible_custom_tooltips(self):
        render_start = self.app_js.index("function renderGraph(graph)")
        render_end = self.app_js.index("function graphEditorToolbarMarkup", render_start)
        graph_source = self.app_js[render_start:render_end]
        self.assertIn('role="button" tabindex="-1" aria-pressed=', graph_source)
        self.assertIn('role="group" tabindex="0"', graph_source)
        self.assertIn("graphNodeAccessibleLabel", graph_source)
        self.assertIn('data-ui-tooltip=', graph_source)
        self.assertIn("function graphNodePortsMarkup(node)", self.app_js)

    def test_graph_and_recipe_cards_share_one_canonical_block_name(self):
        self.assertIn("function canonicalBlockDisplayName(block)", self.app_js)
        self.assertIn("const displayName = canonicalBlockDisplayName(step);", self.app_js)
        self.assertIn("const role = canonicalBlockDisplayName(node);", self.app_js)
        self.assertNotIn('class="node-id"', self.app_js)
        self.assertNotIn(".node-id", self.styles)

    def test_visualization_typography_uses_shared_tokens(self):
        for token in (
            "--visualization-title-font-size",
            "--visualization-axis-font-size",
            "--visualization-label-font-size",
            "--visualization-title-font-weight",
            "--visualization-control-size",
            "--visualization-legend-gap",
            "--visualization-series-line-width",
            "--visualization-text-color",
            "--visualization-muted-color",
        ):
            self.assertIn(token, self.styles)
        self.assertIn("font-family: inherit", self.styles)

    def test_all_result_svgs_are_normalized_through_one_visual_contract(self):
        self.assertIn("const RESULTS_VISUALIZATION_STANDARD", self.app_js)
        self.assertIn("function applyResultsVisualizationStandard(wrap, svg)", self.app_js)
        self.assertIn(
            "function visualizationLegendTooltipText(legendItem)",
            self.app_js,
        )
        self.assertIn(
            'legendItem.setAttribute("data-ui-tooltip", tooltip)',
            self.app_js,
        )
        self.assertIn('wrap.classList.add("visualization-frame")', self.app_js)
        self.assertIn('svg.classList.add("visualization-chart")', self.app_js)
        self.assertIn('svg.setAttribute("data-visualization-standard"', self.app_js)
        self.assertIn('title.setAttribute("text-anchor", "middle")', self.app_js)
        self.assertIn(".visualization-frame .rd-settings-button", self.styles)
        self.assertIn(".visualization-legend-item", self.styles)

    def test_figure_settings_use_one_width_safe_display_and_export_layout(self):
        self.assertIn("function visualizationSettingsBody(...groups)", self.app_js)
        self.assertIn("function visualizationDisplaySettingsGroup(visualizationIds", self.app_js)
        self.assertIn("function bindVisualizationDisplaySettingsFields", self.app_js)
        self.assertIn('figureSettingsToggleGrid("data-visualization-display-setting", controls)', self.app_js)
        for field in ("showGrid", "showValues", "includeBackground"):
            self.assertIn(f'field: "{field}"', self.app_js)
        self.assertIn(".visualization-settings-standard", self.styles)
        for visualization_id in (
            "rate-distortion",
            "runtime-scatter",
            "channel-ber",
            "channel-response",
            "resource-spectral-efficiency",
            "csi-feedback-rate",
            "csi-feedback-quality",
            "csi-feedback-preview",
            "receiver-decision-regions",
        ):
            self.assertIn(f'data-visualization-id="{visualization_id}"', self.app_js)

    def test_numeric_figures_offer_log_axes_while_signed_geometry_does_not(self):
        self.assertIn("function visualizationAxisLogEnabled(key, axis, fallback = false)", self.app_js)
        self.assertIn("function visualizationAxisSettingsGroup(svg, key)", self.app_js)
        self.assertIn('data-log-axes="y"', self.app_js)
        task_start = self.app_js.index("function taskPerformancePlot(rows)")
        task_end = self.app_js.index("function taskPerformancePoints", task_start)
        task_source = self.app_js[task_start:task_end]
        self.assertIn('visualizationAxisLogEnabled(visualizationId, "y", defaultLog)', task_source)
        self.assertIn('data-y-log="${useLog ? "true" : "false"}"', task_source)
        for signed_figure in ("receiver-decision-regions", "receiver-phase-tracking"):
            marker = f'data-visualization-id="{signed_figure}"'
            line = next(line for line in self.app_js.splitlines() if marker in line)
            self.assertNotIn("data-log-axes", line)

        self.assertIn('openCommunicationSettingsDialog(metric, svg?.getAttribute("data-visualization-id") || "")', self.app_js)
        self.assertIn('els.settingsTitle.textContent = `Channel ${rateLabel} Settings`', self.app_js)

    def test_tx_power_legend_categories_are_interactive(self):
        for key in (
            "tx-power-before",
            "tx-power-after",
            "tx-power-rx-antenna",
            "tx-power-post-equalizer",
            "tx-power-noise",
        ):
            self.assertIn(f'data-visualization-legend-key="{key}"', self.app_js)
        self.assertIn("is-visualization-series-highlight", self.styles)

    def test_tx_power_figure_uses_explicit_signal_and_noise_components(self):
        self.assertIn("channel.rx_antenna_signal_power.average", self.app_js)
        self.assertIn("channel.rx_antenna_noise_power.average", self.app_js)
        self.assertIn("channel.post_equalizer_signal_power.average", self.app_js)
        self.assertIn("channel.post_equalizer_noise_power.average", self.app_js)
        self.assertIn("Finite-sample cross term", self.app_js)
        self.assertIn("line = measured total", self.app_js)
        self.assertNotIn("Useful remainder", self.app_js)

    def test_active_results_subtab_is_visibly_taller(self):
        self.assertIn(".results-facet-button {\n  height: 31px;", self.styles)
        self.assertIn(".results-facet-button.active {\n  height: 34px;", self.styles)
        self.assertIn(".results-facet-tabs {\n  min-height: 42px;", self.styles)

    def test_hidden_execution_profile_tag_cannot_render_as_an_empty_pill(self):
        self.assertIn(".execution-profile-tag[hidden] {\n  display: none;", self.styles)
        results_start = self.app_js.index("function renderResultsHeader(rows")
        results_end = self.app_js.index("function hideResultsHeaderControls", results_start)
        self.assertIn("renderExecutionProfileTag(null)", self.app_js[results_start:results_end])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for scatter-domain tests")
    def test_scatter_domains_reserve_pixel_space_for_full_marker_extents(self):
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.slice(0, source.lastIndexOf("init();"));
            const sandbox = {{
              console,
              localStorage: {{ getItem: () => null, setItem: () => {{}} }},
              document: {{ documentElement: {{ dataset: {{}} }}, getElementById: () => null }},
              window: {{ CSS: null }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              const linear = scatterDomainWithMarkerPadding([0, 10], [0, 10], 20, 200, false);
              const linearScale = chartScale(linear, 0, 200, false);
              const logarithmic = scatterDomainWithMarkerPadding([1, 100], [1, 100], 20, 200, true);
              const logScale = chartScale(logarithmic, 0, 200, true);
              globalThis.__result = {{
                linear,
                linearLeft: linearScale(0),
                linearRight: 200 - linearScale(10),
                logarithmic,
                logLeft: logScale(1),
                logRight: 200 - logScale(100),
                hoverExtent: scatterMarkerExtent([{{ size: 1 }}], () => 16),
              }};
            `, sandbox);
            process.stdout.write(JSON.stringify(sandbox.__result));
            """
        )
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            self.fail(completed.stderr)
        result = json.loads(completed.stdout)
        for key in ("linearLeft", "linearRight", "logLeft", "logRight"):
            self.assertGreaterEqual(result[key], 20 - 1e-9)
        self.assertLess(result["linear"][0], 0)
        self.assertGreater(result["linear"][1], 10)
        self.assertLess(result["logarithmic"][0], 1)
        self.assertGreater(result["logarithmic"][1], 100)
        self.assertAlmostEqual(result["hoverExtent"], 22.62, places=6)


if __name__ == "__main__":
    unittest.main()
