from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


class UiResourceSpectralEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.app_js = APP_JS.read_text(encoding="utf-8")
        self.styles = STYLES.read_text(encoding="utf-8")
        start = self.app_js.index("function resourceSpectralEfficiencyFigure(rows)")
        end = self.app_js.index("function resourceAllocationComparisonScopeNote()", start)
        self.figure_source = self.app_js[start:end]

    def test_auto_selects_bars_for_single_budget_and_lines_for_sweep(self):
        self.assertIn("resourceSeriesHasPowerSweep(series)", self.figure_source)
        self.assertIn('const mode = useLines ? "line" : "bar"', self.figure_source)
        self.assertIn('class="resource-se-bar"', self.figure_source)
        self.assertIn('class="rd-series-line"', self.figure_source)
        self.assertIn('data-resource-performance-mode="${mode}"', self.figure_source)

        delivery_start = self.app_js.index("function resourceDeliveryFigure(rows)")
        delivery_end = self.app_js.index("function resourceDeliverySweepFigure(items)", delivery_start)
        delivery_source = self.app_js[delivery_start:delivery_end]
        self.assertIn('resourcePolicySeries(items, "goodput")', delivery_source)
        self.assertIn("resourceSeriesHasPowerSweep(series)", delivery_source)
        self.assertNotIn("distinctPowerBudgets", delivery_source)

    def test_fixed_rate_reference_is_not_drawn(self):
        self.assertNotIn("fixed-rate target", self.figure_source)
        self.assertNotIn("resource-se-target", self.figure_source)
        self.assertNotIn("approximatePowerAtSpectralEfficiency", self.app_js)

    def test_resource_primary_figures_have_one_facet_owner(self):
        overview_start = self.app_js.index("function overviewPrimaryMetricPlot(rows)")
        overview_end = self.app_js.index("function rateDistortionEligibleRows(rows)", overview_start)
        overview_source = self.app_js[overview_start:overview_end]
        self.assertNotIn("resourceSpectralEfficiencyFigure", overview_source)

        performance_start = self.app_js.index("function resultQualityPanel(rows, figureRows)")
        performance_end = self.app_js.index("function resultMetricTableSections(rows)", performance_start)
        performance_source = self.app_js[performance_start:performance_end]
        self.assertEqual(performance_source.count("resourceSpectralEfficiencyFigure"), 1)
        self.assertNotIn("resourceOracleClosenessFigure", performance_source)

        communication_start = self.app_js.index("function resultCommunicationPanel(rows")
        communication_end = self.app_js.index("function rowHasPhysicalWirelessChannel", communication_start)
        communication_source = self.app_js[communication_start:communication_end]
        self.assertIn("resourceAllocationFigure(resourceFigureRows)", communication_source)
        self.assertIn("resourceDeliveryFigure(payloadDeliveryResourceRows)", communication_source)
        self.assertIn("rowHasFiniteBlocklengthAllocationEvidence", communication_source)
        self.assertNotIn("resourceSpectralEfficiencyFigure", communication_source)

    def test_theme_aware_legend_has_focus_highlighting(self):
        self.assertIn("resourcePerformanceLegendMarkup", self.figure_source)
        self.assertIn("data-resource-performance-legend-id", self.app_js)
        self.assertIn("data-visualization-legend-key", self.app_js)
        self.assertIn("data-visualization-series-key", self.figure_source)
        self.assertIn("is-visualization-series-highlight", self.app_js)
        self.assertIn("color: var(--muted)", self.styles)
        self.assertIn("color: var(--text)", self.styles)
        self.assertIn(".visualization-series.is-visualization-series-highlight .rd-series-line", self.styles)

    def test_goodput_uses_external_color_matched_legend_below_plot(self):
        start = self.app_js.index("function resourceDeliverySweepFigure(items)")
        end = self.app_js.index("function resourceSeriesHasPowerSweep(series)", start)
        source = self.app_js[start:end]
        self.assertIn('resourcePerformanceLegendMarkup(legendEntries, "line"', source)
        self.assertIn('data-visualization-series-key="${escapeAttr(item.id)}"', source)
        self.assertNotIn('class="resource-allocation-legend visualization-series-legend"', source)
        self.assertIn(".resource-delivery-chart-wrap,", self.styles)

    def test_power_sweeps_use_domain_navigation_without_scaling_the_chart_frame(self):
        delivery_start = self.app_js.index("function resourceDeliverySweepFigure(items)")
        delivery_end = self.app_js.index("function resourceSeriesHasPowerSweep(series)", delivery_start)
        delivery_source = self.app_js[delivery_start:delivery_end]
        self.assertIn('data-domain-chart', delivery_source)
        self.assertIn('data-domain-view-key="resource-goodput"', delivery_source)
        self.assertIn('resource-goodput-sweep-clip', delivery_source)
        self.assertIn('domainChartVisibleDomains(', delivery_source)

        self.assertIn('data-domain-chart data-domain-view-key="resource-spectral-efficiency"', self.figure_source)
        self.assertIn('resource-spectral-efficiency-sweep-clip', self.figure_source)
        self.assertIn('domainChartVisibleDomains(', self.figure_source)

    def test_resource_performance_figures_expose_minimal_settings(self):
        self.assertIn('data-resource-performance-settings="goodput"', self.app_js)
        self.assertIn('data-resource-performance-settings="spectralEfficiency"', self.figure_source)
        self.assertIn("openResourcePerformanceSettingsDialog", self.app_js)
        self.assertIn("Start Y at zero", self.app_js)
        self.assertIn('data-resource-performance-setting", settings, { x: hasPowerSweep, y: true }', self.app_js)
        self.assertIn('data-x-log="${xLog ? "true" : "false"}"', self.figure_source)
        self.assertIn('data-y-log="${yLog ? "true" : "false"}"', self.figure_source)

    def test_resource_oracle_diagnostics_are_table_only(self):
        self.assertNotIn("function resourceOracleClosenessFigure", self.app_js)
        self.assertNotIn("resource-oracle-closeness", self.app_js)
        self.assertNotIn(".resource-oracle-reference-line", self.styles)
        self.assertIn("function resourceAllocationDiagnosticsTable", self.app_js)
        self.assertIn("Policy / water-filling SE", self.app_js)
        self.assertIn("resource.water_filling_oracle_spectral_efficiency_bps_hz", self.app_js)
        self.assertIn("resource.water_filling_relative_optimality_gap", self.app_js)

    def test_resource_settings_use_responsive_axis_and_display_groups(self):
        dialog_start = self.app_js.index("function openResourcePerformanceSettingsDialog(target)")
        dialog_end = self.app_js.index("function openTimeBoxSettingsDialog()", dialog_start)
        dialog_source = self.app_js[dialog_start:dialog_end]
        self.assertIn("figureSettingsToggleGrid", dialog_source)
        self.assertIn('"Axis Scale"', dialog_source)
        self.assertIn('axisScaleCompactGroup("data-resource-performance-setting"', dialog_source)
        self.assertIn("X log applies to numeric power-budget sweeps", dialog_source)
        self.assertIn(".figure-settings-toggle-grid", self.styles)
        self.assertIn("repeat(auto-fit, minmax(148px, 1fr))", self.styles)
        self.assertIn(".figure-settings-toggle", self.styles)
        self.assertIn("overflow-wrap: anywhere", self.styles)
        self.assertIn("@media (max-width: 430px)", self.styles)

    def test_subcarrier_allocation_exposes_log_power_axis_and_omits_log_zeroes(self):
        start = self.app_js.index("function resourceAllocationFigure(rows)")
        end = self.app_js.index("function resourcePerformanceItems", start)
        source = self.app_js[start:end]
        self.assertIn('data-resource-allocation-settings', source)
        self.assertIn('data-power-y-log="${powerYLog ? "true" : "false"}"', source)
        self.assertIn("resourceAllocationPowerPolylineSegments", source)
        self.assertIn("if (powerYLog && !(power > 0))", source)
        self.assertIn("function openResourceAllocationSettingsDialog()", self.app_js)
        self.assertIn('"powerYLog"', self.app_js)
        self.assertIn('data-domain-view-key="resource-allocation"', source)
        self.assertIn('data-y2-metric="allocated_power"', source)
        self.assertIn('data-y2-log="${powerYLog ? "true" : "false"}"', source)
        self.assertIn('resource-allocation-plot-clip', source)
        self.assertIn("resourceSubcarrierTicks", source)

    def test_zero_based_setting_really_includes_zero(self):
        domain_start = self.app_js.index("function chartDomainForValues(values")
        domain_end = self.app_js.index("function paddedLogDomain", domain_start)
        domain_source = self.app_js[domain_start:domain_end]
        self.assertIn("paddedDomain(values, padding, false)", domain_source)
        self.assertIn("Math.min(0, domain[0])", domain_source)
        self.assertIn("Math.max(0, domain[1])", domain_source)


if __name__ == "__main__":
    unittest.main()
