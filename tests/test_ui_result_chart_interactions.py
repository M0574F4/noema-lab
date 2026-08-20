import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for chart interaction tests")
class UiResultChartInteractionTests(unittest.TestCase):
    def _run_js(self, body: str):
        harness = textwrap.dedent(
            r"""
            class FakeClassList {
              constructor() { this.values = new Set(); }
              add(...values) { values.forEach((value) => this.values.add(value)); }
              remove(...values) { values.forEach((value) => this.values.delete(value)); }
              contains(value) { return this.values.has(value); }
              toggle(value, enabled) {
                if (enabled) this.values.add(value); else this.values.delete(value);
                return Boolean(enabled);
              }
            }
            class FakeSvg {
              constructor(kind = "generic", domains = null) {
                this.kind = kind;
                this.dataset = {};
                this.classList = new FakeClassList();
                this.attributes = new Map([["viewBox", "0 0 640 292"]]);
                this.listeners = new Map();
                this.capture = null;
                if (kind !== "generic") {
                  const xDomain = domains && domains.xDomain ? domains.xDomain : [1, 9];
                  const yDomain = domains && domains.yDomain ? domains.yDomain : [0.01, 0.2];
                  Object.assign(this.dataset, {
                    width: "640", height: "292", left: "68", right: "18", top: "18", bottom: "42",
                    xMin: String(xDomain[0]), xMax: String(xDomain[1]),
                    yMin: String(yDomain[0]), yMax: String(yDomain[1]),
                    xMetric: "snr", yMetric: "ber", xStat: "mean", yStat: "mean",
                    xLog: "false", yLog: "true",
                  });
                }
                if (kind === "domain") {
                  Object.assign(this.dataset, {
                    domainViewKey: "receiver-phase-tracking",
                    domainSignature: "phase:test",
                    visualizationId: "receiver-phase-tracking",
                    xMetric: "frame_symbol_index",
                    yMetric: "unwrapped_phase_rad",
                    xLog: "false",
                    yLog: "false",
                  });
                }
              }
              matches(selector) {
                if (selector === "[data-domain-chart]") return this.kind === "domain";
                return selector === "[data-rd-chart], [data-model-time-chart]" && this.kind !== "generic" && this.kind !== "domain";
              }
              hasAttribute(name) { return this.attributes.has(name); }
              getAttribute(name) { return this.attributes.get(name) || null; }
              setAttribute(name, value) { this.attributes.set(name, String(value)); }
              addEventListener(type, listener) {
                const listeners = this.listeners.get(type) || [];
                listeners.push(listener);
                this.listeners.set(type, listeners);
              }
              listenerCount(type) { return (this.listeners.get(type) || []).length; }
              dispatch(type, event = {}) {
                event.currentTarget = this;
                (this.listeners.get(type) || []).forEach((listener) => listener(event));
              }
              getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 292 }; }
              setPointerCapture(pointerId) { this.capture = pointerId; }
              hasPointerCapture(pointerId) { return this.capture === pointerId; }
              releasePointerCapture(pointerId) { if (this.capture === pointerId) this.capture = null; }
            }
            """
        )
        script = r"""
const fs = require("fs");
const vm = require("vm");
let source = fs.readFileSync(__APP_PATH__, "utf8");
source = source.slice(0, source.lastIndexOf("init();"));
const sandbox = {
  console,
  localStorage: { getItem: () => null, setItem: () => {} },
  document: {
    documentElement: { dataset: {} },
    getElementById: () => null,
    querySelectorAll: () => [],
  },
  window: { CSS: null },
  setTimeout,
  clearTimeout,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const result = vm.runInContext(source + "\n" + __BODY__, sandbox);
process.stdout.write(JSON.stringify(result));
"""
        script = script.replace("__APP_PATH__", json.dumps(str(APP_JS)))
        script = script.replace("__BODY__", json.dumps(harness + "\n" + body))
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            self.fail(completed.stderr)
        return json.loads(completed.stdout)

    def test_generic_result_svg_wheel_pan_and_double_click_reset(self):
        result = self._run_js(
            r"""
            (() => {
              const svg = new FakeSvg();
              bindVisualizationViewportInteractions(svg);
              bindVisualizationViewportInteractions(svg);
              let prevented = 0;
              svg.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => { prevented += 1; },
              });
              const zoomed = visualizationViewBox(svg);
              svg.dispatch("pointerdown", {
                button: 0, pointerId: 7, clientX: 320, clientY: 146,
                preventDefault: () => { prevented += 1; },
              });
              const panningStarted = svg.classList.contains("panning");
              svg.dispatch("pointermove", {
                pointerId: 7, clientX: 200, clientY: 110,
                preventDefault: () => { prevented += 1; },
              });
              const panned = visualizationViewBox(svg);
              svg.dispatch("pointerup", { pointerId: 7 });
              const panningFinished = !svg.classList.contains("panning");
              svg.dispatch("dblclick", { preventDefault: () => { prevented += 1; } });
              const reset = visualizationViewBox(svg);
              const preventedBeforeOrdinaryWheel = prevented;
              svg.dispatch("wheel", {
                deltaY: 120, clientX: 320, clientY: 146,
                altKey: false, ctrlKey: false, metaKey: false,
                preventDefault: () => { prevented += 1; },
              });
              const ordinaryWheelWasConsumed = prevented > preventedBeforeOrdinaryWheel;
              const ordinaryWheelView = visualizationViewBox(svg);
              let browserZoomPrevented = false;
              svg.dispatch("wheel", {
                deltaY: -120, clientX: 320, clientY: 146,
                ctrlKey: true, metaKey: false,
                preventDefault: () => { browserZoomPrevented = true; },
              });
              return {
                listenerCounts: Object.fromEntries(
                  ["wheel", "pointerdown", "pointermove", "pointerup", "dblclick"].map(
                    (type) => [type, svg.listenerCount(type)],
                  ),
                ),
                accessible: svg.getAttribute("tabindex") === "0" && Boolean(svg.getAttribute("aria-keyshortcuts")),
                grabClass: svg.classList.contains("visualization-pan-zoom"),
                zoomed,
                panned,
                reset,
                panningStarted,
                panningFinished,
                ordinaryWheelWasConsumed,
                ordinaryWheelView,
                browserZoomPrevented,
              };
            })()
            """
        )

        self.assertEqual(result["listenerCounts"], {
            "wheel": 1,
            "pointerdown": 1,
            "pointermove": 1,
            "pointerup": 1,
            "dblclick": 1,
        })
        self.assertTrue(result["accessible"])
        self.assertTrue(result["grabClass"])
        self.assertLess(result["zoomed"]["width"], 640)
        self.assertNotEqual(result["panned"]["x"], result["zoomed"]["x"])
        self.assertEqual(result["reset"], {"x": 0, "y": 0, "width": 640, "height": 292})
        self.assertTrue(result["panningStarted"])
        self.assertTrue(result["panningFinished"])
        self.assertFalse(result["ordinaryWheelWasConsumed"])
        self.assertEqual(result["ordinaryWheelView"], {"x": 0, "y": 0, "width": 640, "height": 292})
        self.assertFalse(result["browserZoomPrevented"])

    def test_domain_aware_rd_and_runtime_charts_do_not_double_bind(self):
        result = self._run_js(
            r"""
            (() => {
              let rd = new FakeSvg("rd", state.rdView);
              let runtime = new FakeSvg("runtime", state.modelTimeView);
              const graphSurface = { classList: new FakeClassList() };
              els.graphSurface = graphSurface;
              els.resultsComparison = {
                querySelector: (selector) => selector === "[data-rd-chart]" ? rd
                  : selector === "[data-model-time-chart]" ? runtime : null,
                querySelectorAll: (selector) => selector === ".rd-chart.panning"
                  ? [rd, runtime].filter((chart) => chart.classList.contains("panning")) : [],
              };
              renderResultsDashboard = () => {
                rd = new FakeSvg("rd", state.rdView);
                runtime = new FakeSvg("runtime", state.modelTimeView);
                bindRdChartInteractions();
                bindVisualizationViewportInteractions(rd);
                bindVisualizationViewportInteractions(runtime);
              };
              bindRdChartInteractions();
              bindRdChartInteractions();
              bindVisualizationViewportInteractions(rd);
              bindVisualizationViewportInteractions(rd);
              bindVisualizationViewportInteractions(runtime);
              bindVisualizationViewportInteractions(runtime);
              const initialCounts = {
                rd: ["wheel", "pointerdown", "dblclick", "keydown"].map((type) => rd.listenerCount(type)),
                runtime: ["wheel", "pointerdown", "dblclick", "keydown"].map((type) => runtime.listenerCount(type)),
              };

              let rdPrevented = 0;
              rd.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => { rdPrevented += 1; },
              });
              const rdZoomed = state.rdView.xDomain.slice();
              rd.dispatch("pointerdown", {
                button: 0, clientX: 320, clientY: 146,
                preventDefault: () => { rdPrevented += 1; },
              });
              onPointerMove({
                clientX: 220, clientY: 126,
                preventDefault: () => { rdPrevented += 1; },
              });
              const rdPanned = state.rdView.xDomain.slice();
              const replacementStayedGrabbing = rd.classList.contains("panning") && state.drag.chart === rd;
              onPointerUp({ clientX: 220, clientY: 126 });
              const rdPanFinished = state.drag === null && !rd.classList.contains("panning");
              rd.dispatch("dblclick", { preventDefault: () => { rdPrevented += 1; } });
              const rdReset = state.rdView === null;

              let runtimePrevented = 0;
              runtime.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => { runtimePrevented += 1; },
              });
              const runtimeZoomed = state.modelTimeView.xDomain.slice();
              runtime.dispatch("pointerdown", {
                button: 0, clientX: 320, clientY: 146,
                preventDefault: () => { runtimePrevented += 1; },
              });
              onPointerMove({
                clientX: 210, clientY: 120,
                preventDefault: () => { runtimePrevented += 1; },
              });
              const runtimePanned = state.modelTimeView.xDomain.slice();
              const runtimeReplacementStayedGrabbing = runtime.classList.contains("panning") && state.drag.chart === runtime;
              onPointerUp({ clientX: 210, clientY: 120 });
              runtime.dispatch("dblclick", { preventDefault: () => { runtimePrevented += 1; } });
              const runtimeReset = state.modelTimeView === null;
              return {
                initialCounts,
                allGrab: rd.classList.contains("visualization-pan-zoom") && runtime.classList.contains("visualization-pan-zoom"),
                rdZoomed, rdPanned, replacementStayedGrabbing, rdPanFinished, rdReset, rdPrevented,
                runtimeZoomed, runtimePanned, runtimeReplacementStayedGrabbing, runtimeReset, runtimePrevented,
              };
            })()
            """
        )

        self.assertEqual(result["initialCounts"], {"rd": [1, 1, 1, 1], "runtime": [1, 1, 1, 1]})
        self.assertTrue(result["allGrab"])
        self.assertLess(result["rdZoomed"][1] - result["rdZoomed"][0], 8)
        self.assertNotEqual(result["rdPanned"], result["rdZoomed"])
        self.assertTrue(result["replacementStayedGrabbing"])
        self.assertTrue(result["rdPanFinished"])
        self.assertTrue(result["rdReset"])
        self.assertGreaterEqual(result["rdPrevented"], 4)
        self.assertLess(result["runtimeZoomed"][1] - result["runtimeZoomed"][0], 8)
        self.assertNotEqual(result["runtimePanned"], result["runtimeZoomed"])
        self.assertTrue(result["runtimeReplacementStayedGrabbing"])
        self.assertTrue(result["runtimeReset"])
        self.assertGreaterEqual(result["runtimePrevented"], 4)

    def test_phase_cartesian_chart_changes_domains_without_scaling_the_svg_frame(self):
        result = self._run_js(
            r"""
            (() => {
              const chart = new FakeSvg("domain", { xDomain: [0, 127], yDomain: [-0.8, 1.4] });
              const initialViewBox = chart.getAttribute("viewBox");
              const graphSurface = { classList: new FakeClassList() };
              els.graphSurface = graphSurface;
              els.resultsComparison = {
                querySelector: () => null,
                querySelectorAll: (selector) => {
                  if (selector === "[data-domain-chart]") return [chart];
                  if (selector === ".rd-chart.panning") return chart.classList.contains("panning") ? [chart] : [];
                  return [];
                },
              };
              renderResultsDashboard = () => {
                const view = (state.domainChartViews || {})["receiver-phase-tracking"];
                if (!view) return;
                chart.dataset.xMin = String(view.xDomain[0]);
                chart.dataset.xMax = String(view.xDomain[1]);
                chart.dataset.yMin = String(view.yDomain[0]);
                chart.dataset.yMax = String(view.yDomain[1]);
              };
              bindDomainChartInteractions();
              bindDomainChartInteractions();
              bindVisualizationViewportInteractions(chart);
              let prevented = 0;
              chart.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => { prevented += 1; },
              });
              const zoomed = state.domainChartViews["receiver-phase-tracking"];
              const viewBoxAfterZoom = chart.getAttribute("viewBox");
              chart.dispatch("pointerdown", {
                button: 0, clientX: 320, clientY: 146,
                preventDefault: () => { prevented += 1; },
              });
              onPointerMove({
                clientX: 220, clientY: 126,
                preventDefault: () => { prevented += 1; },
              });
              const panned = state.domainChartViews["receiver-phase-tracking"];
              const viewBoxAfterPan = chart.getAttribute("viewBox");
              onPointerUp({ clientX: 220, clientY: 126 });
              chart.dispatch("dblclick", { preventDefault: () => { prevented += 1; } });
              return {
                listenerCounts: ["wheel", "pointerdown", "dblclick", "keydown"].map((type) => chart.listenerCount(type)),
                genericViewBoxWheelListeners: chart.listenerCount("pointermove"),
                initialViewBox,
                viewBoxAfterZoom,
                viewBoxAfterPan,
                zoomedX: zoomed.xDomain,
                pannedX: panned.xDomain,
                reset: !Object.prototype.hasOwnProperty.call(state.domainChartViews, "receiver-phase-tracking"),
                prevented,
              };
            })()
            """
        )

        self.assertEqual(result["listenerCounts"], [1, 1, 1, 1])
        self.assertEqual(result["genericViewBoxWheelListeners"], 0)
        self.assertEqual(result["initialViewBox"], result["viewBoxAfterZoom"])
        self.assertEqual(result["initialViewBox"], result["viewBoxAfterPan"])
        self.assertLess(result["zoomedX"][1] - result["zoomedX"][0], 127)
        self.assertNotEqual(result["pannedX"], result["zoomedX"])
        self.assertTrue(result["reset"])
        self.assertGreaterEqual(result["prevented"], 4)

    def test_resource_sweep_zoom_changes_power_and_metric_domains_not_viewbox(self):
        result = self._run_js(
            r"""
            (() => {
              const chart = new FakeSvg("domain", { xDomain: [0.1, 4], yDomain: [0.01, 3.2] });
              Object.assign(chart.dataset, {
                domainViewKey: "resource-goodput",
                domainSignature: "power-budget:goodput:log:log",
                visualizationId: "resource-goodput",
                xMetric: "average_power_budget",
                yMetric: "payload_goodput",
                xLog: "true",
                yLog: "true",
              });
              const initialViewBox = chart.getAttribute("viewBox");
              els.graphSurface = { classList: new FakeClassList() };
              els.resultsComparison = {
                querySelector: () => null,
                querySelectorAll: (selector) => {
                  if (selector === "[data-domain-chart]") return [chart];
                  if (selector === ".rd-chart.panning") return chart.classList.contains("panning") ? [chart] : [];
                  return [];
                },
              };
              renderResultsDashboard = () => {};
              bindDomainChartInteractions();
              bindVisualizationViewportInteractions(chart);
              chart.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => {},
              });
              const view = state.domainChartViews["resource-goodput"];
              return {
                initialViewBox,
                finalViewBox: chart.getAttribute("viewBox"),
                xDomain: view.xDomain,
                yDomain: view.yDomain,
                rootViewportPointerListeners: chart.listenerCount("pointermove"),
              };
            })()
            """
        )

        self.assertEqual(result["initialViewBox"], result["finalViewBox"])
        self.assertLess(result["xDomain"][1] / result["xDomain"][0], 4 / 0.1)
        self.assertLess(result["yDomain"][1] / result["yDomain"][0], 3.2 / 0.01)
        self.assertEqual(result["rootViewportPointerListeners"], 0)

    def test_subcarrier_dual_axis_zoom_updates_all_three_domains_not_viewbox(self):
        result = self._run_js(
            r"""
            (() => {
              const chart = new FakeSvg("domain", { xDomain: [-0.5, 15.5], yDomain: [-4, 18] });
              Object.assign(chart.dataset, {
                domainViewKey: "resource-allocation",
                domainSignature: "snapshot:0:channels:16:power-log",
                visualizationId: "resource-allocation",
                xMetric: "subcarrier_index",
                yMetric: "unit_power_snr_db",
                y2Metric: "allocated_power",
                y2Min: "0.001",
                y2Max: "2",
                xLog: "false",
                yLog: "false",
                y2Log: "true",
              });
              const initialViewBox = chart.getAttribute("viewBox");
              els.graphSurface = { classList: new FakeClassList() };
              els.resultsComparison = {
                querySelector: () => null,
                querySelectorAll: (selector) => {
                  if (selector === "[data-domain-chart]") return [chart];
                  if (selector === ".rd-chart.panning") return chart.classList.contains("panning") ? [chart] : [];
                  return [];
                },
              };
              renderResultsDashboard = () => {};
              bindDomainChartInteractions();
              bindVisualizationViewportInteractions(chart);
              chart.dispatch("wheel", {
                deltaY: -180, clientX: 320, clientY: 146,
                altKey: true, ctrlKey: false, metaKey: false,
                preventDefault: () => {},
              });
              const zoomed = state.domainChartViews["resource-allocation"];
              chart.dataset.xMin = String(zoomed.xDomain[0]);
              chart.dataset.xMax = String(zoomed.xDomain[1]);
              chart.dataset.yMin = String(zoomed.yDomain[0]);
              chart.dataset.yMax = String(zoomed.yDomain[1]);
              chart.dataset.y2Min = String(zoomed.y2Domain[0]);
              chart.dataset.y2Max = String(zoomed.y2Domain[1]);
              chart.dispatch("pointerdown", {
                button: 0, clientX: 320, clientY: 146,
                preventDefault: () => {},
              });
              onPointerMove({ clientX: 220, clientY: 126, preventDefault: () => {} });
              const panned = state.domainChartViews["resource-allocation"];
              onPointerUp({ clientX: 220, clientY: 126 });
              chart.dispatch("dblclick", { preventDefault: () => {} });
              return {
                initialViewBox,
                finalViewBox: chart.getAttribute("viewBox"),
                xDomain: zoomed.xDomain,
                yDomain: zoomed.yDomain,
                y2Domain: zoomed.y2Domain,
                pannedXDomain: panned.xDomain,
                pannedY2Domain: panned.y2Domain,
                y2Log: zoomed.y2Log,
                reset: !Object.prototype.hasOwnProperty.call(state.domainChartViews, "resource-allocation"),
                rootViewportPointerListeners: chart.listenerCount("pointermove"),
              };
            })()
            """
        )

        self.assertEqual(result["initialViewBox"], result["finalViewBox"])
        self.assertLess(result["xDomain"][1] - result["xDomain"][0], 16)
        self.assertLess(result["yDomain"][1] - result["yDomain"][0], 22)
        self.assertLess(result["y2Domain"][1] / result["y2Domain"][0], 2000)
        self.assertNotEqual(result["pannedXDomain"], result["xDomain"])
        self.assertNotEqual(result["pannedY2Domain"], result["y2Domain"])
        self.assertTrue(result["y2Log"])
        self.assertTrue(result["reset"])
        self.assertEqual(result["rootViewportPointerListeners"], 0)

    def test_results_svgs_use_hand_cursor_for_idle_and_drag_states(self):
        styles = STYLES_CSS.read_text(encoding="utf-8")
        interaction_start = styles.index(".rd-chart.visualization-pan-zoom {")
        interaction_end = styles.index(".rd-settings-button", interaction_start)
        interaction_styles = styles[interaction_start:interaction_end]
        self.assertIn("cursor: grab;", interaction_styles)
        self.assertIn("cursor: grabbing;", interaction_styles)
        self.assertNotIn("cursor: zoom-in;", interaction_styles)
        self.assertIn(".rd-chart.visualization-pan-zoom *", interaction_styles)


if __name__ == "__main__":
    unittest.main()
