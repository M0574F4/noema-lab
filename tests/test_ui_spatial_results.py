import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for spatial Results tests")
class UiSpatialResultsTests(unittest.TestCase):
    def _run_js(self, body: str):
        script = r"""
const fs = require("fs");
const vm = require("vm");
let source = fs.readFileSync(__APP_PATH__, "utf8");
source = source.replace(/\ninit\(\);\s*$/, "\n");
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
const promise = vm.runInContext(source + "\n;(async () => {\n" + __BODY__ + "\n})()", sandbox);
Promise.resolve(promise).then(
  (result) => process.stdout.write(JSON.stringify(result)),
  (error) => { console.error(error); process.exit(1); },
);
"""
        script = script.replace("__APP_PATH__", json.dumps(str(APP_JS)))
        script = script.replace("__BODY__", json.dumps(body))
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

    def test_localization_and_aoa_visuals_are_evidence_gated_and_domain_specific(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
function rowFor(taskId, runId, output) {
  const recipe = {
    name: `${taskId}_recipe`,
    metadata: { research: { task: { id: taskId } } },
    steps: [{ id: "evaluation", op: "metrics.test", params: {} }],
  };
  const summary = {
    run_id: runId,
    status: "completed",
    recipe,
    metrics: {},
    steps: [
      { id: "evaluation", metrics: taskId === "wireless_localization" ? { "localization.rmse_m": 0.4 } : { "aoa.rmse_deg": 0.8 }, outputs: { report: output } },
    ],
  };
  return {
    recipe: { key: runId, name: recipe.name, displayName: recipe.name, color: "#4b8fd8", recipe },
    summary,
    summaries: [summary],
  };
}

const localization = rowFor("wireless_localization", "loc", {
  kind: "metrics.report",
  path: "/tmp/localization.json",
  metadata: {
    metadata: {
      localization_preview: {
        coordinate_system: "cartesian_xy_metres",
        anchors: [[0, 0], [20, 0], [20, 20], [0, 20]],
        true_positions: [[4, 5], [15, 13]],
        estimated_positions: [[4.3, 5.2], [14.7, 13.1]],
        total_example_count: 2,
      },
    },
  },
});
const aoa = rowFor("aoa_estimation", "aoa", {
  kind: "metrics.report",
  path: "/tmp/aoa.json",
  metadata: {
    metric_family: "aoa_estimation",
    rows: [
      { id: 0, true_angle_deg: -20, estimated_angle_deg: -18.5, error_deg: 1.5 },
      { id: 1, true_angle_deg: 31, estimated_angle_deg: 30.5, error_deg: 0.5 },
    ],
    metadata: {
      problem_metadata: {
        antenna_count: 8,
        element_spacing_wavelengths: 0.5,
        angle_convention: "broadside_azimuth_degrees",
      },
    },
  },
});
const missing = rowFor("wireless_localization", "missing", {
  kind: "metrics.report",
  path: "/tmp/missing.json",
  metadata: { rows: [{ id: 0, error_m: 1.2 }] },
});

const localizationMarkup = spatialResultsFigures([localization], { context: "overview" });
const aoaMarkup = spatialResultsFigures([aoa], { context: "overview" });
const missingMarkup = spatialResultsFigures([missing], { context: "overview" });
const overviewMarkup = resultOverviewPanel([localization], [localization], 0, 0);
const performanceMarkup = resultQualityPanel([aoa], [aoa]);
return {
  localizationMarkup,
  aoaMarkup,
  missingMarkup,
  overviewMarkup,
  performanceMarkup,
  localizationEvidence: spatialResultEvidence(localization.summary, localization),
  aoaEvidence: spatialResultEvidence(aoa.summary, aoa),
};
"""
        )

        localization = result["localizationMarkup"]
        self.assertIn('data-spatial-visualization="localization-map"', localization)
        self.assertIn("Localization geometry", localization)
        self.assertEqual(localization.count('class="spatial-anchor"'), 4)
        self.assertEqual(localization.count('class="spatial-true-position"'), 2)
        self.assertEqual(localization.count('class="spatial-estimated-position"'), 2)
        self.assertEqual(localization.count('class="spatial-error-vector"'), 2)
        self.assertIn("True tag positions", localization)
        self.assertIn("Estimated positions", localization)

        aoa = result["aoaMarkup"]
        self.assertIn('data-spatial-visualization="aoa-bearing"', aoa)
        self.assertIn("AoA bearing", aoa)
        self.assertIn('class="aoa-true-bearing"', aoa)
        self.assertIn('class="aoa-estimated-bearing"', aoa)
        self.assertIn("8-element ULA", aoa)
        self.assertIn("0° = broadside", aoa)
        self.assertIn("Showing example 1 of 2", aoa)
        self.assertEqual(result["aoaEvidence"]["preview"]["shownCount"], 2)

        self.assertEqual(result["missingMarkup"], "")
        self.assertIsNone(result.get("missingEvidence"))
        self.assertIn('data-spatial-visualization="localization-map"', result["overviewMarkup"])
        self.assertIn('data-spatial-visualization="aoa-bearing"', result["performanceMarkup"])
        self.assertAlmostEqual(result["localizationEvidence"]["preview"]["errors"][0], 0.3605551275463989)

    def test_malformed_or_unpaired_previews_do_not_fabricate_figures(self):
        result = self._run_js(
            r"""
return {
  missingEstimate: normalizeLocalizationPreview({
    anchors: [[0, 0], [1, 0], [0, 1]],
    true_positions: [[0.5, 0.5]],
  }),
  malformedCoordinates: normalizeLocalizationPreview({
    anchors: [[0, 0], [1, 0], [0, 1]],
    true_positions: [["not-a-number", 0.5]],
    estimated_positions: [[0.4, 0.5]],
  }),
  missingTruth: normalizeAoaPreview({ estimated_angles_deg: [10] }),
};
"""
        )
        self.assertIsNone(result["missingEstimate"])
        self.assertIsNone(result["malformedCoordinates"])
        self.assertIsNone(result["missingTruth"])

    def test_spatial_figures_follow_global_visualization_controls(self):
        app_js = APP_JS.read_text(encoding="utf-8")
        styles = STYLES_CSS.read_text(encoding="utf-8")

        self.assertIn('visualization: "localization_map"', app_js)
        self.assertIn('visualization: "aoa_bearing"', app_js)
        self.assertIn("function spatialResultsFigures(rows, options = {})", app_js)
        self.assertIn("function spatialResultEvidence(summary, row)", app_js)
        self.assertIn("function localizationMapFigure(entry)", app_js)
        self.assertIn("function aoaBearingFigure(entry)", app_js)
        self.assertIn(".visualization-grid-hidden .spatial-grid", styles)
        self.assertIn(".visualization-values-hidden .spatial-axis-values", styles)
        self.assertIn(".spatial-error-vector", styles)


if __name__ == "__main__":
    unittest.main()
