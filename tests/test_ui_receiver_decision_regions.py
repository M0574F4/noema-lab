import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for Results visualization tests")
class UiReceiverDecisionRegionTests(unittest.TestCase):
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

    def test_receiver_regions_render_all_methods_and_deduplicate_paired_runs(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const constellation = [
  { class_id: 0, bits: "00", i: 0.7071, q: 0.7071 },
  { class_id: 1, bits: "01", i: 0.7071, q: -0.7071 },
  { class_id: 2, bits: "10", i: -0.7071, q: 0.7071 },
  { class_id: 3, bits: "11", i: -0.7071, q: -0.7071 },
];
function preview(mode, label, sha, rows) {
  return {
    schema_version: 1,
    kind: "memoryless_qpsk_iq_decision_regions",
    receiver_mode: mode,
    receiver_label: label,
    model_sha256: sha,
    coordinate_space: "received_iq",
    grid: { i_min: -2, i_max: 2, q_min: -2, q_max: 2, width: 4, height: 4, class_rows: rows },
    constellation,
  };
}
const benchmark = {
  benchmark: {
    metadata: {
        demo: {
        series: [
          { id: "uncompensated_qpsk", label: "Uncompensated QPSK" },
          { id: "calibrated_iq_oracle", label: "Calibrated I/Q oracle" },
          { id: "learned_receiver", label: "Learned I/Q receiver" },
        ],
      },
    },
  },
};
function rowFor(method, runId, evidence) {
  const recipe = {
    name: `${method}_${runId}`,
    metadata: { benchmark_method: method, research: { task: { id: "neural_receiver_demapping" } } },
    steps: [{ id: "wireless_channel", op: "wireless.channel", params: { snr_db: 6 } }],
  };
  const summary = {
    run_id: runId,
    status: "completed",
    recipe,
    metrics: {},
    steps: [{
      id: "demodulator",
      metrics: { "channel.coded.ber": method === "uncompensated_qpsk" ? 0.08 : 0.021 },
      outputs: { bits: { kind: "channel.demod_bits.numpy", path: "/tmp/bits.npz", metadata: { receiver_decision_preview: evidence } } },
    }],
  };
  return {
    recipe: { key: runId, name: recipe.name, displayName: recipe.name, color: "#4b8fd8", recipe },
    summary,
    summaries: [summary],
    benchmarkResult: benchmark,
    benchmarkEntry: { label: `${method} · 6 dB · ${runId}` },
  };
}
const uncompensated = preview("reference_qpsk", "Uncompensated", "", ["3311", "3311", "2200", "2200"]);
const oracle = preview("oracle_frontend_calibrated", "Oracle", "", ["3331", "3331", "2220", "2220"]);
const learned = preview("learned_artifact", "Learned", "abc123", ["3331", "3331", "2220", "2220"]);
const rows = [
  rowFor("uncompensated_qpsk", "u-seed-1", uncompensated),
  rowFor("uncompensated_qpsk", "u-seed-2", uncompensated),
  rowFor("calibrated_iq_oracle", "o-seed-1", oracle),
  rowFor("calibrated_iq_oracle", "o-seed-2", oracle),
  rowFor("learned_receiver", "l-seed-1", learned),
  rowFor("learned_receiver", "l-seed-2", learned),
];
const entries = receiverDecisionEntries(rows);
const figure = receiverDecisionRegionFigure(rows);
const communication = resultCommunicationPanel(rows, rows);
const ordinaryRows = [
  rowFor("uncompensated_qpsk", "ordinary-u", uncompensated),
  rowFor("calibrated_iq_oracle", "ordinary-o", oracle),
  rowFor("learned_receiver", "ordinary-l", learned),
];
[
  ["Uncompensated QPSK", "#2563eb"],
  ["Calibrated I/Q oracle", "#dc2626"],
  ["Learned I/Q receiver", "#16a34a"],
].forEach(([label, color], index) => {
  const row = ordinaryRows[index];
  row.recipe.displayName = label;
  row.recipe.color = color;
  row.benchmarkResult = null;
  row.benchmarkEntry = null;
  delete row.summary.recipe.metadata.benchmark_method;
});
const ordinaryEntries = receiverDecisionEntries(ordinaryRows);
return {
  entryCount: entries.length,
  labels: entries.map((entry) => entry.methodLabel),
  ordinaryLabels: ordinaryEntries.map((entry) => entry.methodLabel),
  ordinaryColors: ordinaryEntries.map((entry) => entry.color),
  figure,
  communication,
  malformed: normalizeReceiverDecisionPreview({
    kind: "memoryless_qpsk_iq_decision_regions",
    grid: { width: 4, height: 4, i_min: -1, i_max: 1, q_min: -1, q_max: 1, class_rows: ["bad"] },
    constellation,
  }),
};
"""
        )

        self.assertEqual(result["entryCount"], 3)
        self.assertEqual(
            result["labels"],
            [
                "Uncompensated QPSK",
                "Calibrated I/Q oracle",
                "Learned I/Q receiver",
            ],
        )
        self.assertEqual(result["ordinaryLabels"], result["labels"])
        self.assertEqual(
            result["ordinaryColors"],
            ["#2563eb", "#dc2626", "#16a34a"],
        )
        figure = result["figure"]
        self.assertIn('data-visualization-id="receiver-decision-regions"', figure)
        self.assertEqual(
            figure.count('<g class="receiver-decision-panel"'),
            1,
            "all receiver methods must be overlaid in one shared I/Q panel",
        )
        self.assertEqual(
            figure.count('<svg class="rd-chart receiver-decision-chart"'),
            1,
        )
        self.assertNotIn('class="receiver-decision-region', figure)
        self.assertEqual(
            figure.count('class="receiver-decision-boundary visualization-series"'),
            3,
        )
        self.assertEqual(figure.count('class="receiver-decision-symbol"'), 4)
        self.assertEqual(
            figure.count(
                'class="receiver-decision-method-legend visualization-series-legend"'
            ),
            3,
        )
        self.assertIn("QPSK receiver decision boundaries", figure)
        self.assertEqual(figure.count('data-method-id="uncompensated_qpsk"'), 1)
        self.assertEqual(figure.count('data-method-id="calibrated_iq_oracle"'), 1)
        self.assertEqual(figure.count('data-method-id="learned_receiver"'), 1)
        self.assertIn('data-visualization-series-key="uncompensated_qpsk"', figure)
        self.assertIn('data-visualization-series-key="calibrated_iq_oracle"', figure)
        self.assertIn('data-visualization-series-key="learned_receiver"', figure)
        self.assertIn('data-visualization-legend-key="uncompensated_qpsk"', figure)
        self.assertIn('data-visualization-legend-key="calibrated_iq_oracle"', figure)
        self.assertIn('data-visualization-legend-key="learned_receiver"', figure)
        self.assertIn('data-ui-tooltip="Uncompensated QPSK"', figure)
        self.assertIn('data-ui-tooltip="Calibrated I/Q oracle"', figure)
        self.assertIn('data-ui-tooltip="Learned I/Q receiver"', figure)
        self.assertIn("stroke-dasharray", figure)
        self.assertIn("In-phase I", figure)
        self.assertIn("Quadrature Q", figure)
        self.assertIn('data-visualization-id="receiver-decision-regions"', result["communication"])
        self.assertIsNone(result["malformed"])

    def test_receiver_region_styles_use_the_global_theme(self):
        styles = STYLES_CSS.read_text(encoding="utf-8")
        self.assertIn(".receiver-decision-chart-wrap {", styles)
        self.assertIn("grid-column: 1 / -1;", styles)
        self.assertIn(".receiver-decision-boundary {", styles)
        self.assertIn(".receiver-decision-method-legend line {", styles)
        self.assertIn(
            ".receiver-decision-boundary.is-visualization-series-highlight {",
            styles,
        )
        self.assertNotIn(".receiver-decision-region-0 {", styles)


if __name__ == "__main__":
    unittest.main()
