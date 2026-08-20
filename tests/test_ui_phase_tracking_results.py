import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for Results visualization tests")
class UiPhaseTrackingResultTests(unittest.TestCase):
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

    def test_phase_figure_separates_truth_estimates_and_direct_detector(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const series = [
  { id: "pilot_interpolation", label: "Pilot interpolation" },
  { id: "learned_receiver", label: "Learned receiver" },
  { id: "oracle_phase", label: "True-phase oracle" },
];
const benchmarkResult = { benchmark: { id: "phase-demo", metadata: { demo: { series } } } };
function rowFor(method, mode, estimate, truthUsed) {
  const recipe = {
    name: method,
    metadata: {
      benchmark_method: method,
      benchmark_paired_seed: 23,
      research: { task: { id: "neural_receiver_demapping" } },
    },
    steps: [{ id: "wireless_channel", op: "wireless.channel", params: { snr_db: 8 } }],
  };
  const diagnostic = {
    receiver_mode: mode,
    phase_estimate_available: Boolean(estimate),
    phase_truth_used_for_decisions: Boolean(truthUsed),
    phase_truth_forwarded_to_learned_runtime: false,
    receiver_decision_preview: {
      receiver_mode: mode,
      context: "one_held_out_packet_position_with_other_packet_samples_fixed",
    },
  };
  if (estimate) {
    diagnostic.phase_tracking_preview = {
      receiver_mode: mode,
      frame_symbol_index: [0, 4, 8],
      estimated_phase_rad: estimate,
    };
  }
  const summary = {
    run_id: method,
    status: "completed",
    recipe,
    steps: [
      {
        id: "carrier_impairment",
        outputs: {
          phase_truth: {
            kind: "channel.carrier_phase_truth.numpy",
            path: "/tmp/truth.npz",
            metadata: {
              phase_truth_preview: {
                packet_index: 0,
                frame_symbol_index: [0, 4, 8],
                true_phase_rad: [0.2, 0.5, 0.9],
              },
            },
          },
        },
      },
      {
        id: "demodulator",
        outputs: {
          diagnostics: {
            kind: "receiver.phase_tracking_diagnostics.numpy",
            path: "/tmp/diagnostics.npz",
            metadata: diagnostic,
          },
        },
      },
    ],
  };
  return {
    recipe: { key: method, name: method, displayName: method, color: "#345678", recipe },
    summary,
    summaries: [summary],
    benchmarkResult,
    benchmarkEntry: { label: method },
  };
}
const rows = [
  rowFor("pilot_interpolation", "pilot_interpolation", [0.18, 0.52, 0.86], false),
  rowFor("learned_receiver", "learned_artifact", null, false),
  rowFor("oracle_phase", "oracle", [0.2, 0.5, 0.9], true),
];
const evidence = receiverPhaseTrackingEvidence(rows);
const figure = receiverPhaseTrackingFigure(rows);
return {
  methodCount: evidence.methods.length,
  estimatedCount: evidence.methods.filter((item) => item.diagnostic.estimateAvailable).length,
  figure,
};
"""
        )

        self.assertEqual(result["methodCount"], 3)
        self.assertEqual(result["estimatedCount"], 2)
        figure = result["figure"]
        self.assertIn('data-visualization-id="receiver-phase-tracking"', figure)
        self.assertIn('data-domain-chart', figure)
        self.assertIn('data-domain-view-key="receiver-phase-tracking"', figure)
        self.assertEqual(figure.count('class="phase-truth-series"'), 1)
        # Each explicit estimate appears once on the truth-aligned trajectory and
        # once on the circular-error panel.
        self.assertEqual(figure.count('class="phase-estimate-series'), 4)
        self.assertIn('data-visualization-series-key="phase-truth"', figure)
        self.assertIn('data-visualization-legend-key="phase-truth"', figure)
        self.assertIn('data-visualization-series-key="phase-estimate-1"', figure)
        self.assertIn('data-visualization-legend-key="phase-estimate-1"', figure)
        self.assertIn("Learned receiver · no explicit phase estimate", figure)
        self.assertIn("True-phase oracle · oracle uses truth", figure)
        self.assertIn("phase truth is not a learned-model input", figure)
        self.assertNotIn("Learned receiver · oracle uses truth", figure)

    def test_phase_figure_lists_every_snr_and_defaults_to_representative_median(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const benchmarkResult = {
  benchmark: {
    id: "phase-snr-sweep",
    metadata: { demo: { series: [{ id: "learned_receiver", label: "Learned receiver" }] } },
  },
};
function rowFor(snr, offset) {
  const recipe = {
    name: `learned_${snr}`,
    metadata: {
      benchmark_method: "learned_receiver",
      benchmark_paired_seed: 23,
      research: { task: { id: "neural_receiver_demapping" } },
    },
    steps: [{ id: "wireless_channel", op: "wireless.channel", params: { snr_db: snr } }],
  };
  const summary = {
    run_id: recipe.name,
    status: "completed",
    recipe,
    steps: [
      {
        id: "carrier_impairment",
        outputs: {
          phase_truth: {
            kind: "channel.carrier_phase_truth.numpy",
            path: `/tmp/truth_${snr}.npz`,
            metadata: {
              phase_truth_preview: {
                packet_index: 0,
                frame_symbol_index: [0, 1, 2],
                true_phase_rad: [-0.3, -6.6, -12.9],
              },
            },
          },
        },
      },
      {
        id: "demodulator",
        outputs: {
          diagnostics: {
            kind: "receiver.phase_tracking_diagnostics.numpy",
            path: `/tmp/diagnostics_${snr}.npz`,
            metadata: {
              receiver_mode: "learned_artifact",
              phase_estimate_available: true,
              phase_truth_used_for_decisions: false,
              phase_truth_forwarded_to_learned_runtime: false,
              phase_tracking_preview: {
                receiver_mode: "learned_artifact",
                frame_symbol_index: [0, 1, 2],
                estimated_phase_rad: [-0.2 + offset, -0.25 + offset, -0.35 + offset],
              },
              receiver_decision_preview: {
                receiver_mode: "learned_artifact",
                context: "one_held_out_packet_position_with_other_packet_samples_fixed",
              },
            },
          },
        },
      },
    ],
  };
  return {
    recipe: { key: recipe.name, name: recipe.name, displayName: recipe.name, color: "#345678", recipe },
    summary,
    summaries: [summary],
    benchmarkResult,
    benchmarkEntry: { label: recipe.name },
  };
}
// Deliberately shuffle the sweep so the default cannot be an insertion-order accident.
const rows = [-2, 10, 2, 6].map((snr, index) => rowFor(snr, index * 0.01));
const figure = receiverPhaseTrackingFigure(rows);
const evidence = receiverPhaseTrackingEvidence(rows);
const tenDb = evidence.conditions.find((condition) => condition.snr === 10);
state.receiverPhaseTrackingSelections = {
  ...(state.receiverPhaseTrackingSelections || {}),
  [evidence.selectionScope]: tenDb.key,
};
const explicitlySelectedFigure = receiverPhaseTrackingFigure(rows);
const syntheticConditions = [-6, 10, -2, 2].map((snr, order) => ({
  snr,
  pairedSeed: 23,
  order,
  methods: new Map([["learned_receiver", {}]]),
}));
return {
  figure,
  explicitlySelectedFigure,
  genericMedianSnr: defaultReceiverPhaseTrackingCondition(syntheticConditions).snr,
};
"""
        )

        figure = result["figure"]
        self.assertIn('data-phase-condition-select', figure)
        self.assertEqual(figure.count("<option "), 4)
        for snr in ("-2 dB", "2 dB", "6 dB", "10 dB"):
            self.assertIn(snr, figure)
        self.assertRegex(figure, r'<option[^>]*selected[^>]*>[^<]*6(?:\.0+)? dB')
        self.assertIn("Carrier phase correction · 6 dB", figure)
        self.assertNotIn("Carrier phase correction · -2 dB", figure)
        self.assertEqual(result["genericMedianSnr"], 2)
        self.assertIn(
            "Carrier phase correction · 10 dB",
            result["explicitlySelectedFigure"],
        )

        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('querySelectorAll("[data-phase-condition-select]")', source)

    def test_phase_estimates_are_truth_aligned_and_show_circular_error_guides(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const tau = 2 * Math.PI;
const truth = {
  packetIndex: 0,
  points: [
    { x: 0, y: -0.3 },
    { x: 1, y: -0.3 - tau },
    { x: 2, y: -0.3 - 2 * tau },
  ],
};
const wrappedEstimate = {
  packetIndex: 0,
  points: [
    { x: 0, y: -0.2 },
    { x: 1, y: -0.2 },
    { x: 2, y: -0.2 },
  ],
};
const aligned = truthAlignedPhaseEstimatePoints(wrappedEstimate, truth);
const errors = aligned.map((point, index) => (
  circularPhaseErrorRad(wrappedEstimate.points[index].y, truth.points[index].y)
));

const series = [{ id: "learned_receiver", label: "Learned receiver" }];
const recipe = {
  name: "learned_receiver_6db",
  metadata: {
    benchmark_method: "learned_receiver",
    benchmark_paired_seed: 23,
    research: { task: { id: "neural_receiver_demapping" } },
  },
  steps: [{ id: "wireless_channel", op: "wireless.channel", params: { snr_db: 6 } }],
};
const summary = {
  run_id: recipe.name,
  status: "completed",
  recipe,
  steps: [
    {
      id: "carrier_impairment",
      outputs: {
        phase_truth: {
          kind: "channel.carrier_phase_truth.numpy",
          path: "/tmp/truth.npz",
          metadata: {
            phase_truth_preview: {
              packet_index: 0,
              frame_symbol_index: truth.points.map((point) => point.x),
              true_phase_rad: truth.points.map((point) => point.y),
            },
          },
        },
      },
    },
    {
      id: "demodulator",
      outputs: {
        diagnostics: {
          kind: "receiver.phase_tracking_diagnostics.numpy",
          path: "/tmp/diagnostics.npz",
          metadata: {
            receiver_mode: "learned_artifact",
            phase_estimate_available: true,
            phase_truth_used_for_decisions: false,
            phase_truth_forwarded_to_learned_runtime: false,
            phase_tracking_preview: {
              receiver_mode: "learned_artifact",
              frame_symbol_index: wrappedEstimate.points.map((point) => point.x),
              estimated_phase_rad: wrappedEstimate.points.map((point) => point.y),
            },
            receiver_decision_preview: {
              receiver_mode: "learned_artifact",
              context: "one_held_out_packet_position_with_other_packet_samples_fixed",
            },
          },
        },
      },
    },
  ],
};
const row = {
  recipe: { key: recipe.name, name: recipe.name, displayName: recipe.name, color: "#345678", recipe },
  summary,
  summaries: [summary],
  benchmarkResult: { benchmark: { id: "phase-demo", metadata: { demo: { series } } } },
  benchmarkEntry: { label: recipe.name },
};
return {
  truth: truth.points,
  aligned,
  errors,
  figure: receiverPhaseTrackingFigure([row]),
};
"""
        )

        self.assertEqual(len(result["aligned"]), 3)
        for truth, aligned, error in zip(result["truth"], result["aligned"], result["errors"]):
            self.assertAlmostEqual(error, 0.1)
            self.assertAlmostEqual(aligned["y"] - truth["y"], 0.1)

        figure = result["figure"]
        self.assertIn("Applied correction versus required correction", figure)
        self.assertIn("Residual phase after correction", figure)
        self.assertIn("wrap(φ − estimate)", figure)
        self.assertIn('data-phase-error-series="learned_receiver"', figure)
        self.assertIn('data-phase-error-guide="positive"', figure)
        self.assertIn('data-phase-error-guide="negative"', figure)
        self.assertIn("π/4", figure)
        self.assertNotIn("Unwrapped phase (rad)", figure)

    def test_phase_alignment_preserves_truth_samples_and_regular_learned_runs_do_not_collapse(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const exactTruth = {
  points: [{ x: 0, y: -0.4 }, { x: 2, y: -2.7 }],
};
const denseEstimate = {
  points: [{ x: 0, y: -0.3 }, { x: 1, y: -1.4 }, { x: 2, y: -2.6 }],
};
const aligned = truthAlignedPhaseEstimatePoints(denseEstimate, exactTruth);

function regularRow(key, offset) {
  const recipe = {
    name: key,
    metadata: { research_stage: "shared_phase_comparison", seed: 23 },
    steps: [
      { id: "wireless_channel", op: "wireless.channel", params: { snr_db: 6, seed: 23001 } },
      { id: "carrier_impairment", op: "wireless.carrier_phase_impairment", params: { seed: 23002 } },
    ],
  };
  const summary = {
    run_id: key,
    status: "completed",
    recipe,
    steps: [
      {
        id: "carrier_impairment",
        outputs: { phase_truth: { metadata: {
          carrier_phase_seed: 23002,
          pilot_seed: 1701,
          wireless_history: [{ seed: 23001 }],
          phase_truth_preview: {
            frame_symbol_index: [0, 2],
            true_phase_rad: [-0.4, -2.7],
          },
        } } },
      },
      {
        id: "demodulator",
        outputs: { diagnostics: { metadata: {
          receiver_mode: "learned_artifact",
          phase_estimate_available: true,
          phase_tracking_preview: {
            frame_symbol_index: [0, 1, 2],
            estimated_phase_rad: [-0.3 + offset, -1.4 + offset, -2.6 + offset],
          },
          receiver_decision_preview: { context: "one_held_out_packet_position" },
        } } },
      },
    ],
  };
  return {
    recipe: { key, name: key, displayName: key, color: "#345678", recipe },
    summary,
    summaries: [summary],
    benchmarkResult: null,
    benchmarkEntry: null,
  };
}
const evidence = receiverPhaseTrackingEvidence([
  regularRow("learned_artifact_a", 0),
  regularRow("learned_artifact_b", 0.03),
]);
return {
  aligned,
  methodIds: evidence.methods.map((method) => method.methodId),
  methodLabels: evidence.methods.map((method) => method.methodLabel),
};
"""
        )

        self.assertEqual([point["x"] for point in result["aligned"]], [0, 2])
        self.assertEqual([point["truth"] for point in result["aligned"]], [-0.4, -2.7])
        self.assertEqual(len(result["methodIds"]), 2)
        self.assertEqual(len(set(result["methodIds"])), 2)
        self.assertEqual(
            set(result["methodLabels"]),
            {"learned_artifact_a", "learned_artifact_b"},
        )

    def test_method_curves_aggregate_paired_seeds_for_ber_and_bler(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const benchmarkResult = {
  benchmark: {
    id: "phase-demo",
    metadata: { demo: { series: [
      { id: "pilot_interpolation", label: "Pilot interpolation" },
      { id: "learned_receiver", label: "Learned receiver" },
    ] } },
  },
};
function rowFor(method, snr, seed, ber, bler) {
  const recipe = {
    name: `${method}_${snr}_${seed}`,
    metadata: {
      benchmark_method: method,
      benchmark_paired_seed: seed,
      research: { task: { id: "neural_receiver_demapping" } },
    },
    steps: [{ id: "wireless_channel", op: "wireless.channel", params: { snr_db: snr } }],
  };
  const summary = {
    run_id: recipe.name,
    status: "completed",
    recipe,
    steps: [
      {
        id: "coded_ber",
        metrics: {
          "channel.coded.ber": ber,
          "channel.coded.error_count": Math.round(ber * 1000),
          "channel.coded.compare_bit_count": 1000,
        },
      },
      {
        id: "coded_bler",
        metrics: {
          "channel.coded.bler": bler,
          "channel.coded.block_error_count": Math.round(bler * 100),
          "channel.coded.block_count": 100,
        },
      },
    ],
  };
  return {
    recipe: { key: recipe.name, name: recipe.name, displayName: recipe.name, color: "#777777", recipe },
    summary,
    summaries: [summary],
    benchmarkResult,
    benchmarkEntry: { label: recipe.name },
  };
}
const rows = [];
const values = {
  pilot_interpolation: { "-2": [0.20, 0.22, 0.24], "2": [0.08, 0.10, 0.12] },
  learned_receiver: { "-2": [0.18, 0.20, 0.22], "2": [0.05, 0.07, 0.09] },
};
Object.entries(values).forEach(([method, bySnr]) => {
  Object.entries(bySnr).forEach(([snr, rates]) => {
    rates.forEach((ber, index) => rows.push(rowFor(method, Number(snr), index + 1, ber, ber * 2)));
  });
});
const labels = recipeDisplayLabels(rows.map((row) => row.recipe));
const settings = { xLog: false, yLog: false, colorBy: "recipe" };
const ber = communicationChartPoints(rows, labels, settings, "ber");
const bler = communicationChartPoints(rows, labels, settings, "bler");
const series = rdPointSeries(ber, (value) => value, (value) => value);
return {
  ber,
  bler,
  seriesCount: series.length,
  performance: neuralReceiverPerformanceFigures(rows),
};
"""
        )

        self.assertEqual(len(result["ber"]), 4)
        self.assertEqual(len(result["bler"]), 4)
        self.assertEqual(result["seriesCount"], 2)
        self.assertTrue(all(point["replicateCount"] == 3 for point in result["ber"]))
        learned_two_db = next(
            point
            for point in result["ber"]
            if point["benchmarkMethod"] == "learned_receiver" and point["snr"] == 2
        )
        self.assertAlmostEqual(learned_two_db["y"], 0.07)
        self.assertLess(learned_two_db["ciLow"], learned_two_db["y"])
        self.assertGreater(learned_two_db["ciHigh"], learned_two_db["y"])
        method_colors = {}
        for point in result["ber"]:
            method_colors.setdefault(point["benchmarkMethod"], set()).add(point["color"])
        self.assertTrue(all(len(colors) == 1 for colors in method_colors.values()))
        self.assertNotEqual(
            next(iter(method_colors["pilot_interpolation"])),
            next(iter(method_colors["learned_receiver"])),
        )
        self.assertIn('data-visualization-id="receiver-ber-vs-snr"', result["performance"])
        self.assertIn('data-visualization-id="receiver-bler-vs-snr"', result["performance"])
        self.assertEqual(result["performance"].count("data-domain-chart"), 2)
        self.assertIn('data-domain-view-key="receiver-ber-vs-snr"', result["performance"])
        self.assertIn('data-domain-view-key="receiver-bler-vs-snr"', result["performance"])
        self.assertEqual(result["performance"].count("data-communication-settings"), 2)
        self.assertIn('data-communication-settings="ber"', result["performance"])
        self.assertIn('data-communication-settings="bler"', result["performance"])
        self.assertEqual(result["performance"].count('class="communication-series-legend visualization-legend"'), 2)
        self.assertIn('data-visualization-legend-key="communication-series-1"', result["performance"])
        self.assertIn('data-visualization-series-key="communication-series-1"', result["performance"])
        self.assertIn("Pilot interpolation", result["performance"])
        self.assertIn("Learned receiver", result["performance"])

    def test_shared_result_viewport_zoom_pan_and_reset_math(self):
        result = self._run_js(
            r"""
const bounds = { x: 0, y: 0, width: 640, height: 292 };
const zoomed = zoomVisualizationViewBox(bounds, bounds, { x: 320, y: 146 }, 0.5);
const panned = panVisualizationViewBox(zoomed, bounds, 50, -30);
const clamped = panVisualizationViewBox(zoomed, bounds, 10000, 10000);
return { zoomed, panned, clamped };
"""
        )

        self.assertEqual(result["zoomed"], {"x": 160, "y": 73, "width": 320, "height": 146})
        self.assertEqual(result["panned"], {"x": 210, "y": 43, "width": 320, "height": 146})
        self.assertEqual(result["clamped"], {"x": 320, "y": 146, "width": 320, "height": 146})

        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("bindVisualizationViewportInteractions(svg);", source)
        self.assertIn('svg.matches("[data-rd-chart], [data-model-time-chart]")', source)
        self.assertIn('svg.addEventListener("wheel"', source)
        self.assertIn('svg.addEventListener("pointerdown"', source)
        self.assertIn('svg.addEventListener("dblclick"', source)

    def test_zero_error_rate_is_retained_on_log_axis_at_half_event_floor(self):
        result = self._run_js(
            r"""
const points = [{
  key: "oracle",
  groupKey: "demo-oracle",
  title: "True-phase oracle",
  label: "",
  seriesTitle: "True-phase oracle",
  groupSize: 2,
  color: "#2563eb",
  seriesColor: "#2563eb",
  x: 12,
  y: 0,
  size: 2000,
  xMetric: "snr",
  rateMetric: "ber",
  rateLabel: "BER",
  channel: "AWGN",
  txBits: 2000,
  channelUses: 1000,
  snr: 12,
  errors: 0,
  total: 2000,
  benchmarkMethod: "oracle_phase",
  replicateCount: 2,
  ciLow: 0,
  ciHigh: 0,
}];
const adjusted = communicationPointsForAxisScale(points, { xLog: false, yLog: true });
return {
  count: adjusted.length,
  point: adjusted[0],
  title: communicationPointTitle(adjusted[0]),
};
"""
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["point"]["observedY"], 0)
        self.assertAlmostEqual(result["point"]["y"], 0.5 / 2000)
        self.assertIn("BER 0", result["title"])
        self.assertIn("half-event floor", result["title"])

    def test_phase_and_interval_styles_are_theme_aware(self):
        styles = STYLES_CSS.read_text(encoding="utf-8")
        for token in (
            ".phase-tracking-chart-wrap {",
            ".phase-tracking-condition-controls {",
            ".phase-truth-series polyline {",
            ".phase-estimate-series polyline {",
            ".phase-error-guide {",
            ".phase-direct-detector i {",
            ".communication-confidence-interval {",
            ".communication-chart .zero-observed-rate circle {",
            ".communication-series-legend {",
            ".rd-chart.visualization-pan-zoom {",
        ):
            self.assertIn(token, styles)
        self.assertIn("var(--visualization-text-color)", styles)
        self.assertIn("var(--recipe-color, var(--accent))", styles)
        self.assertIn("aspect-ratio: 640 / 456", styles)


if __name__ == "__main__":
    unittest.main()
