import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class UiResultTaskMetadataTests(unittest.TestCase):
    def _run_js(self, body: str, payload=None):
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
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const promise = vm.runInContext(
  source + "\n;globalThis.__payload = " + JSON.stringify(payload) +
  "\n;(async () => {\n" + __BODY__ + "\n})()",
  sandbox,
);
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
            input=json.dumps(payload or {}),
        )
        if completed.returncode:
            self.fail(completed.stderr)
        return json.loads(completed.stdout)

    def test_results_honor_nested_research_task_metadata(self):
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function recipeDeclaredTaskId(recipe)", app_js)
        self.assertIn('return String(researchTask.id || metadata.task_id || "").trim();', app_js)
        self.assertIn("recipeDeclaredTaskId(recipe) || benchmarkTaskId", app_js)

    def test_resource_allocation_uses_wireless_metric_group(self):
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn('id: "resource_allocation"', app_js)
        self.assertIn("Power-allocation comparison · complete-run metrics", app_js)
        self.assertIn("Shannon spectral efficiency (bit/s/Hz)", app_js)
        self.assertIn(
            "resourceAllocationComparisonTable(shannonResourceRows)",
            app_js,
        )

    def test_aoa_performance_table_includes_tail_error_statistics(self):
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn('label: "Median error (deg)"', app_js)
        self.assertIn('"aoa.median_error_deg"', app_js)
        self.assertIn('label: "P90 error (deg)"', app_js)
        self.assertIn('"aoa.p90_error_deg"', app_js)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for result presentation tests")
    def test_task_presentations_select_domain_primary_metrics(self):
        cases = [
            {
                "task_id": "pilot_channel_estimation",
                "task_tokens": ["pilot"],
                "metric": "channel_estimation.nmse",
                "label_token": "nmse",
                "direction": "lower_is_better",
                "value": 0.12,
            },
            {
                "task_id": "mimo_ofdm_channel_estimation",
                "task_tokens": ["mimo"],
                "metric": "mimo.channel_estimation.nmse",
                "label_token": "nmse",
                "direction": "lower_is_better",
                "value": 0.08,
            },
            {
                "task_id": "beamforming_precoding",
                "task_tokens": ["beamforming"],
                "metric": "beamforming.spectral_efficiency_bps_hz",
                "label_token": "spectral efficiency",
                "direction": "higher_is_better",
                "value": 4.8,
            },
            {
                "task_id": "beamforming_precoding",
                "task_tokens": ["beamforming"],
                "metric": "beamforming.normalized_gain",
                "label_token": "gain",
                "direction": "higher_is_better",
                "value": 0.91,
            },
            {
                "task_id": "wireless_localization",
                "task_tokens": ["localization"],
                "metric": "localization.rmse_m",
                "label_token": "rmse",
                "direction": "lower_is_better",
                "value": 1.7,
            },
            {
                "task_id": "wireless_localization",
                "task_tokens": ["localization"],
                "metric": "localization.mae_m",
                "label_token": "mae",
                "direction": "lower_is_better",
                "value": 1.2,
            },
            {
                "task_id": "aoa_estimation",
                "task_tokens": ["aoa", "angle-of-arrival", "angle of arrival"],
                "metric": "aoa.rmse_deg",
                "label_token": "rmse",
                "direction": "lower_is_better",
                "value": 3.4,
            },
            {
                "task_id": "aoa_estimation",
                "task_tokens": ["aoa", "angle-of-arrival", "angle of arrival"],
                "metric": "aoa.mae_deg",
                "label_token": "mae",
                "direction": "lower_is_better",
                "value": 2.6,
            },
            {
                "task_id": "neural_receiver_demapping",
                "task_tokens": ["neural receiver"],
                "metric": "channel.coded.ber",
                "label_token": "ber",
                "direction": "lower_is_better",
                "scale": "log",
                "value": 0.004,
            },
            {
                "task_id": "neural_receiver_demapping",
                "task_tokens": ["neural receiver"],
                "metric": "channel.coded.bler",
                "label_token": "bler",
                "direction": "lower_is_better",
                "scale": "log",
                "value": 0.03,
            },
            {
                "task_id": "bit_transport",
                "task_tokens": ["bit transport"],
                "metric": "channel.ber",
                "label_token": "ber",
                "direction": "lower_is_better",
                "scale": "log",
                "value": 0.006,
            },
            {
                "task_id": "bit_transport",
                "task_tokens": ["bit transport"],
                "metric": "channel.bler",
                "label_token": "bler",
                "direction": "lower_is_better",
                "scale": "log",
                "value": 0.04,
            },
            {
                "task_id": "speech_recognition",
                "task_tokens": ["speech"],
                "metric": "task.wer",
                "label_token": "word error",
                "direction": "lower_is_better",
                "value": 0.18,
            },
            {
                "task_id": "speech_recognition",
                "task_tokens": ["speech"],
                "metric": "task.cer",
                "label_token": "character error",
                "direction": "lower_is_better",
                "value": 0.09,
            },
            {
                "task_id": "image_captioning",
                "task_tokens": ["image captioning"],
                "metric": "caption.unigram_bleu_proxy",
                "label_token": "bleu",
                "direction": "higher_is_better",
                "value": 0.31,
            },
        ]
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
return __payload.cases.map((item) => {
  const summary = {
    recipe: {
      metadata: { research: { task: { id: item.task_id } } },
      steps: [],
    },
  };
  const details = {
    metrics: [{ step: "evaluation", metric: item.metric, value: item.value }],
    settings: [],
    artifacts: [],
  };
  const primary = primaryTaskMetric(details, summary, {});
  return {
    task_id: item.task_id,
    task_label: taskLabel(item.task_id),
    primary_label: primary && primary.label,
    primary_value: primary && primary.value,
    primary_direction: primary && primary.direction,
    primary_scale: primary && primary.scale,
  };
});
""",
            payload={"cases": cases},
        )

        for case, actual in zip(cases, result):
            with self.subTest(task_id=case["task_id"], metric=case["metric"]):
                self.assertNotEqual(actual["task_label"], case["task_id"])
                self.assertTrue(
                    any(
                        token.lower() in actual["task_label"].lower()
                        for token in case["task_tokens"]
                    ),
                    actual,
                )
                self.assertIsNotNone(actual["primary_label"], actual)
                self.assertIn(
                    case["label_token"], actual["primary_label"].lower(), actual
                )
                self.assertEqual(actual["primary_value"], case["value"])
                self.assertEqual(actual["primary_direction"], case["direction"])
                if "scale" in case:
                    self.assertEqual(actual["primary_scale"], case["scale"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for result facet tests")
    def test_result_facets_follow_evidence_and_reconcile_stale_selection(self):
        result = self._run_js(
            r"""
function artifact(kind, path, metadata = {}) {
  return { kind, path, metadata };
}
function rowFor(summary, key) {
  return {
    recipe: {
      key,
      name: summary.recipe.name,
      displayName: summary.recipe.name,
      color: "#6aa5ff",
      recipe: summary.recipe,
    },
    summary,
    summaries: [summary],
  };
}
function facetIds(rows) {
  return availableResultFacets(rows, rows).map((facet) => facet.id);
}

const metricSummary = {
  run_id: "metric-only",
  status: "completed",
  recipe: {
    name: "Metric only",
    metadata: { research: { task: { id: "classification" } } },
    steps: [{ id: "evaluation", op: "metrics.classification", params: {} }],
  },
  metrics: {},
  steps: [{ id: "evaluation", metrics: { "task.accuracy": 0.8 }, outputs: {} }],
};
const metricRow = rowFor(metricSummary, "metric");

const exampleSummary = JSON.parse(JSON.stringify(metricSummary));
exampleSummary.run_id = "example";
exampleSummary.recipe.name = "Image example";
exampleSummary.recipe.metadata.research.task.id = "image_reconstruction";
exampleSummary.recipe.steps = [
  { id: "data", op: "source.image_dataset", params: {} },
  { id: "evaluation", op: "metrics.image_reconstruction", params: {} },
];
exampleSummary.steps = [
  { id: "data", metrics: {}, outputs: { images: artifact("image.batch.numpy", "/tmp/reference.npz", { shape: [1, 8, 8, 3] }) } },
  { id: "receiver", metrics: {}, outputs: { images: artifact("image.batch.numpy", "/tmp/received.npz", { shape: [1, 8, 8, 3] }) } },
];
const exampleRow = rowFor(exampleSummary, "example");

const communicationSummary = JSON.parse(JSON.stringify(metricSummary));
communicationSummary.run_id = "communication";
communicationSummary.recipe.name = "BER evidence";
communicationSummary.steps = [
  { id: "evaluation", metrics: { "channel.coded.ber": 0.01 }, outputs: {} },
];
const communicationRow = rowFor(communicationSummary, "communication");

const physicalSummary = JSON.parse(JSON.stringify(metricSummary));
physicalSummary.run_id = "physical";
physicalSummary.recipe.name = "Physical channel";
physicalSummary.recipe.steps.push({ id: "wireless_channel", op: "wireless.channel", params: { channel: "awgn" } });
const physicalRow = rowFor(physicalSummary, "physical");

const systemsSummary = JSON.parse(JSON.stringify(metricSummary));
systemsSummary.run_id = "systems";
systemsSummary.recipe.name = "Memory evidence";
systemsSummary.metrics = { "memory.run.peak_rss_bytes": 4096 };
const systemsRow = rowFor(systemsSummary, "systems");

const metricFacets = availableResultFacets([metricRow], [metricRow]);
state.resultFacet = "communication";
const reconciled = activeResultFacet(metricFacets);
const allRows = [metricRow, exampleRow, communicationRow, systemsRow];
const allFacets = availableResultFacets(allRows, allRows);
return {
  metricFacetIds: metricFacets.map((facet) => facet.id),
  metricFacetLabels: Object.fromEntries(metricFacets.map((facet) => [facet.id, facet.label])),
  exampleKind: resultExampleKind(exampleSummary, exampleRow),
  exampleFacetIds: facetIds([exampleRow]),
  communicationFacetIds: facetIds([communicationRow]),
  physicalFacetIds: facetIds([physicalRow]),
  systemsFacetIds: facetIds([systemsRow]),
  allFacetIds: allFacets.map((facet) => facet.id),
  reconciled,
  storedFacet: state.resultFacet,
};
"""
        )

        self.assertEqual(
            result["metricFacetIds"], ["overview", "quality", "exports"]
        )
        self.assertEqual(result["metricFacetLabels"]["quality"], "Performance")
        self.assertTrue(result["exampleKind"])
        self.assertIn("examples", result["exampleFacetIds"])
        self.assertNotIn("communication", result["metricFacetIds"])
        self.assertIn("communication", result["communicationFacetIds"])
        self.assertIn("communication", result["physicalFacetIds"])
        self.assertNotIn("systems", result["metricFacetIds"])
        self.assertIn("systems", result["systemsFacetIds"])
        self.assertEqual(
            result["allFacetIds"],
            ["overview", "examples", "quality", "communication", "systems", "exports"],
        )
        self.assertEqual(result["reconciled"], "overview")
        self.assertEqual(result["storedFacet"], "overview")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for result overview tests")
    def test_overview_includes_the_primary_available_figure(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
const recipe = {
  name: "Classification baseline",
  metadata: { research: { task: { id: "classification" } } },
  steps: [{ id: "evaluation", op: "metrics.classification", params: {} }],
};
const summary = {
  run_id: "classification-run",
  status: "completed",
  recipe,
  metrics: {},
  steps: [{ id: "evaluation", metrics: { "task.accuracy": 0.8 }, outputs: {} }],
};
const row = {
  recipe: { key: "classification", name: recipe.name, displayName: recipe.name, color: "#6aa5ff", recipe },
  summary,
  summaries: [summary],
};
const markup = resultOverviewPanel([row], [row], 0, 0);
return {
  hasFigureRegion: markup.includes("data-results-overview-figure"),
  hasPrimaryChart: markup.includes("task-performance-chart"),
  overviewBeforeFigure: markup.indexOf("results-summary-strip") < markup.indexOf("data-results-overview-figure"),
};
"""
        )

        self.assertEqual(
            result,
            {
                "hasFigureRegion": True,
                "hasPrimaryChart": True,
                "overviewBeforeFigure": True,
            },
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for result eligibility tests")
    def test_figures_only_count_completed_rows_with_renderable_evidence(self):
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
state.resultHiddenRecipeKeys = [];
function row(key, status, metric = null) {
  const recipe = {
    name: key,
    metadata: { research: { task: { id: "classification" } } },
    steps: [{ id: "evaluation", op: "metrics.classification", params: {} }],
  };
  if (status === null) return { recipe: { key, name: key, recipe }, summaries: [], summary: null };
  const summary = {
    run_id: `${key}-run`, status, recipe, metrics: {},
    steps: metric === null ? [] : [{ id: "evaluation", metrics: { "task.accuracy": metric }, outputs: {} }],
  };
  return { recipe: { key, name: key, recipe }, run: { status }, summaries: [summary], summary };
}
const rows = [
  row("unrun", null),
  row("running", "running", 0.2),
  row("failed", "failed", 0.3),
  row("completed-empty", "completed", null),
  row("completed-evidence", "completed", 0.9),
];
const eligible = completedFigureResultRows(rows, { includeHidden: true });
const visibleBefore = visibleResultRows(rows);
state.resultHiddenRecipeKeys = ["completed-evidence"];
const visibleAfter = visibleResultRows(rows);
return {
  eligible: eligible.map((item) => item.recipe.key),
  visibleBefore: visibleBefore.map((item) => item.recipe.key),
  visibleAfter: visibleAfter.map((item) => item.recipe.key),
  statuses: eligible.flatMap((item) => resultSummaries(item).map((summary) => summary.status)),
};
"""
        )

        self.assertEqual(result["eligible"], ["completed-evidence"])
        self.assertEqual(result["visibleBefore"], ["completed-evidence"])
        self.assertEqual(result["visibleAfter"], [])
        self.assertEqual(result["statuses"], ["completed"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for result overview tests")
    def test_every_template_purpose_can_render_an_overview_figure(self):
        cases = [
            ("image_reconstruction", "quality.psnr_db", 31.0),
            ("text_semantic_similarity", "semantic.lexical_similarity", 0.9),
            ("classification", "task.accuracy", 0.8),
            ("visual_question_answering", "vqa.single_reference_exact_match", 0.7),
            ("object_detection", "detection.f1_at_iou_0p5", 0.6),
            ("segmentation", "segmentation.miou", 0.55),
            ("image_text_retrieval", "retrieval.recall_at_1", 0.4),
            ("image_generation", "generation.text_image_clip.cosine_mean", 0.3),
            ("pilot_channel_estimation", "channel_estimation.nmse", 0.1),
            ("mimo_ofdm_channel_estimation", "mimo.channel_estimation.nmse", 0.08),
            ("beamforming_precoding", "beamforming.spectral_efficiency_bps_hz", 4.2),
            ("wireless_localization", "localization.rmse_m", 0.5),
            ("aoa_estimation", "aoa.rmse_deg", 1.4),
            ("csi_compression_feedback", "csi_feedback.achieved_spectral_efficiency_bps_hz", 3.8),
            ("resource_allocation", "resource.theoretical_shannon_spectral_efficiency_bps_hz", 5.1),
        ]
        result = self._run_js(
            r"""
state.researchCatalog = { tasks: [], datasets: [], metrics: [] };
return __payload.cases.map(([taskId, metric, value], index) => {
  const recipe = {
    name: `Template ${index + 1}`,
    metadata: { research: { task: { id: taskId } } },
    steps: [{ id: "evaluation", op: "metrics.generic", params: {} }],
  };
  const summary = {
    run_id: `run-${index + 1}`,
    status: "completed",
    recipe,
    metrics: {},
    steps: [{ id: "evaluation", metrics: { [metric]: value }, outputs: {} }],
  };
  const row = {
    recipe: { key: `recipe-${index + 1}`, name: recipe.name, displayName: recipe.name, color: "#6aa5ff", recipe },
    summary,
    summaries: [summary],
  };
  const markup = resultOverviewPrimaryFigures([row]);
  return { taskId, hasFigure: markup.includes("<svg"), isEmpty: markup.includes("result-empty") };
});
""",
            payload={"cases": cases},
        )

        for row in result:
            with self.subTest(task_id=row["taskId"]):
                self.assertTrue(row["hasFigure"], row)
                self.assertFalse(row["isEmpty"], row)


if __name__ == "__main__":
    unittest.main()
