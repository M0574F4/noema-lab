from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for recipe matrix UI tests")
class UiRecipeMatrixTests(unittest.TestCase):
    def _run_node(self, body: str):
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.replace(/\\ninit\\(\\);\\s*$/, "\\n");
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
            const promise = vm.runInContext(source + "\\n;(async () => {{\\n" + {json.dumps(body)} + "\\n}})()", sandbox);
            Promise.resolve(promise).then(
              (result) => process.stdout.write(JSON.stringify(result)),
              (error) => {{ console.error(error); process.exit(1); }}
            );
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
        return json.loads(completed.stdout)

    def test_setting_two_sweeps_writes_one_typed_canonical_matrix(self):
        result = self._run_node(
            r"""
state.editRecipe = {
  schema_version: 1,
  name: "two_dimensions",
  metadata: { sweeps: { "old.value": "1,2" } },
  steps: [
    { id: "wireless_channel", op: "wireless.channel", params: { snr_db: 0 } },
    { id: "sender", op: "model.jpeg_encode", params: { quality: 50 } },
  ],
};
setRecipeSweep("wireless_channel.snr_db", "-4,0,4");
setRecipeSweep("sender.quality", "50,75");
return state.editRecipe.metadata;
"""
        )

        self.assertNotIn("sweeps", result)
        self.assertNotIn("ui_sweeps", result)
        self.assertEqual(
            result["matrix"]["dimensions"],
            {"wireless_channel.snr_db": [-4, 0, 4], "sender.quality": [50, 75]},
        )
        self.assertEqual(
            result["matrix"]["step_params"],
            {
                "wireless_channel": {
                    "snr_db": {"matrix": "wireless_channel.snr_db"}
                },
                "sender": {"quality": {"matrix": "sender.quality"}},
            },
        )

    def test_resource_exhaustion_is_terminal_and_rendered_as_memory_failure(self):
        result = self._run_node(
            r"""
const graph = { nodes: [{ id: "source" }, { id: "channel" }] };
const job = {
  status: "resource_exhausted",
  events: [{ kind: "step_started", step_id: "source" }],
};
return {
  terminal: isTerminalJobStatus(job.status),
  label: logStatusLabel(job.status),
  node: nodeStatusFromJob(graph.nodes[0], graph, job, false),
  timedOutTerminal: isTerminalJobStatus("timed_out"),
  timedOutLabel: logStatusLabel("timed_out"),
  timedOutBadge: resultStatusBadge("timed_out"),
};
"""
        )

        self.assertTrue(result["terminal"])
        self.assertIn("memory protection", result["label"])
        self.assertEqual(result["node"], {"kind": "failed", "label": "Failed"})
        self.assertTrue(result["timedOutTerminal"])
        self.assertIn("deadline", result["timedOutLabel"])
        self.assertEqual(result["timedOutBadge"], {"className": "failed", "label": "timed out"})

    def test_run_job_event_updates_are_cursor_based_deduplicated_and_bounded(self):
        result = self._run_node(
            r"""
const previous = [
  { seq: 0, kind: "local", time: "10:00:00", message: "Starting default" },
  { seq: 1, kind: "queued", message: "queued" },
  { seq: 2, kind: "step_started", message: "source" },
];
const incoming = [
  { seq: 2, kind: "step_started", message: "source updated" },
  { seq: 3, kind: "step_completed", message: "source complete" },
];
const merged = mergeRunJobEvents(previous, incoming);
const oversized = mergeRunJobEvents([], Array.from(
  { length: RUN_JOB_EVENT_RETENTION_LIMIT + 25 },
  (_, index) => ({ seq: index + 1, kind: "event", message: String(index + 1) }),
));
let requested = "";
state.runJobs.recipe = { job_id: "job-1", status: "running", events: merged };
api = async (path) => {
  requested = path;
  return { job_id: "job-1", status: "running", events: [] };
};
await pollRunJobWithRetry({ key: "recipe" }, { job_id: "job-1" }, { label: "default" });
return {
  mergedSeqs: merged.map((event) => event.seq),
  mergedMessages: merged.map((event) => event.message),
  cursor: runJobEventCursor({ events: merged }),
  retained: oversized.length,
  firstRetained: oversized[0].seq,
  requested,
};
"""
        )

        self.assertEqual(result["mergedSeqs"], [0, 1, 2, 3])
        self.assertEqual(result["mergedMessages"][-2:], ["source updated", "source complete"])
        self.assertEqual(result["cursor"], 3)
        self.assertEqual(result["retained"], 750)
        self.assertEqual(result["firstRetained"], 26)
        self.assertIn("after_seq=3", result["requested"])
        self.assertIn("event_limit=500", result["requested"])

    def test_settings_matrix_mutations_are_atomic_canonical_and_bounded(self):
        result = self._run_node(
            r"""
state.operations = [{
  id: "test.adjust",
  name: "Adjust",
  status: "implemented",
  input_kinds: {},
  optional_input_kinds: {},
  output_kinds: { value: "json" },
  params_schema: {
    type: "object",
    properties: { gain: { type: "number" } },
    additionalProperties: false,
  },
}];
state.editRecipe = {
  schema_version: 1,
  name: "matrix_settings",
  metadata: {},
  steps: [{ id: "adjust", op: "test.adjust", inputs: {}, params: { gain: 1 } }],
};
let commits = 0;
const notices = [];
commitRecipeSettingsEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};
notify = (message) => notices.push(message);

const added = storeRecipeMatrixFromSettings({
  dimensions: { gain: [1, 2] },
  step_params: { adjust: { gain: { matrix: "gain" } } },
});
const afterAdded = structuredCloneFallback(state.editRecipe.metadata.matrix);
const rejected = storeRecipeMatrixFromSettings({
  dimensions: { gain: [1, 1] },
  step_params: { adjust: { gain: { matrix: "gain" } } },
});
const afterRejected = structuredCloneFallback(state.editRecipe.metadata.matrix);
const removed = storeRecipeMatrixFromSettings({
  dimensions: { gain: [1, 2] },
  step_params: {},
});
return {
  added,
  rejected,
  removed,
  afterAdded,
  afterRejected,
  matrixPresentAfterRemoval: Boolean(state.editRecipe.metadata.matrix),
  commits,
  notices,
};
"""
        )

        self.assertTrue(result["added"])
        self.assertEqual(
            result["afterAdded"],
            {
                "dimensions": {"gain": [1, 2]},
                "step_params": {"adjust": {"gain": {"matrix": "gain"}}},
            },
        )
        self.assertFalse(result["rejected"])
        self.assertEqual(result["afterRejected"], result["afterAdded"])
        self.assertTrue(result["removed"])
        self.assertFalse(result["matrixPresentAfterRemoval"])
        self.assertEqual(result["commits"], 2)
        self.assertTrue(
            any("duplicate typed values" in message for message in result["notices"])
        )

    def test_settings_reject_capture_conflicts_malformed_markers_and_wrong_types(self):
        result = self._run_node(
            r"""
state.operations = [{
  id: "test.adjust",
  name: "Adjust",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
  params_schema: {
    type: "object",
    properties: { gain: { type: "number" } },
    additionalProperties: false,
  },
}];
state.editRecipe = {
  schema_version: 1,
  name: "matrix_guards",
  metadata: {},
  dataset_capture: { sweep: { "adjust.gain": [1, 2] } },
  steps: [{ id: "adjust", op: "test.adjust", inputs: {}, params: { gain: 1 } }],
};
const notices = [];
let commits = 0;
notify = (message) => notices.push(message);
refreshUniversalRecipeSettingsDialog = () => {};
commitRecipeSettingsEdit = () => { commits += 1; };

const conflict = storeRecipeMatrixFromSettings({
  dimensions: { axis: [1, 2] },
  step_params: { adjust: { gain: { matrix: "axis" } } },
});
delete state.editRecipe.dataset_capture.sweep;
const malformed = storeRecipeMatrixFromSettings({
  dimensions: { first: [1], second: [2] },
  step_params: {
    adjust: { gain: { bad: { matrix: "first", extra: true }, good: { matrix: "second" } } },
  },
});
const wrongType = storeRecipeMatrixFromSettings({
  dimensions: { axis: ["low", "high"] },
  step_params: { adjust: { gain: { matrix: "axis" } } },
});
return {
  conflict,
  malformed,
  wrongType,
  commits,
  notices,
  metadata: state.editRecipe.metadata,
};
"""
        )

        self.assertFalse(result["conflict"])
        self.assertFalse(result["malformed"])
        self.assertFalse(result["wrongType"])
        self.assertEqual(result["commits"], 0)
        self.assertEqual(result["metadata"], {})
        self.assertTrue(any("capture-only sweep" in item for item in result["notices"]))
        self.assertTrue(any("invalid matrix marker" in item for item in result["notices"]))
        self.assertTrue(any("value must be number" in item for item in result["notices"]))

    def test_legacy_singleton_and_prototype_named_step_survive_ui_matrix_conversion(self):
        result = self._run_node(
            r"""
state.operations = [{
  id: "test.adjust",
  name: "Adjust",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
  params_schema: {
    type: "object",
    properties: { gain: { type: "number" } },
    additionalProperties: false,
  },
}, {
  id: "test.mode",
  name: "Mode",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
  params_schema: {
    type: "object",
    properties: { mode: { type: "string" } },
    additionalProperties: false,
  },
}];
state.editRecipe = JSON.parse(`{
  "schema_version": 1,
  "name": "legacy_and_safe_keys",
  "metadata": {"sweeps": {"adjust.gain": "5", "choice.mode": ["a", "b"]}},
  "steps": [
    {"id":"adjust","op":"test.adjust","inputs":{},"params":{"gain":5}},
    {"id":"choice","op":"test.mode","inputs":{},"params":{"mode":"a"}}
  ]
}`);
commitRecipeSettingsEdit = () => {};
refreshUniversalRecipeSettingsDialog = () => {};
notify = () => {};
const editable = editableRecipeMatrix();
const stored = storeRecipeMatrixFromSettings(editable);
const legacyMetadata = structuredCloneFallback(state.editRecipe.metadata);

state.editRecipe = JSON.parse(`{
  "schema_version": 1,
  "name": "prototype_step",
  "metadata": {},
  "steps": [{"id":"__proto__","op":"test.adjust","inputs":{},"params":{"gain":1}}]
}`);
setRecipeSweep("__proto__.gain", "1,2");
return {
  editable,
  stored,
  legacyMetadata,
  migrated: editableRecipeMatrix(),
  prototypeMatrixJson: JSON.stringify(state.editRecipe.metadata.matrix),
};
"""
        )

        self.assertEqual(
            result["editable"]["dimensions"],
            {"adjust.gain": [5], "choice.mode": ["a", "b"]},
        )
        self.assertTrue(result["stored"])
        self.assertNotIn("sweeps", result["legacyMetadata"])
        self.assertEqual(
            result["legacyMetadata"]["matrix"]["dimensions"],
            {"adjust.gain": [5], "choice.mode": ["a", "b"]},
        )
        self.assertEqual(result["migrated"]["step_params"]["__proto__"]["gain"], {"matrix": "__proto__.gain"})
        self.assertIn('"__proto__"', result["prototypeMatrixJson"])

    def test_linked_setting_variant_removal_preserves_unrelated_legacy_singleton(self):
        result = self._run_node(
            r"""
state.editRecipe = {
  schema_version: 1,
  name: "legacy_linked",
  metadata: {
    sweeps: {
      "first.runtime": ["auto", "native"],
      "second.gain": [5],
    },
  },
  steps: [
    { id: "first", op: "test.first", params: { runtime: "auto" } },
    { id: "second", op: "test.second", params: { gain: 5 } },
  ],
};
removeRecipeParameterVariants(state.editRecipe, "first", "runtime");
return state.editRecipe.metadata;
"""
        )

        self.assertEqual(
            result["matrix"],
            {
                "dimensions": {"second.gain": [5]},
                "step_params": {
                    "second": {"gain": {"matrix": "second.gain"}}
                },
            },
        )

    def test_step_rename_and_remove_keep_matrix_bindings_in_sync(self):
        result = self._run_node(
            r"""
state.editRecipe = {
  schema_version: 1,
  name: "rename_dimensions",
  metadata: {
    matrix: {
      dimensions: { "first.x": [1, 2], "second.y": [3, 4] },
      step_params: {
        first: { x: { matrix: "first.x" } },
        second: { y: { matrix: "second.y" } },
      },
    },
  },
  steps: [
    { id: "first", op: "test.first", inputs: {}, params: { x: 1 } },
    { id: "second", op: "test.second", inputs: {}, params: { y: 3 } },
  ],
};
state.positions = {};
state.busy = false;
commitRecipeTopologyEdit = () => {};
notify = () => {};
renderRecipeControls = () => {};
renameRecipeStep("first", "renamed");
const afterRename = structuredCloneFallback(state.editRecipe.metadata.matrix);
removeRecipeStep("second");
return { afterRename, afterRemove: state.editRecipe.metadata.matrix };
"""
        )

        self.assertEqual(
            result["afterRename"]["dimensions"],
            {"first.x": [1, 2], "second.y": [3, 4]},
        )
        self.assertEqual(
            result["afterRename"]["step_params"]["renamed"]["x"],
            {"matrix": "first.x"},
        )
        self.assertEqual(
            result["afterRemove"],
            {
                "dimensions": {"first.x": [1, 2]},
                "step_params": {
                    "renamed": {"x": {"matrix": "first.x"}}
                },
            },
        )

    def test_unrelated_edit_preserves_arbitrary_ids_recursive_templates_and_typed_values(self):
        result = self._run_node(
            r"""
state.editRecipe = {
  schema_version: 1,
  name: "preserve_matrix",
  metadata: {
    matrix: {
      dimensions: {
        snr: [1, 2],
        opaque_choice: [{ mode: "a", weights: [1, 2] }],
      },
      step_params: {
        first: { x: { matrix: "snr" } },
        second: { nested: { choice: { matrix: "opaque_choice" } } },
      },
    },
  },
  steps: [
    { id: "first", op: "test.first", params: { x: 1 } },
    { id: "second", op: "test.second", params: { nested: {} } },
    { id: "third", op: "test.third", params: { z: 5 } },
  ],
};
const displayed = sweepSpecForTarget(state.editRecipe, "first.x");
setRecipeSweep("third.z", "5,6");
const emptyCanonical = {
  metadata: { matrix: {}, sweeps: { "third.z": "9,10" } },
  steps: state.editRecipe.steps,
};
return {
  displayed,
  matrix: state.editRecipe.metadata.matrix,
  emptyCanonicalSpecs: recipeSweepSpecs(emptyCanonical),
};
"""
        )

        self.assertEqual(result["displayed"], "1,2")
        self.assertEqual(
            result["matrix"]["dimensions"],
            {
                "snr": [1, 2],
                "opaque_choice": [{"mode": "a", "weights": [1, 2]}],
                "third.z": [5, 6],
            },
        )
        self.assertEqual(
            result["matrix"]["step_params"]["second"],
            {"nested": {"choice": {"matrix": "opaque_choice"}}},
        )
        self.assertEqual(result["emptyCanonicalSpecs"], {})

    def test_shared_dimension_survives_one_binding_removal_and_is_garbage_collected_last(self):
        result = self._run_node(
            r"""
state.editRecipe = {
  schema_version: 1,
  name: "shared_dimension",
  metadata: {
    matrix: {
      dimensions: { shared: [1, 2] },
      step_params: {
        first: { x: { matrix: "shared" } },
        second: { y: { matrix: "shared" } },
      },
    },
  },
  steps: [
    { id: "first", op: "test.first", inputs: {}, params: { x: 1 } },
    { id: "second", op: "test.second", inputs: {}, params: { y: 1 } },
  ],
};
state.positions = {};
state.busy = false;
commitRecipeTopologyEdit = () => {};
notify = () => {};
renderRecipeControls = () => {};
setRecipeSweep("first.x", "7,8");
renameRecipeStep("first", "renamed");
removeRecipeSweep("renamed.x");
const afterFirstRemoval = structuredCloneFallback(state.editRecipe.metadata.matrix);
removeRecipeStep("second");
return {
  afterFirstRemoval,
  hasMatrixAfterLastRemoval: Object.prototype.hasOwnProperty.call(state.editRecipe.metadata, "matrix"),
};
"""
        )

        self.assertEqual(result["afterFirstRemoval"]["dimensions"], {"shared": [7, 8]})
        self.assertNotIn("renamed", result["afterFirstRemoval"]["step_params"])
        self.assertEqual(
            result["afterFirstRemoval"]["step_params"]["second"]["y"],
            {"matrix": "shared"},
        )
        self.assertFalse(result["hasMatrixAfterLastRemoval"])

    def test_run_variants_trust_backend_bindings_even_for_legacy_looking_dimension_id(self):
        result = self._run_node(
            r"""
const base = {
  schema_version: 1,
  name: "backend_authoritative",
  metadata: {
    matrix: {
      dimensions: { "codecParams.encoder.model": ["choice_a", "choice_b"] },
      step_params: {
        channel: { mode: { matrix: "codecParams.encoder.model" } },
      },
    },
  },
  steps: [
    { id: "sender", op: "model.compressai_encode", params: { model: "must_stay" } },
    { id: "channel", op: "test.channel", params: { mode: "choice_a" } },
  ],
};
api = async () => ({
  expanded_count: 1,
  recipes: [{
    ...structuredCloneFallback(base),
    name: "backend_concrete_name",
    metadata: {
      matrix_selection: { "codecParams.encoder.model": "choice_b" },
      matrix_index: 0,
      matrix_variant_id: "mxv1-backend-stable-id",
      backend_marker: { keep: ["exactly", 7] },
    },
    steps: [
      { id: "sender", op: "model.compressai_encode", params: { model: "must_stay", quality: 91 } },
      { id: "channel", op: "test.channel", params: { mode: "choice_b" } },
    ],
  }],
});
const variants = await recipeRunVariants({ recipe: base });
return variants[0];
"""
        )

        concrete = result["recipe"]
        sender = next(step for step in concrete["steps"] if step["id"] == "sender")
        channel = next(step for step in concrete["steps"] if step["id"] == "channel")
        self.assertEqual(concrete["name"], "backend_concrete_name")
        self.assertEqual(sender["params"]["model"], "must_stay")
        self.assertEqual(sender["params"]["quality"], 91)
        self.assertEqual(channel["params"]["mode"], "choice_b")
        self.assertEqual(
            concrete["metadata"],
            {
                "matrix_selection": {
                    "codecParams.encoder.model": "choice_b"
                },
                "matrix_index": 0,
                "matrix_variant_id": "mxv1-backend-stable-id",
                "backend_marker": {"keep": ["exactly", 7]},
            },
        )
        self.assertEqual(result["label"], "model choice b")

    def test_run_variants_keep_backend_order(self):
        result = self._run_node(
            r"""
const base = {
  schema_version: 1,
  name: "authored",
  metadata: {
    matrix: {
      dimensions: { axis: ["first", "second"] },
      step_params: { node: { choice: { matrix: "axis" } } },
    },
  },
  steps: [{ id: "node", op: "test.node", params: { choice: "first" } }],
};
api = async () => ({
  recipes: [
    {
      ...structuredCloneFallback(base),
      name: "backend_first",
      metadata: { matrix_selection: { axis: "second" }, matrix_index: 1, matrix_variant_id: "mxv1-second" },
      steps: [{ id: "node", op: "test.node", params: { choice: "second" } }],
    },
    {
      ...structuredCloneFallback(base),
      name: "backend_second",
      metadata: { matrix_selection: { axis: "first" }, matrix_index: 0, matrix_variant_id: "mxv1-first" },
      steps: [{ id: "node", op: "test.node", params: { choice: "first" } }],
    },
  ],
});
const variants = await recipeRunVariants({ recipe: base });
return variants.map((variant) => ({
  name: variant.recipe.name,
  choice: variant.recipe.steps[0].params.choice,
  index: variant.recipe.metadata.matrix_index,
}));
"""
        )

        self.assertEqual(
            result,
            [
                {"name": "backend_first", "choice": "second", "index": 1},
                {"name": "backend_second", "choice": "first", "index": 0},
            ],
        )

    def test_variant_labels_prefer_canonical_metadata_with_read_only_legacy_fallback(self):
        result = self._run_node(
            r"""
const canonicalSelection = {
  sweep_label: "wrong transient label",
  recipe: {
    metadata: {
      matrix_selection: { "channel.snr_db": 4 },
      matrix_index: 8,
      matrix_variant_id: "mxv1-selection",
      matrix_label: "wrong stored label",
      sweep_values: { "channel.snr_db": -20 },
    },
  },
};
const canonicalIndex = {
  recipe: { metadata: { matrix_selection: {}, matrix_index: 2, matrix_variant_id: "mxv1-index" } },
};
const canonicalId = {
  recipe: { metadata: { matrix_selection: {}, matrix_variant_id: "mxv1-id-only" } },
};
const legacy = {
  sweep_label: "legacy display",
  recipe: { metadata: { sweep_values: { "channel.snr_db": -3 } } },
};
return {
  canonicalSelection: resultVariantLabel(canonicalSelection, 11),
  canonicalIndex: resultVariantLabel(canonicalIndex, 11),
  canonicalId: resultVariantLabel(canonicalId, 11),
  legacy: resultVariantLabel(legacy, 11),
  emptyCanonicalSuppressesLegacySelection: resultMatrixSelection({
    matrix_selection: {},
    sweep_values: { "channel.snr_db": -30 },
  }),
};
"""
        )

        self.assertEqual(result["canonicalSelection"], "SNR 4 dB")
        self.assertEqual(result["canonicalIndex"], "variant 3")
        self.assertEqual(result["canonicalId"], "variant mxv1-id-only")
        self.assertEqual(result["legacy"], "legacy display")
        self.assertEqual(result["emptyCanonicalSuppressesLegacySelection"], {})

    def test_single_variant_run_does_not_decorate_backend_summary_with_sweep_fields(self):
        result = self._run_node(
            r"""
state.runSummaries = {};
state.runs = [];
const backendSummary = {
  run_id: "run-canonical",
  status: "completed",
  recipe_name: "backend_variant",
  recipe: {
    name: "backend_variant",
    metadata: {
      matrix_selection: { "channel.snr_db": 7 },
      matrix_index: 0,
      matrix_variant_id: "mxv1-canonical",
    },
    steps: [],
  },
};
api = async (path, options = {}) => {
  if (path === "/api/run-jobs" && options.method === "POST") {
    return { job_id: "job-canonical", status: "completed", run_id: "run-canonical" };
  }
  if (path === "/api/runs/run-canonical") return backendSummary;
  throw new Error(`unexpected API request: ${path}`);
};
updateRecipeJob = () => {};
const recipe = { lastError: null };
const variant = {
  recipe: backendSummary.recipe,
  label: "SNR 7 dB",
  value: 7,
};
await runSingleRecipeVariant(recipe, variant);
return state.runSummaries["run-canonical"];
"""
        )

        self.assertNotIn("sweep_label", result)
        self.assertNotIn("sweep_value", result)
        self.assertEqual(
            result["recipe"]["metadata"],
            {
                "matrix_selection": {"channel.snr_db": 7},
                "matrix_index": 0,
                "matrix_variant_id": "mxv1-canonical",
            },
        )

    def test_topology_preserving_builder_updates_only_changed_editor_dimension(self):
        result = self._run_node(
            r"""
const previous = {
  schema_version: 1,
  name: "preserved",
  execution_profile: { id: "custom", version: 1 },
  metadata: {
    ui_working_copy: true,
    matrix: {
      dimensions: { quality_axis: [50, 75], snr_axis: [0, 5] },
      step_params: {
        sender: { quality: { matrix: "quality_axis" } },
        wireless_channel: { snr_db: { matrix: "snr_axis" } },
      },
    },
  },
  steps: [
    { id: "sender", op: "model.jpeg_encode", inputs: {}, params: { quality: 50 } },
    { id: "wireless_channel", op: "wireless.channel", inputs: {}, params: { snr_db: 0 } },
  ],
};
const rebuilt = structuredCloneFallback(previous);
rebuilt.metadata = {
  matrix: {
    dimensions: {
      "codecParams.encoder.quality": [40, 60, 80],
      "channel.snr_db": [0, 5],
    },
    step_params: {
      sender: { quality: { matrix: "codecParams.encoder.quality" } },
      wireless_channel: { snr_db: { matrix: "channel.snr_db" } },
    },
  },
};
return recipeWithPreservedEditedTopology(previous, rebuilt).metadata.matrix;
"""
        )

        self.assertEqual(
            result["dimensions"],
            {"quality_axis": [40, 60, 80], "snr_axis": [0, 5]},
        )
        self.assertEqual(
            result["step_params"],
            {
                "sender": {"quality": {"matrix": "quality_axis"}},
                "wireless_channel": {"snr_db": {"matrix": "snr_axis"}},
            },
        )

    def test_builder_sweep_delta_preserves_unrelated_opaque_matrix_dimension(self):
        result = self._run_node(
            r"""
const previous = {
  schema_version: 1,
  name: "preserved_opaque_matrix",
  execution_profile: { id: "custom", version: 1 },
  metadata: {
    ui_working_copy: true,
    matrix: {
      dimensions: {
        opaque_axis: ["conservative", "aggressive"],
        snr_axis: [0, 5],
      },
      step_params: {
        sender: {
          policy: {
            mode: { matrix: "opaque_axis" },
            fixed: true,
          },
        },
        wireless_channel: { snr_db: { matrix: "snr_axis" } },
      },
    },
  },
  steps: [
    {
      id: "sender",
      op: "custom.sender",
      inputs: {},
      params: { policy: { mode: "conservative", fixed: true } },
    },
    {
      id: "wireless_channel",
      op: "wireless.channel",
      inputs: {},
      params: { snr_db: 0 },
    },
  ],
};
const baseline = {
  ...structuredCloneFallback(previous),
  metadata: {
    matrix: {
      dimensions: { "channel.snr_db": [0, 5] },
      step_params: {
        wireless_channel: { snr_db: { matrix: "channel.snr_db" } },
      },
    },
  },
};
const rebuilt = {
  ...structuredCloneFallback(baseline),
  metadata: {
    matrix: {
      dimensions: { "channel.snr_db": [1, 6] },
      step_params: {
        wireless_channel: { snr_db: { matrix: "channel.snr_db" } },
      },
    },
  },
};
return recipeWithPreservedEditedTopology(previous, rebuilt, baseline).metadata.matrix;
"""
        )

        self.assertEqual(
            result,
            {
                "dimensions": {
                    "opaque_axis": ["conservative", "aggressive"],
                    "snr_axis": [1, 6],
                },
                "step_params": {
                    "sender": {
                        "policy": {
                            "mode": {"matrix": "opaque_axis"},
                            "fixed": True,
                        }
                    },
                    "wireless_channel": {
                        "snr_db": {"matrix": "snr_axis"}
                    },
                },
            },
        )

    def test_catalog_owned_execution_blocks_cannot_be_replaced_by_ui_builder(self):
        result = self._run_node(
            r"""
const previous = {
  schema_version: 1,
  name: "catalog_recipe",
  metadata: { ui_working_copy: true },
  steps: [
    {
      id: "data",
      op: "source.image_dataset",
      inputs: {},
      params: { image_ids: "kodim01", catalog_extension: "keep" },
    },
    {
      id: "sender",
      op: "model.compressai_analysis_encode",
      inputs: { images: "data.images" },
      params: { quality: 3 },
    },
  ],
};
const rebuilt = {
  schema_version: 1,
  name: "catalog_recipe",
  metadata: {},
  steps: [
    {
      id: "data",
      op: "source.image_dataset",
      inputs: { unexpected: "other.value" },
      params: { image_ids: "kodim02" },
    },
    {
      id: "sender",
      op: "model.jpeg_encode",
      inputs: { images: "other.images" },
      params: { quality: 75 },
    },
    { id: "injected", op: "channel.identity_link", inputs: {}, params: {} },
  ],
};
return recipeWithPreservedEditedTopology(previous, rebuilt);
"""
        )

        self.assertEqual([step["id"] for step in result["steps"]], ["data", "sender"])
        self.assertEqual(result["steps"][0]["params"]["image_ids"], "kodim02")
        self.assertEqual(result["steps"][0]["params"]["catalog_extension"], "keep")
        self.assertEqual(result["steps"][0]["inputs"], {})
        self.assertEqual(result["steps"][1]["op"], "model.compressai_analysis_encode")
        self.assertEqual(result["steps"][1]["inputs"], {"images": "data.images"})
        self.assertEqual(result["steps"][1]["params"], {"quality": 3})
        self.assertFalse(any(key.startswith("ui_") for key in result["metadata"]))

    def test_builder_applies_only_params_changed_from_its_baseline(self):
        result = self._run_node(
            r"""
const previous = {
  schema_version: 1,
  name: "catalog_recipe",
  metadata: { ui_working_copy: true },
  steps: [{
    id: "sender",
    op: "model.codec",
    inputs: {},
    params: { quality: 3, catalog_tuning: 99 },
  }],
};
const baseline = {
  ...structuredCloneFallback(previous),
  metadata: {},
  steps: [{
    id: "sender",
    op: "model.codec",
    inputs: {},
    params: { quality: 3, catalog_tuning: 0 },
  }],
};
const rebuilt = {
  ...structuredCloneFallback(baseline),
  steps: [{
    id: "sender",
    op: "model.codec",
    inputs: {},
    params: { quality: 5, catalog_tuning: 0 },
  }],
};
return recipeWithPreservedEditedTopology(previous, rebuilt, baseline);
"""
        )

        self.assertEqual(
            result["steps"][0]["params"],
            {"quality": 5, "catalog_tuning": 99},
        )

    def test_ambiguous_editor_alias_preserves_independent_dimensions(self):
        result = self._run_node(
            r"""
const previous = {
  schema_version: 1,
  name: "independent_presets",
  execution_profile: { id: "custom", version: 1 },
  metadata: {
    codec_profile: "evc",
    ui_working_copy: true,
    matrix: {
      dimensions: {
        sender_presets: ["evc_ss_md", "evc_ll"],
        receiver_presets: ["evc_ml_md", "evc_lm_md"],
      },
      step_params: {
        sender: { checkpoint_preset: { matrix: "sender_presets" } },
        receiver: { checkpoint_preset: { matrix: "receiver_presets" } },
      },
    },
  },
  steps: [
    { id: "sender", op: "model.evc_encode", inputs: {}, params: { checkpoint_preset: "evc_ss_md" } },
    { id: "receiver", op: "model.evc_decode", inputs: {}, params: { checkpoint_preset: "evc_ml_md" } },
  ],
};
const rebuilt = structuredCloneFallback(previous);
rebuilt.metadata = {
  codec_profile: "evc",
  matrix: {
    dimensions: {
      "codecParams.encoder.checkpoint_preset": ["evc_ss_md", "evc_ll"],
    },
    step_params: {
      sender: { checkpoint_preset: { matrix: "codecParams.encoder.checkpoint_preset" } },
      receiver: { checkpoint_preset: { matrix: "codecParams.encoder.checkpoint_preset" } },
    },
  },
};
const projected = recipeSweepSpecs(previous);
const preserved = recipeWithPreservedEditedTopology(previous, rebuilt);
return {
  hasAmbiguousAlias: Object.prototype.hasOwnProperty.call(
    projected,
    "codecParams.encoder.checkpoint_preset"
  ),
  matrix: preserved.metadata.matrix,
};
"""
        )

        self.assertFalse(result["hasAmbiguousAlias"])
        self.assertEqual(
            result["matrix"],
            {
                "dimensions": {
                    "sender_presets": ["evc_ss_md", "evc_ll"],
                    "receiver_presets": ["evc_ml_md", "evc_lm_md"],
                },
                "step_params": {
                    "sender": {
                        "checkpoint_preset": {"matrix": "sender_presets"}
                    },
                    "receiver": {
                        "checkpoint_preset": {"matrix": "receiver_presets"}
                    },
                },
            },
        )

    def test_matrix_expansion_failure_is_contained_to_recipe_job(self):
        result = self._run_node(
            r"""
const recipe = {
  key: "working:bad-matrix",
  name: "bad_matrix",
  recipe: {
    schema_version: 1,
    name: "bad_matrix",
    metadata: {},
    steps: [],
  },
};
ensureRecipePayload = async () => {};
recipeUsesUiConfigurator = () => false;
recipeConfigurationIssues = () => [];
recipeRunVariants = async () => { throw new Error("matrix expands past the limit"); };
let localJob = null;
setLocalRecipeJob = (_recipe, status, message, severity) => {
  localJob = { status, message, severity };
};
renderRecipes = () => {};
renderRunLog = () => {};

const status = await runRecipeJob(recipe);
return {
  status,
  transientStatus: recipe.transientStatus,
  lastError: recipe.lastError,
  resultDirty: recipe.resultDirty,
  localJob,
};
"""
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["transientStatus"], "failed")
        self.assertEqual(result["lastError"], "matrix expands past the limit")
        self.assertTrue(result["resultDirty"])
        self.assertEqual(
            result["localJob"],
            {
                "status": "failed",
                "message": "Could not expand recipe matrix: matrix expands past the limit",
                "severity": "error",
            },
        )

    def test_run_all_continues_after_one_recipe_matrix_fails_to_expand(self):
        result = self._run_node(
            r"""
const bad = {
  key: "working:bad-matrix",
  name: "bad_matrix",
  recipe: { schema_version: 1, name: "bad_matrix", metadata: {}, steps: [] },
};
const good = {
  key: "working:good-matrix",
  name: "good_matrix",
  recipe: { schema_version: 1, name: "good_matrix", metadata: {}, steps: [] },
};
state.recipes = [bad, good];
state.graph = null;
state.selectedRecipe = null;
state.selectedRecipeKey = null;
state.editRecipe = null;
state.editRecipeKey = null;
state.resultFacet = "communication";
const expansionAttempts = [];
const variantRuns = [];
const notices = [];

ensureRecipePayload = async () => {};
recipeUsesUiConfigurator = () => false;
recipeConfigurationIssues = () => [];
recipeRunVariants = async (recipe) => {
  expansionAttempts.push(recipe.key);
  if (recipe.key === bad.key) throw new Error("matrix expands past the limit");
  return [{ recipe: structuredCloneFallback(recipe.recipe), label: "default", value: null, sweep: null }];
};
runSingleRecipeVariant = async (recipe) => {
  variantRuns.push(recipe.key);
  return { status: "completed", run_id: null };
};
focusRecipeInGraphForRun = async () => {};
loadAllRecipeResults = async () => {};
setBusy = () => {};
renderRunControls = () => {};
renderRunLog = () => {};
renderRecipes = () => {};
renderResultsDashboard = () => {};
notify = (message) => { notices.push(message); };

await runAllRecipes();
return {
  expansionAttempts,
  variantRuns,
  bad: {
    status: bad.transientStatus,
    error: bad.lastError,
    jobStatus: state.runJobs[bad.key].status,
    message: state.runJobs[bad.key].events[0].message,
  },
  good: {
    status: good.transientStatus,
    error: good.lastError,
    dirty: good.resultDirty,
  },
  notice: notices[notices.length - 1],
  resultFacet: state.resultFacet,
};
"""
        )

        self.assertEqual(
            result["expansionAttempts"],
            ["working:bad-matrix", "working:good-matrix"],
        )
        self.assertEqual(result["variantRuns"], ["working:good-matrix"])
        self.assertEqual(
            result["bad"],
            {
                "status": "failed",
                "error": "matrix expands past the limit",
                "jobStatus": "failed",
                "message": "Could not expand recipe matrix: matrix expands past the limit",
            },
        )
        self.assertEqual(
            result["good"],
            {"status": None, "error": None, "dirty": False},
        )
        self.assertEqual(result["resultFacet"], "overview")
        self.assertEqual(result["notice"], "ran 2 recipes: 1 completed, 1 failed")


if __name__ == "__main__":
    unittest.main()
