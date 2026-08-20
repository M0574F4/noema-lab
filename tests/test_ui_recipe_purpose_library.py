"""Purpose-tag and template-library UI regression coverage."""

import json
from pathlib import Path
import subprocess
import unittest

from noema_lab.core.recipes import recipe_from_dict
from noema_lab.ui.server import _list_recipes, _recipe_purpose_summary


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"


class UiRecipePurposeAndTemplateLibraryTests(unittest.TestCase):
    def test_template_replacement_uses_the_explicit_button_without_an_extra_confirmation(self):
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function bindRecipeLibraryDialog()")
        end = source.index("function selectRecipeLibraryEntry", start)
        dialog_binding = source[start:end]

        self.assertIn("openRecipeFromLibrary(key, inNewTab", dialog_binding)
        self.assertNotIn("confirmRecipePipelineReplacement", dialog_binding)

    def _run_js(self, body: str):
        script = r"""
const fs = require("fs");
const vm = require("vm");
let source = fs.readFileSync(__APP_PATH__, "utf8");
source = source.replace(/\ninit\(\);\s*$/, "\n");
const sandbox = {
  console,
  localStorage: { getItem: () => null, setItem: () => {} },
  document: { documentElement: { dataset: {} }, getElementById: () => null },
  window: { CSS: null },
  setTimeout,
  clearTimeout,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const body = __BODY__;
const promise = vm.runInContext(source + "\n;(async () => {\n" + body + "\n})()", sandbox);
Promise.resolve(promise).then(
  (result) => process.stdout.write(JSON.stringify(result)),
  (error) => { console.error(error); process.exit(1); }
);
"""
        script = script.replace("__APP_PATH__", json.dumps(str(APP_JS)))
        script = script.replace("__BODY__", json.dumps(body))
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_recipe_endpoint_classifies_purpose_without_changing_topology(self):
        rows = {row["path"]: row for row in _list_recipes(ROOT)}

        self.assertEqual(
            rows["recipes/compressai_kodak_default.yaml"]["purpose"]["id"],
            "image_reconstruction",
        )
        self.assertEqual(
            rows["recipes/csi_feedback_sionna_train.yaml"]["purpose"]["id"],
            "csi_compression_feedback",
        )
        self.assertEqual(
            rows["recipes/task_classification_smoke.yaml"]["purpose"]["id"],
            "classification",
        )
        self.assertTrue(
            rows["recipes/compressai_kodak_default.yaml"]["purpose"]["source"]
        )
        self.assertFalse(
            rows["recipes/csi_feedback_sionna_train.yaml"]["run_readiness"]["runnable"]
        )
        self.assertEqual(
            rows["recipes/csi_feedback_sionna_train.yaml"]["run_readiness"]["role"],
            "training_starter",
        )
        self.assertEqual(
            rows["recipes/task_classification_smoke.yaml"]["run_readiness"]["role"],
            "contract_smoke",
        )
        self.assertTrue(
            rows["recipes/aoa_music_ula_baseline.yaml"]["run_readiness"]["runnable"]
        )

    def test_invalid_purpose_contract_does_not_make_recipe_unopenable(self):
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "purpose_mismatch",
                "metadata": {
                    "research": {"task": {"id": "classification"}}
                },
                "steps": [
                    {
                        "id": "evaluation",
                        "op": "metrics.image_reconstruction",
                    }
                ],
            }
        )

        purpose = _recipe_purpose_summary(recipe)

        self.assertEqual(purpose["id"], "classification")
        self.assertEqual(purpose["status"], "invalid")
        self.assertIn("metric graph implies", purpose["error"])
        self.assertEqual(recipe.steps[0].id, "evaluation")

    def test_purpose_is_read_only_and_task_switching_surface_is_absent(self):
        source = APP_JS.read_text(encoding="utf-8")
        for obsolete in (
            'data-task-select',
            'data-recipe-setting="task"',
            "(contract)",
            "switchRecipeTask",
            "bindTaskSelector",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, source)

        markup = self._run_js(
            r"""
state.researchCatalog = {
  tasks: [{ id: "classification", name: "Classification", kind: "task_success", modality: "generic" }],
  datasets: [],
  metrics: [],
};
state.executionProfiles = [];
state.suitesCatalog = { suites: [] };
const recipe = {
  schema_version: 1,
  name: "classification_recipe",
  execution_profile: { id: "custom", version: 1 },
  metadata: { research: { task: { id: "classification" } } },
  steps: [{ id: "evaluation", op: "metrics.classification", inputs: {}, params: {} }],
};
return {
  purposeChip: recipeConfiguratorPurposeChipMarkup(recipe),
  contract: recipeContractSettingsMarkup(recipe),
};
"""
        )

        self.assertIn("Classification", markup["purposeChip"])
        self.assertIn("recipe-purpose-chip", markup["purposeChip"])
        self.assertNotIn("Overview", markup["purposeChip"])
        self.assertNotIn("classification_recipe", markup["purposeChip"])
        self.assertIn("recipe-purpose-setting", markup["contract"])
        self.assertIn("recipe-purpose-chip", markup["contract"])
        self.assertNotRegex(markup["contract"], r"<span>Purpose</span>\s*<input")
        self.assertIn("does not define the block graph", markup["contract"])
        self.assertNotIn("<select", markup["purposeChip"])

    def test_library_keeps_experimental_templates_openable_and_explains_run_readiness(self):
        result = self._run_js(
            r"""
const purpose = { id: "pilot_channel_estimation", label: "Pilot channel estimation", status: "valid" };
const training = {
  name: "training starter",
  path: "recipes/train.yaml",
  step_count: 4,
  execution_profile: { id: "task_evaluation", version: 1 },
  formalTemplate: { id: "train", status: "experimental", available: true, validation: { status: "valid" } },
  run_readiness: { runnable: false, role: "training_starter", issues: ["Export a Workbench contract first."] },
};
const setup = {
  name: "optional model",
  path: "recipes/optional.yaml",
  step_count: 3,
  execution_profile: { id: "task_evaluation", version: 1 },
  formalTemplate: { id: "optional", status: "experimental", available: true, validation: { status: "valid" } },
  run_readiness: { runnable: false, role: "runnable_template", issues: ["Install the optional model runtime."] },
};
return {
  experimentalOpenable: recipeLibraryEntryIsRunnable(training),
  benchmarkReady: recipeLibraryEntryBenchmarkReady(training),
  trainingMarkup: recipeLibraryItemMarkup(training, purpose),
  setupMarkup: recipeLibraryItemMarkup(setup, purpose),
  trainingSummary: recipeLibraryDialogSummaryMarkup(training),
  setupSummary: recipeLibraryDialogSummaryMarkup(setup),
};
"""
        )

        self.assertTrue(result["experimentalOpenable"])
        self.assertFalse(result["benchmarkReady"])
        self.assertNotIn("<small>Export a Workbench contract first.</small>", result["trainingMarkup"])
        self.assertNotIn("<small>Install the optional model runtime.</small>", result["setupMarkup"])
        self.assertIn("Export a Workbench contract first.", result["trainingSummary"])
        self.assertIn("Install the optional model runtime.", result["setupSummary"])
        self.assertNotIn("recipes/train.yaml", result["trainingSummary"])
        self.assertNotIn("recipes/optional.yaml", result["setupSummary"])
        self.assertNotIn("recipe-library-item-state", result["trainingMarkup"])
        self.assertNotIn("recipe-library-item-state", result["setupMarkup"])

    def test_client_reports_conflicting_purpose_contract_and_keeps_custom_metrics(self):
        result = self._run_js(
            r"""
state.researchCatalog = {
  tasks: [{
    id: "classification", name: "Classification", kind: "task_success",
    modality: "generic", metrics: ["task.accuracy"],
  }],
  datasets: [],
  metrics: [
    { id: "custom.team_score", name: "Team score", status: "supported" },
    { id: "task.accuracy", name: "Accuracy", status: "supported" },
  ],
};
const recipe = {
  name: "purpose_contract",
  metadata: {
    task_id: "object_detection",
    research: {
      metrics: [{ id: "custom.team_score" }],
      task: {
        id: "classification", kind: "wrong_kind", modality: "generic",
        metrics: [{ id: "custom.task_contract_score" }],
      },
    },
    metric_specs: [{ id: "custom.legacy_score" }, { id: "custom.team_score" }],
  },
  steps: [{ id: "evaluation", op: "metrics.classification", inputs: {}, params: {} }],
};
return {
  issues: recipePurposeConfigurationIssues(recipe),
  metrics: researchSummaryForRecipe(recipe).metrics,
};
"""
        )

        self.assertTrue(
            any("declarations disagree" in issue for issue in result["issues"])
        )
        self.assertTrue(any("wrong_kind" in issue for issue in result["issues"]))
        self.assertEqual(
            result["metrics"], [
                "custom.team_score",
                "custom.task_contract_score",
                "custom.legacy_score",
                "task.accuracy",
            ]
        )

    def test_result_routing_uses_declared_benchmark_inferred_and_custom_purposes(self):
        result = self._run_js(
            r"""
const declaredSummary = {
  recipe: {
    metadata: { research: { task: { id: "classification" } } },
    steps: [],
  },
};
const benchmarkSummary = { recipe: { metadata: {}, steps: [] } };
const benchmarkRow = {
  benchmarkResult: { benchmark: { task: { id: "visual_question_answering" } } },
};
const inferredSummary = {
  recipe: {
    metadata: {},
    steps: [{ id: "evaluation", op: "metrics.detection" }],
  },
};
const customSummary = {
  recipe: {
    metadata: { research: { task: { id: "custom_team_purpose" } } },
    steps: [],
  },
};
const declaredMetric = primaryTaskMetric(
  { metrics: [{ step: "evaluation", metric: "task.accuracy", value: 0.81 }] },
  declaredSummary,
  {},
);
const benchmarkMetric = primaryTaskMetric(
  { metrics: [{ step: "evaluation", metric: "vqa.single_reference_exact_match", value: 0.72 }] },
  benchmarkSummary,
  benchmarkRow,
);
const customMetric = primaryTaskMetric(
  { metrics: [{ step: "evaluation", metric: "task.score", value: 0.64 }] },
  customSummary,
  {},
);
return {
  declared: summaryTaskId(declaredSummary, {}),
  declaredMetric: declaredMetric.label,
  benchmark: summaryTaskId(benchmarkSummary, benchmarkRow),
  benchmarkMetric: benchmarkMetric.label,
  inferred: summaryTaskId(inferredSummary, {}),
  custom: summaryTaskId(customSummary, {}),
  customMetric: customMetric.label,
  unspecified: summaryTaskId({ recipe: { metadata: {}, steps: [] } }, {}),
};
"""
        )

        self.assertEqual(
            result,
            {
                "declared": "classification",
                "declaredMetric": "Task accuracy",
                "benchmark": "visual_question_answering",
                "benchmarkMetric": "Single-reference exact match",
                "inferred": "object_detection",
                "custom": "custom_team_purpose",
                "customMetric": "Task score",
                "unspecified": "",
            },
        )

    def test_template_catalog_is_deduplicated_and_grouped_by_area_then_purpose(self):
        result = self._run_js(
            r"""
state.recipes = [];
state.researchCatalog = {
  research_areas: [
    { id: "semantic_goal_oriented_communication", name: "Semantic & goal-oriented communication", order: 10 },
    { id: "channel_estimation_feedback", name: "Channel estimation & CSI feedback", order: 20 },
  ],
  tasks: [
    { id: "image_reconstruction", name: "Image reconstruction", area_id: "semantic_goal_oriented_communication", kind: "reconstruction", modality: "image" },
    { id: "classification", name: "Classification", area_id: "semantic_goal_oriented_communication", kind: "task_success", modality: "generic" },
    { id: "csi_compression_feedback", name: "CSI compression and feedback", area_id: "channel_estimation_feedback", kind: "reconstruction", modality: "wireless" },
  ],
};
state.recipeFiles = [
  {
    path: "recipes/image.yaml", name: "Project image", description: "Project description", step_count: 7,
    purpose: { id: "image_reconstruction", kind: "reconstruction", modality: "image" },
    execution_profile: { id: "layered_digital", version: 1 },
  },
  {
    path: "recipes/csi.yaml", name: "CSI study", description: "CSI description", step_count: 5,
    purpose: { id: "csi_compression_feedback", kind: "reconstruction", modality: "wireless" },
    execution_profile: { id: "csi_feedback_downlink", version: 1 },
  },
  {
    path: "recipes/broken.yaml", name: "Broken", description: "invalid yaml", step_count: 0,
    status: "error", purpose: {},
  },
];
state.recipeTemplates = [
  {
    id: "image.alternate", task_id: "image_reconstruction", default: false, order: 1,
    status: "supported", available: true, validation: { status: "valid" },
    recipe_path: "recipes/image.yaml", label: "Image alternate", editor: "image",
    recipe: { name: "Packaged alternate", description: "Alternate starter" },
  },
  {
    id: "image.default", task_id: "image_reconstruction", default: true, order: 99,
    status: "supported", available: true, validation: { status: "valid" },
    recipe_path: "recipes/image.yaml", label: "Image starter", editor: "image",
    recipe: { name: "Packaged image", description: "Starter description" },
  },
  {
    id: "classification.unavailable", task_id: "classification", default: true,
    status: "supported", available: false,
    validation: { status: "unavailable", errors: ["Model adapter is not installed."] }, recipe_path: "recipes/classification.yaml",
    label: "Classification starter", editor: "task",
    recipe: { name: "Classification starter" },
  },
];
const rows = recipeLibraryRows();
const groups = recipeLibraryGroups();
const markup = groups.map((group, index) => recipeLibraryGroupMarkup(group, "", index === 0)).join("");
return {
  rowCount: rows.length,
  benchmarkStarterCount: rows.filter(recipeLibraryEntryCountsAsBenchmarkStarter).length,
  imageRows: rows.filter((row) => row.path === "recipes/image.yaml").length,
  imageKeys: rows.filter((row) => row.path === "recipes/image.yaml").map((row) => row.libraryKey),
  imageGroupKeys: groups
    .find((group) => group.id === "semantic_goal_oriented_communication")
    .purposes.find((purpose) => purpose.id === "image_reconstruction")
    .recipes.map((row) => row.libraryKey),
  semanticPurposeIds: groups
    .find((group) => group.id === "semantic_goal_oriented_communication")
    .purposes.map((purpose) => purpose.id),
  groupIds: groups.map((group) => group.id),
  markup,
};
"""
        )

        self.assertEqual(result["rowCount"], 5)
        self.assertEqual(result["benchmarkStarterCount"], 2)
        self.assertEqual(result["imageRows"], 2)
        self.assertEqual(
            result["imageKeys"],
            ["template:image.alternate", "template:image.default"],
        )
        self.assertEqual(
            result["imageGroupKeys"],
            ["template:image.default", "template:image.alternate"],
        )
        self.assertEqual(
            result["groupIds"],
            [
                "semantic_goal_oriented_communication",
                "channel_estimation_feedback",
                "unavailable",
            ],
        )
        self.assertEqual(
            result["semanticPurposeIds"],
            ["image_reconstruction", "classification"],
        )
        self.assertIn("<details", result["markup"])
        self.assertIn('class="recipe-library-group-summary"', result["markup"])
        self.assertIn('class="recipe-library-group-chevron"', result["markup"])
        self.assertNotIn("recipe-library-group-action", result["markup"])
        self.assertNotIn(">Show<", result["markup"])
        self.assertNotIn(">Hide<", result["markup"])
        self.assertIn('aria-label="2 templates"', result["markup"])
        self.assertIn('data-recipe-library-purpose="image_reconstruction"', result["markup"])
        self.assertIn("Semantic &amp; goal-oriented communication", result["markup"])
        self.assertNotIn("Workspace recipes", result["markup"])
        self.assertNotIn("recipe-library-item-state", result["markup"])
        self.assertIn('aria-disabled="true"', result["markup"])
        self.assertIn("Model adapter is not installed.", result["markup"])

        styles = STYLES_CSS.read_text(encoding="utf-8")
        self.assertIn("grid-auto-rows: max-content", styles)
        self.assertIn(".settings-dialog-body.recipe-library-settings-body", styles)
        self.assertIn("height: min(680px, calc(100vh - 148px))", styles)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto auto", styles)
        self.assertIn(".recipe-library-group[open] > summary", styles)
        self.assertIn(".recipe-library-group > summary:focus-visible", styles)
        self.assertNotIn(".recipe-library-group-action", styles)
        self.assertNotIn(".recipe-library-item-state", styles)
        self.assertIn(".recipe-library-purpose-groups", styles)
        self.assertIn(".recipe-library-purpose-heading", styles)

    def test_formal_template_uses_curated_label_instead_of_internal_recipe_name(self):
        result = self._run_js(
            r"""
state.recipeFiles = [{
  path: "recipes/internal_name.yaml", name: "internal_snake_case_name",
  description: "Project recipe", step_count: 3, purpose: { id: "resource_allocation" },
}];
state.recipeTemplates = [{
  id: "resource.default", task_id: "resource_allocation", default: true,
  status: "supported", available: true, validation: { status: "valid" },
  recipe_path: "recipes/internal_name.yaml", editor: "graph",
  label: "Resource allocation — equal-power baseline",
  recipe: { name: "internal_snake_case_name", description: "Project recipe" },
}];
return recipeLibraryRows()[0].name;
"""
        )

        self.assertEqual(result, "Resource allocation — equal-power baseline")

    def test_formal_template_replaces_pipeline_and_adopts_template_identity(self):
        result = self._run_js(
            r"""
state.operations = [];
state.trainedArtifacts = [];
state.recipeFiles = [{
  path: "recipes/image.yaml", name: "Image template", description: "Template graph", step_count: 1,
  purpose: { id: "image_reconstruction" }, execution_profile: { id: "layered_digital", version: 1 },
}];
state.recipeTemplates = [{
  id: "image.default", task_id: "image_reconstruction", default: true,
  status: "supported", available: true, validation: { status: "valid" },
  recipe_path: "recipes/image.yaml", editor: "image", label: "Image template",
  recipe: { name: "template_image", description: "Template graph" },
}];
state.editRecipe = {
  schema_version: 1,
  name: "authored_study",
  description: "Keep this description.",
  suite: { id: "user_suite" },
  metadata: { owner: "research-team", seed: 99, task_id: "classification" },
  external_contract: { retain: true },
  steps: [{ id: "old", op: "custom.old", inputs: {}, params: {} }],
};
state.editRecipeKey = "working:1";
state.selectedRecipe = { key: "working:1", recipe: state.editRecipe, templatePath: "recipes/old.yaml" };
state.selectedRecipeKey = "working:1";
let request = null;
api = async (path, options) => {
  request = { path, method: options.method, body: JSON.parse(options.body) };
  return { recipe: {
    schema_version: 1,
    name: "template_image",
    description: "Template graph",
    execution_profile: { id: "layered_digital", version: 1 },
    metadata: { task_id: "image_reconstruction", codec_profile: "compressai" },
    steps: [{ id: "data", op: "source.image_dataset", inputs: {}, params: {} }],
  } };
};
applyBuiltRecipe = (recipe) => {
  state.editRecipe = recipe;
  state.selectedRecipe.recipe = recipe;
};
setActiveView = () => {};
notify = () => {};
const opened = await openRecipeFromLibrary("recipes/image.yaml", false);
return {
  opened,
  request,
  recipe: state.editRecipe,
  templatePath: state.selectedRecipe.templatePath,
  templateId: state.selectedRecipe.templateId,
};
"""
        )

        self.assertTrue(result["opened"])
        self.assertEqual(
            result["request"],
            {
                "path": "/api/recipe-templates/instantiate",
                "method": "POST",
                "body": {"template_id": "image.default"},
            },
        )
        recipe = result["recipe"]
        self.assertEqual(recipe["name"], "template_image")
        self.assertEqual(recipe["description"], "Template graph")
        self.assertNotIn("suite", recipe)
        self.assertNotIn("owner", recipe["metadata"])
        self.assertNotIn("seed", recipe["metadata"])
        self.assertEqual(recipe["metadata"]["task_id"], "image_reconstruction")
        self.assertNotIn("external_contract", recipe)
        self.assertEqual([step["id"] for step in recipe["steps"]], ["data"])
        self.assertEqual(result["templatePath"], "recipes/image.yaml")
        self.assertEqual(result["templateId"], "image.default")

    def test_template_identity_is_stable_when_formal_templates_share_a_path(self):
        result = self._run_js(
            r"""
state.recipes = [];
state.recipeFiles = [{
  path: "recipes/shared.yaml", name: "Shared source", step_count: 1,
  purpose: { id: "classification" },
}];
state.recipeTemplates = [
  {
    id: "classification.default", task_id: "classification", default: true,
    status: "supported", available: true, validation: { status: "valid" },
    recipe_path: "recipes/shared.yaml", label: "Default classification",
  },
  {
    id: "classification.alternate", task_id: "classification", default: false,
    status: "supported", available: true, validation: { status: "valid" },
    recipe_path: "recipes/shared.yaml", label: "Alternate classification",
  },
];
let requestedTemplate = "";
api = async (_path, options) => {
  requestedTemplate = JSON.parse(options.body).template_id;
  return { recipe: { name: "alternate", metadata: {}, steps: [] } };
};
selectRecipe = async () => {};
setActiveView = () => {};
notify = () => {};
const opened = await openRecipeFromLibrary("template:classification.alternate", true);
return { opened, requestedTemplate };
"""
        )

        self.assertEqual(
            result,
            {"opened": True, "requestedTemplate": "classification.alternate"},
        )

    def test_pipeline_replacement_replaces_or_removes_template_provenance(self):
        result = self._run_js(
            r"""
const previous = {
  schema_version: 1,
  name: "authored",
  metadata: {
    owner: "research-team",
    template_provenance: { template_id: "template.old", source_digest: "old" },
  },
  steps: [{ id: "old", op: "old.op" }],
};
const formal = recipeWithExplicitTopologyReplacement(previous, {
  schema_version: 1,
  name: "new formal",
  metadata: {
    template_provenance: { template_id: "template.new", source_digest: "new" },
  },
  steps: [{ id: "formal", op: "formal.op" }],
}).recipe;
const project = recipeWithExplicitTopologyReplacement(previous, {
  schema_version: 1,
  name: "project recipe",
  metadata: { purpose_marker: "project" },
  steps: [{ id: "project", op: "project.op" }],
}).recipe;
return { formal: formal.metadata, project: project.metadata };
"""
        )

        self.assertEqual(
            result["formal"]["template_provenance"],
            {"template_id": "template.new", "source_digest": "new"},
        )
        self.assertEqual(result["formal"]["owner"], "research-team")
        self.assertNotIn("template_provenance", result["project"])
        self.assertEqual(result["project"]["owner"], "research-team")

    def test_uncataloged_project_recipe_is_visible_and_openable(self):
        result = self._run_js(
            r"""
state.recipes = [];
state.recipeTemplates = [];
state.recipeFiles = [{
  path: "recipes/custom.yaml", name: "Custom pipeline", description: "Project recipe", step_count: 1,
  purpose: { id: "custom_task" }, execution_profile: { id: "custom", version: 1 },
}];
state.editRecipe = null;
state.selectedRecipe = null;
state.selectedRecipeKey = null;
let requestCount = 0;
let requestedPath = "";
api = async (path) => {
  requestCount += 1;
  requestedPath = path;
  return {
    recipe: {
      schema_version: 1,
      name: "Custom pipeline",
      metadata: { research: { task: { id: "custom_task" } } },
      steps: [{ id: "source", op: "source.custom", params: {} }],
    },
  };
};
selectRecipe = async (recipe) => {
  state.selectedRecipe = recipe;
  state.selectedRecipeKey = recipe.key;
};
setActiveView = () => {};
notify = () => {};
const opened = await openRecipeFromLibrary("recipes/custom.yaml", true);
return {
  opened,
  requestCount,
  rowCount: recipeLibraryRows().length,
  requestedPath,
  selectedName: state.selectedRecipe && state.selectedRecipe.recipe.name,
};
"""
        )

        self.assertEqual(
            result,
            {
                "opened": True,
                "requestCount": 1,
                "rowCount": 1,
                "requestedPath": "/api/recipe?path=recipes%2Fcustom.yaml",
                "selectedName": "Custom pipeline",
            },
        )

    def test_overlapping_template_opens_keep_latest_selection(self):
        result = self._run_js(
            r"""
state.operations = [];
state.trainedArtifacts = [];
state.recipeTemplates = [
  {
    id: "text.default", task_id: "text_semantic_similarity", status: "supported",
    available: true, validation: { status: "valid" }, recipe_path: "recipes/text.yaml", label: "Text",
  },
  {
    id: "classification.default", task_id: "classification", status: "supported",
    available: true, validation: { status: "valid" }, recipe_path: "recipes/classification.yaml", label: "Classification",
  },
];
state.recipeFiles = [
  { path: "recipes/text.yaml", name: "Text", purpose: { id: "text_semantic_similarity" } },
  { path: "recipes/classification.yaml", name: "Classification", purpose: { id: "classification" } },
];
state.editRecipe = { name: "old", metadata: { task_id: "image_reconstruction" }, steps: [] };
state.editRecipeKey = "working:race";
state.selectedRecipe = { key: "working:race", recipe: state.editRecipe };
state.selectedRecipeKey = "working:race";
const pending = {};
api = (_path, options) => new Promise((resolve) => {
  pending[JSON.parse(options.body).template_id] = resolve;
});
applyBuiltRecipe = (recipe) => { state.editRecipe = recipe; state.selectedRecipe.recipe = recipe; };
setActiveView = () => {};
const notices = [];
notify = (message) => { notices.push(message); };

const first = openRecipeFromLibrary("template:text.default", false);
const second = openRecipeFromLibrary("template:classification.default", false);
pending["classification.default"]({
  recipe: { name: "classification", metadata: { task_id: "classification" }, steps: [{ id: "classification" }] },
});
const secondOpened = await second;
pending["text.default"]({
  recipe: { name: "text", metadata: { task_id: "text_semantic_similarity" }, steps: [{ id: "text" }] },
});
const firstOpened = await first;
return {
  firstOpened,
  secondOpened,
  purpose: state.editRecipe.metadata.task_id,
  steps: state.editRecipe.steps.map((step) => step.id),
  notices,
};
"""
        )

        self.assertFalse(result["firstOpened"])
        self.assertTrue(result["secondOpened"])
        self.assertEqual(result["purpose"], "classification")
        self.assertEqual(result["steps"], ["classification"])
        self.assertEqual(len(result["notices"]), 1)

    def test_template_response_does_not_replace_another_recipe_tab(self):
        result = self._run_js(
            r"""
state.recipeTemplates = [{
  id: "text.default", task_id: "text_semantic_similarity", status: "supported",
  available: true, validation: { status: "valid" }, recipe_path: "recipes/text.yaml", label: "Text",
}];
state.recipeFiles = [{ path: "recipes/text.yaml", name: "Text", purpose: { id: "text_semantic_similarity" } }];
const recipeA = { name: "recipe_a", metadata: {}, steps: [] };
const recipeB = { name: "recipe_b", metadata: {}, steps: [] };
const tabA = { key: "working:a", recipe: recipeA };
const tabB = { key: "working:b", recipe: recipeB };
state.editRecipe = recipeA;
state.editRecipeKey = tabA.key;
state.selectedRecipe = tabA;
state.selectedRecipeKey = tabA.key;
let resolveRequest;
api = () => new Promise((resolve) => { resolveRequest = resolve; });
let replacements = 0;
applyBuiltRecipe = () => { replacements += 1; };
notify = () => {};
setActiveView = () => {};

const opening = openRecipeFromLibrary("template:text.default", false);
state.selectedRecipe = tabB;
state.selectedRecipeKey = tabB.key;
state.editRecipe = recipeB;
state.editRecipeKey = tabB.key;
resolveRequest({ recipe: { name: "text", metadata: {}, steps: [] } });
const opened = await opening;
return { opened, replacements, selectedKey: state.selectedRecipeKey, recipeName: state.editRecipe.name };
"""
        )

        self.assertEqual(
            result,
            {
                "opened": False,
                "replacements": 0,
                "selectedKey": "working:b",
                "recipeName": "recipe_b",
            },
        )

    def test_closing_template_library_invalidates_pending_replacement(self):
        result = self._run_js(
            r"""
state.recipeTemplates = [{
  id: "text.default", task_id: "text_semantic_similarity", status: "supported",
  available: true, validation: { status: "valid" }, recipe_path: "recipes/text.yaml", label: "Text",
}];
state.recipeFiles = [{ path: "recipes/text.yaml", name: "Text", purpose: { id: "text_semantic_similarity" } }];
state.editRecipe = { name: "before", metadata: {}, steps: [] };
state.selectedRecipe = { key: "working:close", recipe: state.editRecipe };
state.selectedRecipeKey = "working:close";
state.settingsSection = "recipe-library";
recipeLibraryDialogSession = 7;
let resolveRequest;
api = () => new Promise((resolve) => { resolveRequest = resolve; });
let replacements = 0;
applyBuiltRecipe = () => { replacements += 1; };
notify = () => {};
setActiveView = () => {};
renderRecipeControls = () => {};
els.settingsOverlay = { hidden: false };
const opening = openRecipeFromLibrary("template:text.default", false, { dialogSession: 7 });
closeSettingsDialog();
resolveRequest({ recipe: { name: "text", metadata: {}, steps: [] } });
const opened = await opening;
return { opened, replacements, section: state.settingsSection, overlayHidden: els.settingsOverlay.hidden };
"""
        )

        self.assertEqual(
            result,
            {
                "opened": False,
                "replacements": 0,
                "section": None,
                "overlayHidden": True,
            },
        )

    def test_blocks_routing_is_invariant_to_catalog_editor_preference(self):
        result = self._run_js(
            r"""
state.editRecipe = { name: "ambiguous_editor_recipe", metadata: {}, steps: [] };
state.editRecipeKey = "working:editor";
state.selectedRecipe = {
  key: "working:editor", recipe: state.editRecipe, editorPreference: "text", preserveTopology: true,
};
let route = "";
findStep = (stepId) => stepId ? { id: stepId } : null;
decorateRecipeConfigurator = () => {};
renderRecipeBlocksConfigurator = () => { route = "blocks"; };

state.selectedNodeId = null;
renderUnifiedConfigurator();
const defaultRoute = route;
route = "";
state.selectedNodeId = "sender";
renderUnifiedConfigurator();
return { defaultRoute, selectedRoute: route };
"""
        )

        self.assertEqual(
            result,
            {"defaultRoute": "blocks", "selectedRoute": "blocks"},
        )

    def test_initial_image_override_uses_catalog_binding(self):
        result = self._run_js(
            r"""
state.recipeTemplates = [{
  id: "semantic_comm.image.default", task_id: "image_reconstruction", default: true,
  status: "supported", available: true, validation: { status: "valid" },
  editor_bindings: {
    image_ids: { step_id: "renamed_dataset", op: "source.image_dataset", param: "selected_images" },
  },
}];
let request = null;
api = async (path, options) => {
  request = { path, body: JSON.parse(options.body) };
  return { recipe: { name: "image", metadata: {}, steps: [] } };
};
await createInitialWorkingRecipe();
return { request, preserveTopology: state.recipes[0].preserveTopology, templateId: state.recipes[0].templateId };
"""
        )

        self.assertEqual(result["request"]["path"], "/api/recipe-templates/instantiate")
        self.assertEqual(
            result["request"]["body"],
            {
                "template_id": "semantic_comm.image.default",
                "overrides": {
                    "step_params": {
                        "renamed_dataset": {"selected_images": "kodim01"}
                    }
                },
            },
        )
        self.assertTrue(result["preserveTopology"])
        self.assertEqual(result["templateId"], "semantic_comm.image.default")

    def test_initial_recipe_prefers_dependency_light_text_default(self):
        result = self._run_js(
            r"""
state.recipeTemplates = [
  {
    id: "semantic_comm.image.default", task_id: "image_reconstruction", default: true,
    status: "supported", available: true, validation: { status: "valid" },
  },
  {
    id: "semantic_comm.text.default", task_id: "text_semantic_similarity", default: true,
    status: "supported", available: true, validation: { status: "valid" },
  },
];
let request = null;
api = async (path, options) => {
  request = { path, body: JSON.parse(options.body) };
  return { recipe: { name: "text", metadata: {}, steps: [] } };
};
await createInitialWorkingRecipe();
return { request, templateId: state.recipes[0].templateId };
"""
        )

        self.assertEqual(
            result["request"]["body"],
            {"template_id": "semantic_comm.text.default"},
        )
        self.assertEqual(result["templateId"], "semantic_comm.text.default")


if __name__ == "__main__":
    unittest.main()
