from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import unittest

from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"
IMAGE_RECIPE = ROOT / "recipes" / "compressai_kodak_default.yaml"
TEXT_RECIPE = ROOT / "recipes" / "text_semantic_utf8_clean.yaml"
CSI_RECIPE = ROOT / "recipes" / "csi_feedback_sionna_train.yaml"
RETRIEVAL_RECIPE = ROOT / "recipes" / "clip_retrieval_smoke.yaml"
GENERATION_RECIPE = ROOT / "recipes" / "diffusion_flickr8k_generation.yaml"

SETTINGS_SECTIONS = (
    "general",
    "pipeline",
    "contract",
    "variants",
    "advanced",
)
SETTINGS_FIELDS = (
    "name",
    "description",
    "execution_profile",
    "suite",
    "metadata.matrix",
    "metadata",
)


class _RecipeSettingControlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.controls: dict[str, tuple[str, dict[str, str | None]]] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        field = attributes.get("data-recipe-setting")
        if field:
            self.controls[field] = (tag, attributes)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for recipe settings UI tests")
class RecipeSettingsDialogUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_recipe = load_recipe(IMAGE_RECIPE).to_dict()
        cls.text_recipe = load_recipe(TEXT_RECIPE).to_dict()
        cls.csi_recipe = load_recipe(CSI_RECIPE).to_dict()
        cls.retrieval_recipe = load_recipe(RETRIEVAL_RECIPE).to_dict()
        cls.generation_recipe = load_recipe(GENERATION_RECIPE).to_dict()
        registry = build_registry()
        operation_ids = {
            step["op"]
            for step in cls.image_recipe["steps"]
        } | {
            step["op"]
            for step in cls.text_recipe["steps"]
        } | {
            "noise.source_image_perturbation",
            "modulation.digital_modulate",
            "demodulation.digital_demodulate",
            "wireless.channel",
            "metrics.bit_error_rate",
            "model.jpeg_encode",
            "model.jpeg_decode",
            "channel.payload_passthrough_encoder",
            "channel.payload_passthrough_decoder",
        }
        cls.operations = [
            registry.get(operation_id).describe()
            for operation_id in sorted(operation_ids)
        ]

    def _run_node(self, body: str, **payload: object) -> dict:
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
const payload = __PAYLOAD__;
const body = __BODY__;
const promise = vm.runInContext(
  source + "\n;globalThis.__payload = " + JSON.stringify(payload) +
  "\n;(async () => {\n" + body + "\n})()",
  sandbox,
);
Promise.resolve(promise).then(
  (result) => process.stdout.write(JSON.stringify(result)),
  (error) => { console.error(error); process.exit(1); },
);
"""
        script = script.replace("__APP_PATH__", json.dumps(str(APP_JS)))
        script = script.replace("__PAYLOAD__", json.dumps(payload))
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

    def _render_settings_for_recipes(self) -> dict:
        return self._run_node(
            r"""
state.researchCatalog = {
  tasks: [
    { id: "image_reconstruction", name: "Image reconstruction", modality: "image" },
    { id: "csi_compression_feedback", name: "CSI compression and feedback", modality: "wireless" },
    { id: "image_text_retrieval", name: "Image-text retrieval", modality: "multimodal" },
    { id: "image_generation", name: "Image generation", modality: "multimodal" },
    { id: "custom_task", name: "Custom task", modality: "mixed" },
  ],
  datasets: [],
  metrics: [],
};
state.executionProfiles = [
  { id: "layered_digital", version: 1, label: "Layered digital" },
  { id: "csi_feedback_downlink", version: 1, label: "CSI feedback downlink" },
  { id: "custom", version: 1, label: "Custom" },
];
state.suitesCatalog = { suites: [] };
state.recipeTemplates = [];
els.settingsTitle = { textContent: "" };
els.settingsSubtitle = { textContent: "" };
els.settingsBody = {
  innerHTML: "",
  querySelectorAll: () => [],
  querySelector: () => null,
};
els.settingsOverlay = { hidden: true };

function renderFor(recipe, key, templatePath) {
  const before = JSON.stringify(recipe);
  state.editRecipe = recipe;
  state.editRecipeKey = key;
  state.selectedRecipe = {
    key,
    recipe,
    templatePath,
    preserveTopology: true,
    workingCopy: true,
  };
  state.selectedRecipeKey = key;
  openSettingsDialog("recipe");
  return {
    title: els.settingsTitle.textContent,
    subtitle: els.settingsSubtitle.textContent,
    markup: els.settingsBody.innerHTML,
    section: state.settingsSection,
    visible: !els.settingsOverlay.hidden,
    unchanged: before === JSON.stringify(recipe),
  };
}

const image = renderFor(
  JSON.parse(JSON.stringify(__payload.image)),
  "image",
  "recipes/compressai_kodak_default.yaml",
);
const csi = renderFor(
  JSON.parse(JSON.stringify(__payload.csi)),
  "csi",
  "recipes/csi_feedback_sionna_train.yaml",
);
const retrieval = renderFor(
  JSON.parse(JSON.stringify(__payload.retrieval)),
  "retrieval",
  "recipes/clip_retrieval_smoke.yaml",
);
const generation = renderFor(
  JSON.parse(JSON.stringify(__payload.generation)),
  "generation",
  "recipes/diffusion_flickr8k_generation.yaml",
);
const custom = renderFor({
  schema_version: 1,
  name: "fully_visible_custom_recipe",
  description: "Every top-level authoring surface stays available.",
  execution_profile: { id: "custom", version: 1 },
  suite: { id: "custom_suite", name: "Custom suite", version: "v2" },
  metadata: {
    research: { task: { id: "custom_task", modality: "mixed" } },
    matrix: {
      dimensions: { "source.count": [8, 16] },
      step_params: { source: { count: { matrix: "source.count" } } },
    },
    extension_payload: { owner: "research-team" },
  },
  dataset_capture: {
    split: "validation",
    samples: 16,
    taps: [{ id: "bits", from: "source.bits" }],
  },
  steps: [{ id: "source", op: "source.random_bits", params: { bit_count: 8 } }],
}, "custom", "");
return { image, csi, retrieval, generation, custom };
""",
            image=self.image_recipe,
            csi=self.csi_recipe,
            retrieval=self.retrieval_recipe,
            generation=self.generation_recipe,
        )

    def test_one_universal_dialog_structure_is_used_for_every_recipe_kind(self) -> None:
        result = self._render_settings_for_recipes()
        styles = STYLES_CSS.read_text(encoding="utf-8")
        self.assertIn('.recipe-settings-nav button[aria-current="location"]', styles)
        self.assertEqual(
            len({rendered["subtitle"] for rendered in result.values()}),
            1,
        )

        for rendered in result.values():
            self.assertEqual(rendered["title"], "Recipe Settings")
            self.assertEqual(rendered["section"], "recipe")
            self.assertTrue(rendered["visible"])
            self.assertTrue(rendered["unchanged"])
            markup = rendered["markup"]
            positions = [
                markup.index(f'data-recipe-settings-section="{section}"')
                for section in SETTINGS_SECTIONS
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(markup.count('<header class="config-section-header">'), len(SETTINGS_SECTIONS))
            for section in SETTINGS_SECTIONS:
                self.assertEqual(
                    markup.count(f'data-recipe-settings-section="{section}"'),
                    1,
                )
                self.assertIn(
                    f'aria-labelledby="recipe-settings-{section}-title"',
                    markup,
                )
                self.assertIn(f'id="recipe-settings-{section}-title"', markup)
                self.assertIn(f'aria-controls="recipe-settings-{section}"', markup)

    def test_all_supported_recipe_wide_fields_are_always_available_in_ui(self) -> None:
        result = self._render_settings_for_recipes()

        for recipe_kind, rendered in result.items():
            markup = rendered["markup"]
            parser = _RecipeSettingControlParser()
            parser.feed(markup)
            self.assertIn("<span>Purpose</span>", markup)
            self.assertIn("recipe-purpose-setting", markup)
            self.assertIn("recipe-purpose-chip", markup)
            self.assertNotRegex(markup, r"<span>Purpose</span>\s*<input")
            self.assertIn('readonly aria-readonly="true"', markup)
            self.assertNotIn('data-recipe-setting="task"', markup)
            for field in SETTINGS_FIELDS:
                with self.subTest(recipe=recipe_kind, field=field):
                    self.assertEqual(
                        markup.count(f'data-recipe-setting="{field}"'),
                        1,
                    )
                    tag, attributes = parser.controls[field]
                    self.assertIn(tag, {"input", "select", "textarea"})
                    self.assertNotIn("disabled", attributes)

        custom_markup = result["custom"]["markup"]
        for authored_value in (
            "fully_visible_custom_recipe",
            "Every top-level authoring surface stays available.",
            "custom_task",
            "custom_suite",
            "source.count",
            "extension_payload",
            "research-team",
        ):
            self.assertIn(authored_value, custom_markup)
        self.assertNotIn('data-recipe-setting="dataset_capture"', custom_markup)
        self.assertIn("data-new-recipe-matrix-dimension-target", custom_markup)
        self.assertIn("Add and bind dimension", custom_markup)

        image_markup = result["image"]["markup"]
        self.assertNotIn(
            'data-recipe-pipeline-setting="image:codec"', image_markup
        )
        self.assertNotIn('data-open-recipe-tool="codec"', image_markup)
        self.assertNotIn('data-open-recipe-tool="data"', image_markup)
        self.assertIn(
            'data-recipe-pipeline-setting="image:channel.wireless"',
            image_markup,
        )
        self.assertIn(
            'data-recipe-pipeline-setting="image:channel.coding"',
            image_markup,
        )
        self.assertIn("Updates linked channel blocks", image_markup)
        for owned_field in (
            "codec_profile",
            "channel_mode",
            "measure_codec_timing",
            "codec_timing",
        ):
            with self.subTest(owned_field=owned_field):
                self.assertNotIn(
                    f'data-recipe-metadata-value="{owned_field}"',
                    image_markup,
                )

        for recipe_kind in ("retrieval", "generation"):
            markup = result[recipe_kind]["markup"]
            self.assertIn(
                'data-recipe-linked-setting="task.clip.model_id"',
                markup,
            )
            self.assertIn(
                'data-recipe-linked-setting="task.clip.device"',
                markup,
            )

    def test_generation_safety_checker_defaults_to_enabled(self) -> None:
        result = self._run_node(
            r"""
const defaults = defaultTaskRecipeConfig("image_generation");
const absent = taskRecipeConfigFromRecipe({
  name: "safe_by_default",
  metadata: { research: { task: { id: "image_generation" } } },
  steps: [
    { id: "receiver", op: "foundation.diffusion_state_to_image", params: {} },
  ],
});
const explicitOptOut = taskRecipeConfigFromRecipe({
  name: "explicit_opt_out",
  metadata: { research: { task: { id: "image_generation" } } },
  steps: [
    {
      id: "receiver",
      op: "foundation.diffusion_state_to_image",
      params: { disable_safety_checker: true },
    },
  ],
});
return {
  defaultDisabled: defaults.generation.disable_safety_checker,
  absentDisabled: absent.generation.disable_safety_checker,
  explicitDisabled: explicitOptOut.generation.disable_safety_checker,
};
"""
        )

        self.assertFalse(result["defaultDisabled"])
        self.assertFalse(result["absentDisabled"])
        self.assertTrue(result["explicitDisabled"])

    def test_universal_dialog_binds_general_capture_and_advanced_controls(self) -> None:
        result = self._run_node(
            r"""
function control(attributes, values = {}) {
  const listeners = {};
  return {
    ...values,
    type: values.type || "text",
    value: values.value === undefined ? "" : values.value,
    checked: Boolean(values.checked),
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: (name) => attributes[name] === undefined ? null : attributes[name],
    fire: (event) => listeners[event] && listeners[event](),
  };
}

const name = control({ "data-recipe-setting": "name" }, { value: "renamed_recipe" });
const seed = control({ "data-recipe-global-setting": "seed" }, { type: "number", value: "42" });
const capture = control({ "data-recipe-setting": "dataset_capture" }, { type: "checkbox", checked: true });
const metadata = control({ "data-recipe-metadata-value": "owner" }, { value: '{"team":"vision"}' });
const extension = control({ "data-recipe-extension-value": "priority" }, { value: "3" });
const addExtension = control({});
const extensionKey = control({}, { value: "__proto__" });
const extensionValue = control({}, { value: '{"safe":true}' });

const all = new Map([
  ["[data-recipe-global-setting]", [seed]],
  ["[data-recipe-metadata-value]", [metadata]],
  ["[data-recipe-extension-value]", [extension]],
]);
const one = new Map([
  ['[data-recipe-setting="name"]', name],
  ['[data-recipe-setting="dataset_capture"]', capture],
  ["[data-add-recipe-extension]", addExtension],
  ["[data-new-recipe-extension-key]", extensionKey],
  ["[data-new-recipe-extension-value]", extensionValue],
]);
els.settingsBody = {
  querySelectorAll: (selector) => all.get(selector) || [],
  querySelector: (selector) => one.get(selector) || null,
};
state.operations = [];
state.editRecipe = {
  schema_version: 1,
  name: "before",
  metadata: { owner: "old" },
  priority: 1,
  steps: [],
};
let commits = 0;
let refreshes = 0;
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => { refreshes += 1; };

bindUniversalRecipeSettingsDialog();
name.fire("change");
seed.fire("change");
capture.fire("change");
metadata.fire("change");
extension.fire("change");
addExtension.fire("click");

return { recipe: state.editRecipe, commits, refreshes };
"""
        )

        self.assertEqual(result["recipe"]["name"], "renamed_recipe")
        self.assertEqual(result["recipe"]["metadata"]["seed"], 42)
        self.assertEqual(
            result["recipe"]["metadata"]["owner"],
            {"team": "vision"},
        )
        self.assertEqual(result["recipe"]["priority"], 3)
        self.assertEqual(result["recipe"]["__proto__"], {"safe": True})
        self.assertEqual(
            result["recipe"]["dataset_capture"],
            {"split": "train", "taps": []},
        )
        self.assertEqual(result["commits"], 6)
        self.assertEqual(result["refreshes"], 2)

    def test_shared_clip_setting_is_atomic_schema_safe_and_removes_diverging_variants(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  {
    id: "foundation.clip_image_embed", name: "Image CLIP", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: { type: "object", properties: { model_id: { type: "string" }, device: { type: "string" } }, additionalProperties: false },
  },
  {
    id: "foundation.clip_text_embed", name: "Text CLIP", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: { type: "object", properties: { model_id: { type: "string" }, device: { type: "string" } }, additionalProperties: false },
  },
  {
    id: "custom.strict", name: "Strict replacement", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: { type: "object", properties: {}, additionalProperties: false },
  },
];
state.editRecipe = {
  schema_version: 1,
  name: "retrieval",
  metadata: {
    matrix: {
      dimensions: { clip_model: ["old/a", "old/b"] },
      step_params: {
        image_encoder: { model_id: { matrix: "clip_model" } },
        text_encoder: { model_id: { matrix: "clip_model" } },
      },
    },
  },
  steps: [
    { id: "image_encoder", op: "foundation.clip_image_embed", inputs: {}, params: { model_id: "old/a", device: "cpu" } },
    { id: "text_encoder", op: "foundation.clip_text_embed", inputs: {}, params: { model_id: "old/a", device: "cpu" } },
    { id: "evaluation", op: "metrics.retrieval", inputs: {}, params: {} },
  ],
};
let commits = 0;
commitRecipeSettingsEdit = () => { commits += 1; };
setTaskSharedClipSetting("model_id", "openai/clip-new");
const afterAtomic = structuredCloneFallback(state.editRecipe);
findStep("text_encoder").op = "custom.strict";
const beforeRejected = JSON.stringify(state.editRecipe);
let rejection = "";
try {
  setTaskSharedClipSetting("device", "cuda");
} catch (error) {
  rejection = error.message;
}
return {
  afterAtomic,
  rejectedUnchanged: beforeRejected === JSON.stringify(state.editRecipe),
  rejection,
  commits,
};
"""
        )

        params = [
            step["params"]
            for step in result["afterAtomic"]["steps"]
            if step["id"] in {"image_encoder", "text_encoder"}
        ]
        self.assertEqual([item["model_id"] for item in params], ["openai/clip-new"] * 2)
        self.assertNotIn("matrix", result["afterAtomic"]["metadata"])
        self.assertTrue(result["rejectedUnchanged"])
        self.assertIn("cannot accept shared CLIP", result["rejection"])
        self.assertEqual(result["commits"], 1)

    def test_paired_runtime_controls_use_endpoint_schema_intersections(self) -> None:
        result = self._run_node(
            r"""
function operation(id, runtimeValues, dimensionMaximum = null) {
  const properties = { runtime: { type: "string", enum: runtimeValues } };
  if (dimensionMaximum !== null) properties.feedback_dimension = { type: "integer", minimum: 1, maximum: dimensionMaximum };
  return {
    id, name: id, status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: { type: "object", properties, additionalProperties: false },
  };
}
state.operations = [
  operation("test.csi_encoder", ["training_interface", "identity"], 100),
  operation("test.csi_decoder", ["training_interface"], 10),
  operation("test.jscc_sender", ["training_interface", "external_callable"]),
  operation("test.jscc_receiver", ["training_interface"]),
];
let commits = 0;
commitRecipeValueEdit = () => { commits += 1; };

state.editRecipe = {
  schema_version: 1,
  name: "csi",
  steps: [
    { id: "feedback_encoder", op: "test.csi_encoder", params: { runtime: "training_interface", feedback_dimension: 8 } },
    { id: "feedback_decoder", op: "test.csi_decoder", params: { runtime: "training_interface", feedback_dimension: 8 } },
  ],
};
const csiOptions = csiFeedbackRuntimeOptions(findStep("feedback_encoder"), findStep("feedback_decoder"));
const csiBefore = JSON.stringify(state.editRecipe);
let csiRuntimeError = "";
let csiDimensionError = "";
try { setCsiFeedbackRuntime("identity"); } catch (error) { csiRuntimeError = error.message; }
try { setCsiFeedbackDimension("20"); } catch (error) { csiDimensionError = error.message; }
const csiUnchanged = JSON.stringify(state.editRecipe) === csiBefore;

state.editRecipe = {
  schema_version: 1,
  name: "jscc",
  steps: [
    { id: "sender", op: "test.jscc_sender", params: { runtime: "training_interface" } },
    { id: "receiver", op: "test.jscc_receiver", params: { runtime: "training_interface" } },
  ],
};
const jsccOptions = jointSourceChannelRuntimeOptions(findStep("sender"), findStep("receiver"));
const jsccBefore = JSON.stringify(state.editRecipe);
let jsccError = "";
try { setJointSourceChannelRuntime("external_callable"); } catch (error) { jsccError = error.message; }
return {
  csiOptions,
  csiRuntimeError,
  csiDimensionError,
  csiUnchanged,
  jsccOptions,
  jsccError,
  jsccUnchanged: JSON.stringify(state.editRecipe) === jsccBefore,
  commits,
};
"""
        )

        self.assertNotIn('value="identity"', result["csiOptions"])
        self.assertIn("value must be one of", result["csiRuntimeError"])
        self.assertIn("at most 10", result["csiDimensionError"])
        self.assertTrue(result["csiUnchanged"])
        self.assertNotIn('value="external_callable"', result["jsccOptions"])
        self.assertIn("value must be one of", result["jsccError"])
        self.assertTrue(result["jsccUnchanged"])
        self.assertEqual(result["commits"], 0)

    def test_pipeline_replacement_preserves_recipe_identity_and_global_fields(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  {
    id: "source.old",
    name: "Old source",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {},
    output_kinds: { values: "tensor.numpy" },
    params_schema: { type: "object", properties: { gain: { type: "number" } }, additionalProperties: false },
  },
  {
    id: "source.new",
    name: "New source",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {},
    output_kinds: { values: "tensor.numpy" },
    params_schema: { type: "object", properties: {}, additionalProperties: false },
  },
];
state.editRecipe = {
  schema_version: 1,
  name: "research_working_copy",
  description: "Keep this user-authored description.",
  suite: { id: "research_suite" },
  dataset_capture: {
    split: "validation",
    taps: [{ id: "old_values", from: "old.values" }],
    sweep: { "old.gain": [1, 2] },
  },
  metadata: {
    owner: "lab",
    matrix: {
      dimensions: { gain: [1, 2] },
      step_params: { old: { gain: { matrix: "gain" } } },
    },
    trained_artifact_bindings: { old: { id: "old-artifact" } },
  },
  steps: [{ id: "old", op: "source.old", params: { gain: 1 } }],
};
state.selectedRecipe = null;
let applied = null;
applyBuiltRecipe = (recipe) => {
  applied = recipe;
  state.editRecipe = recipe;
};
refreshUniversalRecipeSettingsDialog = () => {};
notify = () => {};
applyExplicitRecipeTopologyReplacement({
  schema_version: 1,
  name: "builder_default_name",
  description: "Builder default description.",
  execution_profile: { id: "custom", version: 1 },
  metadata: { derived_pipeline: true },
  steps: [{ id: "new", op: "source.new", params: {} }],
}, "Pipeline preset");
return applied;
"""
        )

        self.assertEqual(result["name"], "research_working_copy")
        self.assertEqual(
            result["description"],
            "Keep this user-authored description.",
        )
        self.assertEqual(result["suite"], {"id": "research_suite"})
        self.assertEqual(result["dataset_capture"]["split"], "validation")
        self.assertEqual(result["dataset_capture"]["taps"], [])
        self.assertNotIn("sweep", result["dataset_capture"])
        self.assertEqual(result["metadata"]["owner"], "lab")
        self.assertTrue(result["metadata"]["derived_pipeline"])
        self.assertNotIn("matrix", result["metadata"])
        self.assertNotIn("trained_artifact_bindings", result["metadata"])
        self.assertEqual(result["steps"][0]["id"], "new")

    def test_image_pipeline_setting_confirmation_and_real_builder_routing(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.trainedArtifacts = [];
state.selectedRecipe = null;
state.editRecipe = JSON.parse(JSON.stringify(__payload.image));
state.editRecipe.name = "user_named_image_study";
state.editRecipe.description = "Keep the working-copy description.";
state.editRecipe.global_extension = {
  owner: "research-team",
  policy: { retain: true },
};
state.editRecipe.metadata = {
  ...(state.editRecipe.metadata || {}),
  owner: "lab",
  custom_extension: { campaign: "summer" },
};

let applyCount = 0;
let refreshCount = 0;
let confirmCount = 0;
let notices = [];
applyBuiltRecipe = (recipe) => {
  applyCount += 1;
  state.editRecipe = recipe;
};
refreshUniversalRecipeSettingsDialog = () => { refreshCount += 1; };
notify = (message) => { notices.push(message); };

const beforeCancellation = JSON.stringify(state.editRecipe);
window.confirm = () => {
  confirmCount += 1;
  return false;
};
applyRecipePipelineSetting("image:channel.wireless", "awgn");
const cancelled = {
  byteIdentical: JSON.stringify(state.editRecipe) === beforeCancellation,
  applyCount,
  refreshCount,
  confirmCount,
};

window.confirm = () => {
  confirmCount += 1;
  return true;
};
applyRecipePipelineSetting("image:channel.wireless", "awgn");
const wireless = state.editRecipe.steps.find((step) => step.id === "wireless_channel");
const modulator = state.editRecipe.steps.find((step) => step.id === "modulator");
const demodulator = state.editRecipe.steps.find((step) => step.id === "demodulator");
return {
  cancelled,
  confirmed: {
    applyCount,
    refreshCount,
    confirmCount,
    name: state.editRecipe.name,
    description: state.editRecipe.description,
    suite: state.editRecipe.suite,
    globalExtension: state.editRecipe.global_extension,
    metadataOwner: state.editRecipe.metadata.owner,
    metadataExtension: state.editRecipe.metadata.custom_extension,
    channelEnabled: state.editRecipe.metadata.channel_enabled,
    wireless,
    modulator,
    demodulator,
    notices,
  },
};
""",
            image=self.image_recipe,
            operations=self.operations,
        )

        self.assertEqual(
            result["cancelled"],
            {
                "byteIdentical": True,
                "applyCount": 0,
                "refreshCount": 1,
                "confirmCount": 1,
            },
        )

        confirmed = result["confirmed"]
        self.assertEqual(confirmed["applyCount"], 1)
        self.assertEqual(confirmed["refreshCount"], 2)
        self.assertEqual(confirmed["confirmCount"], 2)
        self.assertEqual(confirmed["name"], "user_named_image_study")
        self.assertEqual(
            confirmed["description"],
            "Keep the working-copy description.",
        )
        self.assertEqual(confirmed["suite"], self.image_recipe["suite"])
        self.assertEqual(
            confirmed["globalExtension"],
            {"owner": "research-team", "policy": {"retain": True}},
        )
        self.assertEqual(confirmed["metadataOwner"], "lab")
        self.assertEqual(
            confirmed["metadataExtension"],
            {"campaign": "summer"},
        )
        self.assertTrue(confirmed["channelEnabled"])
        self.assertEqual(confirmed["wireless"]["op"], "wireless.channel")
        self.assertEqual(confirmed["wireless"]["params"]["channel"], "awgn")
        self.assertEqual(
            confirmed["modulator"]["op"],
            "modulation.digital_modulate",
        )
        self.assertEqual(
            confirmed["demodulator"]["op"],
            "demodulation.digital_demodulate",
        )
        self.assertIn("Image pipeline preset applied", confirmed["notices"][-1])

    def test_block_owned_codec_family_replaces_both_endpoints_without_opening_recipe_settings(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.trainedArtifacts = [];
state.selectedRecipe = null;
state.editRecipe = JSON.parse(JSON.stringify(__payload.image));
state.editRecipe.name = "paired_codec_study";
state.editRecipe.description = "Keep this study identity.";

let applied = null;
let refreshCount = 0;
let notices = [];
applyBuiltRecipe = (recipe) => {
  applied = recipe;
  state.editRecipe = recipe;
};
refreshUniversalRecipeSettingsDialog = () => { refreshCount += 1; };
notify = (message) => { notices.push(message); };
window.confirm = () => true;

applyRecipePipelineSetting("image:codec", "jpeg", { surface: "blocks" });
return {
  applied,
  refreshCount,
  notices,
  sender: applied.steps.find((step) => step.id === "sender"),
  receiver: applied.steps.find((step) => step.id === "receiver"),
  payloadEncoder: applied.steps.find((step) => step.id === "payload_encoder"),
  payloadDecoder: applied.steps.find((step) => step.id === "payload_decoder"),
};
""",
            image=self.image_recipe,
            operations=self.operations,
        )

        self.assertEqual(result["refreshCount"], 0)
        self.assertEqual(result["applied"]["name"], "paired_codec_study")
        self.assertEqual(
            result["applied"]["description"], "Keep this study identity."
        )
        self.assertEqual(result["sender"]["op"], "model.jpeg_encode")
        self.assertEqual(result["receiver"]["op"], "model.jpeg_decode")
        self.assertEqual(
            result["payloadEncoder"]["op"],
            "channel.payload_passthrough_encoder",
        )
        self.assertEqual(
            result["payloadDecoder"]["op"],
            "channel.payload_passthrough_decoder",
        )
        self.assertIn("Image pipeline preset applied", result["notices"][-1])

    def test_structured_value_parser_is_typed_and_rejects_malformed_json(self) -> None:
        result = self._run_node(
            r"""
function parse(raw) {
  try {
    return { ok: true, value: parseRecipeSettingText(raw) };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}
function matrix(raw) {
  try {
    return { ok: true, value: parseRecipeMatrixValues(raw) };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}
return {
  plain: parse("research-note"),
  object: parse('{"owner":"lab"}'),
  array: parse('[1,2]'),
  quoted: parse('"true"'),
  boolean: parse("true"),
  nullValue: parse("null"),
  number: parse("12.5"),
  malformedObject: parse("{broken"),
  malformedArray: matrix("[1,]"),
  malformedQuote: parse('"unterminated'),
  overflow: parse("1e309"),
  nestedOverflow: parse('{"values":[1,1e309]}'),
  unsafeInteger: parse("9007199254740993"),
  duplicateObjectKey: parse('{"seed":1,"seed":9}'),
  csv: matrix("low,medium"),
  quotedComma: matrix('"low,medium"'),
  objectDimension: matrix('{"mode":"fast","weight":2}'),
};
"""
        )

        self.assertEqual(result["plain"], {"ok": True, "value": "research-note"})
        self.assertEqual(result["object"]["value"], {"owner": "lab"})
        self.assertEqual(result["array"]["value"], [1, 2])
        self.assertEqual(result["quoted"]["value"], "true")
        self.assertIs(result["boolean"]["value"], True)
        self.assertIsNone(result["nullValue"]["value"])
        self.assertEqual(result["number"]["value"], 12.5)
        self.assertFalse(result["malformedObject"]["ok"])
        self.assertFalse(result["malformedArray"]["ok"])
        self.assertFalse(result["malformedQuote"]["ok"])
        self.assertFalse(result["overflow"]["ok"])
        self.assertFalse(result["nestedOverflow"]["ok"])
        self.assertFalse(result["unsafeInteger"]["ok"])
        self.assertFalse(result["duplicateObjectKey"]["ok"])
        self.assertIn(
            "duplicate JSON object key",
            result["duplicateObjectKey"]["error"],
        )
        self.assertEqual(result["csv"]["value"], ["low", "medium"])
        self.assertEqual(result["quotedComma"]["value"], ["low,medium"])
        self.assertEqual(
            result["objectDimension"]["value"],
            [{"mode": "fast", "weight": 2}],
        )

    def test_capture_split_updates_are_atomic_complete_and_validated(self) -> None:
        result = self._run_node(
            r"""
const initial = {
  total_samples: 30,
  percentages: { train: 60, validation: 20, test: 20 },
  counts: { train: 18, validation: 6, test: 6 },
};
const valid = updatedRecipeCaptureSplitPlan(initial, "train", "70");
const invalid = updatedRecipeCaptureSplitPlan(initial, "validation", "40");
const tooSmall = updatedRecipeCaptureSplitPlan(initial, "total_samples", "2");
const fallbackTooSmall = updatedRecipeCaptureSplitPlan({}, "train", "60", 2);
return { initial, valid, invalid, tooSmall, fallbackTooSmall };
"""
        )

        self.assertEqual(
            result["valid"]["plan"]["percentages"],
            {"train": 70, "validation": 20, "test": 10},
        )
        self.assertNotIn("counts", result["valid"]["plan"])
        self.assertEqual(sum(result["valid"]["plan"]["percentages"].values()), 100)
        self.assertIsNone(result["invalid"]["plan"])
        self.assertIn("less than 100", result["invalid"]["error"])
        self.assertIsNone(result["tooSmall"]["plan"])
        self.assertIn("at least three", result["tooSmall"]["error"])
        self.assertIsNone(result["fallbackTooSmall"]["plan"])
        self.assertIn("at least three", result["fallbackTooSmall"]["error"])
        self.assertEqual(
            result["initial"]["percentages"],
            {"train": 60, "validation": 20, "test": 20},
        )

    def test_capture_sample_fallback_derives_percentages_and_discards_stale_counts(self) -> None:
        result = self._run_node(
            r"""
function control(attributes, values = {}) {
  const listeners = {};
  return {
    ...values,
    type: values.type || "text",
    value: values.value === undefined ? "" : values.value,
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: (name) => attributes[name] === undefined ? null : attributes[name],
    fire: (event) => listeners[event] && listeners[event](),
  };
}

state.operations = [];
state.editRecipe = {
  schema_version: 1,
  name: "count_backed_capture",
  metadata: {},
  steps: [],
  dataset_capture: {
    split: "train",
    samples: 100,
    taps: [],
    split_plan: { counts: { train: 80, validation: 10, test: 10 } },
  },
};
const beforeMarkup = recipeDatasetCaptureSettingsMarkup(state.editRecipe);
const samples = control({ "data-recipe-capture-field": "samples" }, { type: "number", value: "200" });
els.settingsBody = {
  querySelectorAll: (selector) => selector === "[data-recipe-capture-field]" ? [samples] : [],
  querySelector: () => null,
};
commitRecipeValueEdit = () => {};
refreshUniversalRecipeSettingsDialog = () => {};
bindRecipeDatasetCaptureSettings();
samples.fire("change");
const afterChange = JSON.parse(JSON.stringify(state.editRecipe.dataset_capture));
samples.value = "";
samples.fire("change");
return {
  beforeMarkup,
  derived: recipeCaptureSplitPercentages({ counts: { train: 80, validation: 10, test: 10 } }, 100),
  afterChange,
  afterClear: state.editRecipe.dataset_capture,
};
"""
        )

        self.assertEqual(
            result["derived"],
            {"train": 80, "validation": 10, "test": 10},
        )
        self.assertIn(
            'data-recipe-capture-split-field="total_samples" value="100"',
            result["beforeMarkup"],
        )
        self.assertIn(
            'data-recipe-capture-split-field="train" value="80"',
            result["beforeMarkup"],
        )
        self.assertIn(
            'data-recipe-capture-split-field="validation" value="10"',
            result["beforeMarkup"],
        )
        self.assertIn('data-recipe-capture-derived-test value="10"', result["beforeMarkup"])
        self.assertEqual(result["afterChange"]["samples"], 200)
        self.assertNotIn("total_samples", result["afterChange"]["split_plan"])
        self.assertNotIn("counts", result["afterChange"]["split_plan"])
        self.assertEqual(
            result["afterChange"]["split_plan"]["percentages"],
            {"train": 80, "validation": 10, "test": 10},
        )
        self.assertNotIn("samples", result["afterClear"])
        self.assertEqual(result["afterClear"]["split_plan"]["total_samples"], 200)
        self.assertNotIn("counts", result["afterClear"]["split_plan"])
        self.assertEqual(
            result["afterClear"]["split_plan"]["percentages"],
            {"train": 80, "validation": 10, "test": 10},
        )

    def test_round_robin_capture_sweep_is_edited_without_shape_corruption(self) -> None:
        result = self._run_node(
            r"""
function control(attributes, values = {}) {
  const listeners = {};
  return {
    ...values,
    value: values.value === undefined ? "" : values.value,
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: (name) => attributes[name] === undefined ? null : attributes[name],
    fire: (event) => listeners[event] && listeners[event](),
  };
}

state.operations = [{
  id: "custom.capture_source",
  name: "Capture source",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {},
  output_kinds: { values: "tensor.numpy" },
  params_schema: {
    type: "object",
    properties: {
      gain: { type: "integer", minimum: 0 },
      mode: { type: "string" },
    },
    additionalProperties: false,
  },
}];
state.editRecipe = {
  schema_version: 1,
  name: "round_robin_capture",
  metadata: {},
  steps: [{ id: "source", op: "custom.capture_source", params: { gain: 1, mode: "slow" } }],
  dataset_capture: {
    split: "train",
    taps: [],
    sweep: [
      { "source.gain": 1 },
      { "source.gain": 2, "source.mode": "fast" },
    ],
  },
};
const markup = recipeDatasetCaptureSettingsMarkup(state.editRecipe);
const staleGridEdit = control({ "data-recipe-capture-sweep": "source.gain" }, { value: "99" });
const assignmentEdit = control({ "data-recipe-capture-sweep-assignment": "0" }, { value: '{"source.gain":3}' });
const assignmentRemove = control({ "data-remove-recipe-capture-sweep-assignment": "1" });
const assignmentAdd = control({}, {});
const newAssignment = control({}, { value: '{"source.gain":4,"source.mode":"turbo"}' });
const all = new Map([
  ["[data-recipe-capture-sweep]", [staleGridEdit]],
  ["[data-recipe-capture-sweep-assignment]", [assignmentEdit]],
  ["[data-remove-recipe-capture-sweep-assignment]", [assignmentRemove]],
]);
const one = new Map([
  ["[data-add-recipe-capture-sweep-assignment]", assignmentAdd],
  ["[data-new-recipe-capture-sweep-assignment]", newAssignment],
]);
els.settingsBody = {
  querySelectorAll: (selector) => all.get(selector) || [],
  querySelector: (selector) => one.get(selector) || null,
};
let notices = [];
let commits = 0;
notify = (message) => { notices.push(message); };
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};
bindRecipeDatasetCaptureSettings();

const beforeStaleGridEdit = JSON.stringify(state.editRecipe.dataset_capture.sweep);
staleGridEdit.fire("change");
const staleGridBlocked = JSON.stringify(state.editRecipe.dataset_capture.sweep) === beforeStaleGridEdit;
assignmentEdit.fire("change");
assignmentAdd.fire("click");
assignmentRemove.fire("click");
return {
  markup,
  staleGridBlocked,
  sweep: state.editRecipe.dataset_capture.sweep,
  notices,
  commits,
};
"""
        )

        self.assertIn('data-recipe-capture-sweep-assignment="0"', result["markup"])
        self.assertIn('data-recipe-capture-sweep-assignment="1"', result["markup"])
        self.assertIn("round-robin assignments", result["markup"])
        self.assertNotIn('data-recipe-capture-sweep="source.gain"', result["markup"])
        self.assertTrue(result["staleGridBlocked"])
        self.assertEqual(
            result["sweep"],
            [
                {"source.gain": 3},
                {"source.gain": 4, "source.mode": "turbo"},
            ],
        )
        self.assertIsInstance(result["sweep"], list)
        self.assertEqual(result["commits"], 3)
        self.assertTrue(any("round-robin assignments" in notice for notice in result["notices"]))

    def test_capture_sweeps_reject_values_outside_target_parameter_schema(self) -> None:
        result = self._run_node(
            r"""
function control(attributes, values = {}) {
  const listeners = {};
  return {
    ...values,
    value: values.value === undefined ? "" : values.value,
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: (name) => attributes[name] === undefined ? null : attributes[name],
    fire: (event) => listeners[event] && listeners[event](),
  };
}
function bindOnly(selector, input) {
  els.settingsBody = {
    querySelectorAll: (candidate) => candidate === selector ? [input] : [],
    querySelector: () => null,
  };
  bindRecipeDatasetCaptureSettings();
}
state.operations = [{
  id: "custom.integer_source",
  name: "Integer source",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
  params_schema: {
    type: "object",
    properties: { count: { type: "integer", minimum: 1 } },
    additionalProperties: false,
  },
}];
const recipe = {
  schema_version: 1,
  name: "typed_capture_sweep",
  metadata: {},
  steps: [{ id: "source", op: "custom.integer_source", params: { count: 1 } }],
};
let notices = [];
let commits = 0;
notify = (message) => { notices.push(message); };
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};

state.editRecipe = {
  ...recipe,
  dataset_capture: { split: "train", taps: [], sweep: { "source.count": [1, 2] } },
};
const invalidGridInput = control({ "data-recipe-capture-sweep": "source.count" }, { value: '[1,"two"]' });
bindOnly("[data-recipe-capture-sweep]", invalidGridInput);
const gridBefore = JSON.stringify(state.editRecipe.dataset_capture.sweep);
invalidGridInput.fire("change");
const gridUnchanged = JSON.stringify(state.editRecipe.dataset_capture.sweep) === gridBefore;

state.editRecipe = {
  ...recipe,
  dataset_capture: { split: "train", taps: [], sweep: [{ "source.count": 1 }] },
};
const invalidAssignmentInput = control({ "data-recipe-capture-sweep-assignment": "0" }, { value: '{"source.count":"two"}' });
bindOnly("[data-recipe-capture-sweep-assignment]", invalidAssignmentInput);
const assignmentBefore = JSON.stringify(state.editRecipe.dataset_capture.sweep);
invalidAssignmentInput.fire("change");
const assignmentUnchanged = JSON.stringify(state.editRecipe.dataset_capture.sweep) === assignmentBefore;
return {
  validAssignment: recipeCaptureSweepAssignmentProblem(recipe, { "source.count": 3 }),
  invalidAssignment: recipeCaptureSweepAssignmentProblem(recipe, { "source.count": "three" }),
  validGrid: recipeCaptureSweepGridProblem(recipe, "source.count", [1, 2, 3]),
  invalidGrid: recipeCaptureSweepGridProblem(recipe, "source.count", [1, "two", 3]),
  invalidTextGrid: recipeCaptureSweepGridProblem(recipe, "source.count", "one,two"),
  gridUnchanged,
  assignmentUnchanged,
  notices,
  commits,
};
"""
        )

        self.assertEqual(result["validAssignment"], "")
        self.assertIn("value must be integer", result["invalidAssignment"])
        self.assertEqual(result["validGrid"], "")
        self.assertIn("value must be integer", result["invalidGrid"])
        self.assertIn("must be numeric", result["invalidTextGrid"])
        self.assertTrue(result["gridUnchanged"])
        self.assertTrue(result["assignmentUnchanged"])
        self.assertEqual(result["commits"], 0)
        self.assertEqual(len(result["notices"]), 2)

    def test_capture_sweep_plan_is_bounded_before_grid_expansion(self) -> None:
        result = self._run_node(
            r"""
state.operations = [{
  id: "custom.capture_source",
  name: "Capture source",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
  params_schema: {
    type: "object",
    properties: {
      first: { type: "number" },
      second: { type: "number" },
    },
    additionalProperties: false,
  },
}];
const recipe = {
  schema_version: 1,
  name: "bounded_capture",
  steps: [{ id: "source", op: "custom.capture_source", params: { first: 0, second: 0 } }],
};
return {
  range: recipeCaptureSweepPlanProblem(recipe, { "source.first": "0:0.1:100" }),
  product: recipeCaptureSweepPlanProblem(recipe, {
    "source.first": Array.from({ length: 20 }, (_item, index) => index),
    "source.second": Array.from({ length: 20 }, (_item, index) => index),
  }),
  list: recipeCaptureSweepPlanProblem(
    recipe,
    Array.from({ length: 257 }, (_item, index) => ({ "source.first": index })),
  ),
  small: recipeCaptureSweepPlanProblem(recipe, {
    "source.first": [1, 2],
    "source.second": [3, 4],
  }),
};
"""
        )

        self.assertIn("maximum is 256", result["range"])
        self.assertIn("400 assignments", result["product"])
        self.assertIn("257 round-robin", result["list"])
        self.assertEqual(result["small"], "")

    def test_capture_taps_filter_selected_outputs_and_reject_empty_or_duplicates(self) -> None:
        result = self._run_node(
            r"""
function control(attributes, values = {}) {
  const listeners = {};
  return {
    ...values,
    value: values.value === undefined ? "" : values.value,
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: (name) => attributes[name] === undefined ? null : attributes[name],
    fire: (event) => listeners[event] && listeners[event](),
  };
}
function bindWith(all, one) {
  els.settingsBody = {
    querySelectorAll: (selector) => all.get(selector) || [],
    querySelector: (selector) => one.get(selector) || null,
  };
  bindRecipeDatasetCaptureSettings();
}

state.operations = [{
  id: "custom.two_outputs",
  name: "Two outputs",
  status: "implemented",
  input_kinds: {}, optional_input_kinds: {},
  output_kinds: { first: "tensor.numpy", second: "tensor.numpy" },
  params_schema: { type: "object", properties: {}, additionalProperties: false },
}];
const baseRecipe = {
  schema_version: 1,
  name: "capture_taps",
  metadata: {},
  steps: [{ id: "source", op: "custom.two_outputs", params: {} }],
};
state.editRecipe = {
  ...baseRecipe,
  dataset_capture: {
    split: "train",
    taps: [
      { id: "first_signal", from: "source.first" },
      { id: "second_signal", from: "source.second" },
    ],
  },
};
const fullMarkup = recipeDatasetCaptureSettingsMarkup(state.editRecipe);
const emptyId = control({ "data-recipe-capture-tap-id": "0" }, { value: "   " });
const reservedId = control({ "data-recipe-capture-tap-id": "0" }, { value: "metadata_json" });
const duplicateId = control({ "data-recipe-capture-tap-id": "0" }, { value: "second_signal" });
const emptyFrom = control({ "data-recipe-capture-tap-from": "0" }, { value: "" });
const duplicateFrom = control({ "data-recipe-capture-tap-from": "0" }, { value: "source.second" });
const exhaustedAdd = control({});
let notices = [];
let commits = 0;
notify = (message) => { notices.push(message); };
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};
bindWith(new Map([
  ["[data-recipe-capture-tap-id]", [emptyId, reservedId, duplicateId]],
  ["[data-recipe-capture-tap-from]", [emptyFrom, duplicateFrom]],
]), new Map([["[data-add-recipe-capture-tap]", exhaustedAdd]]));
const beforeRejectedEdits = JSON.stringify(state.editRecipe.dataset_capture.taps);
emptyId.fire("change");
reservedId.fire("change");
duplicateId.fire("change");
emptyFrom.fire("change");
duplicateFrom.fire("change");
exhaustedAdd.fire("click");
const rejectedEditsPreserved = JSON.stringify(state.editRecipe.dataset_capture.taps) === beforeRejectedEdits;

state.editRecipe = {
  ...baseRecipe,
  dataset_capture: { split: "train", taps: [{ id: "first_signal", from: "source.first" }] },
};
const availableMarkup = recipeDatasetCaptureSettingsMarkup(state.editRecipe);
const availableAdd = control({});
bindWith(new Map(), new Map([["[data-add-recipe-capture-tap]", availableAdd]]));
availableAdd.fire("click");
return {
  fullMarkup,
  availableMarkup,
  rejectedEditsPreserved,
  notices,
  commits,
  addedTaps: state.editRecipe.dataset_capture.taps,
  invalidIssues: datasetCaptureConfigurationIssues({
    ...baseRecipe,
    dataset_capture: {
      taps: [{ id: "metadata_json", from: "missing.output" }],
    },
  }),
};
"""
        )

        self.assertIn("data-add-recipe-capture-tap disabled", result["fullMarkup"])
        self.assertNotIn("data-add-recipe-capture-tap disabled", result["availableMarkup"])
        self.assertEqual(result["fullMarkup"].count('value="source.first"'), 1)
        self.assertEqual(result["fullMarkup"].count('value="source.second"'), 1)
        self.assertTrue(result["rejectedEditsPreserved"])
        self.assertEqual(result["commits"], 1)
        self.assertEqual(
            result["addedTaps"],
            [
                {"id": "first_signal", "from": "source.first"},
                {"id": "source_second", "from": "source.second"},
            ],
        )
        self.assertTrue(any("cannot be empty" in notice for notice in result["notices"]))
        self.assertTrue(any("already in use" in notice for notice in result["notices"]))
        self.assertTrue(any("already selected" in notice for notice in result["notices"]))
        self.assertTrue(any("metadata_json is reserved" in notice for notice in result["notices"]))
        self.assertTrue(any("metadata_json is reserved" in issue for issue in result["invalidIssues"]))
        self.assertTrue(any("missing.output" in issue for issue in result["invalidIssues"]))

    def test_generic_master_seed_does_not_correlate_explicit_block_streams(self) -> None:
        result = self._run_node(
            r"""
state.operations = [];
state.editRecipe = {
  schema_version: 1,
  name: "seed_policy",
  metadata: { seed: 7 },
  steps: [
    { id: "first", op: "custom.first", params: { seed: 101 } },
    { id: "second", op: "custom.second", params: { seed: 202 } },
  ],
};
commitRecipeSettingsEdit = () => {};
updateUniversalRecipeGlobalSetting("seed", { type: "number", value: "42" });
return state.editRecipe;
"""
        )

        self.assertEqual(result["metadata"]["seed"], 42)
        self.assertEqual(
            [step["params"]["seed"] for step in result["steps"]],
            [101, 202],
        )

    def test_loaded_recipe_global_settings_update_metadata_and_all_compatible_blocks(self) -> None:
        result = self._run_node(
            r"""
const customOperations = [
  {
    id: "source.custom_primary", name: "Custom primary source", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: { value: "tensor.numpy" },
    params_schema: {
      type: "object",
      properties: {
        repeat_count: { type: "integer", minimum: 1 },
        data_plane_backend: { type: "string", enum: DATA_PLANE_BACKENDS },
        seed: { type: "integer", minimum: 0 },
      },
      additionalProperties: false,
    },
  },
  {
    id: "source.custom_secondary", name: "Custom secondary source", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: { value: "tensor.numpy" },
    params_schema: {
      type: "object",
      properties: { repeat_count: { type: "integer", minimum: 1 } },
      additionalProperties: false,
    },
  },
  {
    id: "custom.backend", name: "Custom backend consumer", status: "implemented",
    input_kinds: { value: "tensor.numpy" }, optional_input_kinds: {}, output_kinds: {},
    params_schema: {
      type: "object",
      properties: {
        data_plane_backend: { type: "string", enum: DATA_PLANE_BACKENDS },
        repeat_count: { type: "integer", minimum: 1 },
      },
      additionalProperties: false,
    },
  },
  {
    id: "custom.seeded", name: "Custom seeded block", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: {
      type: "object",
      properties: { seed: { type: "integer", minimum: 0 } },
      additionalProperties: false,
    },
  },
];
state.operations = [...__payload.operations, ...customOperations];
state.trainedArtifacts = [];
let commits = 0;
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};
applyBuiltRecipe = (recipe) => {
  state.editRecipe = recipe;
  if (state.selectedRecipe) state.selectedRecipe.recipe = recipe;
};

function declaresOrHas(step, field) {
  const operation = operationById(step.op) || {};
  const properties = (((operation.params_schema || {}).properties) || {});
  return Object.prototype.hasOwnProperty.call(step.params || {}, field)
    || Object.prototype.hasOwnProperty.call(properties, field);
}

function compatibleTargets(recipe) {
  return {
    repeat: recipe.steps
      .filter((step) => String(step.op || "").startsWith("source.") && declaresOrHas(step, "repeat_count"))
      .map((step) => step.id),
    backend: recipe.steps
      .filter((step) => declaresOrHas(step, "data_plane_backend"))
      .map((step) => step.id),
  };
}

function explicitSeeds(recipe) {
  return Object.fromEntries(recipe.steps
    .filter((step) => Object.prototype.hasOwnProperty.call(step.params || {}, "seed"))
    .map((step) => [step.id, step.params.seed]));
}

function topology(recipe) {
  return recipe.steps.map((step) => ({
    id: step.id,
    op: step.op,
    inputs: structuredCloneFallback(step.inputs || {}),
  }));
}

function installCanonicalOverrides(recipe, targets) {
  const stepParams = {};
  targets.repeat.forEach((stepId) => {
    stepParams[stepId] = { ...(stepParams[stepId] || {}), repeat_count: { matrix: "global_repeat" } };
  });
  targets.backend.forEach((stepId) => {
    stepParams[stepId] = { ...(stepParams[stepId] || {}), data_plane_backend: { matrix: "global_backend" } };
  });
  const seedStep = recipe.steps.find((step) => Object.prototype.hasOwnProperty.call(step.params || {}, "seed"));
  stepParams[seedStep.id] = { ...(stepParams[seedStep.id] || {}), seed: { matrix: "explicit_stream" } };
  recipe.metadata = {
    ...(recipe.metadata || {}),
    matrix: {
      dimensions: {
        global_repeat: [2, 4],
        global_backend: ["auto", "python_numpy"],
        explicit_stream: [seedStep.params.seed, seedStep.params.seed + 1],
      },
      step_params: stepParams,
    },
  };
}

function installLegacyOverrides(recipe, targets) {
  const sweeps = {};
  targets.repeat.forEach((stepId) => { sweeps[`${stepId}.repeat_count`] = "2,4"; });
  targets.backend.forEach((stepId) => { sweeps[`${stepId}.data_plane_backend`] = "auto,python_numpy"; });
  const seedStep = recipe.steps.find((step) => Object.prototype.hasOwnProperty.call(step.params || {}, "seed"));
  sweeps[`${seedStep.id}.seed`] = `${seedStep.params.seed},${seedStep.params.seed + 1}`;
  recipe.metadata = { ...(recipe.metadata || {}), sweeps };
}

function runCase(recipe, key, legacy = false) {
  state.editRecipe = structuredCloneFallback(recipe);
  state.editRecipeKey = key;
  state.selectedRecipe = {
    key,
    recipe: state.editRecipe,
    preserveTopology: true,
    workingCopy: true,
  };
  const targets = compatibleTargets(state.editRecipe);
  if (legacy) installLegacyOverrides(state.editRecipe, targets);
  else installCanonicalOverrides(state.editRecipe, targets);
  const before = {
    topology: topology(state.editRecipe),
    seeds: explicitSeeds(state.editRecipe),
  };

  updateUniversalRecipeGlobalSetting("seed", { type: "number", value: "444" });
  updateUniversalRecipeGlobalSetting("repeatCount", { type: "number", value: "7" });
  updateUniversalRecipeGlobalSetting("dataPlaneBackend", { type: "select-one", value: "cpp_native" });
  updateUniversalRecipeGlobalSetting("timingEnabled", { type: "checkbox", checked: false });
  updateUniversalRecipeGlobalSetting("timingWarmupRuns", { type: "number", value: "2" });
  updateUniversalRecipeGlobalSetting("timingTimedRuns", { type: "number", value: "5" });

  const matrix = canonicalRecipeMatrix(state.editRecipe);
  return {
    metadata: state.editRecipe.metadata,
    topologyPreserved: JSON.stringify(topology(state.editRecipe)) === JSON.stringify(before.topology),
    explicitSeedsBefore: before.seeds,
    explicitSeedsAfter: explicitSeeds(state.editRecipe),
    repeatTargets: targets.repeat,
    repeatValues: Object.fromEntries(targets.repeat.map((stepId) => [stepId, stepById(state.editRecipe.steps, stepId).params.repeat_count])),
    backendTargets: targets.backend,
    backendValues: Object.fromEntries(targets.backend.map((stepId) => [stepId, stepById(state.editRecipe.steps, stepId).params.data_plane_backend])),
    nonSourceRepeatUnchanged: state.editRecipe.steps
      .filter((step) => !String(step.op || "").startsWith("source.") && declaresOrHas(step, "repeat_count"))
      .every((step) => step.params.repeat_count !== 7),
    remainingBindings: matrix === null ? [] : matrixDirectBindings(matrix)
      .map(({ stepId, param, dimension }) => ({ stepId, param, dimension })),
    hasLegacySweeps: Boolean(state.editRecipe.metadata.sweeps || state.editRecipe.metadata.ui_sweeps),
  };
}

const image = structuredCloneFallback(__payload.image);
const imageSeedStep = image.steps.find((step) => Object.prototype.hasOwnProperty.call(step.params || {}, "seed"));
if (!imageSeedStep) throw new Error("image fixture must contain an explicit per-block seed");

const text = structuredCloneFallback(__payload.text);
text.steps.push({ id: "text_seed_stream", op: "custom.seeded", inputs: {}, params: { seed: 733 } });

const custom = {
  schema_version: 1,
  execution_profile: { id: "custom", version: 1 },
  name: "loaded_custom_recipe",
  metadata: { owner: "lab", codec_timing: { runner: "custom-runner" } },
  steps: [
    { id: "primary", op: "source.custom_primary", inputs: {}, params: { seed: 901 } },
    { id: "secondary", op: "source.custom_secondary", inputs: {}, params: { repeat_count: 2 } },
    { id: "processor", op: "custom.backend", inputs: { value: "primary.value" }, params: { data_plane_backend: "auto", repeat_count: 91 } },
  ],
};

return {
  image: runCase(image, "loaded-image"),
  text: runCase(text, "loaded-text"),
  custom: runCase(custom, "loaded-custom", true),
  commits,
};
""",
            image=self.image_recipe,
            text=self.text_recipe,
            operations=self.operations,
        )

        for recipe_kind in ("image", "text", "custom"):
            with self.subTest(recipe=recipe_kind):
                rendered = result[recipe_kind]
                metadata = rendered["metadata"]
                self.assertEqual(metadata["seed"], 444)
                self.assertEqual(metadata["repeat_count"], 7)
                self.assertEqual(metadata["data_plane_backend"], "cpp_native")
                self.assertIs(metadata["measure_codec_timing"], False)
                self.assertEqual(
                    {
                        key: metadata["codec_timing"][key]
                        for key in ("enabled", "warmup_runs", "timed_runs")
                    },
                    {"enabled": False, "warmup_runs": 2, "timed_runs": 5},
                )
                self.assertTrue(rendered["topologyPreserved"])
                self.assertEqual(
                    rendered["explicitSeedsAfter"],
                    rendered["explicitSeedsBefore"],
                )
                self.assertGreaterEqual(len(rendered["repeatTargets"]), 1)
                self.assertEqual(
                    set(rendered["repeatValues"].values()),
                    {7},
                )
                self.assertGreaterEqual(len(rendered["backendTargets"]), 1)
                self.assertEqual(
                    set(rendered["backendValues"].values()),
                    {"cpp_native"},
                )
                self.assertTrue(rendered["nonSourceRepeatUnchanged"])
                self.assertEqual(
                    [binding["param"] for binding in rendered["remainingBindings"]],
                    ["seed"],
                )
                self.assertFalse(rendered["hasLegacySweeps"])

        self.assertEqual(result["image"]["metadata"]["codec_timing"]["runner"], "pytorch")
        self.assertEqual(result["text"]["metadata"]["codec_timing"]["runner"], "local_python")
        self.assertEqual(result["custom"]["metadata"]["codec_timing"]["runner"], "custom-runner")

    def test_universal_recipe_name_rejects_blank_and_trims_valid_input(self) -> None:
        result = self._run_node(
            r"""
function control(value) {
  const listeners = {};
  return {
    type: "text",
    value,
    addEventListener: (event, callback) => { listeners[event] = callback; },
    getAttribute: () => null,
    fire: (event) => listeners[event] && listeners[event](),
  };
}

const name = control("original_recipe");
els.settingsBody = {
  querySelectorAll: () => [],
  querySelector: (selector) => selector === '[data-recipe-setting="name"]' ? name : null,
};
state.operations = [];
state.editRecipe = {
  schema_version: 1,
  name: "original_recipe",
  metadata: {},
  steps: [],
};
let commits = 0;
let notices = [];
commitRecipeValueEdit = () => { commits += 1; };
refreshUniversalRecipeSettingsDialog = () => {};
notify = (message) => { notices.push(message); };

bindUniversalRecipeSettingsDialog();
name.value = "   \t  ";
name.fire("change");
const blank = {
  recipeName: state.editRecipe.name,
  inputValue: name.value,
  commits,
  notices: [...notices],
};

name.value = "   trimmed_recipe_name   ";
name.fire("change");
return {
  blank,
  valid: {
    recipeName: state.editRecipe.name,
    inputValue: name.value,
    commits,
    notices,
  },
};
"""
        )

        self.assertEqual(result["blank"]["recipeName"], "original_recipe")
        self.assertEqual(result["blank"]["inputValue"], "original_recipe")
        self.assertEqual(result["blank"]["commits"], 0)
        self.assertTrue(result["blank"]["notices"])
        self.assertEqual(result["valid"]["recipeName"], "trimmed_recipe_name")
        self.assertEqual(result["valid"]["inputValue"], "trimmed_recipe_name")
        self.assertEqual(result["valid"]["commits"], 1)


if __name__ == "__main__":
    unittest.main()
