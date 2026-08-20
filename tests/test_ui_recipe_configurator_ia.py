from __future__ import annotations

import json
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
CSI_RECIPE = ROOT / "recipes" / "csi_feedback_sionna_train.yaml"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for configurator UI tests")
class RecipeConfiguratorInformationArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_js = APP_JS.read_text(encoding="utf-8")
        cls.image_recipe = load_recipe(IMAGE_RECIPE).to_dict()
        cls.csi_recipe = load_recipe(CSI_RECIPE).to_dict()
        registry = build_registry()
        operation_ids = {
            step["op"]
            for recipe in (cls.image_recipe, cls.csi_recipe)
            for step in recipe["steps"]
        } | {
            "model.compressai_encode",
            "model.compressai_decode",
            "model.diffusers_autoencoderkl_encode",
            "model.diffusers_autoencoderkl_decode",
            "model.symbol_power_allocator",
            "wireless.channel",
            "wireless.ofdm_channel_state",
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

    def test_configurator_has_no_experiment_blocks_mode_switch(self) -> None:
        """The removed mode switch must not remain as hidden or dead UI chrome."""
        self.assertNotIn("data-configurator-view-switch", self.app_js)
        self.assertNotIn("data-configurator-view-button", self.app_js)

    def test_default_and_template_recipes_always_render_the_same_blocks_view(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
els.recipeConfigurator = { innerHTML: "", querySelectorAll: () => [] };
decorateRecipeConfigurator = () => {};

function renderFor(recipe, key, path) {
  const tab = {
    key,
    recipe,
    templatePath: path,
    // A stale pre-migration value must not bring the old Experiment view back.
    configuratorView: "experiment",
    preserveTopology: true,
    workingCopy: true,
  };
  state.recipes = [tab];
  state.selectedRecipe = tab;
  state.selectedRecipeKey = key;
  state.editRecipe = recipe;
  state.editRecipeKey = key;
  state.selectedNodeId = null;
  renderUnifiedConfigurator();
  return {
    markup: els.recipeConfigurator.innerHTML,
    defaultView: typeof defaultRecipeConfiguratorView === "function"
      ? defaultRecipeConfiguratorView(recipe)
      : "blocks",
    activeView: typeof activeRecipeConfiguratorView === "function"
      ? activeRecipeConfiguratorView(recipe)
      : "blocks",
  };
}

const image = renderFor(
  JSON.parse(JSON.stringify(__payload.image)),
  "default-image",
  "recipes/compressai_kodak_default.yaml",
);
const csi = renderFor(
  JSON.parse(JSON.stringify(__payload.csi)),
  "template-csi",
  "recipes/csi_feedback_sionna_train.yaml",
);
const custom = renderFor({
  schema_version: 1,
  name: "custom_graph",
  execution_profile: { id: "custom", version: 1 },
  metadata: { ui_configured: false },
  steps: [],
}, "custom", "");
return { image, csi, custom };
""",
            operations=self.operations,
            image=self.image_recipe,
            csi=self.csi_recipe,
        )

        for rendered in result.values():
            markup = rendered["markup"]
            self.assertIn("data-configurator-block-list", markup)
            self.assertIn("Execution blocks", markup)
            self.assertLess(
                markup.index("Execution blocks"),
                markup.index("recipe-purpose-chip"),
                "the purpose chip should follow the Execution blocks heading",
            )
            self.assertNotIn("data-add-graph-block", markup)
            self.assertNotIn(
                "Inspect exact operation contracts, connections, and parameters.",
                markup,
            )
            self.assertNotIn(
                "Select a block here or in the graph to edit the exact operation contract, connections, and parameters.",
                markup,
            )
            self.assertNotIn("data-configurator-view-switch", markup)
            self.assertNotIn("data-configurator-view-button", markup)
            self.assertNotIn(">Experiment<", markup)
            self.assertEqual(rendered["defaultView"], "blocks")
            self.assertEqual(rendered["activeView"], "blocks")

        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("renderSelectedBlockConfigurator", source)
        self.assertNotIn("data-add-graph-block", source)
        self.assertIn("data-graph-add-block", source)

    def test_selected_graph_block_keeps_every_card_and_preserves_open_cards(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
const recipe = JSON.parse(JSON.stringify(__payload.image));
const persistedId = recipe.steps[0].id;
const selectedId = recipe.steps[Math.floor(recipe.steps.length / 2)].id;
els.recipeConfigurator = {
  innerHTML: "",
  querySelectorAll: () => [],
};
decorateRecipeConfigurator = () => {};
const tab = {
  key: "working:open-state",
  recipe,
  configuratorOpenStepIds: [persistedId],
};
state.recipes = [tab];
state.selectedRecipe = tab;
state.selectedRecipeKey = tab.key;
state.editRecipe = recipe;
state.editRecipeKey = tab.key;
state.selectedNodeId = selectedId;
renderUnifiedConfigurator();
return { markup: els.recipeConfigurator.innerHTML, persistedId, selectedId, stepCount: recipe.steps.length };
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        markup = result["markup"]
        self.assertEqual(markup.count('class="schema-step-card'), result["stepCount"])
        self.assertEqual(markup.count('class="schema-step-card selected"'), 1)
        self.assertEqual(markup.count('aria-current="true"'), 1)
        self.assertIn(
            f'class="schema-step-card selected" data-schema-step="{result["selectedId"]}" open',
            markup,
        )
        self.assertIn(
            f'data-schema-step="{result["persistedId"]}" open',
            markup,
        )

    def test_dataset_and_linked_codec_controls_have_single_block_owners(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
els.recipeConfigurator = { innerHTML: "", querySelectorAll: () => [] };

const image = JSON.parse(JSON.stringify(__payload.image));
state.editRecipe = image;
renderRecipeBlocksConfigurator(image);
const imageMarkup = els.recipeConfigurator.innerHTML;
const imageConfig = recipeConfigFromRecipe(image);
const imageProfile = codecProfile(imageConfig.codec, imageConfig);
const codecDialogMarkup = settingsDialogBody("codec", imageConfig, imageProfile).html;

const diffusers = {
  schema_version: 1,
  name: "paired_diffusers",
  execution_profile: { id: "layered_digital", version: 1 },
  metadata: { codec_profile: "diffusers_autoencoderkl" },
  steps: [
    { id: "data", op: "source.image_dataset", inputs: {}, params: { dataset: "kodak", dataset_dir: ".noema/datasets/kodak", crop_size: 0 } },
    { id: "sender", op: "model.diffusers_autoencoderkl_encode", inputs: { images: "data.images" }, params: { model_id: "old/model", scale_latents: true, scaling_factor: 0.18 } },
    { id: "receiver", op: "model.diffusers_autoencoderkl_decode", inputs: { latents: "sender.latents" }, params: { model_id: "old/model", scale_latents: true, scaling_factor: 0.18 } },
    { id: "evaluation", op: "metrics.image_reconstruction", inputs: { reference: "data.images", reconstruction: "receiver.images" }, params: {} },
  ],
};
state.editRecipe = diffusers;
renderRecipeBlocksConfigurator(diffusers);
const diffusersMarkup = els.recipeConfigurator.innerHTML;
const synced = syncSchemaCodecOwnerParameter(diffusers, diffusers.steps[1], "model_id", "new/model");
setRecipeSweepOnRecipe(diffusers, "sender.scaling_factor", "0.12,0.18,0.24");
syncSchemaCodecOwnerParameter(diffusers, diffusers.steps[1], "scaling_factor", 0.12);
return { imageMarkup, codecDialogMarkup, diffusersMarkup, diffusers, synced, matrix: diffusers.metadata.matrix };
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        image_markup = result["imageMarkup"]
        def card_markup(markup: str, step_id: str) -> str:
            remainder = markup.split(f'data-schema-step="{step_id}"', 1)[1]
            return remainder.split('<details class="schema-step-card', 1)[0]

        data_card = card_markup(image_markup, "data")
        sender_card = card_markup(image_markup, "sender")
        receiver_card = card_markup(image_markup, "receiver")
        self.assertIn('data-schema-step-param="data"', data_card)
        self.assertIn('data-schema-param-name="dataset_dir"', data_card)
        self.assertIn('data-schema-param-name="crop_size"', data_card)
        self.assertEqual(image_markup.count("data-schema-codec-family="), 1)
        self.assertIn('data-schema-codec-family="sender"', sender_card)
        linked_sender = sender_card.split("Linked settings", 1)[1].split(
            'class="schema-field-group-title">Parameters', 1
        )[0]
        endpoint_sender = sender_card.split(
            'class="schema-field-group-title">Parameters', 1
        )[1]
        self.assertIn('data-schema-linked-codec-contract="compressai"', linked_sender)
        self.assertIn("data-compressai-model-dropdown", linked_sender)
        self.assertIn(
            'data-ui-tooltip="Selecting more than one model runs one variant per model',
            linked_sender,
        )
        self.assertNotIn(
            '<div class="field-hint">Selecting more than one model',
            linked_sender,
        )
        for linked_name in ("quality", "metric", "pretrained"):
            self.assertIn(f'data-setting="codecParams.encoder.{linked_name}"', linked_sender)
            self.assertNotIn(f'data-schema-param-name="{linked_name}"', endpoint_sender)
        self.assertNotIn('data-schema-param-name="model"', endpoint_sender)
        self.assertIn("Model contract", linked_sender)
        self.assertIn("encoder + decoder", sender_card)
        self.assertIn('data-schema-owner-settings="codec"', sender_card)
        self.assertIn("Advanced codec settings", sender_card)
        self.assertIn("Open advanced codec settings", sender_card)
        self.assertIn(
            'class="icon-button schema-linked-settings-button"', sender_card
        )
        self.assertIn(
            'data-ui-tooltip="Advanced codec settings"', sender_card
        )
        self.assertNotIn(">Advanced codec settings<", sender_card)
        self.assertNotIn("Model, checkpoint &amp; runtime", sender_card)
        self.assertNotIn('data-replace-schema-step="sender"', sender_card)
        self.assertNotIn("Advanced block actions", sender_card)
        self.assertLess(
            sender_card.index("Linked settings"),
            sender_card.index('class="schema-field-group-title">Parameters'),
        )
        self.assertIn('data-schema-codec-decoder="receiver"', receiver_card)
        self.assertIn("decoder endpoint", receiver_card)
        self.assertIn("Managed by Encoder · sender", receiver_card)
        self.assertIn("bmshj2018_hyperprior", receiver_card)
        self.assertIn("Advanced codec settings", receiver_card)
        self.assertNotIn(">Advanced codec settings<", receiver_card)
        self.assertIn('data-schema-owner-settings="codec"', receiver_card)
        self.assertNotIn("data-schema-codec-family=", receiver_card)
        self.assertNotIn('data-replace-schema-step="receiver"', receiver_card)
        self.assertNotIn("Advanced block actions", receiver_card)
        self.assertLess(
            receiver_card.index("Linked settings"),
            receiver_card.index('class="schema-field-group-title">Parameters'),
        )
        self.assertIn('data-replace-schema-step="data"', data_card)
        self.assertIn("Advanced block actions", data_card)
        self.assertIn("Change operation&hellip;", data_card)
        self.assertIn(
            'class="schema-operation-contract schema-advanced-block-actions">',
            data_card,
        )
        self.assertEqual(image_markup.count('aria-label="Open advanced codec settings"'), 2)
        self.assertEqual(
            image_markup.count('data-ui-tooltip="Advanced codec settings"'), 2
        )

        styles = STYLES_CSS.read_text(encoding="utf-8")
        linked_layout = styles.split(".schema-linked-owner-control {", 1)[1].split(
            "}", 1
        )[0]
        linked_button = styles.split(".schema-linked-settings-button {", 1)[1].split(
            "}", 1
        )[0]
        self.assertIn("grid-template-columns: minmax(0, 1fr) 28px;", linked_layout)
        self.assertIn("width: 28px;", linked_button)
        self.assertIn("height: 28px;", linked_button)

        codec_dialog = result["codecDialogMarkup"]
        self.assertEqual(codec_dialog.count("data-compressai-model-dropdown"), 0)
        self.assertIn("Encoder Runtime &amp; Transform", codec_dialog)
        self.assertIn("Decoder Runtime &amp; Failure Policy", codec_dialog)
        self.assertNotIn(">Encoder</h3>", codec_dialog)

        diffusers_markup = result["diffusersMarkup"]
        diffusers_receiver_card = card_markup(diffusers_markup, "receiver")
        self.assertEqual(
            diffusers_markup.count('data-schema-param-name="model_id"'),
            1,
        )
        self.assertIn(
            'data-schema-step-param="sender" data-schema-param-name="model_id"',
            diffusers_markup,
        )
        self.assertNotIn(
            'data-schema-step-param="receiver" data-schema-param-name="model_id"',
            diffusers_markup,
        )
        self.assertIn("Managed by Encoder · sender", diffusers_receiver_card)
        self.assertIn("Advanced codec settings", diffusers_receiver_card)
        self.assertNotIn(
            'data-replace-schema-step="receiver"',
            diffusers_receiver_card,
        )
        self.assertEqual(result["synced"], ["sender", "receiver"])
        self.assertEqual(
            result["diffusers"]["steps"][1]["params"]["model_id"],
            "new/model",
        )
        self.assertEqual(
            result["diffusers"]["steps"][2]["params"]["model_id"],
            "new/model",
        )
        matrix = result["matrix"]
        self.assertEqual(
            matrix["dimensions"]["sender.scaling_factor"],
            [0.12, 0.18, 0.24],
        )
        self.assertEqual(
            matrix["step_params"]["sender"]["scaling_factor"],
            {"matrix": "sender.scaling_factor"},
        )
        self.assertEqual(
            matrix["step_params"]["receiver"]["scaling_factor"],
            {"matrix": "sender.scaling_factor"},
        )

    def test_graph_and_recipe_card_resolve_the_same_primary_block_name(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.researchCatalog = null;
state.trainedArtifacts = [];
state.editRecipe = JSON.parse(JSON.stringify(__payload.image));
const sender = state.editRecipe.steps.find((step) => step.id === "sender");
const graphSender = graphFromRecipePayload(state.editRecipe).nodes.find((node) => node.id === "sender");
const card = schemaDrivenStepMarkup(sender, 1, { recipe: state.editRecipe });
graphEditorToolbarMarkup = () => "";
nodeStatus = () => ({ kind: "idle", label: "idle" });
nodeProgress = () => null;
nodeFlowLabel = () => "images -> latents";
nodeEntropyInfo = () => null;
nodeDifferentiabilityInfo = () => null;
nodeRunnerSupportInfo = () => null;
nodeBackendInfo = () => null;
nodeEquivalenceInfo = () => null;
graphNodePortsMarkup = () => "";
nodeContractBadges = () => "";
nodeStatusIcon = () => "";
pendingGraphLinkMarkup = () => "";
bindGraphVisualizationSaveButton = () => {};
bindGraphEditorControls = () => {};
bindGraphPortControls = () => {};
els.graphSurface = { innerHTML: "", querySelectorAll: () => [] };
state.positions = {};
state.selectedRecipe = null;
renderGraph({ nodes: [graphSender], edges: [] });
return {
  cardName: canonicalBlockDisplayName(sender),
  graphName: canonicalBlockDisplayName(graphSender),
  card,
  graphMarkup: els.graphSurface.innerHTML,
  technicalId: graphSender.id,
};
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        self.assertEqual(result["cardName"], "Encoder")
        self.assertEqual(result["graphName"], result["cardName"])
        self.assertIn("<strong>Encoder</strong>", result["card"])
        self.assertIn("sender · model.compressai_analysis_encode", result["card"])
        self.assertIn("Block ID (technical)", result["card"])
        self.assertIn('<text class="node-title" x="10" y="17">Encoder</text>', result["graphMarkup"])
        self.assertNotIn('class="node-id"', result["graphMarkup"])
        self.assertIn("sender", result["graphMarkup"])
        self.assertEqual(result["technicalId"], "sender")

    def test_supported_compressai_vbr_stage_remains_in_linked_settings(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.researchCatalog = null;
state.trainedArtifacts = [];
const recipe = JSON.parse(JSON.stringify(__payload.image));
const sender = recipe.steps.find((step) => step.id === "sender");
const receiver = recipe.steps.find((step) => step.id === "receiver");
sender.op = "model.compressai_encode";
sender.params.vbr_stage = 2;
receiver.op = "model.compressai_decode";
state.editRecipe = recipe;
return { card: schemaDrivenStepMarkup(sender, 1, { recipe }) };
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        linked = result["card"].split("Linked settings", 1)[1].split(
            'class="schema-field-group-title">Parameters', 1
        )[0]
        endpoint = result["card"].split(
            'class="schema-field-group-title">Parameters', 1
        )[1]
        self.assertIn('data-setting="codecParams.encoder.vbr_stage"', linked)
        self.assertNotIn('data-schema-param-name="vbr_stage"', endpoint)

    def test_inline_compressai_model_selection_stays_in_panel_and_preserves_position(self) -> None:
        result = self._run_node(
            r"""
const selected = [
  { value: "bmshj2018_hyperprior" },
  { value: "mbt2018_mean" },
];
const dropdown = {
  querySelectorAll: (selector) => selector === "[data-compressai-model-option]:checked" ? selected : [],
};
const checkbox = { checked: true, closest: () => dropdown };
const config = { codecParams: { encoder: { model: "bmshj2018_hyperprior" } } };
const scroller = { scrollTop: 317 };
let stored = 0;
let applied = null;
let renders = 0;
let modals = 0;
recipeConfiguratorScrollContainer = () => scroller;
storeRecipeConfiguratorOpenBlockIds = () => { stored += 1; };
recipeConfigFromRecipe = () => config;
normalizeRecipeConfig = () => {};
applyRecipeConfig = (next, renderControls) => { applied = { next, renderControls }; };
renderRecipeControls = () => { renders += 1; scroller.scrollTop = 0; };
openSettingsDialog = () => { modals += 1; };
state.editRecipe = { name: "inline-model" };
updateCompressAiModelSelectionFromDropdown(checkbox, { surface: "blocks" });
return {
  model: config.codecParams.encoder.model,
  stored,
  renders,
  modals,
  scrollTop: scroller.scrollTop,
  applyRendered: applied && applied.renderControls,
};
"""
        )

        self.assertEqual(result["model"], "bmshj2018_hyperprior,mbt2018_mean")
        self.assertEqual(result["stored"], 1)
        self.assertEqual(result["renders"], 1)
        self.assertEqual(result["modals"], 0)
        self.assertEqual(result["scrollTop"], 317)
        self.assertFalse(result["applyRendered"])

    def test_inline_model_dropdown_can_escape_the_block_card_clip(self) -> None:
        styles = STYLES_CSS.read_text(encoding="utf-8")
        self.assertIn(".schema-step-card.has-open-dropdown", styles)
        rule = styles.split(".schema-step-card.has-open-dropdown", 1)[1].split("}", 1)[0]
        self.assertIn("overflow: visible", rule)
        self.assertIn('classList.toggle("has-open-dropdown", shouldOpen)', self.app_js)

    def test_graph_selection_expands_and_scrolls_without_replacing_panel(self) -> None:
        result = self._run_node(
            r"""
function classList(initial = []) {
  const values = new Set(initial);
  return {
    toggle: (name, enabled) => enabled ? values.add(name) : values.delete(name),
    contains: (name) => values.has(name),
  };
}
function summary(top) {
  const attributes = new Map();
  return {
    getBoundingClientRect: () => ({ top }),
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
    hasAttribute: (name) => attributes.has(name),
  };
}
function card(id, top, open = false) {
  const header = summary(top);
  return {
    id,
    open,
    header,
    classList: classList(),
    getAttribute: (name) => name === "data-schema-step" ? id : "",
    querySelector: (selector) => selector === ":scope > summary" ? header : null,
  };
}
const first = card("first", 80, true);
const middle = card("middle", 250, false);
const last = card("last", 420, false);
const cards = [first, middle, last];
const scrollCalls = [];
const scrollTail = { style: { height: "" } };
const scroller = {
  classList: { contains: (name) => name === "config-body" },
  scrollTop: 80,
  scrollHeight: 500,
  clientHeight: 400,
  clientTop: 1,
  getBoundingClientRect: () => ({ top: 20 }),
  scrollTo: (options) => scrollCalls.push(options),
};
els.recipeConfigurator = {
  parentElement: scroller,
  querySelectorAll: (selector) => selector === "[data-schema-step]" ? cards : [],
  querySelector: (selector) => selector === "[data-configurator-scroll-tail]" ? scrollTail : null,
};
globalThis.getComputedStyle = () => ({ paddingTop: "10px" });
window.matchMedia = () => ({ matches: false });
let graphRenders = 0;
let controlRenders = 0;
renderGraph = () => { graphRenders += 1; };
renderRecipeControls = () => { controlRenders += 1; };
state.graph = {};
state.editRecipe = { name: "selection", steps: cards.map((item) => ({ id: item.id })) };
const tab = {
  key: "working:selection",
  recipe: state.editRecipe,
  configuratorOpenStepIds: [first.id],
};
state.recipes = [tab];
state.selectedRecipe = tab;
state.selectedRecipeKey = tab.key;
state.editRecipeKey = tab.key;

selectNode("middle", true);
const afterClick = {
  selected: middle.classList.contains("selected"),
  current: middle.header.hasAttribute("aria-current"),
  middleOpen: middle.open,
  firstOpen: first.open,
  storedOpen: tab.configuratorOpenStepIds.slice(),
  scrollTailHeight: scrollTail.style.height,
  scrollCalls: scrollCalls.slice(),
};
selectNode("last", false);
const afterNoScrollSelection = {
  lastSelected: last.classList.contains("selected"),
  lastOpen: last.open,
  middleOpen: middle.open,
  storedOpen: tab.configuratorOpenStepIds.slice(),
  scrollCount: scrollCalls.length,
};
syncRecipeConfiguratorBlockSelection("");
return {
  afterClick,
  afterNoScrollSelection,
  afterClear: { lastSelected: last.classList.contains("selected"), lastOpen: last.open },
  graphRenders,
  controlRenders,
};
"""
        )

        self.assertEqual(
            result["afterClick"],
            {
                "selected": True,
                "current": True,
                "middleOpen": True,
                "firstOpen": True,
                "storedOpen": ["first", "middle"],
                "scrollTailHeight": "199px",
                "scrollCalls": [{"top": 299, "behavior": "smooth"}],
            },
        )
        self.assertEqual(
            result["afterNoScrollSelection"],
            {
                "lastSelected": True,
                "lastOpen": True,
                "middleOpen": True,
                "storedOpen": ["first", "middle", "last"],
                "scrollCount": 1,
            },
        )
        self.assertEqual(
            result["afterClear"],
            {"lastSelected": False, "lastOpen": True},
        )
        self.assertEqual(result["graphRenders"], 1)
        self.assertEqual(result["controlRenders"], 0)

    def test_open_block_state_is_scoped_to_each_recipe_tab(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
els.recipeConfigurator = { innerHTML: "", querySelectorAll: () => [] };
decorateRecipeConfigurator = () => {};

const source = JSON.parse(JSON.stringify(__payload.image));
const sharedId = source.steps[0].id;
const recipeA = { ...source, name: "recipe_a", steps: [source.steps[0]] };
const recipeB = JSON.parse(JSON.stringify(recipeA));
recipeB.name = "recipe_b";
const tabA = {
  key: "working:a",
  recipe: recipeA,
  configuratorOpenStepIds: [sharedId],
};
const tabB = {
  key: "working:b",
  recipe: recipeB,
  configuratorOpenStepIds: [],
};
state.recipes = [tabA, tabB];

function activate(tab) {
  state.selectedRecipe = tab;
  state.selectedRecipeKey = tab.key;
  state.editRecipe = tab.recipe;
  state.editRecipeKey = tab.key;
  state.selectedNodeId = null;
  renderUnifiedConfigurator();
  return els.recipeConfigurator.innerHTML.includes(`data-schema-step="${sharedId}" open`);
}

return {
  aFirst: activate(tabA),
  b: activate(tabB),
  aAgain: activate(tabA),
  aStored: tabA.configuratorOpenStepIds,
  bStored: tabB.configuratorOpenStepIds,
};
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        self.assertEqual(
            result,
            {
                "aFirst": True,
                "b": False,
                "aAgain": True,
                "aStored": ["data"],
                "bStored": [],
            },
        )

    def test_delayed_disclosure_toggle_updates_its_rendered_tab_only(self) -> None:
        result = self._run_node(
            r"""
const recipeA = { name: "a", steps: [{ id: "shared" }] };
const recipeB = { name: "b", steps: [{ id: "shared" }] };
const tabA = { key: "working:a", recipe: recipeA, configuratorOpenStepIds: [] };
const tabB = { key: "working:b", recipe: recipeB, configuratorOpenStepIds: [] };
state.recipes = [tabA, tabB];
state.selectedRecipe = tabA;
state.selectedRecipeKey = tabA.key;
state.editRecipe = recipeA;
state.editRecipeKey = tabA.key;

let toggleListener = null;
const card = {
  open: false,
  getAttribute: (name) => name === "data-schema-step" ? "shared" : "",
  addEventListener: (name, listener) => { if (name === "toggle") toggleListener = listener; },
};
els.recipeConfigurator = {
  querySelectorAll: (selector) => selector === "[data-schema-step]" ? [card] : [],
};
bindSchemaDrivenRecipeConfigurator();

state.selectedRecipe = tabB;
state.selectedRecipeKey = tabB.key;
state.editRecipe = recipeB;
state.editRecipeKey = tabB.key;
card.open = true;
toggleListener();
const afterOpen = { a: tabA.configuratorOpenStepIds.slice(), b: tabB.configuratorOpenStepIds.slice() };
card.open = false;
toggleListener();
return {
  afterOpen,
  afterClose: { a: tabA.configuratorOpenStepIds.slice(), b: tabB.configuratorOpenStepIds.slice() },
};
"""
        )

        self.assertEqual(
            result,
            {
                "afterOpen": {"a": ["shared"], "b": []},
                "afterClose": {"a": [], "b": []},
            },
        )

    def test_dragging_a_graph_node_does_not_trigger_click_navigation(self) -> None:
        result = self._run_node(
            r"""
const selections = [];
selectNode = (nodeId, rerender = true) => selections.push({ nodeId, rerender });
updatePendingGraphLinkPointer = () => {};
updateGraphGeometry = () => {};
state.positions = { middle: { x: 100, y: 60 } };
state.graphZoom = 1;
els.graphSurface = {
  classList: { remove: () => {} },
  scrollLeft: 0,
  scrollTop: 0,
};
els.resultsComparison = { querySelectorAll: () => [] };

function pointer(x, y) {
  return {
    button: 0,
    clientX: x,
    clientY: y,
    preventDefault: () => {},
    stopPropagation: () => {},
  };
}
function deliverClick(nodeId) {
  const event = {
    prevented: false,
    stopped: false,
    immediate: false,
    preventDefault() { this.prevented = true; },
    stopPropagation() { this.stopped = true; },
    stopImmediatePropagation() { this.immediate = true; },
  };
  const suppressed = consumeSuppressedGraphClick(event);
  if (!suppressed) selectNode(nodeId);
  return { suppressed, prevented: event.prevented, stopped: event.stopped, immediate: event.immediate };
}

startNodeDrag(pointer(10, 10), "middle");
const afterPointerDown = selections.slice();
onPointerMove(pointer(20, 10));
onPointerMove(pointer(10, 10));
onPointerUp(pointer(10, 10));
const afterDragClick = deliverClick("middle");
const afterDragSelections = selections.slice();

startNodeDrag(pointer(10, 10), "middle");
onPointerUp(pointer(12, 12));
const ordinaryClick = deliverClick("middle");
return {
  afterPointerDown,
  afterDragClick,
  afterDragSelections,
  ordinaryClick,
  finalSelections: selections,
};
"""
        )

        self.assertEqual(result["afterPointerDown"], [])
        self.assertEqual(
            result["afterDragClick"],
            {"suppressed": True, "prevented": True, "stopped": True, "immediate": True},
        )
        self.assertEqual(result["afterDragSelections"], [])
        self.assertEqual(
            result["ordinaryClick"],
            {"suppressed": False, "prevented": False, "stopped": False, "immediate": False},
        )
        self.assertEqual(result["finalSelections"], [{"nodeId": "middle", "rerender": True}])

    def test_block_cards_expose_complete_non_json_editing_affordances(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  ...__payload.operations,
  {
    id: "custom.extensible",
    name: "Extensible operation",
    status: "implemented",
    input_kinds: {},
    optional_input_kinds: {},
    output_kinds: { value: "json" },
    params_schema: {
      type: "object",
      properties: {},
      additionalProperties: true,
    },
  },
];
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
const recipe = JSON.parse(JSON.stringify(__payload.image));
recipe.metadata = { ...(recipe.metadata || {}), ui_configured: false };
recipe.steps[0].description = "Images entering the reconstruction pipeline.";
recipe.steps[0].params.future_toggle = true;
recipe.steps.push({
  id: "extension_block",
  op: "custom.extensible",
  inputs: {},
  params: {},
});
const before = JSON.stringify(recipe);
state.editRecipe = recipe;
state.editRecipeKey = "generic";
state.selectedRecipe = {
  key: "generic",
  recipe,
  preserveTopology: true,
};
state.selectedRecipeKey = "generic";
state.selectedNodeId = null;
els.recipeConfigurator = { innerHTML: "", querySelectorAll: () => [] };
renderRecipeBlocksConfigurator(recipe);
return {
  markup: els.recipeConfigurator.innerHTML,
  stepCount: recipe.steps.length,
  unchanged: before === JSON.stringify(recipe),
};
""",
            operations=self.operations,
            image=self.image_recipe,
        )

        markup = result["markup"]
        step_count = result["stepCount"]
        self.assertEqual(markup.count('class="schema-step-card'), step_count)
        self.assertEqual(markup.count("data-schema-step-description="), step_count)
        self.assertEqual(
            markup.count("data-replace-schema-step="),
            step_count - 2,
            "linked encoder/decoder topology is replaced atomically by Codec family",
        )
        self.assertNotIn('data-replace-schema-step="sender"', markup)
        self.assertNotIn('data-replace-schema-step="receiver"', markup)
        self.assertIn('data-replace-schema-step="extension_block"', markup)
        self.assertEqual(markup.count("Advanced block actions"), step_count - 2)
        self.assertEqual(markup.count("Change operation&hellip;"), step_count - 2)
        self.assertEqual(markup.count("data-add-schema-param="), 1)
        self.assertIn('data-add-schema-param="extension_block"', markup)
        self.assertNotIn('data-add-schema-param="data"', markup)
        self.assertIn("Images entering the reconstruction pipeline.", markup)
        self.assertIn('data-schema-param-name="future_toggle"', markup)
        self.assertIn("future toggle (extra)", markup)
        self.assertNotIn("data-schema-step-input", markup)
        self.assertNotIn('<div class="schema-field-group-title">Inputs</div>', markup)
        self.assertEqual(markup.count("data-remove-schema-step="), step_count)
        self.assertEqual(markup.count('class="action-icon-glyph"'), step_count)
        self.assertNotIn(">Remove block<", markup)
        self.assertTrue(result["unchanged"])

    def test_noise_and_allocator_snr_controls_only_appear_when_authoritative(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.editRecipe = { steps: [] };
state.trainedArtifacts = [];
const allocator = {
  id: "tx_power",
  op: "model.symbol_power_allocator",
  inputs: { channel_state: "channel_state.state" },
  params: { policy: "water_filling", target_power: 1 },
};
const channel = {
  id: "wireless_channel",
  op: "wireless.channel",
  inputs: {},
  params: { noise_mode: "fixed_variance", noise_variance: 0.2 },
};
const channelState = {
  id: "channel_state",
  op: "wireless.ofdm_channel_state",
  inputs: {},
  params: { noise_variance: 0.2, tdl_model: "A", capacity_multiplier: 16 },
};
state.editRecipe = { steps: [channelState, { ...channel, inputs: { channel_state: "channel_state.state" } }] };
const inheritedState = schemaDrivenStepMarkup(channelState, 0, { recipe: state.editRecipe });
const waterFilling = schemaDrivenStepMarkup(allocator, 0);
allocator.params.policy = "snr_sigmoid";
const sigmoidWithCsi = schemaDrivenStepMarkup(allocator, 0);
delete allocator.inputs.channel_state;
const sigmoidFallback = schemaDrivenStepMarkup(allocator, 0);
const fixedVariance = schemaDrivenStepMarkup(channel, 1);
channel.params.noise_mode = "snr_at_unit_power";
const snrControlled = schemaDrivenStepMarkup(channel, 1);
return { waterFilling, sigmoidWithCsi, sigmoidFallback, fixedVariance, snrControlled, inheritedState };
""",
            operations=self.operations,
        )

        self.assertNotIn('data-schema-param-name="snr_db"', result["waterFilling"])
        self.assertNotIn('data-schema-param-name="snr_db"', result["sigmoidWithCsi"])
        self.assertIn('data-schema-param-name="snr_db"', result["sigmoidFallback"])
        self.assertIn('data-schema-param-name="noise_variance"', result["fixedVariance"])
        self.assertNotIn('data-schema-param-name="snr_db"', result["fixedVariance"])
        self.assertIn('data-schema-param-name="snr_db"', result["snrControlled"])
        self.assertNotIn('data-schema-param-name="noise_variance"', result["snrControlled"])
        self.assertNotIn('data-schema-param-name="noise_variance"', result["inheritedState"])
        self.assertNotIn('data-schema-param-name="tdl_model"', result["inheritedState"])
        self.assertIn('data-schema-param-name="capacity_multiplier"', result["inheritedState"])

    def test_inactive_noise_parameter_is_not_sweepable_and_is_reported(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.trainedArtifacts = [];
const channel = {
  id: "wireless_channel",
  op: "wireless.channel",
  inputs: {},
  params: {
    channel: "awgn",
    noise_mode: "fixed_variance",
    noise_variance: 0.2,
    snr_db: 10,
  },
};
const recipe = {
  schema_version: 1,
  name: "inactive_snr_sweep",
  metadata: {
    matrix: {
      dimensions: { "wireless_channel.snr_db": [0, 10] },
      step_params: {
        wireless_channel: { snr_db: { matrix: "wireless_channel.snr_db" } },
      },
    },
  },
  steps: [channel],
};
state.editRecipe = recipe;
const options = recipeMatrixParameterOptions(recipe).map((item) => item.value);
const problem = recipeMatrixDefinitionProblem(recipe.metadata.matrix, recipe);
const paramIssues = schemaParamConfigurationIssues(channel, (operationById(channel.op) || {}).params_schema || {});
pruneInactiveSchemaParameters(channel, (operationById(channel.op) || {}).params_schema || {});
return {
  options,
  problem,
  paramIssues,
  params: channel.params,
  matrix: canonicalRecipeMatrix(recipe),
};
""",
            operations=self.operations,
        )

        self.assertNotIn("wireless_channel.snr_db", result["options"])
        self.assertIn("inactive parameter wireless_channel.snr_db", result["problem"])
        self.assertTrue(any("snr_db is inactive" in issue for issue in result["paramIssues"]))
        self.assertNotIn("snr_db", result["params"])
        self.assertIsNone(result["matrix"])

    def test_graph_toolbar_uses_icon_only_status_add_and_remove_actions(self) -> None:
        result = self._run_node(
            r"""
state.busy = false;
state.selectedNodeId = "sender";
state.selectedEdgeKey = null;
state.editRecipe = { execution_profile: { id: "layered_digital", version: 1 }, steps: [] };
graphRecipeConfigurationIssues = () => [];
const valid = graphEditorToolbarMarkup({});
graphRecipeConfigurationIssues = () => ["sender input is disconnected"];
const invalid = graphEditorToolbarMarkup({});
return { valid, invalid };
"""
        )

        markup = result["valid"]
        validation = markup.index("graph-validation-state")
        add = markup.index("data-graph-add-block")
        remove = markup.index("data-graph-remove-selected")
        self.assertLess(validation, add)
        self.assertLess(add, remove)
        self.assertNotIn("graph-execution-profile", markup)
        self.assertNotIn("Layered digital", markup)
        self.assertIn('aria-label="Valid topology"', markup)
        self.assertIn('aria-label="Add block"', markup)
        self.assertIn('aria-label="Remove selected block or link"', markup)
        self.assertIn("addBlockIconMarkup", self.app_js)
        self.assertIn("deleteIconMarkup", self.app_js)
        self.assertIn("topologyStatusIconMarkup", self.app_js)
        self.assertNotIn("<rect", markup)
        self.assertNotIn(">Add block<", markup)
        self.assertNotIn(">Remove selected<", markup)
        self.assertIn("graph-validation-state invalid", result["invalid"])
        self.assertIn('aria-label="1 issue"', result["invalid"])

    def test_custom_parameters_are_typed_and_only_allowed_by_extensible_schemas(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  {
    id: "custom.extensible",
    name: "Extensible",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: {
      type: "object",
      properties: { declared: { type: "string" } },
      additionalProperties: true,
    },
  },
  {
    id: "custom.strict",
    name: "Strict",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
  },
];
state.editRecipe = {
  schema_version: 1,
  name: "extensions",
  steps: [
    { id: "open", op: "custom.extensible", inputs: {}, params: {} },
    { id: "closed", op: "custom.strict", inputs: {}, params: {} },
  ],
};
const controls = {
  open: { name: { value: "options", focus: () => {} }, value: { value: '{"enabled":true,"count":2}' } },
  closed: { name: { value: "future", focus: () => {} }, value: { value: "true" } },
};
els.recipeConfigurator = {
  querySelector: (selector) => {
    const stepId = selector.includes('="open"') ? "open" : selector.includes('="closed"') ? "closed" : "";
    if (!stepId) return null;
    return selector.includes("param-name") ? controls[stepId].name : controls[stepId].value;
  },
};
const notices = [];
let commits = 0;
notify = (message) => notices.push(message);
commitRecipeValueEdit = () => { commits += 1; };

addSchemaExtensionParameter("open");
addSchemaExtensionParameter("open");
addSchemaExtensionParameter("closed");
controls.open.name.value = "bad name";
addSchemaExtensionParameter("open");

return { recipe: state.editRecipe, notices, commits };
"""
        )

        self.assertEqual(
            result["recipe"]["steps"][0]["params"]["options"],
            {"enabled": True, "count": 2},
        )
        self.assertEqual(result["recipe"]["steps"][1]["params"], {})
        self.assertEqual(result["commits"], 1)
        self.assertTrue(any("already exists" in notice for notice in result["notices"]))
        self.assertTrue(any("does not accept undeclared" in notice for notice in result["notices"]))
        self.assertTrue(any("parameter names must start" in notice for notice in result["notices"]))

    def test_operation_replacement_resets_and_prunes_incompatible_graph_state(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  {
    id: "test.old",
    name: "Old",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {},
    output_kinds: { old_values: "tensor.numpy" },
    params_schema: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["fast"], default: "fast" },
        gain: { type: "number", minimum: 0, default: 1 },
      },
      additionalProperties: false,
    },
  },
  {
    id: "test.new",
    name: "New",
    status: "implemented",
    input_kinds: {}, optional_input_kinds: {},
    output_kinds: { values: "tensor.numpy" },
    params_schema: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["safe"], default: "safe" },
        gain: { type: "number", minimum: 0, default: 1 },
      },
      additionalProperties: false,
    },
  },
  {
    id: "test.target",
    name: "Target",
    status: "implemented",
    input_kinds: { input: ["tensor.numpy"] }, optional_input_kinds: {},
    output_kinds: {},
    params_schema: { type: "object", properties: {}, additionalProperties: false },
  },
];
state.trainedArtifacts = [];
state.editRecipe = {
  schema_version: 1,
  name: "replace_safely",
  metadata: {
    matrix: {
      dimensions: { mode: ["fast", "slow"] },
      step_params: { subject: { mode: { matrix: "mode" } } },
    },
    trained_artifact_bindings: { subject: { id: "old-artifact" } },
  },
  dataset_capture: {
    taps: [{ id: "old_values", from: "subject.old_values" }],
    sweep: { "subject.gain": [1, 2] },
  },
  steps: [
    { id: "subject", description: "Keep this note", op: "test.old", inputs: {}, params: { mode: "fast", gain: 3 } },
    { id: "target", op: "test.target", inputs: { input: "subject.old_values" }, params: {} },
  ],
};
state.positions = {};
state.busy = false;
const notices = [];
commitRecipeTopologyEdit = () => {};
closeSettingsDialog = () => {};
notify = (message) => notices.push(message);
replaceRecipeOperation("subject", "test.new");
return { recipe: state.editRecipe, notices };
"""
        )

        recipe = result["recipe"]
        subject = recipe["steps"][0]
        self.assertEqual(subject["op"], "test.new")
        self.assertEqual(subject["description"], "Keep this note")
        self.assertEqual(subject["params"], {"mode": "safe", "gain": 3})
        self.assertEqual(recipe["steps"][1]["inputs"], {})
        self.assertNotIn("matrix", recipe["metadata"])
        self.assertNotIn("trained_artifact_bindings", recipe["metadata"])
        self.assertEqual(recipe["dataset_capture"]["taps"], [])
        self.assertNotIn("sweep", recipe["dataset_capture"])
        self.assertTrue(any("reset 1 incompatible parameter" in notice for notice in result["notices"]))

    def test_operation_replacement_preserves_compatible_extension_parameters(self) -> None:
        result = self._run_node(
            r"""
state.operations = [
  {
    id: "test.old", name: "Old", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: { type: "object", properties: { gain: { type: "number" } }, additionalProperties: true },
  },
  {
    id: "test.new", name: "New", status: "implemented",
    input_kinds: {}, optional_input_kinds: {}, output_kinds: {},
    params_schema: {
      type: "object",
      properties: { gain: { type: "number", default: 1 } },
      additionalProperties: { type: "integer" },
    },
  },
];
state.trainedArtifacts = [];
state.editRecipe = {
  schema_version: 1,
  name: "extension_replacement",
  metadata: {},
  steps: [{
    id: "subject",
    op: "test.old",
    inputs: {},
    params: { gain: 2, retained_extension: 7, rejected_extension: "wrong" },
  }],
};
state.positions = {};
const notices = [];
commitRecipeTopologyEdit = () => {};
closeSettingsDialog = () => {};
notify = (message) => notices.push(message);
replaceRecipeOperation("subject", "test.new");
return { step: state.editRecipe.steps[0], notices };
"""
        )

        self.assertEqual(
            result["step"]["params"],
            {"gain": 2, "retained_extension": 7},
        )
        self.assertTrue(
            any("reset 1 incompatible parameter" in notice for notice in result["notices"])
        )


if __name__ == "__main__":
    unittest.main()
