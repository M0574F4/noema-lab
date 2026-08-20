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
RECIPE_PATHS = {
    "csi": "recipes/csi_feedback_sionna_train.yaml",
    "joint": "recipes/deepjscc_kodak_awgn_train.yaml",
    "text": "recipes/text_bart_jscc_clean.yaml",
    "classification": "recipes/task_classification_smoke.yaml",
    "vqa": "recipes/vqa_goal_oriented_smoke.yaml",
    "detection": "recipes/yolo_coco_detection.yaml",
    "segmentation": "recipes/yolo_coco_segmentation.yaml",
    "retrieval": "recipes/clip_retrieval_smoke.yaml",
    "generation": "recipes/diffusion_flickr8k_generation.yaml",
    "image": "recipes/compressai_kodak_default.yaml",
    "contract": "recipes/semantic_artifacts_smoke.yaml",
}


@unittest.skipUnless(shutil.which("node"), "Node.js is required for presentation registry tests")
class RecipePresentationRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipes = {
            key: load_recipe(ROOT / path).to_dict()
            for key, path in RECIPE_PATHS.items()
        }
        cls.recipes["contract"]["metadata"]["ui_configured"] = True
        registry = build_registry()
        operation_ids = {
            step["op"]
            for recipe in cls.recipes.values()
            for step in recipe["steps"]
        }
        cls.operations = [
            registry.get(operation_id).describe()
            for operation_id in sorted(operation_ids)
        ]

    def _run_node(self, body: str, **payload: object) -> object:
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
            input=json.dumps(payload),
        )
        if completed.returncode:
            self.fail(completed.stderr)
        return json.loads(completed.stdout)

    def test_registry_is_versioned_ordered_and_has_one_fallback(self) -> None:
        result = self._run_node(
            r"""
return {
  contractVersion: RECIPE_PRESENTATION_CONTRACT_VERSION,
  facetContractVersion: RECIPE_PRESENTATION_FACET_CONTRACT_VERSION,
  facetIds: RECIPE_PRESENTATION_FACETS.map((facet) => facet.id),
  adapters: RECIPE_PRESENTATION_ADAPTERS.map((adapter) => ({
    id: adapter.id,
    version: adapter.version,
    order: adapter.order,
    guided: adapter.guided,
    fallback: Boolean(adapter.fallback),
    defaultView: adapter.defaultView,
    facetIds: Array.from(adapter.facetIds),
  })),
};
"""
        )

        self.assertEqual(result["contractVersion"], 1)
        self.assertEqual(result["facetContractVersion"], 1)
        self.assertEqual(
            result["facetIds"],
            ["data", "method", "communication", "evaluation"],
        )
        adapters = result["adapters"]
        self.assertEqual(len({item["id"] for item in adapters}), len(adapters))
        self.assertEqual(
            [item["order"] for item in adapters],
            sorted(item["order"] for item in adapters),
        )
        self.assertTrue(all(item["version"] > 0 for item in adapters))
        self.assertEqual(sum(item["fallback"] for item in adapters), 1)
        self.assertTrue(adapters[-1]["fallback"])
        self.assertEqual(adapters[-1]["id"], "generic-graph")
        self.assertEqual(adapters[-1]["defaultView"], "blocks")
        for adapter in adapters[:-1]:
            self.assertTrue(adapter["guided"])
            self.assertEqual(adapter["defaultView"], "blocks")
            self.assertEqual(adapter["facetIds"], result["facetIds"])

    def test_adapter_resolution_depends_only_on_recipe_content(self) -> None:
        result = self._run_node(
            r"""
const expectedRecipes = __payload.recipes;
const resolved = {};
for (const [key, recipe] of Object.entries(expectedRecipes)) {
  const before = JSON.stringify(recipe);
  resolved[key] = {
    adapter: resolveRecipePresentation(recipe).adapter.id,
    defaultView: defaultRecipeConfiguratorView(recipe),
    unchanged: before === JSON.stringify(recipe),
  };
}
const image = expectedRecipes.image;
const editorRoutes = [];
for (const editorPreference of ["image", "text", "task", "graph", ""]) {
  state.selectedRecipe = { key: editorPreference || "none", recipe: image, editorPreference };
  state.selectedRecipeKey = state.selectedRecipe.key;
  state.editRecipe = image;
  state.editRecipeKey = state.selectedRecipe.key;
  editorRoutes.push(resolveRecipePresentation(image).adapter.id);
}
const optedOut = JSON.parse(JSON.stringify(image));
optedOut.metadata = { ...(optedOut.metadata || {}), ui_configured: false };
return {
  resolved,
  editorRoutes,
  optedOut: resolveRecipePresentation(optedOut).adapter.id,
  optedOutView: defaultRecipeConfiguratorView(optedOut),
};
""",
            recipes=self.recipes,
        )

        expected = {
            "csi": "csi-feedback",
            "joint": "joint-image-jscc",
            "text": "text-semantic",
            "classification": "task-builder",
            "vqa": "task-builder",
            "detection": "task-builder",
            "segmentation": "task-builder",
            "retrieval": "task-builder",
            "generation": "task-builder",
            "image": "layered-image",
            "contract": "task-contract",
        }
        self.assertEqual(
            {key: value["adapter"] for key, value in result["resolved"].items()},
            expected,
        )
        self.assertTrue(all(value["unchanged"] for value in result["resolved"].values()))
        self.assertTrue(all(value["defaultView"] == "blocks" for value in result["resolved"].values()))
        self.assertEqual(result["editorRoutes"], ["layered-image"] * 5)
        self.assertEqual(result["optedOut"], "generic-graph")
        self.assertEqual(result["optedOutView"], "blocks")

    def test_every_registered_recipe_renders_the_universal_blocks_surface(self) -> None:
        result = self._run_node(
            r"""
state.operations = __payload.operations;
state.recipeTemplates = [];
state.researchCatalog = null;
state.trainedArtifacts = [];
els.recipeConfigurator = { innerHTML: "", querySelectorAll: () => [] };
decorateRecipeConfigurator = () => {};

const rendered = {};
for (const [key, sourceRecipe] of Object.entries(__payload.recipes)) {
  const recipe = JSON.parse(JSON.stringify(sourceRecipe));
  const tab = {
    key,
    recipe,
    configuratorView: "experiment",
    preserveTopology: false,
    workingCopy: true,
    editorPreference: "graph",
  };
  state.recipes = [tab];
  state.selectedRecipe = tab;
  state.selectedRecipeKey = key;
  state.editRecipe = recipe;
  state.editRecipeKey = key;
  state.selectedNodeId = null;
  const before = JSON.stringify(recipe);
  renderUnifiedConfigurator();
  rendered[key] = {
    adapter: resolveRecipePresentation(recipe).adapter.id,
    defaultView: defaultRecipeConfiguratorView(recipe),
    activeView: activeRecipeConfiguratorView(recipe),
    markup: els.recipeConfigurator.innerHTML,
    stepCount: recipe.steps.length,
    unchanged: before === JSON.stringify(recipe),
  };
}
return rendered;
""",
            recipes=self.recipes,
            operations=self.operations,
        )

        for key, row in result.items():
            markup = row["markup"]
            self.assertEqual(row["defaultView"], "blocks", key)
            self.assertEqual(row["activeView"], "blocks", key)
            self.assertIn("data-configurator-block-list", markup, key)
            self.assertEqual(
                markup.count('class="schema-step-card'),
                row["stepCount"],
                key,
            )
            self.assertNotIn("data-configurator-view-switch", markup, key)
            self.assertNotIn("data-configurator-view-button", markup, key)
            self.assertNotIn(">Experiment<", markup, key)
            self.assertTrue(row["unchanged"], key)


if __name__ == "__main__":
    unittest.main()
