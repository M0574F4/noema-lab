from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest

from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
DEEPJSCC_RECIPE = ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"
LAYERED_RECIPE = ROOT / "recipes" / "compressai_kodak_default.yaml"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the configurator UI tests")
class JointSourceChannelConfiguratorTests(unittest.TestCase):
    def _run_node(self, body: str) -> dict:
        body_source = textwrap.dedent(body)
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.replace(/\ninit\(\);\s*$/, "\n");
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
            vm.runInContext(source + "\n" + {json.dumps(body_source)}, sandbox);
            process.stdout.write(JSON.stringify(sandbox.__result));
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

    def test_deepjscc_high_level_panel_edits_the_existing_recipe(self):
        recipe = load_recipe(DEEPJSCC_RECIPE).to_dict()
        for step in recipe["steps"]:
            if step["id"] in {"sender", "receiver"}:
                step["params"].update(
                    {
                        "checkpoint_path": "obsolete.npz",
                        "checkpoint_sha256": "0" * 64,
                    }
                )
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in sorted({step["op"] for step in recipe["steps"]})
        ]
        manifest_path = "differentiable_exports/deepjscc_demo/trained_artifact.yaml"
        artifact = {
            "id": "deepjscc.demo",
            "label": "Learned DeepJSCC demo",
            "manifest_path": manifest_path,
            "ready": True,
            "artifact": {"format": "onnx", "sha256": "a" * 64},
            "application": {"mode": "all_group_bindings"},
            "compatible_operations": [
                {
                    "operation": "model.deepjscc_external_encode",
                    "binding_group": "deepjscc_sender_receiver",
                    "role": "encoder",
                    "preferred_step_id": "sender",
                    "required_inputs": ["images"],
                    "params": {
                        "runtime": "learned_artifact",
                        "artifact_manifest_path": manifest_path,
                        "artifact_entrypoint": "encoder",
                    },
                },
                {
                    "operation": "model.deepjscc_external_decode",
                    "binding_group": "deepjscc_sender_receiver",
                    "role": "decoder",
                    "preferred_step_id": "receiver",
                    "required_inputs": ["symbols"],
                    "params": {
                        "runtime": "learned_artifact",
                        "artifact_manifest_path": manifest_path,
                        "artifact_entrypoint": "decoder",
                    },
                },
            ],
        }
        result = self._run_node(
            rf"""
              state.operations = {json.dumps(operations)};
              state.editRecipe = {json.dumps(recipe)};
              state.editRecipeKey = "working:deepjscc";
              state.selectedRecipe = {{ key: state.editRecipeKey, recipe: state.editRecipe }};
              state.selectedNodeId = null;
              state.trainedArtifacts = [{json.dumps(artifact)}];
              els.recipeConfigurator = {{ innerHTML: "", querySelectorAll: () => [] }};
              bindJointSourceChannelConfigurator = () => {{}};
              const topology = (value) => (value.steps || []).map((step) => ({{
                id: step.id,
                op: step.op,
                inputs: step.inputs || {{}},
              }}));
              const before = topology(state.editRecipe);
              renderJointSourceChannelConfigurator(state.editRecipe);
              const markup = els.recipeConfigurator.innerHTML;
              const afterRender = topology(state.editRecipe);

              const routeCalls = [];
              decorateRecipeConfigurator = () => {{}};
              renderRecipeBlocksConfigurator = () => routeCalls.push("blocks");
              state.selectedNodeId = null;
              renderUnifiedConfigurator();
              state.selectedNodeId = "sender";
              renderUnifiedConfigurator();

              globalThis.__result = {{ markup, before, afterRender, routeCalls }};
            """
        )

        self.assertEqual(result["before"], result["afterRender"])
        self.assertEqual(result["routeCalls"], ["blocks", "blocks"])
        markup = result["markup"]
        facet_positions = [
            markup.index(f'data-config-facet="{facet}"')
            for facet in ("data", "method", "communication", "evaluation")
        ]
        self.assertEqual(facet_positions, sorted(facet_positions))
        for facet in ("data", "method", "communication", "evaluation"):
            self.assertEqual(markup.count(f'data-config-facet="{facet}"'), 1)
        for heading in ("Data", "JSCC Model", "Transmit Power", "Wireless Channel", "Evaluation Metrics"):
            self.assertIn(f"<h3>{heading}</h3>", markup)
        self.assertIn("research-section", markup)
        for parameter in (
            "dataset",
            "dataset_dir",
            "image_ids",
            "resize_shorter_side",
            "crop_size",
            "repeat_count",
            "target_power",
            "channel",
            "noise_mode",
            "wireless_backend",
            "data_plane_backend",
            "seed",
        ):
            self.assertIn(f'data-schema-param-name="{parameter}"', markup)
        self.assertIn('data-joint-symbol-runtime', markup)
        self.assertIn("Learned artifact (trained or reference)", markup)
        self.assertNotIn('value="learned_checkpoint"', markup)
        self.assertNotIn("Reference checkpoint", markup)
        self.assertIn('data-joint-symbol-snr="wireless_channel"', markup)
        self.assertIn('data-schema-trained-artifact="sender"', markup)
        self.assertIn("sender + receiver", markup)
        self.assertIn("PSNR", markup)
        self.assertIn("MSE", markup)
        self.assertNotIn("<h2>Working recipe</h2>", markup)

    def test_runtime_and_snr_changes_are_atomic_value_edits(self):
        recipe = load_recipe(DEEPJSCC_RECIPE).to_dict()
        for step in recipe["steps"]:
            if step["id"] in {"sender", "receiver"}:
                step["params"].update(
                    {
                        "checkpoint_path": "obsolete.npz",
                        "checkpoint_sha256": "0" * 64,
                        "checkpoint_format": "noema_deepjscc_reference_cnn_npz_v1",
                    }
                )
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in (
                "model.deepjscc_external_encode",
                "model.deepjscc_external_decode",
                "wireless.channel",
            )
        ]
        result = self._run_node(
            rf"""
              state.operations = {json.dumps(operations)};
              state.editRecipe = {json.dumps(recipe)};
              state.trainedArtifacts = [];
              const topology = (value) => (value.steps || []).map((step) => ({{
                id: step.id,
                op: step.op,
                inputs: step.inputs || {{}},
              }}));
              const before = topology(state.editRecipe);
              globalThis.__commitCount = 0;
              commitRecipeValueEdit = () => {{ globalThis.__commitCount += 1; }};
              setJointSourceChannelRuntime("learned_artifact");
              setJointSourceChannelSnrSpec("wireless_channel", "8,10,12");
              globalThis.__result = {{
                before,
                after: topology(state.editRecipe),
                sender: findStep("sender").params,
                receiver: findStep("receiver").params,
                channel: findStep("wireless_channel").params,
                matrix: state.editRecipe.metadata.matrix,
                commitCount: globalThis.__commitCount,
              }};
            """
        )

        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["commitCount"], 2)
        for endpoint in (result["sender"], result["receiver"]):
            self.assertEqual(endpoint["runtime"], "learned_artifact")
            self.assertNotIn("checkpoint_path", endpoint)
            self.assertNotIn("checkpoint_sha256", endpoint)
            self.assertNotIn("checkpoint_format", endpoint)
        self.assertEqual(result["channel"]["snr_db"], 8)
        self.assertEqual(
            result["matrix"],
            {
                "dimensions": {"wireless_channel.snr_db": [8, 10, 12]},
                "step_params": {
                    "wireless_channel": {
                        "snr_db": {"matrix": "wireless_channel.snr_db"}
                    }
                },
            },
        )

    def test_capability_routing_keeps_supported_templates_out_of_generic_fallback(self):
        deepjscc = load_recipe(DEEPJSCC_RECIPE).to_dict()
        layered = load_recipe(LAYERED_RECIPE).to_dict()
        result = self._run_node(
            rf"""
              const deepjscc = {json.dumps(deepjscc)};
              const layered = {json.dumps(layered)};
              const unsupported = {{
                schema_version: 1,
                name: "custom_graph",
                execution_profile: {{ id: "custom", version: 1 }},
                steps: [{{ id: "data", op: "source.image_dataset", params: {{}} }}],
              }};
              const explicitlyGeneric = JSON.parse(JSON.stringify(layered));
              explicitlyGeneric.metadata = {{ ...(explicitlyGeneric.metadata || {{}}), ui_configured: false }};
              globalThis.__result = {{
                deepjscc: recipeUsesUiConfigurator(deepjscc),
                layered: recipeUsesUiConfigurator(layered),
                unsupported: recipeUsesUiConfigurator(unsupported),
                explicitlyGeneric: recipeUsesUiConfigurator(explicitlyGeneric),
                preserveWorkingCopy: recipePreservesWorkingTopology({{ metadata: {{ ui_working_copy: true }} }}),
              }};
            """
        )

        self.assertTrue(result["deepjscc"])
        self.assertTrue(result["layered"])
        self.assertFalse(result["unsupported"])
        self.assertFalse(result["explicitlyGeneric"])
        self.assertTrue(result["preserveWorkingCopy"])

    def test_catalog_open_and_run_preserve_the_loaded_working_topology(self):
        source = APP_JS.read_text(encoding="utf-8")
        catalog_open = source.split("async function openRecipeFromLibrary", 1)[1].split(
            "function renderRecipes", 1
        )[0]

        self.assertIn("preserveTopology: true", catalog_open)
        self.assertNotIn("ui_preserve_topology", catalog_open)
        self.assertNotIn("ui_configured: false", catalog_open)
        self.assertIn(
            "if (!recipeUsesUiConfigurator(recipe.recipe) || recipePreservesWorkingTopology(recipe.recipe))",
            source,
        )


if __name__ == "__main__":
    unittest.main()
