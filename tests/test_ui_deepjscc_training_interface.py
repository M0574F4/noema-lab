from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest

from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
RECIPE_PATH = ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"


class DeepJsccTrainingInterfaceRecipeTests(unittest.TestCase):
    def test_training_recipe_declares_valid_export_only_interfaces(self):
        recipe = load_recipe(RECIPE_PATH)
        validate_recipe_against_registry(recipe, build_registry())
        endpoints = {
            step.id: step
            for step in recipe.steps
            if step.op
            in {"model.deepjscc_external_encode", "model.deepjscc_external_decode"}
        }
        self.assertEqual(set(endpoints), {"sender", "receiver"})
        self.assertEqual(
            {step.params.get("runtime") for step in endpoints.values()},
            {"training_interface"},
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI warning regression test")
    def test_ui_distinguishes_training_callable_checkpoint_and_artifact_modes(self):
        recipe = load_recipe(RECIPE_PATH).to_dict()
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in sorted({step["op"] for step in recipe["steps"]})
        ]
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
            vm.runInContext(source + `
              state.operations = ${{JSON.stringify({json.dumps(operations)})}};
              const trainingRecipe = ${{JSON.stringify({json.dumps(recipe)})}};
              trainingRecipe.steps.find((step) => step.id === "wireless_channel").params.wireless_backend = "numpy";
              const callableRecipe = JSON.parse(JSON.stringify(trainingRecipe));
              callableRecipe.steps.find((step) => step.id === "sender").params = {{ runtime: "external_callable" }};
              const checkpointRecipe = JSON.parse(JSON.stringify(trainingRecipe));
              checkpointRecipe.steps.find((step) => step.id === "sender").params = {{ runtime: "learned_checkpoint" }};
              const artifactRecipe = JSON.parse(JSON.stringify(trainingRecipe));
              artifactRecipe.steps.find((step) => step.id === "sender").params = {{ runtime: "learned_artifact" }};
              const configuredArtifactRecipe = JSON.parse(JSON.stringify(trainingRecipe));
              configuredArtifactRecipe.steps.find((step) => step.id === "sender").params = {{
                runtime: "learned_artifact",
                artifact_manifest_path: "differentiable_exports/deepjscc/trained_artifact.yaml",
                artifact_entrypoint: "encoder",
              }};
              globalThis.__result = {{
                training: recipeConfigurationIssues(trainingRecipe),
                callable: recipeConfigurationIssues(callableRecipe),
                checkpoint: recipeConfigurationIssues(checkpointRecipe),
                artifact: recipeConfigurationIssues(artifactRecipe),
                configuredArtifact: recipeConfigurationIssues(configuredArtifactRecipe),
              }};
            `, sandbox);
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
        result = json.loads(completed.stdout)
        self.assertEqual(len(result["training"]), 1)
        self.assertIn("export-only training starter", result["training"][0])
        self.assertIn("generic training contract", result["training"][0])
        self.assertEqual(len(result["callable"]), 2)
        self.assertTrue(any("export-only training starter" in issue for issue in result["callable"]))
        self.assertTrue(any("custom external codecs require a Python callable" in issue for issue in result["callable"]))
        self.assertEqual(len(result["checkpoint"]), 2)
        self.assertTrue(any("select a learned DeepJSCC checkpoint" in issue for issue in result["checkpoint"]))
        self.assertEqual(len(result["artifact"]), 2)
        self.assertTrue(any("select a trained DeepJSCC artifact" in issue for issue in result["artifact"]))
        self.assertEqual(len(result["configuredArtifact"]), 1)
        self.assertIn("export-only training starter", result["configuredArtifact"][0])


if __name__ == "__main__":
    unittest.main()
