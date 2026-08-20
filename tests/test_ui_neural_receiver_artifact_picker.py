from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path

from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


class NeuralReceiverArtifactPickerUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the neural-receiver UI test")
    def test_mode_and_artifact_picker_match_allocator_interaction_pattern(self):
        operation = build_registry().get("demodulation.neural_receiver_adapter").describe()
        artifact_ui = operation["params_schema"]["properties"]["artifact_manifest_path"]["x-noema-ui"]
        self.assertEqual(artifact_ui["control"], "trained_artifact")
        self.assertEqual(artifact_ui["visible_when"], {"mode": "learned_artifact"})
        self.assertIn("mode", artifact_ui["derived_params"])
        self.assertIn("artifact_manifest_path", artifact_ui["derived_params"])
        app_js = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
        step = {
            "id": "demodulator",
            "op": "demodulation.neural_receiver_adapter",
            "inputs": {"rx_symbols": "wireless_channel.rx_symbols"},
            "params": {"mode": "reference_qpsk"},
        }
        artifact = {
            "id": "learned-qpsk-demapper",
            "label": "Learned QPSK demapper",
            "manifest_path": "trained_artifacts/qpsk/trained_artifact.yaml",
            "source_class": "project_trained",
            "ready": True,
            "compatible_operations": [
                {
                    "operation": "demodulation.neural_receiver_adapter",
                    "required_inputs": ["rx_symbols"],
                    "params": {
                        "mode": "learned_artifact",
                        "modulation": "qpsk",
                        "artifact_manifest_path": "trained_artifacts/qpsk/trained_artifact.yaml",
                        "artifact_entrypoint": "neural_receiver",
                    },
                }
            ],
        }
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(app_js))}, "utf8");
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
              state.operations = [${{JSON.stringify({json.dumps(operation)})}}];
              state.editRecipe = {{ name: "qpsk", metadata: {{}}, steps: [${{JSON.stringify({json.dumps(step)})}}] }};
              state.trainedArtifacts = ${{JSON.stringify({json.dumps([artifact])})}};
              const receiver = state.editRecipe.steps[0];
              const referenceMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});
              receiver.params.mode = "linear_npz";
              const linearMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});
              receiver.params.mode = "learned_artifact";
              const learnedMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});
              state.trainedArtifacts = [];
              const learnedWithoutArtifactsMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});
              globalThis.__result = {{ referenceMarkup, linearMarkup, learnedMarkup, learnedWithoutArtifactsMarkup }};
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

        for markup in (result["referenceMarkup"], result["linearMarkup"], result["learnedMarkup"]):
            self.assertIn('class="field-grid compact schema-primary-parameters with-artifact"', markup)
            self.assertIn('data-schema-trained-artifact="demodulator"', markup)
            self.assertLess(markup.index('data-schema-param-name="mode"'), markup.index("data-schema-trained-artifact"))
            self.assertLess(markup.index("data-schema-trained-artifact"), markup.index('data-schema-param-name="modulation"'))
            self.assertIn(">Uncompensated QPSK</option>", markup)
            self.assertIn(">Calibrated I/Q oracle</option>", markup)
            self.assertIn(">Linear NPZ checkpoint</option>", markup)
            self.assertIn(">Learned model</option>", markup)

        for markup in (result["referenceMarkup"], result["linearMarkup"]):
            self.assertIn('class="trained-artifact-picker managed-checkpoint-picker disabled"', markup)
            select = markup.split('data-schema-trained-artifact="demodulator"', 1)[1].split(">", 1)[0]
            self.assertIn("disabled", select)
            self.assertIn("Set Mode to Learned model", markup)

        learned_select = result["learnedMarkup"].split(
            'data-schema-trained-artifact="demodulator"', 1
        )[1].split(">", 1)[0]
        self.assertNotIn("disabled", learned_select)
        self.assertIn('<optgroup label="Project-trained">', result["learnedMarkup"])
        self.assertIn("Learned QPSK demapper", result["learnedMarkup"])
        self.assertIn('data-schema-trained-artifact-open="demodulator"', result["learnedMarkup"])
        self.assertIn("Import external artifact", result["learnedMarkup"])

        empty_select = result["learnedWithoutArtifactsMarkup"].split(
            'data-schema-trained-artifact="demodulator"', 1
        )[1].split(">", 1)[0]
        self.assertNotIn("disabled", empty_select)
        self.assertIn('data-schema-trained-artifact-open="demodulator"', result["learnedWithoutArtifactsMarkup"])


if __name__ == "__main__":
    unittest.main()
