from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path

from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


class PhaseTrackingArtifactPickerUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the phase-tracking UI test")
    def test_picker_tracks_mode_and_refreshes_project_trained_artifacts(self):
        operation = build_registry().get(
            "demodulation.phase_tracking_receiver_adapter"
        ).describe()
        artifact_ui = operation["params_schema"]["properties"][
            "artifact_manifest_path"
        ]["x-noema-ui"]
        self.assertEqual(artifact_ui["control"], "trained_artifact")
        self.assertEqual(artifact_ui["visible_when"], {"mode": "learned_artifact"})

        app_js = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
        step = {
            "id": "demodulator",
            "op": "demodulation.phase_tracking_receiver_adapter",
            "inputs": {
                "rx_symbols": "wireless_channel.rx_symbols",
                "pilot_context": "modulator.pilot_context",
            },
            "params": {
                "mode": "pilot_interpolation",
                "pll_alpha": 0.12,
                "pll_beta": 0.005,
            },
        }
        artifact = {
            "id": "qpsk_pilot_phase_tracking.demodulator.phase_tracking_receiver",
            "label": "Learned phase tracker · qpsk_pilot_phase_tracking",
            "manifest_path": ".noema/training_exports/qpsk_phase_tracking/trained_artifact.yaml",
            "source_class": "project_trained",
            "ready": True,
            "compatible_operations": [
                {
                    "operation": "demodulation.phase_tracking_receiver_adapter",
                    "preferred_step_id": "demodulator",
                    "required_inputs": ["rx_symbols", "pilot_context"],
                    "params": {
                        "mode": "learned_artifact",
                        "artifact_manifest_path": "trained_artifact.yaml",
                        "artifact_entrypoint": "phase_tracking_receiver",
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
              process,
              localStorage: {{ getItem: () => null, setItem: () => {{}} }},
              document: {{ documentElement: {{ dataset: {{}} }}, getElementById: () => null }},
              window: {{ CSS: null }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              (async () => {{
                state.operations = [${{JSON.stringify({json.dumps(operation)})}}];
                state.editRecipe = {{ name: "phase", metadata: {{}}, steps: [${{JSON.stringify({json.dumps(step)})}}] }};
                state.trainedArtifacts = ${{JSON.stringify({json.dumps([artifact])})}};
                const receiver = state.editRecipe.steps[0];
                const modes = ["uncompensated", "pilot_interpolation", "pilot_smoothing", "decision_directed_pll", "oracle"];
                const classicalMarkup = Object.fromEntries(modes.map((mode) => {{
                  receiver.params.mode = mode;
                  return [mode, schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }})];
                }}));
                receiver.params.mode = "learned_artifact";
                const learnedMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});

                state.trainedArtifacts = [];
                api = async (path) => {{
                  if (path !== "/api/trained-artifacts") throw new Error("unexpected API path: " + path);
                  globalThis.__apiCalls = Number(globalThis.__apiCalls || 0) + 1;
                  return {{ artifacts: ${{JSON.stringify({json.dumps([artifact])})}} }};
                }};
                renderRecipeControls = () => {{
                  globalThis.__immediateRenders = Number(globalThis.__immediateRenders || 0) + 1;
                }};
                recipeConfiguratorScrollContainer = () => null;
                snapshotRenderedRecipeConfiguratorBlockIds = () => new Set();
                receiver.params.mode = "pilot_interpolation";
                const inactiveRefresh = refreshSchemaManagedArtifactsForController(
                  receiver,
                  operationById(receiver.op).params_schema,
                  "mode",
                );
                receiver.params.mode = "learned_artifact";
                await Promise.all([
                  refreshSchemaManagedArtifactsForController(
                    receiver,
                    operationById(receiver.op).params_schema,
                    "mode",
                  ),
                  refreshSchemaManagedArtifactsForController(
                    receiver,
                    operationById(receiver.op).params_schema,
                    "mode",
                  ),
                ]);
                const refreshedMarkup = schemaDrivenStepMarkup(receiver, 0, {{ recipe: state.editRecipe, open: true }});
                globalThis.__result = {{
                  classicalMarkup,
                  learnedMarkup,
                  refreshedMarkup,
                  apiCalls: globalThis.__apiCalls,
                  immediateRenders: globalThis.__immediateRenders,
                  inactiveRefreshWasNull: inactiveRefresh === null,
                }};
              }})().then(() => process.stdout.write(JSON.stringify(globalThis.__result))).catch((error) => {{
                console.error(error);
                process.exitCode = 1;
              }});
            `, sandbox);
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

        for mode, markup in result["classicalMarkup"].items():
            with self.subTest(mode=mode):
                self.assertIn(
                    'class="trained-artifact-picker managed-checkpoint-picker disabled"',
                    markup,
                )
                select = markup.split(
                    'data-schema-trained-artifact="demodulator"', 1
                )[1].split(">", 1)[0]
                self.assertIn("disabled", select)
                self.assertIn("Set Mode to Learned artifact", markup)

        learned = result["learnedMarkup"]
        learned_select = learned.split(
            'data-schema-trained-artifact="demodulator"', 1
        )[1].split(">", 1)[0]
        self.assertNotIn("disabled", learned_select)
        self.assertIn('<optgroup label="Project-trained">', learned)
        self.assertIn("Learned phase tracker", learned)
        self.assertLess(
            learned.index('data-schema-param-name="mode"'),
            learned.index('data-schema-trained-artifact="demodulator"'),
        )
        self.assertLess(
            learned.index('data-schema-trained-artifact="demodulator"'),
            learned.index('data-schema-param-name="pll_alpha"'),
        )

        self.assertIn('<optgroup label="Project-trained">', result["refreshedMarkup"])
        self.assertIn("Learned phase tracker", result["refreshedMarkup"])
        self.assertEqual(result["apiCalls"], 1)
        self.assertEqual(result["immediateRenders"], 1)
        self.assertTrue(result["inactiveRefreshWasNull"])

        binder_source = app_js.read_text(encoding="utf-8")
        binder_source = binder_source[
            binder_source.index("function bindSchemaDrivenRecipeConfigurator()") :
            binder_source.index("function addSchemaExtensionParameter(")
        ]
        self.assertIn(
            "void refreshSchemaManagedArtifactsForController(",
            binder_source,
        )


if __name__ == "__main__":
    unittest.main()
