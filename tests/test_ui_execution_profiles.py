from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import urllib.request

from noema_lab.ui.server import start_ui_server_in_thread


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"


class ExecutionProfileApiTests(unittest.TestCase):
    def test_catalog_and_recipe_rows_include_execution_profile_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            recipe_dir = project_root / "recipes"
            recipe_dir.mkdir(parents=True)
            recipe = {
                "schema_version": 1,
                "name": "profiled_image_path",
                "execution_profile": {"id": "layered_digital", "version": 1},
                "steps": [
                    {"id": "payload_bit_boundary", "op": "channel.bit_boundary"},
                    {"id": "channel_encoder", "op": "channel.identity_encoder"},
                    {"id": "tx_bit_boundary", "op": "channel.bit_boundary"},
                    {"id": "wireless_channel", "op": "wireless.digital_link"},
                    {"id": "rx_bit_boundary", "op": "channel.bit_boundary"},
                    {"id": "channel_bit_count_match", "op": "channel.bit_count_match"},
                    {"id": "channel_decoder", "op": "channel.identity_decoder"},
                ],
            }
            (recipe_dir / "profiled.json").write_text(json.dumps(recipe), encoding="utf-8")
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                project_root / ".workspace",
                project_root,
            )
            host, port = server.server_address
            base = "http://%s:%d" % (host, port)
            try:
                catalog = _get_json(base + "/api/execution-profiles")
                self.assertEqual(
                    [profile["id"] for profile in catalog["profiles"]],
                    [
                        "layered_digital",
                        "joint_source_channel_symbols",
                        "csi_feedback_downlink",
                        "pilot_channel_estimation",
                        "mimo_ofdm_channel_estimation",
                        "beamforming_link_evaluation",
                        "range_localization",
                        "aoa_array_estimation",
                        "task_inference",
                        "task_evaluation",
                    ],
                )
                self.assertEqual(catalog["custom"]["id"], "custom")

                rows = _get_json(base + "/api/recipes")["recipes"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    rows[0]["execution_profile"],
                    {"id": "layered_digital", "version": 1},
                )
                self.assertEqual(rows[0]["execution_profile_report"]["status"], "conformant")
                self.assertEqual(
                    rows[0]["execution_profile_report"]["reference"],
                    rows[0]["execution_profile"],
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI profile regression test")
class ExecutionProfileUiTests(unittest.TestCase):
    def test_value_and_topology_edits_have_distinct_profile_semantics(self):
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
              state.executionProfiles = [
                {{ id: "layered_digital", label: "Layered digital", summary: "Separate digital stages." }},
                {{ id: "joint_source_channel_symbols", label: "Joint source-channel symbols", summary: "Symbol-native path." }},
              ];
              state.editRecipe = {{
                name: "image_path",
                execution_profile: {{ id: "layered_digital", version: 1 }},
                metadata: {{}},
                steps: [{{ id: "data", op: "source.image_dataset", params: {{}}, inputs: {{}} }}],
              }};
              state.editRecipeKey = "working:profile";
              state.selectedRecipe = {{
                key: "working:profile",
                editorPreference: "image",
                preserveTopology: false,
                recipe: state.editRecipe,
              }};
              applyBuiltRecipe = (recipe) => {{ globalThis.__lastApplied = JSON.parse(JSON.stringify(recipe)); }};
              state.editRecipe.name = "renamed path";
              commitRecipeValueEdit(false);
              const afterValue = JSON.parse(JSON.stringify(state.editRecipe.execution_profile));
              commitRecipeTopologyEdit(false);
              const afterTopology = JSON.parse(JSON.stringify(state.editRecipe.execution_profile));
              const firstMetadata = JSON.parse(JSON.stringify(state.editRecipe.metadata));
              commitRecipeTopologyEdit(false);
              globalThis.__result = {{
                afterValue,
                afterTopology,
                afterSecondTopology: JSON.parse(JSON.stringify(state.editRecipe.execution_profile)),
                firstMetadata,
                editorPreference: state.selectedRecipe.editorPreference,
                preserveTopology: state.selectedRecipe.preserveTopology,
                standardMarkup: executionProfileSummaryMarkup({{
                  execution_profile: {{ id: "layered_digital", version: 1 }},
                }}),
                customMarkup: executionProfileSummaryMarkup(state.editRecipe),
                inferredJoint: inferredExecutionProfile({{
                  metadata: {{ codec_output_form: "symbols" }},
                  steps: [{{ op: "model.deepjscc_external_encode" }}],
                }}),
                inferredLocalization: inferredExecutionProfile({{
                  steps: [
                    {{ id: "data", op: "source.localization_geometry" }},
                    {{ id: "evaluation", op: "metrics.localization" }},
                  ],
                }}),
                inferredMimo: inferredExecutionProfile({{
                  steps: [
                    {{ id: "data", op: "source.ai_phy_channel_realization", params: {{ scenario: "mimo_ofdm", tx_antennas: 2 }} }},
                    {{ id: "pilots", op: "source.ai_phy_pilot_pattern" }},
                    {{ id: "pilot_observation", op: "wireless.pilot_observation" }},
                    {{ id: "evaluation", op: "metrics.channel_estimation" }},
                  ],
                }}),
                taskInferenceLabel: executionProfileLabel("task_inference"),
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

        self.assertEqual(result["afterValue"], {"id": "layered_digital", "version": 1})
        self.assertEqual(
            result["afterTopology"],
            {
                "id": "custom",
                "version": 1,
                "based_on": {"id": "layered_digital", "version": 1},
            },
        )
        self.assertEqual(result["afterSecondTopology"], result["afterTopology"])
        self.assertFalse(any(key.startswith("ui_") for key in result["firstMetadata"]))
        self.assertEqual(result["editorPreference"], "graph")
        self.assertTrue(result["preserveTopology"])
        self.assertIn("Layered digital", result["standardMarkup"])
        self.assertIn("standard", result["standardMarkup"])
        self.assertIn('data-ui-tooltip="Separate digital stages."', result["standardMarkup"])
        self.assertNotIn("field-hint", result["standardMarkup"])
        self.assertIn("Custom pipeline", result["customMarkup"])
        self.assertIn("custom topology", result["customMarkup"])
        self.assertEqual(
            result["inferredJoint"],
            {"id": "joint_source_channel_symbols", "version": 1},
        )
        self.assertEqual(
            result["inferredLocalization"],
            {"id": "range_localization", "version": 1},
        )
        self.assertEqual(
            result["inferredMimo"],
            {"id": "mimo_ofdm_channel_estimation", "version": 1},
        )
        self.assertEqual(result["taskInferenceLabel"], "Task inference")

    def test_canonical_builders_declare_the_two_standard_profiles(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn(
            'id: profile.pipeline === "symbols" ? EXECUTION_PROFILE_IDS.jointSymbols : EXECUTION_PROFILE_IDS.layered',
            source,
        )
        self.assertIn(
            "id: neuralJsccMode ? EXECUTION_PROFILE_IDS.jointSymbols : EXECUTION_PROFILE_IDS.layered",
            source,
        )
        self.assertIn('api("/api/execution-profiles")', source)
        self.assertNotIn("commitDirectRecipeEdit", source)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
