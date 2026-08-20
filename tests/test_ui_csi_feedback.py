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
STYLES = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"
RECIPE = ROOT / "recipes" / "csi_feedback_sionna_train.yaml"


def _artifact(metadata: dict) -> dict:
    return {
        "kind": "metrics.report",
        "path": "/tmp/csi-feedback-report.json",
        "metadata": metadata,
    }


def _preview() -> dict:
    return {
        "axes": {"rows": "tx_antenna", "columns": "subcarrier"},
        "samples": [
            {
                "sample_index": 0,
                "true_magnitude": [[1.0, 0.5], [0.25, 0.75]],
                "reconstructed_magnitude": [[0.9, 0.45], [0.3, 0.7]],
                "absolute_error": [[0.1, 0.05], [0.05, 0.05]],
            },
            {
                "sample_index": 1,
                "true_magnitude": [[0.8, 0.4], [0.2, 0.6]],
                "reconstructed_magnitude": [[0.7, 0.35], [0.25, 0.55]],
                "absolute_error": [[0.1, 0.05], [0.05, 0.05]],
            },
        ],
    }


def _summary(
    run_id: str,
    *,
    feedback_dimension: int,
    feedback_bits: int | None,
    nmse_db: float,
    cosine: float,
    achieved: float,
    perfect: float,
    runtime: str = "truncated_angular_delay",
    downlink_snr: float = 10.0,
) -> dict:
    mode = "uniform_quantized" if feedback_bits is not None else "ideal_noiseless"
    metrics = {
        "csi_feedback.feedback_dimension": feedback_dimension,
        "csi_feedback.compression_factor": 512 / feedback_dimension,
        "csi_feedback.nmse_db": nmse_db,
        "csi_feedback.phase_invariant_cosine": cosine,
        "csi_feedback.achieved_spectral_efficiency_bps_hz": achieved,
        "csi_feedback.perfect_csi_spectral_efficiency_bps_hz": perfect,
        "csi_feedback.spectral_efficiency_retention": achieved / perfect,
        "csi_feedback.spectral_efficiency_loss_bps_hz": perfect - achieved,
        "channel.snr_db": downlink_snr,
    }
    if feedback_bits is not None:
        metrics["csi_feedback.feedback_bits_per_sample"] = feedback_bits
    recipe = {
        "name": run_id,
        "execution_profile": {"id": "csi_feedback_downlink", "version": 1},
        "metadata": {
            "research": {
                "task": {
                    "id": "csi_compression_feedback",
                    "kind": "feedback_control",
                    "modality": "wireless",
                }
            }
        },
        "steps": [
            {
                "id": "channel_state",
                "op": "wireless.miso_ofdm_csi",
                "params": {"downlink_snr_db": downlink_snr},
            },
            {
                "id": "feedback_encoder",
                "op": "model.csi_feedback_encoder",
                "params": {
                    "runtime": runtime,
                    "feedback_dimension": feedback_dimension,
                },
            },
            {
                "id": "feedback_link",
                "op": "channel.csi_feedback_link",
                "params": {"mode": mode, "bits_per_latent": 4},
            },
            {
                "id": "feedback_decoder",
                "op": "model.csi_feedback_decoder",
                "params": {
                    "runtime": runtime,
                    "feedback_dimension": feedback_dimension,
                },
            },
            {"id": "precoder", "op": "model.csi_mrt_precoder", "params": {}},
            {"id": "evaluation", "op": "metrics.csi_feedback", "params": {}},
        ],
    }
    return {
        "run_id": run_id,
        "status": "completed",
        "recipe": recipe,
        "metrics": metrics,
        "steps": [
            {
                "id": "evaluation",
                "metrics": metrics,
                "outputs": {
                    "report": _artifact({"csi_feedback_preview": _preview()})
                },
            }
        ],
    }


class CsiFeedbackRecipeTests(unittest.TestCase):
    def test_templates_keep_downlink_snr_only_on_executable_channel_step(self):
        for filename in (
            "csi_feedback_sionna_train.yaml",
            "csi_feedback_truncated_angular_delay.yaml",
            "csi_feedback_perfect_csit_upper_bound.yaml",
        ):
            recipe = load_recipe(ROOT / "recipes" / filename).to_dict()
            self.assertNotIn("downlink_snr_db", recipe.get("metadata", {}), filename)
            channel = next(
                step for step in recipe["steps"] if step["id"] == "channel_state"
            )
            self.assertEqual(channel["params"]["downlink_snr_db"], 10.0, filename)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for CSI feedback UI tests")
class CsiFeedbackUiTests(unittest.TestCase):
    def _run_node(self, body: str) -> dict:
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
              {body}
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
        return json.loads(completed.stdout)

    def test_high_level_configurator_edits_existing_profile_and_pair_atomically(self):
        recipe = load_recipe(RECIPE).to_dict()
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in sorted({step["op"] for step in recipe["steps"]})
        ]
        artifact = {
            "id": "csi.demo",
            "label": "Learned CSI feedback demo",
            "manifest_path": "trained_artifacts/csi-demo/trained_artifact.yaml",
            "ready": True,
            "artifact": {"format": "noema_trained_artifact_manifest_v2", "sha256": "a" * 64},
            "runtime": {
                "entrypoints": [
                    {
                        "id": "encoder",
                        "outputs": [
                            {"name": "feedback_code", "dtype": "float32", "shape": ["batch", 24]}
                        ],
                    },
                    {
                        "id": "decoder",
                        "inputs": [
                            {"name": "feedback_code", "dtype": "float32", "shape": ["batch", 24]}
                        ],
                    },
                ]
            },
            "application": {"mode": "all_group_bindings"},
            "compatible_operations": [
                {
                    "operation": "model.csi_feedback_encoder",
                    "runtime_entrypoint": "encoder",
                    "binding_group": "csi_feedback_codec",
                    "role": "encoder",
                    "preferred_step_id": "feedback_encoder",
                    "required_inputs": ["csi"],
                    "params": {
                        "runtime": "learned_artifact",
                        "artifact_manifest_path": "trained_artifacts/csi-demo/trained_artifact.yaml",
                        "artifact_entrypoint": "encoder",
                    },
                },
                {
                    "operation": "model.csi_feedback_decoder",
                    "runtime_entrypoint": "decoder",
                    "binding_group": "csi_feedback_codec",
                    "role": "decoder",
                    "preferred_step_id": "feedback_decoder",
                    "required_inputs": ["received_code"],
                    "params": {
                        "runtime": "learned_artifact",
                        "artifact_manifest_path": "trained_artifacts/csi-demo/trained_artifact.yaml",
                        "artifact_entrypoint": "decoder",
                    },
                },
            ],
        }
        result = self._run_node(
            rf"""
              state.operations = {json.dumps(operations)};
              state.editRecipe = {json.dumps(recipe)};
              state.editRecipeKey = "working:csi";
              state.selectedRecipe = {{ key: state.editRecipeKey, recipe: state.editRecipe }};
              state.selectedNodeId = null;
              state.trainedArtifacts = [{json.dumps(artifact)}];
              els.recipeConfigurator = {{ innerHTML: "", querySelectorAll: () => [] }};
              bindCsiFeedbackConfigurator = () => {{}};
              const topology = (value) => (value.steps || []).map((step) => ({{
                id: step.id, op: step.op, inputs: step.inputs || {{}},
              }}));
              const before = topology(state.editRecipe);
              renderCsiFeedbackConfigurator(state.editRecipe);
              const markup = els.recipeConfigurator.innerHTML;
              const afterRender = topology(state.editRecipe);
              globalThis.__commitCount = 0;
              globalThis.__notices = [];
              commitRecipeValueEdit = () => {{ globalThis.__commitCount += 1; }};
              notify = (message) => globalThis.__notices.push(message);
              renderRecipeControls = () => {{}};
              setCsiFeedbackRuntime("identity");
              const identityDimensions = [
                findStep("feedback_encoder").params.feedback_dimension,
                findStep("feedback_decoder").params.feedback_dimension,
              ];
              setCsiFeedbackRuntime("truncated_angular_delay");
              setCsiFeedbackDimension("40");
              const choice = compatibleTrainedArtifactChoices(findStep("feedback_encoder"))[0];
              applyTrainedArtifactToStep("feedback_encoder", choice.key);
              const routeCalls = [];
              decorateRecipeConfigurator = () => {{}};
              renderRecipeBlocksConfigurator = () => routeCalls.push("blocks");
              state.selectedNodeId = null;
              renderUnifiedConfigurator();
              state.selectedNodeId = "feedback_encoder";
              renderUnifiedConfigurator();
              globalThis.__result = {{
                markup,
                before,
                afterRender,
                after: topology(state.editRecipe),
                encoder: findStep("feedback_encoder").params,
                decoder: findStep("feedback_decoder").params,
                commitCount: globalThis.__commitCount,
                identityDimensions,
                notices: globalThis.__notices,
                routeCalls,
              }};
            """
        )

        self.assertEqual(result["before"], result["afterRender"])
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["routeCalls"], ["blocks", "blocks"])
        self.assertEqual(result["commitCount"], 4)
        self.assertEqual(result["identityDimensions"], [512, 512])
        facet_positions = [
            result["markup"].index(f'data-config-facet="{facet}"')
            for facet in ("data", "method", "communication", "evaluation")
        ]
        self.assertEqual(facet_positions, sorted(facet_positions))
        for heading in (
            "Downlink Channel &amp; CSI",
            "CSI Feedback Model",
            "Feedback Link",
            "Downlink Evaluation",
        ):
            self.assertIn(f"<h3>{heading}</h3>", result["markup"])
        for parameter in (
            "tx_antennas",
            "ofdm_fft_size",
            "num_ofdm_symbols",
            "tdl_model",
            "wireless_backend",
            "seed",
            "downlink_snr_db",
            "sample_count",
            "mode",
            "bits_per_latent",
            "clip_value",
        ):
            self.assertIn(f'data-schema-param-name="{parameter}"', result["markup"])
        self.assertIn("data-csi-feedback-runtime", result["markup"])
        self.assertIn("data-csi-feedback-dimension", result["markup"])
        self.assertIn("Evaluation CSI samples / realizations", result["markup"])
        self.assertIn('data-schema-trained-artifact="feedback_encoder"', result["markup"])
        self.assertIn("feedback_encoder + feedback_decoder", result["markup"])
        self.assertNotIn("<h2>Working recipe</h2>", result["markup"])
        for endpoint in (result["encoder"], result["decoder"]):
            self.assertEqual(endpoint["runtime"], "learned_artifact")
            self.assertEqual(endpoint["feedback_dimension"], 24)
            self.assertEqual(
                endpoint["artifact_manifest_path"],
                "trained_artifacts/csi-demo/trained_artifact.yaml",
            )
        self.assertEqual(
            result["notices"],
            ["Learned CSI feedback demo applied to feedback_encoder + feedback_decoder"],
        )

    def test_workbench_csi_contract_stays_architecture_neutral(self):
        recipe = load_recipe(RECIPE).to_dict()
        result = self._run_node(
            rf"""
              const recipe = {json.dumps(recipe)};
              const report = {{
                optimizable_candidate_blocks: ["feedback_encoder", "feedback_decoder"],
                frozen_exportable_blocks: ["feedback_link", "precoder", "evaluation"],
              }};
              state.differentiableExportSettings = {{
                optimizable: "feedback_encoder,feedback_decoder",
                framework: "torch-sionna",
                out: "differentiable_exports/csi_feedback",
                force: true,
              }};
              state.trainingCaptureContract = {{
                ready: true,
                mode: "captured_tensors",
                sample_unit: "CSI realizations",
                total_samples: 768,
                required_taps: [{{ id: "true_csi", from: "channel_state.csi", role: "encoder_input_csi" }}],
                selected_taps: [{{ id: "true_csi", from: "channel_state.csi" }}],
                candidates: [{{ id: "true_csi", from: "channel_state.csi", kind: "channel.miso_ofdm_csi.numpy", selectable: true, required: true }}],
                split_plan: {{ total_samples: 768, percentages: {{ train: 66.6666666667, validation: 16.66666666665, test: 16.66666666665 }} }},
              }};
              const markup = differentiableExportFormMarkup(recipe, report, report.optimizable_candidate_blocks, null);
              globalThis.__result = {{
                markup,
              }};
            """
        )

        self.assertNotIn('data-differentiable-export-field="starter"', result["markup"])
        self.assertNotIn('data-differentiable-export-field="loss"', result["markup"])
        self.assertIn('value="torch-sionna" selected', result["markup"])
        self.assertIn("true_csi", result["markup"])
        self.assertIn("channel_state.csi", result["markup"])
        self.assertIn("Train %", result["markup"])
        self.assertIn('value="66.6667"', result["markup"])
        self.assertNotIn("Capture CSI datasets", result["markup"])

    def test_task_specific_results_use_real_feedback_accounting_and_preview(self):
        quantized = [
            _summary(
                "quantized-16",
                feedback_dimension=16,
                feedback_bits=64,
                nmse_db=-8.0,
                cosine=0.90,
                achieved=4.0,
                perfect=5.0,
            ),
            _summary(
                "quantized-32",
                feedback_dimension=32,
                feedback_bits=128,
                nmse_db=-12.0,
                cosine=0.96,
                achieved=4.6,
                perfect=5.0,
            ),
        ]
        continuous = _summary(
            "continuous-24",
            feedback_dimension=24,
            feedback_bits=None,
            nmse_db=-10.0,
            cosine=0.94,
            achieved=4.4,
            perfect=5.0,
        )
        rows = [
            {
                "recipe": {
                    "key": "quantized",
                    "displayName": "Quantized CSI codec",
                    "color": "#6aa5ff",
                    "recipe": quantized[0]["recipe"],
                },
                "summaries": quantized,
            }
        ]
        continuous_rows = [
            {
                "recipe": {
                    "key": "continuous",
                    "displayName": "Continuous CSI codec",
                    "color": "#ff9d5c",
                    "recipe": continuous["recipe"],
                },
                "summary": continuous,
            }
        ]
        result = self._run_node(
            rf"""
              const rows = {json.dumps(rows)};
              const continuousRows = {json.dumps(continuous_rows)};
              state.resultRows = rows;
              state.csiFeedbackSettings = {{ entryId: "", sampleIndex: 1 }};
              const details = collectRunDetails(rows[0].summaries[0]);
              const preview = csiFeedbackPreviewFigure(rows);
              const examples = resultExamplesPanel(rows, 0);
              const performance = resultQualityPanel(rows, rows);
              const communication = resultCommunicationPanel(rows, rows);
              els.settingsTitle = {{ textContent: "" }};
              els.settingsSubtitle = {{ textContent: "" }};
              els.settingsBody = {{ innerHTML: "", querySelectorAll: () => [] }};
              els.settingsOverlay = {{ hidden: true }};
              openCsiFeedbackSettingsDialog("rate");
              const rateSettings = els.settingsBody.innerHTML;
              openCsiFeedbackSettingsDialog("quality");
              const qualitySettings = els.settingsBody.innerHTML;
              openCsiFeedbackSettingsDialog("preview");
              globalThis.__result = {{
                inferredTask: researchTaskForSteps(rows[0].summaries[0].recipe.steps),
                primary: primaryTaskMetric(details, rows[0].summaries[0], rows[0]),
                rowDetected: rowIsCsiFeedback(rows[0]),
                rate: csiFeedbackRateFigure(rows),
                continuousRate: csiFeedbackRateFigure(continuousRows),
                quality: csiFeedbackQualityFigure(rows),
                preview,
                examples,
                performance,
                communication,
                previewSettings: els.settingsBody.innerHTML,
                rateSettings,
                qualitySettings,
                tables: resultMetricTableSections([...rows, ...continuousRows]),
                overview: overviewPrimaryMetricPlot(rows),
              }};
            """
        )

        self.assertEqual(result["inferredTask"]["id"], "csi_compression_feedback")
        self.assertEqual(result["primary"]["label"], "Achievable spectral efficiency (bit/s/Hz)")
        self.assertAlmostEqual(result["primary"]["value"], 4.0)
        self.assertTrue(result["rowDetected"])
        self.assertIn("Achieved Downlink Spectral Efficiency", result["rate"])
        self.assertIn("CSI feedback method", result["rate"])
        self.assertNotIn("Rate retention", result["rate"])
        self.assertIn("data-visualization-series-key", result["rate"])
        self.assertIn("data-visualization-legend-key", result["rate"])
        self.assertIn("continuous feedback; no synthetic bit count", result["continuousRate"])
        self.assertIn("CSI Reconstruction NMSE", result["quality"])
        self.assertIn("NMSE (dB)", result["quality"])
        self.assertIn("phase-invariant cosine", result["quality"])
        for heading in (
            "True CSI magnitude",
            "Reconstructed CSI magnitude",
            "Absolute error",
        ):
            self.assertIn(heading, result["preview"])
        self.assertIn('data-csi-feedback-settings="preview"', result["preview"])
        self.assertIn('data-csi-feedback-preview-entry', result["preview"])
        self.assertIn('data-csi-feedback-preview-sample', result["preview"])
        self.assertIn('data-csi-feedback-setting="entryId"', result["previewSettings"])
        self.assertIn('data-csi-feedback-setting="sampleIndex"', result["previewSettings"])
        self.assertIn('data-csi-feedback-setting="previewScale"', result["previewSettings"])
        self.assertIn('data-csi-feedback-setting="independentErrorScale"', result["previewSettings"])
        self.assertIn('data-csi-feedback-setting="rateYLog"', result["rateSettings"])
        self.assertNotIn('data-csi-feedback-setting="rateXLog"', result["rateSettings"])
        self.assertIn("SNR is already expressed in dB", result["rateSettings"])
        self.assertIn('data-csi-feedback-setting="qualityXLog"', result["qualitySettings"])
        self.assertNotIn('data-csi-feedback-setting="qualityYLog"', result["qualitySettings"])
        self.assertIn("NMSE is already expressed in dB", result["qualitySettings"])
        self.assertIn('data-visualization-display-setting="showGrid"', result["previewSettings"])
        self.assertIn('data-visualization-display-setting="showValues"', result["previewSettings"])
        self.assertIn('data-visualization-display-setting="includeBackground"', result["previewSettings"])
        self.assertIn("Realization 2", result["preview"])
        self.assertIn("csi-feedback-heatmap-cell", result["preview"])
        self.assertIn("CSI Reconstruction Preview", result["examples"])
        self.assertNotIn("CSI Reconstruction Preview", result["communication"])
        self.assertNotIn('class="result-card"', result["examples"])
        self.assertIn("CSI compression and feedback · complete-run comparison", result["tables"])
        self.assertIn("continuous — not bits", result["tables"])
        self.assertNotIn('data-visualization-id="csi-feedback-rate"', result["overview"])
        self.assertNotIn('data-visualization-id="csi-feedback-quality"', result["overview"])
        self.assertEqual(result["performance"].count('data-visualization-id="csi-feedback-quality"'), 1)
        self.assertNotIn('data-visualization-id="csi-feedback-rate"', result["performance"])
        self.assertEqual(result["communication"].count('data-visualization-id="csi-feedback-rate"'), 1)
        self.assertNotIn('data-visualization-id="csi-feedback-quality"', result["communication"])

    def test_csi_figures_reuse_global_theme_legend_save_and_settings_contract(self):
        app_js = APP_JS.read_text(encoding="utf-8")
        styles = STYLES.read_text(encoding="utf-8")
        self.assertIn('data-visualization-id="csi-feedback-rate"', app_js)
        self.assertIn('data-visualization-id="csi-feedback-quality"', app_js)
        self.assertIn('data-visualization-id="csi-feedback-preview"', app_js)
        self.assertIn("bindVisualizationControls();", app_js)
        self.assertIn("ensureVisualizationSaveButton", app_js)
        self.assertIn("ensureMinimalVisualizationSettingsButton", app_js)
        self.assertIn("bindVisualizationLegendInteractions", app_js)
        self.assertIn("var(--visualization-muted-color)", styles)
        self.assertIn('data-csi-feedback-settings="rate"', app_js)
        self.assertIn('data-csi-feedback-settings="quality"', app_js)
        self.assertIn('data-csi-feedback-settings="preview"', app_js)
        self.assertIn(".csi-feedback-scale-label", styles)
        self.assertIn(".csi-feedback-legend-entry.is-visualization-series-highlight", styles)

    def test_snr_sweep_uses_one_primary_curve_per_named_csi_method(self):
        learned = [
            _summary(
                f"learned-{snr}",
                feedback_dimension=32,
                feedback_bits=128,
                nmse_db=-11.0,
                cosine=0.96,
                achieved=2.0 + snr / 10,
                perfect=2.2 + snr / 10,
                runtime="learned_artifact",
                downlink_snr=snr,
            )
            for snr in (0.0, 5.0, 10.0)
        ]
        classical = [
            _summary(
                f"classical-{snr}",
                feedback_dimension=32,
                feedback_bits=128,
                nmse_db=-8.5,
                cosine=0.93,
                achieved=1.8 + snr / 10,
                perfect=2.2 + snr / 10,
                runtime="truncated_angular_delay",
                downlink_snr=snr,
            )
            for snr in (0.0, 5.0, 10.0)
        ]
        rows = [
            {
                "recipe": {"key": "learned", "color": "#6aa5ff", "recipe": learned[0]["recipe"]},
                "summaries": learned,
            },
            {
                "recipe": {"key": "classical", "color": "#ff9d5c", "recipe": classical[0]["recipe"]},
                "summaries": classical,
            },
        ]
        result = self._run_node(
            rf"""
              const rows = {json.dumps(rows)};
              const labels = recipeDisplayLabels(rows.map((row) => row.recipe));
              globalThis.__result = {{
                learned: recipeDisplayName(rows[0].recipe, labels),
                classical: recipeDisplayName(rows[1].recipe, labels),
                figure: csiFeedbackRateFigure(rows),
              }};
            """
        )

        self.assertEqual(result["learned"], "Learned CSI codec")
        self.assertEqual(result["classical"], "Truncated angular-delay")
        self.assertIn("Achieved Downlink Spectral Efficiency vs Downlink SNR", result["figure"])
        self.assertIn("Downlink SNR (dB)", result["figure"])
        self.assertEqual(result["figure"].count('<polyline class="rd-series-line"'), 2)
        self.assertNotIn("csi-feedback-retention-axis", result["figure"])

    def test_all_result_svg_titles_are_centered(self):
        app_js = APP_JS.read_text(encoding="utf-8")
        title_tags = [line for line in app_js.splitlines() if '<text class="rd-plot-title"' in line]
        self.assertTrue(title_tags)
        for title in title_tags:
            self.assertIn('x="${width / 2}"', title)
            self.assertIn('text-anchor="middle"', title)


if __name__ == "__main__":
    unittest.main()
