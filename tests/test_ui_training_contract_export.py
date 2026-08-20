from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest

from noema_lab.ui.server import (
    _exported_project_payload,
    _exported_project_watch_payload,
)


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
STYLES_CSS = ROOT / "src" / "noema_lab" / "ui" / "static" / "styles.css"
ARCHITECTURE_DOC = ROOT / "docs" / "differentiable_export_architecture.md"


class TrainingContractWorkbenchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the Workbench custom-capture test")
    def test_workbench_uses_one_managed_training_data_flow(self):
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
              const events = [];
              const handlerName = (handler) => {{
                if (handler === updateDatasetCaptureSetting) return "updateDatasetCaptureSetting";
                if (handler === updateDatasetCaptureConfigField) return "updateDatasetCaptureConfigField";
                if (handler === updateDatasetCaptureDraft) return "updateDatasetCaptureDraft";
                if (handler === addDatasetCaptureTap) return "addDatasetCaptureTap";
                if (handler === runDatasetCapture) return "runDatasetCapture";
                return "wrapped";
              }};
              const control = (name, extra = {{}}) => ({{
                ...extra,
                dataset: extra.dataset || {{}},
                addEventListener: (event, handler) => events.push({{ name, event, handler: handlerName(handler) }}),
              }});
              const out = control("out", {{ type: "text" }});
              const force = control("force", {{ type: "checkbox" }});
              const split = control("split", {{ type: "text" }});
              const signal = control("signal", {{ tagName: "SELECT" }});
              const signalName = control("signalName", {{ tagName: "INPUT" }});
              const remove = control("remove", {{ dataset: {{ datasetCaptureRemoveTap: "0" }} }});
              const add = control("add");
              const submit = control("submit");
              const all = new Map([
                ["[data-dataset-capture-field]", [out, force]],
                ["[data-dataset-capture-config-field]", [split]],
                ["[data-dataset-capture-draft]", [signal, signalName]],
                ["[data-dataset-capture-remove-tap]", [remove]],
              ]);
              const one = new Map([
                ["[data-dataset-capture-add-tap]", add],
                ["[data-dataset-capture-submit]", submit],
              ]);
              const panel = {{
                innerHTML: "",
                querySelectorAll: (selector) => all.get(selector) || [],
                querySelector: (selector) => one.get(selector) || null,
              }};
              const recipe = {{
                name: "arbitrary_recipe",
                dataset_capture: {{
                  split: "train",
                  samples: 12,
                  taps: [{{ id: "samples", from: "source.samples" }}],
                }},
                steps: [{{ id: "source", op: "test.numpy_source" }}],
              }};
              state.operations = [{{
                id: "test.numpy_source",
                name: "Array source",
                status: "implemented",
                input_kinds: {{}},
                output_kinds: {{ samples: "data.real_numpy" }},
              }}];
              state.editRecipe = recipe;
              els.trainingPanel = panel;
              trainingStepTable = () => "<div>operation inspection</div>";
              differentiableExportFormMarkup = () => "<section data-derived-training-contract>derived training contract</section>";
              renderTrainingView();
              globalThis.__result = {{ markup: panel.innerHTML, events }};
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
        markup = result["markup"]
        events = {(item["name"], item["event"], item["handler"]) for item in result["events"]}

        self.assertIn("Operation Training Capabilities", markup)
        self.assertNotIn("Operation Differentiability", markup)
        self.assertIn("data-derived-training-contract", markup)
        self.assertNotIn("Custom Dataset Capture", markup)
        self.assertNotIn('data-dataset-capture-config-field="split"', markup)
        self.assertNotIn('data-dataset-capture-draft="from"', markup)
        self.assertNotIn('data-dataset-capture-submit', markup)
        self.assertEqual(events, set())

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the Workbench markup test")
    def test_contract_is_neutral_and_uses_block_picker(self):
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
              const recipe = {{
                name: "contract_test",
                steps: [
                  {{ id: "sender", op: "model.deepjscc_external_encode" }},
                  {{ id: "receiver", op: "model.deepjscc_external_decode" }},
                ],
              }};
              const report = {{
                steps: [
                  {{ id: "sender", op: "model.deepjscc_external_encode", gradient: "none", trainable_params: false, fine_tuning_supported: false, replacement_ready: true, exportable: false }},
                  {{ id: "receiver", op: "model.deepjscc_external_decode", gradient: "full", trainable_params: true, fine_tuning_supported: false, replacement_ready: true, exportable: true }},
                  {{ id: "fine_tunable_only", op: "model.internal", gradient: "full", trainable_params: true, fine_tuning_supported: true, replacement_ready: false, exportable: true }},
                  {{ id: "differentiable_support", op: "channel.support", gradient: "full", trainable_params: false, fine_tuning_supported: false, replacement_ready: false, exportable: true }},
                ],
                replacement_candidate_blocks: ["sender", "receiver"],
                // A stale/legacy list must never override operation-owned ABI readiness.
                optimizable_candidate_blocks: ["sender", "receiver", "fine_tunable_only", "differentiable_support"],
                portable_artifact_return_blocks: ["sender", "receiver"],
                frozen_exportable_blocks: ["unrelated_global_block"],
                selected_downstream_support_blocks: ["tx_power", "wireless_channel"],
                loss_step_candidates: ["evaluation", "secondary_evaluation"],
                selected_loss_steps: ["evaluation"],
                recommended_mode: "differentiable_export",
              }};
              const graph = {{ blocks: [{{ block: "AwgnChannelBlock", noema_op_id: "wireless.channel" }}] }};
              state.editRecipe = recipe;
              Object.assign(ensureCurrentTrainingPlan(), {{
                objective: "stale.example.objective",
                starter: "deepjscc-image",
              }});
              state.differentiableExportSettings = {{
                optimizable: "sender,receiver",
                framework: "torch",
                out: "differentiable_exports/contract_test",
                force: true,
              }};
              state.trainingCaptureContract = {{
                mode: "captured_tensors",
                ready: true,
                sample_unit: "records",
                total_samples: 10,
                required_taps: [{{ id: "source", from: "source.values" }}],
                selected_taps: [{{ id: "source", from: "source.values" }}],
                candidates: [{{ id: "source", from: "source.values", kind: "data.real_numpy" }}],
                split_plan: {{ percentages: {{ train: 60, validation: 20, test: 20 }}, counts: {{ train: 6, validation: 2, test: 2 }} }},
              }};
              const operationTable = trainingStepTable(report);
              const soleRequiredToggle = trainingContractStepToggle(
                {{ id: "estimator", replacement_ready: true }},
                new Set(["estimator"]),
                new Set(["estimator"]),
              );
              const portableReplacementToggle = trainingContractStepToggle(
                {{ id: "decoder", replacement_ready: true }},
                new Set(["decoder", "encoder"]),
                new Set(["decoder", "encoder"]),
              );
              const misleadingFineTuneToggle = trainingContractStepToggle(
                {{ id: "fine_tunable_only", trainable_params: true, replacement_ready: false, gradient: "full", exportable: true }},
                new Set(["fine_tunable_only"]),
                new Set(["fine_tunable_only"]),
              );
              const replacementCandidates = Array.from(differentiableExportCandidateSet(
                {{ steps: report.steps }},
                report.optimizable_candidate_blocks,
                {{ nodes: [{{ id: "graph_fine_tunable", differentiability: {{ gradient: "full", trainable_params: true, exportable: true }} }}] }},
              ));
              const capabilityHint = trainingGradientBlockHint(report);
              const noReplacementNotice = differentiableExportScopeNotice({{
                optimizable_candidate_blocks: [],
                frozen_exportable_blocks: ["differentiable_support"],
              }}, [], "contract");
              const captureBackedNotice = differentiableExportScopeNotice({{
                replacement_candidate_blocks: ["receiver"],
                selected_loss_steps: ["evaluation"],
                selected_downstream_support_blocks: ["hard_decision"],
                recommended_mode: "dataset_capture",
              }}, ["receiver"]);
              const neutral = differentiableExportFormMarkup(recipe, report, report.replacement_candidate_blocks, graph);
              const neutralPlan = currentTrainingPlanPayload();
              const capture = exportedProjectCapturePlanMarkup({{
                ready_for_external_training: false,
                captures: [
                  {{
                    split: "train",
                    label: "Train",
                    requested_samples: 80,
                    status: "pending",
                    output_dir: "differentiable_exports/contract_test/data/train",
                    expected_taps: [{{ id: "channel_gain", from: "wireless_channel.channel_gain" }}],
                  }},
                  {{
                    split: "validation",
                    label: "Validation",
                    requested_samples: 20,
                    status: "pending",
                    output_dir: "differentiable_exports/contract_test/data/validation",
                    expected_taps: [{{ id: "channel_gain", from: "wireless_channel.channel_gain" }}],
                  }},
                ],
              }});
              state.exportedProjectCaptureAllLoading = true;
              const busyCapture = exportedProjectCapturePlanMarkup({{
                ready_for_external_training: false,
                captures: [{{ split: "train", label: "Train", status: "pending" }}],
              }});
              state.exportedProjectCaptureAllLoading = false;
              state.exportedTrainingProjectPath = "differentiable_exports/pending_contract";
              state.exportedTrainingProject = {{
                ready_for_external_training: false,
                artifacts_ready: false,
                captures: [{{
                  split: "train",
                  label: "Train",
                  requested_samples: 80,
                  status: "pending",
                }}],
              }};
              const pendingWorkflow = exportedTrainingProjectWorkflowMarkup("differentiable_exports/pending_contract");
              state.exportedTrainingProjectPath = "differentiable_exports/neutral_contract";
              state.exportedTrainingProject = {{
                out_dir: "differentiable_exports/neutral_contract",
                ready_for_external_training: true,
                artifacts_ready: false,
                captures: [],
                training: {{
                  working_directory: "differentiable_exports/neutral_contract",
                  trainer_included: false,
                  command_status: "not_supplied",
                }},
                manifest: {{ helpers: {{ validate: "python validate_contract.py" }} }},
              }};
              const neutralReadyWorkflow = exportedTrainingProjectWorkflowMarkup("differentiable_exports/neutral_contract");
              state.exportedTrainingProjectPath = "differentiable_exports/contract_test";
              state.exportedTrainingProject = {{
                ready_for_external_training: true,
                artifacts_ready: false,
                captures: [{{
                  split: "train",
                  label: "Train",
                  requested_samples: 80,
                  captured_samples: 80,
                  status: "complete",
                }}],
                demo_starter: {{
                  included: true,
                  training: {{ working_directory: "differentiable_exports/contract_test/reference_training", command: "python train.py" }},
                  evaluation: {{ working_directory: "differentiable_exports/contract_test/reference_training", command: "python evaluate.py" }},
                }},
              }};
              const workflow = exportedTrainingProjectWorkflowMarkup("differentiable_exports/contract_test");
              const previewError = differentiableExportGraphMarkup(null, "preview details", recipe);
              const typedDagPreview = differentiableExportGraphMarkup({{
                blocks: [{{ block: "BranchBlock", noema_op_id: "test.branch" }}],
                metadata: {{ execution_issue: "The downstream support route is a branched DAG." }},
              }}, "", recipe);
              globalThis.__result = {{ operationTable, soleRequiredToggle, portableReplacementToggle, misleadingFineTuneToggle, replacementCandidates, capabilityHint, noReplacementNotice, captureBackedNotice, neutral, neutralPlan, capture, busyCapture, pendingWorkflow, neutralReadyWorkflow, workflow, previewError, typedDagPreview }};
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
        neutral = result["neutral"]
        operation_table = result["operationTable"]
        sole_required_toggle = result["soleRequiredToggle"]
        portable_replacement_toggle = result["portableReplacementToggle"]
        capture = result["capture"]
        busy_capture = result["busyCapture"]
        pending_workflow = result["pendingWorkflow"]
        neutral_ready_workflow = result["neutralReadyWorkflow"]
        workflow = result["workflow"]
        preview_error = result["previewError"]
        typed_dag_preview = result["typedDagPreview"]
        self.assertNotIn("Typed DAG required", typed_dag_preview)
        self.assertIn("BranchBlock", typed_dag_preview)
        self.assertNotIn("objective", result["neutralPlan"])
        self.assertNotIn("starter", result["neutralPlan"])

        self.assertIn("Training bundle", neutral)
        self.assertIn("model ABI + data contract", neutral)
        self.assertNotIn('data-differentiable-export-field="optimizable"', neutral)
        self.assertIn("Built-in fine-tuning", operation_table)
        self.assertIn("Portable replacement", operation_table)
        self.assertIn(">Train/replace</th>", operation_table)
        self.assertIn("Differentiable support", operation_table)
        self.assertNotIn("Train/replace block", operation_table)
        self.assertNotIn("Replaceable slot", operation_table)
        self.assertNotIn("<th>Exportable</th>", operation_table)
        self.assertIn("<th>Block</th>", operation_table)
        self.assertIn(">Encoder</td>", operation_table)
        self.assertIn(">Decoder</td>", operation_table)
        self.assertNotIn("<th>Step</th>", operation_table)
        self.assertIn('data-differentiable-export-slot="sender"', operation_table)
        self.assertIn('data-differentiable-export-slot="receiver"', operation_table)
        self.assertNotIn(
            'data-differentiable-export-slot="fine_tunable_only"', operation_table
        )
        self.assertNotIn(
            'data-differentiable-export-slot="differentiable_support"', operation_table
        )
        self.assertEqual(
            result["replacementCandidates"], ["sender", "receiver"]
        )
        self.assertIn("built-in fine-tuning", operation_table.lower())
        self.assertIn("portable trained-artifact", operation_table.lower())
        self.assertIn("does not fine-tune", operation_table.lower())
        self.assertNotIn(
            "Can serve as a typed externally implemented training slot",
            operation_table,
        )
        self.assertNotIn("Frozen support block", operation_table)
        self.assertIn('data-differentiable-export-slot="estimator" checked disabled', sole_required_toggle)
        self.assertIn("required", sole_required_toggle)
        self.assertIn("requires at least one selected replacement boundary", sole_required_toggle)
        self.assertIn("portable trained-artifact replacement ABI", portable_replacement_toggle)
        self.assertNotIn("data-differentiable-export-slot", result["misleadingFineTuneToggle"])
        self.assertIn("No portable replacement ABI", result["misleadingFineTuneToggle"])
        self.assertIn("1 built-in fine-tuning", result["capabilityHint"])
        self.assertIn("2 portable replacement", result["capabilityHint"])
        self.assertIn("no portable replacement boundary", result["noReplacementNotice"].lower())
        self.assertNotIn("no typed trainable slot", result["noReplacementNotice"].lower())
        self.assertIn("Capture-backed training", result["captureBackedNotice"])
        self.assertIn("No recipe backward route required", result["captureBackedNotice"])
        self.assertNotIn("hard_decision", result["captureBackedNotice"])
        self.assertNotIn('data-differentiable-export-slot="sender"', neutral)
        self.assertNotIn('differentiable-export-slot-list', neutral)
        self.assertNotIn("Selected replacement blocks", neutral)
        self.assertNotIn("Downstream route endpoint", neutral)
        self.assertNotIn("data-training-loss-step", neutral)
        self.assertIn("tx_power, wireless_channel", neutral)
        self.assertNotIn("unrelated_global_block", neutral)
        self.assertNotIn("Training targets", neutral)
        self.assertIn('data-differentiable-export-field="framework"', neutral)
        self.assertNotIn('data-differentiable-export-field="starter"', neutral)
        self.assertNotIn('data-differentiable-export-field="loss"', neutral)
        self.assertNotIn("Demo starter", neutral)
        self.assertNotIn("Example loss", neutral)
        self.assertNotIn('class="workbench-help"', neutral)
        self.assertIn('data-ui-tooltip="The bundle records the returned-model ABI', neutral)
        self.assertIn(
            'class="primary workbench-primary-action differentiable-export-submit"',
            neutral,
        )
        self.assertIn(">Export training bundle</button>", neutral)
        self.assertIn('data-workbench-copy', neutral)
        self.assertIn('class="action-icon-glyph"', neutral)
        self.assertLess(neutral.index("Dataset definition"), neutral.index("Training bundle"))
        self.assertLess(neutral.index("Bundle directory"), neutral.index("Support framework"))
        self.assertLess(neutral.index("Support framework"), neutral.index("Export training bundle"))
        self.assertLess(neutral.index("Export training bundle"), neutral.index("Dataset capture"))
        self.assertLess(neutral.index("Dataset capture"), neutral.index("External model"))
        self.assertIn('class="training-bundle-support-row"', neutral)
        self.assertNotIn("External training handoff", neutral)
        self.assertNotIn("Materialize the frozen data plan.", neutral)
        self.assertNotIn("Available after the training bundle is exported.", neutral)
        self.assertNotIn("Train externally", neutral)
        self.assertNotIn("Return and validate model", neutral)
        self.assertIn('class="exported-project-workflow locked"', neutral)
        self.assertIn("export bundle first", neutral)
        self.assertIn('data-exported-project-capture-all disabled', neutral)
        self.assertIn('data-exported-project-validate disabled', neutral)
        self.assertNotIn("Select on Graph", neutral)
        self.assertNotIn("Trainable slots and return targets", neutral)
        self.assertNotIn(
            '<div class="training-alert compact">The contract records', neutral
        )

        self.assertIn('class="exported-project-capture-strip"', capture)
        self.assertIn(
            'class="primary workbench-primary-action exported-project-capture-all"',
            capture,
        )
        self.assertNotIn('class="exported-project-stage-heading"', capture)
        self.assertIn("Train", capture)
        self.assertIn("80", capture)
        self.assertIn("80%", capture)
        self.assertIn("Validation", capture)
        self.assertIn("20%", capture)
        self.assertIn("channel_gain", capture)
        self.assertIn("wireless_channel.channel_gain", capture)
        self.assertIn("Dataset capture", workflow)
        self.assertIn("External model", workflow)
        self.assertNotIn("External training handoff", workflow)
        self.assertIn("Validate returned model", workflow)
        self.assertNotIn("Refresh bundle", pending_workflow)
        self.assertIn("Capture all datasets", capture)
        self.assertIn("Capturing…", busy_capture)
        self.assertIn('data-exported-project-capture-all disabled', busy_capture)
        self.assertIn('class="training-card wide exported-project-dataset-card', pending_workflow)
        self.assertIn('data-exported-project-validate disabled', pending_workflow)
        self.assertNotIn("Capture datasets before training", pending_workflow)
        self.assertIn("No trainer included.", neutral_ready_workflow)
        self.assertIn("No capture required", neutral_ready_workflow)
        self.assertIn('data-exported-project-capture-all disabled', neutral_ready_workflow)
        self.assertIn("No trainer included.", neutral_ready_workflow)
        self.assertIn("Check bundle", neutral_ready_workflow)
        self.assertIn(
            "cd -- differentiable_exports/neutral_contract &amp;&amp; python validate_contract.py",
            neutral_ready_workflow,
        )
        self.assertNotIn('aria-label="Copy training command"', neutral_ready_workflow)
        self.assertNotIn(
            '<code class="exported-project-command">cd -- differentiable_exports/neutral_contract</code>',
            neutral_ready_workflow,
        )
        self.assertIn("Train · attached demo", workflow)
        self.assertIn(
            "cd -- differentiable_exports/contract_test/reference_training &amp;&amp; python train.py",
            workflow,
        )
        self.assertIn(
            "cd -- differentiable_exports/contract_test/reference_training &amp;&amp; python evaluate.py",
            workflow,
        )
        self.assertLess(workflow.index("Dataset capture"), workflow.index("External model"))
        self.assertLess(workflow.index("External model"), workflow.index("Validate returned model"))
        self.assertIn(
            'class="primary workbench-primary-action exported-project-validate"',
            workflow,
        )
        self.assertIn("Recapture all datasets", workflow)
        self.assertNotIn('data-exported-project-capture-all disabled', workflow)
        self.assertNotIn('data-exported-project-validate disabled', workflow)
        self.assertNotIn("Capture datasets before training", workflow)
        self.assertNotIn("Train demo starter", workflow)
        self.assertNotIn("No trainer included.", workflow)
        self.assertNotIn("Use trained artifact", workflow)
        self.assertEqual(preview_error, "")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the Workbench recapture test")
    def test_capture_all_overwrites_every_managed_split_after_completion(self):
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
            const execution = vm.runInContext(source + `
              (async () => {{
                const calls = [];
                state.selectedRecipe = {{ key: "allocation" }};
                state.exportedTrainingProject = {{
                  ready_for_external_training: true,
                  captures: [
                    {{ split: "train", label: "Train", status: "complete" }},
                    {{ split: "validation", label: "Validation", status: "complete" }},
                    {{ split: "test", label: "Held-out test", status: "complete" }},
                  ],
                }};
                renderTrainingView = () => {{}};
                startRecipeActivityProgress = () => {{}};
                refreshExportedTrainingProject = async () => state.exportedTrainingProject;
                runExportedProjectCapture = async (split, context) => {{
                  calls.push({{ split, captureStatus: context.capture.status }});
                  return true;
                }};
                await runAllExportedProjectCaptures();
                globalThis.__result = calls;
              }})()
            `, sandbox);
            execution.then(() => process.stdout.write(JSON.stringify(sandbox.__result)));
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
        self.assertEqual(
            json.loads(completed.stdout),
            [
                {"split": "train", "captureStatus": "complete"},
                {"split": "validation", "captureStatus": "complete"},
                {"split": "test", "captureStatus": "complete"},
            ],
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the Workbench capture-plan test")
    def test_capture_plan_is_visible_and_edits_persist_to_the_training_plan(self):
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
              const recipe = {{
                name: "capture_ui",
                dataset_capture: {{
                  taps: [{{ id: "channel_gains", from: "channel_state.state" }}],
                  split_plan: {{
                    total_samples: 192,
                    percentages: {{ train: 66.6666666667, validation: 16.66666666665, test: 16.66666666665 }},
                  }},
                }},
                steps: [{{
                  id: "wireless_channel",
                  op: "wireless.channel",
                  inputs: {{}},
                  params: {{ snr_db: 6 }},
                }}],
              }};
              const contract = {{
                mode: "captured_tensors",
                ready: true,
                sample_unit: "CSI realizations",
                total_samples: 192,
                suggested_total_samples: 24576,
                required_taps: [{{ id: "channel_gains", from: "channel_state.state", role: "replacement_input:tx_power.channel_state" }}],
                selected_taps: [{{ id: "channel_gains", from: "channel_state.state" }}],
                candidates: [
                  {{ id: "channel_gains", from: "channel_state.state", kind: "channel.ofdm_channel_state.numpy", selectable: true, relationship: "scenario_signal" }},
                  {{ id: "allocation", from: "tx_power.allocation", kind: "channel.power_allocation.numpy", selectable: true, relationship: "current_replacement_output", reason: "Produced by the currently installed allocator." }},
                  {{ id: "rx_bits", from: "receiver.bits", kind: "channel.bits.numpy", selectable: true, relationship: "current_pipeline_dependent", reason: "Produced downstream under the current pipeline." }},
                ],
                split_plan: {{
                  percentages: {{ train: 66.6666666667, validation: 16.66666666665, test: 16.66666666665 }},
                  counts: {{ train: 128, validation: 32, test: 32 }},
                }},
              }};
              state.operations = [{{
                id: "wireless.channel",
                name: "Wireless channel",
                status: "implemented",
                input_kinds: {{}},
                optional_input_kinds: {{}},
                output_kinds: {{ rx_symbols: "channel.rx_symbols.complex_numpy" }},
                params_schema: {{
                  type: "object",
                  properties: {{ snr_db: {{ type: "number" }} }},
                  additionalProperties: false,
                }},
              }}];
              const markup = trainingCaptureContractMarkup(recipe, contract);
              const pipelineSelectedMarkup = trainingCaptureContractMarkup({{
                ...recipe,
                dataset_capture: {{
                  ...recipe.dataset_capture,
                  taps: [...recipe.dataset_capture.taps, {{ id: "allocation", from: "tx_power.allocation" }}],
                }},
              }}, {{
                ...contract,
                selected_taps: [...contract.selected_taps, {{ id: "allocation", from: "tx_power.allocation" }}],
              }});
              state.editRecipe = recipe;
              state.trainingCaptureContract = contract;
              let saved = null;
              applyDatasetCaptureRecipe = (value) => {{
                saved = value;
                state.trainingPlan = {{
                  ...ensureCurrentTrainingPlan(),
                  dataset_capture: structuredCloneFallback(value.dataset_capture || {{}}),
                }};
              }};
              updateTrainingCaptureSweep({{ currentTarget: {{
                value: '{{"wireless_channel.snr_db":[-2,2,6,10]}}',
                setAttribute: () => {{}},
                removeAttribute: () => {{}},
              }} }});
              const sweepSaved = structuredCloneFallback(saved);
              updateTrainingCaptureSweep({{ currentTarget: {{
                value: '{{"missing.snr_db":[0,1]}}',
                setAttribute: () => {{}},
                removeAttribute: () => {{}},
              }} }});
              const invalidSweepError = state.trainingDatasetCaptureError;
              const afterInvalidSweep = structuredCloneFallback(saved);
              updateTrainingCaptureSignal({{ currentTarget: {{ checked: true, dataset: {{ trainingCaptureSignal: "receiver.bits", trainingCaptureSignalId: "rx_bits" }} }} }});
              updateTrainingCapturePlanField({{ currentTarget: {{ value: "300", dataset: {{ trainingCapturePlanField: "total_samples" }} }} }});
              updateTrainingCapturePlanField({{ currentTarget: {{ value: "60", dataset: {{ trainingCapturePlanField: "train" }} }} }});
              updateTrainingCapturePlanField({{ currentTarget: {{ value: "20", dataset: {{ trainingCapturePlanField: "validation" }} }} }});
              const normalSaved = structuredCloneFallback(saved);
              const updatedMarkup = trainingCaptureContractMarkup(normalSaved, contract);
              updateTrainingCapturePlanField({{ currentTarget: {{ value: "90", dataset: {{ trainingCapturePlanField: "train" }} }} }});
              const invalidSplitError = state.trainingDatasetCaptureError;
              const invalidSplitMarkup = trainingCaptureContractMarkup(currentTrainingRecipe(), contract);
              state.editRecipe = {{
                name: "implicit_required",
                dataset_capture: {{
                  split_plan: {{
                    total_samples: 10,
                    percentages: {{ train: 50, validation: 25, test: 25 }},
                    counts: {{ train: 5, validation: 3, test: 2 }},
                  }},
                }},
                steps: [],
              }};
              updateTrainingCaptureSignal({{ currentTarget: {{ checked: true, dataset: {{ trainingCaptureSignal: "receiver.bits", trainingCaptureSignalId: "rx_bits" }} }} }});
              updateTrainingCapturePlanField({{ currentTarget: {{ value: "20", dataset: {{ trainingCapturePlanField: "total_samples" }} }} }});
              globalThis.__result = {{ markup, updatedMarkup, pipelineSelectedMarkup, normalSaved, implicitSaved: saved, invalidSplitError, invalidSplitMarkup, sweepSaved, invalidSweepError, afterInvalidSweep }};
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
        markup = result["markup"]
        saved = result["normalSaved"]
        implicit_saved = result["implicitSaved"]

        self.assertIn("Captured signals", markup)
        self.assertIn("Capture coordinates", markup)
        self.assertIn('data-training-capture-sweep', markup)
        self.assertIn('class="workbench-selectable-list training-capture-signal-list all-signals"', markup)
        self.assertNotIn("Add or remove optional graph outputs", markup)
        self.assertIn("CSI realizations", markup)
        self.assertIn('data-training-capture-plan-field="total_samples"', markup)
        self.assertIn('data-training-capture-plan-field="train"', markup)
        self.assertIn('data-training-capture-plan-field="validation"', markup)
        self.assertNotIn('data-training-capture-plan-field="test"', markup)
        self.assertIn("Held-out test", markup)
        self.assertIn("16.6667%", markup)
        self.assertIn("derived", markup)
        self.assertIn("128 CSI realizations", markup)
        self.assertIn('data-training-capture-signal="channel_state.state"', markup)
        self.assertIn('data-training-capture-signal="tx_power.allocation"', markup)
        self.assertIn("current pipeline", markup)
        self.assertNotIn("Use suggested", markup)
        self.assertNotIn("data-training-capture-use-suggestion", markup)
        self.assertIn("Produced by the currently installed allocator.", markup)
        original_positions = [
            markup.index('data-training-capture-signal="channel_state.state"'),
            markup.index('data-training-capture-signal="tx_power.allocation"'),
            markup.index('data-training-capture-signal="receiver.bits"'),
        ]
        updated_positions = [
            result["updatedMarkup"].index(
                'data-training-capture-signal="channel_state.state"'
            ),
            result["updatedMarkup"].index(
                'data-training-capture-signal="tx_power.allocation"'
            ),
            result["updatedMarkup"].index(
                'data-training-capture-signal="receiver.bits"'
            ),
        ]
        self.assertEqual(original_positions, sorted(original_positions))
        self.assertEqual(updated_positions, sorted(updated_positions))
        required_input = markup.split(
            'data-training-capture-signal="channel_state.state"', 1
        )[1].split("/>", 1)[0]
        self.assertIn("checked", required_input)
        self.assertIn("disabled", required_input)
        self.assertIn('value="300"', result["updatedMarkup"])
        self.assertIn('value="60"', result["updatedMarkup"])
        self.assertIn("20%", result["updatedMarkup"])
        self.assertIn("180 CSI realizations", result["updatedMarkup"])
        pipeline_input = result["pipelineSelectedMarkup"].split(
            'data-training-capture-signal="tx_power.allocation"', 1
        )[1].split("/>", 1)[0]
        self.assertIn("checked", pipeline_input)
        self.assertNotIn("disabled", pipeline_input)
        self.assertEqual(saved["dataset_capture"]["split_plan"]["total_samples"], 300)
        self.assertEqual(
            saved["dataset_capture"]["split_plan"]["percentages"],
            {"train": 60, "validation": 20, "test": 20},
        )
        self.assertEqual(
            result["sweepSaved"]["dataset_capture"]["sweep"],
            {"wireless_channel.snr_db": [-2, 2, 6, 10]},
        )
        self.assertEqual(
            result["afterInvalidSweep"]["dataset_capture"]["sweep"],
            {"wireless_channel.snr_db": [-2, 2, 6, 10]},
        )
        self.assertIn("not an editable parameter", result["invalidSweepError"])
        self.assertIn("less than 100%", result["invalidSplitError"])
        self.assertIn("training-capture-plan-error", result["invalidSplitMarkup"])
        self.assertIn("less than 100%", result["invalidSplitMarkup"])
        self.assertIn(
            {"id": "rx_bits", "from": "receiver.bits"},
            saved["dataset_capture"]["taps"],
        )
        self.assertIn(
            {"id": "channel_gains", "from": "channel_state.state"},
            implicit_saved["dataset_capture"]["taps"],
        )
        self.assertNotIn("counts", implicit_saved["dataset_capture"]["split_plan"])
        self.assertEqual(implicit_saved["dataset_capture"]["split_plan"]["total_samples"], 20)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the Workbench source-data test")
    def test_file_backed_training_data_is_explicitly_not_a_capture_job(self):
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
              globalThis.__markup = trainingCaptureContractMarkup({{ name: "deepjscc" }}, {{
                mode: "file_backed_live_differentiable",
                total_samples: 20,
                sample_unit: "images",
                split_plan: {{ editable: false, counts: {{ train: 16, validation: 4 }}, test: "ordinary benchmark" }},
                explanation: "No tensor capture is required.",
              }});
              globalThis.__invalidMarkup = trainingCaptureContractMarkup({{ name: "deepjscc" }}, {{
                mode: "file_backed_live_differentiable",
                ready: false,
                issue: "missing source images",
                total_samples: 0,
                sample_unit: "images",
                split_plan: {{ editable: false, counts: {{ train: 0, validation: 0 }} }},
              }});
            `, sandbox);
            process.stdout.write(JSON.stringify({{ markup: sandbox.__markup, invalidMarkup: sandbox.__invalidMarkup }}));
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
        markup = result["markup"]
        self.assertIn("20 images", markup)
        self.assertIn("16</b> train", markup)
        self.assertIn("4</b> validation", markup)
        self.assertIn('data-ui-tooltip="No tensor capture is required."', markup)
        self.assertNotIn("Training and validation come from", markup)
        self.assertIn("fixed source partition", markup)
        self.assertIn("Source unresolved", result["invalidMarkup"])
        self.assertIn("missing source images", result["invalidMarkup"])

    def test_primary_request_is_one_neutral_contract_endpoint(self):
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("async function runDifferentiableExport()")
        end = source.index("function currentExportedTrainingProjectPath()", start)
        request_source = source[start:end]

        self.assertIn('api("/api/recipe/differentiable-export"', request_source)
        self.assertNotIn("differentiable-export-graph", request_source)
        self.assertNotIn("include_starter", request_source)
        self.assertNotIn("request.starter", request_source)
        self.assertNotIn("request.exporter", request_source)
        self.assertNotIn("request.loss", request_source)

    def test_exported_project_state_is_watched_without_a_manual_refresh_action(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn("/api/workbench/exported-project-watch?path=", source)
        self.assertIn("restartExportedTrainingProjectWatch()", source)
        self.assertIn('document.addEventListener("visibilitychange"', source)
        self.assertIn('window.addEventListener("focus"', source)
        self.assertNotIn("data-exported-project-refresh", source)
        self.assertNotIn("Refresh bundle", source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the bundle-watch test")
    def test_exported_project_watch_refreshes_only_when_revision_changes(self):
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const vm = require("vm");
            let source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            source = source.replace(/\ninit\(\);\s*$/, "\n");
            const sandbox = {{
              console,
              localStorage: {{ getItem: () => null, setItem: () => {{}} }},
              document: {{
                hidden: false,
                documentElement: {{ dataset: {{}} }},
                getElementById: () => null,
              }},
              window: {{ CSS: null }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              globalThis.__done = (async () => {{
                const calls = [];
                let revision = "revision-1";
                state.activeView = "training";
                state.editRecipe = {{ name: "watch_demo", steps: [] }};
                state.differentiableExportSettings = {{
                  optimizable: "receiver",
                  framework: "torch",
                  out: "differentiable_exports/watch_demo",
                  force: false,
                }};
                renderTrainingView = () => {{}};
                renderRecipeControls = () => {{}};
                api = async (path) => {{
                  calls.push(path);
                  if (path.startsWith("/api/workbench/exported-project-watch")) {{
                    return {{ status: "ok", exists: true, revision }};
                  }}
                  if (path.startsWith("/api/workbench/exported-project?")) {{
                    return {{ status: "ok", revision, ready_for_external_training: true, captures: [] }};
                  }}
                  if (path === "/api/trained-artifacts") return {{ status: "ok", artifacts: [] }};
                  throw new Error("unexpected request " + path);
                }};
                await pollExportedTrainingProjectWatch(state.exportedProjectWatchGeneration, false);
                const firstCalls = calls.slice();
                calls.length = 0;
                await pollExportedTrainingProjectWatch(state.exportedProjectWatchGeneration, false);
                const unchangedCalls = calls.slice();
                calls.length = 0;
                revision = "revision-2";
                await pollExportedTrainingProjectWatch(state.exportedProjectWatchGeneration, false);
                return {{
                  firstCalls,
                  unchangedCalls,
                  changedCalls: calls.slice(),
                  finalRevision: state.exportedProjectWatchToken,
                }};
              }})();
            `, sandbox);
            sandbox.__done.then((result) => process.stdout.write(JSON.stringify(result))).catch((error) => {{
              console.error(error);
              process.exitCode = 1;
            }});
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
        self.assertEqual(len(result["firstCalls"]), 3)
        self.assertEqual(len(result["unchangedCalls"]), 1)
        self.assertEqual(len(result["changedCalls"]), 3)
        self.assertEqual(result["finalRevision"], "revision-2")

    def test_exported_project_watch_revision_tracks_external_handoff_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "watch_demo"
            missing = _exported_project_watch_payload(project, root)
            self.assertFalse(missing["exists"])

            project.mkdir(parents=True)
            artifact_manifest = project / "trained_artifact.yaml"
            capture_dir = project / "data" / "train"
            manifest = {
                "schema_version": 1,
                "kind": "noema.training_interface_bundle@1",
                "capture_jobs": [
                    {
                        "split": "train",
                        "output_dir": str(capture_dir.relative_to(root)),
                    }
                ],
                "trained_artifacts": [
                    {"manifest_path": str(artifact_manifest.relative_to(root))}
                ],
                "training": {"working_directory": str(project.relative_to(root))},
                "evaluation": {},
            }
            (project / "project_manifest.yaml").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            initial = _exported_project_watch_payload(project, root)
            self.assertTrue(initial["exists"])

            capture_dir.mkdir(parents=True)
            (capture_dir / "schema.json").write_text("{}", encoding="utf-8")
            captured = _exported_project_watch_payload(project, root)
            self.assertNotEqual(captured["revision"], initial["revision"])

            component = project / "artifacts" / "receiver.onnx"
            component.parent.mkdir()
            component.write_bytes(b"model-v1")
            artifact_manifest.write_text(
                json.dumps(
                    {
                        "kind": "noema.trained_block_artifact",
                        "schema_version": 2,
                        "components": [
                            {"id": "receiver", "path": "artifacts/receiver.onnx"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            returned = _exported_project_watch_payload(project, root)
            self.assertNotEqual(returned["revision"], captured["revision"])

            component.write_bytes(b"model-v2-with-different-size")
            changed_component = _exported_project_watch_payload(project, root)
            self.assertNotEqual(changed_component["revision"], returned["revision"])

    def test_exported_project_watch_surfaces_invalid_manifest_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "differentiable_exports" / "invalid_watch"
            project.mkdir(parents=True)
            manifest_path = project / "project_manifest.yaml"
            manifest_path.write_text('{"capture_jobs": [], "capture_jobs": []}', encoding="utf-8")

            duplicate = _exported_project_watch_payload(project, root)
            self.assertEqual(duplicate["status"], "invalid")
            self.assertTrue(any("Duplicate" in issue for issue in duplicate["issues"]))

            manifest_path.write_text(
                json.dumps(
                    {
                        "capture_jobs": [{"output_dir": ""}],
                        "trained_artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            empty_path = _exported_project_watch_payload(project, root)
            self.assertEqual(empty_path["status"], "invalid")
            self.assertTrue(
                any("capture_jobs[0].output_dir" in issue for issue in empty_path["issues"])
            )

    def test_operation_table_uses_themed_workbench_tooltips(self):
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function trainingStepTable(report)")
        end = source.index("function differentiableExportGraphMarkup", start)
        table_source = source[start:end]
        self.assertIn("data-ui-tooltip", table_source)
        self.assertNotIn(' title="', table_source)

    def test_capture_inspection_freshness_includes_trainable_slot_selection(self):
        source = APP_JS.read_text(encoding="utf-8")
        start = source.index("function trainingInspectionRequestSignature")
        end = source.index("async function refreshTrainingInspection", start)
        signature_source = source[start:end]
        self.assertIn("differentiableExportSettings.optimizable", signature_source)
        self.assertIn("::slots=", signature_source)
        self.assertNotIn("::loss_steps=", signature_source)

    def test_training_data_edits_preserve_the_workbench_dom(self):
        source = APP_JS.read_text(encoding="utf-8")
        signal_start = source.index("function updateTrainingCaptureSignal")
        signal_end = source.index("function updateTrainingCapturePlanField", signal_start)
        plan_end = source.index("async function runDifferentiableExport", signal_end)
        apply_start = source.index("function applyDatasetCaptureRecipe")
        apply_end = source.index("async function runDifferentiableGraphExport", apply_start)
        refresh_start = source.index("async function refreshTrainingInspection")
        refresh_end = source.index("function removeStaleTrainingBundleWorkflow", refresh_start)

        self.assertIn("preserveEditor: true", source[signal_start:plan_end])
        preserve_branch = source[apply_start:apply_end].split(
            "if (options.preserveEditor)", 1
        )[1].split("return;", 1)[0]
        self.assertIn("syncTrainingCaptureEditor()", preserve_branch)
        self.assertIn("scheduleSilentTrainingInspection()", preserve_branch)
        self.assertNotIn("renderTrainingView()", preserve_branch)
        self.assertIn("if (render) renderTrainingView()", source[refresh_start:refresh_end])
        self.assertIn("else syncTrainingCaptureEditor()", source[refresh_start:refresh_end])
        reset_start = source.index("function removeStaleTrainingBundleWorkflow")
        reset_end = source.index("function scheduleSilentTrainingInspection", reset_start)
        reset_source = source[reset_start:reset_end]
        self.assertIn("exportedTrainingProjectWorkflowMarkup", reset_source)
        self.assertNotIn("workflow.remove()", reset_source)

    def test_training_handoff_primary_actions_share_run_button_sizing(self):
        styles = STYLES_CSS.read_text(encoding="utf-8")
        shared_start = styles.index(".workbench-primary-action {")
        shared_end = styles.index("}", shared_start)
        shared_rule = styles[shared_start:shared_end]
        primary_start = styles.index("button.primary {")
        primary_end = styles.index("}", primary_start)
        primary_rule = styles[primary_start:primary_end]

        self.assertIn("width: 190px", shared_rule)
        self.assertIn("min-width: 190px", shared_rule)
        self.assertIn("max-width: 100%", shared_rule)
        self.assertIn("height: 34px", shared_rule)
        self.assertIn("background: var(--accent)", primary_rule)
        self.assertIn("border-color: var(--accent)", primary_rule)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the shell quoting test")
    def test_training_handoff_shell_quotes_copyable_working_directories(self):
        script = textwrap.dedent(
            rf"""
            const fs = require("fs");
            const source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
            const start = source.indexOf("function shellQuoteCommandPath");
            const end = source.indexOf("function escapeHtml", start);
            eval(source.slice(start, end));
            process.stdout.write(JSON.stringify([
              shellQuoteCommandPath("differentiable_exports/demo"),
              shellQuoteCommandPath("/tmp/research bundle; $(touch bad)"),
              shellQuoteCommandPath("/tmp/researcher's bundle"),
            ]));
            """
        )
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            [
                "differentiable_exports/demo",
                "'/tmp/research bundle; $(touch bad)'",
                "'/tmp/researcher'\"'\"'s bundle'",
            ],
        )

    def test_bundle_overwrite_is_opt_in_and_directory_edits_invalidate_stale_handoff(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn(
            'differentiableExportSettings: { optimizable: "", framework: "torch", out: "", force: false }',
            source,
        )
        setting_start = source.index("function updateDifferentiableExportSetting")
        setting_end = source.index("function updateDifferentiableExportSlotSelection", setting_start)
        setting_source = source[setting_start:setting_end]
        self.assertIn('if (field === "out"', setting_source)
        self.assertIn("removeStaleTrainingBundleWorkflow()", setting_source)

    def test_workbench_omits_the_advanced_route_endpoint_control(self):
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn('[data-training-loss-step]', source)
        self.assertNotIn("function updateTrainingLossStepSelection", source)
        start = source.index("function currentTrainingPlanPayload")
        end = source.index("function currentTrainingRecipe", start)
        payload_builder = source[start:end]
        self.assertIn("delete plan.loss_steps", payload_builder)

    def test_exported_project_surfaces_optional_starter_without_overwriting_neutral_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            project = root / "differentiable_exports" / "demo"
            starter = project / "reference_training"
            starter.mkdir(parents=True)
            handoff_files = {
                "RUN_DEMO.md": "# Run demo\n",
                "train_demo.py": "print('train')\n",
                "evaluate_demo.py": "print('evaluate')\n",
            }
            for relative, content in handoff_files.items():
                (project / relative).write_text(content, encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "kind": "noema.training_interface_bundle@1",
                "training": {
                    "owner": "external_researcher",
                    "working_directory": str(project.relative_to(root)),
                },
                "evaluation": {"owner": "noema_ordinary_recipe_or_benchmark"},
                "external_training": {
                    "architecture": "not_supplied",
                    "loss": "not_supplied",
                    "trainer": "not_supplied",
                    "optional_demo_scaffold": {
                        "included": True,
                        "normative": False,
                        "path": "reference_training",
                        "training": {
                            "working_directory": str(starter.relative_to(root)),
                            "command": "python train.py",
                            "history_path": str((starter / "training_history.json").relative_to(root)),
                        },
                        "evaluation": {
                            "command": "python evaluate.py",
                            "metrics_path": str((starter / "evaluation_metrics.json").relative_to(root)),
                        },
                        "root_handoff": {
                            "ownership": "noema_demo_helper",
                            "normative": False,
                            "working_directory": ".",
                            "quickstart": "RUN_DEMO.md",
                            "training": {
                                "path": "train_demo.py",
                                "command": "python train_demo.py",
                                "target": "reference_training/train.py",
                            },
                            "evaluation": {
                                "path": "evaluate_demo.py",
                                "command": "python evaluate_demo.py",
                                "target": "reference_training/evaluate.py",
                            },
                            "managed_files": [
                                {
                                    "path": relative,
                                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                                }
                                for relative, content in handoff_files.items()
                            ],
                        },
                    },
                },
                "capture_jobs": [],
                "trained_artifacts": [],
            }
            (starter / "training_history.json").write_text("[]", encoding="utf-8")
            (starter / "evaluation_metrics.json").write_text("{}", encoding="utf-8")
            (project / "project_manifest.yaml").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            payload = _exported_project_payload(project, root)

            self.assertEqual(payload["training"]["owner"], "external_researcher")
            self.assertNotIn("command", payload["training"])
            self.assertEqual(
                payload["training"]["history_path"],
                "differentiable_exports/demo/training_history.json",
            )
            self.assertFalse(payload["training"]["history_exists"])
            self.assertEqual(
                payload["evaluation"]["owner"],
                "noema_ordinary_recipe_or_benchmark",
            )
            self.assertEqual(
                payload["evaluation"]["metrics_path"],
                "differentiable_exports/demo/evaluation_metrics.json",
            )
            self.assertFalse(payload["evaluation"]["metrics_exist"])
            self.assertEqual(
                payload["demo_starter"]["training"]["command"],
                "uv run --project %s --extra onnx python train_demo.py"
                % shlex.quote(str(root.resolve())),
            )
            self.assertEqual(
                payload["demo_starter"]["training"]["working_directory"],
                "differentiable_exports/demo",
            )
            self.assertEqual(
                payload["demo_starter"]["evaluation"]["command"],
                "uv run --project %s --extra onnx python evaluate_demo.py"
                % shlex.quote(str(root.resolve())),
            )
            self.assertEqual(
                payload["demo_starter"]["evaluation"]["working_directory"],
                "differentiable_exports/demo",
            )
            self.assertTrue(payload["demo_starter"]["training"]["history_exists"])
            self.assertTrue(payload["demo_starter"]["evaluation"]["metrics_exist"])
            self.assertTrue(payload["demo_starter"]["root_handoff"]["active"])
            self.assertEqual(payload["demo_starter"]["root_handoff"]["issues"], [])

            (project / "train_demo.py").write_text("print('changed')\n", encoding="utf-8")
            stale = _exported_project_payload(project, root)
            self.assertEqual(
                stale["demo_starter"]["training"]["command"],
                "uv run --project %s --extra onnx python train.py"
                % shlex.quote(str(root.resolve())),
            )
            self.assertEqual(
                stale["demo_starter"]["training"]["working_directory"],
                "differentiable_exports/demo/reference_training",
            )
            self.assertFalse(stale["demo_starter"]["root_handoff"]["active"])
            self.assertIn(
                "training launcher hash does not match",
                stale["demo_starter"]["root_handoff"]["issues"],
            )

    def test_architecture_document_separates_contract_and_external_training(self):
        document = ARCHITECTURE_DOC.read_text(encoding="utf-8")
        self.assertIn("architecture- and loss-neutral **training bundle**", document)
        self.assertIn("Workbench exports only the neutral contract", document)
        self.assertNotIn("**Include demo starter**", document)
        self.assertIn("The preferred neutral ABI is ONNX", document)
        self.assertIn("Researchers own model architecture, loss, optimization, trainer, and model selection", document)


if __name__ == "__main__":
    unittest.main()
