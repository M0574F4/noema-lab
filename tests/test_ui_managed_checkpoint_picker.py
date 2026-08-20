from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path

from noema_lab.core.recipes import load_recipe
from noema_lab.ops import build_registry


ROOT = Path(__file__).resolve().parents[1]


class ManagedCheckpointPickerTests(unittest.TestCase):
    def test_symbol_allocator_declares_one_managed_checkpoint_control(self):
        operation = build_registry().get("model.symbol_power_allocator").describe()
        properties = operation["params_schema"]["properties"]
        picker = properties["checkpoint_path"]["x-noema-ui"]

        model_batch = properties["model_batch_size"]
        self.assertEqual(model_batch["default"], 1024)
        self.assertEqual(model_batch["minimum"], 1)
        self.assertFalse(model_batch["x-noema-ui"]["allow_sweep"])
        self.assertEqual(
            model_batch["x-noema-ui"]["enabled_when"],
            {"policy": ["learned_checkpoint", "learned_artifact"]},
        )

        self.assertEqual(picker["control"], "trained_artifact")
        self.assertEqual(picker["visible_when"], {"policy": "learned_checkpoint"})
        self.assertIn(".npz", picker["accept"])
        self.assertEqual(picker["label"], "Learned checkpoint")
        for name in (
            "checkpoint_sha256",
            "checkpoint_format",
            "checkpoint_strict",
            "checkpoint_max_bytes",
        ):
            self.assertTrue(properties[name]["x-noema-ui"]["hidden"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the allocator-policy UI test")
    def test_resource_allocator_keeps_model_picker_beside_policy_and_enables_it_for_learned_policy(self):
        operation = build_registry().get("model.symbol_power_allocator").describe()
        app_js = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
        step = {
            "id": "tx_power",
            "op": "model.symbol_power_allocator",
            "inputs": {
                "symbols": "modulator.symbols",
                "channel_state": "channel_state.state",
            },
            "params": {
                "policy": "fixed",
                "granularity": "per_subcarrier",
                "budget_mode": "fixed_average",
                "target_power": 1.0,
            },
        }
        artifact = {
            "id": "learned-power-policy",
            "label": "Learned power policy",
            "manifest_path": "trained_artifacts/learned-power-policy/trained_artifact.yaml",
            "ready": True,
            "compatible_operations": [
                {
                    "operation": "model.symbol_power_allocator",
                    "required_inputs": ["channel_state"],
                    "params": {
                        "policy": "learned_artifact",
                        "artifact_manifest_path": "trained_artifacts/learned-power-policy/trained_artifact.yaml",
                        "artifact_entrypoint": "power_policy",
                    },
                }
            ],
        }
        checkpoint_artifact = {
            "id": "learned-power-checkpoint",
            "label": "Learned NPZ power policy",
            "manifest_path": "trained_artifacts/learned-power-checkpoint/trained_artifact.yaml",
            "ready": True,
            "compatible_operations": [
                {
                    "operation": "model.symbol_power_allocator",
                    "required_inputs": ["channel_state"],
                    "params": {
                        "policy": "learned_checkpoint",
                        "checkpoint_path": "trained_artifacts/learned-power-checkpoint/policy.npz",
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_format": "noema_csi_power_deepset_npz_v1",
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
              state.editRecipe = {{ name: "resource", metadata: {{}}, steps: [${{JSON.stringify({json.dumps(step)})}}] }};
              state.trainedArtifacts = ${{JSON.stringify({json.dumps([artifact, checkpoint_artifact])})}};
              const allocator = state.editRecipe.steps[0];
              const fixedMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              allocator.params.policy = "snr_sigmoid";
              const sigmoidMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              allocator.params.policy = "water_filling";
              const waterMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              allocator.params.policy = "learned_artifact";
              const learnedMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              state.trainedArtifactImportError = "Choose a safe NPZ checkpoint or a portable ZIP artifact package.";
              state.trainedArtifactImportErrorStepId = allocator.id;
              const importErrorMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              state.trainedArtifactImportError = "";
              state.trainedArtifactImportErrorStepId = "";
              state.trainedArtifacts = [];
              const learnedWithoutArtifactsMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              state.trainedArtifacts = ${{JSON.stringify({json.dumps([artifact, checkpoint_artifact])})}};
              allocator.params.policy = "learned_checkpoint";
              const legacyLearnedMarkup = schemaDrivenStepMarkup(allocator, 0, {{ recipe: state.editRecipe, open: true }});
              allocator.params.policy = "learned_artifact";
              allocator.params.granularity = "global";
              allocator.params.budget_mode = "variable_average";
              applySchemaOperationParamImplications(allocator, "policy", "learned_artifact");
              const impliedParams = {{ ...allocator.params }};
              allocator.params.artifact_manifest_path = "trained_artifact.yaml";
              allocator.params.artifact_entrypoint = "power_policy";
              allocator.params.policy = "fixed";
              pruneInactiveSchemaManagedArtifactParams(allocator, operationById(allocator.op).params_schema);
              globalThis.__result = {{
                fixedMarkup,
                sigmoidMarkup,
                waterMarkup,
                learnedMarkup,
                importErrorMarkup,
                learnedWithoutArtifactsMarkup,
                legacyLearnedMarkup,
                impliedParams,
                prunedParams: allocator.params,
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

        self.assertIn("Policy", result["fixedMarkup"])
        self.assertIn("Equal power", result["fixedMarkup"])
        for markup in (result["fixedMarkup"], result["sigmoidMarkup"], result["waterMarkup"]):
            self.assertIn("data-schema-trained-artifact", markup)
            self.assertIn('class="trained-artifact-picker managed-checkpoint-picker disabled"', markup)
            self.assertIn("Select Learned model under Policy", markup)
        self.assertNotIn(
            '<div class="field-hint">Choose a registered artifact or use the folder button',
            result["fixedMarkup"],
        )
        policy_index = result["learnedMarkup"].index('data-schema-param-name="policy"')
        model_batch_index = result["learnedMarkup"].index('data-schema-param-name="model_batch_size"')
        artifact_index = result["learnedMarkup"].index("data-schema-trained-artifact")
        granularity_index = result["learnedMarkup"].index('data-schema-param-name="granularity"')
        self.assertLess(policy_index, model_batch_index)
        self.assertLess(model_batch_index, artifact_index)
        self.assertLess(artifact_index, granularity_index)
        self.assertIn('class="field-grid compact schema-primary-parameters with-artifact"', result["learnedMarkup"])
        learned_select = result["learnedMarkup"].split('data-schema-trained-artifact="tx_power"', 1)[1].split(">", 1)[0]
        self.assertNotIn("disabled", learned_select)
        for markup in (result["fixedMarkup"], result["sigmoidMarkup"], result["waterMarkup"]):
            model_batch_control = markup.split(
                'data-schema-param-name="model_batch_size"', 1
            )[1].split(">", 1)[0]
            self.assertIn("disabled", model_batch_control)
            self.assertIn('value="1024"', markup)
        for markup in (result["learnedMarkup"], result["legacyLearnedMarkup"]):
            model_batch_control = markup.split(
                'data-schema-param-name="model_batch_size"', 1
            )[1].split(">", 1)[0]
            self.assertNotIn("disabled", model_batch_control)
        self.assertNotIn("Choose a safe NPZ checkpoint", result["importErrorMarkup"])
        empty_learned_select = result["learnedWithoutArtifactsMarkup"].split(
            'data-schema-trained-artifact="tx_power"', 1
        )[1].split(">", 1)[0]
        self.assertNotIn("disabled", empty_learned_select)
        self.assertIn(">Learned model</option>", result["learnedMarkup"])
        self.assertNotIn("Learned model · NPZ checkpoint", result["learnedMarkup"])
        self.assertNotIn("Learned model · registered artifact", result["learnedMarkup"])
        for markup in (result["learnedMarkup"], result["legacyLearnedMarkup"]):
            self.assertIn("Learned power policy", markup)
            self.assertIn("Learned NPZ power policy", markup)
            self.assertIn(".npz", markup)
            self.assertIn(".zip", markup)
        self.assertEqual(result["impliedParams"]["granularity"], "per_subcarrier")
        self.assertEqual(result["impliedParams"]["budget_mode"], "fixed_average")
        self.assertEqual(result["prunedParams"]["policy"], "fixed")
        self.assertNotIn("artifact_manifest_path", result["prunedParams"])
        self.assertNotIn("artifact_entrypoint", result["prunedParams"])

        binder_source = app_js.read_text(encoding="utf-8")
        binder_source = binder_source[
            binder_source.index("function bindSchemaDrivenRecipeConfigurator()"):
            binder_source.index("function addSchemaExtensionParameter(")
        ]
        self.assertIn("const stepOperation = operationById(step.op) || {};", binder_source)
        self.assertNotIn("pruneInactiveSchemaManagedArtifactParams(step, operation.params_schema", binder_source)

    def test_both_deepjscc_interfaces_expose_the_same_managed_artifact_control(self):
        registry = build_registry()
        for operation_id in (
            "model.deepjscc_external_encode",
            "model.deepjscc_external_decode",
        ):
            operation = registry.get(operation_id).describe()
            properties = operation["params_schema"]["properties"]
            picker = properties["checkpoint_path"]["x-noema-ui"]
            self.assertEqual(picker["control"], "trained_artifact")
            self.assertEqual(picker["visible_when"], {"runtime": "learned_checkpoint"})
            self.assertIn(".npz", picker["accept"])

    def test_ui_imports_external_checkpoint_and_applies_derived_binding(self):
        app_js = (ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('ui.control !== "trained_artifact"', app_js)
        self.assertIn("data-schema-trained-artifact-open", app_js)
        self.assertIn('type="file" hidden data-schema-trained-artifact-file', app_js)
        self.assertIn("/api/trained-artifacts/import", app_js)
        self.assertIn('"Content-Type": "application/octet-stream"', app_js)
        self.assertIn("applyTrainedArtifactToStep(stepId, choice.key)", app_js)

    def test_ui_applies_grouped_artifact_bindings_as_one_recipe_edit(self):
        app_js = (ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        start = app_js.index("function applyTrainedArtifactToStep")
        end = app_js.index("async function importTrainedArtifactForStep", start)
        binding_function = app_js[start:end]

        self.assertIn('mode === "all_group_bindings"', app_js)
        self.assertIn("resolveTrainedArtifactApplication(choice, step)", app_js)
        self.assertIn("preferred_step_id", app_js)
        self.assertIn("binding_group", app_js)
        self.assertIn("applications.forEach((application)", app_js)
        self.assertNotIn("training_performed", binding_function)
        self.assertIn("applies together to", app_js)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the artifact-binding UI test")
    def test_deepjscc_artifact_is_directly_selectable_and_applied_atomically(self):
        recipe = load_recipe(ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml").to_dict()
        recipe.setdefault("metadata", {})["training_performed"] = False
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
            for operation_id in (
                "model.deepjscc_external_encode",
                "model.deepjscc_external_decode",
            )
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
        app_js = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
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
              state.operations = ${{JSON.stringify({json.dumps(operations)})}};
              state.editRecipe = ${{JSON.stringify({json.dumps(recipe)})}};
              state.trainedArtifacts = [${{JSON.stringify({json.dumps(artifact)})}}];
              globalThis.__commitCount = 0;
              globalThis.__notices = [];
              commitRecipeValueEdit = () => {{ globalThis.__commitCount += 1; }};
              notify = (message) => {{ globalThis.__notices.push(message); }};
              renderRecipeControls = () => {{}};
              const sender = findStep("sender");
              const senderOperation = operationById(sender.op);
              const markupBeforeRuntimeChange = schemaDrivenTrainedArtifactMarkup(
                sender,
                senderOperation.params_schema
              );
              const choice = compatibleTrainedArtifactChoices(sender)[0];
              applyTrainedArtifactToStep("sender", choice.key);
              globalThis.__result = {{
                markupBeforeRuntimeChange,
                sender: findStep("sender").params,
                receiver: findStep("receiver").params,
                metadata: state.editRecipe.metadata,
                commitCount: globalThis.__commitCount,
                notices: globalThis.__notices,
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

        self.assertIn('data-schema-trained-artifact="sender"', result["markupBeforeRuntimeChange"])
        self.assertIn("sender + receiver", result["markupBeforeRuntimeChange"])
        self.assertEqual(result["commitCount"], 1)
        self.assertEqual(result["sender"]["runtime"], "learned_artifact")
        self.assertEqual(result["sender"]["artifact_entrypoint"], "encoder")
        self.assertEqual(result["receiver"]["runtime"], "learned_artifact")
        self.assertEqual(result["receiver"]["artifact_entrypoint"], "decoder")
        self.assertEqual(result["sender"]["artifact_manifest_path"], manifest_path)
        self.assertEqual(result["receiver"]["artifact_manifest_path"], manifest_path)
        self.assertNotIn("checkpoint_path", result["sender"])
        self.assertNotIn("checkpoint_path", result["receiver"])
        bindings = result["metadata"]["trained_artifact_bindings"]
        self.assertEqual(set(bindings), {"sender", "receiver"})
        self.assertEqual(
            {binding["binding_group"] for binding in bindings.values()},
            {"deepjscc_sender_receiver"},
        )
        self.assertIn("training_performed", result["metadata"])
        self.assertFalse(result["metadata"]["training_performed"])
        self.assertEqual(result["notices"], ["Learned DeepJSCC demo applied to sender + receiver"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the artifact-binding UI test")
    def test_csi_picker_groups_references_gates_protocol_and_preserves_method_label(self):
        recipe = load_recipe(ROOT / "recipes" / "csi_feedback_sionna_train.yaml").to_dict()
        recipe.setdefault("metadata", {})["training_performed"] = True

        def artifact(artifact_id: str, label: str, source_origin: str, tx_antennas: int) -> dict:
            manifest_path = f"trained_artifacts/references/{artifact_id}/trained_artifact.yaml"
            return {
                "id": artifact_id,
                "label": label,
                "manifest_path": manifest_path,
                "ready": True,
                "source": {"origin": source_origin},
                "artifact": {"format": "noema_trained_artifact_manifest_v2", "sha256": "a" * 64},
                "application": {"mode": "all_group_bindings"},
                "runtime": {
                    "entrypoints": [
                        {
                            "id": "encoder",
                            "validated_signature": {
                                "inputs": [{"name": "csi_ri", "shape": ["batch", 2, tx_antennas, 32]}],
                                "outputs": [{"name": "feedback_code", "shape": ["batch", 32]}],
                            },
                        },
                        {
                            "id": "decoder",
                            "validated_signature": {
                                "inputs": [{"name": "feedback_code", "shape": ["batch", 32]}],
                                "outputs": [{"name": "csi_hat_ri", "shape": ["batch", 2, tx_antennas, 32]}],
                            },
                        },
                    ]
                },
                "training": {
                    "feedback_constraint": {
                        "mode": "uniform_quantized",
                        "feedback_bits_per_sample": 128,
                    }
                },
                "compatible_operations": [
                    {
                        "operation": "model.csi_feedback_encoder",
                        "binding_group": "csi_feedback_codec",
                        "role": "encoder",
                        "preferred_step_id": "feedback_encoder",
                        "runtime_entrypoint": "encoder",
                        "required_inputs": ["csi"],
                        "params": {
                            "runtime": "learned_artifact",
                            "feedback_dimension": 32,
                            "artifact_manifest_path": manifest_path,
                            "artifact_entrypoint": "encoder",
                        },
                    },
                    {
                        "operation": "model.csi_feedback_decoder",
                        "binding_group": "csi_feedback_codec",
                        "role": "decoder",
                        "preferred_step_id": "feedback_decoder",
                        "runtime_entrypoint": "decoder",
                        "required_inputs": ["received_code"],
                        "params": {
                            "runtime": "learned_artifact",
                            "feedback_dimension": 32,
                            "artifact_manifest_path": manifest_path,
                            "artifact_entrypoint": "decoder",
                        },
                    },
                ],
            }

        exact = artifact(
            "matched-klt",
            "Matched KLT/PCA · 128 bit",
            "noema_reference_baseline",
            8,
        )
        incompatible = artifact(
            "published-csinet",
            "Published CsiNet · COST2100",
            "published_reference_checkpoint",
            32,
        )
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in (
                "model.csi_feedback_encoder",
                "model.csi_feedback_decoder",
            )
        ]
        app_js = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
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
              state.operations = ${{JSON.stringify({json.dumps(operations)})}};
              state.editRecipe = ${{JSON.stringify({json.dumps(recipe)})}};
              state.trainedArtifacts = ${{JSON.stringify({json.dumps([exact, incompatible])})}};
              commitRecipeValueEdit = () => {{}};
              notify = () => {{}};
              renderRecipeControls = () => {{}};
              const encoder = findStep("feedback_encoder");
              const choices = compatibleTrainedArtifactChoices(encoder);
              const exactChoice = choices.find((choice) => choice.artifact.id === "matched-klt");
              const incompatibleChoice = choices.find((choice) => choice.artifact.id === "published-csinet");
              const markup = schemaDrivenTrainedArtifactMarkup(encoder, operationById(encoder.op).params_schema);
              applyTrainedArtifactToStep("feedback_encoder", exactChoice.key);
              const method = recipeLabelFields({{ recipe: state.editRecipe }}).get("csi_feedback_runtime");
              globalThis.__result = {{
                markup,
                exactReady: exactChoice.ready,
                incompatibleReady: incompatibleChoice.ready,
                incompatibleIssues: incompatibleChoice.compatibilityIssues,
                methodLabel: method.label,
                binding: state.editRecipe.metadata.trained_artifact_bindings.feedback_encoder,
                trainingPerformed: state.editRecipe.metadata.training_performed,
                trainingPerformedPresent: Object.prototype.hasOwnProperty.call(state.editRecipe.metadata, "training_performed"),
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
        self.assertTrue(result["exactReady"])
        self.assertFalse(result["incompatibleReady"])
        self.assertIn("requires 32 transmit antennas", "; ".join(result["incompatibleIssues"]))
        self.assertIn('<optgroup label="Reference baseline">', result["markup"])
        self.assertIn('<optgroup label="Published reference">', result["markup"])
        self.assertIn("Published CsiNet · COST2100 (incompatible:", result["markup"])
        self.assertEqual(result["methodLabel"], "Matched KLT/PCA · 128 bit")
        self.assertEqual(result["binding"]["label"], "Matched KLT/PCA · 128 bit")
        self.assertEqual(result["binding"]["source_class"], "reference_baseline")
        self.assertTrue(result["trainingPerformedPresent"])
        self.assertTrue(result["trainingPerformed"])

    def test_workbench_describes_neutral_contract_slots_and_optional_starter(self):
        app_js = (ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("Operation Training Capabilities", app_js)
        self.assertIn("Built-in fine-tuning", app_js)
        self.assertIn("Portable replacement", app_js)
        self.assertIn(">Train/replace</th>", app_js)
        self.assertIn("Differentiable support", app_js)
        self.assertIn("does not fine-tune", app_js)
        self.assertIn("Select which portable recipe blocks the separate training plan will replace", app_js)
        self.assertIn("data-differentiable-export-slot", app_js)
        self.assertIn("workbenchSelectableListMarkup", app_js)
        self.assertNotIn("Include demo starter", app_js)
        self.assertNotIn("Operation Differentiability", app_js)
        self.assertNotIn("Train/replace block", app_js)
        self.assertNotIn("Replaceable slot", app_js)
        self.assertNotIn("Select on Graph", app_js)
        self.assertNotIn("Trainable slots and return targets", app_js)
        self.assertNotIn("Model architecture, loss, optimizer, and trainer remain external choices", app_js)
        self.assertNotIn('starter === "deepjscc-image"', app_js)
        self.assertIn('return "torch"', app_js)
        self.assertNotIn("Model steps to train", app_js)
        self.assertNotIn("Optimizable model markers", app_js)


if __name__ == "__main__":
    unittest.main()
