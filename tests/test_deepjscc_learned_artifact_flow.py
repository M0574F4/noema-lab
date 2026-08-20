from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from noema_lab.core.artifacts import Artifact
from noema_lab.core.operations import OperationContext, OperationError
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.trained_artifacts import inspect_trained_artifact
from noema_lab.ops import build_registry
from noema_lab.ops.models.external import (
    DeepJsccExternalDecodeOperation,
    DeepJsccExternalEncodeOperation,
)
from noema_lab.ui.server import start_ui_server_in_thread


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src" / "noema_lab" / "ui" / "static" / "app.js"
RECIPE_PATH = ROOT / "recipes" / "deepjscc_kodak_awgn_train.yaml"
ONNX_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


class DeepJsccLearnedArtifactFlowTests(unittest.TestCase):
    def test_portable_pair_picker_excludes_individual_onnx_and_bare_manifests(self):
        registry = build_registry()
        for operation_id in (
            "model.deepjscc_external_encode",
            "model.deepjscc_external_decode",
        ):
            properties = registry.get(operation_id).describe()["params_schema"]["properties"]
            accept = properties["artifact_manifest_path"]["x-noema-ui"]["accept"]
            self.assertIn(".zip", accept)
            self.assertIn(".noema-artifact", accept)
            self.assertNotIn(".onnx", accept)
            self.assertNotIn(".yaml", accept)
            self.assertNotIn(".json", accept)

    def test_import_api_rejects_individual_deepjscc_onnx_with_pair_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, thread = start_ui_server_in_thread(
                "127.0.0.1",
                0,
                root / ".workspace",
                root,
            )
            try:
                host, port = server.server_address
                query = urllib.parse.urlencode(
                    {
                        "operation": "model.deepjscc_external_encode",
                        "filename": "encoder.onnx",
                    }
                )
                request = urllib.request.Request(
                    "http://%s:%d/api/trained-artifacts/import?%s"
                    % (host, port, query),
                    data=b"not-a-complete-paired-artifact",
                    method="POST",
                    headers={"Content-Type": "application/octet-stream"},
                )
                with self.assertRaises(urllib.error.HTTPError) as captured:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(captured.exception.code, 400)
                payload = json.loads(captured.exception.read().decode("utf-8"))
                message = str(payload.get("error") or payload)
                self.assertIn("not a complete artifact", message)
                self.assertIn("both encoder and decoder", message)
                self.assertFalse((root / ".noema" / "trained_artifacts").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the import UX test")
    def test_picker_explains_pair_package_and_rejects_encoder_file_locally(self):
        registry = build_registry()
        operation = registry.get("model.deepjscc_external_encode").describe()
        recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
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
              window: {{ CSS: null, setTimeout, clearTimeout }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              state.operations = [${{JSON.stringify({json.dumps(operation)})}}];
              state.editRecipe = ${{JSON.stringify({json.dumps(recipe)})}};
              state.trainedArtifacts = [];
              renderRecipeControls = () => {{}};
              const sender = findStep("sender");
              sender.params.runtime = "learned_artifact";
              const markup = schemaDrivenTrainedArtifactMarkup(sender, operationById(sender.op).params_schema);
              importTrainedArtifactForStep("sender", {{ name: "encoder.onnx" }});
              globalThis.__result = {{
                markup,
                error: state.trainedArtifactImportError,
                errorStep: state.trainedArtifactImportErrorStepId,
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
        self.assertIn("Individual ONNX files are not complete DeepJSCC artifacts", result["markup"])
        self.assertIn('accept=".zip,.noema-artifact,application/zip,application/octet-stream"', result["markup"])
        self.assertIn("cannot be imported by itself", result["error"])
        self.assertIn("both ONNX components", result["error"])
        self.assertEqual(result["errorStep"], "sender")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI regression test")
    def test_runtime_change_reveals_picker_and_selection_applies_pair_atomically(self):
        registry = build_registry()
        operations = [
            registry.get(operation_id).describe()
            for operation_id in (
                "model.deepjscc_external_encode",
                "model.deepjscc_external_decode",
            )
        ]
        recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
        artifact = {
            "id": "test.deepjscc.pair",
            "package_sha256": "b" * 64,
            "label": "Test DeepJSCC pair",
            "manifest_path": "differentiable_exports/test/trained_artifact.yaml",
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
                        "artifact_manifest_path": "differentiable_exports/test/trained_artifact.yaml",
                        "artifact_entrypoint": "encoder",
                        "artifact_package_sha256": "b" * 64,
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
                        "artifact_manifest_path": "differentiable_exports/test/trained_artifact.yaml",
                        "artifact_entrypoint": "decoder",
                        "artifact_package_sha256": "b" * 64,
                    },
                },
            ],
        }
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
              window: {{ CSS: null, setTimeout, clearTimeout }},
              setTimeout,
              clearTimeout,
            }};
            sandbox.globalThis = sandbox;
            vm.createContext(sandbox);
            vm.runInContext(source + `
              state.operations = ${{JSON.stringify({json.dumps(operations)})}};
              state.editRecipe = ${{JSON.stringify({json.dumps(recipe)})}};
              state.trainedArtifacts = ${{JSON.stringify({json.dumps([artifact])})}};
              let commits = 0;
              let notice = "";
              applyBuiltRecipe = () => {{ commits += 1; }};
              renderRecipeControls = () => {{}};
              removeRecipeSweep = () => {{}};
              notify = (message) => {{ notice = String(message); }};

              const sender = findStep("sender");
              const receiver = findStep("receiver");
              const senderSchema = operationById(sender.op).params_schema;
              const before = schemaDrivenTrainedArtifactMarkup(sender, senderSchema);

              // This mirrors the schema parameter change handler before the artifact is chosen.
              sender.params.runtime = "learned_artifact";
              reconcileTrainedArtifactBindingMetadata(sender);
              commitRecipeValueEdit(true);
              const afterRuntimeChange = schemaDrivenTrainedArtifactMarkup(sender, senderSchema);
              const choice = compatibleTrainedArtifactChoices(sender)[0];
              applyTrainedArtifactToStep("sender", choice.key);

              globalThis.__result = {{
                before,
                afterRuntimeChange,
                senderParams: sender.params,
                receiverParams: receiver.params,
                bindingKeys: Object.keys(state.editRecipe.metadata.trained_artifact_bindings || {{}}).sort(),
                trainingPerformedPresent: Object.prototype.hasOwnProperty.call(
                  state.editRecipe.metadata,
                  "training_performed",
                ),
                commits,
                notice,
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

        # A discovered paired artifact is offered directly even while the slot is still
        # in its export-only training-interface mode. Changing the runtime first must
        # keep that picker available rather than losing the compatible choice.
        self.assertIn("Test DeepJSCC pair", result["before"])
        self.assertIn("sender + receiver", result["before"])
        self.assertIn("Test DeepJSCC pair", result["afterRuntimeChange"])
        self.assertIn("sender + receiver", result["afterRuntimeChange"])
        self.assertEqual(
            result["senderParams"],
            {
                "runtime": "learned_artifact",
                "artifact_manifest_path": "differentiable_exports/test/trained_artifact.yaml",
                "artifact_entrypoint": "encoder",
                "artifact_package_sha256": artifact["package_sha256"],
            },
        )
        self.assertEqual(
            result["receiverParams"],
            {
                "runtime": "learned_artifact",
                "artifact_manifest_path": "differentiable_exports/test/trained_artifact.yaml",
                "artifact_entrypoint": "decoder",
                "artifact_package_sha256": artifact["package_sha256"],
            },
        )
        self.assertEqual(result["bindingKeys"], ["receiver", "sender"])
        # Selecting a returned artifact is evidence of binding, not proof that
        # this UI session performed the training.  The legacy flag must not be
        # synthesized from artifact selection.
        self.assertFalse(result["trainingPerformedPresent"])
        self.assertEqual(result["commits"], 2)
        self.assertIn("applied to sender + receiver", result["notice"])

    @unittest.skipUnless(
        ONNX_RUNTIME_AVAILABLE,
        "ONNX and ONNX Runtime are required for the learned-artifact codec test",
    )
    def test_learned_artifact_operations_execute_paired_onnx_codec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = _write_round_trip_package(root / "returned-codec")
            inspected = inspect_trained_artifact(
                manifest_path,
                project_root=root,
                registry=build_registry(),
            )
            self.assertTrue(inspected["ready"], inspected["issues"])
            bindings = {
                binding["role"]: binding
                for binding in inspected["compatible_operations"]
            }

            images = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
            image_path = root / "images.npz"
            image_metadata = {"shape": list(images.shape), "dataset": "round_trip"}
            np.savez_compressed(
                image_path,
                images=images,
                metadata_json=json.dumps(image_metadata),
            )
            encoder_params = {
                **bindings["encoder"]["params"],
                "artifact_manifest_path": str(manifest_path),
            }
            encoded = DeepJsccExternalEncodeOperation().run(
                OperationContext(
                    recipe_name="deepjscc_learned_artifact_round_trip",
                    step_id="sender",
                    params=encoder_params,
                    inputs={
                        "images": Artifact(
                            "image.batch.numpy",
                            image_path,
                            image_metadata,
                        )
                    },
                    run_dir=root,
                    step_dir=root / "sender",
                )
            )
            symbols = encoded.outputs["symbols"]
            self.assertEqual(symbols.metadata["adapter"], "learned_artifact")
            self.assertEqual(symbols.metadata["symbol_shape"], [2, 3, 4, 5])
            self.assertEqual(symbols.metadata["image_shape"], list(images.shape))

            decoder_params = {
                **bindings["decoder"]["params"],
                "artifact_manifest_path": str(manifest_path),
            }
            decoded = DeepJsccExternalDecodeOperation().run(
                OperationContext(
                    recipe_name="deepjscc_learned_artifact_round_trip",
                    step_id="receiver",
                    params=decoder_params,
                    inputs={"symbols": symbols},
                    run_dir=root,
                    step_dir=root / "receiver",
                )
            )
            with np.load(decoded.outputs["images"].path, allow_pickle=False) as payload:
                reconstruction = np.asarray(payload["images"])
            np.testing.assert_array_equal(reconstruction, images)
            self.assertEqual(
                decoded.outputs["images"].metadata["adapter"],
                "learned_artifact",
            )

            missing_manifest = dict(encoder_params)
            missing_manifest["artifact_manifest_path"] = ""
            with self.assertRaisesRegex(
                OperationError,
                "runtime=learned_artifact requires artifact_manifest_path",
            ):
                DeepJsccExternalEncodeOperation().run(
                    OperationContext(
                        recipe_name="deepjscc_missing_artifact",
                        step_id="sender",
                        params=missing_manifest,
                        inputs={
                            "images": Artifact(
                                "image.batch.numpy",
                                image_path,
                                image_metadata,
                            )
                        },
                        run_dir=root,
                        step_dir=root / "missing_sender",
                    )
                )


def _write_round_trip_package(package: Path) -> Path:
    package.mkdir(parents=True, exist_ok=True)
    contract_path = package / "slot_contract.yaml"
    contract = {
        "schema_version": 1,
        "kind": "noema.trainable_slot_contract@1",
        "id": "noema.slot.deepjscc-round-trip",
        "version": 1,
    }
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    encoder_path = package / "encoder.onnx"
    decoder_path = package / "decoder.onnx"
    _write_codec_onnx(encoder_path, decoder_path)

    image_shape = ["batch", 3, "height", "width"]
    symbol_shape = ["batch", "real_imag_channel", "symbol_height", "symbol_width"]

    def tensor(name: str, semantic: str, shape: list) -> dict:
        return {
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "semantic": semantic,
        }

    manifest = {
        "schema_version": 2,
        "kind": "noema.trained_block_artifact",
        "id": "test.deepjscc-round-trip",
        "name": "Round-trip DeepJSCC codec",
        "contract": {
            "id": contract["id"],
            "version": 1,
            "path": contract_path.name,
            "sha256": canonical_json_sha256(contract),
            "file_sha256": _sha256(contract_path),
        },
        "components": [
            {
                "id": "encoder",
                "role": "encoder",
                "path": encoder_path.name,
                "format": "onnx",
                "sha256": _sha256(encoder_path),
            },
            {
                "id": "decoder",
                "role": "decoder",
                "path": decoder_path.name,
                "format": "onnx",
                "sha256": _sha256(decoder_path),
            },
        ],
        "runtime": {
            "backend": "onnxruntime",
            "abi_version": 1,
            "entrypoints": [
                {
                    "id": "encoder",
                    "component": "encoder",
                    "inputs": [tensor("images", "normalized image tensor", image_shape)],
                    "outputs": [
                        tensor(
                            "symbols_ri",
                            "real/imaginary channel symbols",
                            symbol_shape,
                        )
                    ],
                },
                {
                    "id": "decoder",
                    "component": "decoder",
                    "inputs": [
                        tensor(
                            "symbols_ri",
                            "real/imaginary received symbols",
                            symbol_shape,
                        )
                    ],
                    "outputs": [
                        tensor(
                            "reconstruction",
                            "normalized image reconstruction",
                            image_shape,
                        )
                    ],
                },
            ],
        },
        "application": {"mode": "all_group_bindings"},
        "compatible_operations": [
            {
                "operation": "model.deepjscc_external_encode",
                "runtime_entrypoint": "encoder",
                "binding_group": "deepjscc_sender_receiver",
                "role": "encoder",
                "preferred_step_id": "sender",
                "required_inputs": ["images"],
                "params": {},
            },
            {
                "operation": "model.deepjscc_external_decode",
                "runtime_entrypoint": "decoder",
                "binding_group": "deepjscc_sender_receiver",
                "role": "decoder",
                "preferred_step_id": "receiver",
                "required_inputs": ["symbols"],
                "params": {},
            },
        ],
    }
    manifest_path = package / "trained_artifact.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_codec_onnx(encoder_path: Path, decoder_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception as exc:  # pragma: no cover - optional dependency gate
        raise unittest.SkipTest("onnx is unavailable: %s" % exc) from exc

    encoder_graph = helper.make_graph(
        [helper.make_node("Concat", ["images", "images"], ["symbols_ri"], axis=1)],
        "deepjscc_round_trip_encoder",
        [
            helper.make_tensor_value_info(
                "images",
                TensorProto.FLOAT,
                ["batch", 3, "height", "width"],
            )
        ],
        [
            helper.make_tensor_value_info(
                "symbols_ri",
                TensorProto.FLOAT,
                ["batch", "real_imag_channel", "symbol_height", "symbol_width"],
            )
        ],
    )
    decoder_graph = helper.make_graph(
        [
            helper.make_node(
                "Split",
                ["symbols_ri"],
                ["reconstruction", "unused_imaginary"],
                axis=1,
            )
        ],
        "deepjscc_round_trip_decoder",
        [
            helper.make_tensor_value_info(
                "symbols_ri",
                TensorProto.FLOAT,
                ["batch", "real_imag_channel", "symbol_height", "symbol_width"],
            )
        ],
        [
            helper.make_tensor_value_info(
                "reconstruction",
                TensorProto.FLOAT,
                ["batch", 3, "height", "width"],
            )
        ],
    )
    encoder = helper.make_model(
        encoder_graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    decoder = helper.make_model(
        decoder_graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(encoder)
    onnx.checker.check_model(decoder)
    onnx.save_model(encoder, str(encoder_path))
    onnx.save_model(decoder, str(decoder_path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
