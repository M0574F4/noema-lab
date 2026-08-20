# Return a Schema-v2 ONNX Artifact and Run an Ordinary Benchmark

This tutorial closes the contract-first DeepJSCC loop started in
[Export a Contract-First DeepJSCC Training Project](export_differentiable_training_scenario.md):

```text
ordinary Noema recipe
  -> neutral training contract
  -> external architecture, loss, optimizer, and trainer
  -> schema-v2 ONNX artifact package
  -> import and bind to the existing sender/receiver slots
  -> ordinary recipe run and benchmark
```

Noema does not need the researcher's model class or training code. The returned ONNX graphs carry the
architecture, while the exported contract and artifact manifest define the executable interface. A
model that cannot use the portable ONNX ABI must use an explicitly trusted adapter/plugin instead; it
must not be smuggled into the portable package as arbitrary Python.

## 1. Train outside Noema

The neutral export in `differentiable_exports/deepjscc_kodak_awgn/` gives you
`training_contract.yaml`, `scenario_graph.json`, `interfaces.py`, test vectors, and the generated
artifact template. Implement encoder and decoder modules against those interfaces, then choose the
architecture, objective, optimizer, trainer, and checkpoint-selection rule in your own training
project.

To exercise the handoff with the checked-in DeepJSCC demonstration, first prepare its external
training project as described in the contract-export tutorial. Then copy this block from any directory inside the
clone; it returns to the repository root, enters the bundle directory, validates the neutral
contract, then trains and evaluates through the bundle-root launchers:

```bash
cd "$(git rev-parse --show-toplevel)"
cd differentiable_exports/deepjscc_kodak_awgn
uv run --project ../.. --extra onnx python validate_contract.py
uv run --project ../.. --extra onnx python train_demo.py
uv run --project ../.. --extra onnx python evaluate_demo.py
cd ../..
```

The demonstration project is example code. Its reference CNN and MSE objective do not become
compatibility requirements. A successful run writes files with this shape:

```text
training_contract.yaml
trained_artifact.yaml
artifacts/
  encoder.onnx
  decoder.onnx
reference_training/
  training_history.json
  evaluation_metrics.json
```

If you train your own implementation, export ONNX components with the exact names, dtypes, layouts,
symbolic dimensions, and real/imaginary representation declared by the contract's runtime ABI. Run
`python validate_contract.py` from the neutral export root before packaging anything.

## 2. Create the portable artifact package

Start from `trained_artifact.template.yaml` and save the completed manifest as
`trained_artifact.yaml`. A portable paired DeepJSCC package contains only declared data and executable
graphs:

```text
trained_artifact.yaml
training_contract.yaml       unchanged from the export
artifacts/
  encoder.onnx
  decoder.onnx
```

The schema-v2 manifest has four important parts:

```yaml
schema_version: 2
kind: noema.trained_block_artifact
id: my_lab.deepjscc.kodak_v1
contract:
  id: <exported contract id>
  version: 1
  path: training_contract.yaml
  sha256: <canonical semantic hash>
  file_sha256: <exact file hash>
components:
  - id: encoder
    role: encoder
    path: artifacts/encoder.onnx
    format: onnx
    sha256: <full encoder file hash>
  - id: decoder
    role: decoder
    path: artifacts/decoder.onnx
    format: onnx
    sha256: <full decoder file hash>
runtime:
  backend: onnxruntime
  abi_version: 1
  entrypoints: <keep the generated encoder and decoder tensor ABIs>
application:
  mode: all_group_bindings
compatible_operations:
  - operation: model.deepjscc_external_encode
    runtime_entrypoint: encoder
    binding_group: deepjscc_sender_receiver
    role: encoder
    preferred_step_id: sender
  - operation: model.deepjscc_external_decode
    runtime_entrypoint: decoder
    binding_group: deepjscc_sender_receiver
    role: decoder
    preferred_step_id: receiver
```

Keep the generated entrypoints, required inputs, binding parameters, and contract hashes intact.
Replace the component files, component hashes, artifact ID, name, and optional training provenance.
Architecture, loss, optimizer, and trainer may be recorded under `training`, but Noema does not use
those descriptive fields as compatibility gates.

For Workbench import, put the manifest, unchanged contract, and declared ONNX files in a ZIP archive;
the `.noema-artifact` suffix is also accepted. From any directory inside the clone, copy this block;
it returns to the repository root and enters the bundle directory before creating the package:

```bash
cd "$(git rev-parse --show-toplevel)"
cd differentiable_exports/deepjscc_kodak_awgn
zip -r deepjscc_kodak_v1.noema-artifact \
  trained_artifact.yaml training_contract.yaml artifacts/encoder.onnx artifacts/decoder.onnx
cd ../..
```

`package_artifact.py` is not the portable-package validator. It only creates a validation request for
a runtime that needs a separate adapter. The Workbench importer performs the schema-v2 ONNX validation
described below.

## 3. Import and bind the artifact in Workbench

Open the ordinary DeepJSCC recipe, or a duplicate you want to benchmark. Then:

1. Open **Graph** and select the **Encoder** Block (technical step ID `sender`).
2. In **Recipe**, find **Model artifact**.
3. Select the folder button beside the artifact selector.
4. Choose the ZIP or `.noema-artifact` package.

Noema validates the package before registering it. Validation covers:

- safe archive paths and declared-file-only import;
- the contract's semantic SHA-256 and exact file SHA-256;
- every ONNX component's full file SHA-256;
- ONNX signatures against the declared tensor ABI;
- operation-owned component roles, entrypoints, and required inputs;
- compatibility with the selected recipe operation and installed runtime.

Only declared files are copied, atomically, under `.noema/trained_artifacts/imported/`. External Python
files are neither imported nor executed. After a successful import, Workbench applies the artifact to
the current working recipe. Because DeepJSCC uses `all_group_bindings`, selecting or importing it at
the sender also applies the matching decoder binding; unrelated encoder and decoder artifacts cannot
be mixed accidentally.

The effective parameters have this form, with Noema supplying the managed paths and entrypoints:

```yaml
# sender
runtime: learned_artifact
artifact_manifest_path: .noema/trained_artifacts/imported/<managed-id>/trained_artifact.yaml
artifact_entrypoint: encoder

# receiver
runtime: learned_artifact
artifact_manifest_path: .noema/trained_artifacts/imported/<managed-id>/trained_artifact.yaml
artifact_entrypoint: decoder
```

If an artifact produced by the demonstration project is already discovered through the exported
project, you can choose it directly from the same **Model artifact** selector instead of importing it again.
The source recipe and original recipe file are never modified; only the open working recipe changes.

## 4. Run the ordinary recipe first

Save the working recipe under a meaningful method name and run it through the normal recipe controls.
This is a runtime and contract smoke test, not the final comparison. Verify that:

- both sender and receiver show the same managed artifact;
- the reconstruction output is present;
- the configured channel-use and image-quality metrics are present;
- the run manifest records the artifact manifest, ONNX component hashes, and recipe provenance.

Power normalization, TX/RX symbol boundaries, AWGN, symbol-count matching, and image metrics remain in
the recipe. The learned artifact replaces only the declared sender and receiver slots. There is no
DeepJSCC-specific run action or generated benchmark.

## 5. Compare under an ordinary benchmark pack

Add the returned-model recipe and whichever baselines you want to an ordinary benchmark pack. Keep
the source images, SNR grid, channel seed policy, power normalization, and channel-use accounting fixed
across methods. From any directory inside the clone, copy this block; it returns to the repository
root before using the normal benchmark CLI:

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/noema benchmark validate benchmarks/<suite>/<pack>.yaml
.venv/bin/noema benchmark run benchmarks/<suite>/<pack>.yaml
```

For image DeepJSCC, the central comparison normally includes reconstruction quality versus channel
conditions and bandwidth use, such as PSNR, MS-SSIM, or LPIPS versus SNR at a fixed
channel-uses-per-pixel budget. A training-time validation loss is useful for model selection; it is not
the benchmark conclusion.

## 6. Inspect reproducibility evidence

Use the ordinary Results and Benchmark views. To inspect a saved bundle from the CLI, copy this block
from any directory inside the clone; it returns to the repository root first:

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/noema benchmark verify <result-id>
.venv/bin/noema benchmark result <result-id>
```

For a paper or demo report, retain:

- benchmark pack ID and version;
- returned-model recipe SHA-256;
- trained-artifact manifest and contract IDs;
- ONNX component SHA-256 values and runtime evidence;
- training provenance you chose to report;
- image set, SNR grid, seeds, power constraint, and channel-use accounting;
- plotted data and result-bundle path.

That closes the loop without making Noema the trainer: Noema exports the scenario and interface
contracts, the researcher chooses and trains the model, the portable artifact returns through the
declared ABI, and the ordinary benchmark machinery evaluates it alongside every other method.
