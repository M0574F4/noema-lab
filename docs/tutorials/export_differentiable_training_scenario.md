# Export a Contract-First DeepJSCC Training Project

This tutorial exports a DeepJSCC training contract from a normal Noema communication recipe. The
result defines portable replacement interfaces, the required recipe-derived frozen channel path, data requirements, and
artifact return ABI. A researcher supplies and trains the model outside Noema.

The export does **not** copy an AI model out of the canonical recipe. The recipe's
`model.deepjscc_external_encode` and `model.deepjscc_external_decode` steps are typed sender and
receiver interfaces. They tell the contract compiler where external encoder and decoder modules connect and
where their trained artifact must return. They do not select or contain a neural architecture.

The closed loop is:

```text
ordinary recipe
  -> typed sender/receiver replacement interfaces + recipe channel constraints
  -> neutral slot/scenario/data/artifact contracts
  -> external architecture, loss, optimizer, and trainer
  -> ABI-conformant paired encoder/decoder artifact
  -> the same operation slots in an ordinary recipe
  -> ordinary benchmark
```

## CLI training summary

Use a fresh `BUNDLE` path for this all-in-one route; the detailed Workbench walkthrough below is an
alternative, not an additional sequence to run.
This example is file-backed, so its CLI route has no `dataset-capture` command or capture-progress
display.

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/differentiable_exports/deepjscc_kodak_awgn"

  cd "$ROOT"
  # This exact reproducibility block requires a Noema source checkout.
  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema data fetch kodak \
    --directory "$ROOT/.noema/datasets/kodak"
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/deepjscc_kodak_awgn_train.yaml" \
    --training-plan "$ROOT/demo_trainings/deepjscc_image_reconstruction/training_plan.yaml" \
    --out "$BUNDLE"
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" deepjscc-image "$BUNDLE" \
    --project-root "$ROOT"

  cd "$BUNDLE"
  uv run --project "$ROOT" --extra onnx python validate_contract.py
  uv run --project "$ROOT" --extra onnx python train_demo.py
  uv run --project "$ROOT" --extra onnx python evaluate_demo.py
)
```

## 1. Inspect the recipe path

The scenario recipe is
[recipes/deepjscc_kodak_awgn_train.yaml](../../recipes/deepjscc_kodak_awgn_train.yaml):

```text
Kodak image
  -> sender interface
  -> average-power normalization
  -> TX symbol boundary
  -> AWGN channel
  -> RX symbol boundary and count check
  -> receiver interface
  -> image reconstruction metrics
```

The relevant endpoint declarations contain contracts, not model layers:

```yaml
- id: sender
  op: model.deepjscc_external_encode
  inputs:
    images: data.images

- id: receiver
  op: model.deepjscc_external_decode
  inputs:
    symbols: rx_symbol_boundary.symbols
```

Inspect the path from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/noema differentiable inspect recipes/deepjscc_kodak_awgn_train.yaml \
  --replacement sender,receiver \
  --loss evaluation
```

The report should show `Status: full_gradient_possible`,
`Recommendation: differentiable_export`, and
`runner support: benchmark_run=yes, dataset_capture=yes, differentiable_export=yes`. Its gradient path
runs from `sender` through power normalization, the symbol boundaries, and `wireless_channel` to
`receiver` and the loss, with no gradient break.

In **Operation Training Capabilities**, the visible Blocks are **Encoder** and **Decoder**. Their
recipe step IDs are `sender` and `receiver`, which is why those values appear in `--replacement` and
the report; their current operation IDs are `model.deepjscc_external_encode` and
`model.deepjscc_external_decode`. The two Blocks are selected replacement interfaces and
artifact-return targets. Their current operations' own gradient/exportability is irrelevant because
researcher-supplied modules take their place.

This is a live task-loss export, so frozen downstream support after a selected replacement output
must remain differentiable and materializable until the loss. Power normalization, symbol boundaries,
and AWGN therefore remain in the forward pass but are not optimized. The original selected operations
are not counted as frozen support.

For a branched recipe `A -> B -> {C1, C2} -> D -> loss`, replacing **B** omits upstream **A**
and B's current implementation from backpropagation/export. A live task loss includes both **C1**
and **C2**, plus **D**—not just one shortest branch. Capture-backed local or supervised training
exports none of C1/C2/D because it trains from frozen input/target records instead.

## 2. Export the training bundle

Use the Workbench controls in their displayed order:

1. In **Operation Training Capabilities**, choose the **Encoder** and **Decoder** Blocks with
   **Train/replace**.
2. In **Training bundle**, set **Bundle directory** to
   `differentiable_exports/deepjscc_kodak_awgn`.
3. Review **Dataset definition**. This file-backed example resolves the recipe-selected Kodak source
   files, so it does not need the **Captured signals** and **Dataset size and splits** editors used by
   capture-backed bundles.
4. Select **PyTorch** under **Support framework**. Enable **Overwrite generated files** only when
   intentionally replacing a prior bundle.
5. Select **Export training bundle**.

Workbench automatically analyzes all reachable evaluation sinks; it does not present them as
training choices or select a model, objective, optimizer, or trainer. Export freezes the replacement
ABI, selected data, deterministic partitions, paths, and file hashes before external training reads
them. In a capture-backed workflow, the same boundary also freezes the selected tensors, split
recipes, seeds, managed paths, and capture hashes before **Dataset capture** can materialize the
managed train, validation, and held-out-test splits.

From any directory inside the repository clone, copy this block. It returns to the repository root
before performing the equivalent CLI export:

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/noema differentiable export recipes/deepjscc_kodak_awgn_train.yaml \
  --training-plan demo_trainings/deepjscc_image_reconstruction/training_plan.yaml \
  --out differentiable_exports/deepjscc_kodak_awgn
```

Noema derives the reachable endpoint (`evaluation`) and records it under the schema field
`loss_steps` in `training_plan.yaml`. Advanced CLI users can apply `--route-loss evaluation` to
intentionally narrow a live differentiable export; the Workbench UI always uses automatic route
analysis.

If inspection chooses **Capture-backed training**, the chosen `loss_steps` value remains in
`training_plan.yaml` as the analyzed recipe boundary, while `live_route_loss_steps` is empty and the
typed scenario omits all downstream backpropagation support. For a custom array-valued replacement,
Workbench derives its incoming model tensors as required capture signals and lets you add optional
targets/auxiliary outputs before choosing the train/validation/held-out-test split. Export writes the
generic data contract and three runnable capture recipes whose outputs are managed under
`<bundle>/data/train`, `<bundle>/data/validation`, and `<bundle>/data/test`; it does not require a
hard-coded demo.
Those generated recipes are capture-only projections, so their execution profile is `custom` with
the complete scenario profile recorded as `based_on`. This does not change the source recipe or its
standard-profile validation.

The initial closed DeepJSCC demonstration intentionally uses one native PyTorch autograd graph, so
use `--framework torch` for this specialized export. Noema rejects `torch-sionna` for this particular
reference-project exporter because that generated trainer has only been qualified with its native
Torch channel blocks. The generic typed graph separately registers Sionna 2.x/PyTorch AWGN and
flat-fading differentiable materializations for supported recipes. Both implementations carry the
same explicit `receiver_processing` selector as benchmark execution; this tutorial does not imply
support for unregistered Sionna operations.

A neutral contract exported with `--framework torch-sionna` keeps that combined framework label for
researcher-owned slots and selects the concrete `sionna` backend for supported frozen PHY nodes in
`scenario_graph.json`. It does not substitute the native-Torch channel materialization. This neutral
contract path is distinct from requesting the optional packaged DeepJSCC starter described above.

The export contains:

```text
differentiable_exports/deepjscc_kodak_awgn/
  training_contract.yaml
  scenario_graph.json
  data_contract.yaml
  trained_artifact.template.yaml
  interfaces.py
  validate_contract.py
  package_artifact.py
  project_manifest.yaml
  noema_recipe.yaml
  README.md
```

The files have distinct ownership:

- `training_contract.yaml` defines encoder/decoder inputs, outputs, layouts, symbolic shapes, and atomic
  return binding.
- `scenario_graph.json` records the minimal derived training DAG: selected replacement placeholders,
  every required inter-placeholder or downstream branch, and external typed boundaries for omitted
  producers. The full recipe remains in `noema_recipe.yaml`; only unchanged nodes on the external loss
  route are instantiated during training.
- `trained_artifact.template.yaml` defines the executable ONNX return ABI and conformance bindings
  without naming a training loss or architecture.
- `data_contract.yaml` resolves the recipe's selected image IDs into explicit `train` and
  `validation` file lists. It pins every source file by SHA-256 and records the exact preprocessing
  contract an external trainer must follow.
- `noema_recipe.yaml` is the unchanged recipe snapshot used to create the project.
- `project_manifest.yaml` lets Workbench discover the project and its returned trained artifact.

## 3. Verify the differentiable path

The important forward path implemented by an external DeepJSCC training scenario is conceptually:

```python
symbols_before_power = encoder(images)
symbols = power_normalize(symbols_before_power)
rx_symbols = awgn(symbols, snr_db)
reconstruction = decoder(rx_symbols)
```

The researcher-supplied encoder and decoder own trainable parameters. The selected recipe operations
are placeholders and are not exported. Power normalization and AWGN are frozen modules,
but both are implemented with PyTorch tensor operations, so an externally defined loss can reach the
encoder. The contract exposes source and reconstruction tensors needed by candidate losses; it does
not choose one. Typed symbol boundaries and the count-match assertion remain in the ordinary benchmark
recipe; they do not invent trainable parameters.

This is the differentiable-module feature demonstrated by the export: recipe-defined fixed modules
are materialized into the training graph around externally supplied modules. It is different from the
resource-allocation example, which trains from captured CSI and does not require gradients through a
channel module.

### Auditable image partitions

The image source is not rediscovered or randomly partitioned by the external trainer. At export time,
Noema resolves the ordered image IDs selected by the recipe, verifies that each file exists, records
its resolved path, byte count, and SHA-256, and materializes a deterministic partition. The last
`max(1, floor(N/5))` selected images are validation; all preceding images are training. The generated
demonstration trainer consumes those exact lists and fails if the contract or any selected image changes.

The checked-in Kodak demonstration selects `kodim01` through `kodim20`, which therefore materialize as:

- training: `kodim01` through `kodim16`;
- validation and checkpoint selection: `kodim17` through `kodim20`.

`kodim21` through `kodim24` are deliberately absent from the training bundle. Keep them for the
ordinary artifact-bound Noema recipe used after training. Neither `train.py` nor the demonstration project's
`evaluate.py` may read those held-out test images. `evaluate.py` reports validation evidence only;
the ordinary Noema benchmark supplies test evidence.

## 4. Implement and train externally

Review the slot, scenario, and data contracts. Implement encoder and decoder modules with the declared
I/O. Choose the reconstruction or task loss, optimizer, scheduler, trainer, and checkpoint-selection
rule in your own project. From any directory inside the clone, copy this block. It returns to the
repository root, enters the export directory, validates the contract, then returns to the repository
root:

```bash
cd "$(git rev-parse --show-toplevel)"
cd differentiable_exports/deepjscc_kodak_awgn
uv run --project ../.. python validate_contract.py
cd ../..
```

This verifies the recipe, contract, scenario-graph, and file hashes. Test your own modules against
`interfaces.py`, the declared tensor ABI, constraints, and gradient route before starting a long
training campaign.

To reproduce the checked-in DeepJSCC demonstration, copy this block from any directory inside the
clone. It returns to the repository root, prepares the example training project, then uses the safe
bundle-root launchers to validate, train, and evaluate it:

```bash
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/differentiable_exports/deepjscc_kodak_awgn"

cd "$ROOT"
uv run --extra onnx python demo_trainings/prepare_example.py deepjscc-image "$BUNDLE"
cd "$BUNDLE"
uv run --project "$ROOT" --extra onnx python validate_contract.py
uv run --project "$ROOT" --extra onnx python train_demo.py
uv run --project "$ROOT" --extra onnx python evaluate_demo.py
cd "$ROOT"
```

This helper is part of the demonstration material, not the Workbench interface. It supplies the
tutorial's ready `reference_training/` directory. Skip it when using your own training stack; you
may add your own model, loss, and trainer under the bundle directory instead. Workbench detects
attached files automatically. Under **Dataset capture**, this file-backed example reports that no
dataset materialization is required, so it has no **Capture all datasets** action. Capture-backed
bundles use that action and write their three splits under `<bundle>/data/`.

Run the researcher-owned commands above. After training returns `trained_artifact.yaml` and its
declared components, select **Validate returned model** under **External model**. The interface shows
the validation phase and elapsed time; a successful check reports **model interface valid**.
It checks the tensor ABI, component and contract hashes, operation bindings, and bundle
integrity—not reconstruction quality. The ordinary held-out benchmark remains the quality test.

Run a fresh bundle directory when changing the selected IDs or source files. The export hashes are a
deliberate audit boundary; editing the data contract, silently replacing an image, or asking a
trainer to invent a new split must fail validation instead of changing the experiment unnoticed.

## 5. Package and return the trained artifact

The preferred architecture-neutral return is a schema-v2 ONNX package. Start from
`trained_artifact.template.yaml`, save the completed manifest as `trained_artifact.yaml`, and keep the
exported contract unchanged beside it:

```text
training_contract.yaml
trained_artifact.yaml
artifacts/encoder.onnx
artifacts/decoder.onnx
```

The executable ONNX graphs carry the architecture. The manifest pins every component with a full
SHA-256, pins both the semantic and byte hashes of `training_contract.yaml`, declares the tensor
entrypoints and real/imaginary representation for complex symbols, and groups encoder and decoder
bindings so they cannot accidentally be applied as unrelated models. Architecture, loss, optimizer,
and trainer may be recorded as provenance, but they are not compatibility gates.

The DeepJSCC demonstration project writes the same schema-v2 structure from `reference_training/`.
For an external implementation, complete the generated template and place the declared files in a ZIP or
`.noema-artifact` archive. Workbench performs the actual contract, hash, ONNX-signature, tensor-ABI,
operation-binding, and runtime checks when the package is imported. `package_artifact.py` has a narrower
purpose: it creates a runtime-adapter validation request for formats that need a separate adapter. It
does not turn an arbitrary file into a ready Noema artifact by itself.

The next tutorial returns the paired artifact to the ordinary recipe and benchmarks it. There is no
DeepJSCC-specific benchmark command: compatible artifacts bind to the declared slots, then the normal
recipe and benchmark workflow runs.
