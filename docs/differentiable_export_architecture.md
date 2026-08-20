# Training and Export Architecture

This document defines how Noema supports trainable semantic-communication research without becoming
a general-purpose training framework. Noema owns reproducible scenarios, typed replacement slots, data
capture, frozen differentiable support modules, artifact compatibility, and benchmark evaluation.
Researchers own model architecture, loss, optimization, trainer, and model selection.

The product export is an architecture- and loss-neutral **training bundle**. Demonstration trainers
live separately under `demo_trainings/`; they are not part of the neutral export. Tutorial
preparation helpers may attach a ready `reference_training/` example afterward. In a DeepJSCC
recipe, for example, explicit attachment also adds root-level `train_demo.py` and
`evaluate_demo.py` launchers plus `RUN_DEMO.md`. The manifest records these as non-normative files
owned by the demonstration helper; a neutral bundle never contains them. In the recipe,
`model.deepjscc_external_encode` and
`model.deepjscc_external_decode` are typed sender and receiver slots. They define expected inputs,
outputs, and artifact-return locations. They are not pre-existing neural networks to extract from the
recipe.

## Intended Workflow

```text
recipe
  + separate training plan (replacement steps, recipe loss steps, capture, framework)
  -> neutral training contract + frozen differentiable support
  -> external model/loss/trainer chosen by researcher
  -> ABI-conformant, hash-pinned trained artifact
  -> artifact bound to the compatible slots in an ordinary recipe
  -> ordinary frozen benchmark evaluation
```

The same recipe graph is the source of truth throughout the workflow. A separate training plan records
the researcher's later choices without mutating that graph. A recipe describes the research
scenario before execution. A run or benchmark result bundle is evidence produced after execution.
Training-contract export does not fork the scenario definition into a second hidden format. It derives
slot interfaces, dataset settings, frozen differentiable modules, channel assumptions, constraints,
accounting points, and artifact bindings from recipe steps and operation contracts. The checked-in
demo projects exercise that interface without becoming part of it.

The Workbench presents this contract in one order. The researcher selects **Train/replace** in the
**Operation Training Capabilities** table, sets **Training bundle** > **Bundle directory**, reviews
**Dataset definition** > **Captured signals** and **Dataset size and splits**, chooses the **Support
framework** and **Overwrite generated files**, and selects **Export training bundle**. **Dataset
capture** and **External model** are always visible below, but their actions remain disabled until
bundle export.
Export must precede managed capture because it freezes the replacement ABI, selected tensors, split
recipes and seeds, output paths, and integrity hashes. A tutorial may then attach its optional demo
project. Workbench monitors the selected exported bundle and discovers attached project or artifact
files automatically. **Capture all datasets** materializes `<bundle>/data/train`,
`<bundle>/data/validation`, and `<bundle>/data/test` in **Dataset capture**.
After a successful capture, this action becomes **Recapture all datasets** and remains enabled as an
explicit overwrite of every managed split. A recapture is written to a sibling staging directory
and replaces the managed split only after the new capture succeeds; a failed recapture leaves the
previous dataset intact. The researcher then runs the researcher-owned training process. Once
**External model** detects the returned artifact, the
researcher selects **Validate returned model**. That action checks interface conformance and integrity;
benchmark or held-out metrics, not interface validation, measure model quality.

Canonical templates therefore never embed `dataset_capture` or `metadata.training_performed`.
Legacy embedded capture configuration remains readable and is migrated into a plan; generated
capture-job recipes may still contain a materialized `dataset_capture` section because they are jobs,
not reusable scenarios. Export writes `training_plan.yaml` and records a plan SHA independently from
the neutral scenario recipe SHA.

Within that plan, `selected_steps` names portable replacement boundaries and `loss_steps` names the
recipe evaluation sinks used for route analysis. A capture-backed plan keeps that selection as
provenance even though its exported typed scenario has no live loss route; `live_route_loss_steps`
is empty in that case. An externally authored plan may record an `objective` for provenance, but
Workbench does not choose or inject one.

## Replacement Boundary Versus Gradient Boundary

Choosing **Train/replace** in Workbench answers *which visible Block will the researcher's new
model replace?* It does not assert that the operation currently installed in that Block is itself
differentiable or exportable. The current operation can be an analytical rule, a hard-decision
implementation, a NumPy routine, or an adapter placeholder. Training substitutes a new implementation
at that boundary, so the current operation's gradient metadata is irrelevant to the new model's
backward pass.

Three identifiers appear at this boundary and should not be conflated:

| Term | Example | Where it is used |
| --- | --- | --- |
| Visible **Block** | **Classifier** | Graph, Recipe panel, and the **Block** column in Workbench; use this name in researcher instructions. |
| Recipe step ID | `receiver` | Recipe-local wiring, delayed hover details, exported contracts, and CLI arguments such as `--replacement receiver`. |
| Operation ID | `model.modulation_classifier_adapter` | Exact contract currently assigned to the step; it describes the present implementation and its ABI, not the researcher's replacement architecture. |

The **Operation Training Capabilities** table consequently reports independent facts:

- **Built-in fine-tuning** is an explicit property of the installed implementation
  (`Operation.fine_tuning_supported` plus a callable operation-owned `fine_tuning_provider`, exposed
  as `training_capabilities.built_in_fine_tuning`). It means Noema has a concrete path for updating
  that implementation's own parameters. No bundled operation currently advertises this action; the
  pretrained BART evaluation path is intentionally reported as **No**. The legacy
  `differentiability.trainable_params` value alone does not grant this capability.
- **Portable replacement** derives only from a complete `trained_artifact_abi`, so an independently
  trained model can be returned and bound at that Block. Neither `trainable_params: true` nor
  differentiability alone makes a Block portable-replacement ready. Readiness requires a non-empty
  `component_id`, `component_role`, matching entrypoint/binding, typed ports, and declared runtime
  binding parameters—the same fields enforced again during artifact import.
- **Gradient** reports the current operation's backward behavior, while **Differentiable support**
  requires a `full` or `surrogate` gradient plus an exportable runner materialization for that
  unchanged operation on a live replacement-to-loss route. Neither decides whether that same row may
  be selected as a replacement target.

The gradient requirement depends on the training style:

- In a **live task-loss** export, the new model's output is passed through recipe-defined support to a
  task loss in the same autograd graph. Only the *frozen downstream support Blocks* on the route from
  a selected replacement's output to that loss must have differentiable, exportable materializations
  in the chosen framework. The original operation at the selected boundary is replaced and is not
  checked as support. If the DAG splits and reconverges before the loss, every live branch is included;
  the compiler does not choose a single shortest path. Blocks upstream of the first selected replacement only produce its inputs and
  do not need a backward path. The researcher-supplied replacement must participate in the external
  trainer's autograd graph, but that is a property of the new module—not metadata inherited from the
  original operation.
- In **capture-backed external training**, ordinary capture execution writes the researcher-selected
  inputs, targets, or auxiliary tensors, and the external trainer defines how to use them. No gradient
  route through the recipe is required. Capture-producing operations need only run under the capture
  contract and emit the declared tensors. The selected Block still needs a replacement/artifact ABI
  so the trained model can return for frozen benchmarking.

For custom array-valued slots, Noema derives the required capture taps from inputs entering the
selected replacement boundary, skips connections internal to a jointly selected group, and exposes
all other capturable graph outputs as optional signals. Workbench always exposes the tap and split
editor when capture is required. Export then writes a generic `captured_generic_tensors` data
contract plus train/validation/held-out-test capture jobs. A non-array slot input is reported as an
unsupported boundary instead of silently producing an empty bundle; that operation needs an
operation-owned file-backed provider or an explicit array boundary.

Each generated capture recipe executes only the ancestors of its selected tensors. It therefore
declares a `custom` execution profile with the source recipe's standard profile retained in
`based_on`; the unchanged source recipe remains the object validated against the complete standard
topology contract.

The incoming tensors are required. Targets remain a researcher choice: select the original slot
output for distillation, a later graph output for task supervision, or neither when the external
trainer computes a label-free/local objective. That choice is made after the neutral recipe exists.

Outputs of the currently installed replacement and its descendants are not forbidden. Workbench
marks them **current pipeline** because an offline capture records the present implementation's
value and will not react to the future replacement. Researchers may still select such a signal for
distillation, task supervision, diagnostics, or auxiliary objectives. A checked-in demonstration
project may validate a narrower set of signals for its own method, but that validation is not part
of recipe inspection or generic contract export.

A capture-backed local objective, such as label-free OFDM allocation trained from captured channel
gains and a Shannon formula inside the trainer, has the same boundary: its local loss must be
differentiable with respect to the new model, but no recipe-DAG gradient route is required.

## Scope

Noema provides:

- reproducible datasets, channel settings, task settings, seeds, and benchmark protocols;
- typed recipe DAGs with visible sender, payload, channel, receiver, and metric boundaries;
- capture datasets generated from named taps in a recipe graph;
- neutral training-contract export with either captured data or an honestly materialized downstream
  task-loss route in the selected gradient framework;
- differentiability metadata for every operation contract;
- lint reports that explain which parts of a recipe are trainable, frozen, non-differentiable, or
  externally supplied;
- safe, hash-pinned trained-artifact contracts for returning supported checkpoints into existing recipe
  operations;
- explicit adapter/plugin contracts for custom runtimes outside a built-in safe checkpoint format;
- frozen benchmark evaluation of returned artifacts under the same protocol used for baselines.

Noema does not aim to become:

- a full experiment manager for arbitrary training campaigns;
- a replacement for PyTorch Lightning, Hydra, Accelerate, Sionna training scripts, or a lab-specific
  training stack;
- a service that silently changes benchmark protocols to make training easier;
- a system that treats an exported training harness result as a publishable benchmark number without
  rerunning the frozen evaluation path.

## Modes

### Benchmark Mode

Benchmark mode is the existing execution path. It includes both ordinary recipe runs and benchmark
pack runs.

Recipe run:

```bash
noema recipe run recipes/my_recipe.yaml
```

Benchmark run:

```bash
noema benchmark run benchmarks/benchmark_v1/my_protocol.yaml
```

Benchmark mode is for comparable measurement, not training. It:

- execute the recipe DAG with registered operation contracts;
- enforce strict lint for publishable benchmarks;
- preserve fixed bit/symbol/accounting boundaries;
- write `recipe.json`, `summary.json`, `manifest.json`, metrics, and artifacts;
- record operation contracts, environment evidence, seed policy, artifact hashes, and protocol IDs;
- reject or warn on non-comparable status such as failed, canceled, missing required metrics, missing
  bit boundaries, or unsupported channel accounting.

Training artifacts may be consumed by benchmark mode only after their format and SHA-256 are verified.
For a built-in checkpoint runtime, the artifact configures existing compatible operations; it does not
create task-specific operation IDs or a separate benchmark product. A custom model that needs arbitrary
Python execution still returns through an explicit adapter/plugin boundary. In both cases, a normal
recipe or benchmark pack is run again.

### Capture Mode

Dataset capture mode generates training datasets from recipe taps. A tap is a declared point in the DAG whose
outputs should be serialized for training. Examples include:

```text
data.images
sender.latents
payload_encoder.bits
modulator.symbols
wireless_channel.rx_symbols
demodulator.llr
tx_bit_boundary.bits
rx_bit_boundary.bits
receiver.images
evaluation.report
```

Dataset capture mode runs a recipe under a declared scenario until it has written the requested number of
aligned capture records. For the MVP, capture records are serialized as compressed `.npz` shards:

```text
.noema/dataset_captures/<capture_id>/
  shards/
    shard_0000.npz
    shard_0001.npz
  schema.json                capture manifest and schema
  tap_manifest.json          tap source, dtype, shape, and artifact provenance
  recipe.json                recipe used to create the capture
  channel_distribution.json  per-run channel/sweep metadata
  split.json                 split and shard index
```

This `.noema/dataset_captures/<capture_id>` layout is the standalone CLI capture workspace. Captures
managed by an exported training bundle instead remain inside that bundle under `data/train`,
`data/validation`, and `data/test`.

`capture.samples` means aligned capture records, not raw source files or channel symbols. A tapped
array with shape `[B, ...]` contributes `B` records. A 1D vector such as one packet of bits or one
sequence of received symbols is treated as one record unless the artifact metadata explicitly declares
`record_axis: 0`, `sample_axis: 0`, or `capture_record_axis: 0`. All taps in a dataset capture must expose
the same record count or capture fails with a clear error.

The current shard controls are:

```yaml
capture:
  split: train
  samples: 100000
  shard_size: 1024
  max_runs: 2000
  seed_mode: increment_run_seed  # fixed_seed | increment_run_seed | recipe_seed_plus_shard
  sweep:
    wireless_channel.snr_db: 0:5:30
```

Sweeps are deterministic round-robin over the configured values/grid. `.npz` is the only capture
format for the MVP; larger appendable formats such as Zarr can be added later without changing the
recipe-level capture contract.

Dataset capture mode should support at least two use cases:

- supervised receiver training, such as `rx_symbols -> target_bits` or `llr -> target_bits`;
- semantic/model training, such as `source_image -> reconstructed_image` or
  `semantic_state -> target_answer`.

Captured datasets must keep sample IDs, split names, seed derivation, channel settings, units, dtype,
shape, and bit-count metadata. If a tap contains channel bits, the canonical bit contract remains:

```text
np.uint8 unpacked bits, one bit per element, values 0 or 1
```

Dataset capture mode is allowed to be large and offline. It should not be treated as a benchmark result by
itself because it is training evidence, not final method evaluation.

Capture is an execution and serialization requirement, not an autograd requirement. A supervised
capture contract may cross hard or otherwise non-differentiable operations before producing its
input/target tensors. Those operations do not block training because the external trainer starts a
new graph at the captured model input and computes loss against the captured target.

### Training Contract Export Mode

Training contract export creates a neutral handoff bundle from a recipe. It does not choose a network,
objective, optimizer, scheduler, trainer, or checkpoint-selection rule. The normative files are:

`scenario_graph.json` is the authoritative DAG representation. The optional differentiable-graph
preview may produce a simple block sequence only when the selected downstream support is one fully
materialized linear chain. Workbench does not show a warning merely because the authoritative graph
is branched; genuine contract or export failures are reported through the normal validation errors.

```text
differentiable_exports/<export_id>/
  training_contract.yaml      typed slots, signals, constraints, and return bindings
  scenario_graph.json         named-port DAG, placeholders, frozen context, and gradient capabilities
  data_contract.yaml          capture tensors and splits when offline capture is required
  capture_*_recipe.yaml       generic capture jobs when required by the data contract
  trained_artifact.template.yaml  ONNX entrypoints and ordinary-operation bindings
  interfaces.py               architecture-neutral callable protocols
  validate_contract.py        contract, recipe, graph, and hash checks
  package_artifact.py         creates a runtime-adapter validation request
  project_manifest.yaml       generic Workbench capture and artifact-discovery contract
  noema_recipe.yaml           unchanged source recipe snapshot
  README.md                   handoff and ordinary benchmark workflow
  data/                       managed capture outputs when capture is required
    train/
    validation/
    test/
```

The current bundle keeps these ownership boundaries explicit. In particular, `model.py`, `losses.py`,
and `train.py` are never normative Noema contracts.

Selected recipe steps define **replacement slots**, not implementations. The selected current
operation supplies the boundary contract; its own gradient and differentiable-export flags are not
requirements because the researcher's model takes its place. A slot contract must state:

- whether it replaces a complete operation or an internal policy within a composite operation;
- input/output kinds, names, dtype, layout, symbolic shape, and dynamic axes;
- runtime conditioning such as CSI, noise variance, SNR, or a power budget;
- constraints owned by Noema before or after the model;
- the compatible artifact binding and any atomic multi-block binding group.

The generated bundle distinguishes two interfaces that must not be conflated. The **operation
boundary** names the complete ports used by the recipe and captured dataset. The **runtime artifact
ABI** names the exact entrypoint tensors exposed by the returned model; its names, dtypes, shapes,
and layouts may differ because the existing operation adapter can unpack structured inputs, derive
values from recipe parameters, or convert layouts. `interfaces.py` publishes both boundaries and
identifies the captured operation-input subset required by the adapter. It copies a value only when
an identity mapping is proven; every other runtime input is marked `operation_adapter_required`
instead of guessing preprocessing. Returned-model validation checks the actual model against this
runtime ABI.

The export supports two data styles without creating task-specific Workbench sections:

- **live differentiable scenario**: recipe-derived frozen downstream support connects replacement
  outputs to the chosen task loss during external training;
- **capture-backed contract**: Workbench runs the bundle's frozen capture jobs into its managed
  `data/{train,validation,test}` directories before external training, with no gradient route through
  the recipe required.

Workbench automatically analyzes every reachable evaluation sink and does not expose a route-endpoint
selector. This keeps recipe metrics from looking like training objectives. Advanced CLI workflows
may narrow route analysis with `training_plan.loss_steps` or `--route-loss` when intentionally
exporting one live differentiable route; that remains separate from the external trainer's objective.

For a file-backed live-differentiable image scenario, the neutral export also materializes a root
`data_contract.yaml`. It contains the recipe-selected image IDs, an explicit train/validation
partition, resolved paths, file sizes, per-file SHA-256 values, and preprocessing. An external
trainer must consume those exact records; it may not scan a directory, synthesize fallback samples,
or perform its own implicit split. The contract exposes no test split. Held-out test inputs belong to
an ordinary artifact-bound Noema recipe or benchmark after training.

The generic capture runner already accepts recipe-declared taps for any operation. Operation-owned
providers currently generate `data_contract.yaml` and split capture jobs for per-subcarrier resource
allocation, QPSK receiver demapping, automatic modulation recognition, and CSI feedback. Other
capture-backed slots must declare their taps and split recipes explicitly until more providers are
registered from operation contracts. This is a compiler limitation, not a reason to add task-specific
sections to Workbench.

Workbench exports only the neutral contract. It has no model, loss, optimizer, trainer, or demo
selector. Researchers may add those files under the bundle directory. Repository demonstrations
provide separate example projects:

| Demonstration project | Example content |
| --- | --- |
| `deepjscc-image` | reference CNN encoder/decoder, image MSE, and a small PyTorch loop |
| `neural-receiver` | receiver example with bit BCE and a small capture-backed loop |
| `modulation-recognition` | 1-D CNN classifier, cross entropy, and a capture-backed loop |
| `resource-allocation` | Deep Sets policy, negative Shannon objective, and exact-power example |
| `csi-feedback` | paired multiresolution autoencoder and hybrid NMSE/rate example |

These projects demonstrate the workflow; they do not constrain researchers to those architectures or
losses. Each demo tutorial's preparation helper attaches a ready `reference_training/` project after
the neutral Workbench export. The helper also creates safe root launchers for its nested training and
evaluation scripts and refuses to overwrite same-named researcher files. A normal researcher workflow
skips that step and trains directly against the same slot, scenario, data, and artifact contracts.

The contract-only form needs only the recipe, selected slots, framework, and output:

```bash
noema differentiable export recipes/deepjscc_kodak_awgn_train.yaml \
  --training-plan demo_trainings/deepjscc_image_reconstruction/training_plan.yaml \
  --out differentiable_exports/deepjscc_kodak_awgn
```

Workbench uses one primary API workflow, `POST /api/recipe/differentiable-export`. The request carries
the recipe, selected slots, framework, output directory, and overwrite choice. Demo-specific model
and objective identifiers are not part of that request.

Training-contract export preserves the benchmark scenario in metadata, but it is not a benchmark.
External training returns an ABI-conformant artifact, which Noema then evaluates through an ordinary
recipe and benchmark pack.

### Trained-Artifact Return Path

A portable runtime returns a model through `trained_artifact.yaml`. The manifest declares:

- one or more self-contained ONNX components with exact file SHA-256 values;
- the neutral contract's canonical semantic SHA-256 and a separate exact contract-file SHA-256;
- typed entrypoints checked against both ONNX signatures and operation-owned adapter ABIs;
- compatible existing operation contracts;
- required inputs and the parameters applied to each operation;
- a binding group when one artifact must configure several steps atomically;
- source recipe/contract identity and training provenance.

Workbench's **Validate returned model** action verifies these interface and integrity properties. It
does not require a particular loss, trainer, training history, or evaluation score, and a successful
validation must not be described as evidence of model quality.

The preferred neutral ABI is ONNX because the executable graph carries its architecture and can be
validated without importing researcher training code. Complex values cross this ABI as explicitly
declared real/imaginary tensors when direct complex-tensor support is not portable. A paired DeepJSCC
artifact therefore normally contains encoder and decoder graphs in one atomic binding group; a learned
allocation policy normally contains one graph plus Noema-owned feasibility postprocessing.

Architecture-specific safe NPZ adapters may remain as explicit built-in runtimes, but they are not
the general return contract or the current demo output. A model that cannot use the
ONNX-first ABI must return through an explicitly declared trusted adapter/plugin. Runtime compatibility
must validate I/O and constraints; it must not reject an artifact merely because it used a different
training loss, optimizer, or architecture.

## Differentiability Metadata

Every operation contract should eventually expose differentiability metadata. The goal is not to make
every block differentiable; the goal is to make gradient behavior explicit and lintable.

Suggested operation-level fields:

```yaml
differentiability:
  framework: torch | sionna | tensorflow | numpy | blackbox | none
  gradient: full | stop | surrogate | none
  trainable_params: true
  exportable: true
  reason: "Uses differentiable AWGN channel; hard demodulation stops gradients."
```

Recommended gradient meanings:

| Gradient | Meaning | Example |
| --- | --- | --- |
| `full` | Native autograd path exists in the declared framework. | PyTorch neural encoder and pure-PyTorch AWGN addition. |
| `surrogate` | Uses a documented surrogate gradient. | Quantizer with straight-through estimator. |
| `stop` | Forward execution is valid but gradients intentionally stop here. | JPEG entropy coding, hard demodulation, packet checks. |
| `none` | No meaningful gradient behavior is declared. | Dataset loaders, metric reports, black-box adapters. |

Operation metadata should distinguish four questions:

- If this operation is retained as frozen support, can gradients pass through its inputs and outputs?
- If retained as frozen support, can it be materialized by the selected training backend?
- Does the installed operation support built-in fine-tuning of its own parameters?
- Does its Block expose a complete trained-artifact ABI for an external replacement?

These answers are independent. For example, a frozen neural feature extractor may pass gradients
without supporting built-in fine-tuning, while a non-differentiable analytical classifier can still
sit in a Block with **Portable replacement**. Selecting that classifier Block substitutes researcher
code; it does not try to differentiate the analytical implementation.

## Linter Behavior

The recipe linter should report training/export feasibility separately from benchmark validity.
A recipe can be valid for benchmark evaluation while impossible to train end-to-end.

Training lint should classify each selected training question as one of:

- **fully differentiable**: every frozen downstream support Block between each selected replacement
  output and the chosen task loss has a supported gradient materialization;
- **differentiable with surrogates**: at least one such downstream support Block uses a declared
  straight-through or other surrogate gradient;
- **partially differentiable**: a task-loss route exists, but a named frozen downstream support Block
  stops it;
- **dataset-capture-only**: inputs and targets can be captured for external supervised training, so
  no recipe gradient route is required;
- **not exportable**: a required replacement ABI, capture contract, or frozen downstream support
  materialization is unavailable.

The installed operation at a selected replacement boundary must be excluded from gradient breaks and
differentiable-export blockers. Its complete trained-artifact ABI is checked separately. Likewise,
non-differentiable Blocks that merely produce captured inputs or targets are not gradient blockers.

Example linter output should be explicit:

```text
training lint: dataset-capture-only
ok: Demodulator is a typed replacement boundary (step `demodulator`)
note: current operation `demodulation.hard_qpsk` is replaced; its gradient is not a requirement
ok: capture `wireless_channel.rx_symbols` -> input and `tx_bit_boundary.bits` -> target
note: supervised bit loss is computed in the external trainer; no recipe gradient route is required
```

For a live task-loss export, lint should instead name only a failing frozen downstream Block, for
example **Wireless channel** (step `wireless_channel`, operation `wireless.awgn`), and explain which
selected Block-to-loss route it interrupts.

For frozen benchmark evaluation, the same recipe may still pass strict benchmark lint if all required
accounting boundaries and metrics exist.

The CLI exposes this training lint directly:

```bash
noema differentiable inspect recipes/text_bart_jscc_clean.yaml
noema differentiable inspect recipes/text_bart_jscc_clean.yaml --json
```

The command validates the recipe against the operation registry, reads each step differentiability
metadata, treats `metrics.*` steps as loss/evaluation sinks, and reports one of:
`full_gradient_possible`, `partial_gradient_possible`, `dataset_capture_only`, or `not_trainable`. The
recommendation maps those statuses onto `differentiable_export`, `receiver_only`, `dataset_capture`, or
`benchmark_run`.

`noema differentiable inspect` can also inspect a selected training question instead of only the whole recipe:

```bash
noema differentiable inspect recipes/deepjscc_kodak_awgn_train.yaml \
  --replacement sender,receiver \
  --loss evaluation \
  --json
```

In that mode the report includes each selected replacement Block's route to loss, the portion that
crosses channel/PHY support, exact downstream gradient breaks, exact downstream export blockers,
path issues such as a selected Block not reaching the loss, and suggested capture taps such as
`wireless_channel.rx_symbols` and `channel_encoder.coded_bits` for capture-backed receiver training.
The CLI values passed to `--replacement` are recipe step IDs even though the UI and explanatory text
lead with visible Block names. `--optimizable` remains accepted as a compatibility alias.

## Minimal Differentiable PHY Blocks

Differentiable training modules are intentionally separate from the NumPy artifact executor. They are
used in generated training projects and are not replacements for ordinary recipe operations. The
available building blocks include:

- `PowerNormalizationBlock` for differentiable complex-symbol power normalization;
- `AwgnChannelBlock` and `FlatRayleighChannelBlock` as pure PyTorch fallback PHY blocks;
- `SionnaAwgnChannelBlock` and `SionnaFlatFadingChannelBlock` for the supported Sionna 2.x/PyTorch
  channel materializations;
- `QamPamMapperBlock` for frozen digital mapping;
- `SoftDemapperBlock` for max-log LLR output;
- Sionna 2.x/PyTorch mapper and soft-demapper blocks for the matching registered digital path.

The closed DeepJSCC demonstration deliberately uses the pure-PyTorch path. Noema's generic typed
graph supports the registered Sionna 2.x/PyTorch AWGN and flat-fading blocks, but the specialized
DeepJSCC reference-project exporter currently accepts only `framework=torch`. That restriction is a
property of this checked-in trainer/export template, not a claim that Sionna 2.x breaks PyTorch
autograd.

The supported differentiable MVP path is:

```text
encoder output symbols -> power normalization -> AWGN channel -> decoder -> loss
```

The gradient metadata is intentionally conservative: soft demapping is differentiable with respect to
received symbols, bit-to-constellation mapping stops gradients to hard input bits, CRC stops gradients,
and hard channel decoding should be treated as a gradient break unless a future adapter declares a
surrogate.

Full OFDM, LDPC, and 5G NR differentiable export remain separate follow-up layers. LDPC is expected to
enter evaluation/capture flows before Noema claims full end-to-end differentiability through a
protected digital receiver.

## Example Workflow 1: DeepJSCC-Style Image Transmission

Goal: train an image semantic communication system end to end with a differentiable wireless channel.

Scenario:

```text
image -> external encoder slot -> power normalization -> pure-PyTorch AWGN -> external decoder slot -> external loss
```

1. Define a Noema recipe with an image source, typed sender/receiver interfaces, power normalization,
   AWGN, symbol accounting, and image reconstruction metrics. The interfaces contain no neural
   architecture.
2. Select the sender and receiver slots.
3. Run training lint on the selected subgraph:

   ```text
   data.images -> sender.symbols -> wireless_channel.rx_symbols -> receiver.images -> loss
   ```

4. Export the pure-PyTorch training contract. It includes:

   - source/capture and slot contracts derived from recipe settings;
   - differentiable power normalization and complex AWGN with the recipe power/SNR policy;
   - encoder/decoder placeholders into which external modules are injected;
   - paired encoder/decoder artifact-return bindings and conformance tests.
   - an explicit, hash-pinned train/validation image data contract when the recipe source is
     file-backed. Validation may select checkpoints; held-out test images are intentionally absent.

5. The researcher chooses the encoder/decoder architecture, loss, optimizer, and trainer externally.
   The separate DeepJSCC demonstration project is one non-normative example.
6. Package an ONNX-first paired artifact and `trained_artifact.yaml` with its SHA-256. A cataloged
   architecture-specific demo format may be used only when explicitly selected.
7. In Workbench, selecting that artifact on the sender or receiver applies both compatible bindings to
   an ordinary recipe copy.
8. Noema runs the frozen benchmark protocol with that recipe and records PSNR/MS-SSIM/LPIPS, bit or
   channel-use accounting, BER if relevant, timing, memory, environment, and artifact hashes.

The demonstration project's `evaluate.py` evaluates the exported validation list and must label its output
as validation. It is not the test step. Test images are selected only in the later ordinary recipe so
`train.py` cannot use them for optimization, early stopping, or model selection.

Important rule: training-time channel randomization may use a range or distribution, but the frozen
benchmark must evaluate the declared benchmark SNR/rate grid exactly.

## Example Workflow 2: Neural Receiver Training

Goal: keep a conventional digital transmitter fixed and train a neural receiver offline.

Scenario:

```text
digital transmitter -> Sionna channel -> captured rx symbols + target bits -> offline receiver training
```

1. Define a Noema recipe with source bits, channel coding/modulation, Sionna channel, and a receiver
   placeholder.
2. Use dataset capture mode to record taps:

   ```text
   tx_bit_boundary.bits
   modulator.symbols
   wireless_channel.rx_symbols
   wireless_channel.channel_state
   demodulator.llr       optional baseline feature
   ```

3. Capture bundle writes sample IDs, SNR, modulation, code rate, channel realization metadata, symbol
   dtype/shape, and target bit count.
4. Export a receiver training contract backed by the capture dataset. The training target is usually
   `tx_bit_boundary.bits` or a coded-bit boundary, depending on whether the neural receiver replaces
   only demodulation or also channel decoding.
5. Researcher trains the receiver externally with a bitwise or codeword loss.
6. The trained receiver returns as a schema-v2 artifact bound to the compatible receiver operation. A
   portable ONNX artifact consumes `channel.rx_symbols.complex_numpy` or `channel.llr.numpy` and emits
   canonical unpacked bits; a non-portable runtime requires an explicit trusted adapter/plugin.
7. Noema evaluates the bound artifact in frozen benchmark mode, reporting BER/BLER, task metrics if
   there is an end task, channel uses, latency, memory, and full reproducibility metadata.

Important rule: the receiver binding must declare whether it replaces the demodulator, channel
decoder, or both. This avoids comparing systems with different accounting boundaries.

No recipe gradient path is needed in this workflow. The original receiver implementation and the
capture-producing transmitter/channel may be non-differentiable: they produce frozen input/target
records, and the new receiver's supervised graph begins at those records.

## Stable Boundaries

Training/export features should reuse the same canonical boundaries already used for evaluation:

- source data boundary: sample IDs, splits, dtypes, shapes, and units;
- semantic/model boundary: latents, indices, symbols, text states, detections, masks, or answers;
- payload boundary: internal representation converted to canonical bits/bytes;
- channel boundary: coded bits, symbols, received symbols, LLRs, and channel state;
- receiver boundary: reconstructed image, text, answer, detections, masks, or task output;
- metric boundary: task/reconstruction/communication/system metrics.

A differentiable export may omit some evaluation-only metrics, but it should not redefine the meaning of
bits, symbols, sample IDs, or channel settings.

## Current Limits And Extension Points

- Capture storage format: NPZ is simple and portable; Parquet/Arrow/Zarr may be better for large
  tabular or chunked tensor captures.
- Distributed training: exported scripts are single-process; a distributed launcher must preserve
  Noema's benchmark contracts.
- Differentiable approximations: surrogate gradients should be opt-in and visible in manifests.
- Sionna versioning: every differentiable harness must pin compatible Sionna 2 and PyTorch versions
  and pass a finite sender-gradient test before registration.
- Security: built-in artifact paths should use non-executable, strictly validated formats. External
  training code and custom adapters/plugins remain explicit researcher code and are not silently
  executed by benchmark validation.
