# Backend Materialization Architecture

## Core Model

Noema uses one recipe contract with multiple faithful materializations.

```text
recipe DAG + operation contracts
  -> benchmark materialization
  -> capture materialization
  -> differentiable training/export materialization
```

The recipe graph and operation contracts are the source of truth. Benchmark execution, capture
dataset generation, and differentiable export may execute the same operation with different
implementation technologies, but they must share the same operation identity, parameter schema,
artifact schema, boundary contracts, accounting rules, and declared equivalence behavior.

This avoids two failure modes:

- making all of Noema depend on a single tensor framework even when the work is discrete,
  file-based, or classical-codec based;
- letting benchmark and training paths drift into two hidden definitions of the same experiment.

## Terminology

### Format

A format is what crosses stable recipe boundaries or is saved for later inspection.

Examples:

- `.npz` array shard;
- JSON or YAML metadata;
- CSV metric table;
- PNG/JPEG image artifact;
- codec bitstream;
- unpacked `uint8` bit vector;
- `complex64` symbol array;
- PyTorch checkpoint;
- ONNX or OpenVINO model artifact.

Formats are about representation and persistence. They are not the same as implementation backend.
For example, a Torch backend may still export an `.npz` capture shard, and a NumPy benchmark block may
load a PyTorch checkpoint through an external adapter.

### Backend

A backend is the technology used to implement an operation.

Examples:

- NumPy;
- Torch;
- Sionna;
- TensorFlow;
- ONNX Runtime;
- OpenVINO;
- C++ or native extension;
- external executable;
- black-box remote or lab-specific adapter.

Backends are about computation. A backend can be appropriate for benchmarking, differentiable export, both,
or neither depending on the operation.

### Runner

A runner is the purpose and execution style used for a recipe.

Noema currently treats these as distinct runner families:

- **benchmark runner**: executes recipes and benchmark packs to produce comparable evidence,
  manifests, metrics, and artifacts;
- **dataset capture runner**: executes recipes to materialize researcher-selected tensor taps;
- **training/export runner**: compiles a differentiable PyTorch/Sionna-style harness or graph from a
  supported recipe subgraph.

The benchmark runner is evidence-oriented. The training/export runner is gradient-oriented. The
dataset capture runner is dataset-oriented.

### Materialization

A materialization is a concrete implementation of an operation for a specific runner/backend pair.

Examples:

```text
wireless.channel
  benchmark + numpy  -> NumPy AWGN/Rayleigh artifact operation
  benchmark + sionna -> Sionna-backed channel artifact operation
  differentiable_export + torch -> torch.nn.Module AWGN block
  differentiable_export + sionna -> Sionna channel wrapper

modulation.digital_modulate
  benchmark + numpy -> hard QPSK symbols from uint8 bits
  differentiable_export + torch -> differentiable mapper or surrogate mapper

channel.identity_link
  benchmark + numpy -> bit-perfect uint8 bypass

channel.identity_symbol_link
  differentiable_export + torch -> gradient-friendly symbol identity
```

The recipe does not change from `wireless.channel` to `wireless.channel_torch`. The selected runner
resolves `wireless.channel` into a compatible materialization using operation metadata.

## Materialization Registry

Noema exposes an internal materialization registry for runner code that needs to turn a stable
operation ID into a concrete backend implementation.

Conceptually, operation contracts now form a table like this:

```text
wireless.channel
  benchmark    + numpy  -> numpy_awgn_rayleigh_artifact
  capture      + numpy  -> numpy_awgn_rayleigh_artifact
  benchmark    + sionna -> sionna_awgn_rayleigh_artifact
  capture      + sionna -> sionna_awgn_rayleigh_artifact
  differentiable_export + torch  -> torch_awgn_or_flat_rayleigh_module
  differentiable_export + sionna -> sionna_awgn_or_flat_fading_module
```

The recipe still says:

```yaml
- id: wireless_channel
  op: wireless.channel
```

Runner code resolves the operation against its purpose and backend:

```python
from noema_lab.core.materialization import build_materialization_registry
from noema_lab.ops import build_registry

operations = build_registry()
materializations = build_materialization_registry(operations)
resolved = materializations.resolve("wireless.channel", runner="benchmark_run", backend="numpy")

assert resolved.operation_id == "wireless.channel"
assert resolved.implementation == "numpy_awgn_rayleigh_artifact"
```

The resolved object preserves both pieces of information:

- the stable `Operation` object, including its input/output contracts and artifact `run()` method;
- the selected materialization metadata, including runner, backend, implementation name, status, and
  notes.

This is separate from recipe syntax. A recipe encodes `wireless.channel`, not
`wireless.channel_sionna` or `wireless.channel_torch`; backend choice is runner policy, CLI/UI
selection, benchmark protocol, or differentiable-export configuration.

## Execution Planning

The benchmark executor resolves the complete recipe before creating a run directory or executing an
operation. `plan_recipe(...)` returns an immutable `noema.execution_plan` contract containing:

- the effective recipe digest and runner;
- one ordered binding for every recipe step;
- the selected backend, implementation, and stable materialization identity;
- the bound Python implementation identity and normalized selection metadata;
- a canonical digest of every operation contract and every step binding;
- a canonical digest of the complete plan.

Backend precedence is deterministic:

1. an explicit backend supplied to the planner;
2. an exact materialization `parameter_bindings` match, such as
   `runtime: learned_artifact` selecting an ONNX Runtime implementation;
3. a backend-selecting operation parameter such as `wireless_backend`;
4. the first implemented materialization in the operation's declared contract order.

`auto` is deterministic and does not inspect which optional packages happen to be installed. It
selects the first compatible implemented materialization in declared contract order and the planner
then replaces the runtime selector with the concrete activating value. For ordinary wireless
benchmark and capture operations this means NumPy; Sionna is a strict explicit selection. An
explicit backend is strict: if that runner/backend pair is not declared as an implemented
materialization, planning fails before storage or operation side effects. A generic operation
parameter named `backend` is treated as a materialization selector only when its schema enum overlaps
the operation's advertised materialization backends; model-library choices such as `diffusers`
therefore remain operation parameters rather than runner policy.

Operations that dispatch internally on a mode, policy, or runtime parameter declare the activating
value directly on each materialization as `parameter_bindings`. The planner uses those bindings in
both directions: authored parameters select the matching materialization, while an explicit planner
backend/implementation injects the unique declared parameter value needed to activate it. The bound
parameter mapping is schema-validated before a run directory is created and is persisted in the plan
as `implementation_metadata.parameter_overrides`. This prevents evidence such as
`backend: external` from being recorded while `runtime: learned_artifact` actually invokes ONNX.

Backend parameter vocabularies do not have to duplicate materialization IDs. A parameter schema may
declare `x-noema-materialization-selector` with a value-to-backend mapping. The shared
`data_plane_backend` schema uses this to translate `python_numpy` to `numpy` and `cpp_native` to
`cpp`. Its declared `automatic_value` resolves `auto` to `python_numpy`. When those translated
backends exactly match the operation's materialization backend domain, that concrete value activates
the selected top-level materialization. Otherwise the control is a subordinate kernel selector: the
planner still pins its concrete effective value and records the authored value, effective value, and
translated target separately as `implementation_metadata.subordinate_runtime_selectors`. It cannot
override a wrapper backend or a broader selector such as `wireless_backend`. Consequently neither a
native extension appearing on `PYTHONPATH` nor a Sionna installation can change the implementation
behind a previously hashed `auto` plan; accelerated data-plane kernels require explicit
`cpp_native`, and Sionna requires explicit `sionna` (except when an explicit Sionna channel-state
artifact makes that backend the only compatible materialization).

The local benchmark executor consumes the operation objects captured by this plan. It does not look
the operations up again while stepping through the recipe. Each successful run stores:

- `recipe.authored.json`, preserving the submitted recipe;
- `recipe.json`, preserving the default-materialized effective recipe;
- `execution-plan.json`, preserving all planned bindings and their digests;
- plan and authored/effective recipe digests in `manifest.json` and `summary.json`;
- the exact execution binding on every completed step in both manifest and summary evidence.

The older `validate_recipe_against_registry(...)` entry point remains validation-only. It checks
operation and graph contracts without choosing a runner, while `plan_recipe(...)` is the authoritative
API whenever execution or export needs a concrete materialization decision.

Run verification treats the plan as evidence rather than decoration. It recomputes the whole-plan,
operation-contract, and per-step binding digests; links authored/effective recipe hashes across the
plan, manifest, summary, and recipe files; and confirms the planned step order and operation IDs
match the effective DAG. It also derives each materialization ID from the operation/runner/backend/
implementation tuple, requires one matching implemented contract entry, and checks that the recorded
parameter bindings are the ones that activate that entry. When verification is given the live
operation registry, the embedded contracts must also match that trusted registry snapshot. Without a
registry, verification proves internal consistency and detects accidental edits, but recomputed hashes
alone are not an authenticity mechanism. A missing or altered plan is therefore an invalid new run
bundle. Legacy bundles without a manifest plan declaration remain readable.

## Contract-First Rule

The canonical object is not a NumPy array and not a Torch tensor. The canonical object is the operation
contract.

Each operation contract describes:

- operation ID and stable role in the DAG;
- input and output artifact kinds;
- parameter schema and defaults;
- stable boundary formats, dtypes, shapes, and units;
- rate/accounting fixed points;
- differentiability metadata;
- supported runners and backends;
- equivalence class across materializations;
- timing and memory metric categories;
- artifact metadata needed to interpret outputs.

Runtime values may be NumPy arrays, Torch tensors, JSON payloads, bytes, paths, native handles, or
external artifacts. They are valid only if they satisfy the contract for the boundary they cross.

## Runner Responsibilities

### Benchmark Runner

The benchmark runner is the official evidence path. It prioritizes reproducibility, inspectable
artifacts, exact rate accounting, and manifest completeness.

It may use mixed implementations: NumPy, Torch, C++, ONNX Runtime, OpenVINO, Sionna, external codec
repositories, and subprocesses. That is acceptable as long as each materialization is declared and the
run manifest records the actual backend used.

Benchmark runs record:

- recipe SHA;
- expanded execution plan or operation contract metadata;
- materialization backend per step;
- environment and dependency evidence;
- fixed bit/symbol/accounting boundary metadata;
- artifacts with hashes, dtypes, shapes, units, and interpretation metadata;
- metrics with finite values and units;
- dataset, channel, benchmark, and seed policy metadata.

### Dataset Capture Runner

The dataset capture runner is artifact-based but dataset-oriented. It executes a recipe repeatedly or in
batches, reads declared taps, and writes aligned records into dataset shards. Canonical
`metadata.matrix` points are planned before output-directory changes and cycled deterministically;
run records retain `matrix_selection`, `matrix_index`, and `matrix_variant_id`. The older
`dataset_capture.sweep` input remains a one-way compatibility boundary and cannot be combined with a
canonical recipe matrix.

Capture preserves the same boundary contracts as benchmark mode. For example, channel payload
bits remain unpacked `np.uint8` values `0` or `1`, and received symbols remain `complex64` or a
declared equivalent. Capture datasets are training evidence, not final benchmark evidence.

### Training/Export Runner

The training/export runner compiles a differentiable subgraph or harness. It uses Torch/Sionna
or another differentiable backend only for frozen downstream support on a selected replacement-to-loss
route. The operation currently installed at the replacement boundary is substituted, so its own
gradient/export metadata is not a prerequisite. Blocks that only generate captured supervised inputs
or targets are executed by the capture runner and need no gradient materialization.

Differentiable export does not silently reinterpret non-differentiable frozen support as
differentiable. JPEG, CRC, hard demodulation, packet failure, hard entropy decoding, and byte container
parsing remain gradient breaks when they lie downstream of a replacement on the chosen live task-loss
route, unless an operation explicitly declares a surrogate or differentiable materialization. They
are not blockers merely because they occur before a replacement, feed a capture, or are themselves
the operation being replaced.

The exported harness records the source recipe, selected subgraph, selected materializations,
loss, seed/channel settings, and adapter return path. A checkpoint trained from an export must return
to Noema as an adapter and be evaluated again by the benchmark runner before it is treated as a
comparable result.

## Equivalence Classes

Materializations are not assumed to produce identical outputs. Each operation with multiple
materializations declares an equivalence class.

### Exact

Outputs must match exactly.

Examples:

- bit packing and unpacking;
- bytes-to-bits and bits-to-bytes;
- CRC computation;
- deterministic identity bit link;
- deterministic rate/accounting calculations.

### Numerical

Outputs match within a declared tolerance.

Examples:

- power normalization;
- constellation mapping;
- deterministic floating-point transforms;
- PyTorch vs ONNX/OpenVINO inference for supported model exports.

### Statistical

Outputs follow the same declared distribution rather than matching sample-by-sample.

Examples:

- AWGN samples;
- Rayleigh fading;
- random interference;
- stochastic channel sweeps.

Tests compare distribution properties such as measured noise variance, output power, estimated
SNR, BER curves, or channel response statistics.

### Behavioral

The operation satisfies the same task contract but may not produce identical outputs.

Examples:

- generative receivers;
- VQA or captioning models;
- diffusion-based reconstruction;
- LLM-assisted semantic reconstruction;
- black-box external adapters.

Behavioral equivalence is acceptable for task benchmarks only when the metric, dataset, prompt/input
contract, and benchmark protocol are explicit.

## Boundary Rules

Shared boundaries are where Noema prevents drift between materializations.

Important canonical boundaries include:

```text
payload.bits       unpacked uint8 bits, one bit per element, values 0 or 1
channel.bits       channel-coded unpacked uint8 bits, one bit per element, values 0 or 1
channel.symbols    complex64 or declared real-valued symbol tensor/array
rx.symbols         received complex64 or declared real-valued symbol tensor/array
llr                floating soft reliability values with declared sign convention
image.tensor       declared layout, dtype, value range, and color space
text.utf8_bytes    exact UTF-8 byte payload
metric.scalar      finite numeric value with unit and direction
artifact.file      path with hash and interpretation metadata
```

Runner-specific internal values may differ, but values crossing these boundaries must satisfy the
declared contract. This is especially important for rate accounting, channel accounting, BER, packet
success, and reconstruction metrics.

## Disabled Channel Modes

A disabled channel must be explicit about what is disabled.

### Bit-Perfect Bypass

```text
payload bits -> identity bit link -> payload bits
```

This is exact, useful for clean digital evaluation, and non-differentiable because it operates on
discrete bit decisions.

### Identity PHY

```text
bits/logits -> null or identity modulator -> continuous symbols
continuous symbols -> identity symbol channel -> continuous symbols
continuous symbols -> null, soft, or hard demodulator -> bits/logits
```

This preserves the physical-layer skeleton and can be gradient-friendly through the continuous symbol
region. Hard decisions remain gradient breaks unless a soft/surrogate materialization is selected.

The UI and manifests label these modes differently. Recipes cannot hide a non-gradient bit bypass
behind a label that implies differentiable physical transport.

## Why Not Make Everything Torch?

Noema does not use Torch as a universal runtime value format.

Reasons:

- classical codecs, entropy coding, CRC, packet failure policies, byte streams, files, manifests, and
  external binaries are not naturally Torch operations;
- storing benchmark evidence requires files, hashes, schemas, metrics, and manifests, not only live
  tensors;
- Torch as a mandatory base dependency would make lightweight benchmark inspection and verification
  heavier;
- small discrete CPU operations may be simpler or faster in NumPy, Python, C, or a native library;
- using Torch tensors for discrete operations does not make those operations differentiable.

Native Torch is the dependency-light in-process differentiable materialization. The supported
Sionna 2.x PHY package is PyTorch-based, and Noema registers its AWGN and flat-fading modules for
the corresponding differentiable-export regions; Sionna also remains first-class for declared
benchmark/capture regions. Discrete mapping and hard decisions still stop gradients, and operations
outside the registered combinations must not be inferred to have a differentiable Sionna path.

## Why Not Keep Independent Benchmark And Training Systems?

Independent systems would be easier to implement initially but would undermine scientific credibility.
The same method could train under one channel normalization and benchmark under another, or use a
different demapper, rate count, randomization policy, or preprocessing path.

Noema therefore requires shared contracts and conformance tests wherever an operation has more than
one materialization.

## Conformance Coverage

Operations with multiple materializations require equivalence metadata and focused conformance
checks where the implementations overlap.

Current low-level conformance targets are:

- bit boundary and bit packing/unpacking: exact;
- symbol boundary and symbol identity: exact or numerical;
- power normalization: numerical;
- AWGN: statistical;
- QPSK/PAM mapper: numerical;
- soft demapper: numerical;
- rate and channel-use accounting: exact.

The unit-level suite in `tests/test_backend_conformance.py` checks:

- bit packing/unpacking exactness, including native dataplane when available;
- symbol identity exactness across artifact and Torch identity materializations;
- Torch power normalization against a NumPy reference;
- AWGN noise statistics across artifact NumPy and Torch materializations;
- QPSK mapper convention equivalence between artifact NumPy and Torch;
- QPSK demapper hard-bit and max-log LLR convention equivalence;
- exact rate-accounting fixed points for transmitted bits, symbols, and bits-per-symbol.

## Operational Requirements

The implemented architecture:

1. records backend, materialization, and equivalence metadata in operation contracts;
2. keeps benchmark execution artifact-based and mixed-backend;
3. limits tensor-based differentiable export to differentiable subgraphs;
4. validates shared boundaries for bits, symbols, images, text, metrics, and artifacts;
5. distinguishes bit-perfect bypass from an identity PHY;
6. records selected materializations in run manifests and exported training bundles;
7. tests shared low-level operations before relying on backend substitutions;
8. shows backend, gradient, and equivalence information without changing recipe semantics.

## Scope Boundaries

This architecture does not require:

- rewriting all existing operations to Torch;
- removing NumPy or artifact-based execution;
- making Sionna a mandatory dependency;
- turning Noema into a full training framework;
- guaranteeing sample-identical stochastic channels across all backends;
- treating differentiable-export results as publishable benchmark results without frozen re-evaluation.

## Summary

Noema is contract-first, not NumPy-first or Torch-first.

Benchmarking needs durable evidence. Training needs differentiable in-memory graphs. Capture needs
dataset generation. Those runners can coexist if they share one recipe contract, declare their
materializations honestly, enforce canonical boundaries, and test equivalence where multiple backends
claim to implement the same operation.
