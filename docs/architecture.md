# Architecture

The platform is intentionally CLI-first. The implemented dashboard reads operation contracts,
recipes, graphs, and run summaries from the same backend instead of maintaining a separate workflow
model.

## Planes

- **Control plane**: CLI, operation registry, recipe validation, graph rendering, run storage.
- **Execution plane**: local DAG executor with sequential defaults, opt-in bounded branch
  parallelism, cooperative cancellation, and process-local execution-plan caching; subprocesses,
  containers, and remote workers are not part of the current runtime.
- **Artifact plane**: datasets, images, semantic indices, bits, channel outputs, reports.
- **Communication plane**: semantic encoders/decoders, channel encoders/decoders, modulation,
  wireless channel models.
- **Measurement plane**: image quality, bit error rate, invalid semantic index rate, task accuracy,
  and task-specific perceptual metrics.
- **Reproducibility plane**: run manifests, recipe fingerprints, operation contracts, environment
  evidence, git state, deterministic seed policy, and artifact hashes.
- **Research object plane**: normalized dataset, task, metric, and benchmark specs inferred from
  recipes or supplied explicitly through recipe metadata.

## Operation Rule

Every block is a typed operation:

```text
source.image_dataset           image.batch.numpy
source.local_npz_images        image.batch.numpy
source.kodak_files             image.files
model.toy_vqvae_encode         image.batch.numpy -> semantic.indices.numpy
noise.representation_indices         semantic.indices.numpy -> semantic.indices.numpy
channel.indices_to_bits        semantic.indices.numpy -> channel.payload_bits.numpy
channel.identity_encoder       channel.payload_bits.numpy -> channel.coded_bits.numpy
channel.repetition_encoder     channel.payload_bits.numpy -> channel.coded_bits.numpy
modulation.digital_modulate    channel.coded_bits.numpy -> channel.symbols.complex_numpy
wireless.channel               channel.symbols.complex_numpy -> channel.rx_symbols.complex_numpy
demodulation.digital_demodulate channel.rx_symbols.complex_numpy -> channel.demod_bits.numpy + channel.llr.numpy
channel.identity_decoder       channel.demod_bits.numpy -> channel.payload_bits.numpy
channel.repetition_decoder     channel.demod_bits.numpy -> channel.payload_bits.numpy
channel.bits_to_indices        channel.payload_bits.numpy -> semantic.indices.numpy
wireless.digital_link          channel.payload_bits.numpy -> channel.payload_bits.numpy
metrics.bit_error_rate         bit streams -> metrics.report
model.toy_vqvae_decode         semantic.indices.numpy -> image.batch.numpy
metrics.image_reconstruction   image.batch.numpy + image.batch.numpy -> metrics.report
```

This keeps source semantic perturbation, receiver representation perturbation, and channel noise visible in the graph.
Researchers can replace one block without rewriting the whole pipeline.

Each operation contract may also declare differentiability metadata. Operations that do not declare
metadata are treated as NumPy/artifact-only blocks:

```yaml
differentiability:
  framework: numpy        # torch | sionna | tensorflow | numpy | blackbox | none
  gradient: none          # full | stop | surrogate | none
  trainable_params: false
  exportable: false
  reason: optional explanation
```

Training/export lint uses this field to explain whether gradients can pass through an operation when
it remains as frozen support and whether a PyTorch/Sionna harness can materialize it. Built-in
fine-tuning is declared independently by `Operation.fine_tuning_supported` **and** a callable
operation-owned `fine_tuning_provider`, then exposed as
`training_capabilities.built_in_fine_tuning`. No bundled operation currently advertises that action;
pretrained BART evaluation is not presented as fine-tuning. The legacy `trainable_params` value is retained only for
contract compatibility. Selecting a Block for replacement is a third, separate question: the current
operation's gradient and exportability do not constrain its replacement. A Block has **Portable
replacement** only when it exposes a complete validated trained-artifact ABI.

For a live task-loss graph, only frozen downstream support between the selected replacement's output
and the loss must be differentiable and exportable. Capture-backed supervised training requires no
recipe gradient route at all. For example, JPEG, entropy coding, hard modulation/demodulation, CRC
checks, and packet checks remain non-differentiable, but they may still participate in capture or be
the current implementation at a typed replacement boundary.

Operation contracts also declare backend materialization metadata. These fields implement the
contract-first rule described in [Backend Materialization Architecture](architecture/backend_materialization.md):

```yaml
backends:
  benchmark: [numpy, sionna]
  capture: [numpy, sionna]
  differentiable_export: [torch, sionna]
equivalence:
  type: exact | numerical | statistical | behavioral
  tolerance: optional
formats:
  artifact: npz
  tensor: torch.Tensor
materializations:
  - runner: benchmark
    backend: numpy
    implementation: default
    status: implemented
```

If an operation does not declare these fields, Noema reports only a generic Python dispatch label for
benchmark run/dataset capture, no differentiable-export backend, behavioral equivalence, and
operation-defined artifact formats. The generic label is not a claim that NumPy, PyTorch, or any
other numerical backend implements the operation. Noema names a concrete backend or claims exact,
numerical, or statistical equivalence only when explicit metadata supports it.

The `wireless.digital_link` operation remains as a shortcut for smoke tests. Research recipes use
the explicit encoder/modulator/channel/demodulator/decoder chain when they need visible channel
coding, coded BER, payload BER, code rate, symbol count, and approximate LLR artifacts.

## Artifact Kinds

The channel path uses deliberately specific artifact kinds:

```text
channel.payload_bits.numpy       information bits before/after channel coding
channel.coded_bits.numpy         protected bits after a channel encoder
channel.symbols.complex_numpy    transmitted complex baseband symbols
channel.rx_symbols.complex_numpy received/equalized complex symbols
channel.demod_bits.numpy         hard demodulator decisions
channel.llr.numpy                approximate soft reliability values
```

Those names are the adapter boundary for libraries such as Sionna and custom research code.

## Canonical Boundary Contracts

Noema uses reusable boundary contracts to keep tasks from inventing incompatible data conventions.
The core validators live in `noema_lab.core.boundaries` and are used by the bit/symbol fixed-point
operations.

| Contract | Rule |
| --- | --- |
| `payload.bits` | 1-D `np.uint8`, unpacked one bit per element, values exactly `0` or `1` |
| `channel.bits` | same physical representation as `payload.bits`, used after channel coding/demodulation/fixed points |
| `channel.symbols` | finite `complex64` baseband symbols, or finite `float32` real-valued symbols for explicit real PHY surrogates |
| `image.tensor` | `[N, H, W, C]`, `uint8` in `[0,255]` or `float32` in `[0,1]`, `C in {1,3,4}` |
| `text.utf8_bytes` | valid UTF-8 byte sequence, either Python bytes/string or 1-D `uint8` byte array |
| `semantic.embedding` | finite `float32` `[N, D]` embedding batch |
| `metric.scalar` | finite numeric scalar, with non-negativity enforced for rates/counts/latencies where appropriate |
| `artifact.file` | existing file; recorded SHA-256 must match when present |

Boundary operations write the contract ID into artifact metadata, for example
`boundary_contract: channel.bits` or `boundary_contract: channel.symbols`. Rate accounting points use
the same strict path: bit counts, byte counts, symbol counts, channel uses, pixel counts, and sample
counts must be finite, non-negative, and integer-valued unless the metric explicitly declares a
continuous unit.

## Wireless Channel Rule

Wireless simulation stays behind the same typed symbol boundary:

```text
channel.symbols.complex_numpy -> wireless.channel -> channel.rx_symbols.complex_numpy
```

The operation contract exposes realistic presets without changing codec recipes:

```text
awgn                  complex AWGN baseline
flat_rayleigh         flat fading with perfect one-tap equalization
interference_awgn     AWGN plus co-channel interference
mimo_flat             flat MIMO with perfect-CSI equalization
ofdm_tdl              OFDM-grid tapped-delay-line emulation
ofdm_cdl              clustered-delay-line inspired OFDM emulation
urban_micro           mobility/interference-oriented urban microcell preset
```

Every run records the selected preset, backend, SNR, MIMO antenna counts, OFDM grid parameters,
interference settings, and mobility metadata in the channel artifact. The built-in `numpy` backend
is the portable deterministic baseline. The optional `sionna` backend is registered only for the
presets that have real adapters today; unsupported Sionna combinations raise an operation error
rather than silently using another backend. Native Sionna 2/PyTorch TDL-OFDM is available through
the explicit `wireless.ofdm_channel_state` artifact and matched `wireless.channel` path; the AI-PHY
and CSI-feedback operations also expose registered Sionna TDL-OFDM realizations. Direct Sionna CDL,
urban-micro, and interference presets, ns-3, GNU Radio, and hardware-in-the-loop are not included in
the current runtime; the same backend-specific adapter contract can accommodate them.

## Codec Timing Vocabulary

Run summaries may keep implementation-stage timings such as `encoder.entropy_encode`,
`encoder.bitstream_encode`, `decoder.entropy_decode`, or `decoder.bitstream_decode`, but
comparative UI metrics collapse them into canonical buckets:

```text
encoder.inference       neural/model/transform/quantizer work on the sender side
encoder.payload_encode  internal representation -> transmitted payload bits/bytes
decoder.payload_decode  transmitted payload bits/bytes -> internal representation
decoder.inference       neural/model/transform/reconstruction work on the receiver side
```

Entropy coding is therefore a subtype of payload coding, not the generic name for the whole
payload step. EF-LIC fixed-length index packing and JPEG byte/container writing also count as
payload coding while keeping their own implementation-stage labels for detailed inspection.

## Recipe Rule

Recipes are ordered DAGs. Inputs always point backward with `step_id.output_name`, for example:

```yaml
inputs:
  bits: channel_encoder.bits
```

The CLI validates:

- recipe fields, explicit step IDs, and the exact schema generation in strict validation mode
- each operation exists and is runnable
- all required inputs are connected
- connected artifact kinds are compatible
- parameters match the operation schema
- the declared standard execution profile matches the graph topology

The executor also runs the same registry validation before creating a run directory. This keeps CLI,
dashboard, and direct Python usage on the same safety path. Purpose, execution topology, and
concrete blocks are intentionally separate: `metadata.research.task` tags what is evaluated,
`execution_profile` says which stable execution spine is claimed, and `steps` say what actually runs.
The full ownership and template rules are defined in the [schema reference](reference/schemas.md).

Runtime scheduling is separate from recipe semantics. Runs are sequential by default; an explicit
worker count can overlap dependency-independent branches while keeping durable step evidence in
plan order. Cancellation is cooperative and flows through `OperationContext`. The cache,
scheduler, safety boundary, CLI/API controls, and evidence fields are defined in
[Execution Runtime](architecture/execution_runtime.md).

## Template And Matrix Boundary

The backend owns template discovery through `recipe_templates.yaml`. Its catalog maps stable
template IDs to purpose IDs, editing hints, source recipes, and expected execution profiles,
including exactly one supported default for each purpose it exposes. `GET /api/recipe-templates`
returns the ordered entries with availability and validation evidence. The dashboard combines that
catalog with `GET /api/recipes`, deduplicates shared sources, and presents collapsible purpose
groups. The purpose is read-only semantic metadata for validation and result routing; it never
selects or rewires blocks. Filename conventions and frontend purpose-to-template tables are not
part of the contract. `POST /api/recipe-templates/instantiate` resolves a stable ID,
prefers an explicit project override, otherwise reads the packaged starter, applies typed parameter
overrides, and strictly validates the resulting standalone recipe in one call. Once selected, a
template has no special execution privileges. Its provenance records the exact catalog entry,
source, and override digests. Editor and working-copy flags stay in the browser tab wrapper rather
than executable metadata, and browser-side builders cannot replace catalog-owned blocks or inputs.
Catalog declarations are package-cached, while validation evidence is re-inspected against mutable
project recipes on each request.

Parameter spaces follow the same compile-to-an-ordinary-recipe rule. `metadata.matrix` declares
dimension lists and their `steps[].params` bindings. Expansion produces concrete recipes whose
coordinates and deterministic ordinal live in `metadata.matrix_selection` and
`metadata.matrix_index`; `metadata.matrix_variant_id` supplies a stable, type-sensitive coordinate
identity. Planning strictly validates every concrete point, and single-run entry points reject an
unresolved definition before creating a run directory. CLI, dashboard, benchmark, and dataset
capture paths all consume the same concrete-recipe service rather than running separate sweep
engines. Benchmark packs also use `params.matrix_selection` for an already-materialized
comparison point. If the source recipe owns matching dimensions, loading the benchmark materializes
their parameter bindings and rejects disagreement with explicit overrides before dropping the
authored matrix definition. Legacy `sweeps`, `ui_sweeps`, and benchmark `sweep_values` fields are
read for v1 compatibility, but new writes use the matrix names.

## Run Manifest Rule

Every run directory contains:

```text
recipe.authored.json  normalized author choices for this run variant
recipe.json           default-expanded recipe payload actually executed
execution-plan.json   immutable runner/backend/implementation bindings
summary.json          compact status, metrics, and UI-facing step outputs
manifest.json         reproducibility evidence and artifact ledger
```

`manifest.json` is the durable run-evidence record. It stores:

- canonical recipe SHA-256
- operation contracts for every operation used by the recipe
- normalized research specs for dataset, task, metrics, and benchmark
- Python/platform/dependency versions
- `pyproject.toml` and `uv.lock` hashes
- native dataplane availability
- git commit, branch, dirty state, and diff stat when available
- seed policy and master seed source
- produced artifact ledger with kind, relative path, SHA-256, dtype, shape, and metadata

If a recipe defines `metadata.seed`, `metadata.experiment_seed`, or `metadata.ui_seed`, random
operations can derive independent streams from the master seed. A step-level `params.seed` remains
an explicit override. The derivation is intentionally stable:

```text
sha256(master_seed | recipe_name | step_id | stream) mod 2^31-1
```

## Research Object Rule

Noema defines stable research objects that benchmark packs build on:

```text
DatasetSpec    dataset identity, modality, source operation, split/version, parameters
TaskSpec       objective such as reconstruction, classification, VQA, ASR, or retrieval
MetricSpec     metric identity, unit, direction, family, and reduction
BenchmarkSpec  named dataset + task + channel + metric bundle
```

Recipes can provide these explicitly under `metadata.research`, or Noema can infer a conservative
view from the operation graph. For example, a recipe with `source.image_dataset` and
`metrics.image_reconstruction` is normalized as an image reconstruction task over the selected image
dataset. The CLI exposes this with:

```bash
noema recipe specs recipes/compressai_kodak_default.yaml
```

## Research Catalog Rule

Datasets, tasks, and metrics are promoted from inferred metadata into a machine-readable catalog:

```text
src/noema_lab/research_catalog.yaml
  datasets[]  id/name/modality/status/source_ops/versions
  tasks[]     id/name/kind/modality/status/required_artifacts/metrics
  metrics[]   id/name/family/unit/direction/reduction
```

The catalog is not meant to block custom research. Unknown datasets and metrics are allowed as
warnings unless strict validation is requested. Known catalog entries are checked for task kind,
modality, dataset modality, and task-metric compatibility. The CLI exposes the catalog with:

```bash
noema research catalog
noema research tasks
noema research metrics
noema research validate-recipe recipes/compressai_kodak_default.yaml
```

The dashboard API exposes the same data at `/api/research/catalog`. Benchmark validation includes
catalog validation in its JSON output, and new benchmark result bundles embed the declared
dataset/task/metric contract under `result.json -> benchmark`.

## Text Task Rule

The text workflow keeps the same artifact and channel contracts as other tasks:

```text
source.text_dataset                text.batch.json
noise.source_text_perturbation                text.batch.json -> text.batch.json
model.text_utf8_encode             text.batch.json -> channel.payload_bits.numpy
channel.*                          canonical unpacked uint8 bits
model.text_utf8_decode             channel.payload_bits.numpy -> text.batch.json
metrics.text_semantic_similarity   text.batch.json + text.batch.json -> metrics.report
```

The initial text metrics are deterministic lexical/task metrics:

```text
semantic.lexical_similarity   token-F1 lexical-overlap proxy (not embedding-based semantic similarity)
text.unigram_bleu_proxy       sentence unigram precision with brevity penalty (not corpus BLEU)
text.edit_similarity          normalized Levenshtein similarity
text.exact_match              literal case-, punctuation-, and whitespace-sensitive exact-match fraction
```

Embedding metrics such as BERTScore are not implemented by this lexical evaluator. They require
separate metric adapters with explicit model, dependency, and runtime contracts.

The reference smoke presets live in `benchmarks/text_semantic_presets_v1.yaml` and are documented in
`docs/text_task.md`. They are workflow examples, not canonical publication evidence. The important
compatibility rule is that literal `[MASK]` receiver repair is
valid only for bit-preserving text codecs such as raw UTF-8. Neural/generative symbol codecs such as
BART JSCC-lite must not rely on `[MASK]` surviving decode; use symbol/channel noise and semantic text
metrics for those experiments.

## Foundation And Generative Layer

The typed foundation layer preserves the same recipe and operation engine. A runnable path is text +
KB + SemanticState:

```text
source.text_dataset                         text.batch.json
foundation.knowledge_base                   foundation.kb.json
foundation.text_semantic_state_encode       text.batch.json -> semantic.state.json
foundation.semantic_state_ground            semantic.state.json + foundation.kb.json -> semantic.state.json
foundation.semantic_state_payload_encode    semantic.state.json -> channel.payload_bits.numpy
channel.*                                   canonical unpacked uint8 bits
foundation.semantic_state_payload_decode    channel.payload_bits.numpy -> semantic.state.json
foundation.semantic_state_to_text           semantic.state.json -> text.batch.json
metrics.text_semantic_similarity            text.batch.json + text.batch.json -> metrics.report
metrics.semantic_state_faithfulness         semantic.state.json + semantic.state.json + foundation.kb.json -> metrics.report
```

`SemanticState` is a JSON artifact with `kind: semantic.state`, `modality`, and per-example
`states[]` containing concepts, entities, facts, optional source text, and KB grounding matches.
`foundation.kb.json` is a versioned knowledge-base artifact with normalized fact triples. Payload
transport is deliberately explicit: SemanticState JSON is serialized to UTF-8 bytes and then to
canonical unpacked `uint8` bits before channel coding/modulation, so BER/rate accounting uses the
same fixed bit boundary as image and text byte payloads.

The first faithfulness metrics are deterministic and dependency-light:

```text
faithfulness.concept_precision / recall / f1
faithfulness.entity_precision / recall / f1
faithfulness.fact_precision / recall / f1
faithfulness.unsupported_assertion_rate
faithfulness.fact_omission_rate
faithfulness.kb_fact_precision
```

The two fact-error rates are explicitly reference-relative constructs. Unsupported assertion divides
candidate facts absent from the supplied reference by the candidate fact count and is zero for an
empty candidate; omission divides missing reference facts by reference fact count and is zero for an
empty reference. Neither is labeled a real-world hallucination rate. SemanticState IDs must be
non-empty, unique, and identical across the paired inputs.

Foundation adapter contracts are registered as ordinary operations:

```text
foundation.clip_text_embed       text.batch.json -> foundation.embedding.numpy
foundation.clip_image_embed      image.batch.numpy -> foundation.embedding.numpy
foundation.sam_segment           image.batch.numpy -> semantic.state.json
foundation.vlm_image_to_state    image.batch.numpy -> semantic.state.json
foundation.diffusion_state_to_image semantic.state.json -> image.batch.numpy
```

The local CLIP/SAM/VLM-style implementations are deterministic baselines for smoke testing and
debugging artifact contracts. Heavyweight CLIP, SAM, VLM, or diffusion backends connect through
runtime-specific adapters with explicit model/checkpoint parameters. Embedding comparison and
retrieval require both inputs to declare and agree on an embedding-space provenance record covering
the backend, model, revision status, preprocessing contract, and dimensions; a free-form metric label
cannot substitute for that agreement. The `foundation` optional extra
declares the expected Python ecosystem dependencies, but no adapter silently downloads checkpoints
or falls back to a fake generative output.

## Task-Oriented Benchmark Rule

Downstream task success is a first-class benchmark contract. The CLI and catalog support these task
IDs:

```text
classification
visual_question_answering
object_detection
segmentation
image_text_retrieval
```

The evaluator operations are intentionally model-agnostic:

```text
metrics.classification   task.labels.json + task.predictions.json -> metrics.report
metrics.vqa              vqa.answers.json + vqa.answers.json -> metrics.report
metrics.detection        vision.detections.json + vision.detections.json -> metrics.report
metrics.segmentation     vision.segmentation_mask.numpy + vision.segmentation_mask.numpy -> metrics.report
metrics.retrieval        retrieval.rankings.json -> metrics.report
```

`metrics.captioning` remains available for typed caption artifacts, but `image_captioning` is not
advertised as runnable without a real captioning receiver or model adapter.

This separates task results from the model adapters that produce the predictions. A compatible VQA
model, detector, segmenter, captioner, or CLIP retrieval model emits the corresponding typed
artifact; the benchmark and report layer remains unchanged.

## Semantic Artifact Rule

The following typed semantic objects are exchanged by task-oriented and foundation-model recipes:

```text
vision.embedding.clip.numpy
multimodal.embedding.numpy
vision.detections.json
vision.segmentation_mask.numpy
vision.scene_graph.json
vision.semantic_map.json
text.caption.json
vqa.answers.json
retrieval.rankings.json
semantic.importance_map.numpy
video.frame_sequence.numpy
```

Model adapters produce these artifacts directly rather than hiding semantic objects inside opaque
JSON blobs. The smoke operation `source.semantic_artifacts_smoke` emits all of them and the
recipe `recipes/semantic_artifacts_smoke.yaml` verifies the executable task-metric subset.

## Benchmark Pack Rule

Benchmark packs are versioned YAML or JSON documents under `benchmarks/`. They do not replace
recipes; they group existing recipes into a comparable suite:

```text
BenchmarkPack
  id/version/name
  dataset/task/metrics/baselines
  recipes[]
    id/label/role/path
```

This is also the higher-level **documentation demo/campaign** object. A showcase such as “learned
water filling versus equal power and the theoretical oracle” is one benchmark pack with three
method entries, not three public scenario templates. A method entry may point to the same canonical
scenario recipe with typed block overrides, or to a frozen materialized recipe when an exact paper
artifact must be preserved. The result bundle records every resolved recipe hash and run manifest
automatically.

The layers have separate jobs:

- a Template Recipe helps a researcher start from the right scenario topology;
- Workbench derives capture and training from a compatible method slot in that scenario;
- a benchmark pack fixes the methods, coordinates, metrics, and evidence used for comparison;
- a stored benchmark result is the read-only webpage or documentation demo.

The benchmark runner validates each referenced recipe, checks that the recipe's normalized
`DatasetSpec` and `TaskSpec` match the pack when those fields are declared, then runs each recipe
through `LocalExecutor`. Results are written under:

```text
.noema/benchmarks/<result_id>/benchmark.json
.noema/benchmarks/<result_id>/result.json
.noema/benchmarks/<result_id>/metrics.csv
.noema/benchmarks/<result_id>/recipes.csv
.noema/benchmarks/<result_id>/summary.md
```

`result.json` is the benchmark result schema. It records benchmark identity, pack hash, each recipe's
run id, manifest path, normalized research specs, status, and flattened metrics. This keeps
benchmarking reproducible while individual recipe runs retain their full manifest. The CSV and
Markdown files are generated reports derived from `result.json`; they can be regenerated with
`noema benchmark export <result_id>`.

## External Adapter SDK

External researcher models can be registered without editing `src/noema_lab/ops`. An adapter folder
provides a `noema_adapter.yaml` manifest and Python callables. The manifest defines operation
IDs that wrap one of the built-in external contracts, such as `model.external_encode_bits`,
`model.external_encode_indices`, `model.external_encode_latents`, or `model.deepjscc_external_encode`.

The registry loads those manifests through:

```bash
noema --adapter adapters/my_codec ops list
noema --adapter adapters/my_codec recipe run recipes/my_recipe.yaml
noema --adapter adapters/my_codec ui serve --port 8766
```

The SDK keeps bit payloads on the same fixed channel contract as built-in models:
`np.uint8` unpacked bits, one bit per element, values `0` or `1`. Packed-byte adapters can still be
used by declaring `bit_storage: packed_bytes`; Noema unpacks before channel transmission and repacks
before the external decoder. See [external_adapter_sdk.md](external_adapter_sdk.md) for the manifest
schema and callable contract.

## Adapter Direction

The likely next adapter steps are:

- add CLIC image decoding and resizing
- wrap a pretrained VQ-VAE/VQGAN encoder/decoder
- extend native Sionna OFDM coverage beyond the bound TDL-state path and add CDL adapters
- add ns-3, GNU Radio, or hardware backends behind the same symbol-channel contract
- broaden the current Sionna NR-LDPC transport-block path with Polar codes and more soft-decision receivers

## Dashboard Boundary

The dashboard calls the same APIs used by these CLI operations:

```bash
noema ops list --json
noema ops show wireless.digital_link --json
noema recipe validate recipes/compressai_kodak_default.yaml
noema recipe graph recipes/compressai_kodak_default.yaml --format json
noema recipe run recipes/jpeg_kodak_smoke.yaml
noema recipe run recipes/jpeg_q75_kodak_repetition_awgn.yaml
noema recipe expand-matrix recipes/deepjscc_kodak_awgn_train.yaml
noema recipe run-matrix recipes/deepjscc_kodak_awgn_train.yaml
noema template instantiate semantic_comm.image_reconstruction.default
noema runs show <run_id>
noema ui serve
```

That makes the graph editor a visual recipe editor, not a separate experiment engine.

The dashboard server lives under `noema_lab.ui` and calls the same Python contracts as the CLI:

```text
GET  /api/ops
GET  /api/recipes
GET  /api/recipe-templates
POST /api/recipe-templates/instantiate
GET  /api/recipe/graph?path=...
GET  /api/recipe/matrix?path=...
POST /api/recipe/matrix/expand
POST /api/recipe/run
POST /api/recipe/run-payload
POST /api/recipe/save
GET  /api/runs
GET  /api/runs/<run_id>
GET  /api/artifact?path=...
GET  /api/image?path=...
```

The graph canvas is interactive in the browser: nodes are selectable and draggable. Recipes are
selected as tabs, and a recipe can be duplicated into an editable in-browser copy. Selecting a block
edits the loaded recipe directly, and `POST /api/recipe/run-payload` runs the edited recipe without
saving a copy first. The Results view follows the selected recipe's latest run and keeps visual
artifacts, measured metrics, recipe settings, and artifact records in separate panes so channel
parameters such as SNR do not get confused with quality metrics. The current editor intentionally
supports block selection, parameter editing, duplication, and execution, but not manual edge editing.
