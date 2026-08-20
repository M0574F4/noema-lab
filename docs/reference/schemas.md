# Schema And Contract Reference

Noema separates *what is being evaluated* from *how it is executed*. A recipe is the compiled
experiment instance; a template is only a reusable way to create that instance.

## Recipe Field Ownership

| Concern | Canonical owner | Purpose |
| --- | --- | --- |
| Schema generation | `schema_version` | Selects the recipe contract. It is an integer, not a task or template version. |
| Research purpose | `metadata.research.task` | Tags reconstruction, classification, VQA, retrieval, and other scored outcomes for validation and result presentation. It does not select blocks. |
| Execution topology | `execution_profile` | Declares a versioned executable spine such as `layered_digital` or `range_localization`; it is not a task label. |
| Executable graph | `steps[]` | Orders operation instances and connects typed outputs to inputs. |
| Block configuration | `steps[].params` | Configures one operation instance and is validated by that operation's parameter schema. |
| Experiment-wide protocol | `metadata` | Holds seeds, measurement policy, provenance, and other values that apply across steps. |
| Dataset-capture policy | training plan `dataset_capture` | Configures capture taps, partitions, and sampling without changing the scenario recipe. An embedded recipe field is accepted only for legacy files and transient capture jobs. |
| Research-area discovery | research-catalog `task.area_id` | Groups canonical scenarios for exploration without changing recipe semantics. |
| Benchmark ownership | `suite` | Associates a recipe with a protocol family; it is not the template-browser taxonomy. |
| Parameter sweep | `metadata.matrix` | Defines dimensions and binds selected values into `steps[].params`. |
| Training campaign | Workbench `noema.training_plan` | Selects replaceable blocks, capture, and support framework after an ordinary recipe is chosen. Workbench derives reachable evaluation sinks automatically. An external objective may be recorded as provenance, but Workbench does not select it. |
| Cross-recipe comparison | benchmark pack | Selects recipes, fixed overrides, roles, datasets, tasks, and metrics for a comparison. |

`metadata.task_id` remains a v1 compatibility alias. When it is present, it is resolved through the
research catalog and must agree with `metadata.research.task.id`. New recipes should use
`metadata.research.task` when an explicit declaration is needed. If the task can be inferred from a
specific source/metric graph, the declared and inferred task must also agree.

The training-plan fields are intentionally orthogonal: `selected_steps` chooses portable replacement
boundaries; optional `loss_steps` lets advanced CLI workflows narrow recipe route analysis, while
Workbench analyzes all reachable evaluation sinks automatically;
`dataset_capture` defines offline records/splits; and `objective` names an external or optional-demo
loss. An objective such as `image.mse` is never treated as a recipe step ID.

## Purpose, Profile, And Steps Are Orthogonal

These fields answer different questions:

- `metadata.research.task`: **What purpose/outcome is scored?**
- `execution_profile`: **What stable execution spine does this graph claim?**
- `steps`: **Which concrete operations execute, in what order?**
- `steps[].params`: **How is each concrete operation configured?**

For example, image reconstruction can use either a `layered_digital` JPEG graph or a
`joint_source_channel_symbols` DeepJSCC graph. Conversely, a `layered_digital` profile can carry
image, text, or another payload. Combining purpose and profile into one discriminator would duplicate
the graph and make those valid combinations hard to represent.

Standard execution profiles are mandatory topology contracts during planning. `custom` is an
explicit opt-out for graphs that do not claim a standard spine; it may retain `based_on` provenance
after a standard graph is edited. See [Recipe Execution Profiles](../architecture/execution_profiles.md).

## Configurator Presentation

The browser presents every open recipe through one stable hierarchy without adding another recipe
schema:

1. **Overview** resolves the read-only purpose tag, working-copy origin, execution profile, graph
   size, and validation state from the ordinary recipe and its UI-only tab wrapper.
2. **Blocks** is the consistent primary editor. It exposes exact operation IDs, typed inputs,
   operation-schema parameters, wiring, sweeps, and artifact bindings for the current graph.
3. **Recipe Settings** owns recipe-wide identity and execution defaults plus explicit atomic controls
   that must update linked blocks together. Whole-pipeline presets are labeled as replacements and
   require confirmation.

Graph selection is transient browser state and is not written to recipe JSON. A template may retain
an editor hint for compatibility, but opening the same compiled recipe through a formal template,
the project recipe catalog, or a working copy produces the same Overview, Blocks, and Recipe
Settings structure.

Graph nodes permanently display only the canonical block role, a compact configuration summary, and
the input-to-output flow. Technical step IDs remain available in delayed hover details and block
settings. The Workbench **Operation Training Capabilities** table follows the same convention: its
**Block** column uses the graph's canonical role name, while the exact operation has its own column
and the technical step ID is available on delayed hover and in exported contracts. **Built-in
fine-tuning** is an explicit implementation capability (`Operation.fine_tuning_supported` plus a
callable operation-owned `fine_tuning_provider`, exposed as
`training_capabilities.built_in_fine_tuning`). No bundled operation currently advertises that action;
**Portable replacement** instead requires a complete
validated `trained_artifact_abi`. The older `differentiability.trainable_params` field is retained for
contract compatibility but is not a replacement or fine-tuning permission switch. **Gradient** and **Differentiable
support** describe the current operation when retained in a live graph; they are not selection gates
for replacing it.

Presentation routing is a versioned, code-only adapter registry. Adapter matching depends only on
canonical recipe content, with `metadata.ui_configured: false` retained as a compatibility opt-out
to the generic Blocks view. Template IDs, file paths, and tab editor preferences never select an
adapter. The registry contract and facet contract are currently version 1; their versions and
adapter IDs are UI implementation details and are never serialized into recipe JSON.

The header's raw JSON editor is also a working-copy view. Apply sends the authored object to the
strict validation-only endpoint, adopts the normalized response only after successful validation,
and never writes the originating template or recipe file. Copy and paste operate on the editor
draft, so invalid text remains recoverable until the user fixes or cancels it.

## Templates

A template is not a second runtime schema. It is a stored starter recipe or a deterministic builder
that emits an ordinary recipe. After instantiation, validation and execution do not depend on which
template produced it.

Template rules:

- A public template defines a research scenario or genuinely distinct protocol topology, not a
  particular method, training mode, maturity label, smoke test, or saved workspace file.
- Classical methods, learned fallbacks, and returned artifacts that share a topology belong behind
  a stable typed block selector. They are compared as benchmark-pack methods, not duplicated as
  separate templates.
- Choosing a different template replaces the pipeline graph. It must not retain blocks from the
  previous pipeline, while user-authored recipe-wide identity and global protocol fields are
  preserved when replacing the current working copy.
- Purpose is a declared or inferred semantic tag used for catalog grouping, metric validation, and
  result visualization. Editing purpose never inserts, removes, or rewires blocks.
- An ordinary parameter edit preserves the current graph and changes only the selected values.
- Template/source provenance may be kept for the UI, but it must not override recipe semantics.
- Every generated recipe declares an `execution_profile`; task builders do not infer topology from a
  UI label at execution time.
- Shared defaults belong in operation parameter schemas. Templates should specify deliberate
  experiment choices, not copy every operation default.

The resulting object model is deliberately small:

1. **Scenario template** discovers a canonical topology.
2. **Recipe working copy** records one configured method and protocol coordinate.
3. **Training plan** separately selects replacement blocks, capture taps/splits, and any external
   objective for one study.
4. **Workbench training-interface bundle** derives typed tensor and artifact-return contracts from
   the recipe plus that plan.
5. **Benchmark pack** stores the multi-method comparison used by a paper or documentation demo.
6. **Run/benchmark result bundles** preserve hashes, manifests, metrics, and artifacts as evidence.

There is no `training template` or `workspace template` layer. A training campaign must be derived
from any compatible ordinary recipe. A scenario is not advertised as closed-loop trainable until
its selected block has an operation-owned data contract and executable artifact-return ABI.

The recipe and training plan have independent SHA-256 identities. The recipe SHA answers “was the
same runnable scientific scenario used?”; the training-plan SHA answers “were the same replacement
targets, capture policy, route boundary, and framework used?” Export materializes the
overlay only temporarily and writes both identities. Consequently, changing a training choice does
not silently create a different scenario, while changing a recipe block or parameter does.

The backend template catalog in `src/noema_lab/recipe_templates.yaml` owns discovery and default
selection. Each entry declares `id`, `label`, `task_id`, `editor`, `recipe_path`, `order`, `default`,
`status`, the expected `execution_profile`, and optionally a packaged `starter_resource`. Every task
represented in the catalog has exactly one supported default; the catalog does not need to
represent every research-catalog task. Here `task_id` is the catalog's purpose classification, not
a graph-building command. The Template Recipes browser is populated only by
`GET /api/recipe-templates` and presents those curated rows under research area and purpose.
`GET /api/recipes` remains available for explicit file loading, benchmark packs, tests, and saved
research assets, but it does not create a catch-all Workspace Recipes section. The browser must not
rediscover defaults from filename substrings or maintain a second purpose-to-template table.

Instantiation is by stable catalog ID. A file at `<project_root>/<recipe_path>` is an explicit
project override and takes precedence over `starter_resource`; an invalid override is reported and
never silently replaced by the built-in. If the override is absent, the packaged starter makes the
same template usable from an installed wheel without a repository checkout. Resolution, one-time
source reading, strict compilation, operation-registry validation, and task/profile contract checks
are performed together for every call, so prior inspection is not stale authorization to use a
changed file.

An instantiation may apply a typed `step_params` mapping, plus optional `name` and `description`,
before its final strict compilation. Step IDs and parameter names must exist in the starter or its
operation contract; parameter values retain their JSON types and still pass operation-owned schema
validation. The standalone recipe records only canonical provenance, not UI working-copy state:

```yaml
metadata:
  template_provenance:
    schema_version: 1
    kind: noema.recipe_template_provenance
    template_id: ai_phy.pilot_channel_estimation.adapter
    catalog_schema_version: 1
    template_digest: sha256:<digest-of-template-catalog-entry>
    source_kind: packaged_builtin  # or project_override
    source_reference: noema_lab:recipe_starters/channel_estimation_adapter_awgn.yaml
    source_digest: sha256:<digest-of-exact-source-bytes>
    overrides_digest: sha256:<digest-of-canonical-typed-overrides>
```

Use `noema template instantiate <template-id>` from the CLI or
`POST /api/recipe-templates/instantiate` from the dashboard. The browser keeps the active
configurator view, graph selection, working-copy state, and topology-preservation flags in its tab
wrapper; those UI concerns are not
written into the instantiated executable recipe. Template-time editor overrides use explicit
catalog `editor_bindings` (`step_id`, `op`, and `param`), which are checked against the selected
project override or packaged starter before the template is advertised as valid. A local editor
may merge parameter updates into existing step IDs, but cannot replace the catalog-owned operation
IDs, inputs, graph shape, or unrelated catalog-defined parameters.

The inspection response has catalog-level `schema_version`, `status` (`valid`, `degraded`, or
`invalid`), and `templates`. The server re-inspects the referenced project recipes when the endpoint
is requested, so a saved or externally edited template cannot leave process-lifetime validation
evidence stale. Each template row includes `available`, row validation (`valid`,
`unavailable`, or `invalid`), the selected source kind/digest, and a compact resolved-recipe summary.
A missing project override uses the packaged starter; discovery degrades only if neither source is
available. A task, profile, or recipe-contract mismatch is invalid. Multiple
templates may implement the same task only when they represent genuinely different protocols or
topologies—for example layered-digital text transport and continuous-symbol text JSCC. A baseline
operation and a learned replacement inside the same graph are method choices, not two templates.
The `editor` selects a compatible editing surface; it does not change recipe semantics.

## Matrices And Concrete Coordinates

`metadata.matrix` is the canonical authored parameter-space definition. Dimension values are
explicit lists. `step_params` binds dimensions to operation parameters by step ID; a
`{matrix: <dimension>}` reference may appear at any depth in the parameter value.

```yaml
metadata:
  matrix:
    dimensions:
      seed: [1, 2]
      payload.bit_count: [1024, 4096]
    step_params:
      data:
        seed: {matrix: seed}
        bit_count: {matrix: payload.bit_count}
steps:
  - id: data
    op: source.random_bits
    params:
      seed: 0
      bit_count: 1024
```

Expansion computes the Cartesian product of the dimensions, up to 256 concrete points. Unknown
steps or dimensions, unused dimensions, and malformed references are errors. Each concrete recipe
has the selected values applied to `steps[].params`, removes all matrix-definition aliases, and
records `metadata.matrix_selection`, its deterministic `metadata.matrix_index`, and a type-sensitive
stable `metadata.matrix_variant_id` of the form `mxv1-<sha256>`. The ID hashes canonical typed
coordinates, so booleans, integers, and floats remain distinct and harmless recipe renames or
dimension reordering do not change identity. The selection is provenance for one concrete point,
not another sweep definition.

Matrix planning strictly compiles the authored recipe and every concrete point through the same
operation registry. Single-run entry points reject an unresolved matrix before run-directory side
effects; use `noema recipe run-matrix` or the backend matrix-expansion API to execute its concrete
recipes. The dashboard uses those returned recipes verbatim rather than reapplying coordinates in
JavaScript.

The old flat authoring fields `metadata.sweeps` and `metadata.ui_sweeps` are compatibility-only
inputs. Readers normalize them when possible, but new recipes and UI edits write `metadata.matrix`.
When forms coexist, canonical `matrix` wins with a compatibility warning; without it, `sweeps` wins
over `ui_sweeps`. Strict legacy checking rejects either alias, which is useful for enforcing new
authored contracts without breaking v1 reads.

Benchmark recipe entries use the same coordinate name for already-materialized points:

```yaml
recipes:
  - id: jpeg_snr_8
    path: recipes/jpeg_q75_kodak_repetition_awgn.yaml
    params:
      matrix_selection:
        channel.snr_db: 8
      step_params:
        wireless_channel:
          snr_db: 8
```

`params.matrix_selection` is canonical. When the source recipe declares matching matrix dimensions,
the benchmark loader requires a complete in-range selection, applies those bindings to
`steps[].params`, and rejects any conflicting explicit `step_params` override. The instantiated
benchmark recipe then contains the concrete selection and index, but no remaining matrix definition.
It also carries the same stable `matrix_variant_id` used by direct matrix and capture execution.
Additional coordinates that are not dimensions of the source recipe remain typed provenance; their
executable effects must still be declared through `step_params`.

The deprecated benchmark field `params.sweep_values` is accepted when it appears alone and is
normalized to `metadata.matrix_selection` in the concrete recipe. If both names appear, their
mappings must be identical. Concrete recipes never gain a new `metadata.sweep_values` alias.

## Authored, Normalized, And Effective Forms

Recipe compilation exposes three representations:

1. **Raw**: the decoded YAML/JSON mapping supplied by the author.
2. **Normalized**: the author form parsed into typed recipe objects. Compatibility conveniences are
   made explicit, while the author's sparse parameter choices remain sparse.
3. **Effective**: a deep copy of the normalized form with operation-schema defaults recursively
   materialized. This is the form executed and fingerprinted.

Local runs record both views:

```text
recipe.authored.json  normalized author choices for the concrete run variant
recipe.json           effective, default-expanded recipe actually executed
manifest.json         fingerprint and reproducibility evidence for recipe.json
```

This prevents hidden runtime defaults while keeping recipe files readable. A default belongs in one
place—the operation contract—and is copied into the effective run recipe by the compiler.

## Runtime Controls And Evidence

Execution controls are invocation parameters, not recipe fields. `parallel_workers` selects the
local scheduler bound and `use_plan_cache` selects a planning optimization; neither belongs under
recipe `metadata`, `execution_profile`, or `steps[].params`. The CLI, HTTP API, and dashboard pass
them separately when starting a run.

The resulting `summary.json` and `manifest.json` both record `execution.mode` and
`execution.parallel_workers`. Their `execution_plan.cache` records the cache outcome and links it to
the immutable execution plan and authored/effective recipe hashes. Verification accepts older
bundles without these optional P3 fields, but if either copy is present its schema and cross-file
links must be consistent. See [Execution Runtime](../architecture/execution_runtime.md).

## Compilation Modes

`compile_recipe(..., mode="compat")` supports existing v1 inputs. Unknown root or step fields and
implicit step IDs produce diagnostics, and unknown fields are retained so round-trips are lossless.

`compile_recipe(..., mode="strict")` turns those conditions into errors. The CLI `recipe validate`
command and the UI save endpoint use strict compilation. UI saves validate the default-expanded
recipe against the operation registry before atomically replacing a file, so invalid edits do not
damage the previous recipe.

Both modes require an exact supported integer `schema_version` when the field is supplied. Values
such as `0`, `2`, `false`, `"1"`, or `1.0` are rejected.

## Validation Boundaries

Validation is layered deliberately:

- The recipe compiler checks recipe shape, schema version, field ownership, IDs, and references.
- Operation schemas validate each step's local parameters, including nested objects and arrays.
- The planner validates operation availability, required inputs, artifact kinds, research metadata,
  and declared execution-profile conformance.
- Recipe lint adds cross-step research and communication invariants and can return a complete issue
  report without bypassing the planner used for execution.
- Benchmark validation checks comparison-level dataset, task, metric, and override consistency.

## Source-Of-Truth Locations

| Contract | Source of truth | Reflected in docs |
| --- | --- | --- |
| Recipe parsing and compilation | `noema_lab.core.recipes` | This page and Python API reference |
| Recipe-template discovery and instantiation | `recipe_templates.yaml` and `noema_lab.core.recipe_templates` | This page, `template` CLI, and `/api/recipe-templates` |
| Recipe matrices, concrete identities, and legacy sweep reads | `noema_lab.core.matrix` and `noema_lab.core.variants` | This page, `recipe expand-matrix`, and `recipe run-matrix` |
| Task/dataset/metric identities | `research_catalog.yaml` and `noema_lab.core.research` | Architecture and research CLI |
| Execution profiles | `noema_lab.core.execution_profiles` | Execution-profile architecture page |
| Operation inputs, outputs, params, and defaults | operation `describe()` contracts in `src/noema_lab/ops/` | Generated operation reference |
| Benchmark packs | `noema_lab.core.benchmarks` and benchmark YAML files | Benchmark docs and validation commands |
| Run/result verification | `noema_lab.core.verification` | Verification docs and API reference |
| Runtime scheduling, cancellation, and plan caching | `noema_lab.core.executor` and `noema_lab.core.plan_cache` | Execution-runtime architecture page |
| External adapters | `noema_lab.core.external_adapters` and adapter manifests | External adapter SDK |
| Capture bundles | `noema_lab.core.capture` | Capture docs and API reference |

If a field is part of a public recipe, operation, adapter, benchmark, capture, run, result, or
submission contract, document it in the closest implementation source first. Public documentation
should then generate it or link to the generated reference.
