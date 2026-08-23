# Noema Documentation

```{figure} _static/launch/f0-launch-hero.svg
:alt: Noema launch hero stating Make the comparison traceable, with the bounded experimental receiver headline and its scientific-status disclosure
:class: noema-home-hero
:figclass: noema-home-hero-figure
```

Noema is an experiment-contract and evidence system for learned-communication comparisons. It
keeps the protocol, execution plan, resource accounting, returned model, terminal outcome, and
retained evidence attached to the same verifiable object.

```{raw} html
<nav class="noema-home-actions" aria-label="Primary Noema destinations">
  <a class="noema-home-action noema-home-action-primary" href="break_the_comparison.html">Try the flagship demo</a>
  <a class="noema-home-action" href="#watch-the-complete-workflow">Watch the complete workflow</a>
  <a class="noema-home-action" href="tutorials.html">Start a workflow</a>
  <a class="noema-home-action" href="demos.html">Explore demonstrations</a>
  <a class="noema-home-action" href="reference/index.html">Open the reference</a>
</nav>
```

Noema's verifier establishes local, profile-scoped consistency. It does not establish scientific
fairness, standards conformance, authenticity, or independent reproduction.

## Start with the failure

[Break the comparison](break_the_comparison.md) is the shortest introduction to Noema. Begin with
a valid paired result, then change the condition, aggregation, metric, or comparator role. The
arithmetic remains precise while the claim loses the contract that made it interpretable.

## Watch the complete workflow

```{include} _includes/launch_video.md
```

## Choose your route

```{raw} html
<div class="noema-home-routes">
  <a class="noema-home-route" href="tutorials.html">
    <strong>Run an experiment</strong>
    <span>Install from source, instantiate a recipe, and retain a verifiable result.</span>
  </a>
  <a class="noema-home-route" href="tutorials/export_differentiable_training_scenario.html">
    <strong>Train your method</strong>
    <span>Export a typed training contract and return a portable trained artifact.</span>
  </a>
  <a class="noema-home-route" href="demos.html">
    <strong>Study examples</strong>
    <span>Open the physical-layer and end-to-end communication demonstrations.</span>
  </a>
  <a class="noema-home-route" href="result_verification.html">
    <strong>Audit evidence</strong>
    <span>Understand manifests, traceability profiles, and verification boundaries.</span>
  </a>
  <a class="noema-home-route" href="external_adapter_sdk.html">
    <strong>Bring your own model</strong>
    <span>Connect an encoder, metric, dataset, or portable inference runtime.</span>
  </a>
  <a class="noema-home-route" href="publication_artifact_readiness.html">
    <strong>Prepare a publication</strong>
    <span>Separate completed evidence, release controls, and scientific readiness.</span>
  </a>
</div>
```

## How Noema works

```{figure} _static/noema-training-loop.svg
:alt: Six-step Noema workflow from opening a scenario and choosing trainable blocks through contract export, external model training, recipe binding, and verified benchmarking
:class: noema-overview-workflow
:figclass: noema-overview-workflow-figure noema-home-workflow-figure
:target: _static/noema-training-loop.svg

Noema's contract-first model-development loop. Open the diagram for a full-size view or
[download the editable draw.io source](_static/diagrams/noema-training-loop.drawio).
```

An executable experiment contract links a schema-validated protocol to a concrete execution plan,
declared rate and channel-use accounting, optional returned-model interfaces, terminal outcomes,
and retained result and plot evidence. Noema runs typed benchmark and capture recipes, exports
architecture-neutral training-return contracts, and checks the identities and relations covered by
the selected traceability profile.

## Evidence, with its boundaries

```{figure} _static/launch/f3-receiver-ber-vs-snr.svg
:alt: Log-scale pre-decoder BER versus SNR for uncompensated QPSK, a calibrated diagnostic oracle, and a learned receiver with an observed min-max band over paired seeds
:class: noema-home-result
:target: _static/launch/f3-receiver-ber-vs-snr.svg

The selected launch demonstration. The learned band is the observed minimum and maximum over three
paired held-out seeds—not a confidence interval—and the calibrated oracle is a diagnostic reference.
Open the chart for a full-size view.
```

The demo, chart, and tables read their values and warnings from the same generated evidence file,
so their claims cannot silently drift apart. Open the complete
[F0–F5 visual and T0–T2 table set](launch_assets.md), inspect the
[canonical launch evidence](launch_evidence.md), or browse the
[static result explorer](https://M0574F4.github.io/noema-lab/demo/).
This is completed experimental evidence, not a publication-ready canonical benchmark.

## Browse the documentation

Use the navigation menu for the full documentation tree. New readers can start with the
[tutorials](tutorials.md), [demonstrations](demos.md), [architecture](architecture.md), or generated
[reference](reference/index.md). For the source-checkout quickstart, see the project
[README](https://github.com/M0574F4/noema-lab/blob/main/README.md).

```{toctree}
:maxdepth: 2
:caption: Overview
:titlesonly:
:hidden:

architecture/ai_native_positioning
break_the_comparison
tutorials
demos
```

```{toctree}
:maxdepth: 2
:caption: Suites
:titlesonly:
:hidden:

suites/index
```

```{toctree}
:maxdepth: 2
:caption: Core Architecture
:titlesonly:
:hidden:

architecture
benchmarking
result_verification
traceability_profiles
architecture/backend_materialization
architecture/execution_profiles
architecture/execution_runtime
architecture/execution_plan_cache
rate_channel_accounting
codec_payload_security
differentiable_export_architecture
semantic_artifacts
```

```{toctree}
:maxdepth: 2
:caption: Workflows
:titlesonly:
:hidden:

tutorials/export_differentiable_training_scenario
tutorials/external_training_checkpoint_adapter
tutorials/physical_layer_demo_workflow
text_task
task_oriented_benchmarks
vqa_goal_oriented_benchmarks
static_demo
```

```{toctree}
:maxdepth: 2
:caption: Evidence and Submissions
:titlesonly:
:hidden:

publication_artifact_readiness
submissions
launch_evidence
launch_assets
```

```{toctree}
:maxdepth: 2
:caption: Extending Noema
:titlesonly:
:hidden:

external_adapter_sdk
```

```{toctree}
:maxdepth: 2
:caption: Reference
:titlesonly:
:hidden:

reference/index
```
