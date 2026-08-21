<p align="center">
  <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/launch/f0-launch-hero.svg" alt="Noema: make the comparison traceable, with bounded experimental launch evidence" width="100%" />
</p>

<p align="center">
  <a href="https://M0574F4.github.io/noema-lab/">
    <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/noema-logo.svg" alt="Noema logo" width="96" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/M0574F4/noema-lab/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/M0574F4/noema-lab/ci.yml?branch=main&label=ci"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/M0574F4/noema-lab"></a>
  <a href="CITATION.cff"><img alt="Cite" src="https://img.shields.io/badge/cite-CITATION.cff-53b889"></a>
  <a href="https://M0574F4.github.io/noema-lab/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-4eb7c4"></a>
  <img alt="Pre-release" src="https://img.shields.io/badge/status-pre--release-e2a646">
</p>

# Noema

**Executable experiment contracts for learned-communication comparisons.**

Noema binds a schema-validated comparison protocol to its execution plan, communication-resource
accounting, returned models, terminal outcomes, and retained evidence. The result is a local,
inspectable chain from a declared question to a reported figure.

Noema checks the identities and relations covered by the selected traceability profile. It does
not prove scientific fairness, standards conformance, authenticity, or independent reproduction.

<p align="center">
  <strong><a href="https://M0574F4.github.io/noema-lab/break_the_comparison.html">Try the flagship demo</a></strong>
  · <a href="https://www.youtube.com/watch?v=bKNXS_vHLHc">Watch the complete workflow</a>
  · <a href="https://M0574F4.github.io/noema-lab/">Read the documentation</a>
  · <a href="https://M0574F4.github.io/noema-lab/demos.html">Explore demonstrations</a>
  · <a href="https://M0574F4.github.io/noema-lab/launch_assets.html">Inspect launch evidence</a>
</p>

## Start here

| I want to… | Start with… |
| --- | --- |
| understand the central idea | the interactive [Break the comparison](https://M0574F4.github.io/noema-lab/break_the_comparison.html) evidence lab |
| watch Noema train and compare a model | the [learned QPSK I/Q calibration walkthrough](https://www.youtube.com/watch?v=bKNXS_vHLHc) |
| run a dependency-light example | the source-checkout quickstart below |
| bring my own model | the [external adapter SDK](https://M0574F4.github.io/noema-lab/external_adapter_sdk.html) |
| export a training contract | the [architecture-neutral export workflow](https://M0574F4.github.io/noema-lab/tutorials/export_differentiable_training_scenario.html) |
| train and return a model | the [trained-artifact workflow](https://M0574F4.github.io/noema-lab/tutorials/external_training_checkpoint_adapter.html) |
| inspect retained evidence | [result verification](https://M0574F4.github.io/noema-lab/result_verification.html) |
| build a paper figure | the [physical-layer demo workflow](https://M0574F4.github.io/noema-lab/tutorials/physical_layer_demo_workflow.html) |
| browse implementation contracts | the generated [reference](https://M0574F4.github.io/noema-lab/reference/index.html) |

## Watch the complete workflow

<p align="center">
  <a href="https://www.youtube.com/watch?v=bKNXS_vHLHc">
    <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/noema-training-loop.svg" alt="Watch Noema export a training contract, return a model, benchmark it, and verify the result" width="88%" />
  </a>
</p>

The recorded walkthrough starts from a clean environment and shows contract export, dataset
capture, external model training, returned-model validation, and the UI comparison against the
uncompensated and calibrated-oracle baselines.

## Quickstart

Noema is not yet published on PyPI and currently supports Python 3.11–3.13. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/M0574F4/noema-lab.git
cd noema-lab
uv sync
uv run noema template instantiate semantic_comm.text_semantic_similarity.default > noema-quickstart.yaml
uv run noema recipe lint noema-quickstart.yaml
uv run noema recipe run noema-quickstart.yaml
uv run noema ui serve --port 8766
```

Open `http://127.0.0.1:8766`.

Sionna-backed paths are optional. Install the current no-ray-tracing Sionna 2/PyTorch stack with
`uv sync --extra wireless`; install the CompressAI examples with `uv sync --extra compressai`.
The [tutorials](https://M0574F4.github.io/noema-lab/tutorials.html) identify which workflows need
large downloads, external datasets, or additional rights review.

## See why the contract matters

<p align="center">
  <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/launch/f3-receiver-ber-vs-snr.svg" alt="Canonical launch plot of pre-decoder BER versus SNR with bounded experimental disclosure" width="100%" />
</p>

This is one completed experimental demonstration. The learned band is the observed minimum and
maximum over three paired held-out seeds—not a confidence interval—and the calibrated oracle is a
diagnostic reference with additional calibration knowledge. The result is not presented as a
publication-ready canonical benchmark.

Every displayed number and warning in the launch surfaces comes from
[`launch_evidence.json`](launch_evidence.json). The generated [F0–F5 figures and T0–T2
tables](https://M0574F4.github.io/noema-lab/launch_assets.html) are projections of that source.

## How Noema works

<p align="center">
  <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/launch/f2-contract-to-evidence.svg" alt="Noema chain from typed protocol through execution, accounting, retained evidence, and launch projection" width="100%" />
</p>

```text
experiment contract (recipe + benchmark protocol)
  -> concrete pre-execution plan
  -> benchmark evaluation or typed capture/export
  -> external training by the researcher
  -> returned model bound to declared artifact slots
  -> selected benchmark evaluation
  -> locally verified result bundle and evidence-bound plot
```

Noema is CLI-first. The dashboard, benchmark runner, capture/export paths, and evidence tools
read the same operation contracts and result evidence instead of maintaining separate workflow
models.

## What Noema retains

```text
.noema/benchmarks/<result_id>/
  result.json          benchmark-level evidence
  metrics.csv          table-ready metrics
  recipes.csv          exact recipe membership
  summary.md           human-readable report
  figures/*.png        optional evidence-bound plots

.noema/runs/<run_id>/
  recipe.json          normalized plan that was executed
  manifest.json        hashes, environment, operation contracts
  summary.json         metrics, artifacts, terminal status
  artifacts/           images, text, plots, arrays
```

A publication can cite the recipe SHA-256, benchmark ID and version, dataset/task/channel
conditions, transmitted bits and channel uses, BER/BLER, returned-model hashes, environment
manifest, table CSV, and plotted-data CSV. Verification establishes the declared local relations;
reviewers still assess whether the scientific comparison itself is appropriate.

## How Noema fits

- **Sionna** supplies simulation and physical-layer building blocks.
- **CompressAI** supplies learned-compression models, codecs, and evaluation utilities.
- **DeepMIMO** supplies scenario-based channel data for MIMO research.
- **Noema** binds work across such tools into one comparison contract and retained evidence chain.

Noema does not replace those projects. It is an experiment-contract runner, resource-accounting
layer, capture/export bridge, and local evidence verifier. It is not a full model trainer,
standards-conformance validator, private leaderboard, or guarantee of fairness or reproducibility.

## Suites and maturity

| Status | Suite | Examples |
| --- | --- | --- |
| Active development | Semantic Communication | image reconstruction, text, VQA, retrieval, generative receiver |
| Experimental | Neural Receiver / AI-PHY | learned QPSK demapping, carrier tracking, modulation recognition |
| Experimental | Channel Estimation and MIMO-OFDM | pilot estimation, CSI feedback, learned channel estimators |
| Experimental | Resource Allocation | equal-power baselines, delayed-CSI learned allocation |
| Experimental | Beamforming, Localization, and Sensing | codebooks, learned policies, range and AoA tasks |

The documentation labels smoke, experimental, completed evidence, and release-candidate surfaces
separately. A configured platform or workflow becomes supported only after its exact release archive
passes the frozen release checks.

## Documentation map

| Section | Use it for |
| --- | --- |
| [Overview](https://M0574F4.github.io/noema-lab/) | product thesis, routes, workflow, and evidence boundary |
| [Tutorials](https://M0574F4.github.io/noema-lab/tutorials.html) | source checkout, training export, returned models, and publication workflows |
| [Demonstrations](https://M0574F4.github.io/noema-lab/demos.html) | physical-layer and end-to-end examples |
| [Core architecture](https://M0574F4.github.io/noema-lab/architecture.html) | typed operations, execution, accounting, and artifacts |
| [Result verification](https://M0574F4.github.io/noema-lab/result_verification.html) | manifests, identities, profiles, and verifier limits |
| [Evidence and submissions](https://M0574F4.github.io/noema-lab/publication_artifact_readiness.html) | artifact readiness, result submissions, and launch evidence |
| [Reference](https://M0574F4.github.io/noema-lab/reference/index.html) | generated CLI, operations, API, and schemas |

## Release status

Noema is pre-release software. Source builds currently report `0.2.0.dev0`; the first stable release
will use a separately qualified version and tag. The repository contains the public software,
documentation, runnable demonstrations, and curated demonstration evidence—not the private paper,
research notes, raw experiment workspace, or model checkpoints.

## Cite and contribute

Use [`CITATION.cff`](CITATION.cff) for citation metadata. Contribution expectations, governance,
security reporting, and community conduct are documented in [CONTRIBUTING.md](CONTRIBUTING.md),
[GOVERNANCE.md](GOVERNANCE.md), [SECURITY.md](SECURITY.md), and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

<p align="center">
  <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/assets/benchmarked-with-noema.svg" alt="Benchmarked with Noema badge" />
</p>
