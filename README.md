<p align="center">
  <a href="https://M0574F4.github.io/noema-lab/">
    <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/noema-logo-dark.svg" alt="Noema — Semantic communication research toolkit" width="420" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/M0574F4/noema-lab/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/M0574F4/noema-lab/ci.yml?branch=main&label=ci"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/M0574F4/noema-lab"></a>
  <a href="CITATION.cff"><img alt="Cite" src="https://img.shields.io/badge/cite-CITATION.cff-53b889"></a>
  <a href="https://M0574F4.github.io/noema-lab/"><img alt="Documentation" src="https://img.shields.io/badge/docs-GitHub%20Pages-4eb7c4?logo=githubpages&logoColor=white"></a>
  <img alt="Pre-release" src="https://img.shields.io/badge/status-pre--release-e2a646">
</p>

# Noema

**Executable experiment contracts for learned-communication comparisons.**

Noema binds a schema-validated comparison protocol to its execution plan, communication-resource
tracking, returned models, terminal outcomes, and retained evidence. The result is a local,
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

## Watch the complete workflow

<p align="center">
  <a href="https://www.youtube.com/watch?v=bKNXS_vHLHc"><img alt="Watch the complete Noema workflow on YouTube" src="https://img.shields.io/badge/Watch-YouTube-FF0000?logo=youtube&logoColor=white" width="220"></a>
</p>

The recorded walkthrough starts from a clean environment and shows contract export, dataset
capture, external model training, returned-model validation, and the UI comparison against the
uncompensated and calibrated-oracle baselines.

## Start here

| I want to… | Start with… |
| --- | --- |
| understand the central idea | the interactive [Break the comparison](https://M0574F4.github.io/noema-lab/break_the_comparison.html) evidence lab |
| watch an external model train, then compare it with Noema | the [learned QPSK I/Q calibration walkthrough](https://www.youtube.com/watch?v=bKNXS_vHLHc) |
| **choose a system to train** | **the [ready-to-train matrix](#ready-to-train-systems) below** |
| run a dependency-light example | the source-checkout quickstart below |
| bring my own model | the [external adapter SDK](https://M0574F4.github.io/noema-lab/external_adapter_sdk.html) |
| export a training contract | the [architecture-neutral export workflow](https://M0574F4.github.io/noema-lab/tutorials/export_differentiable_training_scenario.html) |
| train and return a model | the [trained-artifact workflow](https://M0574F4.github.io/noema-lab/tutorials/external_training_checkpoint_adapter.html) |
| inspect retained evidence | [result verification](https://M0574F4.github.io/noema-lab/result_verification.html) |
| build a paper figure | the [physical-layer demo workflow](https://M0574F4.github.io/noema-lab/tutorials/physical_layer_demo_workflow.html) |
| browse implementation contracts | the generated [reference](https://M0574F4.github.io/noema-lab/reference/index.html) |

## Ready-to-train systems

Every **Train + compare** page starts with one copyable CLI block that exports the contract,
prepares the data, trains the included starter model, returns its artifact, and builds or runs the
baseline comparison. Replace the starter with your own model while keeping the exported interface
and experiment protocol fixed.

| Problem | What you can train | Included comparisons | Availability |
| --- | --- | --- | --- |
| Receiver I/Q calibration | affine QPSK receiver | uncompensated QPSK; calibrated I/Q oracle | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_qpsk_demapper_demo.html) |
| Carrier tracking | packet-context QPSK phase tracker | interpolation; smoothing; decision-directed PLL; true-phase reference | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_qpsk_phase_tracking_demo.html) |
| Automatic modulation recognition | blind I/Q classifier | differential cumulants; synchronized-likelihood reference | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/automatic_modulation_recognition_demo.html) |
| MIMO-OFDM channel estimation | 2×2 sparse-pilot estimator | LS interpolation; fixed-prior LMMSE; exact-channel diagnostic | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_mimo_ofdm_channel_estimation_demo.html) |
| CSI compression and feedback | 128-bit encoder/decoder pair | matched KLT/PCA codec | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_csi_feedback.html) |
| OFDM subcarrier allocation | power-allocation policy | equal power; water filling | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/ofdm_resource_allocation_demo.html) |
| Delayed-CSI OFDM allocation | reliability-aware causal allocator | equal power; delayed-CSI and uncertainty-aware water filling | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/reliability_aware_ofdm_allocation_demo.html) |
| Joint communication and sensing | OFDM power-allocation policy | equal power; communication water filling; iterative scalarized reference | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_isac_ofdm_allocation_demo.html) |
| Image delivery over AWGN | DeepJSCC image encoder/decoder | capacity-matched JPEG | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/digital_vs_deepjscc_sionna.html) |
| Image delivery over slow fading | blind, nested-rate DeepJSCC pair | outage-aware capacity-matched JPEG | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/deepjscc_slow_rayleigh.html) |
| 2D range localization | geometry-aware residual localizer | linear and regularized trilateration | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_range_localization_demo.html) |
| Narrowband AoA estimation | covariance-domain array estimator | MUSIC; Bartlett | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_aoa_estimation_demo.html) |
| MISO beam selection | learned eight-beam codebook | perfect-CSIT MRT; equal-size fixed DFT-codebook sweep | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_beam_selection_demo.html) |
| Near-field XL-MIMO focusing | physics-informed range/angle estimator | far-field steering; polar codebook; simulation-truth oracle | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_near_field_xl_mimo_demo.html) |
| LEO-NTN Doppler and handover | causal Doppler/next-beam tracker | hold-last; linear extrapolation; future-state oracle | ✅ [Train + compare](https://M0574F4.github.io/noema-lab/tutorials/learned_leo_ntn_tracking_demo.html) |

## Quickstart

Noema is not yet published on PyPI and currently supports Python 3.11–3.13. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/M0574F4/noema-lab.git
cd noema-lab
uv sync --frozen
uv run noema template instantiate semantic_comm.text_semantic_similarity.default > noema-quickstart.yaml
uv run noema recipe lint noema-quickstart.yaml
uv run noema recipe run noema-quickstart.yaml
uv run noema ui serve --port 8766
```

Open `http://127.0.0.1:8766`. Press `Ctrl+C` in the terminal to stop the UI. The generated
`noema-quickstart.yaml` file is ignored by Git, so the checkout stays clean.

Sionna-backed paths are optional. Install the current no-ray-tracing Sionna 2/PyTorch stack with
`uv sync --extra wireless`; install the CompressAI examples with `uv sync --extra compressai`.
The [tutorials](https://M0574F4.github.io/noema-lab/tutorials.html) identify which workflows need
large downloads, external datasets, or additional rights review.

## How Noema works

### Direct benchmark path

<p align="center">
  <img src="https://raw.githubusercontent.com/M0574F4/noema-lab/main/docs/_static/launch/f2-contract-to-evidence.svg" alt="Noema chain from typed protocol through execution, metrics, retained evidence, and launch projection" width="100%" />
</p>

The diagram above shows the direct path from a declared benchmark to retained evidence and a
public result. When a researcher trains a replacement model, Noema uses the longer handoff below.

### External-training path

```text
experiment contract (recipe + benchmark protocol)
  -> concrete pre-execution plan
  -> benchmark evaluation or typed capture/export
  -> external training by the researcher
  -> returned model bound to declared artifact slots
  -> selected benchmark evaluation
  -> locally verified result bundle and plot linked to its source evidence
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
  figures/*.png        optional plots linked to result evidence

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

Noema does not replace those projects. It is an experiment-contract runner, resource-tracking
layer, capture/export bridge, and local evidence verifier. It is not a full model trainer,
standards-conformance validator, private leaderboard, or guarantee of fairness or reproducibility.

## Cite and contribute

Use [`CITATION.cff`](CITATION.cff) for citation metadata. Contribution expectations, governance,
security reporting, and community conduct are documented in [CONTRIBUTING.md](CONTRIBUTING.md),
[GOVERNANCE.md](GOVERNANCE.md), [SECURITY.md](SECURITY.md), and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
