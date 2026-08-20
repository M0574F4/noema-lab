# DeepJSCC vs. Capacity-Matched JPEG over AWGN

Train an image encoder and decoder jointly over an additive white Gaussian noise (AWGN) channel,
then compare the learned system with a capacity-matched JPEG reference at a fixed bandwidth ratio.

Use this page for a compact implementation and training check. For a research-facing comparison
with no-CSI slow fading and a bandwidth sweep, use
[DeepJSCC over slow Rayleigh fading](deepjscc_slow_rayleigh.md).

```{warning}
**Experimental tutorial evidence.** The completed result on this page uses one trained model, four
Kodak crops, and three channel-noise seeds. It is not a paper-grade population claim or a
publication-ready result. Authoritative Kodak redistribution terms have not been archived, so this
site does not publish the source crop or reconstruction gallery.
```

## What this demo measures

The two paths are:

```text
Capacity-matched JPEG: image → highest JPEG quality fitting the capacity budget
                              → ideal error-free delivery → JPEG decode
DeepJSCC:              image → 0.5 learned complex symbols/pixel → AWGN
                              → learned reconstruction
```

The bandwidth ratio is
`κ = 0.5 complex channel uses per source pixel`. For an SNR coordinate in decibels,
`γ = 10^(SNR_dB/10)` and the ideal complex-AWGN budget is
`capacity bpp = κ × log2(1 + γ)`.

At each SNR, the JPEG reference searches qualities `1–95` and chooses the highest native bitstream
that fits this budget. If quality `1` does not fit, it records an outage and returns the benchmark's
declared gray reconstruction. This is an optimistic channel-coding assumption for one fixed JPEG
implementation, not an upper bound over digital separation or all source codecs. The comparison is
modeled on the protocol in the
[DeepJSCC paper](https://doi.org/10.1109/TCCN.2019.2919300).

PSNR, measured in decibels, and MS-SSIM, reported on an approximately `0–1` scale here, both increase
with reconstruction quality. Training minimizes image mean squared error (MSE); the benchmark
reports all three measures. This experiment varies SNR at one fixed `κ`, so it is not a
rate-distortion sweep.

## Before you start

You need:

- a Git clone of Noema, a supported Python version (`3.11`–`3.13`), and `uv`;
- network access for dependencies and the 24 hash-verified Kodak PNGs;
- port `8766` available if you choose the Workbench route;
- enough time for an 80-epoch starter-model training run and a 32-run benchmark.

The starter selects CUDA when available and otherwise uses CPU. Seeds and file hashes reproduce the
protocol, but training is not guaranteed to be bit-identical across hardware and library builds.
Your result ID and exact metrics may therefore differ from the reference result below.

Fetch the dataset once before choosing a workflow:

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync --extra onnx --extra wireless
uv run --extra onnx noema data fetch kodak \
  --directory .noema/datasets/kodak
```

```{caution}
The fetcher records the mirror URL and file hashes, but the repository currently records the Kodak
usage statement as an unverified third-party statement. Confirm the applicable terms before using
the data, and do not redistribute the images or their reconstructions from this tutorial.
```

## How the Noema pieces fit

```text
recipe
  → exported training bundle and interface contract
  → hash-verified data contract and train/validation split
  → paired encoder/decoder artifact
  → frozen benchmark pack
  → result bundle with metrics and provenance
```

The recipe defines the communication problem and replaceable interfaces. The exported bundle
contains the trainer-neutral contract plus a PyTorch starter. Training returns the encoder and
decoder together, and the benchmark evaluates that frozen pair without changing the declared
resource budget.

## Choose one workflow

The CLI quick start and the guided Workbench route are alternatives. Do not run both into the same
bundle directory. The exporter rejects a non-empty output directory unless overwrite is explicitly
enabled, so choose a fresh bundle path for each attempt.

### Option A — CLI training summary

Run this block from any directory inside the Git clone after fetching Kodak. It exports the bundle,
attaches the starter, validates the contract, trains and evaluates the pair, and runs the benchmark.

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/deepjscc_image"
  cd "$ROOT"

  uv run --project "$ROOT" --extra onnx --extra wireless noema differentiable export \
    "$ROOT/recipes/deepjscc_kodak_awgn_train.yaml" \
    --training-plan \
    "$ROOT/demo_trainings/deepjscc_image_reconstruction/training_plan.yaml" \
    --out "$BUNDLE"
  uv run --project "$ROOT" --extra onnx --extra wireless python \
    "$ROOT/demo_trainings/prepare_example.py" deepjscc-image "$BUNDLE" \
    --project-root "$ROOT"

  cd "$BUNDLE"
  uv run --project "$ROOT" --extra onnx --extra wireless python validate_contract.py
  uv run --project "$ROOT" --extra onnx --extra wireless python train_demo.py
  uv run --project "$ROOT" --extra onnx --extra wireless python evaluate_demo.py

  cd "$BUNDLE/reference_training"
  uv run --project "$ROOT" --extra onnx --extra wireless python build_benchmark.py
  cd "$ROOT"
  uv run --project "$ROOT" --extra onnx --extra wireless noema benchmark validate \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
  uv run --project "$ROOT" --extra onnx --extra wireless noema benchmark run \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
)
```

This example reads a file-backed dataset directly, so it does not need a tensor-capture stage.
The final command prints the result ID and paths to `result.json`, `metrics.csv`, `recipes.csv`, and
`summary.md`.

### Option B — guided Workbench workflow

#### 1. Open the template

Start Noema in Terminal 1 and leave the process running:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx --extra wireless noema ui serve --port 8766
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766), select **Browse template recipes**, then open
**Semantic communication** > **Image reconstruction** >
**Image reconstruction — DeepJSCC over AWGN**.

The recipe declares:

- Kodak images `01–20` for the deterministic train/validation partition;
- Kodak images `21–24` for the benchmark only;
- a training SNR grid of `[-4, 0, 4, 8, 12, 16]` dB;
- unit average transmitted complex-symbol power;
- paired trainable **Sender** and **Receiver** interfaces.

#### 2. Export from Workbench

Configure Workbench:

1. Under **Operation Training Capabilities**, select **Train/replace** for **Sender** and
   **Receiver**.
2. Set **Bundle directory** to `.noema/training_exports/deepjscc_image_workbench`.
3. Select **PyTorch** under **Support framework**.
4. Leave **Overwrite generated files** disabled for a new bundle. Enable it only when you
   intentionally want to regenerate Noema-owned contract files in an existing bundle.
5. Select **Export training bundle**.

In Terminal 2, attach the included starter:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx --extra wireless python demo_trainings/prepare_example.py \
  deepjscc-image .noema/training_exports/deepjscc_image_workbench
```

Inspect `data_contract.yaml` to confirm the selected files, hashes, split, and crop policy.

#### 3. Train and return the endpoint pair

```bash
cd "$(git rev-parse --show-toplevel)/.noema/training_exports/deepjscc_image_workbench"
uv run --project ../../.. --extra onnx --extra wireless python validate_contract.py
uv run --project ../../.. --extra onnx --extra wireless python train_demo.py
uv run --project ../../.. --extra onnx --extra wireless python evaluate_demo.py
```

The residual convolutional model downsamples each spatial dimension by eight and emits 32 complex
feature maps, giving `32 / (8 × 8) = 0.5` complex channel uses per source pixel. Crops, flips, and
rotations augment the 16 training images while the recorded source files and hashes remain fixed.

The paired artifact contains:

- `artifacts/encoder.onnx` and `artifacts/decoder.onnx`;
- `trained_artifact.yaml`, which binds the pair to its interfaces;
- `reference_training/training_history.json`;
- `reference_training/evaluation_metrics.json`.

Workbench discovers the pair automatically. Select **Validate returned model** and continue only
after the interface validation succeeds.

#### 4. Build and run the benchmark

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/deepjscc_image_workbench/reference_training
uv run --project ../../../.. --extra onnx --extra wireless python build_benchmark.py
cd ../../../..
uv run --extra onnx --extra wireless noema benchmark validate \
  .noema/training_exports/deepjscc_image_workbench/reference_training/benchmark_pack.yaml
uv run --extra onnx --extra wireless noema benchmark run \
  .noema/training_exports/deepjscc_image_workbench/reference_training/benchmark_pack.yaml
```

### Progress and recovery

| Stage | Successful checkpoint | Safe recovery |
| --- | --- | --- |
| Dataset fetch | 24 Kodak files pass the recorded hashes | Rerun the fetch command |
| Export | `data_contract.yaml` and starter support files exist | Choose a fresh bundle, or deliberately enable overwrite |
| Contract validation | `validate_contract.py` reports a valid contract | Fix the reported file, hash, or interface error before training |
| Training/evaluation | Both ONNX files and `trained_artifact.yaml` exist | Rerun from the bundle; training does not resume an interrupted epoch sequence |
| Benchmark | The CLI prints `RESULT_ID` and result paths | Rebuild the pack only if its inputs changed; otherwise rerun the benchmark |

## Method and benchmark protocol

The benchmark grid is `[-6, -4, -2, 0, 4, 8, 12, 16]` dB. The model trains on
`[-4, 0, 4, 8, 12, 16]` dB, so `−2` dB is an unseen interpolation point and `−6` dB tests
extrapolation below the training range.

The same channel-noise seeds—`71001`, `72001`, and `73001`—are reused at every SNR. This
common-random-number design pairs the SNR comparisons; the seeds are not fresh independent draws at
each coordinate. The capacity-matched JPEG reference is deterministic and runs once per SNR.

Both methods receive a maximum budget of `0.5` complex channel uses per source pixel. DeepJSCC emits
that number directly. The JPEG model reserves the same full block even when its native bitstream
uses less of the ideal capacity or the run is in outage.

The current JPEG recipe fixes the encoder configuration, including 4:2:0 subsampling, non-progressive
output, and qualities `1–95`. The
[current recipe template](../../recipes/jpeg_capacity_matched_kodak_awgn.yaml) may evolve; the
snapshot manifest records hashes for the exact per-run recipes used by the completed result.

## Checks before interpretation

Before interpreting a new result, verify that:

1. every learned run uses the same paired-artifact identity;
2. images `21–24` do not appear in the training contract;
3. both methods are admitted under the `0.5`-use resource limit;
4. every selected JPEG bitstream is at or below its ideal capacity budget;
5. quality `0` in the operating-point table is interpreted as the declared outage sentinel.

## Completed reference result

```{warning}
This stored result is descriptive tutorial evidence. Its 95% intervals cover three channel-noise
seeds for one frozen model and the same four crops. They do not cover training initialization,
model-selection, or image-population uncertainty.
```

<details>
<summary>Reference result identity</summary>

`20260727T235925Z_semantic_comm.digital_vs_deepjscc_post_training_v3` contains 32 runs: one
deterministic capacity-matched JPEG run and three learned noise realizations at each of eight SNR
coordinates.

</details>

<div
  data-noema-chart="deepjscc-psnr"
  data-chart-summary="DeepJSCC has higher mean PSNR at −6 and −4 dB; capacity-matched JPEG is higher from −2 through 16 dB."
>
  <p><strong>PSNR chart unavailable.</strong> The accessible quality table below contains the
  plotted means; the downloadable chart data contains interval bounds.</p>
</div>

<div
  data-noema-chart="deepjscc-ms-ssim"
  data-chart-summary="MS-SSIM follows the same ordering as PSNR: DeepJSCC leads at −6 and −4 dB, then capacity-matched JPEG leads from −2 dB."
>
  <p><strong>MS-SSIM chart unavailable.</strong> The accessible quality table below contains the
  plotted means; the downloadable chart data contains interval bounds.</p>
</div>

```{csv-table} Reconstruction-quality means
:file: ../demo/data/digital_vs_deepjscc/quality_summary_table.csv
:header-rows: 1
:class: noema-compact-table
```

```{csv-table} Capacity-matched JPEG operating points
:file: ../demo/data/digital_vs_deepjscc/jpeg_operating_points_table.csv
:header-rows: 1
:class: noema-compact-table
```

```{csv-table} Communication-resource audit
:file: ../demo/data/digital_vs_deepjscc/resource_audit.csv
:header-rows: 1
:class: noema-compact-table
```

In this completed run, DeepJSCC reaches `21.99` dB PSNR at −6 dB versus `13.72` dB for the JPEG
reference, and `23.14` versus `17.64` dB at −4 dB. The ordering reverses at −2 dB, where JPEG reaches
`25.34` dB and DeepJSCC `24.01` dB. At 16 dB, JPEG reaches `37.50` dB and DeepJSCC `26.02` dB.
MS-SSIM follows the same ordering. These observations illustrate low-SNR graceful degradation for
this artifact, dataset slice, and fixed bandwidth ratio; they do not establish universal learned
superiority or rate-distortion optimality.

The reader-facing tables are rounded for comparison. Download the
[full-precision summary](../demo/data/digital_vs_deepjscc/summary_table.csv),
[per-run benchmark projection](../demo/data/digital_vs_deepjscc/benchmark_projection.csv),
[chart data](../demo/data/digital_vs_deepjscc/chart_data.json), and
[snapshot manifest](../demo/data/digital_vs_deepjscc/snapshot_manifest.json) for the stored values,
run IDs, recipe hashes, resource checks, and statistical design. The manifest identifies the local
raw result files by hash; the raw result bundle is not distributed by this documentation snapshot.

### Why there is no reconstruction gallery

The local benchmark records representative reconstructions, including explicit gray outage images.
They are intentionally excluded from the published documentation until authoritative Kodak terms
covering redistribution and web display are archived. This keeps the experimental metrics
reproducible without treating uncertain dataset rights as a cosmetic warning.

## Verify your local result

Set the variable to the ID printed by your benchmark:

```bash
cd "$(git rev-parse --show-toplevel)"
RESULT_ID="<paste-the-result-id>"
uv run --extra onnx --extra wireless noema benchmark verify "$RESULT_ID"
```

Read every verification warning before using the result outside a local experiment. Do not suppress
warnings to make an experimental result appear publication-ready.

## Publication status

This benchmark remains `experimental`, the stored result is not paper-grade evidence, and the Kodak
rights record is unresolved. A future publication path should require:

1. authoritative dataset terms and an archived dataset citation;
2. a canonical benchmark with `traceability_profile_requested: true` and the exact current profile
   binding;
3. substantially more held-out images and channel seeds;
4. independently trained models when making population-level claims;
5. a persistently archived result bundle, environment identity, and verification report;
6. publication without `--allow-warnings`.

## References and implementation notes

- Bourtsoulatze, Kurka, and Gündüz,
  [“Deep Joint Source-Channel Coding for Wireless Image Transmission”](https://doi.org/10.1109/TCCN.2019.2919300).
- This tutorial preserves the paper's fixed-bandwidth AWGN comparison idea but uses Noema's
  CPU-sized residual model, Kodak split, training grid, and metric implementations. It is not a
  reproduction of the paper's reported numbers.
- JPEG thresholds depend on the Pillow/libjpeg build and the recipe's exact encoder settings.
  MS-SSIM uses Noema's configured `pytorch-msssim` implementation. The local result bundle and lock
  file provide the implementation context for a reproduced run.
