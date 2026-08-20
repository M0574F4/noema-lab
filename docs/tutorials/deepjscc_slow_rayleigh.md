# Blind DeepJSCC over Slow Rayleigh Fading

## Goal

Train one image encoder/decoder pair for a channel whose complex gain is unknown and
constant over each image:

```text
Blind DeepJSCC: image → learned complex symbols → h·symbols + noise
                       → learned reconstruction

Digital reference: image → JPEG rate chosen from average SNR
                         → ideal capacity-achieving code
                         → perfect JPEG decode or block-fading outage
```

Here `h ~ CN(0,1)` changes independently between images. Neither learned endpoint
receives `h`, an estimate of `h`, or pilots. The digital reference is deliberately
strong: when the instantaneous channel supports its fixed rate, delivery is perfect.
When it does not, the entire image packet is in outage.

This is the useful contrast. Digital separation has a sharp outage boundary, while a
continuous joint source-channel representation can learn a graceful reconstruction
from the distorted symbols. The protocol follows the slow-fading comparison in the
[original DeepJSCC paper](https://arxiv.org/abs/1809.01733).

The example trains a nested representation at three bandwidth ratios:
`κ = 0.125, 0.25, 0.5` complex channel uses per source pixel. The benchmark therefore
reports both quality versus average SNR and a rate–distortion slice at 10 dB.

## CLI training summary

Run this block from anywhere inside the repository:

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/deepjscc_slow_rayleigh"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/deepjscc_kodak_slow_rayleigh_train.yaml" \
    --training-plan \
    "$ROOT/demo_trainings/deepjscc_image_reconstruction/training_plan.yaml" \
    --out "$BUNDLE"
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" deepjscc-image "$BUNDLE" \
    --project-root "$ROOT"

  cd "$BUNDLE"
  uv run --project "$ROOT" --extra onnx python validate_contract.py
  uv run --project "$ROOT" --extra onnx python train_demo.py
  uv run --project "$ROOT" --extra onnx python evaluate_demo.py

  cd "$BUNDLE/reference_training"
  uv run --project "$ROOT" --extra onnx python \
    build_slow_fading_benchmark.py
  cd "$ROOT"
  uv run --project "$ROOT" --extra onnx noema benchmark validate \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
  uv run --project "$ROOT" --extra onnx noema benchmark run \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
)
```

No dataset-capture command is required. The example reads the hash-pinned Kodak files
through the exported data contract. The last command prints the benchmark result ID.

## UI workflow

1. Start Noema with `uv run --extra onnx noema ui serve --port 8766`.
2. Select **Browse template recipes** > **Semantic communication** >
   **Image reconstruction** > **Image reconstruction — blind DeepJSCC over slow
   fading**.
3. In **Workbench**, select **Train/replace** for both **Sender** and **Receiver**.
4. Set the bundle directory to
   `.noema/training_exports/deepjscc_slow_rayleigh`, select **PyTorch**, and export
   the training bundle.
5. Attach the example, train, and evaluate with the commands above.
6. Return to Workbench and select **Validate returned model**.

The recipe owns the system assumptions: data split, slow-fading law, missing CSI,
power normalization, and SNR sweep. The example scaffold owns only the illustrative
model, loss, and trainer.

## What the model learns

The encoder emits a progressively ordered latent representation. Training randomly
uses 8, 16, or 32 complex feature maps, so the first maps must carry the most useful
image information. A single complex gain rotates and scales every latent symbol of an
image before noise is added. A small trainable front end pools global received-symbol
statistics and predicts one complex correction per image; the image decoder then works
from the corrected latent tensor. It is not given the true gain. The encoder can learn
codeword statistics that make this blind estimate identifiable, effectively learning
synchronization structure jointly with the image representation.

Training exports three paired ONNX artifacts:

- `trained_artifact_kappa_0p125.yaml`;
- `trained_artifact_kappa_0p25.yaml`;
- `trained_artifact.yaml` for `κ = 0.5`.

All three come from the same selected checkpoint. This makes the rate sweep a property
of one trained representation rather than three unrelated favorable trials.

## Fair comparison

For every SNR, bandwidth ratio, held-out image, and seed:

- both methods use the same channel-use budget;
- both use the same slow-Rayleigh realization;
- the learned decoder receives no CSI or pilots;
- JPEG quality is selected only from average SNR;
- the ideal digital packet is decoded perfectly when instantaneous capacity supports
  its rate;
- an unsupported digital packet is reconstructed with that source image's per-channel
  mean, as declared before the run.

The digital oracle uses the simulated gain only after the fact to determine whether
its fixed rate is in outage. It does not adapt JPEG rate to the instantaneous gain.
This makes it an optimistic digital upper bound, not a practical no-CSI receiver.

The primary plot fixes `κ = 0.5` and sweeps average SNR. The rate–distortion plot fixes
average SNR at 10 dB and sweeps `κ`. PSNR and MS-SSIM are reported together with the
digital outage rate and exact channel uses per pixel.

## Completed result

Result
`20260728T002720Z_semantic_comm.deepjscc_slow_rayleigh_post_training_v1`
contains 48 completed runs. Each plotted cell uses the four held-out Kodak images and
three channel seeds. Within every coordinate and seed, the learned and digital runs
use the same per-image fading gains.

<div data-noema-chart="deepjscc-slow-psnr-snr"></div>

<div data-noema-chart="deepjscc-slow-ms-ssim-snr"></div>

At `κ = 0.5`, Blind DeepJSCC has higher mean PSNR at every tested SNR. Its advantage
decreases from 1.72 dB at 0 dB average SNR to 0.23 dB at 20 dB. The MS-SSIM
comparison is stronger: at 20 dB it reaches 0.823 versus 0.451 for the digital
reference.

The SNR-sweep table reports a JPEG outage rate of `0.75` at every coordinate. The
same digital fades remain in outage as average SNR rises because JPEG rate also rises
with average channel capacity. Without instantaneous CSI, that operating rule keeps
the normalized outage threshold approximately fixed. Successful packets improve with
the selected JPEG quality, while failed packets still use the declared fallback.
DeepJSCC has no binary packet-outage state; its degradation is measured by PSNR and
MS-SSIM instead.

<div data-noema-chart="deepjscc-slow-rate-psnr"></div>

The nested learned representation improves from 20.14 dB at `κ = 0.125` to 21.04 dB
at `κ = 0.5`. The digital rate-slice means are non-monotonic because each κ cell has
only three deterministic fading replicates and packet outage dominates the average.
The confidence bands expose that variance; this experimental slice is not evidence
that increasing digital bandwidth reduces source-coding quality.

```{csv-table} SNR-sweep means
:file: ../demo/data/deepjscc_slow_rayleigh/snr_summary.csv
:header-rows: 1
:class: noema-compact-table
```

```{csv-table} Rate-slice means at 10 dB
:file: ../demo/data/deepjscc_slow_rayleigh/rate_summary.csv
:header-rows: 1
:class: noema-compact-table
```

### Reconstruction previews and dataset rights

The benchmark evidence can produce local source/reconstruction previews, but Noema
does not redistribute those Kodak-derived images. Obtain the dataset under its
authoritative terms and generate previews only in your local workspace. The public
snapshot retains aggregate metrics and content-bound run identities, not source pixels
or derivative reconstructions.

The [benchmark projection](../demo/data/deepjscc_slow_rayleigh/benchmark_projection.csv),
[paired-fading audit](../demo/data/deepjscc_slow_rayleigh/pairing_audit.csv),
[interactive chart data](../demo/data/deepjscc_slow_rayleigh/chart_data.json), and
[snapshot manifest](../demo/data/deepjscc_slow_rayleigh/snapshot_manifest.json)
preserve the plotted values, run identities, evidence hashes, and statistical design.
The three-seed intervals make this an experimental demonstration rather than a
paper-grade population claim.
