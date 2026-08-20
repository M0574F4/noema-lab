# Learned CSI Compression and Feedback

## Goal

Train a paired encoder/decoder that compresses MISO-OFDM CSI to a 128-bit feedback message, then
compare it with a matched KLT/PCA codec at the same feedback budget. The base station reconstructs
CSI, forms an MRT precoder, and evaluates spectral efficiency on the original channel realization.

The fixed protocol is Sionna TDL-A with CSI shape `2 × 8 × 32`, 32 feedback values, 4-bit uniform
quantization, and 512 channel realizations per benchmark run. The learned codec changes only the
feedback encoder and decoder. For the general contract and artifact workflow, see
[Physical-Layer System Templates and Learned-Method Examples](physical_layer_demo_workflow.md).

## CLI training summary

Use a fresh `BUNDLE` path for this all-in-one route; the detailed Workbench walkthrough below is an
alternative, not an additional sequence to run.
Each `dataset-capture` command reports progress for its own split; let it complete before the next
command starts. Add `--force` only when intentionally replacing an existing capture.

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/csi_feedback"

  cd "$ROOT"
  # This exact reproducibility block requires a Noema source checkout.
  uv sync --extra onnx --extra wireless
  uv run --project "$ROOT" --extra onnx --extra wireless noema differentiable export \
    "$ROOT/recipes/csi_feedback_sionna_train.yaml" \
    --training-plan "$ROOT/demo_trainings/csi_feedback_autoencoder/training_plan.yaml" \
    --out "$BUNDLE"
  uv run --project "$ROOT" --extra onnx --extra wireless python \
    "$ROOT/demo_trainings/prepare_example.py" csi-feedback "$BUNDLE" \
    --project-root "$ROOT"

  uv run --project "$ROOT" --extra onnx --extra wireless noema dataset-capture run \
    "$BUNDLE/capture_train_recipe.yaml" --out "$BUNDLE/data/train"
  uv run --project "$ROOT" --extra onnx --extra wireless noema dataset-capture run \
    "$BUNDLE/capture_validation_recipe.yaml" --out "$BUNDLE/data/validation"
  uv run --project "$ROOT" --extra onnx --extra wireless noema dataset-capture run \
    "$BUNDLE/capture_test_recipe.yaml" --out "$BUNDLE/data/test"

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

## 1. Export the training bundle

Open **Limited-feedback MISO-OFDM CSI** from **Template Recipes**, then select **Workbench**.

1. Under **Operation Training Capabilities**, select **Train/replace** for both **Feedback encoder**
   and **Feedback decoder**. Confirm **Portable replacement** is **Yes** for both.
2. Under **Dataset definition** > **Captured signals**, keep `channel_state.csi` selected. The
   example autoencoder uses the same CSI tensor as its reconstruction target.
3. Under **Dataset size and splits**, set:

   - **Total CSI realizations:** `12288`
   - **Train:** `80%`
   - **Validation:** `10%`

   The remaining `10%` is the held-out test split.
4. Under **Training bundle**, set:

   - **Bundle directory:** `.noema/training_exports/csi_feedback`
   - **Support framework:** **PyTorch**
   - **Overwrite generated files:** enable only when rebuilding this bundle

5. Select **Export training bundle**.

Attach the checked-in example codec, loss, trainer, and evaluator from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python demo_trainings/prepare_example.py csi-feedback \
  .noema/training_exports/csi_feedback
```

Return to Workbench. It detects the attached example automatically. Under **Dataset capture**, select
**Capture all datasets** when the action becomes available. Continue when the status is **ready to
train**. The three splits are stored under `.noema/training_exports/csi_feedback/data/`.

## 2. Validate, train, and evaluate

From the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/csi_feedback
uv run --project ../../.. --extra onnx python validate_contract.py
uv run --project ../../.. --extra onnx python train_demo.py
uv run --project ../../.. --extra onnx python evaluate_demo.py
cd ../../..
```

The included CPU-sized trainer begins from a KLT/PCA codec fitted only on the training split. It
then learns a quantization-aware nonlinear residual in two stages: quantized NMSE pretraining,
followed by per-subcarrier beam-direction and MRT-rate fine-tuning. The untouched KLT initialization
remains an eligible checkpoint, so fine-tuning cannot silently return a validation candidate worse
than its own starting point. Training exports two ONNX components in one atomic schema-v2 artifact:

- `encoder`: complex CSI represented by real/imaginary channels to feedback values;
- `decoder`: quantized feedback values to reconstructed CSI.

The returned manifest is `.noema/training_exports/csi_feedback/trained_artifact.yaml`. Return to
**Workbench**; once **External model** detects the returned artifact, select **Validate returned
model**. Continue when the status is **model interface valid**.

The example is not a reproduction of one published network. It keeps the CPU-sized ideas most
relevant to this protocol: learned channel structure from
[CsiNet](https://arxiv.org/abs/1712.08919), multi-resolution residual processing and leaky
activations from [CRNet](https://arxiv.org/abs/1910.14322), and explicit quantization-aware training
as emphasized by [CsiNet+](https://arxiv.org/abs/1906.06007). It does not add the entropy model from
[DeepCMC](https://arxiv.org/abs/1907.02942), because that would change this demonstration's fixed
`32 × 4 = 128`-bit scalar-quantization protocol.

## 3. Run the comparison

Create ordinary working copies of **Limited-feedback MISO-OFDM CSI** and configure:

| Method | Model artifact |
| --- | --- |
| Learned codec | The returned project-trained paired artifact |
| Matched linear baseline | **Matched KLT/PCA · Sionna TDL-A · 128 bit** |
| Optional diagnostic | Truncated angular-delay transform |
| Optional upper bound | `recipes/csi_feedback_perfect_csit_upper_bound.yaml` |

Keep the CSI shape, 32-value bottleneck, 4-bit quantizer, channel distribution, sample count, and
evaluation settings identical. Use the same fresh channel seed for every method at a coordinate and
several seeds that were not used for training or tuning. **Perfect CSIT** is an upper bound, not a
matched-feedback competitor.

Select **Run All**, then compare the methods in **Results**. The primary result is achieved downlink
spectral efficiency, reported with rate retention or loss relative to perfect CSIT. Use CSI NMSE,
phase-invariant correlation, CSI/error heatmaps, quantization MSE, and clipped fraction as
diagnostics.

The checked-in demonstration also provides a reproducible paired campaign. After training and
evaluation, build and run it from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/csi_feedback/reference_training
uv run --project ../../../.. --extra onnx --extra wireless python build_benchmark.py
cd ../../../..
uv run --extra onnx --extra wireless noema benchmark validate \
  .noema/training_exports/csi_feedback/reference_training/benchmark_pack.yaml
uv run --extra onnx --extra wireless noema benchmark run \
  .noema/training_exports/csi_feedback/reference_training/benchmark_pack.yaml
```

## 4. Refit KLT after changing the training data

KLT must be fitted on the same training corpus available to the learned codec. If you regenerate the
captures, copy the following block from any directory inside the clone. It returns to the repository
root before building a new matched reference from the new train and validation directories:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python tools/build_csi_klt_reference_artifact.py \
  --train-capture <train-capture-directory> \
  --validation-capture <validation-capture-directory> \
  --output trained_artifacts/references/klt-sionna-tdl-a-8x32-128bit-v2 \
  --artifact-id noema.csi_feedback.klt_sionna_tdl_a_8x32_128bit_v2 \
  --label "Matched KLT/PCA · Sionna TDL-A · 128 bit · v2" \
  --feedback-dimension 32 \
  --bits-per-latent 4 \
  --overwrite
```

Fit the basis on train data only; use validation data for coefficient-scaling choices. Never fit or
tune on held-out test or benchmark realizations.

## Validity Checks

A comparable demonstration has:

1. disjoint train, validation, held-out-test, and benchmark seeds;
2. one paired encoder/decoder artifact bound atomically;
3. learned and KLT methods evaluated on identical channel realizations at 128 bits per sample;
4. spectral efficiency as the main comparison, with reconstruction metrics as diagnostics;
5. several fresh paired benchmark seeds after all model choices are frozen.

A matched KLT transform can be very strong on a single normalized TDL-A distribution. Any comparison
therefore includes the matched transform rather than only a truncated transform. The included
trainer is small enough for CPU use; the held-out benchmark, rather than training loss, determines
the relative performance.

## Completed benchmark result

The internally validated experimental result
`20260727T125921Z_channel_estimation.learned_csi_feedback_post_training_v1`
contains 60 completed runs: four methods, five downlink SNRs, and three paired held-out channel
seeds per SNR. Each run evaluates 512 fresh TDL-A channel realizations. The bands are two-sided
Student-t 95% confidence intervals over the three paired seeds; they show tutorial-scale
uncertainty and are not a paper-grade population claim.

<div data-noema-chart="csi-feedback-rate"></div>

<div data-noema-chart="csi-feedback-retention"></div>

<div data-noema-chart="csi-feedback-nmse"></div>

```{csv-table} Completed paired benchmark means
:file: ../demo/data/csi_feedback/summary_table.csv
:header-rows: 1
:class: noema-compact-table
```

At the matched 128-bit budget, the learned codec retains `99.13%` of perfect-CSIT rate at 0 dB and
`99.67%` at 20 dB. It gains about `0.091–0.107 bit/s/Hz` over the fixed truncated angular-delay
codec across the sweep. The learned codec and the distribution-fitted KLT/PCA codec are effectively
tied: their mean rate differs by less than `0.00005 bit/s/Hz`, and their mean reconstruction NMSE
differs by about `0.009 dB`. The learned mean is marginally higher, but the paired difference
changes sign across seeds, so these three-seed results do not support a claim that either method
outperforms the other.

The NMSE curves are flat across downlink SNR because the same reconstructed CSI is scored at each
MRT operating SNR. The SNR sweep changes how reconstruction error affects downstream rate; it does
not add observation noise to the CSI codec.

### What one compressed channel looks like

These figures show transmit antenna 0 for sample 0 at 10 dB and paired seed 92001. The response plot
compares the true frequency response with all three 128-bit reconstructions; the error plot shows
the corresponding absolute complex reconstruction error.

<div data-noema-chart="csi-feedback-response-preview"></div>

<div data-noema-chart="csi-feedback-error-preview"></div>

The compact [benchmark projection](../demo/data/csi_feedback/benchmark_projection.csv),
[chart data](../demo/data/csi_feedback/chart_data.json), and
[snapshot manifest](../demo/data/csi_feedback/snapshot_manifest.json) preserve the plotted result,
all 60 run identities, source hashes, returned-artifact evidence, and the representative report
hashes without copying the large runtime tensors into the documentation.

## Verify a Local Result

After the benchmark command prints `RESULT_ID`:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx --extra wireless noema benchmark verify RESULT_ID
```
