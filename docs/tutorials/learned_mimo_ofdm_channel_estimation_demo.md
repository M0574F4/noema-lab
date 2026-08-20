# Learned 2×2 MIMO-OFDM Channel Estimation

## Goal

Train a portable estimator for a `2×2`, 64-subcarrier MIMO-OFDM link with sparse
orthogonal comb pilots. Training and benchmarking mix Sionna 3GPP TDL-A,
TDL-C, and TDL-E profiles at 3.5 GHz, 30 kHz subcarrier spacing, 300 ns RMS
delay spread, and 30 km/h mobility. The receiver is not told which profile
generated a realization.

Noema owns pilot division and frequency interpolation. The returned model gets:

- sparse pilot LS values before interpolation, shaped
  `[batch, rx, tx, subcarrier, 2]`;
- the binary pilot mask, shaped `[batch, tx, subcarrier]`;
- the operation-owned LS estimate, shaped `[batch, rx, tx, subcarrier, 2]`;
- one complex-noise variance per channel realization.

It predicts the same real/imaginary channel tensor. Simulated channel truth is
an offline training target and never enters the returned runtime ABI.

## CLI training summary

Run this block from any directory inside the repository. It recreates the
generated bundle and all captures, trains and evaluates the example model, and
builds the paired benchmark.

```bash
(
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/.noema/training_exports/mimo_ofdm_channel_estimation"
cd "$ROOT"

uv sync --extra wireless --extra onnx
uv run --project "$ROOT" --extra wireless --extra onnx noema differentiable export \
  "$ROOT/recipes/mimo_ofdm_adapter_channel_estimation.yaml" \
  --training-plan "$ROOT/demo_trainings/mimo_ofdm_channel_estimation_cnn/training_plan.yaml" \
  --out "$BUNDLE" --force
uv run --project "$ROOT" --extra wireless --extra onnx python \
  "$ROOT/demo_trainings/prepare_example.py" mimo-ofdm-channel-estimation "$BUNDLE" \
  --project-root "$ROOT"

uv run --project "$ROOT" --extra wireless --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_train_recipe.yaml" --out "$BUNDLE/data/train" --force
uv run --project "$ROOT" --extra wireless --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_validation_recipe.yaml" --out "$BUNDLE/data/validation" --force
uv run --project "$ROOT" --extra wireless --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_test_recipe.yaml" --out "$BUNDLE/data/test" --force

cd "$BUNDLE"
uv run --project "$ROOT" --extra wireless --extra onnx python validate_contract.py
uv run --project "$ROOT" --extra wireless --extra onnx python train_demo.py
uv run --project "$ROOT" --extra wireless --extra onnx python evaluate_demo.py

cd "$BUNDLE/reference_training"
uv run --project "$ROOT" --extra wireless --extra onnx python build_benchmark.py
cd "$ROOT"
uv run --project "$ROOT" --extra wireless --extra onnx noema benchmark validate \
  "$BUNDLE/reference_training/benchmark_pack.yaml"
uv run --project "$ROOT" --extra wireless --extra onnx noema benchmark run \
  "$BUNDLE/reference_training/benchmark_pack.yaml"
)
```

Each capture command displays its own progress. The full workflow requires
Python 3.11 through 3.13.

## 1. Export and capture from Workbench

Open **Browse template recipes** > **Channel estimation & CSI feedback** >
**MIMO-OFDM channel estimation** > **Learned 2×2 MIMO-OFDM channel
estimation**, then select **Workbench**:

1. Under **Operation Training Capabilities**, select **Train/replace** for
   **Channel estimator**.
2. Under **Dataset definition** > **Captured signals**, keep these selected:

   - `pilot_observations`: the estimator block's recipe-boundary input;
   - `pilot_ls`: divided sparse-pilot values before interpolation;
   - `pilot_mask`: locations of the active pilots;
   - `ls_estimate`: the operation-owned interpolated fallback;
   - `noise_variance`: runtime conditioning;
   - `channel_truth`: the offline supervised target.

3. Leave **Parameter sweep** blank. The template matrix already defines
   TDL-A/C/E at `-5, 0, 5, 10, 15, 20` dB.
4. Set **Total recipe records** to `18432`, **Train %** to `66.6667`, and
   **Validation %** to `16.6667`.
5. Set **Bundle directory** to
   `.noema/training_exports/mimo_ofdm_channel_estimation`, select **PyTorch**,
   and export the training bundle.
6. From the project root, attach the checked-in example:

   ```bash
   cd "$(git rev-parse --show-toplevel)"
   uv run --extra wireless --extra onnx python demo_trainings/prepare_example.py \
     mimo-ofdm-channel-estimation \
     .noema/training_exports/mimo_ofdm_channel_estimation
   ```

7. Return to **Dataset capture**, select **Capture all datasets**, and wait for
   train `12288`, validation `3072`, and held-out test `3072`.

## 2. Train and return the estimator

The current reference model is a compact, noise-conditioned residual network.
Its frequency branch combines sparse pilots, their mask, and interpolated LS
with dilated circular convolutions. A second branch applies fixed real-valued
DFT transforms and refines the delay-domain representation. The output starts
exactly at LS.

The residual parameterization does not prevent full channel estimation. The
network can learn a residual equal to `channel truth − LS`, canceling the input
completely when needed. Keeping the LS anchor gives training and checkpoint
selection a safe fallback. Validation compares every SNR cell and aggregate
NMSE with both LS and the fixed-prior LMMSE before selecting among two compact
widths and two seeds. The final training line reports the aggregate gap and
the number of SNR bins won against the stronger classical baseline.

Run from the exported bundle:

```bash
cd "$(git rev-parse --show-toplevel)/.noema/training_exports/mimo_ofdm_channel_estimation"
uv run --project ../../.. --extra wireless --extra onnx python validate_contract.py
uv run --project ../../.. --extra wireless --extra onnx python train_demo.py
uv run --project ../../.. --extra wireless --extra onnx python evaluate_demo.py
```

The trainer returns `artifacts/channel_estimator.onnx` and
`trained_artifact.yaml`. Workbench detects the returned artifact; select
**Validate returned model**. For an ordinary run, set **Channel estimator** >
**Reference method** to **Learned artifact** and select **Learned estimator ·
2×2 mixed TDL**.

## 3. Compare estimators

The paired campaign uses identical held-out TDL profile, channel realization,
pilots, and observation noise for:

- least squares with frequency interpolation;
- fixed-prior linear MMSE, which assumes the same four-tap exponential PDP for
  every TDL-A/C/E realization;
- the returned learned dual-domain estimator.

The default post-training campaign has 54 runs: three methods, six SNRs, and
three profile/seed units (TDL-A, TDL-C, and TDL-E). This keeps the demo
lightweight while retaining common-random-number pairing.

Perfect channel knowledge is not a fourth deployable estimator. It appears
only as the diagnostic ZF spectral-efficiency ceiling.

In **Results**, compare:

- NMSE in dB versus SNR;
- post-ZF spectral efficiency versus SNR;
- estimated-CSI ZF rate relative to exact-channel ZF;
- true, estimated, and absolute-error channel heatmaps.

This is a practical covariance-mismatch problem, not a comparison with an
exact covariance-aware MMSE oracle. Compare LS, fixed-prior LMMSE, and the
learned estimator without assuming their order. When NMSE improves but
post-ZF rate does not, the per-link heatmaps can reveal errors concentrated
on channel modes that ZF amplifies.

## 4. Completed benchmark result

The frozen experimental result
`20260726T235356Z_mimo_ofdm.learned_channel_estimation_post_training_v2`
contains 54 completed runs: three estimators, six SNRs, and three paired
profile/channel units per SNR: TDL-A with seed 91001, TDL-C with seed 92001,
and TDL-E with seed 93001. The bands are two-sided Student-t 95% confidence
intervals over those three heterogeneous units, so they are tutorial-scale
uncertainty indicators rather than population claims.

<div data-noema-chart="mimo-channel-estimation-nmse"></div>

<div data-noema-chart="mimo-channel-estimation-profile-gain"></div>

<div data-noema-chart="mimo-channel-estimation-zf-rate"></div>

<div data-noema-chart="mimo-channel-estimation-zf-retention"></div>

<div data-noema-chart="mimo-channel-estimation-response-preview"></div>

```{csv-table} Completed paired benchmark means
:file: ../demo/data/mimo_ofdm_channel_estimation/summary_table.csv
:header-rows: 1
:class: noema-compact-table
```

The learned estimator has the lowest NMSE in every one of the 18 paired
profile/SNR cells. After averaging the three profile/channel units, its NMSE
advantage over the stronger classical baseline ranges from `0.50` dB at
20 dB SNR to `3.18` dB at 10 dB SNR. The independent held-out capture
evaluation also reports a `1.82` dB aggregate NMSE improvement over the
fixed-prior LMMSE.

The post-ZF result is intentionally reported separately. Learned estimation
has the highest mean post-ZF spectral efficiency at 10, 15, and 20 dB. At
−5, 0, and 5 dB, a classical estimate gives a slightly higher mean ZF rate
despite having worse NMSE. This illustrates that minimizing channel MSE does
not guarantee the best downstream equalizer at every operating point.

The exact-channel ZF curve is a diagnostic, not a capacity oracle. Its rate
ratio can exceed one at low SNR because estimation error can accidentally
regularize an otherwise noise-amplifying zero-forcing inverse.

The compact [benchmark projection](../demo/data/mimo_ofdm_channel_estimation/benchmark_projection.csv),
[chart data](../demo/data/mimo_ofdm_channel_estimation/chart_data.json), and
[snapshot manifest](../demo/data/mimo_ofdm_channel_estimation/snapshot_manifest.json)
preserve the documentation result without copying the benchmark's large run
artifacts.

## 5. Verify and publish

After `noema benchmark run` prints `RESULT_ID`:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run noema benchmark verify RESULT_ID
uv run noema benchmark publish RESULT_ID \
  --slug learned-mimo-ofdm-channel-estimation \
  --out docs/demo/experiments/learned-mimo-ofdm-channel-estimation
```
