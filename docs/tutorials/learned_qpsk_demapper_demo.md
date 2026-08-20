# Learned QPSK Receiver Calibration

## Goal

Train a receiver that maps one impaired QPSK symbol, represented as `[I, Q]`, to
two bit logits. The scenario applies AWGN followed by one stable receiver
front-end distortion: I/Q gain imbalance, quadrature error, carrier-phase
offset, and DC offset.

The paired benchmark compares:

- **Uncompensated QPSK:** the ordinary `I=0` and `Q=0` decision axes;
- **Calibrated I/Q oracle:** knows the simulated front-end transform and inverts it;
- **Learned I/Q receiver:** sees only captured impaired I/Q samples.

This is a data-driven calibration example. A successful learned receiver should
approach the calibrated oracle and outperform the mismatched uncompensated
detector, especially at high SNR.

## Watch the end-to-end workflow

```{include} ../_includes/launch_video.md
```

## CLI training summary

Run this complete block from any directory inside the repository. `BUNDLE` must
not already exist; choose another name when keeping an older run.

```bash
(
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/.noema/training_exports/qpsk_iq_calibration"
cd "$ROOT"

uv run --extra onnx noema differentiable export \
  "$ROOT/recipes/neural_receiver_qpsk_iq_calibration.yaml" \
  --training-plan "$ROOT/demo_trainings/neural_receiver_supervised_qpsk/training_plan.yaml" \
  --out "$BUNDLE"
uv run --project "$ROOT" --extra onnx python \
  "$ROOT/demo_trainings/prepare_example.py" neural-receiver "$BUNDLE" \
  --project-root "$ROOT"

uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_train_recipe.yaml" --out "$BUNDLE/data/train"
uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_validation_recipe.yaml" --out "$BUNDLE/data/validation"
uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_test_recipe.yaml" --out "$BUNDLE/data/test"

cd "$BUNDLE"
uv run --project "$ROOT" --extra onnx python validate_contract.py
uv run --project "$ROOT" --extra onnx python train_demo.py
uv run --project "$ROOT" --extra onnx python evaluate_demo.py --summary

cd "$BUNDLE/reference_training"
uv run --project "$ROOT" --extra onnx python build_benchmark.py
cd "$ROOT"
uv run --extra onnx noema benchmark validate \
  "$BUNDLE/reference_training/benchmark_pack.yaml" --summary
uv run --extra onnx noema benchmark run \
  "$BUNDLE/reference_training/benchmark_pack.yaml"
)
```

Each capture command displays progress for its split. Add `--force` only when
intentionally replacing an existing capture directory.

## 1. Open the template

Start Noema from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python --version
uv run noema ui serve --port 8766
```

Python must be 3.11 through 3.13. Open `http://127.0.0.1:8766`, select **Browse
template recipes**, then open **Physical layer & resource optimization** >
**Neural receiver demapping** > **QPSK receiver calibration under I/Q
imbalance**.

The recipe evaluates `-2`, `2`, `6`, and `10` dB. Its **Receiver I/Q
front-end impairment** block holds the stable device parameters; the learned
receiver does not receive those parameters.

## 2. Export and capture from Workbench

Open **Workbench** and configure:

1. Under **Operation Training Capabilities**, select **Train/replace** for
   **Demodulator**.
2. Under **Dataset definition** > **Captured signals**, select:
   - `receiver_frontend_rx_symbols` (`receiver_frontend.rx_symbols`) as input;
   - `tx_bit_boundary_bits` (`tx_bit_boundary.bits`) as target.
3. Under **Capture coordinates**, leave **Parameter sweep** blank. The template
   already defines `-2`, `2`, `6`, and `10` dB as its recipe matrix.
4. Set **Total recipe records** to `48`, **Train %** to `66.6667`, and
   **Validation %** to `16.6667`.
5. Set **Bundle directory** to
   `.noema/training_exports/qpsk_iq_calibration`, select **PyTorch**, and
   enable **Overwrite generated files** only when rebuilding.
6. Select **Export training bundle**.

Attach the checked-in example trainer from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python demo_trainings/prepare_example.py neural-receiver \
  .noema/training_exports/qpsk_iq_calibration
```

Return to **Workbench** > **Dataset capture**, select **Capture all datasets**,
and wait for train, validation, and held-out test to finish.

## 3. Train, evaluate, and return the model

```bash
cd "$(git rev-parse --show-toplevel)/.noema/training_exports/qpsk_iq_calibration"
uv run --project ../../.. --extra onnx python validate_contract.py
uv run --project ../../.. --extra onnx python train_demo.py
uv run --project ../../.. --extra onnx python evaluate_demo.py --summary
```

The example uses one affine calibration model because the configured front-end
distortion is a fixed invertible affine transform. It estimates an initial pair
of decision lines from captured I/Q and bit labels, then converges the logistic
loss with full-batch L-BFGS. This matched model cannot invent unnecessary curved
or wavy regions between finite samples. Training uses transmitted bits only as
offline labels; runtime input remains `[I,Q]`.

The returned files are:

- `artifacts/neural_receiver.onnx`;
- `trained_artifact.yaml`;
- `reference_training/training_history.json`;
- `reference_training/evaluation_metrics.json`.

`evaluation_metrics.json` also reports
`decision_boundary_agreement.symbol_region_agreement_rate`, measured against
the simulation-only oracle on a dense held-out I/Q plane. This separates
boundary geometry from finite-test BER fluctuations.

Workbench detects the returned artifact automatically. Select **Validate
returned model**. For a recipe smoke run, set **Demodulator** > **Mode** to
**Learned model** and select **Learned receiver ·
qpsk_iq_imbalance_receiver_calibration**.

## 4. Run the paired benchmark

```bash
cd "$(git rev-parse --show-toplevel)/.noema/training_exports/qpsk_iq_calibration/reference_training"
uv run --project ../../../.. --extra onnx python build_benchmark.py
cd ../../../..

uv run --extra onnx noema benchmark validate \
  .noema/training_exports/qpsk_iq_calibration/reference_training/benchmark_pack.yaml \
  --summary
uv run --extra onnx noema benchmark run \
  .noema/training_exports/qpsk_iq_calibration/reference_training/benchmark_pack.yaml
```

The campaign pairs payload bits, AWGN realizations, front-end parameters, and
SNR across all three receiver methods. The default uses 1,048,576 bits per run
and three held-out seeds.

In **Results** > **Communication**, the decision-boundary figure overlays the
three methods on one received-I/Q plane. Hover or focus a legend entry to
highlight its boundary. The three model classes have distinct geometry:

- uncompensated: horizontal and vertical axes through the origin;
- calibrated oracle: shifted and rotated affine boundaries;
- learned: boundaries close to the oracle.

Use the BER curve to measure whether the uncompensated detector has an error
floor and whether either calibrated receiver reduces it. The oracle is a
diagnostic reference, not a deployable competitor: its hidden calibration
parameters are never passed to the learned runtime.

## Completed benchmark result

Result
`20260726T155403Z_neural_receiver_ai_phy.learned_qpsk_iq_calibration_v1`
contains 63 completed runs: three receiver methods, seven SNR values, and three
paired held-out seeds. Every run compares 1,048,576 bits. The shaded ranges
below show the minimum and maximum BER across the three seeds; each line is
their arithmetic mean.

<div data-noema-chart="qpsk-iq-calibration-ber"></div>

Completed paired benchmark for the fixed receiver I/Q impairment. Hover over a
point for exact values or over a method name to isolate it visually.

```{csv-table} Mean BER and observed learned-receiver range
:file: ../demo/data/qpsk_iq_calibration/ber_table.csv
:header-rows: 1
:align: center
```

<div data-noema-chart="qpsk-iq-calibration-boundaries"></div>

The same method titles and colors are used in the BER and boundary figures.
The markers are the four noiseless received constellation centroids after the
fixed front-end impairment. The learned boundary follows the calibrated oracle
instead of the mismatched horizontal and vertical uncompensated axes.

The learned receiver remains close to the calibrated oracle while materially
outperforming the uncompensated detector. Its mean BER reduction relative to
uncompensated QPSK grows from 16.18% at `-2` dB to 98.87% at `10` dB. Across
the seven SNR cells, its mean BER differs from the oracle by at most 0.25%
relative; at `10` dB the absolute means are `7.51e-4` for the learned receiver
and `7.50e-4` for the oracle. At `-2` dB the learned mean is lower by 0.01%
relative, a negligible finite-sample reversal rather than evidence of
outperforming the oracle. In this completed run, the matched affine model
recovers an unknown, stable receiver calibration from labeled impaired I/Q
samples without receiving the hidden impairment parameters at runtime.

BLER is not used for the main figure because 1,024-bit blocks make it saturate
near one over most of this SNR range. BER is the more informative metric for
this uncoded calibration experiment.

The [per-run projection](../demo/data/qpsk_iq_calibration/benchmark_projection.csv)
and
[provenance manifest](../demo/data/qpsk_iq_calibration/benchmark_manifest.json)
preserve all run IDs, paired seeds, BER/BLER counts, recipe hashes, the trained
artifact runtime identity, and hashes of the original completed result files.
The result passes Noema's bundle-integrity checks. It remains explicitly
marked experimental and is not yet designated as a publication-ready canonical
benchmark.

## 5. Verify a Local Result

Replace `<result_id>` with the ID printed by the benchmark run:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx noema benchmark verify <result_id>
```

A comparable result has disjoint capture splits, paired benchmark conditions,
one unchanged artifact hash, complete BER denominators, and explicit
aggregation-cell and statistical-unit declarations. Performance relative to
the uncompensated detector and oracle is an outcome, not a validity condition.
