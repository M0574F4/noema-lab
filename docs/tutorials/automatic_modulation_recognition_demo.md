# Learned Automatic Modulation Recognition

## Goal

Train an I/Q-only classifier to recognize BPSK, QPSK, and 16-QAM when each received frame has an
unknown carrier phase, residual frequency offset, and AWGN. The learned model replaces only the
**Classifier** block; the waveform source, impairment distribution, class vocabulary, SNR grid,
and evaluation protocol remain fixed.

The comparison contains three methods:

- **Blind differential cumulant:** a deployable I/Q-only classical baseline;
- **Learned blind-carrier classifier:** the returned ONNX model;
- **Oracle-synchronized likelihood:** a diagnostic reference that receives simulator-only carrier
  phase, frequency offset, and noise variance.

The oracle is not a deployable competitor. It shows the performance available when the nuisance
parameters hidden from the two blind methods are known exactly.

## CLI training summary

Run this complete block from the repository. It exports a fresh training bundle, captures disjoint
datasets, trains and evaluates the example model, and runs the paired benchmark.

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/amc_blind_carrier"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/modulation_recognition_awgn.yaml" \
    --training-plan \
    "$ROOT/demo_trainings/modulation_recognition_supervised_cnn/training_plan.yaml" \
    --out "$BUNDLE"
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" modulation-recognition "$BUNDLE" \
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
  uv run --project "$ROOT" --extra onnx python evaluate_demo.py

  cd "$BUNDLE/reference_training"
  uv run --project "$ROOT" --extra onnx python build_benchmark.py
  cd "$ROOT"
  uv run --project "$ROOT" --extra onnx noema benchmark validate \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
  uv run --project "$ROOT" --extra onnx noema benchmark run \
    "$BUNDLE/reference_training/benchmark_pack.yaml"
)
```

Each capture command reports progress for its split. The final command prints the benchmark result
ID and the paths to `result.json`, `metrics.csv`, `recipes.csv`, and `summary.md`.

## 1. Open the reusable scenario

Start Noema from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run noema ui serve --port 8766
```

Open `http://127.0.0.1:8766`, select **Browse template recipes**, and open **Physical layer &
resource optimization** > **Automatic modulation recognition** > **Automatic modulation
recognition with blind carrier impairments**.

The scenario uses:

- 128 complex symbols per frame;
- balanced `bpsk`, `qpsk`, and `qam16` classes;
- SNR coordinates `[-2, 2, 6, 10, 14, 18]` dB;
- carrier phase uniformly distributed over \([-\pi,\pi]\);
- residual frequency offset uniformly distributed over
  \([-0.006,0.006]\) cycles per symbol.

The template opens with the blind differential-cumulant classifier. These system parameters belong
to the recipe; no training architecture or loss is encoded in it.

## 2. Export and capture from Workbench

Open **Workbench** and configure:

1. Under **Operation Training Capabilities**, select **Train/replace** for **Classifier**.
2. Under **Dataset definition** > **Captured signals**, keep:

   - `iq_frames` from `observation.observation`;
   - `modulation_labels` from `data.labels`.

   The labels are offline training targets, not runtime classifier inputs.
3. Under **Capture coordinates**, leave **Parameter sweep** empty. The recipe matrix already
   supplies the six SNR coordinates.
4. Under **Dataset size and splits**, set:

   - **Total modulation frames:** `10368`;
   - **Train:** `66.6667%`;
   - **Validation:** `16.6667%`;
   - **Held-out test:** `16.6666%`.

5. Under **Training bundle**, set:

   - **Bundle directory:** `.noema/training_exports/amc_blind_carrier`;
   - **Support framework:** **PyTorch**.

6. Select **Export training bundle**.

Attach the checked-in example:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python demo_trainings/prepare_example.py modulation-recognition \
  .noema/training_exports/amc_blind_carrier
```

Return to **Workbench**, select **Capture all datasets**, and wait until train, validation, and
held-out test contain 6,912, 1,728, and 1,728 frames. These counts divide evenly across the six
SNR coordinates.

## 3. Train and return the classifier

From the bundle directory:

```bash
cd "$(git rev-parse --show-toplevel)/.noema/training_exports/amc_blind_carrier"
uv run --project ../../.. --extra onnx python validate_contract.py
uv run --project ../../.. --extra onnx python train_demo.py
uv run --project ../../.. --extra onnx python evaluate_demo.py
```

The model receives only `[batch, 128, 2]` real-valued I/Q frames. It does not receive SNR, carrier
phase, frequency offset, or oracle synchronization. Its example architecture uses communication
knowledge without changing the generic runtime interface:

- per-frame power normalization removes an irrelevant amplitude scale;
- adjacent-symbol differential features cancel constant carrier phase and expose modulation-order
  structure under residual frequency offset;
- second- and fourth-order local products describe BPSK/QPSK rotational symmetries;
- the differential-cumulant rule supplies an explicit classical prior;
- a smaller, dropout-regularized temporal CNN learns residual corrections;
- random carrier rotations augment the training frames, and validation-loss early stopping can
  retain the untrained prior if a learned correction does not generalize.

This compact design combines the low-SNR strength of raw-I/Q temporal convolutional
classifiers described by [O'Shea et al.](https://arxiv.org/abs/1602.04105) with the
phase-invariant signal structure used by the deployable cumulant baseline. Rotation augmentation
is supported by the AMC study of [Huang et al.](https://arxiv.org/abs/1912.03026); the residual
path follows the broader finding that structured complex/residual models improve high-SNR
classification without requiring a much deeper network
([Krzyston et al.](https://arxiv.org/abs/2010.10717)).

The returned files are:

- `artifacts/modulation_classifier.onnx`;
- `trained_artifact.yaml`;
- `reference_training/training_history.json`;
- `reference_training/evaluation_metrics.json`.

Workbench detects the returned artifact automatically. Select **Validate returned model** and
continue when the status is **model interface valid**.

## 4. Run the paired comparison

The checked-in benchmark uses six SNR coordinates, three fresh paired seeds, three methods, and
1,536 balanced frames per method and coordinate. At each coordinate, all methods receive the same
transmitted symbols, carrier impairments, and AWGN realization.

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/amc_blind_carrier/reference_training
uv run --project ../../../.. --extra onnx python build_benchmark.py
cd ../../../..
uv run --extra onnx noema benchmark validate \
  .noema/training_exports/amc_blind_carrier/reference_training/benchmark_pack.yaml
uv run --extra onnx noema benchmark run \
  .noema/training_exports/amc_blind_carrier/reference_training/benchmark_pack.yaml
```

The primary metric is balanced accuracy. Accuracy and macro F1 are companion metrics; the
per-method confusion matrices show which modulation pairs remain ambiguous.

## Validity Checks

A comparable demonstration requires:

1. balanced class support and disjoint train, validation, test, and benchmark records;
2. identical paired payload, noise, and carrier conditions across methods;
3. an oracle-synchronized likelihood reference labeled as non-deployable;
4. complete classwise denominators and confusion matrices;
5. one frozen artifact hash across every learned benchmark run.

The demonstration is CPU-sized and controlled. It establishes the full Noema
replacement and benchmarking workflow; it is not a RadioML or over-the-air performance claim.

## Completed benchmark result

The internally validated experimental result
`20260727T150344Z_neural_receiver_ai_phy.learned_modulation_recognition_blind_carrier_post_training_v1`
contains 54 completed runs: three methods, six SNRs, and three paired held-out seeds per SNR. Each
run classifies 1,536 balanced frames. The bands are two-sided Student-t 95% confidence intervals
over the three paired seeds and show tutorial-scale uncertainty.

<div data-noema-chart="modulation-recognition-accuracy"></div>

<div data-noema-chart="modulation-recognition-macro-f1"></div>

```{csv-table} Completed paired benchmark means
:file: ../demo/data/modulation_recognition/summary_table.csv
:header-rows: 1
:class: noema-compact-table
```

The learned classifier improves substantially on the deployable blind cumulant baseline throughout
the difficult −2 to 10 dB range. Its mean accuracy rises from `42.1%` at −2 dB to `82.2%` at
6 dB and `97.6%` at 10 dB; the cumulant baseline obtains `33.3%`, `33.3%`, and `66.9%` at those
coordinates. At 14 and 18 dB the cumulant and oracle methods reach `100%`, while the learned model
reaches `99.5%` and `99.7%`. The remaining errors are a small number of 16-QAM frames classified
as QPSK, so this result does not claim that learning dominates the classical rule at every SNR.
The oracle-synchronized likelihood remains the diagnostic upper reference because it receives the
simulator-only carrier phase and frequency offset.

The compact [benchmark projection](../demo/data/modulation_recognition/benchmark_projection.csv),
[chart data](../demo/data/modulation_recognition/chart_data.json), and
[snapshot manifest](../demo/data/modulation_recognition/snapshot_manifest.json) preserve the
plotted values, run identities, source hashes, paired-seed protocol, and training-evidence binding
without copying runtime tensors into the documentation.

## Verify a Local Result

After the benchmark prints `RESULT_ID`:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx noema benchmark verify RESULT_ID
```
