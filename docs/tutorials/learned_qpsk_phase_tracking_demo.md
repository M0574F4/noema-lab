# Learned QPSK Carrier Tracking

## Goal

Train a packet-context receiver for pilot-aided QPSK with unknown packet phase, residual carrier
frequency offset, Wiener phase noise, and AWGN. At frame position `t`, the receiver observes

`y[t] = x[t] exp(j phi[t]) + noise`,

where `x[t]` is the transmitted QPSK symbol and `phi[t]` is a phase trajectory that drifts across
the packet. The model estimates the phase correction at every symbol, derotates `y[t]`, and then
uses the ordinary QPSK sign decisions. Its runtime inputs are the complete received I/Q packet and
the public pilot mask and pilot values; simulator phase truth is used only as a training target.

The returned ONNX model is compared with six receivers using the same payload, noise, and
carrier-impairment seeds:

| Receiver | Information used at inference | Role |
| --- | --- | --- |
| Uncompensated QPSK | Received I/Q | Weak practical baseline |
| Pilot interpolation | Received I/Q and public pilots | Feed-forward classical tracker |
| Pilot smoothing | Received I/Q and public pilots | Denoised local-linear classical tracker |
| Decision-directed PLL | Received I/Q and public pilots | Adaptive classical tracker |
| Learned temporal receiver | Received I/Q and public pilots | Trainable method under study |
| True-phase correction | Received I/Q and simulator phase truth | Non-deployable expected-BER reference |

The research question is whether a learned packet-context detector can use the same public pilot
information as the practical trackers to improve carrier-impaired reception. It is not expected to
systematically beat a receiver that receives hidden simulator phase truth.

The reusable system template is the built-in **Pilot-aided QPSK carrier tracking** entry under
**Template Recipes** > **Physical layer & resource optimization** > **Neural receiver demapping**.
Its [version-controlled recipe source](../../recipes/neural_receiver_qpsk_phase_tracking.yaml)
describes the communication system independently of the example model and trainer; users normally
open it from **Template Recipes** rather than loading the YAML file manually.

## CLI training summary

For a terminal-only run, `BUNDLE` must name a directory that does not already exist; change it before
running this block if necessary. The numbered walkthrough below remains the Workbench alternative.
Each `dataset-capture` command reports progress for its own split; let it complete before the next
command starts. Add `--force` only when intentionally replacing an existing capture.

```bash
(
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/.noema/training_exports/qpsk_phase_tracking"
cd "$ROOT"

# This exact reproducibility block requires a Noema source checkout.
uv sync --extra onnx
uv run --project "$ROOT" --extra onnx noema differentiable export \
  "$ROOT/recipes/neural_receiver_qpsk_phase_tracking.yaml" \
  --training-plan "$ROOT/demo_trainings/neural_receiver_phase_tracking_qpsk/training_plan.yaml" \
  --out "$BUNDLE"
uv run --project "$ROOT" --extra onnx python \
  "$ROOT/demo_trainings/prepare_example.py" phase-tracking-receiver "$BUNDLE" \
  --project-root "$ROOT"

uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_train_recipe.yaml" \
  --out "$BUNDLE/data/train"
uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_validation_recipe.yaml" \
  --out "$BUNDLE/data/validation"
uv run --project "$ROOT" --extra onnx noema dataset-capture run \
  "$BUNDLE/capture_test_recipe.yaml" \
  --out "$BUNDLE/data/test"

cd "$BUNDLE"
uv run --project "$ROOT" --extra onnx python validate_contract.py
uv run --project "$ROOT" --extra onnx python train_demo.py
uv run --project "$ROOT" --extra onnx python evaluate_demo.py
)
```

## Scenario

| Property | Demo setting |
| --- | --- |
| Payload | 1,024 data bits per packet |
| Modulation | Pilot-aided QPSK |
| Pilot pattern | 16-symbol preamble, then one pilot after every 16 data symbols |
| Initial phase | Uniform over `[-pi, pi]` |
| Residual CFO | Uniform over `[-0.01, 0.01]` cycles per symbol |
| Phase noise | Wiener increments with standard deviation `0.04` rad per symbol |
| Template SNR sweep | `-2, 2, 6, 10` dB with fixed unit-power QPSK and varying AWGN variance |
| Benchmark SNR | `-2, 0, 2, 4, 6, 8, 10` dB |
| Paired evaluation | Three held-out payload, AWGN, and carrier-impairment seeds |
| Metrics | Data-bit BER and packet BLER |

Raw interpolation is noisy because it forces the estimate through every noisy pilot. The learned
receiver therefore starts from a five-nearest-pilot local-linear smoother and predicts a
full-strength circular correction at every symbol. This is not a restriction to a small residual:
`final phase = smoother phase + learned correction`, and the correction can span the complete
`[-pi, pi]` circle, so it can replace the smoother modulo one carrier cycle.

The model receives eleven observable features per symbol: smoother-corrected I/Q, raw I/Q, the
pilot mask, pilot innovation, the smoother phasor, and a QPSK fourth-power cue. The known pilots
anchor the absolute QPSK quadrant; the temporal trend reveals residual frequency offset; and the
fourth-power cue cancels the unknown QPSK data symbols, exposing phase motion modulo 90 degrees.
A temporal convolutional network combines those cues across the packet. It returns only the phase
correction; Noema derives bit scores deterministically after derotation. Training begins with
circular phase supervision before optimizing the data-bit objective.

## 1. Open the template

From the repository root, check the environment and start Noema:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python --version
uv run noema ui serve --port 8766
```

Python must be 3.11 through 3.13. Open `http://127.0.0.1:8766`, select **Browse template recipes**, then
open **Physical layer & resource optimization** > **Neural receiver demapping** >
**Pilot-aided QPSK carrier tracking**.

The template's four-point **Run All** sweep keeps the transmitted QPSK power and carrier
impairment distribution fixed and changes only the AWGN variance through
`wireless_channel.snr_db`. This is the conventional BER-versus-SNR experiment for a normalized
constellation. The denser seven-point grid used by the final benchmark is generated in step 5 as
concrete recipes; it does not reuse or nest the template's interactive matrix.

## 2. Export the neutral bundle

Open **Workbench** and configure these fields in order:

1. Under **Operation Training Capabilities**, find **Demodulator**, confirm **Portable replacement**
   is **Yes**, and select **Train/replace**.
2. Under **Dataset definition** > **Captured signals**, keep the two required model inputs selected
   and select the target:

   - `carrier_impairment_rx_symbols` (`carrier_impairment.rx_symbols`): impaired received I/Q;
   - `modulator_pilot_context` (`modulator.pilot_context`): public pilot mask and known pilot
     values;
   - `tx_bit_boundary_bits` (`tx_bit_boundary.bits`): transmitted data bits.

   Attaching the example in the next section adds the simulator's carrier-phase trace to the
   generated capture plan as a training-only auxiliary target. You do not need to select that
   demo-specific signal manually, and it is never a runtime input to the learned model.
3. Under **Capture coordinates**, leave **Parameter sweep** blank. The built-in template already
   supplies the four SNR coordinates through its recipe matrix; defining a second capture sweep
   would be ambiguous.
4. Under **Dataset size and splits**, set **Total recipe records** to `1536`, **Train %** to
   `66.6667`, and **Validation %** to `16.6667`. This gives 1,024 train, 256 validation, and 256 held-out
   test records; this template emits one 1,024-bit packet per record.
5. Under **Training bundle**, set:

   - **Bundle directory:** `.noema/training_exports/qpsk_phase_tracking`
   - **Support framework:** **PyTorch**
   - **Overwrite generated files:** enabled when intentionally rebuilding the bundle

6. Select **Export training bundle**.

The bundle defines only the replacement interface and capture plan. The learned ABI accepts
`receiver_features_v3[packet, frame_symbol, 11]` and returns
`residual_phase_rad[packet, frame_symbol]`. Noema applies that phase correction and computes QPSK
bit scores; carrier-phase truth is not an ABI input.

## 3. Attach the example and capture data

From the repository root, attach the checked-in temporal-convolution candidates, loss, trainer,
evaluator, and benchmark builder:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python demo_trainings/prepare_example.py phase-tracking-receiver \
  .noema/training_exports/qpsk_phase_tracking
```

The helper adds `carrier_impairment.phase_truth` only to this external demo's training plan, keeps
the recipe matrix as the single SNR-variant definition, and regenerates the Noema-owned capture
recipes. It does not change the reusable system recipe or the returned model ABI.

Return to **Workbench**. It detects the attached example automatically. Under **Dataset capture**,
select **Capture all datasets** when the action becomes available, then wait until train, validation,
and held-out test show **1,024**, **256**, and **256** captured records respectively.

The captures are stored under
`.noema/training_exports/qpsk_phase_tracking/data/{train,validation,test}`.

## 4. Train, evaluate, and validate

Run the contract check and trainer from the bundle directory:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/qpsk_phase_tracking
uv run --project ../../.. --extra onnx python validate_contract.py
uv run --project ../../.. --extra onnx python train_demo.py
```

Wait for training to finish successfully and create `trained_artifact.yaml`. Only then run the
held-out evaluator:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/qpsk_phase_tracking
uv run --project ../../.. --extra onnx python evaluate_demo.py
cd ../../..
```

Training uses `data/train` and `data/validation` for checkpoint selection. Evaluation opens
`data/test` once. The returned files are:

- `.noema/training_exports/qpsk_phase_tracking/artifacts/phase_tracking_receiver.onnx`;
- `.noema/training_exports/qpsk_phase_tracking/trained_artifact.yaml`;
- `.noema/training_exports/qpsk_phase_tracking/reference_training/training_history.json`;
- `.noema/training_exports/qpsk_phase_tracking/reference_training/evaluation_metrics.json`.

Return to **Workbench**. Once **External model** detects the returned artifact, select **Validate
returned model**. Continue when the status is **model interface valid**.

### Optional single-recipe smoke test

To check the returned model in the currently open recipe, open **Graph** > **Demodulator**, set
**Mode** to **Learned artifact**, then under **Trained artifact** > **Project-trained** select
**Learned phase tracker · qpsk_pilot_phase_tracking**. Select **Run All**, then open **Results**.
The artifact selector is used only in **Learned artifact** mode and remains unavailable for the
five non-learned receiver modes.

This smoke test is optional and is not how the six-method benchmark is configured. The benchmark
builder in the next section reads `trained_artifact.yaml` directly and creates all six receiver
variants, including the learned-artifact binding. You therefore do not need to switch modes or
select an artifact manually before building or running the benchmark campaign.

## 5. Run the paired six-method benchmark

Build the campaign from `reference_training`, then validate and run it from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/qpsk_phase_tracking/reference_training
uv run --project ../../../.. --extra onnx python build_benchmark.py
cd ../../../..

uv run --extra onnx noema benchmark validate \
  .noema/training_exports/qpsk_phase_tracking/reference_training/benchmark_pack.yaml
uv run --extra onnx noema benchmark run \
  .noema/training_exports/qpsk_phase_tracking/reference_training/benchmark_pack.yaml
```

Record the printed `result_id`. Reload the dashboard if needed, open **Results**, then select the new
entry under **Open** > **Benchmark results**.

- **Overview** opens with receiver BER versus SNR.
- **Performance** shows paired BER and BLER curves for all six methods.
- **Communication** shows carrier-phase truth and the explicit estimates produced by the classical
  trackers, learned phase tracker, and oracle.

The oracle uses simulator truth only to define a non-deployable expected-BER reference. Exact phase
correction is optimal in expectation, but its measured BER need not be the smallest on every short,
finite noise realization. The learned runtime receives only I/Q and public pilots. The benchmark
compares it with both practical trackers; any reported learned gain must come from the stored paired
evidence, not access to simulator truth.

## Completed paired benchmark

Result
`20260727T005814Z_neural_receiver_ai_phy.learned_qpsk_phase_tracking_v2`
contains 126 completed runs: six receivers, seven SNR values from `-2` to
`10` dB, and three held-out paired seeds. Each run compares 262,144 data
bits. At a given SNR and seed, every receiver uses the same payload, AWGN,
carrier impairment, and public pilots.

<div data-noema-chart="qpsk-phase-tracking-ber"></div>

The learned temporal receiver has lower mean BER than every deployable
classical tracker at all seven SNR coordinates. Pilot smoothing is the
strongest classical tracker throughout the sweep; the learned BER reduction
relative to it grows from about 7.9% at `-2` dB to 57.8% at `10` dB. In this
completed result, the simulator-truth oracle remains better than the learned
receiver at every coordinate.

```{csv-table} Mean paired BER and learned improvement
:file: ../demo/data/qpsk_phase_tracking/paired_summary_table.csv
:header-rows: 1
:align: center
```

Packet BLER is retained in the benchmark evidence but is not used as the
headline figure: with 1,024-bit packets it is saturated across much of this
BER range and adds little information.

### What the phase tracker is correcting

The next two interactive figures use the same simulated phase truth and
received packet for every shown method. The first compares absolute,
unwrapped carrier-phase estimates. The second subtracts simulator truth and
wraps the error into `[-π, π]`, which makes the quality of the applied
correction easier to compare.

<div data-noema-chart="qpsk-phase-tracking-phase-estimates"></div>

<div data-noema-chart="qpsk-phase-tracking-phase-error"></div>

The orange line is the carrier phase applied by the simulator. The oracle
copies that truth and therefore overlaps it with zero phase error. Pilot
smoothing, the decision-directed PLL, and the learned receiver use only
deployable observations. The learned estimate follows the drift more closely
than the two classical estimates in this representative 6 dB packet.

The decision-directed PLL initializes phase and frequency from the pilot
preamble. It then derotates each symbol and updates its state from a known
pilot or its own nearest-QPSK payload decision. Incorrect payload decisions
can feed back into the loop, especially at low SNR.

The [paired run projection](../demo/data/qpsk_phase_tracking/paired_benchmark_projection.csv),
[summary table](../demo/data/qpsk_phase_tracking/paired_summary_table.csv), and
[benchmark snapshot manifest](../demo/data/qpsk_phase_tracking/paired_snapshot_manifest.json)
bind the 126-run result, recipe hashes, paired seeds, BER denominators,
training evidence, and learned-artifact identity. The
[representative phase trace](../demo/data/qpsk_phase_tracking/representative_phase_trace.csv)
has a separate provenance manifest because it is a qualitative packet view;
all statistical claims above come from the completed benchmark.

## Read the completed demo

The completed benchmark views answer different questions:

| View | What to inspect |
| --- | --- |
| **Overview** | Primary BER-versus-SNR comparison for all six receivers |
| **Performance** | Paired BER and the stored packet-BLER measurements |
| **Communication** | Simulator phase truth and explicit estimates from pilot interpolation, pilot smoothing, the PLL, the learned tracker, and the oracle |
| Published static page | Plot CSV files, scalar metrics, method definitions, recipe hashes, paired seeds, artifact hashes, and verification status |

The learned receiver returns its circular correction relative to pilot smoothing. Noema combines
the two into the final unwrapped carrier-phase trajectory and derives the QPSK decisions. A single
fixed I/Q decision boundary is not representative here because the correct boundary rotates over
the packet. Static decision regions remain available in the synchronized, memoryless QPSK
demapping demo; this demonstration uses phase trajectories instead.

## Validity Checks

A completed demonstration is comparable only when:

1. Train, validation, held-out test, and benchmark records are disjoint.
2. All six methods at a coordinate use identical payload, AWGN, and carrier-impairment seeds.
3. The learned artifact hash stays unchanged across every learned benchmark run.
4. The learned receiver is compared with interpolation, smoothing, and the PLL without assuming
   that it must beat any one of them.
5. The true-phase method is reported as a non-deployable expected-BER reference, not as an ordinary
   baseline or a guaranteed pointwise lower envelope for finite empirical BER.
6. Every BER and BLER value retains its bit or packet denominator.

Do not claim a learned advantage unless the paired benchmark and its confidence intervals support
it. A learned method that underperforms a classical tracker is still a valid result and remains in
the evidence.

## 6. Verify and Create a Static Result Page

Replace `<result_id>` with the identifier printed by the benchmark run:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx noema benchmark verify <result_id>
uv run --extra onnx noema benchmark publish <result_id> \
  --slug learned-qpsk-phase-tracking-receiver \
  --out docs/demo/experiments/learned-qpsk-phase-tracking-receiver
```

The published page reads the stored metrics, paired seeds, recipe hashes, training evidence, and
artifact hashes. It also generates BER/BLER SVG figures, their plotted-data CSV files, a complete
metrics CSV, compact evidence files, and a publication manifest. It does not retrain or rerun the
experiment.

The tutorial figures above are deterministic projections of the completed
benchmark and its bound training evidence; rebuilding the documentation does
not retrain the model or rerun the campaign.
