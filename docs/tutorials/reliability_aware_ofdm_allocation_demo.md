# Reliability-Aware OFDM Allocation with Delayed CSI

## Goal

Train an OFDM power allocator for short packets when the transmitter sees an old, noisy channel
history. Each example gives the model four consecutive complex CSI snapshots, ordered oldest to
newest. The newest snapshot is five OFDM symbols older than the channel used for transmission. The
current state is captured only for the training objective and held-out evaluation. In other words,
the current channel never enters the returned model ABI.

The history and current state are causal slices of one Sionna 3GPP TDL-C trajectory. A 1 µs RMS
delay spread creates frequency selectivity across 128 subcarriers, while 120 km/h mobility creates
temporal evolution. Channel normalization is disabled, so the policy must also handle absolute
fading changes. This gives delayed CSI useful but imperfect predictive information.

The objective is expected short-packet goodput,
`target rate × (1 − predicted BLER)`. Predicted BLER uses the finite-blocklength normal
approximation for parallel complex AWGN channels. It is a reliability model, not a claim that a
specific deployed FEC decoder was simulated.

## CLI training summary

Run this block from the project root. The explicit `--force` flags recreate the generated bundle
and all three captures, so it is also the correct block for replacing an older version of this demo.
Each capture command displays its own progress.

```bash
(
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/.noema/training_exports/ofdm_delayed_csi_reliability"
cd "$ROOT"

uv sync --extra wireless --extra onnx
uv run --project "$ROOT" --extra wireless --extra onnx noema differentiable export \
  "$ROOT/recipes/resource_delayed_csi_finite_blocklength.yaml" \
  --training-plan "$ROOT/demo_trainings/resource_allocation_delayed_csi_finite_blocklength/training_plan.yaml" \
  --out "$BUNDLE" --force
uv run --project "$ROOT" --extra wireless --extra onnx python \
  "$ROOT/demo_trainings/prepare_example.py" delayed-csi-resource-allocation "$BUNDLE" \
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

## Scenario

| Property | Template setting |
| --- | --- |
| Channel | Sionna 3GPP TDL-C, channel normalization disabled |
| OFDM bandwidth | `128 × 15 kHz = 1.92 MHz` |
| RMS delay spread | `1 µs` |
| Mobility | `120 km/h` at `3.5 GHz` |
| Transmitter input | `4 × 128 × 2` real values: four complex CSI snapshots |
| CSI age | newest history snapshot is `5` OFDM symbols old |
| CSI estimation SNR | `20 dB` |
| Allocation interval | `24` OFDM symbols |
| Capture stride | every `12th` allocation state within a trajectory |
| Short-packet model | Normal approximation, blocklength `128` channel uses |
| Target rate | `2 bit/s/Hz` |
| Template sweep | Average power budget `0.4, 0.6, 0.8, 1.0, 1.4` |

The allocator receives the noisy complex history, noise variance, and power budget. Its temporal and
frequency convolutions can use phase evolution across snapshots and correlation across neighboring
subcarriers to predict a useful allocation for the later channel. The output is projected onto the
nonnegative fixed-sum power simplex. During training, the loss scores that allocation on the aligned
current channel; it does not imitate a water-filling label.

## 1. Export and capture from Workbench

Start Noema, then open **Browse template recipes** > **Physical layer & resource optimization** >
**Resource allocation** > **Reliability-aware OFDM allocation with delayed CSI**. Select
**Workbench** and configure the sections in their displayed order:

1. Under **Operation Training Capabilities**, find **TX power**, confirm **Portable replacement**
   is **Yes**, and select **Train/replace**.
2. Under **Dataset definition** > **Captured signals**, keep both aligned CSI tensors selected:

   - `csi_observation_transmitter_csi`: four-snapshot delayed/noisy complex CSI history; required
     model input;
   - `csi_observation_actual_state`: later channel state; training-only objective input.

3. Under **Capture coordinates**, leave **Parameter sweep** blank. Training samples the supported
   power-budget range itself; the template's power matrix is reserved for benchmark runs.
4. Under **Dataset size and splits**, set **Total recipe records** to `3072`, **Train %** to
   `66.6667`, and **Validation %** to `16.6667`. This produces `2048` training, `512` validation,
   and `512` held-out test records.
5. Under **Training bundle**, set:

   - **Bundle directory:** `.noema/training_exports/ofdm_delayed_csi_reliability`
   - **Support framework:** **PyTorch**
   - **Overwrite generated files:** enable only when intentionally rebuilding the bundle

6. Select **Export training bundle**.

From the repository root, attach the checked-in example model, loss, trainer, evaluator, and
benchmark builder:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra wireless --extra onnx python demo_trainings/prepare_example.py \
  delayed-csi-resource-allocation \
  .noema/training_exports/ofdm_delayed_csi_reliability
```

Return to **Workbench**. Under **Dataset capture**, select **Capture all datasets** and wait until
train, validation, and held-out test are ready. Use **Recapture all datasets** when intentionally
replacing an existing capture.

## 2. Train and return the model

Run from the exported bundle:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/ofdm_delayed_csi_reliability
uv run --project ../../.. --extra wireless --extra onnx python validate_contract.py
uv run --project ../../.. --extra wireless --extra onnx python train_demo.py
uv run --project ../../.. --extra wireless --extra onnx python evaluate_demo.py
cd ../../..
```

Training uses only `data/train` and `data/validation` for model selection. Evaluation reads
`data/test`. Return to **Workbench**, wait for **External model** to detect
`trained_artifact.yaml`, and select **Validate returned model**.

The example accepts a learned checkpoint only when all validation checks pass:

- at least `0.5%` aggregate expected-goodput improvement over the strongest deployable baseline;
- a positive lower bound for the paired `95%` trajectory-cluster confidence interval;
- no operating-point regression larger than `0.2%`; and
- at least `30` independent trajectory clusters.

The comparison baseline is selected from equal power, water filling on the newest delayed/noisy
snapshot, and uncertainty-shrunk delayed-CSI water filling. A checkpoint that misses the gate can
still be inspected as a runtime-compatible artifact, but the benchmark builder will not present it
as a successful learned candidate.

For an ordinary recipe smoke test, open **Graph** > **TX power**, set **Policy** to
**Learned model**, select the returned project-trained artifact, and select **Run All**.

## 3. Compare policies

The current paired benchmark builder compares:

- equal power;
- water filling applied to the newest delayed/noisy CSI snapshot;
- uncertainty-shrunk water filling applied to that snapshot;
- complex-AR prediction of the current CSI followed by water filling; and
- the returned learned allocator.

Water filling on delayed CSI is a mismatched practical baseline, not a Shannon oracle. Every method
uses the same current channel, noise, and power budget at a benchmark coordinate. Compare
finite-blocklength expected goodput, predicted BLER, and power-constraint error. The dashboard also
plots the newest delayed/noisy transmitter snapshot beside the current channel so the aging
mismatch is visible.

The complex-AR policy is a stronger classical comparator because it first predicts the later
complex channel from the same causal history available to the learned policy. New campaigns also
report a perfect-current-CSI numerical finite-blocklength optimizer as a non-deployable diagnostic
reference, not as a practical baseline. The completed result below predates both additions and
contains the first four-policy campaign: equal power, two delayed-CSI water-filling policies, and
the learned allocator.

## Completed benchmark result

Result
`20260726T192946Z_resource_allocation.delayed_csi_finite_blocklength_post_training_v2`
completed all `60` recipes: five power budgets, three paired held-out TDL trajectory seeds, and four
policies. At every power budget and seed, all policies use the same payload, channel trajectory,
CSI-estimation error, and AWGN realization.

```{note}
This is verified experimental demonstration evidence. Each point averages three paired held-out
trajectory seeds; the shaded bands are two-sided Student-t 95% confidence intervals with two
degrees of freedom. More independent trajectories are needed for a paper-grade population claim.
```

### Objective-aligned comparison

<div data-noema-chart="delayed-csi-goodput"></div>

The learned allocator has the highest expected finite-blocklength goodput at every tested power
budget and every paired seed. Uncertainty-shrunk water filling is the strongest baseline at all five
coordinates. The paired mean improvement decreases from `16.54%` at power `0.4` to `3.46%` at
power `1.4`, and the paired 95% confidence interval for the absolute gain remains positive at every
coordinate.

<div data-noema-chart="delayed-csi-predicted-bler"></div>

This BLER is the normal-approximation reliability estimate used by the training objective:
`expected goodput = 2 bit/s/Hz × (1 − predicted BLER)`. It is not a measured decoder BLER. The
ordinary uncoded-QPSK payload BER and BLER produced by the recipe's link smoke path are deliberately
excluded from this demonstration because they are not the optimized finite-blocklength objective.

```{csv-table} Paired held-out benchmark summary
:file: ../demo/data/reliability_aware_delayed_csi_ofdm_allocation/summary_table.csv
:header-rows: 1
:align: center
```

Across the full campaign, delayed/current channel-gain correlation averages `0.6976`. The maximum
sum-power error is `6.14e-6`, and no policy assigns negative power.

### What one paired channel state looks like

This figure uses snapshot `0` at power `0.8` and paired seed `95101`. It shows every second
subcarrier for readability; the values come from the stored benchmark run evidence.

<div data-noema-chart="delayed-csi-representative-state"></div>

The horizontal axis is frequency (subcarrier index), not time. At each subcarrier `k`, the gray
curve is the transmitter's noisy observation `|Ĥ[t−5,k]|²`, while the green curve is the perfect
current state `|H[t,k]|²` used only for evaluation. They are therefore not expected to be
horizontally shifted copies: the complex channel evolves during the five-symbol feedback delay,
and estimation noise adds another mismatch. The right axis shows how every policy distributes the
same `128 × 0.8` total power budget using its permitted input. The learned policy is scored on the
green current channel but never receives that channel as a runtime input. Use the legend to isolate
a policy or highlight its allocation.

### Provenance

The compact [snapshot manifest](../demo/data/reliability_aware_delayed_csi_ofdm_allocation/snapshot_manifest.json)
records all 60 run IDs, recipe and semantic recipe hashes, source-file hashes, the paired-seed
design, representative-preview evidence, and the returned artifact:

- result JSON SHA-256:
  `9208bea28be10b8e90329ed4a2b620c07460089de7329d0cb26a7417a3eb77c8`;
- metrics CSV SHA-256:
  `b5b39b81d4374a5b6bb3226a17c64a14e310f79f53654f6a229dd5c6db97793f`;
- returned ONNX component SHA-256:
  `18868922a2bd109c65150bd6f10aec039c80ec42b180a43a44af6e1e486a8225`;
- trained-artifact manifest SHA-256:
  `94473e9417cc1fb65afe9aefb1710482df1a047e4f9c21334fc2be294447550b`.

The checked-in [60-run projection](../demo/data/reliability_aware_delayed_csi_ofdm_allocation/benchmark_projection.csv)
contains only the metrics and identities needed by this page, rather than duplicating the large
runtime tensors.

## 4. Verify a Local Result

Replace `<result_id>` with the identifier printed by the benchmark run:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra wireless --extra onnx noema benchmark verify <result_id>
uv run --extra wireless --extra onnx noema benchmark publish <result_id> \
  --slug reliability-aware-delayed-csi-ofdm-allocation \
  --out docs/demo/experiments/reliability-aware-delayed-csi-ofdm-allocation
```

This experimental result is not publication-ready by default. Verification
and static export do not retrain the model or rerun the recipes.
