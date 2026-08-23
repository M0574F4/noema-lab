# Learned Joint ISAC OFDM Allocation

## Goal

Train a portable power-allocation policy for one controlled frequency-selective OFDM protocol. Each
record exposes per-subcarrier communication gain, sensing gain, noise level, and the declared
sensing weight. The policy must return a non-negative unit-sum power vector.

The paired comparison contains equal power, communication-only water filling, a per-scene
projected-gradient reference for the declared scalarized objective, and the returned learned policy.
Channel gains, noise, tradeoff weight, power budget, and random seeds remain fixed within each
paired unit.

## CLI training summary

Run this complete block from the repository:

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/isac_joint_allocation"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/isac_ofdm_joint_allocation.yaml" \
    --training-plan "$ROOT/demo_trainings/isac_joint_allocation_deepsets/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" isac-joint-allocation "$BUNDLE" \
    --project-root "$ROOT"

  uv run --project "$ROOT" --extra onnx noema dataset-capture run \
    "$BUNDLE/capture_train_recipe.yaml" --out "$BUNDLE/data/train" --force
  uv run --project "$ROOT" --extra onnx noema dataset-capture run \
    "$BUNDLE/capture_validation_recipe.yaml" --out "$BUNDLE/data/validation" --force
  uv run --project "$ROOT" --extra onnx noema dataset-capture run \
    "$BUNDLE/capture_test_recipe.yaml" --out "$BUNDLE/data/test" --force

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

## 1. Export and capture

Select **Train/replace** for **Allocator** and retain `isac_features` from `data.problem`. No oracle
allocation labels are captured. The default plan collects 4,096 disjoint records over 0–20 dB.

## 2. Train and return the policy

The starter uses a permutation-equivariant subcarrier scorer followed by softmax. Its power vector
therefore satisfies the budget exactly. It trains directly against negative scalarized utility;
validation utility selects the checkpoint and the test split remains sealed until evaluation.

The returned package contains `artifacts/isac_allocator.onnx`, `trained_artifact.yaml`, training
history, and held-out evaluation metrics.

## 3. Compare the allocation rules

The generated campaign sweeps held-out SNRs and three fresh paired seeds. Read
`isac.scalarized_utility` together with communication rate and sensing SNR: the scalarized number is
meaningful only for the fixed weight declared by this contract.

## Completed benchmark result

The completed paired campaign contains 24 runs: four allocation rules, two held-out SNRs, and three
fresh seeds. Points are means over the three seeds; bands are two-sided Student-t 95% confidence
intervals.

```{note}
Three paired seeds make this a compact workflow result, not a publication-strength population
claim.
```

<div data-noema-chart="isac-joint-allocation-utility"></div>

The learned allocator has a higher mean than equal power and communication-only water filling at
both tested SNRs and closely follows the per-scene projected-gradient reference. At 0 dB it is
0.02% above that finite-iteration reference; this tiny reversal is solver tolerance, not evidence
that it exceeds the declared objective's exact optimum.

```{csv-table} Paired benchmark summary
:file: ../demo/data/isac_joint_allocation/summary_table.csv
:header-rows: 1
:align: center
```

Download the [run-level projection](../demo/data/isac_joint_allocation/benchmark_projection.csv),
[chart data](../demo/data/isac_joint_allocation/chart_data.json), or
[provenance manifest](../demo/data/isac_joint_allocation/snapshot_manifest.json).

This is a synthetic resource-allocation demonstration. It has no waveform-level target detector,
range/Doppler ambiguity function, clutter model, multi-user interference, or standards-conformance
claim.
