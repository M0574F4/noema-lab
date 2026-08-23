# Learned LEO-NTN Doppler Prediction and Beam Handover

## Goal

Train a causal portable tracker from six noisy Doppler/azimuth observations. At a fixed one-second
horizon, the model predicts future Doppler and scores one of nine fixed beam sectors. Future state
and the correct next beam are captured separately and never enter runtime inference.

The paired comparison contains hold-last, linear extrapolation, the learned tracker, and a
simulation-future-state oracle.

## CLI training summary

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/leo_ntn_tracking"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/leo_ntn_doppler_beam_tracking.yaml" \
    --training-plan "$ROOT/demo_trainings/leo_ntn_tracking_mlp/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" leo-ntn-tracking "$BUNDLE" \
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

Select **Train/replace** for **Tracker**. Capture `track_features` from `data.problem` and
`future_state` from `data.truth`. The default plan uses 4,096 disjoint records across five
observation-SNR settings.

## 2. Train and return the tracker

The starter jointly minimizes normalized future-Doppler error and next-beam cross entropy.
Validation data selects the checkpoint. Its ONNX output contains one Doppler estimate followed by
nine beam logits; Noema performs the same deterministic argmax projection for every run.

## 3. Compare causal handover rules

The generated campaign reports both Doppler MAE and beam-handover accuracy at the same fixed
horizon. Hold-last and linear extrapolation see exactly the same history as the learned model. The
oracle consumes future simulator state and is only an upper bound.

## Completed benchmark result

The completed paired campaign contains 24 runs: four causal handover rules, two held-out SNRs, and
three fresh track/noise seeds. Points are means over the three seeds; bands are two-sided Student-t
95% confidence intervals.

```{note}
Three paired seeds make this a compact workflow result, not a publication-strength population
claim.
```

<div data-noema-chart="leo-ntn-handover-accuracy"></div>

The learned tracker has higher mean next-beam accuracy than hold-last at 0 dB (0.719 versus 0.635),
while hold-last is slightly better at 15 dB (0.911 versus 0.896). Linear extrapolation is weaker at
both points; the future-state oracle is 1.0 by construction. This is a mixed result, not a universal
learned-tracker win.

```{csv-table} Paired benchmark summary
:file: ../demo/data/leo_ntn_tracking/summary_table.csv
:header-rows: 1
:align: center
```

Download the [run-level projection](../demo/data/leo_ntn_tracking/benchmark_projection.csv),
[chart data](../demo/data/leo_ntn_tracking/chart_data.json), or
[provenance manifest](../demo/data/leo_ntn_tracking/snapshot_manifest.json).

This bounded kinematic generator is not an orbital propagator, ephemeris product, 3GPP NTN channel
model, link budget, beam-management protocol, or conformance test.
