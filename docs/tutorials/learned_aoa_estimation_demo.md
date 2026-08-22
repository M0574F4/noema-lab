# Learned Narrowband Angle-of-Arrival Estimation

## Goal

Train a portable single-source AoA estimator for an eight-element half-wavelength uniform linear
array. The returned model receives noisy complex snapshots only. Source angles are offline labels
and are unavailable at runtime.

The paired comparison holds angles, array snapshots, and AWGN seeds fixed across Bartlett, MUSIC,
and the returned learned estimator.

## CLI training summary

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/aoa_estimation"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/aoa_adapter_ula_baseline.yaml" \
    --training-plan "$ROOT/demo_trainings/aoa_estimation_covariance_mlp/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" aoa-estimation "$BUNDLE" \
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

## 1. Export and capture from Workbench

Open **Localization / Sensing** > **ULA angle-of-arrival estimation**, then select
**Train/replace** for **Estimator**. Keep:

- `snapshots` from `array_observation.observation`;
- `angles_deg` from `array_observation.truth`.

Use 4,096 records, the default three-way split, and the training plan's `0, 5, 10, 15, 20` dB
capture sweep. The captured snapshot tensor preserves the complex array and time axes; the
exporter presents it to ONNX as a final real/imaginary axis.

## 2. Train and return the estimator

The starter converts each frame to a normalized complex sample covariance and uses a bounded MLP
regressor. This removes arbitrary snapshot ordering and follows the same second-order structure
used by the classical array processors. Validation angle MSE selects the checkpoint; held-out
angles remain sealed until `evaluate_demo.py`.

The returned `aoa_estimator.onnx` implements:

- input `snapshots_ri`: `[batch, antenna, snapshot, 2]`;
- output `angles_deg`: `[batch]`.

The schema-v2 artifact records the exact trained antenna and snapshot sizes and binds the model to
`model.aoa_estimator_adapter`.

## 3. Compare estimators

The post-training pack compares Bartlett, MUSIC, and the learned estimator at 0 and 15 dB over
three fresh paired seeds. Use `aoa.rmse_deg` as the primary metric and inspect `aoa.mae_deg` as a
companion measure.

This is deliberately a far-field, single-source, narrowband, calibrated-array protocol. Multiple
sources, coherent multipath, calibration error, near-field propagation, and wideband beam squint
need separate contracts and should not be inferred from this demo.
