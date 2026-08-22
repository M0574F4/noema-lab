# Learned Two-Dimensional Range Localization

## Goal

Train a portable localizer for the controlled four-anchor scenario. The runtime model receives
only anchor coordinates and noisy ranges. True positions are separate offline targets; they never
enter the returned block.

The paired comparison contains linear trilateration, centroid-regularized trilateration, and the
returned geometry-aware residual model. Geometry, SNR, range-noise floor, target positions, and
random seeds are held fixed within each paired unit.

## CLI training summary

Run this complete block from the repository. It exports the neutral contract, captures disjoint
splits, trains and evaluates the included starter, and runs the post-training comparison.

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/range_localization"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/localization_adapter_baseline.yaml" \
    --training-plan "$ROOT/demo_trainings/localization_supervised_mlp/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" range-localization "$BUNDLE" \
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

Open **Localization / Sensing** > **Range-based wireless localization**, then select
**Train/replace** for **Localizer**. Keep these captured signals:

- `range_features` from `range_observation.observation`;
- `positions` from `range_observation.truth`.

The first tap stores anchor (x,y) coordinates beside each measured range. The second is an
offline label. Use 3,072 records with the default train, validation, and held-out test percentages,
keep the training plan's `0, 5, 10, 15, 20` dB capture sweep, select **PyTorch**, and export the
bundle. Attach the **range-localization** example, then capture all three splits.

## 2. Train and return the model

The starter computes differentiable linear trilateration and adds a bounded MLP residual. It keeps
the physical geometry visible and gives optimization a meaningful classical starting point. Model
selection uses validation position MSE; byte-level record fingerprints reject overlap between
train, validation, and test.

Training returns:

- `artifacts/localization_estimator.onnx`;
- `trained_artifact.yaml` with the `localization_estimator` ABI;
- `reference_training/training_history.json`;
- `reference_training/evaluation_metrics.json`.

The runtime ABI is two inputs—`anchors` and `ranges`—and one `positions` output. No truth position
is available during inference.

## 3. Compare localizers

`build_benchmark.py` creates paired held-out runs at 0 and 15 dB over three fresh seeds. Every
method receives the same target positions, anchors, and noisy ranges. Compare
`localization.rmse_m`; `task.score` is a companion presentation metric.

This protocol is synthetic range localization, not synchronized UWB ranging. It does not support
clock bias, waveform-level ToA extraction, learned NLOS identification, or radio-map claims.
