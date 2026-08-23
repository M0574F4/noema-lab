# Learned Near-Field XL-MIMO Range-Angle Focusing

## Goal

Train a portable range-angle estimator for a controlled 28 GHz spherical-wave scenario. The
runtime model receives one noisy phase-referenced coherent-pilot observation from a 32-element
half-wavelength array. The declared 0.5–5 m interval lies approximately within this aperture's
radiative near-field region; true range and angle remain separate offline targets.

The paired comparison contains far-field steering search, polar range-angle codebook search, the
returned learned estimator, and true-position focusing as an explicitly labeled simulation oracle.

## CLI training summary

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/near_field_xl_mimo"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/near_field_xl_mimo_focusing.yaml" \
    --training-plan "$ROOT/demo_trainings/near_field_range_angle_mlp/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" near-field-range-angle "$BUNDLE" \
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

Select **Train/replace** for **Estimator**. Capture `array_observation` from `data.problem` and
`range_angle` from `data.truth`. The default plan uses 4,096 disjoint records over 0–20 dB.

## 2. Train and return the estimator

The starter searches a dense spherical-wave bank over the declared 0.5–5 m and ±55° region, then
fits a bounded residual head for sub-grid and low-SNR corrections from the full coherent array.
Both paths are packaged in one portable ONNX component. Model selection uses validation loss only.
The ABI accepts `array_ri[batch, antenna, 2]` and returns `range_angle[batch, 2]`.

## 3. Compare estimation and focusing

The generated paired campaign reports range RMSE, angle RMSE, and normalized focusing gain. The
far-field baseline deliberately ignores spherical-wave range curvature. The polar codebook searches
the declared range-angle grid, while true-position focusing uses evaluator truth and is not an
implementable receiver.

## Completed benchmark result

The completed paired campaign contains 24 runs: four focusing rules, two held-out SNRs, and three
fresh target/noise seeds. Points are means over the three seeds; bands are two-sided Student-t 95%
confidence intervals.

```{note}
Three paired seeds make this a compact workflow result, not a publication-strength population
claim.
```

<div data-noema-chart="near-field-focusing-gain"></div>

The learned estimator reaches 0.958 normalized focusing gain at 0 dB and 0.979 at 15 dB. The two
coarser classical searches remain around 0.88–0.90, while the simulation-only true-position oracle
is 1.0. The gain comes mainly from the denser physics-based search; the learned residual supplies
bounded sub-grid correction rather than replacing the propagation model.

```{csv-table} Paired benchmark summary
:file: ../demo/data/near_field_xl_mimo/summary_table.csv
:header-rows: 1
:align: center
```

Download the [run-level projection](../demo/data/near_field_xl_mimo/benchmark_projection.csv),
[chart data](../demo/data/near_field_xl_mimo/chart_data.json), or
[provenance manifest](../demo/data/near_field_xl_mimo/snapshot_manifest.json).

This narrowband, single-user synthetic protocol omits beam squint, mutual coupling, blockage,
calibration error, wideband delay, and multi-user interference.
