# Learned Single-User MISO Beam Selection

## Goal

Train a portable eight-beam, constant-modulus codebook for a clustered ULA channel distribution.
The returned policy receives an eight-antenna complex MISO channel vector and selects the strongest
learned beam. The loss optimizes normalized channel gain directly; no label or oracle score enters
the deployed block.

The paired comparison holds the beam count and channel realizations fixed across the learned
codebook and an equal-size DFT codebook. Perfect-CSIT maximum-ratio transmission (MRT) remains an
upper bound.

## CLI training summary

```bash
(
  set -euo pipefail
  ROOT="$(git rev-parse --show-toplevel)"
  BUNDLE="$ROOT/.noema/training_exports/miso_beam_selection"
  cd "$ROOT"

  uv sync --extra onnx
  uv run --project "$ROOT" --extra onnx noema differentiable export \
    "$ROOT/recipes/beamforming_adapter_baseline.yaml" \
    --training-plan "$ROOT/demo_trainings/beam_selection_supervised_mlp/training_plan.yaml" \
    --out "$BUNDLE" --force
  uv run --project "$ROOT" --extra onnx python \
    "$ROOT/demo_trainings/prepare_example.py" beam-selection "$BUNDLE" \
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

Open **Beamforming / Precoding** > **Single-user MISO beamforming**, select **Train/replace** for
**Beamformer**, and keep `channels` from `data.problem`. Use 4,096 records and the default train,
validation, and held-out test split.

There is no separately captured target tap. The trainer evaluates normalized gain from each channel
vector directly, making the objective explicit and reproducible.

## 2. Train and return the policy

The starter initializes eight ULA beams from the DFT grid, learns their spatial frequencies from
the training split, and exports exhaustive selection within that learned codebook. The
`beam_policy` ABI is:

- input `channels_ri`: `[batch, tx_antenna, 2]`;
- output `weights_ri`: `[batch, tx_antenna, 2]`.

The Noema runtime independently checks finite values and nonzero norm, then normalizes the returned
beam before link evaluation. `evaluate_demo.py` reports held-out normalized gain, the fixed-DFT
reference gain, and their ratio without exposing test records during training.

## 3. Compare beam policies

The generated campaign uses identical held-out channels for all three methods at 0 and 15 dB over
three fresh seeds. The primary outcome is `beamforming.spectral_efficiency_bps_hz`.

## Completed benchmark result

The completed paired campaign contains 18 runs: the learned eight-beam codebook, equal-size fixed
DFT-codebook search, and perfect-CSIT MRT at two held-out SNRs over three fresh channel seeds.
Points are means over the three seeds; bands are two-sided Student-t 95% confidence intervals.

```{note}
Three paired seeds make this a compact workflow result, not a publication-strength population
claim.
```

<div data-noema-chart="miso-beam-selection-rate"></div>

The learned codebook improves mean spectral efficiency from 2.596 to 2.761 bit/s/Hz at 0 dB and
from 7.179 to 7.365 bit/s/Hz at 15 dB. Perfect-CSIT MRT remains the expected upper bound at 2.922
and 7.601 bit/s/Hz because it is not restricted to eight reusable beams.

```{csv-table} Paired benchmark summary
:file: ../demo/data/miso_beam_selection/summary_table.csv
:header-rows: 1
:align: center
```

Download the [run-level projection](../demo/data/miso_beam_selection/benchmark_projection.csv),
[chart data](../demo/data/miso_beam_selection/chart_data.json), or
[provenance manifest](../demo/data/miso_beam_selection/snapshot_manifest.json).

This scenario gives every policy the realized channel and draws users from three declared spatial
hotspots. The learned-codebook advantage is distribution-specific. It does not claim a feedback-bit
budget, beam-training overhead, mobility, multi-user interference, or robustness to CSI error.
