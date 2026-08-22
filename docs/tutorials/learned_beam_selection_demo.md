# Learned Single-User MISO Beam Selection

## Goal

Train a portable policy that maps an eight-antenna complex MISO channel vector to one unit-norm
beam from a finite DFT codebook. Exhaustive codebook search supplies offline class labels inside the
trainer; no label or oracle score enters the deployed block.

The paired comparison includes the learned policy, exhaustive DFT-codebook selection, and
perfect-CSIT maximum-ratio transmission (MRT). MRT is an upper bound and exhaustive search is an
oracle within the declared codebook.

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

There is no separately captured target tap. The trainer derives the best finite-codebook index from
each channel vector, making the supervision rule explicit and reproducible.

## 2. Train and return the policy

The starter normalizes channel magnitude, trains a compact codebook classifier with cross entropy,
and exports the selected DFT beam rather than class logits. The `beam_policy` ABI is:

- input `channels_ri`: `[batch, tx_antenna, 2]`;
- output `weights_ri`: `[batch, tx_antenna, 2]`.

The Noema runtime independently checks finite values and nonzero norm, then normalizes the returned
beam before link evaluation. `evaluate_demo.py` reports held-out codebook accuracy and mean channel
gain without exposing test records during training.

## 3. Compare beam policies

The generated campaign uses identical held-out channels for all three methods at 0 and 15 dB over
three fresh seeds. The primary outcome is `beamforming.spectral_efficiency_bps_hz`.

This scenario gives every policy the realized channel. It does not claim a feedback-bit budget,
beam training overhead, mobility, multi-user interference, or robustness to CSI error. Those are
separate protocols—not extra labels to add silently to this one.
