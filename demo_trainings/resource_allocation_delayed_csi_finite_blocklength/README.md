# Reliability-aware OFDM allocation with delayed CSI

This reference project trains one allocator block from a neutral Noema export.
The policy receives four consecutive delayed/noisy complex CSI snapshots,
ordered oldest to newest, plus noise variance and the power budget. The newest
snapshot is five OFDM symbols old. Aligned current channel gains are used only
to evaluate the finite-blocklength loss during training; they are not a runtime
input and are not target allocations.

The companion recipe takes causal slices from one non-normalized Sionna TDL-C
trajectory: 128 subcarriers at 15 kHz spacing, 1 µs RMS delay spread, and
120 km/h mobility. Capture keeps every twelfth allocation state to reduce
within-trajectory duplication while runtime scoring retains all 24 states.

The causal history CNN uses temporal phase evolution and local frequency
correlation, then directly maximizes normal-approximation expected goodput for
blocklength 128 and rate 2 bit/s/Hz. It never imitates water filling. Its output
is projected exactly onto the nonnegative fixed-sum power simplex; epoch zero
is equal power.

A learned checkpoint is accepted only when it improves validation goodput over
the strongest deployable baseline by at least 0.5%, has a positive paired 95%
trajectory-cluster confidence-interval lower bound, regresses by no more than
0.2% at every configured operating point, and has at least 30 independent
trajectory clusters. A weaker artifact remains runtime-valid but cannot be
presented as the learned benchmark candidate.

## Run the checked-in example

From the Noema repository root, run this block. The `--force` flags recreate a
stale bundle and all three captures before training.

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
  "$ROOT/demo_trainings/prepare_example.py" delayed-csi-resource-allocation \
  "$BUNDLE" --project-root "$ROOT"

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

`train_demo.py` verifies that train and validation contain no duplicate or
overlapping aligned channel pairs before optimizing. `evaluate_demo.py` repeats
the check against held-out test data and compares equal power, water filling on
the delayed observation, uncertainty-shrunk water filling, a causal
per-subcarrier complex-AR predictor followed by water filling, and the learned
policy. The predictive baseline fits only the four transmitter-visible CSI
snapshots and forecasts across the declared feedback delay. Current-CSI
Shannon water filling and a direct perfect-current-CSI numerical optimization
of the modeled finite-blocklength objective are reported only as diagnostics.
Neither is deployable under delayed CSI, and the numerical reference is not
claimed to find the global optimum.

The resulting campaign keeps the payload, physical trajectory, CSI-estimation
error, and channel noise paired across methods.
