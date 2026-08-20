# Learned OFDM Subcarrier Allocation

## Goal

Train an OFDM power-allocation policy and compare it with equal power and theoretical water filling
under the same nonnegative, fixed-sum power constraint. The scenario stays fixed; only the
**TX power** Block changes.

- **Equal power** is the classical baseline.
- **Learned allocation** is the returned ONNX model.
- **Water filling** is the exact Gaussian-input Shannon-capacity reference.

The example trainer receives channel gains and maximizes Shannon spectral efficiency. It does not
copy water-filling labels. For background on contracts, captures, and artifact return, see
[Physical-Layer System Templates and Learned-Method Examples](physical_layer_demo_workflow.md).

## CLI training summary

For a terminal-only run, `BUNDLE` must name a directory that does not already exist; change it before
running this block if necessary. The numbered walkthrough below remains the Workbench alternative.
Each `dataset-capture` command reports progress for its own split; let it complete before the next
command starts. Add `--force` only when intentionally replacing an existing capture.

```bash
(
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
BUNDLE="$ROOT/.noema/training_exports/ofdm_allocator"
cd "$ROOT"

# This exact reproducibility block requires a Noema source checkout.
uv sync --extra onnx
uv run --project "$ROOT" --extra onnx noema differentiable export \
  "$ROOT/recipes/resource_equal_power_baseline.yaml" \
  --training-plan "$ROOT/demo_trainings/resource_allocation_unsupervised_shannon/training_plan.yaml" \
  --out "$BUNDLE"
uv run --project "$ROOT" --extra onnx python \
  "$ROOT/demo_trainings/prepare_example.py" resource-allocation "$BUNDLE" \
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

## 1. Export and capture from Workbench

Open **OFDM subcarrier resource allocation** from **Template Recipes**, then select **Workbench**.

1. Under **Operation Training Capabilities**, find **TX power**. Confirm **Portable replacement** is
   **Yes**, then select **Train/replace**.
2. Under **Dataset definition**, keep `channel_state.state` selected. The generated signal name may
   appear as `channel_state_state`.
3. Under **Dataset size and splits**, set:

   - **Total recipe records:** `192`
   - **Train:** `66.6667%`
   - **Validation:** `16.6667%`

   The resulting split contains 128 train, 32 validation, and 32 held-out test records.
4. Under **Training bundle**, set:

   - **Bundle directory:** `.noema/training_exports/ofdm_allocator`
   - **Support framework:** **PyTorch**
   - **Overwrite generated files:** enable only when intentionally regenerating the bundle

5. Select **Export training bundle**.

From the repository root, attach the included example model, objective, and trainer:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx python demo_trainings/prepare_example.py resource-allocation \
  .noema/training_exports/ofdm_allocator
```

Return to Workbench. It detects the attached example automatically. Under **Dataset capture**, select
**Capture all datasets** when the action becomes available. Continue when every split shows captured
counts and the action changes to **Recapture all datasets**.

## 2. Train, evaluate, and return the model

Run the prepared launchers from the repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/ofdm_allocator
uv run --project ../../.. --extra onnx python validate_contract.py
uv run --project ../../.. --extra onnx python train_demo.py
uv run --project ../../.. --extra onnx python evaluate_demo.py
cd ../../..
```

Training uses `data/train` and `data/validation` for checkpoint selection. Evaluation reads
`data/test` once and registers the returned artifact as `trained_artifact.yaml`.

In Workbench, select **Validate returned model**. Continue when the status reports
**model interface valid**.

To smoke-test the returned model in the ordinary recipe:

1. Select **Graph**, then open **TX power**.
2. Set **Policy** to **Learned model**.
3. Under **Model artifact** > **Project-trained**, select
   **Learned · ofdm_subcarrier_resource_allocation**.
4. Select **Run All** and inspect **Results**.

Select the registered artifact entry, not the standalone ONNX file.

## 3. Run the paired comparison

From any directory inside the clone, run the following block. It returns to the repository root,
enters the bundle's `reference_training` directory to build the equal-power, learned, and
water-filling pack, then returns to the repository root to validate and run it:

```bash
cd "$(git rev-parse --show-toplevel)"
cd .noema/training_exports/ofdm_allocator/reference_training
uv run --project ../../../.. --extra onnx python build_benchmark.py
cd ../../../..

uv run --extra wireless --extra onnx noema benchmark validate \
  .noema/training_exports/ofdm_allocator/reference_training/benchmark_pack.yaml
uv run --extra wireless --extra onnx noema benchmark run \
  .noema/training_exports/ofdm_allocator/reference_training/benchmark_pack.yaml
```

Save the printed `result_id`. Reload the dashboard if needed, select **Results**, then open the new
entry under **Open** > **Benchmark results**.

## Completed benchmark result

The frozen result
`20260727T002419Z_resource_allocation.learned_allocator_post_training_v2`
contains 27 completed runs: three policies, three power budgets, and three
paired held-out payload/channel/noise seeds. Every method sees the same
Sionna TDL-A channel realization within a paired cell. Points below are means
over the three seeds; bands are two-sided Student-t 95% confidence intervals.

```{note}
Three paired seeds are sufficient for this compact workflow demonstration,
but not for a publication-strength population claim. Extend the seed list
before reporting formal confidence intervals in research.
```

<div data-noema-chart="ofdm-allocation-spectral-efficiency"></div>

The learned policy and exact water filling overlap to plotting precision.
Across the three tested budgets, the learned policy reaches at least
`99.9999969%` of the paired water-filling spectral efficiency.

```{csv-table} Paired benchmark summary
:file: ../demo/data/ofdm_resource_allocation/paired_summary_table.csv
:header-rows: 1
:align: center
```

<div data-noema-chart="ofdm-allocation-shannon-optimum-achievement"></div>

This view divides each run by water filling on the same paired channel state.
The oracle is therefore exactly `100%`; the learned curve approaches it from
below while equal power retains a visible optimality gap.

<div data-noema-chart="ofdm-allocation-representative-power"></div>

The representative state shows why the learned result is meaningful. Equal
power remains flat, while both the learned policy and water filling move power
away from the deep fade near subcarriers 20–22 and toward stronger
subcarriers. The learned and oracle allocations nearly coincide without
using water-filling allocations as training labels.

Payload BLER is deliberately not plotted as an allocator-optimality figure.
The payload path uses an identity code and discrete SNR-threshold bit loading,
whereas water filling optimizes Gaussian-input Shannon rate. Changing the
allocator also changes the adaptive schedule and modulation-order map, so a
lower payload BLER for one policy does not imply that it exceeds
water-filling capacity.

The [paired run projection](../demo/data/ofdm_resource_allocation/paired_benchmark_projection.csv),
[summary table](../demo/data/ofdm_resource_allocation/paired_summary_table.csv),
and [provenance manifest](../demo/data/ofdm_resource_allocation/paired_snapshot_manifest.json)
preserve the plotted metrics, run IDs, recipe hashes, paired seeds,
representative reports, and trained-artifact evidence. The earlier
[single-seed UI export](../demo/data/ofdm_resource_allocation/current_results_all_metrics.csv)
remains archived for workflow regression checks but is not used by these
figures.

## Interpret the comparison

Use these as the primary checks:

1. Shannon spectral efficiency versus average transmit-power budget.
2. Equal-power, learned, and water-filling results at every paired coordinate.
3. Maximum sum-power error and negative-power violations.

Water filling is optimal for the declared parallel Gaussian-channel objective.
BER, BLER, and goodput from the discrete transport path are diagnostics, not
optimality evidence for this task. The default held-out campaign uses three
paired seeds for a quick demonstration; increase its seed list for
publication-strength confidence intervals.

## 4. Verify and publish

From any directory inside the clone, run the following block. It returns to the repository root;
replace `<result_id>` with the identifier printed by the benchmark run:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra wireless --extra onnx noema benchmark verify <result_id>
uv run --extra wireless --extra onnx noema benchmark publish <result_id> \
  --slug learned-ofdm-resource-allocation \
  --out docs/demo/experiments/learned-ofdm-resource-allocation
```

The generated page reads the stored benchmark result and includes metric curves, tables, run IDs,
recipe hashes, verification status, and training evidence. It does not retrain or rerun recipes.
