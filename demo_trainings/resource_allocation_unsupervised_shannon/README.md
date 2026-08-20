# Label-free CSI power-allocation demo training

This directory contains the model-training project used by the OFDM allocation demonstration. It is
deliberately outside the Noema execution workflow: Workbench first exports a neutral training bundle,
then captures channel states against that frozen bundle. The tutorial-only
`demo_trainings/prepare_example.py` helper supplies a ready `reference_training/` project inside it.
PyTorch training remains external.

`training_plan.yaml` records this example's CSI-only capture and local Shannon objective. Apply it
with the CLI when reproducing the demonstration; it is intentionally separate from the reusable
OFDM system recipe.

## Quick start from the default Workbench bundle

The bundle created by **Export training bundle** is intentionally trainer-neutral. Its root contains
files such as `training_contract.yaml`, `interfaces.py`, and `validate_contract.py`, but it does not
contain a training launcher. Attaching this checked-in demo adds `RUN_DEMO.md`, `train_demo.py`, and
`evaluate_demo.py` at the bundle root, with the implementation kept under `reference_training/`.

From the Noema repository root (the directory containing `.venv/`, `demo_trainings/`, and
`.noema/`), attach the demo with a supported Python version (3.11–3.13):

```bash
uv run --extra onnx python --version
uv run --extra onnx python demo_trainings/prepare_example.py resource-allocation \
  .noema/training_exports/ofdm_allocator
```

The version command should report Python 3.11 through 3.13. `uv run`
selects the project interpreter and locked ONNX/PyTorch dependencies; do not use an unrelated
system `python` or run `pip` against the checkout's `.venv`.

Then return to Workbench. It detects the attached example automatically. Use **Capture all datasets**
when the action becomes available. Once all three splits say **captured**, run these commands directly
from the bundle root:

```bash
cd .noema/training_exports/ofdm_allocator
uv run --project ../../.. --extra onnx python train_demo.py
uv run --project ../../.. --extra onnx python evaluate_demo.py
cd ../../..
```

The launchers enter `reference_training/` automatically. `train_demo.py` reads the training split and
uses the validation split automatically to select a checkpoint; there is no separate
validation-training command. `evaluate_demo.py` then measures the selected checkpoint on the
held-out test split. After both commands succeed, return to Workbench. Once **External model** detects
the returned artifact, select **Validate returned model** to check its manifest, hashes, and runtime
ABI.

The network is not trained to imitate water filling. Its only training signal is

```text
-mean(log2(1 + |H|^2 * allocated_power / noise_variance))
```

and its output is projected onto the nonnegative fixed-sum power simplex. Thus every prediction
satisfies the same instantaneous constraint as the theoretical water-filling problem. Water filling
appears only in `evaluate.py`, after checkpoint selection, and in Noema's benchmark metrics.

## Why this model

Each OFDM subcarrier is one element of an unordered set. A shared carrier encoder, mean-pooled global
context, and shared score decoder make the policy permutation equivariant: permuting CSI entries
permutes the allocated powers. A graph neural network becomes appropriate for a multiuser
interference graph, but is unnecessary for this independent parallel-channel demonstration.

Euclidean simplex projection was chosen over softmax because it enforces the power budget exactly
and can emit the true zero-power inactive tones found in water filling.

## Python environment

Run the populated project under Python 3.11 through 3.13 with the dependencies declared in
`reference_training/requirements.txt`. The source directory intentionally has no `train_config.yaml`; the demo
preparation helper creates a configured copy beside the Workbench export.

For a bundle at a non-default path inside a uv-managed Noema checkout, stay at the repository root
and replace `<bundle>` below with its path relative to that root:

```bash
uv run --extra onnx python <bundle>/train_demo.py
uv run --extra onnx python <bundle>/evaluate_demo.py
```

For an independent training environment, create and activate a virtual environment, then install
the exported dependency manifest before running the script:

```bash
python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r reference_training/requirements.txt
python train_demo.py
python evaluate_demo.py
```

PyTorch publishes platform-specific CPU and CUDA wheels. If a particular CUDA build is required,
install PyTorch using its official platform selector first; the remaining requirements are still
declared by this project. The trainer automatically uses CUDA when available and otherwise uses CPU.

## Demonstration project workflow

After Workbench selects **Train/replace** in **Operation Training Capabilities**, the user sets
**Training bundle** > **Bundle directory**, reviews **Dataset definition** > **Captured signals** and
**Dataset size and splits**, chooses the **Support framework** and **Overwrite generated files**, and
selects **Export training bundle**. Export comes before capture because it freezes the replacement
ABI, selected tensors, split recipes and seeds, managed paths, and integrity hashes.
**Dataset capture** and **External model** remain visible throughout, but their actions are disabled
until bundle export. Training itself runs in the researcher's environment between capture and
returned-model validation.

The demo preparation helper copies the implementation under `reference_training/` and creates the
root launchers plus `RUN_DEMO.md`. The helper owns only those named demo files and refuses to replace
a same-named researcher file. Its ownership and non-normative status are recorded under
`external_training.optional_demo_scaffold.root_handoff` in `project_manifest.yaml`. The helper is
optional only for researchers who provide their own trainer; it is required before using this
README's example training and evaluation commands.
The neutral bundle root provides:

- `training_contract.yaml`, the architecture- and objective-neutral slot contract;
- `scenario_graph.json`, the typed frozen-context graph;
- `data_contract.yaml`, the integrity-bound CSI capture contract;
- `capture_train_recipe.yaml`, `capture_validation_recipe.yaml`, and `capture_test_recipe.yaml`,
  configured to capture CSI only with disjoint channel seeds;
- `project_manifest.yaml`, the durable Workbench contract for capture status, the external-training
  boundary, and returned-artifact discovery;
- `data/train`, `data/validation`, and `data/test`, the managed capture destinations.

The populated `train_config.yaml` and the example model, loss, and trainer remain under
`reference_training/`; they are replaceable and are not part of artifact compatibility. Researchers
who skip the helper can add their own model, loss, and trainer anywhere under the bundle directory.

After attaching the demo, Workbench detects the new files automatically. Use **Dataset capture** >
**Capture all datasets** to run the generated train, validation, and held-out-test recipes. Train
outside Noema, then return to **External model** and select **Validate returned model** once the
returned artifact is detected. That check verifies the returned ABI, hashes, contract binding, and
bundle integrity; it does not measure allocation quality. Held-out evaluation and the paired
benchmark provide the quality evidence.

The prepared `reference_training/train_config.yaml` already points to the bundle-owned train,
validation, and test captures. From the bundle root, run
`uv run --extra onnx python train_demo.py`. It trains several initialization seeds, selects only by
validation Shannon spectral efficiency, writes a portable ONNX policy, registers it in a schema-v2
`trained_artifact.yaml` at the bundle root bound to `training_contract.yaml`, and never reads an
oracle allocation. Run `uv run --extra onnx python evaluate_demo.py` for held-out water-filling
regret and KKT metrics. In Noema, select the artifact on any compatible
`model.symbol_power_allocator` block and compare it using the normal recipe/benchmark workflow.

The ONNX graph returns unconstrained allocation scores. Noema, not the external model, owns and
enforces the exact nonnegative fixed-sum projection at deployment.

From the Noema project root, the complete CLI alternative to Workbench capture followed by training
is:

```bash
uv run --extra onnx noema dataset-capture run <bundle>/capture_train_recipe.yaml \
  --out <bundle>/data/train
uv run --extra onnx noema dataset-capture run <bundle>/capture_validation_recipe.yaml \
  --out <bundle>/data/validation
uv run --extra onnx noema dataset-capture run <bundle>/capture_test_recipe.yaml \
  --out <bundle>/data/test

uv run --extra onnx python <bundle>/train_demo.py
uv run --extra onnx python <bundle>/evaluate_demo.py
cd <bundle>/reference_training
uv run --extra onnx python build_benchmark.py
```

Use the three bundle-owned paths already written in `train_config.yaml` as the `--out` values.

After training returns `trained_artifact.yaml`, `build_benchmark.py` binds that artifact to the
canonical allocator block and writes `benchmark_pack.yaml`. Every equal-power, learned, and
water-filling entry comes from the same `noema_recipe.yaml`; only the allocator policy, power
budget, and paired held-out seeds change. The helper defaults to three budgets and three held-out
seeds, embeds the `metadata.demo` publishing specification and training-evidence paths, and prints
the exact validation and run commands. Customize the campaign without editing YAML directly:

```bash
uv run --extra onnx python build_benchmark.py --budgets 0.5,1,2 --seeds 71001,72001,73001
```

Run the printed commands from the Noema project root. The generated pack resolves its recipe and
artifact references relative to its own location, while benchmark results continue to use the
project's ordinary `.noema` workspace.

Important: the classical oracle is optimal for Gaussian signaling and the Shannon objective. BER,
BLER, and goodput under finite QAM/coding are downstream practical measurements; they are not the
mathematical proof of convergence to water filling.
