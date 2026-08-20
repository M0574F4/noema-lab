# Physical-Layer Learned-Method Workflow

These demonstrations replace one operation in a fixed communication recipe, train the replacement
outside Noema, return a portable artifact, and compare it with classical references. Each demo page
provides its exact template, signals, sizes, paths, and commands.

## Common UI workflow

Copy complete shell blocks. Their first command returns to the repository root from any directory
inside the clone. Start Noema with:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run noema ui serve --port 8766
```

Open `http://127.0.0.1:8766`, select **Browse template recipes**, then use this sequence:

1. Open the template named by the demo and select **Workbench**.
2. Under **Operation Training Capabilities**, select **Train/replace** for the demo's Block.
3. Under **Dataset definition**, keep required model inputs selected and add only the targets or
   auxiliary signals used by the external loss.
4. Set the demo's record count, train percentage, validation percentage, and any capture-coordinate
   sweep. The remaining records form the held-out test split.
5. Under **Training bundle**, set the documented directory and framework. Enable **Overwrite
   generated files** only when intentionally rebuilding the bundle.
6. Select **Export training bundle**. This freezes the model ABI, capture plan, split seeds, paths,
   and integrity hashes before data is generated.
7. Run the demo's `prepare_example.py` command. Workbench detects changes to the selected bundle
   automatically while it is open.
8. Select **Capture all datasets** and wait for **ready to train**. Existing data can be replaced with
   **Recapture all datasets**.
9. Run the documented contract check, trainer, and held-out evaluator from the bundle directory.
10. When **External model** detects the returned artifact, select **Validate returned model**.
    Continue when the status is **model interface valid**.
11. Bind the returned artifact to the ordinary recipe for a smoke run, then run the paired benchmark
    described by the demo.

Workbench exports the interface and captures data; it does not execute researcher training code.
The checked-in trainers are optional examples, not part of the reusable recipe.

## What the capability table means

| Column | Meaning |
| --- | --- |
| **Block** | Human-readable role used in the graph and Recipe panel |
| **Built-in fine-tuning** | Whether Noema can optimize the installed operation itself |
| **Portable replacement** | Whether an external artifact can replace this Block through a complete ABI |
| **Gradient** | Backward behavior of the currently installed operation |
| **Differentiable support** | Whether the unchanged operation can remain on a live route to a loss |

Selecting a Block for replacement does not require its current implementation to be differentiable:
that implementation is removed. Differentiability matters only for unchanged downstream Blocks when
training through a live recipe loss.

All demos listed below use capture-backed training. They train from frozen records, so they do not
need a differentiable recipe route:

| Demo | Required input | Example target or objective |
| --- | --- | --- |
| OFDM allocation | Channel gains | Local Shannon objective; no policy labels |
| Delayed-CSI OFDM allocation | Four-snapshot delayed/noisy complex CSI history | Expected finite-blocklength goodput on the aligned later channel; no policy labels |
| MIMO-OFDM channel estimation | Operation-owned LS estimate and noise variance | Simulated 2×2 frequency-domain channel truth |
| QPSK demapping | Received I/Q symbols | Transmitted bits |
| QPSK carrier tracking | Impaired I/Q frame and public pilot context | Transmitted data bits |
| Modulation recognition | I/Q frames | Modulation class |

Additional graph outputs remain selectable because they may be valid labels, teacher outputs, or
diagnostics. Choosing them is part of a training plan, not the identity of the system template.

(ui-and-cli-boundary)=
## UI and CLI boundary

The UI is the recommended path. For automation, copy the block below from any directory inside the
clone. It returns to the repository root, exports a bundle, and materializes its three generated
capture recipes:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx noema differentiable export <recipe.yaml> \
  --training-plan <training_plan.yaml> \
  --out <bundle>

uv run --extra onnx noema dataset-capture run <bundle>/capture_train_recipe.yaml \
  --out <bundle>/data/train
uv run --extra onnx noema dataset-capture run <bundle>/capture_validation_recipe.yaml \
  --out <bundle>/data/validation
uv run --extra onnx noema dataset-capture run <bundle>/capture_test_recipe.yaml \
  --out <bundle>/data/test
```

Each `dataset-capture` command reports progress for its own split; let it complete before the next
command starts. Add `--force` only when intentionally replacing existing generated files or
captures. The exact demo pages then supply the trainer, evaluator, benchmark builder, and
artifact-return commands.

Benchmark creation, campaign launch, result verification, and static publication currently remain
command-line operations. After a run, reload Noema, select **Results**, and open its entry under
**Benchmark results** to inspect the same stored evidence in the UI.

## Reproducibility checklist

- Select checkpoints using train and validation data only.
- Keep train, validation, held-out test, and benchmark records disjoint.
- Freeze the returned artifact before comparison.
- Use identical data and channel seeds for all methods at each benchmark coordinate.
- Change only the operation under study; keep the scenario, budgets, and metrics fixed.
- Treat an upper bound as an upper bound, not a matched competitor.
- Verify the stored result before publishing it.

## Publish a completed benchmark

The benchmark run prints a result ID whose evidence is stored under
`.noema/benchmarks/<result_id>/`. From any directory inside the clone, copy the block below; it
returns to the repository root before publishing that immutable result:

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --extra onnx noema benchmark verify <result_id>
uv run --extra onnx noema benchmark publish <result_id> \
  --slug <stable-demo-slug> \
  --out docs/demo/experiments/<stable-demo-slug>
```

Publication reads stored results; it does not rerun recipes or training. Use `--force` only when
intentionally replacing an existing deterministic publication.

## Demo guides

All completed and in-progress guides are grouped on the
[Demonstrations](../demos.md) page.

The [Shannon-allocation demo](ofdm_resource_allocation_demo.md) assumes current CSI and has an
analytical water-filling reference. The
[finite-blocklength delayed-CSI demo](reliability_aware_ofdm_allocation_demo.md) instead gives every
deployable allocator the same causal four-snapshot CSI history and scores reliability on the
aligned later channel. Memoryless QPSK demapping predicts bits from synchronized symbols. QPSK
carrier tracking uses a whole pilot-aided frame to handle phase and frequency drift. Modulation
recognition predicts the modulation family from a frame. They are separate benchmarks.
