# DeepJSCC image reconstruction demonstration project

This template trains image-to-symbol and symbol-to-image endpoint slots jointly
through the frozen differentiable channel selected by the recipe. The supported
demonstrations use either complex AWGN or blind slow Rayleigh fading. Noema binds
the copied project to the selected recipe; model optimization remains external.

The packaged workflow uses one pure-PyTorch autograd graph. Sionna 2 is
PyTorch-native and available through Noema's generic differentiable export graph,
but this specialized starter deliberately keeps its own qualified native channel
blocks. Ordinary returned-artifact benchmarks keep the recipe's explicit
`wireless.channel` semantics.

`training_plan.yaml` is the demonstration-owned selection of endpoint slots and MSE objective. It
is separate from the runnable image-communication recipe.

## Exported project

The exporter copies `model.py`, `scenario.py`, `datamodule.py`, `losses.py`,
`structured_input.py`, `train.py`, `evaluate.py`, both benchmark builders, and
`requirements.txt`, then generates recipe-bound configuration, contract, and
project-manifest files.

Populate the configured image dataset, then run from the exported directory. The export resolves the
recipe-selected files once, records a SHA-256 for every file, and materializes explicit `train` and
`validation` lists in the root `data_contract.yaml`. The loader consumes those exact lists and rejects
missing, changed, overlapping, or undeclared files; it does not recompute an implicit split at runtime.

```bash
python -m pip install -r requirements.txt
python train.py
python evaluate.py
```

Held-out test images are intentionally absent from the training project. Select them only in a
separate ordinary Noema recipe after the paired artifact has returned. `evaluate.py` reports
validation behavior and must not be presented as held-out test evidence.

Training writes:

- `artifacts/encoder.onnx` and `artifacts/decoder.onnx`: portable executable endpoint graphs;
- `trained_artifact.yaml`: schema-v2, contract-bound paired encoder/decoder bindings;
- `training_contract.yaml`: the exact neutral Noema contract copied into the returned package;
- `training_history.json`: training and validation history.

The compact residual CNN downsamples by eight. The AWGN example uses 32 complex
symbol channels (`κ=0.5`). The slow-fading example trains one nested
representation with 8, 16, and 32 active channels (`κ=0.125, 0.25, 0.5`).
Deterministic training crops and dihedral
augmentations increase sample diversity without exposing held-out files. This CNN is only the
demonstration's architecture. The returned artifact ABI is
architecture- and loss-neutral: ONNX Runtime executes typed `encoder` and `decoder` entrypoints,
while the manifest pins both graphs and the neutral contract by SHA-256. Selecting the artifact
applies both endpoint roles atomically; no generated operation ID or Python adapter is required.

After training and validation, use the post-training command recorded in
`project_manifest.yaml`. `build_benchmark.py` creates the fixed-bandwidth AWGN
campaign; `build_slow_fading_benchmark.py` creates the paired no-CSI SNR and
bandwidth sweeps.
