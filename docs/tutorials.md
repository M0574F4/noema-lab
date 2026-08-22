# Tutorials

## Flagship Workflows

These tutorials show the recommended public workflow for communication research results:

1. [Blind DeepJSCC versus ideal digital separation over slow fading](tutorials/deepjscc_slow_rayleigh.md)
2. [Export an architecture-neutral training contract](tutorials/export_differentiable_training_scenario.md)
3. [Return a schema-v2 ONNX artifact and run an ordinary benchmark](tutorials/external_training_checkpoint_adapter.md)
4. [Train and benchmark learned CSI compression and feedback](tutorials/learned_csi_feedback.md)
5. [Build and publish physical-layer learned-method demos](tutorials/physical_layer_demo_workflow.md)

Use them in order when developing a new learned communication method: first understand the baseline
plot, then export a neutral training contract, then return the trained model through its declared
artifact ABI.

## End-to-End Communication Demonstrations

- [DeepJSCC vs. capacity-matched JPEG over AWGN](tutorials/digital_vs_deepjscc_sionna.md) is a
  compact fixed-bandwidth implementation and training check.
- [Blind DeepJSCC versus ideal digital separation over slow fading](tutorials/deepjscc_slow_rayleigh.md)
  adds a no-CSI outage regime and a bandwidth sweep for research-facing analysis.

## Physical-Layer Demonstrations

Each page opens with one copyable CLI training summary, then gives the equivalent UI selections,
bundle path, trainer details, fair benchmark methods, result views, and publication command:

- [Learned OFDM subcarrier allocation](tutorials/ofdm_resource_allocation_demo.md)
- [Reliability-aware OFDM allocation with delayed CSI](tutorials/reliability_aware_ofdm_allocation_demo.md)
- [Learned 2×2 MIMO-OFDM channel estimation](tutorials/learned_mimo_ofdm_channel_estimation_demo.md)
- [Learned CSI compression and feedback](tutorials/learned_csi_feedback.md)
- [Learned QPSK receiver calibration](tutorials/learned_qpsk_demapper_demo.md)
- [Learned QPSK carrier tracking](tutorials/learned_qpsk_phase_tracking_demo.md)
- [Automatic modulation recognition](tutorials/automatic_modulation_recognition_demo.md)
- [Learned two-dimensional range localization](tutorials/learned_range_localization_demo.md)
- [Learned narrowband AoA estimation](tutorials/learned_aoa_estimation_demo.md)
- [Learned MISO beam selection](tutorials/learned_beam_selection_demo.md)

## Run the Kodak Development Benchmark

This experimental, unfrozen image-reconstruction pack includes a pretrained CompressAI baseline.
It is a source-checkout workflow because the benchmark pack and recipes are repository assets; the
project has not yet been published on PyPI. A cold run downloads the Kodak images and pretrained
checkpoint, may take several minutes on CPU, and requires the researcher to confirm the applicable
dataset and model rights. Run the block from the repository root. The result is development
evidence, not publication evidence.

```bash
uv sync --extra compressai
uv run noema benchmark validate benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
uv run noema benchmark run benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
uv run noema benchmark results
```

Open the generated `.noema/benchmarks/<result_id>/summary.md` and `metrics.csv`.

## Add an External Codec

```bash
uv run noema adapter scaffold adapters/my_codec --name my_codec --kind bits
uv run noema adapter validate adapters/my_codec/noema_adapter.yaml
```

Edit `adapters/my_codec/adapter.py`, then create a recipe that uses
`model.my_codec_encode_bits` and `model.my_codec_decode_bits` in the canonical bit spine.

## Add an External Metric or Task Dataset

```bash
uv run noema adapter scaffold adapters/my_metric --name my_metric --kind classification_metric
uv run noema adapter scaffold adapters/my_dataset --name my_dataset --kind classification_dataset
```

For a complete example:

```bash
uv run noema --adapter examples/adapters/classification_task \
  benchmark run examples/benchmarks/external_classification_adapter_smoke.yaml
```

## Interpret a Manifest

After any recipe run:

```bash
uv run noema runs list
uv run noema runs manifest <run_id>
```

Look for:

- `recipe.sha256`: exact recipe identity;
- `operation_contracts`: operation versions and typed input/output contracts;
- `environment`: Python, package, native extension, and git evidence;
- `seed_policy`: deterministic seed derivation;
- `artifacts`: output paths, hashes, dtype, shape, and metadata.

## Reproduce a Reported Table

1. Check the reported benchmark ID/version.
2. Validate the same benchmark pack.
3. Register the submitted adapter, if any.
4. Run the submitted recipe or benchmark pack.
5. Compare `metrics.csv`, `recipes.csv`, and recipe SHA-256 values.
6. Validate the submission bundle with `noema submission validate`.
