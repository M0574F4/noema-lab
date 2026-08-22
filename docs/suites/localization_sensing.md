# Localization / Sensing Suite

This experimental suite covers two explicit, runnable sensing protocols: four-anchor 2D range localization and
single-source narrowband angle-of-arrival estimation with a uniform linear array.

## Benchmark Pack

- `benchmarks/localization_sensing/range_localization_v1.yaml`
- `benchmarks/localization_sensing/aoa_estimation_v1.yaml`

The pack compares:

- `recipes/localization_trilateration_baseline.yaml` using `model.trilateration_localizer`;
- `recipes/localization_adapter_baseline.yaml` using `model.localization_adapter`.

The range graph separates geometry from observation. Declared SNR now causally controls range
uncertainty; a separately declared measurement floor and optional NLOS bias remain visible
parameters.

The range recipes declare `range_localization`, requiring geometry, range observation, localizer,
and position evaluation as separate stages.

The AoA pack compares:

- `recipes/aoa_music_ula_baseline.yaml` using `model.music_aoa_estimator`;
- `recipes/aoa_adapter_ula_baseline.yaml` using a typed endpoint whose ordinary-run reference is a Bartlett spatial spectrum.

Its initial popular scenario is intentionally focused: one far-field narrowband source, an
eight-element half-wavelength ULA, 64 complex snapshots, and AWGN. The graph is:

```text
source-angle scene
  -> ULA noisy snapshots
  -> MUSIC or Bartlett-reference endpoint
  -> angular-error evaluation
```

The AoA recipes declare `aoa_array_estimation`, requiring the angular scene, array observation,
estimator, and angular-error stages independently of whether MUSIC or an adapter is selected.

## Adapter Point

Both replacement points now implement the full capture, training, artifact-return, and paired-
benchmark loop:

- `model.localization_adapter` passes `[batch, anchor, 2]` anchor coordinates and
  `[batch, anchor]` noisy ranges to a returned `localization_estimator` ONNX entrypoint. The
  checked-in trainer uses true positions only as offline supervision and compares the result with
  linear and regularized trilateration. See [Learned range localization](../tutorials/learned_range_localization_demo.md).
- `model.aoa_estimator_adapter` passes complex snapshots as
  `[batch, antenna, snapshot, 2]` real/imaginary tensors to a returned `aoa_estimator` ONNX
  entrypoint. The checked-in covariance-domain trainer compares the returned model with Bartlett
  and MUSIC. See [Learned AoA estimation](../tutorials/learned_aoa_estimation_demo.md).

Each artifact is hash-pinned, validated against the operation-owned ABI, and evaluated on a sealed
test capture before its post-training benchmark is built.

## Metrics and Plots

Core metrics:

- `localization.rmse_m`;
- `localization.mae_m`;
- `localization.p90_error_m`;
- `task.score`;
- `channel.snr_db`.

AoA metrics:

- `aoa.rmse_deg`;
- `aoa.mae_deg`;
- `aoa.median_error_deg` and `aoa.p90_error_deg`;
- `task.score` and `channel.snr_db`.

Default plot:

```bash
uv run noema benchmark run benchmarks/localization_sensing/range_localization_v1.yaml
uv run noema benchmark plot <result_id> --plot graceful-degradation --x channel.snr_db --y task.score --group method --out figures/localization_score.png
uv run noema benchmark run benchmarks/localization_sensing/aoa_estimation_v1.yaml
```

## Boundary

The range protocol is not a synchronized UWB waveform, and the AoA protocol does not model multiple
sources, coherent multipath, array calibration error, or near-field propagation. AoA/ToA fusion,
NLOS protocols, Sionna RT scenes, and radio maps are outside these baselines.
