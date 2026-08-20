# MIMO-OFDM Suite

This experimental suite provides a runnable 2×2 frequency-selective MIMO-OFDM
channel-estimation protocol. The learned-estimator template uses a mixture of
Sionna 3GPP TDL-A/C/E channels while retaining explicit example, receive-antenna,
transmit-antenna, and subcarrier axes. Orthogonal comb pilots produce noisy
observations, and Noema materializes sparse divided pilots, their mask, and an
LS/interpolation fallback for portable learned estimators.

## Benchmark Pack

- `benchmarks/mimo_ofdm/channel_estimation_v1.yaml`

The pack compares:

- `recipes/mimo_ofdm_ls_channel_estimation.yaml` as an interpolation-aware LS baseline;
- `recipes/mimo_ofdm_adapter_channel_estimation.yaml` as the mixed-profile
  LS/fixed-prior-LMMSE/portable-learned endpoint.

```text
2×2 Sionna 3GPP TDL-A/C/E channel truth
  -> orthogonal comb-pilot grid
  -> SNR-controlled noisy pilot observation
  -> LS/interpolation, fixed-prior LMMSE, or portable learned endpoint
  -> NMSE and post-ZF spectral-efficiency evaluation
```

Both recipes declare `mimo_ofdm_channel_estimation`, which distinguishes this pilot-grid spine from
the flat/SISO `pilot_channel_estimation` profile without turning antenna or pilot counts into
profiles.

## Adapter Point

`model.channel_estimator_adapter` is a selectable **Train/replace** boundary.
Its portable ONNX ABI receives sparse divided pilot observations, their binary
mask, an operation-owned LS channel estimate, and noise variance. It returns
an estimated complex channel tensor of the same shape. Channel truth remains
an offline capture target.

The complete training and benchmark workflow is
[Learned 2×2 MIMO-OFDM channel estimation](../tutorials/learned_mimo_ofdm_channel_estimation_demo.md).

## Metrics and Plots

Core metrics:

- `mimo.channel_estimation.nmse`;
- `mimo.channel_estimation.nmse_db`;
- `mimo.channel_estimation.zf_spectral_efficiency_bps_hz`;
- `mimo.channel_estimation.zf_rate_retention`;
- `channel_estimation.nmse`;
- `task.score`;
- `channel.snr_db`.

Default plot:

```bash
uv run noema benchmark run benchmarks/mimo_ofdm/channel_estimation_v1.yaml
uv run noema benchmark plot <result_id> --plot graceful-degradation --x channel.snr_db --y task.score --group method --out figures/mimo_ofdm_channel_estimation.png
```

## Boundary

This is a channel-estimation benchmark, not a complete 5G NR conformance test.
The post-ZF metric measures the downstream sensitivity of the estimate; it
does not add a full coded waveform, standardized DMRS, or BER/BLER claim.
