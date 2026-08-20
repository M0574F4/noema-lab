# Channel Estimation Suite

This experimental suite benchmarks AI-assisted channel acquisition under controlled pilot and feedback
protocols. It contains both pilot-channel estimation and a complete limited-feedback CSI loop in
which reconstructed CSI drives downlink precoding.

## Benchmark Pack

- `benchmarks/channel_estimation/pilot_awgn_v1.yaml`
- `benchmarks/channel_estimation/csi_feedback_v1.yaml`

The pilot pack includes low- and high-SNR runs for:

- `recipes/channel_estimation_ls_awgn.yaml` using `model.ls_channel_estimator`;
- `recipes/channel_estimation_adapter_awgn.yaml` using `model.channel_estimator_adapter`.

Both recipes expose the same five-block protocol:

```text
seeded flat-SISO channel truth
  -> known QPSK pilot pattern
  -> SNR-controlled noisy pilot observation
  -> LS or LMMSE-reference estimator
  -> exact-shape NMSE evaluation
```

They declare `pilot_channel_estimation`; planner validation therefore requires every one of those
physical boundaries while allowing either supported estimator implementation.

The adapter currently provides a typed, runnable reference endpoint, not a portable replacement
boundary. Its ordinary-run materialization is an explicitly labeled scalar LMMSE reference, not a
claim that training has occurred. The separation follows the usual pilot-position estimation
contract used by Sionna channel-estimation components while keeping the default scenario fast and
dependency-light.

The CSI-feedback pack compares matched-budget truncated angular-delay compression with a learned
encoder/decoder artifact and reports a separately labeled perfect-CSIT upper bound. Every method
uses the same seeded, spatially correlated Sionna TDL MISO-OFDM realizations. The standard feedback
path uses a fixed-dimensional latent and explicit uniform quantization; continuous latents are never
reported as feedback bits.

Its standard executable profile is:

```text
Sionna MISO-OFDM CSI
  -> UE feedback encoder
  -> limited feedback link
  -> BS feedback decoder
  -> reconstructed-CSI MRT precoder
  -> true-channel rate evaluation
```

The architecture-neutral training contract captures only realized complex CSI and channel
metadata. The paired encoder and decoder return as one atomic artifact. The optional training
project used by the documentation demo is separate example code; model architecture, loss,
optimizer, and trainer remain
external choices.

## Adapter Point

The reference estimator uses the typed input/output shape
`model.channel_estimator_adapter`: `ai_phy.channel_estimation_problem.numpy` in and
`ai_phy.channel_estimate.numpy` out. The current operation does not yet declare a trained-artifact ABI,
so Workbench correctly shows **Portable replacement: No** and does not offer **Train/replace**.

The recipe keeps channel realization, pilot design, observation noise, and estimation separate, so
users can capture and study any one of them without editing a fused source block. Portable model
return becomes available only after the estimator ABI and runtime binding are implemented.

## Metrics and Plots

Core metrics:

- `channel_estimation.nmse`;
- `channel_estimation.mse`;
- `task.score = 1 / (1 + NMSE)`;
- `channel.snr_db`.

CSI-feedback system metrics:

- `csi_feedback.achieved_spectral_efficiency_bps_hz`;
- `csi_feedback.spectral_efficiency_retention` and rate loss versus perfect CSIT;
- `csi_feedback.nmse_db` and phase-invariant cosine correlation;
- latent dimension/compression factor;
- actual feedback bits only when the feedback link quantizes the latent.

Achieved downlink spectral efficiency and its perfect-CSIT retention are the primary conclusions.
CSI NMSE is a reconstruction diagnostic, not a substitute for downstream link performance.

Default plot:

```bash
uv run noema benchmark run benchmarks/channel_estimation/pilot_awgn_v1.yaml
uv run noema benchmark plot <result_id> --plot graceful-degradation --x channel.snr_db --y task.score --group method --out figures/channel_estimation_score.png
```

## Boundary

The CSI-feedback protocol assumes perfect CSI at the UE and evaluates single-user MRT. It does
not claim a 3GPP Type-II codebook, multi-user precoding, noisy pilot acquisition, or feedback-channel
errors.

The pilot-AWGN protocol is a flat-fading SISO baseline. Frequency-selective 2×2 comb pilots are
covered by the MIMO-OFDM suite; coded BER/BLER and mobility are outside this protocol.
