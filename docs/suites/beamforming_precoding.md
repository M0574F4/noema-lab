# Beamforming / Precoding Suite

This experimental suite benchmarks AI-native beam selection and precoding under fixed channel and SNR protocols.
The current v1-draft pack uses synthetic flat-fading MISO channels and explicitly labels both
policies by the information they receive: perfect-CSIT MRT is an upper bound, while exhaustive DFT
codebook selection is an oracle within a finite beam codebook.

## Benchmark Pack

- `benchmarks/beamforming_precoding/beam_selection_v1.yaml`

The pack compares:

- `recipes/beamforming_mrt_baseline.yaml` using `model.mrt_beamformer` as the perfect-CSIT upper bound;
- `recipes/beamforming_adapter_baseline.yaml` using `model.beamforming_adapter` as the finite-codebook oracle and typed adapter boundary.

Both declare `beamforming_link_evaluation`: a realized link must feed a beamformer and the selected
weights must be scored on that link. The profile does not choose MRT, a codebook, or a learned
policy.

## Adapter Point

`model.beamforming_adapter` consumes `ai_phy.beamforming_problem.numpy` and emits
`ai_phy.beamforming_decision.numpy`. It currently has no trained-artifact ABI, so it is not
selectable under **Train/replace**; its present role is a runnable finite-codebook reference.

## Metrics and Plots

Core metrics:

- `beamforming.spectral_efficiency_bps_hz`;
- `beamforming.normalized_gain`;
- `beamforming.array_gain_db`;
- `task.score`;
- `channel.snr_db`.

Default plot:

```bash
uv run noema benchmark run benchmarks/beamforming_precoding/beam_selection_v1.yaml
uv run noema benchmark plot <result_id> --plot graceful-degradation --x channel.snr_db --y task.score --group method --out figures/beamforming_score.png
```

## Boundary

This is not a multi-user, mobility-aware, beam-tracking, or Sionna RT beam-alignment benchmark.
It also does not claim a feedback-bit budget: both current policies observe the realized channel.
Feedback-constrained beam selection, multi-user ZF/RZF, and beam tracking require separate protocols
with explicit CSI acquisition, feedback errors, baselines, and channel-use accounting.
