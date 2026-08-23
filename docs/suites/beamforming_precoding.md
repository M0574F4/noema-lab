# Beamforming / Precoding Suite

This experimental suite benchmarks AI-native beam selection and precoding under fixed channel and SNR protocols.
The current v1-draft pack uses synthetic clustered-ULA MISO channels and explicitly labels methods
by the information and beam budget they receive: perfect-CSIT MRT is an upper bound, while the
fixed DFT method exhaustively searches eight reusable beams.

## Benchmark Pack

- `benchmarks/beamforming_precoding/beam_selection_v1.yaml`

The pack compares:

- `recipes/beamforming_mrt_baseline.yaml` using `model.mrt_beamformer` as the perfect-CSIT upper bound;
- `recipes/beamforming_adapter_baseline.yaml` using `model.beamforming_adapter` as the fixed eight-beam DFT baseline and typed adapter boundary.

Both declare `beamforming_link_evaluation`: a realized link must feed a beamformer and the selected
weights must be scored on that link. The profile does not choose MRT, a codebook, or a learned
policy.

## Adapter Point

`model.beamforming_adapter` consumes `ai_phy.beamforming_problem.numpy` and emits
`ai_phy.beamforming_decision.numpy`. Its portable `beam_policy` ABI passes the realized channel as
`[batch, tx_antenna, 2]` real/imaginary values and requires a unit-norm beam with the same shape.

The checked-in [learned beam-selection workflow](../tutorials/learned_beam_selection_demo.md)
captures channel vectors, learns eight constant-modulus beam directions by normalized gain,
exports a hash-pinned ONNX policy, evaluates it on a sealed test split, and builds a paired
comparison with an equal-size fixed DFT codebook and the perfect-CSIT MRT upper bound.

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

The suite now includes one such separate mobility protocol: [LEO-NTN Doppler prediction and beam
handover](../tutorials/learned_leo_ntn_tracking_demo.md). It fixes a causal observation history,
one-second horizon, and nine beam sectors, then compares hold-last, linear extrapolation, a portable
learned tracker, and a future-state oracle. Its bounded kinematic generator is intentionally not
treated as an orbital or 3GPP NTN channel model.
