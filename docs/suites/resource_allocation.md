# Resource Allocation Suite

The suite also includes a separate [joint ISAC OFDM allocation demonstration](../tutorials/learned_isac_ofdm_allocation_demo.md).
That synthetic protocol exposes communication and sensing gains together, enforces one exact
sum-power budget, and compares equal power, communication-only water filling, a scalarized
per-scene optimizer, and a portable learned policy. It does not reuse the communication-only pack
or present the scalarized objective as a waveform-level sensing metric.

This experimental suite compares subcarrier power-allocation policies by configuring the canonical communication
pipeline rather than constructing a separate resource-allocation graph. Both recipes use the same
seeded random-bit source, canonical payload and TX bit boundaries, channel-code stage, QPSK
modulator, Sionna 3GPP TDL-A OFDM realization, `wireless.channel`, demodulator, channel decoder, BER/
BLER accounting, and resource-allocation metrics. Only the policy on
`model.symbol_power_allocator` changes.

## Benchmark Pack

- `benchmarks/resource_allocation/power_allocation_v1.yaml`

The pack compares:

- `recipes/resource_equal_power_baseline.yaml` with `policy: fixed`;
- `recipes/resource_water_filling_baseline.yaml` with `policy: water_filling`.

`wireless.ofdm_channel_state` is the explicit CSI boundary added to the main pipeline. It generates a
reproducible Sionna TDL frequency response sized to the actual modulated bit payload. Both
the allocator and `wireless.channel` have a visible `channel_state` edge from that artifact, so the
oracle label, applied fading, receiver metrics, and training capture refer to the same channel
sample. This is the perfect instantaneous CSIT/CSIR assumption used by the theoretical oracle.

Each allocatable channel is one OFDM subcarrier for one OFDM-symbol channel state, not an antenna
and not a user-data symbol. Noise variance is fixed at 0.2. The template matrix evaluates normalized
average transmit-power budgets 0.5, 1, and 2; within every matrix point all policies receive the same
fixed budget. Thus the allocator redistributes power across subcarriers without changing a point's
mean TX power or sweeping the noise floor. An SNR parameter is inactive in this fixed-variance mode
and cannot be bound to a recipe or capture sweep.

The oracle uses
`p_k = max(mu - noise_variance / gain_k, 0)`, choosing `mu` so the powers sum to
`fft_size * average_tx_power`. The `noise_variance / gain_k` term is the inverse-unit-SNR floor.
Higher-gain subcarriers generally receive more power, while sufficiently weak subcarriers receive
zero; water-filling is not allocation proportional to inverse SNR.

## Training Boundary

The training capture contains only `channel_state.state`, with the realized Sionna frequency-response
gains. Noise variance and the instantaneous average-power budget are runtime conditions. Oracle power
is deliberately not a training tap: the OFDM demonstration trainer minimizes negative Shannon spectral
efficiency and satisfies nonnegativity and the exact sum-power constraint through a differentiable
simplex projection. A researcher may replace the demonstration's model, objective, and trainer while
keeping the same data, constraint, and artifact-return contracts.

Water filling is introduced only after checkpoint selection, on held-out channel seeds. A learned
allocator replaces the policy at `tx_power` while the source, CSI generator, noise variance, channel,
receiver, metrics, and seeds stay fixed. This makes learned-versus-oracle comparisons sample-aligned
without turning training into imitation of the oracle.

## Metrics and Dashboard

Core evidence includes the theoretical parallel-Gaussian-channel sum objective per OFDM symbol,
theoretical full-band Shannon spectral efficiency, maximum power-budget error, active resource-element
fraction, TX power, coded and post-decoder BER, and image reconstruction metrics. The theoretical
spectral efficiency is `mean_k log2(1 + |h[k]|² p[k] / noise_variance)`; it is not achieved QPSK
payload throughput. In the Communication tab, canonical
resource-allocation runs show both the ordinary link figures and a selectable per-state plot with
unit-power subcarrier SNR and allocated power. Water filling is the Shannon sum-rate oracle; without
adaptive bit loading, it is not expected to minimize fixed-QPSK BER or image-decoder outages.

## Delayed-CSI finite-blocklength demonstration

The separate **Reliability-aware OFDM allocation with delayed CSI** template keeps the same
resource-allocation purpose but changes the research question. It gives the allocator four causal
complex CSI snapshots from a non-normalized wideband Sionna TDL-C trajectory and withholds the
channel state five OFDM symbols later. Training and evaluation score finite-blocklength expected
goodput on that later state. In this setting, water filling on the newest delayed snapshot is a
mismatched baseline rather than an oracle. See
[the step-by-step demonstration](../tutorials/reliability_aware_ofdm_allocation_demo.md) for the
training workflow, interactive paired-seed result figures, and immutable benchmark provenance.
