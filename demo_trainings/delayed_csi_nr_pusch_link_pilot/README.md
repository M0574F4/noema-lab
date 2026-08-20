# Delayed-CSI NR-PUSCH link pilot

This is a bounded development-only go/no-go pilot. It leaves the frozen
equal-power NR-PUSCH study untouched and asks whether an existing TDL-trained
learned allocator transfers directionally to a more concrete PUSCH chain.

The chain includes NR transport-block coding and CRC, QPSK MCS 7, DMRS-based
receiver channel estimation, a continuous TDL-C300 channel at 3.5 GHz and
120 km/h, normal-CP OFDM, and a fixed 5 dB normalized Eb/N0. Four noisy CSI
snapshots end five OFDM symbols before the PUSCH slot. All methods share the
payload, physical channel, CSI noise, AWGN, exact waveform energy, and
per-PRB power bounds.

The learned model is seed 35023 from the codec-aligned development study. Its
use here is explicitly an out-of-distribution transfer test. The successful
16-tone two-path model is not reused because its interface and channel are
incompatible with PUSCH.

Run from the repository root:

```bash
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/pilot.py self-test
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/pilot.py smoke
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/pilot.py run --max-units 1
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/pilot.py run
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/pilot.py analyze
```

Five units contain eight independent trajectories each, for 40 paired CRC
attempts per method. This is far too small for a paper superiority claim. It
only decides whether a separately frozen, larger prospective NR study is worth
running.

Even a successful follow-up would be a normalized NR-framed link-level case,
not a deployment, field, or NR-conformance benchmark. Per-PRB spectral power
shaping and the reciprocal/SRS-like CSI history are explicit abstractions.

## Post-pilot SNR diagnostics

The original eight-block result was too coarse to rule out a CRC-waterfall
effect. A development-only paired sweep therefore evaluated 3--7 dB in 0.5 dB
steps with 32 attempts per method and fresh seed namespaces:

```bash
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/snr_sweep.py run \
  --design broad_v2
```

At 6.5 dB the learned transfer model had 22/32 deliveries, versus 21/32 for
equal power and 20/32 for both delayed-CSI water-filling methods. Because that
one-delivery advantage was exploratory, a post-outcome focused refinement used
96 fresh paired attempts per method at 6.25, 6.50, and 6.75 dB:

```bash
.venv/bin/python -I -B \
  demo_trainings/delayed_csi_nr_pusch_link_pilot/snr_sweep.py run \
  --design focused_v1
```

The learned/equal delivery counts were 63/63, 64/64, and 65/67,
respectively. No candidate coordinate passed the development gate. These
results do not support opening a confirmatory NR campaign with the transferred
model.

An earlier `delayed_csi_nr_pusch_link_snr_sweep_v1` directory is retained but
excluded: its worker changed the nominal SNR label while the fixed-coordinate
pilot's 5 dB noise helper remained active. The corrected broad sweep is
`delayed_csi_nr_pusch_link_snr_sweep_v2`; the focused refinement is
`delayed_csi_nr_pusch_link_snr_refinement_v1`. This bug does not change the
original pilot because that pilot was intentionally fixed at 5 dB.
