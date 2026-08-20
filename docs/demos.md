# Demonstrations

These guides start from reusable system templates, train external replacements, return portable
artifacts, and run paired benchmarks. Physical-layer examples capture selected tensors at a block
boundary. End-to-end image examples instead use file-backed data contracts and train both
communication endpoints together.

For the shared UI concepts and reproducibility rules, read the
[physical-layer learned-method workflow](tutorials/physical_layer_demo_workflow.md).

## Flagship Interactive Demo

[Break the comparison](break_the_comparison.md) is the shortest route into Noema's core idea. It
loads the canonical launch evidence, starts with a valid paired comparison, and lets you introduce
condition, aggregation, metric, and comparator-role faults one at a time.
Its static companions and canonical result table are collected in the
[F0–F5 launch visual system](launch_assets.md).
Watch the [complete learned QPSK I/Q calibration workflow](https://www.youtube.com/watch?v=bKNXS_vHLHc)
for the clean installation, contract export, external training, returned model, and UI comparison.

## Resource Allocation

- [Learned OFDM subcarrier allocation](tutorials/ofdm_resource_allocation_demo.md)
- [Reliability-aware OFDM allocation with delayed CSI](tutorials/reliability_aware_ofdm_allocation_demo.md)

## Channel Estimation and Feedback

- [Learned 2×2 MIMO-OFDM channel estimation](tutorials/learned_mimo_ofdm_channel_estimation_demo.md)
- [Learned CSI compression and feedback](tutorials/learned_csi_feedback.md)

## Receivers and Signal Classification

- [Learned QPSK receiver calibration](tutorials/learned_qpsk_demapper_demo.md)
- [Learned QPSK carrier tracking](tutorials/learned_qpsk_phase_tracking_demo.md)
- [Automatic modulation recognition](tutorials/automatic_modulation_recognition_demo.md)

## End-to-End Communication

- [DeepJSCC vs. capacity-matched JPEG over AWGN](tutorials/digital_vs_deepjscc_sionna.md)
- [Blind DeepJSCC over slow Rayleigh fading](tutorials/deepjscc_slow_rayleigh.md)

```{toctree}
:hidden:
:maxdepth: 1

tutorials/ofdm_resource_allocation_demo
tutorials/reliability_aware_ofdm_allocation_demo
tutorials/learned_mimo_ofdm_channel_estimation_demo
tutorials/learned_csi_feedback
tutorials/learned_qpsk_demapper_demo
tutorials/learned_qpsk_phase_tracking_demo
tutorials/automatic_modulation_recognition_demo
tutorials/digital_vs_deepjscc_sionna
tutorials/deepjscc_slow_rayleigh
```
