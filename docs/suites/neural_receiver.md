# Neural Receiver / AI-PHY Suite

This experimental suite covers AI-native physical-layer receiver methods: learned demappers, receiver-only
training, and BER/BLER link benchmarks. It is the first non-semantic expansion suite because it
reuses Noema's existing symbol boundaries, channel blocks, dataset capture runner, differentiable export, and
bit-accounting checks.

## Runnable Proof Benchmark

The first benchmark pack is:

```text
benchmarks/neural_receiver_ai_phy/qpsk_awgn_receiver_v1.yaml
```

It runs a defensible small link-level chain:

```text
seeded random bits
  -> canonical payload/tx bit boundaries
  -> QPSK mapper
  -> AWGN channel
  -> classical demapper baseline or neural receiver adapter
  -> BER / BLER
```

The built-in neural receiver adapter currently has a deterministic `reference_qpsk` mode and a tiny
`linear_npz` checkpoint mode. The reference mode is not claimed to be a trained neural receiver; it is
the stable adapter slot that lets benchmark packs, graph rendering, capture taps, manifests, and
result plots exercise the same path a researcher would replace with a trained receiver.

The reusable **QPSK receiver calibration under I/Q imbalance** template adds a
stable receiver front-end distortion after AWGN. Its post-training campaign
compares ordinary uncompensated QPSK, a simulation-only calibrated oracle, and
a learned artifact on paired bits and noise. Unlike the ideal-AWGN smoke pack,
this scenario has a meaningful learned objective: infer the shifted and rotated
decision boundaries without receiving the hidden calibration parameters.

## Capture Shape

Receiver-only training data can be captured from taps such as:

```text
rx_symbols + noise variance + target_bits -> offline receiver training dataset
```

The benchmark recipes keep `tx_bit_boundary` and `rx_bit_boundary` in the standard Noema transport
spine so bit accounting, BER, BLER, and channel-use accounting stay comparable to semantic
communication recipes.

## Current Boundary

The current suite does not include Sionna LDPC/coded receiver baselines, soft-output neural receivers
with LLR losses, channel-mismatch generalization grids, or OFDM/MIMO receiver tasks. These capabilities
require separate benchmark protocols rather than implicit extensions of the QPSK/AWGN workflow.
