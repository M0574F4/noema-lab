# Packet-context QPSK phase-tracking example

This optional example trains a temporal receiver for the exported
`demodulation.phase_tracking_receiver_adapter` contract. It consumes the
pilot-smoother-corrected frame plus observable raw, pilot-innovation, smoother,
and fourth-power phase cues. Its depthwise-separable temporal network predicts
one full-circle residual phase per symbol. A zero residual exactly reproduces
the strong pilot-smoothing baseline.

The ONNX ABI takes `receiver_features_v3 [packet, frame_symbol, 11]` and returns
only `residual_phase_rad [packet, frame_symbol]`. Noema deterministically rotates
the corrected I/Q in feature channels 0 and 1 to obtain bit scores; the model
cannot improve its loss by learning an unrelated logit gain.

The channel's phase trace supplies an auxiliary circular loss during training
and the oracle evaluation diagnostic. It is not a runtime model input and is
not included in the returned ONNX ABI.

Reference training first fits phase alone, then fine-tunes with bit BCE and a
strong circular-phase loss. A learned checkpoint is returned only when it
materially improves validation BER without a per-SNR regression. Otherwise the
zero-residual pilot-smoothing baseline is packaged explicitly.

From the training-bundle root, after all three capture jobs are ready, run:

```bash
python train_demo.py
python evaluate_demo.py
```

Then, from `reference_training/`, build the paired six-method campaign:

```bash
python build_benchmark.py
```

The generated `benchmark_recipe.yaml` is the concrete benchmark source. The
generated `benchmark_pack.yaml` compares uncompensated QPSK, raw pilot
interpolation, pilot smoothing, a decision-directed PLL, the returned learned
receiver, and oracle phase correction over paired payload, AWGN, and impairment
seeds.
