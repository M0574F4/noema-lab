# QPSK receiver-calibration reference training

This optional demonstration trainer closes one
`demodulation.neural_receiver_adapter` loop. The recipe applies AWGN followed
by a fixed affine receiver front-end impairment. The model receives only the
impaired `[I,Q]` sample and returns two bit logits; transmitted bits are offline
supervision, not runtime inputs.

The example fits an affine calibration model because the simulated front end is
itself a fixed invertible affine transform. It initializes the two bit
boundaries from captured I/Q and bit pairs, then converges the logistic loss
with full-batch L-BFGS. This keeps the learned boundary class physically matched
to the task instead of allowing an unnecessarily flexible MLP to wiggle between
finite training samples.

The trainer exports ONNX and writes a hash-pinned schema-v2 artifact. Packet
fingerprints enforce disjoint train, validation, and held-out test captures.

From the exported bundle:

1. capture train, validation, and test in Workbench;
2. run `python train_demo.py`;
3. run `python evaluate_demo.py`;
4. from `reference_training/`, run `python build_benchmark.py`;
5. run the printed Noema benchmark commands.

The generated campaign pairs bits and AWGN seeds across three methods:
uncompensated QPSK, a simulation-only calibrated I/Q oracle, and the learned
artifact. A useful result moves the learned decision boundaries toward the
oracle and removes much of the uncompensated detector’s high-SNR error floor.

The oracle knows the simulated front-end transform. The learned artifact does
not; it infers calibration from captured I/Q and bit pairs. This distinction is
recorded in the evaluation evidence.
