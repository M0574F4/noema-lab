# LEO-NTN Doppler and beam-handover reference training

This optional starter trains a causal predictor from a fixed history of noisy Doppler and azimuth
observations. The returned model predicts future Doppler and scores the next beam at a declared
one-second horizon. Future state and the correct next beam are offline labels only.

The paired campaign compares hold-last, linear extrapolation, the learned ONNX tracker, and a
simulation-truth oracle on identical held-out tracks and observation noise. Report Doppler MAE and
beam-handover accuracy together; neither metric alone represents the joint decision.

Capture all three splits, run `python train_demo.py` and `python evaluate_demo.py`, then run
`python build_benchmark.py` from `reference_training/`. The kinematic generator is deliberately
bounded and synthetic; it is not an orbital propagator, 3GPP NTN channel model, ephemeris test, or
handover-protocol conformance claim.
