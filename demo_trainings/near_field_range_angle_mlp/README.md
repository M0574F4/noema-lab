# Physics-informed near-field range-angle reference training

This optional starter trains the `model.near_field_estimator_adapter` boundary for a controlled
28 GHz spherical-wave scenario. It uses the half-wavelength phase-slope invariant for angle and a
compact MLP over phase-curvature features for range. Runtime input remains only a noisy,
phase-referenced coherent-pilot observation; labels are separate offline artifacts.

The paired benchmark holds array geometry, carrier frequency, target state, SNR, and noise seed
fixed while comparing far-field steering, an exhaustive polar range-angle codebook, a learned ONNX
estimator, and simulation-truth focusing as an explicitly labeled upper bound.

Capture train, validation, and held-out test from the exported bundle; run `python train_demo.py`,
`python evaluate_demo.py`, and then `python build_benchmark.py` from `reference_training/`. The
protocol does not model wideband beam squint, mutual coupling, blockage, hardware calibration, or
multi-user interference.
