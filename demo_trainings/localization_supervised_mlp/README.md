# Range-localization reference training

This optional trainer closes the `model.localization_adapter` loop for the
four-anchor two-dimensional localization scenario. The model receives anchor
coordinates and noisy ranges. True positions are captured separately for
offline supervision and are never runtime inputs.

The reference model keeps linear trilateration in the forward path and learns a
small bounded residual. This geometry-aware design has a clear classical
fallback and avoids asking an unconstrained network to rediscover the entire
ranging geometry.

From the exported bundle:

1. capture train, validation, and test in Workbench;
2. run `python train_demo.py`;
3. run `python evaluate_demo.py`;
4. from `reference_training/`, run `python build_benchmark.py`;
5. run the printed Noema benchmark command.

The generated campaign compares linear trilateration, centroid-regularized
trilateration, and the learned artifact on paired held-out geometries, range
noise, and SNR values. Train and validation records select the model; the test
capture remains sealed until evaluation.
