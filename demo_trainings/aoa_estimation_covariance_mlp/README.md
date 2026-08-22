# Narrowband AoA-estimation reference training

This optional trainer closes the `model.aoa_estimator_adapter` loop for one
far-field narrowband source observed by a half-wavelength uniform linear array.
The model receives noisy complex snapshots; the source angle is offline
supervision and is not available to the returned runtime block.

The network operates on the normalized complex sample covariance rather than a
flattened waveform. That preserves the array-processing structure and gives a
direct, fair comparison with Bartlett and MUSIC under the same controlled
single-source protocol.

From the exported bundle:

1. capture train, validation, and test in Workbench;
2. run `python train_demo.py`;
3. run `python evaluate_demo.py`;
4. from `reference_training/`, run `python build_benchmark.py`;
5. run the printed Noema benchmark command.

The generated campaign pairs source angles, array snapshots, and noise seeds
across Bartlett, MUSIC, and the learned estimator. It does not claim multipath
or multiple-source coverage; those require separate protocols.
