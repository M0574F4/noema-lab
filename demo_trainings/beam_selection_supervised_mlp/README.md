# MISO beam-selection reference training

This optional trainer closes the `model.beamforming_adapter` loop for the
single-user flat-fading MISO scenario. The model receives only the realized
complex channel vector and selects one beam from the same finite DFT codebook
used by the exhaustive reference.

The class label is generated inside the trainer by exhaustively evaluating that
codebook. No oracle label enters the deployed ONNX block. The returned block
emits a unit-norm complex beam through the operation-owned portable ABI.

From the exported bundle:

1. capture train, validation, and test in Workbench;
2. run `python train_demo.py`;
3. run `python evaluate_demo.py`;
4. from `reference_training/`, run `python build_benchmark.py`;
5. run the printed Noema benchmark command.

The generated campaign compares the learned policy with exhaustive finite-
codebook selection and perfect-CSIT MRT on identical held-out channels. MRT is
an upper bound and exhaustive search is a codebook oracle; both are labeled as
such instead of being presented as deployable peers.
