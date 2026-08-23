# MISO beam-selection reference training

This optional trainer closes the `model.beamforming_adapter` loop for the
single-user clustered-ULA MISO scenario. Training learns eight constant-modulus
beam directions from captured channel vectors. At runtime the returned block
exhaustively selects the best member of that learned codebook.

The loss directly maximizes normalized channel gain; there is no captured label
or hidden oracle input. The returned block emits a unit-norm complex beam through
the operation-owned portable ABI.

From the exported bundle:

1. capture train, validation, and test in Workbench;
2. run `python train_demo.py`;
3. run `python evaluate_demo.py`;
4. from `reference_training/`, run `python build_benchmark.py`;
5. run the printed Noema benchmark command.

The generated campaign compares an eight-beam learned codebook with an
equal-size fixed DFT codebook and perfect-CSIT MRT on identical held-out
channels. MRT remains an upper bound; the fixed DFT sweep is exhaustive only
within its declared eight-beam codebook.
