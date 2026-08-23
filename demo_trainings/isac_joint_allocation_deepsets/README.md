# Joint ISAC allocation reference training

This optional starter trains only the OFDM power-allocation block. Each runtime record contains
per-subcarrier communication gain, sensing gain, noise level, and the declared sensing weight. The
network returns a non-negative unit-sum power vector, so the transmit-power budget is exact by
construction.

The starter is label-free: it minimizes the negative scalarized utility recorded in the contract.
The paired benchmark compares equal power, communication-only water filling, a per-scene iterative
reference optimizer, and the returned ONNX policy on identical held-out scenes.

From the exported bundle, capture all three splits, run `python train_demo.py`, then
`python evaluate_demo.py`. Run `python build_benchmark.py` from `reference_training/` and execute the
printed Noema benchmark command. This is a synthetic allocation protocol, not a waveform-level
radar detector or standards-conformant ISAC link.
