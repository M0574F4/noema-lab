# Automatic modulation-recognition demonstration trainer

This project reproduces the model-training stage of the AMC demonstration. The Noema training
contract fixes the I/O ABI and class order; it does not require this temporal CNN or this optimizer.

`training_plan.yaml` separately records that this example captures both I/Q frames and class labels
and trains with cross-entropy. The reusable AMC recipe itself does not prescribe that method.

1. Run the three capture jobs listed in `project_manifest.yaml` from Noema. They create separate train, validation, and held-out test datasets with `iq_frames` and `modulation_labels` taps.
2. Create an environment and install `requirements.txt`.
3. Review `train_config.yaml`, especially the capture directories and seeds.
4. Run `python train.py`. It starts from the blind differential-cumulant prior, trains a small regularized residual classifier, and selects the checkpoint by validation cross-entropy. The untrained prior is retained as an epoch-zero fallback.
5. Run `python evaluate.py` to score only the held-out test capture.
6. Run `python build_benchmark.py`. It creates a paired blind-cumulant, learned, and oracle-synchronized SNR campaign using the same held-out frames, carrier impairments, and AWGN seeds for every method. It rejects capture-seed reuse and defaults to 1,536 balanced frames per evaluation run.
7. Validate and run the printed benchmark command. For an individual run, you can also set the canonical recipe classifier to `learned_artifact` and select the returned manifest.

The built-in dataset is intentionally controlled: balanced fixed-length BPSK, QPSK, and 16-QAM frames with unknown per-frame carrier phase, residual frequency offset, and AWGN. It is a clean blind-receiver demonstration boundary, not a claim of RadioML benchmark parity.
