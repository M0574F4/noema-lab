# CSI-feedback demonstration trainer

This directory contains the training project used by the CSI-feedback demonstration. The
`demo_trainings/prepare_example.py` helper binds it beside a neutral exported contract. The contract
fixes tensor interfaces, paired atomic
artifact return, captured CSI provenance, and the feedback-link budget. It does
not prescribe this model, loss, optimizer, or trainer.

`training_plan.yaml` separately records this demonstration's true-CSI capture, split, and NMSE
objective. Those choices are not properties of the reusable limited-feedback system recipe.

The v3 example fits a KLT/PCA codec on the declared training split, installs it
as the exact epoch-zero encoder/decoder, and trains a small angular-delay
nonlinear residual around it. The KLT path is initially frozen and later moves
at one tenth of the residual path's learning rate. Its forward pass uses the
recipe's exact uniform quantizer and bit depth; only the backward pass uses a
straight-through estimator.

Training first minimizes quantized NMSE, then combines normalized CSI
reconstruction error, per-subcarrier beam-direction agreement, MRT
spectral-efficiency retention over an SNR grid, and a small
quantization-consistency term. Checkpoints are selected by validation mean
spectral-efficiency retention first and NMSE second. The untouched KLT prior is
always a candidate, so unsuccessful fine-tuning cannot silently replace it with
a worse validation checkpoint. Set
`objective.loss: csi.normalized_reconstruction_mse` for a controlled NMSE-only
ablation; that mode reverses checkpoint priority. The test split is never opened
during training or model selection.

From the exported `reference_training` directory, after all three Noema capture
jobs finish:

```bash
python -m pip install -r requirements.txt
python train.py
python evaluate.py
```

`train.py` uses AdamW, learning-rate warm-up followed by cosine decay, staged
validation early stopping, gradient clipping, and quantization-aware fine-tuning. It
returns one schema-v2 atomic artifact containing ONNX `encoder` and `decoder`
components. Select that single artifact in either paired recipe slot; Noema
applies both bindings together. `evaluate.py` uses the held-out test capture and
verifies that train, validation, and test CSI records are disjoint. Before
packaging the artifact, the trainer also records the SHA-256 of every captured
split shard in `data_contract.yaml`; this makes later benchmark evidence
verifiable without exposing test tensors to checkpoint selection.

After training and evaluation, `build_benchmark.py` creates the paired
truncated/KLT/learned/perfect-CSIT comparison:

```bash
python build_benchmark.py
noema benchmark validate benchmark_pack.yaml
noema benchmark run benchmark_pack.yaml
```

For a practical rerun, capture 12,288 independent CSI realizations with an
80/10/10 train/validation/test split. If you change the training data, refit any
data-derived KLT/PCA comparison on exactly the same training split.
