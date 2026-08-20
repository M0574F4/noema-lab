# Noema demonstration training projects

This directory contains model, loss, and trainer implementations used by the checked-in Noema
demonstrations. They are not part of Workbench and are not a general Noema training subsystem.
Workbench exports a neutral interface and capture contract. Each project keeps its demo-specific
capture selections and objective in its own `training_plan.yaml`; `prepare_example.py` then binds
the checked-in model and trainer to that contract. Only the resulting frozen,
provenance-recorded checkpoint returns to Noema for benchmark evaluation.

Available projects:

- `deepjscc_image_reconstruction`: jointly trained image-to-symbol encoder and symbol-to-image decoder
  around recipe-derived pure-PyTorch power normalization and either AWGN or blind slow Rayleigh
  fading. The template owns the reference CNN architecture and returns an atomic pair of portable
  ONNX encoder/decoder bindings.
- `resource_allocation_unsupervised_shannon`: label-free CSI-conditioned OFDM power allocation with
  a permutation-equivariant Deep Sets policy, negative Shannon spectral-efficiency loss, and exact
  simplex feasibility.
- `resource_allocation_delayed_csi_finite_blocklength`: reliability-aware allocation from delayed,
  noisy CSI histories using a finite-blocklength goodput objective and validation-only acceptance
  against deployable causal baselines.
- `csi_feedback_autoencoder`: paired quantization-aware CSI encoder/decoder with an NMSE and
  MRT-spectral-efficiency objective under a fixed feedback-bit contract.
- `neural_receiver_supervised_qpsk`: supervised QPSK receiver mapping noisy complex symbols to
  bit LLRs with BCE loss and a portable single-block return artifact.
- `neural_receiver_phase_tracking_qpsk`: temporal residual phase tracking from observable packet
  context, returned through a portable receiver artifact without exposing the oracle phase trace.
- `modulation_recognition_supervised_cnn`: blind BPSK/QPSK/16-QAM recognition under carrier phase,
  frequency-offset, and AWGN impairments with captured labels used only for supervision.
- `mimo_ofdm_channel_estimation_cnn`: supervised residual frequency-domain refinement of
  operation-owned LS estimates for a portable 2×2 MIMO-OFDM channel-estimator artifact.

These are demonstration implementations, not template types or product-level training choices.
Researchers may ignore them and use any model, loss, optimizer, and trainer that satisfy the same
data and returned-artifact interfaces.
