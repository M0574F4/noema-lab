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
- `localization_supervised_mlp`: geometry-aware residual range localization from captured anchors
  and noisy ranges, with positions used only as offline supervision.
- `aoa_estimation_covariance_mlp`: covariance-domain single-source ULA angle estimation from
  complex snapshots, compared with Bartlett and MUSIC.
- `beam_selection_supervised_mlp`: finite-DFT-codebook beam classification from captured MISO
  channels, with exhaustive-search labels derived inside the trainer.
- `isac_joint_allocation_deepsets`: label-free joint communication/sensing OFDM allocation under an
  exact sum-power constraint and an explicit scalarized utility.
- `near_field_range_angle_mlp`: bounded range-angle regression from coherent spherical-wave array
  observations, evaluated by both estimation error and focusing gain.
- `leo_ntn_tracking_mlp`: causal future-Doppler regression and next-beam classification from a
  fixed noisy observation history.

These six compact adapter projects share the small capture, train, evaluate, and post-training benchmark
harness in `_portable_ai_phy_adapter_common`; each task still owns its model, ABI, objective, and
comparison roles explicitly.

These are demonstration implementations, not template types or product-level training choices.
Researchers may ignore them and use any model, loss, optimizer, and trainer that satisfy the same
data and returned-artifact interfaces.
