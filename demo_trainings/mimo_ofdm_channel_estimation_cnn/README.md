# Reference MIMO-OFDM channel estimator

This optional project demonstrates one external implementation of the portable
`model.channel_estimator_adapter` contract. Noema captures sparse divided pilot
observations, their mask, the operation-owned interpolated LS estimate, noise
variance, and simulated channel truth. The trainer fits a compact
noise-conditioned residual estimator with frequency- and delay-domain
branches. It trains across mixed 3GPP TDL-A/C/E profiles, selects a checkpoint
using per-SNR and aggregate validation NMSE, exports ONNX, and returns a
schema-v2 trained artifact. Zero-initialized residual heads supply a safe LS
fallback.

Run `train.py`, then `evaluate.py`, only after the bundle's train, validation,
and held-out test captures are complete. Run `build_benchmark.py` to create the
paired LS/fixed-prior-LMMSE/learned SNR-and-profile campaign.
