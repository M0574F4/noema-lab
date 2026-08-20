from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import torch


def _validate_csi_pair(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reconstruction.shape != target.shape:
        raise ValueError(
            "CSI reconstruction and target shapes must match; got %s and %s"
            % (tuple(reconstruction.shape), tuple(target.shape))
        )
    if target.ndim != 4 or int(target.shape[1]) != 2:
        raise ValueError(
            "CSI tensors must have shape [batch,2,tx_antenna,subcarrier], got %s"
            % (tuple(target.shape),)
        )
    return reconstruction.float(), target.float()


def _per_sample_nmse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    reconstruction, target = _validate_csi_pair(reconstruction, target)
    error_energy = (reconstruction - target).square().flatten(1).sum(1)
    reference_energy = target.square().flatten(1).sum(1)
    return error_energy / reference_energy.clamp_min(float(eps))


def normalized_reconstruction_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean per-sample CSI NMSE, kept as the pure-NMSE ablation objective."""

    return _per_sample_nmse(reconstruction, target, eps=eps).mean()


def phase_invariant_cosine_similarity(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean complex cosine magnitude, invariant to one global phase per sample."""

    true_h = _as_complex(target)
    reconstructed_h = _as_complex(reconstruction)
    true_flat = true_h.flatten(1)
    reconstructed_flat = reconstructed_h.flatten(1)
    inner = (true_flat.conj() * reconstructed_flat).sum(1).abs()
    denominator = torch.linalg.vector_norm(true_flat, dim=1) * torch.linalg.vector_norm(
        reconstructed_flat, dim=1
    )
    return (inner / denominator.clamp_min(float(eps))).mean()


def subcarrier_direction_cosine_similarity(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean beam-direction agreement over samples and OFDM subcarriers."""

    true_h = _as_complex(target)
    reconstructed_h = _as_complex(reconstruction)
    inner = (true_h.conj() * reconstructed_h).sum(dim=1).abs()
    denominator = torch.linalg.vector_norm(
        true_h, dim=1
    ) * torch.linalg.vector_norm(reconstructed_h, dim=1)
    return (inner / denominator.clamp_min(float(eps))).mean()


def _mrt_rates(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    downlink_snr_db: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample achieved and perfect-CSI MRT spectral efficiencies."""

    true_h = _as_complex(target)
    reconstructed_h = _as_complex(reconstruction)
    norm = reconstructed_h.abs().square().sum(dim=1, keepdim=True).sqrt()
    weights = reconstructed_h / norm.clamp_min(float(eps))

    # This branch matches Noema's benchmark fallback for an all-zero estimate.
    # torch.where keeps the non-degenerate path differentiable and vectorized.
    fallback = torch.zeros_like(weights)
    fallback[:, 0, :] = 1.0 + 0.0j
    weights = torch.where((norm <= float(eps)).expand_as(weights), fallback, weights)

    effective_gain = (true_h * weights.conj()).sum(dim=1).abs().square()
    perfect_gain = true_h.abs().square().sum(dim=1)
    snr_linear = float(10.0 ** (float(downlink_snr_db) / 10.0))
    achieved_rate = torch.log2(1.0 + snr_linear * effective_gain).mean(dim=1)
    perfect_rate = torch.log2(1.0 + snr_linear * perfect_gain).mean(dim=1)
    return achieved_rate, perfect_rate


def csi_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    downlink_snr_db: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Differentiable equivalents of Noema's CSI-feedback benchmark metrics."""

    per_sample_nmse = _per_sample_nmse(reconstruction, target, eps=eps)
    cosine = phase_invariant_cosine_similarity(reconstruction, target, eps=eps)
    subcarrier_cosine = subcarrier_direction_cosine_similarity(
        reconstruction, target, eps=eps
    )
    achieved_rate, perfect_rate = _mrt_rates(
        reconstruction,
        target,
        downlink_snr_db=downlink_snr_db,
        eps=eps,
    )
    achieved_mean = achieved_rate.mean()
    perfect_mean = perfect_rate.mean()
    retention = achieved_mean / perfect_mean.clamp_min(float(eps))
    nmse = per_sample_nmse.mean()
    return {
        "nmse": nmse,
        "nmse_db": 10.0 * torch.log10(nmse.clamp_min(1e-15)),
        "phase_invariant_cosine": cosine,
        "subcarrier_direction_cosine": subcarrier_cosine,
        "spectral_efficiency_bps_hz": achieved_mean,
        "perfect_csi_spectral_efficiency_bps_hz": perfect_mean,
        "spectral_efficiency_retention": retention,
        "spectral_efficiency_loss_bps_hz": perfect_mean - achieved_mean,
    }


def csi_metrics_over_snrs(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    snr_db_values: Sequence[float],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Aggregate CSI and MRT metrics with every configured SNR weighted equally."""

    snrs = [float(value) for value in snr_db_values]
    if not snrs:
        raise ValueError("snr_db_values must contain at least one SNR")

    per_snr = [
        csi_metrics(
            reconstruction,
            target,
            downlink_snr_db=snr_db,
            eps=eps,
        )
        for snr_db in snrs
    ]
    achieved = torch.stack([item["spectral_efficiency_bps_hz"] for item in per_snr])
    perfect = torch.stack(
        [item["perfect_csi_spectral_efficiency_bps_hz"] for item in per_snr]
    )
    retention = torch.stack(
        [item["spectral_efficiency_retention"] for item in per_snr]
    )
    first = per_snr[0]
    mean_retention = retention.mean()
    return {
        "nmse": first["nmse"],
        "nmse_db": first["nmse_db"],
        "phase_invariant_cosine": first["phase_invariant_cosine"],
        "subcarrier_direction_cosine": first["subcarrier_direction_cosine"],
        "spectral_efficiency_bps_hz": achieved.mean(),
        "perfect_csi_spectral_efficiency_bps_hz": perfect.mean(),
        "spectral_efficiency_retention": mean_retention,
        "mean_spectral_efficiency_retention": mean_retention,
        "spectral_efficiency_loss_bps_hz": (perfect - achieved).mean(),
    }


def hybrid_csi_objective(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    feedback_code: torch.Tensor | None,
    received_code: torch.Tensor | None,
    *,
    snr_db_values: Sequence[float],
    nmse_weight: float,
    direction_weight: float,
    rate_weight: float,
    quantization_weight: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Label-free task-aware objective for the CSI-feedback demonstration project.

    The objective combines reconstruction fidelity, per-subcarrier beam direction,
    and reconstructed-CSI MRT rate retention. Setting all weights except
    ``nmse_weight`` to zero recovers the pure-NMSE training ablation. The
    quantization term compares the encoder output to a detached quantized code,
    so it pulls latents toward representable levels even when transport uses an
    identity-gradient straight-through estimator.
    """

    weights = {
        "nmse_weight": float(nmse_weight),
        "direction_weight": float(direction_weight),
        "rate_weight": float(rate_weight),
        "quantization_weight": float(quantization_weight),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("CSI objective weights must be non-negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("At least one CSI objective weight must be positive")

    metrics = csi_metrics_over_snrs(
        reconstruction,
        target,
        snr_db_values=snr_db_values,
        eps=eps,
    )
    nmse_loss = metrics["nmse"]
    direction_loss = (1.0 - metrics["subcarrier_direction_cosine"]).clamp_min(0.0)
    rate_loss = (1.0 - metrics["mean_spectral_efficiency_retention"]).clamp_min(0.0)

    if feedback_code is None or received_code is None:
        if weights["quantization_weight"] > 0.0:
            raise ValueError(
                "feedback_code and received_code are required when quantization_weight > 0"
            )
        quantization_loss = reconstruction.new_zeros(())
    else:
        if feedback_code.shape != received_code.shape:
            raise ValueError(
                "Feedback and received code shapes must match; got %s and %s"
                % (tuple(feedback_code.shape), tuple(received_code.shape))
            )
        quantization_loss = (
            feedback_code.float() - received_code.float().detach()
        ).square().mean()

    loss = (
        weights["nmse_weight"] * nmse_loss
        + weights["direction_weight"] * direction_loss
        + weights["rate_weight"] * rate_loss
        + weights["quantization_weight"] * quantization_loss
    )
    return {
        "loss": loss,
        "nmse_loss": nmse_loss,
        "direction_loss": direction_loss,
        "mrt_rate_loss": rate_loss,
        "quantization_consistency_loss": quantization_loss,
        **metrics,
    }


def _as_complex(csi_ri: torch.Tensor) -> torch.Tensor:
    if csi_ri.ndim != 4 or int(csi_ri.shape[1]) != 2:
        raise ValueError(
            "CSI tensor must have shape [batch,2,tx_antenna,subcarrier], got %s"
            % (tuple(csi_ri.shape),)
        )
    values = csi_ri.float()
    return torch.complex(values[:, 0], values[:, 1])
