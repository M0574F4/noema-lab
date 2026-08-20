from __future__ import annotations

import torch


def bit_scores_from_residual(
    receiver_features_v3: torch.Tensor,
    residual_phase: torch.Tensor,
) -> torch.Tensor:
    """Rotate the strong baseline I/Q by the learned residual phase.

    The returned scores follow Noema's positive-for-bit-zero convention.  No
    learned temperature or gain is available in this path.
    """

    if receiver_features_v3.ndim != 3 or receiver_features_v3.shape[-1] < 2:
        raise ValueError("receiver_features_v3 must have shape [packet, symbol, feature]")
    if residual_phase.shape != receiver_features_v3.shape[:2]:
        raise ValueError("residual phase must match packet and symbol dimensions")
    baseline = receiver_features_v3[..., :2].float()
    cosine = torch.cos(residual_phase)
    sine = torch.sin(residual_phase)
    corrected_real = baseline[..., 0] * cosine + baseline[..., 1] * sine
    corrected_imag = baseline[..., 1] * cosine - baseline[..., 0] * sine
    return torch.stack((corrected_real, corrected_imag), dim=-1)


def _packet_weight_grid(
    reference: torch.Tensor,
    packet_weights: torch.Tensor | None,
) -> torch.Tensor:
    if packet_weights is None:
        return torch.ones(reference.shape[:2], dtype=reference.dtype, device=reference.device)
    weights = packet_weights.to(dtype=reference.dtype, device=reference.device)
    if weights.ndim != 1 or weights.shape[0] != reference.shape[0]:
        raise ValueError("packet_weights must have shape [packet]")
    return weights[:, None].expand(reference.shape[0], reference.shape[1])


def masked_bit_bce(
    bit_llr: torch.Tensor,
    target_bits: torch.Tensor,
    data_mask: torch.Tensor,
    packet_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy over data positions only.

    Noema's LLR convention is positive for bit zero. PyTorch BCE logits are
    positive for target one, hence the sign inversion below.
    """

    if bit_llr.shape != target_bits.shape:
        raise ValueError(
            "bit_llr and target_bits must have the same shape; got %s and %s"
            % (tuple(bit_llr.shape), tuple(target_bits.shape))
        )
    if data_mask.shape != bit_llr.shape[:-1]:
        raise ValueError(
            "data_mask must match packet and symbol dimensions; got %s for %s"
            % (tuple(data_mask.shape), tuple(bit_llr.shape))
        )
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        -bit_llr,
        target_bits.float(),
        reduction="none",
    )
    weights = _packet_weight_grid(bit_llr, packet_weights)
    expanded = (data_mask.to(dtype=losses.dtype) * weights).unsqueeze(-1)
    denominator = torch.clamp(expanded.sum() * losses.shape[-1], min=1.0)
    return torch.sum(losses * expanded) / denominator


def masked_circular_phase_loss(
    residual_phase: torch.Tensor,
    target_phase: torch.Tensor,
    phase_mask: torch.Tensor,
    packet_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Circular residual-phase supervision used only while training.

    The target comes from simulator captures.  It is not part of the deployed
    receiver ABI and is never passed to the returned ONNX model.
    """

    if residual_phase.shape != target_phase.shape:
        raise ValueError("residual and target phase tensors must have equal shape")
    if phase_mask.shape != residual_phase.shape:
        raise ValueError("phase_mask must match residual phase shape")
    weights = phase_mask.to(dtype=residual_phase.dtype) * _packet_weight_grid(
        residual_phase.unsqueeze(-1),
        packet_weights,
    )
    denominator = torch.clamp(weights.sum(), min=1.0)
    return torch.sum(
        (1.0 - torch.cos(residual_phase - target_phase)) * weights
    ) / denominator


def phase_tracking_loss(
    receiver_features_v3: torch.Tensor,
    target_bits: torch.Tensor,
    data_mask: torch.Tensor,
    residual_phase: torch.Tensor,
    target_phase: torch.Tensor,
    phase_mask: torch.Tensor,
    *,
    phase_weight: float,
    packet_weights: torch.Tensor | None = None,
    phase_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bit_scores = bit_scores_from_residual(receiver_features_v3, residual_phase)
    bit_loss = masked_bit_bce(
        bit_scores,
        target_bits,
        data_mask,
        packet_weights,
    )
    phase_loss = masked_circular_phase_loss(
        residual_phase,
        target_phase,
        phase_mask,
        packet_weights,
    )
    total = phase_loss if phase_only else bit_loss + max(0.0, float(phase_weight)) * phase_loss
    return total, bit_loss, phase_loss, bit_scores
