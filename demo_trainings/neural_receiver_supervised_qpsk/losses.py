from __future__ import annotations

import torch


def bit_bce(bit_llr: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    """BCE under the runtime convention: positive LLR predicts bit zero."""

    return torch.nn.functional.binary_cross_entropy_with_logits(
        -bit_llr.float(), target_bits.float()
    )


def bit_error_rate(bit_llr: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    decisions = bit_llr < 0.0
    return torch.mean((decisions != target_bits.bool()).float())
