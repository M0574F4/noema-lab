from __future__ import annotations

import torch


def spectral_efficiency(
    power: torch.Tensor,
    channel_gain: torch.Tensor,
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    noise = torch.clamp(noise_variance.float().reshape(-1, 1), min=1e-12)
    return torch.mean(torch.log2(1.0 + channel_gain.float() * power.float() / noise), dim=1)


def negative_shannon_spectral_efficiency(
    power: torch.Tensor,
    channel_gain: torch.Tensor,
    noise_variance: torch.Tensor,
) -> torch.Tensor:
    """Label-free loss: maximize the native parallel-channel Shannon objective."""
    return -torch.mean(spectral_efficiency(power, channel_gain, noise_variance))


def feasibility_metrics(power: torch.Tensor, average_power_budget: torch.Tensor) -> dict[str, float]:
    expected = average_power_budget.float().reshape(-1) * float(power.shape[1])
    error = torch.abs(torch.sum(power.float(), dim=1) - expected)
    negative = torch.clamp(-power.float(), min=0.0)
    return {
        "max_power_budget_error": float(torch.max(error).detach().cpu()),
        "max_negative_power_violation": float(torch.max(negative).detach().cpu()),
    }
