from __future__ import annotations

import math

import torch


def finite_blocklength_state_metrics(
    power: torch.Tensor,
    actual_channel_gain: torch.Tensor,
    noise_variance: torch.Tensor,
    *,
    blocklength_channel_uses: int,
    target_rate_bps_hz: float,
    include_third_order_term: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel-channel normal approximation for one packet per state row."""

    if power.shape != actual_channel_gain.shape or power.ndim != 2:
        raise ValueError(
            "power and actual_channel_gain must share [batch,subcarrier] shape"
        )
    blocklength = int(blocklength_channel_uses)
    rate = float(target_rate_bps_hz)
    if blocklength < 16:
        raise ValueError("blocklength_channel_uses must be >= 16")
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("target_rate_bps_hz must be finite and nonnegative")
    noise = noise_variance.float().reshape(-1, 1)
    if noise.shape[0] == 1 and power.shape[0] != 1:
        noise = noise.expand(power.shape[0], 1)
    if noise.shape[0] != power.shape[0] or bool(torch.any(noise <= 0.0)):
        raise ValueError("noise_variance must be positive and batch-aligned")

    snr = actual_channel_gain.float() * power.float() / noise
    capacity = torch.mean(torch.log2(1.0 + snr), dim=1)
    log2e = math.log2(math.e)
    dispersion = torch.mean(
        (1.0 - torch.pow(1.0 + snr, -2.0)) * (log2e**2),
        dim=1,
    )
    correction = (
        math.log2(float(blocklength)) / (2.0 * float(blocklength))
        if include_third_order_term
        else 0.0
    )
    z = (
        (capacity - rate + correction)
        * math.sqrt(float(blocklength))
        / torch.sqrt(torch.clamp(dispersion, min=1e-12))
    )
    predicted_bler = 0.5 * torch.special.erfc(z / math.sqrt(2.0))
    goodput = rate * (1.0 - predicted_bler)
    return goodput, predicted_bler, capacity, dispersion


def negative_expected_finite_blocklength_goodput(
    power: torch.Tensor,
    actual_channel_gain: torch.Tensor,
    noise_variance: torch.Tensor,
    *,
    blocklength_channel_uses: int,
    target_rate_bps_hz: float,
    include_third_order_term: bool = True,
) -> torch.Tensor:
    """No-allocation-label loss evaluated on the aligned later channel state."""

    goodput, _, _, _ = finite_blocklength_state_metrics(
        power,
        actual_channel_gain,
        noise_variance,
        blocklength_channel_uses=blocklength_channel_uses,
        target_rate_bps_hz=target_rate_bps_hz,
        include_third_order_term=include_third_order_term,
    )
    return -torch.mean(goodput)


def feasibility_metrics(
    power: torch.Tensor,
    average_power_budget: torch.Tensor,
) -> dict[str, float]:
    expected = average_power_budget.float().reshape(-1) * float(power.shape[1])
    error = torch.abs(torch.sum(power.float(), dim=1) - expected)
    negative = torch.clamp(-power.float(), min=0.0)
    return {
        "max_power_budget_error": float(torch.max(error).detach().cpu()),
        "max_negative_power_violation": float(
            torch.max(negative).detach().cpu()
        ),
    }
