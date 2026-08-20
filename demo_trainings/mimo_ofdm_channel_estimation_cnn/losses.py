from __future__ import annotations

import torch


def normalized_complex_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Normalize each independent channel realization before averaging so a
    # high-power realization or SNR cell cannot dominate the update.
    dimensions = tuple(range(1, prediction.ndim))
    error = torch.sum((prediction - target) ** 2, dim=dimensions)
    power = torch.sum(target**2, dim=dimensions)
    return torch.mean(error / torch.clamp(power, min=1e-12))


def nmse_db(prediction: torch.Tensor, target: torch.Tensor) -> float:
    value = normalized_complex_mse(prediction, target)
    return float(10.0 * torch.log10(torch.clamp(value, min=1e-12)).detach().cpu())
