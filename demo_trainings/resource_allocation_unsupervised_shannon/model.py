from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def simplex_projection(scores: torch.Tensor) -> torch.Tensor:
    """Project each score row onto {x >= 0, sum(x) = 1}."""
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [batch, subcarrier], got {tuple(scores.shape)}")
    sorted_scores, _ = torch.sort(scores, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_scores, dim=1) - 1.0
    indices = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype).reshape(1, -1)
    active = sorted_scores - cumulative / indices > 0.0
    rho = torch.sum(active, dim=1, keepdim=True).clamp(min=1) - 1
    theta = torch.gather(cumulative, 1, rho) / (rho.to(scores.dtype) + 1.0)
    projected = torch.clamp(scores - theta, min=0.0)
    return projected / torch.clamp(torch.sum(projected, dim=1, keepdim=True), min=1e-12)


class DeepSetPowerAllocator(torch.nn.Module):
    """Permutation-equivariant CSI-to-power policy with exact instantaneous feasibility."""

    def __init__(
        self,
        hidden_dim: int = 64,
        feature_mean: float = 0.0,
        feature_scale: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.eps = float(eps)
        self.phi_0 = torch.nn.Linear(1, hidden_dim)
        self.phi_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.rho_0 = torch.nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.rho_out = torch.nn.Linear(hidden_dim, 1)
        self.register_buffer("feature_mean", torch.tensor(float(feature_mean), dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(max(float(feature_scale), 1e-6), dtype=torch.float32))

    def forward(
        self,
        channel_gain: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> torch.Tensor:
        scores, budget = self.allocation_scores(
            channel_gain,
            noise_variance,
            average_power_budget,
        )
        fractions = simplex_projection(scores)
        total_power = budget * float(scores.shape[1])
        return fractions * total_power

    def allocation_scores(
        self,
        channel_gain: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gains = channel_gain.float()
        if gains.ndim != 2:
            raise ValueError(f"channel_gain must have shape [batch, subcarrier], got {tuple(gains.shape)}")
        noise = noise_variance.float().reshape(-1, 1)
        budget = average_power_budget.float().reshape(-1, 1)
        if noise.shape[0] == 1 and gains.shape[0] != 1:
            noise = noise.expand(gains.shape[0], 1)
        if budget.shape[0] == 1 and gains.shape[0] != 1:
            budget = budget.expand(gains.shape[0], 1)
        if noise.shape[0] != gains.shape[0] or budget.shape[0] != gains.shape[0]:
            raise ValueError("noise and power-budget batches must align with channel_gain")
        return self._score_network(gains, noise, budget), budget

    def _score_network(
        self,
        gains: torch.Tensor,
        noise: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        """Architecture body after the public slot's shape normalization."""
        raw_feature = torch.log(torch.clamp(gains * budget / torch.clamp(noise, min=self.eps), min=self.eps))
        carrier_input = ((raw_feature - self.feature_mean) / self.feature_scale).unsqueeze(-1)
        embedding = torch.relu(self.phi_0(carrier_input))
        embedding = torch.relu(self.phi_1(embedding))
        context = torch.mean(embedding, dim=1, keepdim=True).expand_as(embedding)
        score_input = torch.cat([embedding, context, carrier_input], dim=-1)
        scores = self.rho_out(torch.relu(self.rho_0(score_input))).squeeze(-1)
        return scores


class _PowerPolicyOnnxView(torch.nn.Module):
    def __init__(self, model: DeepSetPowerAllocator):
        super().__init__()
        self.model = model

    def forward(
        self,
        channel_gain: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> torch.Tensor:
        return self.model._score_network(
            channel_gain.float(),
            noise_variance.float().reshape(-1, 1),
            average_power_budget.float().reshape(-1, 1),
        )


def export_onnx_policy(
    model: DeepSetPowerAllocator,
    path: Path,
    *,
    subcarrier_count: int,
) -> str:
    """Export executable policy scores; Noema owns the deployment projection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(subcarrier_count))
    model = model.cpu().eval()
    gains = torch.ones((2, count), dtype=torch.float32)
    noise = torch.full((2, 1), 0.2, dtype=torch.float32)
    budget = torch.ones((2, 1), dtype=torch.float32)
    torch.onnx.export(
        _PowerPolicyOnnxView(model),
        (gains, noise, budget),
        str(destination),
        input_names=["channel_gain", "noise_variance", "average_power_budget"],
        output_names=["allocation_scores"],
        dynamic_axes={
            "channel_gain": {0: "batch", 1: "subcarrier"},
            "noise_variance": {0: "batch"},
            "average_power_budget": {0: "batch"},
            "allocation_scores": {0: "batch", 1: "subcarrier"},
        },
        opset_version=17,
        dynamo=False,
    )
    return _sha256(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
