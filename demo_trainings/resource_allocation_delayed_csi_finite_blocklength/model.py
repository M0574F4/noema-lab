from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def simplex_projection(scores: torch.Tensor) -> torch.Tensor:
    """Project each row onto the unit simplex exactly.

    The projection is piecewise differentiable and, unlike a softmax, can assign
    exactly zero power to a subcarrier.
    """

    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError(
            "scores must have shape [batch, subcarrier], got %s"
            % (tuple(scores.shape),)
        )
    ordered, _ = torch.sort(scores, dim=1, descending=True)
    cumulative = torch.cumsum(ordered, dim=1) - 1.0
    rank = torch.arange(
        1,
        scores.shape[1] + 1,
        dtype=scores.dtype,
        device=scores.device,
    ).reshape(1, -1)
    active = ordered - cumulative / rank > 0.0
    rho = torch.sum(active, dim=1, keepdim=True).clamp(min=1) - 1
    threshold = torch.gather(cumulative, 1, rho) / (
        rho.to(dtype=scores.dtype) + 1.0
    )
    projected = torch.clamp(scores - threshold, min=0.0)
    return projected / torch.clamp(
        torch.sum(projected, dim=1, keepdim=True),
        min=1e-12,
    )


@dataclass(frozen=True)
class AllocatorCandidate:
    id: str
    hidden_dim: int
    dilations: tuple[int, ...]
    kernel_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "architecture": "causal_csi_history_frequency_residual_cnn",
            "hidden_dim": self.hidden_dim,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
        }


class _FrequencyResidualBlock(torch.nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = torch.nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=width,
        )
        self.pointwise = torch.nn.Conv1d(width, width, 1)
        self.activation = torch.nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.activation(self.depthwise(value)))
        return self.activation(value + residual)


class CausalCsiHistoryPowerAllocator(torch.nn.Module):
    """Predict feasible power from a causal complex-CSI history.

    The front end exposes field-informed features: complex samples, log power,
    and normalized complex changes between consecutive CSI reports. Dilated
    convolutions then exploit wideband frequency correlation while a pooled
    context couples all tones through the shared power budget. The current
    channel realization is deliberately absent from the runtime interface and
    is used only to evaluate the external training objective.
    """

    def __init__(
        self,
        *,
        history_length: int = 4,
        hidden_dim: int = 48,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        feature_mean: float = 0.0,
        feature_scale: float = 1.0,
        baseline_score_scale: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        width = int(hidden_dim)
        dilation_values = tuple(int(value) for value in dilations)
        kernel = int(kernel_size)
        if width < 1:
            raise ValueError("hidden_dim must be positive")
        if not dilation_values or any(value < 1 for value in dilation_values):
            raise ValueError("dilations must contain positive integers")
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        history = int(history_length)
        if history < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history
        self.hidden_dim = width
        self.dilations = dilation_values
        self.kernel_size = kernel
        self.eps = float(eps)
        self.baseline_score_scale = float(baseline_score_scale)
        # Per history sample: log power and normalized complex correlation
        # between neighboring tones. Consecutive samples additionally
        # contribute normalized temporal complex correlation. These features
        # preserve physically useful frequency/phase evolution while remaining
        # invariant to an arbitrary common carrier phase. Two scalar channels
        # carry the public noise and average-power controls.
        input_channels = 3 * history + 2 * max(history - 1, 0) + 2
        self.input_projection = torch.nn.Conv1d(input_channels, width, 1)
        self.blocks = torch.nn.ModuleList(
            _FrequencyResidualBlock(width, kernel, dilation)
            for dilation in dilation_values
        )
        self.context_projection = torch.nn.Conv1d(2 * width, width, 1)
        self.output_projection = torch.nn.Conv1d(width, 1, 1)
        # Epoch zero is exactly equal power, a stable deployable baseline.
        torch.nn.init.zeros_(self.output_projection.weight)
        torch.nn.init.zeros_(self.output_projection.bias)
        self.register_buffer(
            "feature_mean",
            torch.tensor(float(feature_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.tensor(max(float(feature_scale), 1e-6), dtype=torch.float32),
        )

    def forward(
        self,
        csi_history: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> torch.Tensor:
        scores, budget = self.allocation_scores(
            csi_history,
            noise_variance,
            average_power_budget,
        )
        fractions = simplex_projection(scores)
        return fractions * budget * float(scores.shape[1])

    def allocation_scores(
        self,
        csi_history: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._validated_history(csi_history)
        if history.shape[2] < 1:
            raise ValueError(
                "csi_history must contain at least one subcarrier"
            )
        noise = _batch_scalar(
            noise_variance, history.shape[0], "noise_variance"
        )
        budget = _batch_scalar(
            average_power_budget,
            history.shape[0],
            "average_power_budget",
        )
        if bool(torch.any(noise <= 0.0)):
            raise ValueError("noise_variance must be positive")
        if bool(torch.any(budget <= 0.0)):
            raise ValueError("average_power_budget must be positive")
        return self._score_network(history, noise, budget), budget

    def _score_network(
        self,
        csi_history: torch.Tensor,
        noise: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        # allocation_scores validates researcher/runtime calls. The ONNX view
        # invokes this tensor-only core directly so export remains free of
        # Python shape guards and supports a dynamic batch/subcarrier axis.
        history = csi_history.float()
        real = history[..., 0]
        imag = history[..., 1]
        gain = torch.clamp(real.square() + imag.square(), min=self.eps)
        log_gain = (
            torch.log(gain) - self.feature_mean
        ) / self.feature_scale
        frequency_scale = torch.sqrt(
            torch.clamp(
                gain[:, :, 1:] * gain[:, :, :-1],
                min=self.eps,
            )
        )
        frequency_real = (
            real[:, :, 1:] * real[:, :, :-1]
            + imag[:, :, 1:] * imag[:, :, :-1]
        ) / frequency_scale
        frequency_imag = (
            imag[:, :, 1:] * real[:, :, :-1]
            - real[:, :, 1:] * imag[:, :, :-1]
        ) / frequency_scale
        frequency_real = torch.cat(
            (torch.ones_like(real[:, :, :1]), frequency_real),
            dim=2,
        )
        frequency_imag = torch.cat(
            (torch.zeros_like(imag[:, :, :1]), frequency_imag),
            dim=2,
        )
        components = [
            frequency_real,
            frequency_imag,
            log_gain,
        ]
        if self.history_length > 1:
            change_scale = torch.sqrt(
                torch.clamp(
                    gain[:, 1:] * gain[:, :-1],
                    min=self.eps,
                )
            )
            components.extend(
                (
                    (
                        real[:, 1:] * real[:, :-1]
                        + imag[:, 1:] * imag[:, :-1]
                    )
                    / change_scale,
                    (
                        imag[:, 1:] * real[:, :-1]
                        - real[:, 1:] * imag[:, :-1]
                    )
                    / change_scale,
                )
            )
        log_noise = torch.log(torch.clamp(noise, min=self.eps))
        log_budget = torch.log(torch.clamp(budget, min=self.eps))
        components.extend(
            (
                log_noise.reshape(-1, 1, 1).expand(
                    -1, 1, history.shape[2]
                ),
                log_budget.reshape(-1, 1, 1).expand(
                    -1, 1, history.shape[2]
                ),
            )
        )
        features = torch.cat(
            [
                value
                if value.ndim == 3
                else value.reshape(
                    history.shape[0], -1, history.shape[2]
                )
                for value in components
            ],
            dim=1,
        )
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden)
        context = torch.mean(hidden, dim=2, keepdim=True).expand_as(hidden)
        hidden = torch.nn.functional.silu(
            self.context_projection(torch.cat((hidden, context), dim=1))
        )
        residual = self.output_projection(hidden).squeeze(1)
        newest_log_gain = log_gain[:, -1]
        return self.baseline_score_scale * newest_log_gain + residual

    def _validated_history(self, value: torch.Tensor) -> torch.Tensor:
        history = value.float()
        # Retain a convenient legacy inspection path for H=1 unit tests and
        # researcher experiments. Exported artifacts always use the exact
        # four-dimensional causal-history ABI.
        if history.ndim == 2 and self.history_length == 1:
            gain = torch.clamp(history, min=0.0)
            history = torch.stack(
                (torch.sqrt(gain), torch.zeros_like(gain)),
                dim=-1,
            ).unsqueeze(1)
        if (
            history.ndim != 4
            or history.shape[1] != self.history_length
            or history.shape[3] != 2
        ):
            raise ValueError(
                "csi_history must have shape [batch,%d,subcarrier,2], got %s"
                % (self.history_length, tuple(history.shape))
            )
        return history


class _PowerPolicyOnnxView(torch.nn.Module):
    """Expose scores; Noema applies the same exact simplex projection."""

    def __init__(self, model: CausalCsiHistoryPowerAllocator) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        csi_history: torch.Tensor,
        noise_variance: torch.Tensor,
        average_power_budget: torch.Tensor,
    ) -> torch.Tensor:
        return self.model._score_network(
            csi_history.float(),
            noise_variance.float().reshape(-1, 1),
            average_power_budget.float().reshape(-1, 1),
        )


# Keep the earlier public class name import-compatible for existing external
# scripts. Its implementation now consumes causal complex CSI history.
FrequencyResidualPowerAllocator = CausalCsiHistoryPowerAllocator


def allocator_candidates(
    model_config: Mapping[str, Any],
) -> tuple[AllocatorCandidate, ...]:
    raw = model_config.get("candidates")
    if raw is None:
        raw = [
            {
                "id": "causal_history_frequency_cnn_48",
                "hidden_dim": int(model_config.get("hidden_dim") or 48),
                "dilations": model_config.get("dilations") or [1, 2, 4, 8],
                "kernel_size": int(model_config.get("kernel_size") or 3),
            }
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("model.candidates must be a non-empty list")
    result = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("model.candidates[%d] must be a mapping" % index)
        candidate_id = str(item.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError("model candidate IDs must be non-empty and unique")
        seen.add(candidate_id)
        result.append(
            AllocatorCandidate(
                id=candidate_id,
                hidden_dim=int(item.get("hidden_dim") or 48),
                dilations=tuple(
                    int(value)
                    for value in (item.get("dilations") or [1, 2, 4, 8])
                ),
                kernel_size=int(item.get("kernel_size") or 3),
            )
        )
    return tuple(result)


def build_allocator(
    candidate: AllocatorCandidate,
    *,
    feature_mean: float,
    feature_scale: float,
    history_length: int = 4,
) -> CausalCsiHistoryPowerAllocator:
    return CausalCsiHistoryPowerAllocator(
        history_length=history_length,
        hidden_dim=candidate.hidden_dim,
        dilations=candidate.dilations,
        kernel_size=candidate.kernel_size,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def export_onnx_policy(
    model: CausalCsiHistoryPowerAllocator,
    path: Path,
    *,
    subcarrier_count: int,
    history_length: int | None = None,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    count = max(1, int(subcarrier_count))
    history = int(
        model.history_length
        if history_length is None
        else history_length
    )
    if history != model.history_length:
        raise ValueError(
            "history_length does not match the trained model"
        )
    csi_history = torch.zeros(
        (2, history, count, 2), dtype=torch.float32
    )
    csi_history[..., 0] = 1.0
    noise = torch.full((2, 1), 0.2, dtype=torch.float32)
    budget = torch.ones((2, 1), dtype=torch.float32)
    torch.onnx.export(
        _PowerPolicyOnnxView(model),
        (csi_history, noise, budget),
        str(destination),
        input_names=[
            "csi_history",
            "noise_variance",
            "average_power_budget",
        ],
        output_names=["allocation_scores"],
        dynamic_axes={
            "csi_history": {0: "batch", 2: "subcarrier"},
            "noise_variance": {0: "batch"},
            "average_power_budget": {0: "batch"},
            "allocation_scores": {0: "batch", 1: "subcarrier"},
        },
        opset_version=17,
        dynamo=False,
    )
    return _sha256(destination)


def _batch_scalar(
    value: torch.Tensor,
    batch_size: int,
    label: str,
) -> torch.Tensor:
    result = value.float().reshape(-1, 1)
    if result.shape[0] == 1 and batch_size != 1:
        result = result.expand(batch_size, 1)
    if result.shape[0] != batch_size:
        raise ValueError("%s batch does not align with csi_history" % label)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
