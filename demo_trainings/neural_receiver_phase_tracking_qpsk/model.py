from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ReceiverCandidate:
    id: str
    architecture: str
    hidden_dim: int
    dilations: tuple[int, ...]
    kernel_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
        }


class _TemporalResidualBlock(torch.nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(
                width,
                width,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=width,
            ),
            torch.nn.SiLU(),
            torch.nn.Conv1d(width, width, 1),
        )
        self.activation = torch.nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.network(value))


class PilotSmootherResidualTcnReceiver(torch.nn.Module):
    """Predict a residual correction around the deterministic pilot smoother.

    ``receiver_features_v3[..., :2]`` is I/Q after the public pilot smoother.
    The network returns only a full-circle residual phase.  Bit scores are
    always derived outside the model by applying that phase to the first two
    channels, so the learned component cannot improve BCE merely by changing a
    positive logit gain.  A zero-initialized one-channel head makes epoch zero
    exactly reproduce the strong pilot-smoothing baseline.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 5,
    ):
        super().__init__()
        if hidden_dim < 1 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("context_tcn requires positive width and an odd kernel >= 3")
        dilation_values = tuple(int(value) for value in dilations)
        if not dilation_values or any(value < 1 for value in dilation_values):
            raise ValueError("context_tcn dilations must be positive")
        self.input_projection = torch.nn.Conv1d(11, hidden_dim, 1)
        self.blocks = torch.nn.ModuleList(
            _TemporalResidualBlock(hidden_dim, kernel_size, dilation)
            for dilation in dilation_values
        )
        self.output_projection = torch.nn.Conv1d(hidden_dim, 1, 1)
        torch.nn.init.zeros_(self.output_projection.weight)
        torch.nn.init.zeros_(self.output_projection.bias)

    def forward(self, receiver_features_v3: torch.Tensor) -> torch.Tensor:
        value = receiver_features_v3.float()
        hidden = self.input_projection(value.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        raw_phase = self.output_projection(hidden).squeeze(1)
        return torch.pi * torch.tanh(raw_phase)


def receiver_candidates(model_config: Mapping[str, Any]) -> tuple[ReceiverCandidate, ...]:
    raw_candidates = model_config.get("candidates")
    if raw_candidates is None:
        raw_candidates = [
            {
                "id": "pilot_smoother_residual_tcn_32",
                "architecture": "pilot_smoother_residual_tcn",
                "hidden_dim": 32,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "kernel_size": 5,
            },
            {
                "id": "pilot_smoother_residual_tcn_48",
                "architecture": "pilot_smoother_residual_tcn",
                "hidden_dim": 48,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "kernel_size": 5,
            },
        ]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("model.candidates must be a non-empty list")
    resolved = []
    seen = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError("model.candidates[%d] must be a mapping" % index)
        candidate_id = str(raw.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError("model candidate IDs must be non-empty and unique")
        seen.add(candidate_id)
        architecture = str(raw.get("architecture") or "").strip()
        if architecture not in {
            "pilot_smoother_residual_tcn",
            "interpolation_residual_tcn",
        }:
            raise ValueError("unsupported phase-tracking architecture %r" % architecture)
        if architecture == "interpolation_residual_tcn":
            architecture = "pilot_smoother_residual_tcn"
        hidden_dim = int(raw.get("hidden_dim") or 32)
        dilations = tuple(int(value) for value in raw.get("dilations") or [])
        kernel_size = int(raw.get("kernel_size") or 5)
        resolved.append(
            ReceiverCandidate(
                id=candidate_id,
                architecture=architecture,
                hidden_dim=hidden_dim,
                dilations=dilations,
                kernel_size=kernel_size,
            )
        )
    return tuple(resolved)


def build_receiver(candidate: ReceiverCandidate) -> torch.nn.Module:
    if candidate.architecture not in {
        "pilot_smoother_residual_tcn",
        "interpolation_residual_tcn",
    }:
        raise ValueError("unsupported receiver architecture %r" % candidate.architecture)
    return PilotSmootherResidualTcnReceiver(
        hidden_dim=candidate.hidden_dim,
        dilations=candidate.dilations,
        kernel_size=candidate.kernel_size,
    )


def export_onnx_receiver(model: torch.nn.Module, path: Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()

    example = torch.zeros((1, 257, 11), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(destination),
        input_names=["receiver_features_v3"],
        output_names=["residual_phase_rad"],
        dynamic_axes={
            "receiver_features_v3": {0: "packet", 1: "frame_symbol"},
            "residual_phase_rad": {0: "packet", 1: "frame_symbol"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return hashlib.sha256(destination.read_bytes()).hexdigest()
