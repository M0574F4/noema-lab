from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ReceiverCandidate:
    """Resolved, serializable architecture choice for validation selection."""

    id: str
    architecture: str
    hidden_dims: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "architecture": self.architecture,
        }
        if self.hidden_dims:
            payload["hidden_dims"] = list(self.hidden_dims)
        return payload


class AffineNeuralReceiver(torch.nn.Module):
    """Learned affine I/Q calibration, matched to the demo's fixed front end."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.network.weight.copy_(torch.eye(2, dtype=torch.float32))
            self.network.bias.zero_()

    def forward(self, rx_symbols_ri: torch.Tensor) -> torch.Tensor:
        return self.network(rx_symbols_ri.float())


class ReferenceNeuralReceiver(torch.nn.Module):
    """Small example only; the exported slot does not prescribe this architecture."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        widths = tuple(
            int(value)
            for value in (
                hidden_dims if hidden_dims is not None else (hidden_dim, hidden_dim)
            )
        )
        if not widths or any(value < 1 for value in widths):
            raise ValueError("symbol_mlp hidden_dims must contain positive integers")
        layers: list[torch.nn.Module] = []
        input_dim = 2
        for width in widths:
            layers.extend((torch.nn.Linear(input_dim, width), torch.nn.SiLU()))
            input_dim = width
        layers.append(torch.nn.Linear(input_dim, 2))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, rx_symbols_ri: torch.Tensor) -> torch.Tensor:
        return self.network(rx_symbols_ri.float())


def receiver_candidates(model_config: Mapping[str, Any]) -> tuple[ReceiverCandidate, ...]:
    """Resolve the configured search set while preserving old bundle compatibility."""

    raw_candidates = model_config.get("candidates")
    if raw_candidates is None:
        hidden_dim = int(model_config.get("hidden_dim", 32))
        raw_candidates = [
            {"id": "affine", "architecture": "affine"},
            {
                "id": "symbol_mlp_%dx%d" % (hidden_dim, hidden_dim),
                "architecture": "symbol_mlp",
                "hidden_dims": [hidden_dim, hidden_dim],
            },
        ]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("model.candidates must be a non-empty list")

    resolved = []
    seen_ids = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError("model.candidates[%d] must be a mapping" % index)
        candidate_id = str(raw.get("id") or "").strip()
        if not candidate_id:
            raise ValueError("model.candidates[%d].id is required" % index)
        if candidate_id in seen_ids:
            raise ValueError("model candidate id %r is duplicated" % candidate_id)
        seen_ids.add(candidate_id)
        architecture = str(raw.get("architecture") or "").strip()
        if architecture == "affine":
            hidden_dims: tuple[int, ...] = ()
        elif architecture == "symbol_mlp":
            configured_widths = raw.get("hidden_dims")
            if configured_widths is None:
                configured_widths = [int(model_config.get("hidden_dim", 32))] * 2
            if not isinstance(configured_widths, (list, tuple)):
                raise ValueError(
                    "model candidate %r hidden_dims must be a list" % candidate_id
                )
            hidden_dims = tuple(int(value) for value in configured_widths)
            if not hidden_dims or any(value < 1 for value in hidden_dims):
                raise ValueError(
                    "model candidate %r hidden_dims must contain positive integers"
                    % candidate_id
                )
        else:
            raise ValueError(
                "model candidate %r has unsupported architecture %r"
                % (candidate_id, architecture)
            )
        resolved.append(
            ReceiverCandidate(
                id=candidate_id,
                architecture=architecture,
                hidden_dims=hidden_dims,
            )
        )
    return tuple(resolved)


def build_receiver(candidate: ReceiverCandidate) -> torch.nn.Module:
    if candidate.architecture == "affine":
        return AffineNeuralReceiver()
    if candidate.architecture == "symbol_mlp":
        return ReferenceNeuralReceiver(hidden_dims=candidate.hidden_dims)
    raise ValueError("unsupported receiver architecture %r" % candidate.architecture)


def export_onnx_receiver(
    model: torch.nn.Module,
    path: Path,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((16, 2), dtype=torch.float32)
    symbol_count = torch.export.Dim("symbol", min=1)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["rx_symbols_ri"],
        output_names=["bit_llr"],
        dynamic_shapes={
            "rx_symbols_ri": {0: symbol_count},
        },
        opset_version=18,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
