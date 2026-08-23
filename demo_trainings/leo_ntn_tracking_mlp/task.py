from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "leo_ntn_tracking"
COMPONENT_ID = "tracker"
COMPONENT_ROLE = "leo_ntn_doppler_beam_tracker"
ENTRYPOINT_ID = "ntn_tracker"
OPERATION_ID = "model.leo_ntn_tracking_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "beam_handover_accuracy"
DESCRIPTION = "Causal history model jointly predicting future Doppler and the next LEO-NTN beam through a portable ABI."
BEAM_COUNT = 9
MAX_DOPPLER_HZ = 48000.0


class LeoNtnTracker(nn.Module):
    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(history_length * 3, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, BEAM_COUNT + 1))

    def forward(self, track_features: torch.Tensor) -> torch.Tensor:
        raw = self.network(track_features.reshape(track_features.shape[0], -1))
        doppler_hz = MAX_DOPPLER_HZ * torch.tanh(raw[:, :1])
        return torch.cat([doppler_hz, raw[:, 1:]], dim=1)


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[2] != 3 or features.shape[1] < 3:
        raise ValueError("LEO-NTN features must have shape [record, history>=3, 3]")
    return LeoNtnTracker(int(features.shape[1]))


def predictions(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model(features)


def training_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = model(features)
    doppler_loss = torch.mean(((output[:, 0] - targets[:, 0]) / MAX_DOPPLER_HZ) ** 2)
    beam_loss = nn.functional.cross_entropy(output[:, 1:], targets[:, 2].long())
    return doppler_loss + 0.35 * beam_loss


def metrics(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def metrics_from_predictions(output: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    del features
    selected = torch.argmax(output[:, 1:], dim=1)
    return {
        "doppler_mae_hz": float(torch.mean(torch.abs(output[:, 0] - targets[:, 0])).cpu()),
        "beam_handover_accuracy": float(torch.mean((selected == targets[:, 2].long()).float()).cpu()),
    }


def runtime_inputs(features: np.ndarray):
    return [{"name": "track_features", "dtype": "float32", "shape": ["batch", int(features.shape[1]), 3], "semantic": "causal_time_doppler_angle_history"}]


def runtime_outputs(features: np.ndarray):
    del features
    return [{"name": "decision", "dtype": "float32", "shape": ["batch", BEAM_COUNT + 1], "semantic": "future_doppler_hz_and_beam_logits"}]


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {"track_features": np.ascontiguousarray(features, dtype=np.float32)}


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(model.cpu().eval(), sample.cpu(), str(path), input_names=["track_features"], output_names=["decision"], dynamic_axes={"track_features": {0: "batch"}, "decision": {0: "batch"}}, opset_version=18, dynamo=False)


def binding_params() -> dict[str, str]:
    return {"mode": "learned_artifact", "artifact_manifest_path": "trained_artifact.yaml", "artifact_entrypoint": ENTRYPOINT_ID}


def benchmark_methods(artifact_path: str, package_sha256: str):
    return (
        ("hold_last", "Hold last observation", "baseline", {"mode": "hold_last"}),
        ("linear_extrapolation", "Linear Doppler/angle extrapolation", "baseline", {"mode": "linear_extrapolation"}),
        ("oracle_future", "True future state", "oracle", {"mode": "oracle_future"}),
        ("learned_ntn_tracker", "Learned causal tracker", "candidate", {"mode": "learned_artifact", "artifact_manifest_path": artifact_path, "artifact_entrypoint": ENTRYPOINT_ID, "artifact_package_sha256": package_sha256}),
    )
