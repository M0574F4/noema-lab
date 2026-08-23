from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "isac_joint_allocation"
COMPONENT_ID = "allocator"
COMPONENT_ROLE = "joint_isac_ofdm_power_allocator"
ENTRYPOINT_ID = "isac_allocator"
OPERATION_ID = "model.isac_ofdm_allocator_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "scalarized_utility"
DESCRIPTION = (
    "Permutation-equivariant OFDM power allocator trained without oracle labels "
    "against the declared communication/sensing utility and returned through a portable ABI."
)
SUPERVISED_LABELS_USED = False


class IsacAllocator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(6, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = features
        pooled = torch.mean(features[..., :2], dim=1, keepdim=True).expand(-1, features.shape[1], -1)
        logits = self.score(torch.cat([local, pooled], dim=-1)).squeeze(-1)
        return torch.softmax(logits, dim=1)


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[2] != 4 or features.shape[1] < 4:
        raise ValueError("ISAC features must have shape [record, subcarrier>=4, 4]")
    return IsacAllocator()


def predictions(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model(features)


def training_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    del targets
    power = model(features)
    comm = torch.clamp(features[..., 0], min=1e-8)
    sensing = torch.clamp(features[..., 1], min=1e-8)
    noise = torch.clamp(features[:, 0, 2], min=1e-8)
    alpha = torch.clamp(features[:, 0, 3], min=0.0, max=1.0)
    rate = torch.sum(torch.log2(1.0 + power * comm / noise[:, None]), dim=1)
    sensing_information = torch.log2(1.0 + torch.sum(power * sensing, dim=1) / noise)
    utility = (1.0 - alpha) * rate / comm.shape[1] + alpha * sensing_information
    return -torch.mean(utility)


def metrics(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def metrics_from_predictions(output: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    del targets
    comm = torch.clamp(features[..., 0], min=1e-8)
    sensing = torch.clamp(features[..., 1], min=1e-8)
    noise = torch.clamp(features[:, 0, 2], min=1e-8)
    alpha = torch.clamp(features[:, 0, 3], min=0.0, max=1.0)
    rate = torch.sum(torch.log2(1.0 + output * comm / noise[:, None]), dim=1)
    sensing_information = torch.log2(1.0 + torch.sum(output * sensing, dim=1) / noise)
    utility = (1.0 - alpha) * rate / comm.shape[1] + alpha * sensing_information
    return {
        "communication_rate_bps_hz": float(torch.mean(rate).cpu()),
        "sensing_information": float(torch.mean(sensing_information).cpu()),
        "scalarized_utility": float(torch.mean(utility).cpu()),
        "power_sum_error": float(torch.max(torch.abs(torch.sum(output, dim=1) - 1.0)).cpu()),
    }


def runtime_inputs(features: np.ndarray):
    return [{"name": "features", "dtype": "float32", "shape": ["batch", int(features.shape[1]), 4], "semantic": "per_subcarrier_communication_sensing_noise_weight"}]


def runtime_outputs(features: np.ndarray):
    return [{"name": "power", "dtype": "float32", "shape": ["batch", int(features.shape[1])], "semantic": "unit_simplex_subcarrier_power"}]


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {"features": np.ascontiguousarray(features, dtype=np.float32)}


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model.cpu().eval(), sample.cpu(), str(path), input_names=["features"], output_names=["power"],
        dynamic_axes={"features": {0: "batch"}, "power": {0: "batch"}}, opset_version=18, dynamo=False,
    )


def binding_params() -> dict[str, str]:
    return {"mode": "learned_artifact", "artifact_manifest_path": "trained_artifact.yaml", "artifact_entrypoint": ENTRYPOINT_ID}


def benchmark_methods(artifact_path: str, package_sha256: str):
    return (
        ("equal_power", "Equal power", "baseline", {"mode": "equal_power"}),
        ("communications_water_filling", "Communication-only water filling", "baseline", {"mode": "communications_water_filling"}),
        ("scalarized_reference", "Per-scene scalarized optimization", "oracle", {"mode": "scalarized_reference"}),
        ("learned_isac_allocator", "Learned joint allocator", "candidate", {"mode": "learned_artifact", "artifact_manifest_path": artifact_path, "artifact_entrypoint": ENTRYPOINT_ID, "artifact_package_sha256": package_sha256}),
    )
