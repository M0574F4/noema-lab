from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "range_localization"
COMPONENT_ID = "localizer"
COMPONENT_ROLE = "two_dimensional_range_localizer"
ENTRYPOINT_ID = "localization_estimator"
OPERATION_ID = "model.localization_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "rmse_m"
DESCRIPTION = (
    "Geometry-aware residual localizer trained from captured noisy ranges and "
    "returned through Noema's portable range-localization ABI."
)
RUNTIME_INPUTS = [
    {"name": "anchors", "dtype": "float32", "shape": ["batch", "anchor", 2], "semantic": "two_dimensional_anchor_coordinates"},
    {"name": "ranges", "dtype": "float32", "shape": ["batch", "anchor"], "semantic": "measured_anchor_ranges_m"},
]
RUNTIME_OUTPUTS = [
    {"name": "positions", "dtype": "float32", "shape": ["batch", 2], "semantic": "estimated_two_dimensional_position_m"},
]


def runtime_inputs(features: np.ndarray):
    anchor_count = int(features.shape[1])
    return [
        {**RUNTIME_INPUTS[0], "shape": ["batch", anchor_count, 2]},
        {**RUNTIME_INPUTS[1], "shape": ["batch", anchor_count]},
    ]


def runtime_outputs(features: np.ndarray):
    del features
    return list(RUNTIME_OUTPUTS)


class RangeLocalizer(nn.Module):
    def __init__(self, anchor_count: int) -> None:
        super().__init__()
        self.anchor_count = int(anchor_count)
        self.residual = nn.Sequential(
            nn.Linear(self.anchor_count * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, anchors: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        baseline = _linear_trilateration(anchors, ranges)
        scale = torch.amax(anchors, dim=(1, 2), keepdim=False).clamp_min(1.0)
        normalized = torch.cat(
            [anchors / scale[:, None, None], ranges[..., None] / scale[:, None, None]],
            dim=2,
        ).reshape(anchors.shape[0], -1)
        correction = 0.15 * scale[:, None] * torch.tanh(self.residual(normalized))
        return baseline + correction


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[2] != 3 or features.shape[1] < 3:
        raise ValueError("range features must have shape [record, anchor>=3, 3]")
    return RangeLocalizer(int(features.shape[1]))


def model_inputs(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return features[..., :2], features[..., 2]


def training_loss(
    model: nn.Module, features: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("localization targets must have shape [record, 2]")
    anchors, ranges = model_inputs(features)
    return torch.mean((model(anchors, ranges) - targets) ** 2)


def metrics(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def predictions(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    anchors, ranges = model_inputs(features)
    return model(anchors, ranges)


def metrics_from_predictions(
    output: torch.Tensor, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    del features
    errors = torch.linalg.vector_norm(output - targets, dim=1)
    return {
        "mse": float(torch.mean(errors**2).cpu()),
        "rmse_m": float(torch.sqrt(torch.mean(errors**2)).cpu()),
        "mae_m": float(torch.mean(errors).cpu()),
    }


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "anchors": np.ascontiguousarray(features[..., :2], dtype=np.float32),
        "ranges": np.ascontiguousarray(features[..., 2], dtype=np.float32),
    }


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    anchors, ranges = model_inputs(sample[:2])
    torch.onnx.export(
        model.cpu().eval(),
        (anchors.cpu(), ranges.cpu()),
        str(path),
        input_names=["anchors", "ranges"],
        output_names=["positions"],
        dynamic_axes={"anchors": {0: "batch"}, "ranges": {0: "batch"}, "positions": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )


def binding_params() -> dict[str, str]:
    return {
        "mode": "learned_artifact",
        "artifact_manifest_path": "trained_artifact.yaml",
        "artifact_entrypoint": ENTRYPOINT_ID,
    }


def benchmark_methods(artifact_path: str, package_sha256: str):
    return (
        ("trilateration", "Linear trilateration", "baseline", {"mode": "trilateration"}),
        (
            "regularized_trilateration",
            "Regularized trilateration",
            "baseline",
            {"mode": "regularized_trilateration"},
        ),
        (
            "learned_localizer",
            "Learned residual localizer",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_path,
                "artifact_entrypoint": ENTRYPOINT_ID,
                "artifact_package_sha256": package_sha256,
            },
        ),
    )


def _linear_trilateration(anchors: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    origin = anchors[:, :1, :]
    matrix = 2.0 * (anchors[:, 1:, :] - origin)
    rhs = (
        ranges[:, :1] ** 2
        - ranges[:, 1:] ** 2
        + torch.sum(anchors[:, 1:, :] ** 2, dim=2)
        - torch.sum(origin**2, dim=2)
    )
    normal = torch.matmul(matrix.transpose(1, 2), matrix)
    projected = torch.matmul(matrix.transpose(1, 2), rhs[..., None]).squeeze(-1)
    a = normal[:, 0, 0] + 1e-4
    b = normal[:, 0, 1]
    c = normal[:, 1, 0]
    d = normal[:, 1, 1] + 1e-4
    determinant = (a * d - b * c).clamp_min(1e-8)
    x = (d * projected[:, 0] - b * projected[:, 1]) / determinant
    y = (a * projected[:, 1] - c * projected[:, 0]) / determinant
    return torch.stack([x, y], dim=1)
