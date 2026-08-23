from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "aoa_estimation"
COMPONENT_ID = "aoa_estimator"
COMPONENT_ROLE = "single_source_ula_aoa_estimator"
ENTRYPOINT_ID = "aoa_estimator"
OPERATION_ID = "model.aoa_estimator_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "rmse_deg"
DESCRIPTION = (
    "Bartlett-initialized covariance-residual estimator for one narrowband "
    "source on a half-wavelength ULA, returned through Noema's portable AoA ABI."
)
RUNTIME_INPUTS = [
    {
        "name": "snapshots_ri",
        "dtype": "float32",
        "shape": ["batch", "antenna", "snapshot", 2],
        "semantic": "complex_narrowband_ula_snapshots_real_imag",
    }
]
RUNTIME_OUTPUTS = [
    {"name": "angles_deg", "dtype": "float32", "shape": ["batch"], "semantic": "estimated_source_angle_deg"},
]


def runtime_inputs(features: np.ndarray):
    return [
        {
            **RUNTIME_INPUTS[0],
            "shape": ["batch", int(features.shape[1]), int(features.shape[2]), 2],
        }
    ]


def runtime_outputs(features: np.ndarray):
    del features
    return list(RUNTIME_OUTPUTS)


class CovarianceAoaEstimator(nn.Module):
    def __init__(self, antenna_count: int) -> None:
        super().__init__()
        self.antenna_count = int(antenna_count)
        dimension = 2 * int(antenna_count) * int(antenna_count)
        self.residual = nn.Sequential(
            nn.Linear(dimension, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # The fixed physical bank supplies a strong continuous-data baseline.
        # Training learns only the bounded correction from the full covariance,
        # so a weak checkpoint cannot destroy the array-processing solution.
        grid = torch.linspace(-60.0, 60.0, 481, dtype=torch.float32)
        antenna = torch.arange(self.antenna_count, dtype=torch.float32)[None, :]
        phase = torch.pi * torch.sin(torch.deg2rad(grid))[:, None] * antenna
        self.register_buffer("angle_grid_deg", grid)
        self.register_buffer("steering_real", torch.cos(phase))
        self.register_buffer("steering_imag", torch.sin(phase))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, snapshots_ri: torch.Tensor) -> torch.Tensor:
        real = snapshots_ri[..., 0]
        imag = snapshots_ri[..., 1]
        real = real - torch.mean(real, dim=2, keepdim=True)
        imag = imag - torch.mean(imag, dim=2, keepdim=True)
        count = snapshots_ri.shape[2]
        covariance_real = (
            torch.matmul(real, real.transpose(1, 2))
            + torch.matmul(imag, imag.transpose(1, 2))
        ) / count
        covariance_imag = (
            torch.matmul(imag, real.transpose(1, 2))
            - torch.matmul(real, imag.transpose(1, 2))
        ) / count
        scale = torch.mean(torch.diagonal(covariance_real, dim1=1, dim2=2), dim=1)
        features = torch.cat([covariance_real, covariance_imag], dim=2)
        features = features / scale[:, None, None].clamp_min(1e-6)
        sample_real = real.transpose(1, 2)
        sample_imag = imag.transpose(1, 2)
        projection_real = (
            torch.matmul(sample_real, self.steering_real.T)
            + torch.matmul(sample_imag, self.steering_imag.T)
        )
        projection_imag = (
            torch.matmul(sample_imag, self.steering_real.T)
            - torch.matmul(sample_real, self.steering_imag.T)
        )
        spectrum = torch.mean(
            projection_real.square() + projection_imag.square(), dim=1
        )
        base_angle = self.angle_grid_deg[torch.argmax(spectrum, dim=1)]
        correction = 2.0 * torch.tanh(
            self.residual(features.flatten(1)).squeeze(1)
        )
        return torch.clamp(base_angle + correction, -60.0, 60.0)


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 4 or features.shape[-1] != 2:
        raise ValueError("AoA snapshots must have shape [record, antenna, snapshot, 2]")
    return CovarianceAoaEstimator(int(features.shape[1]))


def training_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    labels = targets.reshape(-1)
    return torch.mean((model(features) - labels) ** 2)


def metrics(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def predictions(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model(features)


def metrics_from_predictions(
    output: torch.Tensor, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    del features
    errors = torch.abs(output.reshape(-1) - targets.reshape(-1))
    return {
        "mse": float(torch.mean(errors**2).cpu()),
        "rmse_deg": float(torch.sqrt(torch.mean(errors**2)).cpu()),
        "mae_deg": float(torch.mean(errors).cpu()),
    }


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {"snapshots_ri": np.ascontiguousarray(features, dtype=np.float32)}


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model.cpu().eval(),
        sample[:2].cpu(),
        str(path),
        input_names=["snapshots_ri"],
        output_names=["angles_deg"],
        dynamic_axes={"snapshots_ri": {0: "batch"}, "angles_deg": {0: "batch"}},
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
        ("bartlett", "Bartlett reference", "baseline", {"mode": "bartlett_reference"}),
        ("music", "MUSIC", "baseline", {"mode": "music"}),
        (
            "learned_estimator",
            "Learned Bartlett-residual estimator",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_path,
                "artifact_entrypoint": ENTRYPOINT_ID,
                "artifact_package_sha256": package_sha256,
            },
        ),
    )
