from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "near_field_range_angle"
COMPONENT_ID = "estimator"
COMPONENT_ROLE = "near_field_range_angle_estimator"
ENTRYPOINT_ID = "near_field_estimator"
OPERATION_ID = "model.near_field_estimator_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "normalized_focusing_gain"
DESCRIPTION = "Physics-informed coherent-array range/angle estimator returned through Noema's portable near-field ABI."
RANGE_MIN_M = 0.5
RANGE_MAX_M = 5.0
ANGLE_LIMIT_DEG = 55.0


class NearFieldEstimator(nn.Module):
    def __init__(self, antennas: int) -> None:
        super().__init__()
        self.antennas = int(antennas)
        feature_count = self.antennas * 3 + 2
        self.residual = nn.Sequential(
            nn.Linear(feature_count, 192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        # A dense spherical-wave bank uses the declared physical aperture and
        # range/angle bounds. The trainable head performs bounded sub-grid and
        # low-SNR corrections from the full coherent observation.
        ranges = torch.linspace(RANGE_MIN_M, RANGE_MAX_M, 46)
        angles = torch.linspace(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG, 111)
        candidate_range = ranges[:, None].expand(-1, angles.numel()).reshape(-1)
        candidate_angle = angles[None, :].expand(ranges.numel(), -1).reshape(-1)
        speed_of_light = 299_792_458.0
        wavelength = speed_of_light / 28.0e9
        aperture = (
            torch.arange(self.antennas, dtype=torch.float32)
            - (self.antennas - 1) / 2.0
        ) * (wavelength / 2.0)
        theta = torch.deg2rad(candidate_angle)[:, None]
        radius = candidate_range[:, None]
        target_x = radius * torch.sin(theta)
        target_y = radius * torch.cos(theta)
        distance = torch.sqrt((target_x - aperture[None, :]).square() + target_y.square())
        relative = distance - radius
        amplitude = radius / distance.clamp_min(1e-9)
        phase = -2.0 * torch.pi * relative / wavelength
        steering_real = amplitude * torch.cos(phase) / self.antennas**0.5
        steering_imag = amplitude * torch.sin(phase) / self.antennas**0.5
        self.register_buffer("candidate_range", candidate_range)
        self.register_buffer("candidate_angle", candidate_angle)
        self.register_buffer("steering_real", steering_real)
        self.register_buffer("steering_imag", steering_imag)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _unit_pair(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(1e-6)
        return real / magnitude, imag / magnitude

    def forward(self, array_ri: torch.Tensor) -> torch.Tensor:
        real = array_ri[:, :, 0]
        imag = array_ri[:, :, 1]
        unit_real, unit_imag = self._unit_pair(real, imag)

        log_magnitude = torch.log1p(torch.sqrt(real.square() + imag.square()))
        projection_real = (
            torch.matmul(real, self.steering_real.T)
            + torch.matmul(imag, self.steering_imag.T)
        )
        projection_imag = (
            torch.matmul(imag, self.steering_real.T)
            - torch.matmul(real, self.steering_imag.T)
        )
        scores = projection_real.square() + projection_imag.square()
        indices = torch.argmax(scores, dim=1)
        base_range = self.candidate_range[indices]
        base_angle = self.candidate_angle[indices]
        features = torch.cat(
            [
                unit_real,
                unit_imag,
                log_magnitude,
                ((base_range - RANGE_MIN_M) / (RANGE_MAX_M - RANGE_MIN_M))[:, None],
                (base_angle / ANGLE_LIMIT_DEG)[:, None],
            ],
            dim=1,
        )
        correction = self.residual(features)
        range_m = torch.clamp(
            base_range + 0.25 * torch.tanh(correction[:, 0]),
            RANGE_MIN_M,
            RANGE_MAX_M,
        )
        angle_deg = torch.clamp(
            base_angle + torch.tanh(correction[:, 1]),
            -ANGLE_LIMIT_DEG,
            ANGLE_LIMIT_DEG,
        )
        return torch.stack([range_m, angle_deg], dim=1)


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[2] != 2 or features.shape[1] < 8:
        raise ValueError("Near-field array features must have shape [record, antenna>=8, 2]")
    return NearFieldEstimator(int(features.shape[1]))


def predictions(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model(features)


def training_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prediction = model(features)
    range_error = (prediction[:, 0] - targets[:, 0]) / (RANGE_MAX_M - RANGE_MIN_M)
    angle_error = (prediction[:, 1] - targets[:, 1]) / (2.0 * ANGLE_LIMIT_DEG)
    return torch.mean(range_error**2 + angle_error**2)


def metrics(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def metrics_from_predictions(output: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    del features
    range_error = output[:, 0] - targets[:, 0]
    angle_error = output[:, 1] - targets[:, 1]
    return {
        "range_rmse_m": float(torch.sqrt(torch.mean(range_error**2)).cpu()),
        "angle_rmse_deg": float(torch.sqrt(torch.mean(angle_error**2)).cpu()),
        "range_mae_m": float(torch.mean(torch.abs(range_error)).cpu()),
        "angle_mae_deg": float(torch.mean(torch.abs(angle_error)).cpu()),
    }


def runtime_inputs(features: np.ndarray):
    return [{"name": "array_ri", "dtype": "float32", "shape": ["batch", int(features.shape[1]), 2], "semantic": "phase_referenced_coherent_array_observation"}]


def runtime_outputs(features: np.ndarray):
    del features
    return [{"name": "range_angle", "dtype": "float32", "shape": ["batch", 2], "semantic": "range_m_and_angle_deg"}]


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {"array_ri": np.ascontiguousarray(features, dtype=np.float32)}


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(model.cpu().eval(), sample.cpu(), str(path), input_names=["array_ri"], output_names=["range_angle"], dynamic_axes={"array_ri": {0: "batch"}, "range_angle": {0: "batch"}}, opset_version=18, dynamo=False)


def binding_params() -> dict[str, str]:
    return {"mode": "learned_artifact", "artifact_manifest_path": "trained_artifact.yaml", "artifact_entrypoint": ENTRYPOINT_ID}


def benchmark_methods(artifact_path: str, package_sha256: str):
    return (
        ("far_field_steering", "Far-field steering search", "baseline", {"mode": "far_field_steering"}),
        ("polar_codebook", "Polar range-angle codebook", "baseline", {"mode": "polar_codebook"}),
        ("oracle_focus", "True-position focusing", "oracle", {"mode": "oracle_focus"}),
        ("learned_near_field_estimator", "Physics-informed learned estimator", "candidate", {"mode": "learned_artifact", "artifact_manifest_path": artifact_path, "artifact_entrypoint": ENTRYPOINT_ID, "artifact_package_sha256": package_sha256}),
    )
