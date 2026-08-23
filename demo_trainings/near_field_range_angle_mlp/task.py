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
        # Absolute pilot phase is arbitrary.  Adjacent-element correlations retain
        # the linear phase slope (angle), while correlations between adjacent
        # slopes retain the spherical-wave curvature (range).  Supplying both
        # invariants makes the starter genuinely useful without baking a search
        # grid or the simulator's labels into the learned component.
        feature_count = antennas * 3 + (antennas - 1) * 2 + (antennas - 2) * 2
        self.network = nn.Sequential(
            nn.Linear(feature_count, 192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _unit_pair(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(1e-6)
        return real / magnitude, imag / magnitude

    def forward(self, array_ri: torch.Tensor) -> torch.Tensor:
        real = array_ri[:, :, 0]
        imag = array_ri[:, :, 1]
        unit_real, unit_imag = self._unit_pair(real, imag)

        slope_real = unit_real[:, 1:] * unit_real[:, :-1] + unit_imag[:, 1:] * unit_imag[:, :-1]
        slope_imag = unit_imag[:, 1:] * unit_real[:, :-1] - unit_real[:, 1:] * unit_imag[:, :-1]
        slope_real, slope_imag = self._unit_pair(slope_real, slope_imag)

        curve_real = slope_real[:, 1:] * slope_real[:, :-1] + slope_imag[:, 1:] * slope_imag[:, :-1]
        curve_imag = slope_imag[:, 1:] * slope_real[:, :-1] - slope_real[:, 1:] * slope_imag[:, :-1]
        curve_real, curve_imag = self._unit_pair(curve_real, curve_imag)

        log_magnitude = torch.log1p(torch.sqrt(real.square() + imag.square()))
        features = torch.cat(
            [unit_real, unit_imag, log_magnitude, slope_real, slope_imag, curve_real, curve_imag],
            dim=1,
        )
        raw = self.network(features)
        range_m = RANGE_MIN_M + (RANGE_MAX_M - RANGE_MIN_M) * torch.sigmoid(raw[:, 0])
        mean_slope_real = torch.mean(slope_real, dim=1)
        mean_slope_imag = torch.mean(slope_imag, dim=1)
        spatial_phase = torch.atan2(mean_slope_imag, mean_slope_real)
        angle_deg = torch.asin(torch.clamp(spatial_phase / torch.pi, -0.999, 0.999)) * (180.0 / torch.pi)
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
