from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK_ID = "beamforming_precoding"
COMPONENT_ID = "beam_policy"
COMPONENT_ROLE = "single_user_miso_beam_policy"
ENTRYPOINT_ID = "beam_policy"
OPERATION_ID = "model.beamforming_adapter"
REQUIRED_INPUTS = ["problem"]
PRIMARY_METRIC = "normalized_channel_gain"
DESCRIPTION = (
    "Distribution-aware eight-beam MISO codebook learned from captured channel "
    "vectors and returned through Noema's portable beam-policy ABI."
)
RUNTIME_INPUTS = [
    {"name": "channels_ri", "dtype": "float32", "shape": ["batch", "tx_antenna", 2], "semantic": "complex_single_user_miso_channel_real_imag"},
]
RUNTIME_OUTPUTS = [
    {"name": "weights_ri", "dtype": "float32", "shape": ["batch", "tx_antenna", 2], "semantic": "unit_norm_complex_precoder_real_imag"},
]


def runtime_inputs(features: np.ndarray):
    return [{**RUNTIME_INPUTS[0], "shape": ["batch", int(features.shape[1]), 2]}]


def runtime_outputs(features: np.ndarray):
    return [{**RUNTIME_OUTPUTS[0], "shape": ["batch", int(features.shape[1]), 2]}]


class BeamPolicy(nn.Module):
    def __init__(self, antenna_count: int) -> None:
        super().__init__()
        self.antenna_count = int(antenna_count)
        initial_frequency = torch.arange(
            -self.antenna_count // 2,
            self.antenna_count - self.antenna_count // 2,
            dtype=torch.float32,
        ) * (2.0 / float(self.antenna_count))
        self.spatial_frequency_raw = nn.Parameter(
            torch.atanh(initial_frequency / 1.05)
        )
        self.register_buffer(
            "antenna_index", torch.arange(self.antenna_count, dtype=torch.float32)
        )

    def codebook(self) -> torch.Tensor:
        spatial_frequency = 1.05 * torch.tanh(self.spatial_frequency_raw)
        phase = math.pi * spatial_frequency[:, None] * self.antenna_index[None, :]
        return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1) / math.sqrt(
            float(self.antenna_count)
        )

    def normalized_gains(self, channels_ri: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook()
        channel_real = channels_ri[..., 0]
        channel_imag = channels_ri[..., 1]
        beam_real = codebook[..., 0]
        beam_imag = codebook[..., 1]
        gain_real = channel_real @ beam_real.T + channel_imag @ beam_imag.T
        gain_imag = channel_imag @ beam_real.T - channel_real @ beam_imag.T
        channel_power = torch.sum(channels_ri.square(), dim=(1, 2)).clamp_min(1e-8)
        return (gain_real.square() + gain_imag.square()) / channel_power[:, None]

    def forward(self, channels_ri: torch.Tensor) -> torch.Tensor:
        indices = torch.argmax(self.normalized_gains(channels_ri), dim=1)
        return self.codebook()[indices]


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[-1] != 2:
        raise ValueError("beam channels must have shape [record, tx_antenna, 2]")
    return BeamPolicy(int(features.shape[1]))


def training_loss(model: BeamPolicy, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    del targets
    # Hard best-beam assignment gives a compact Lloyd-style codebook update:
    # each channel trains the currently selected beam directly for array gain.
    return -torch.mean(torch.max(model.normalized_gains(features), dim=1).values)


def metrics(model: BeamPolicy, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return metrics_from_predictions(predictions(model, features), features, targets)


def predictions(model: BeamPolicy, features: torch.Tensor) -> torch.Tensor:
    return model(features)


def metrics_from_predictions(
    weights: torch.Tensor, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    del targets
    antenna_count = int(features.shape[1])
    antenna = torch.arange(antenna_count, dtype=features.dtype, device=features.device)[:, None]
    beam = torch.arange(antenna_count, dtype=features.dtype, device=features.device)[None, :]
    phase = 2.0 * math.pi * antenna * beam / float(antenna_count)
    codebook = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).permute(1, 0, 2)
    codebook = codebook / math.sqrt(float(antenna_count))
    channel_real, channel_imag = features[..., 0], features[..., 1]
    beam_real, beam_imag = codebook[..., 0], codebook[..., 1]
    oracle_real = channel_real @ beam_real.T + channel_imag @ beam_imag.T
    oracle_imag = channel_imag @ beam_real.T - channel_real @ beam_imag.T
    dft_gain = torch.max(oracle_real**2 + oracle_imag**2, dim=1).values
    weight_real, weight_imag = weights[..., 0], weights[..., 1]
    gain_real = torch.sum(channel_real * weight_real + channel_imag * weight_imag, dim=1)
    gain_imag = torch.sum(channel_imag * weight_real - channel_real * weight_imag, dim=1)
    gain = gain_real**2 + gain_imag**2
    channel_power = torch.sum(features.square(), dim=(1, 2)).clamp_min(1e-8)
    normalized_gain = gain / channel_power
    dft_normalized_gain = dft_gain / channel_power
    return {
        "normalized_channel_gain": float(torch.mean(normalized_gain).cpu()),
        "dft_normalized_gain": float(torch.mean(dft_normalized_gain).cpu()),
        "learned_to_dft_gain": float(
            (torch.mean(normalized_gain) / torch.mean(dft_normalized_gain)).cpu()
        ),
        "mean_channel_gain": float(torch.mean(gain).cpu()),
    }


def onnx_feed(features: np.ndarray) -> dict[str, np.ndarray]:
    return {"channels_ri": np.ascontiguousarray(features, dtype=np.float32)}


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model.cpu().eval(),
        sample[:2].cpu(),
        str(path),
        input_names=["channels_ri"],
        output_names=["weights_ri"],
        dynamic_axes={"channels_ri": {0: "batch"}, "weights_ri": {0: "batch"}},
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
        ("mrt", "Perfect-CSIT MRT upper bound", "oracle", {"mode": "mrt"}),
        (
            "dft_codebook_sweep",
            "Fixed eight-beam DFT sweep",
            "baseline",
            {"mode": "codebook_sweep_reference"},
        ),
        (
            "learned_beam_policy",
            "Learned eight-beam codebook",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_path,
                "artifact_entrypoint": ENTRYPOINT_ID,
                "artifact_package_sha256": package_sha256,
            },
        ),
    )
