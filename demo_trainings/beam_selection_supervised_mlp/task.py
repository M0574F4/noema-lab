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
PRIMARY_METRIC = "codebook_accuracy"
DESCRIPTION = (
    "Supervised finite-codebook MISO beam policy trained from captured channel "
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
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.antenna_count, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.antenna_count),
        )
        antenna = torch.arange(self.antenna_count, dtype=torch.float32)[:, None]
        beam = torch.arange(self.antenna_count, dtype=torch.float32)[None, :]
        phase = 2.0 * math.pi * antenna * beam / float(self.antenna_count)
        codebook = torch.stack(
            [torch.cos(phase), torch.sin(phase)], dim=-1
        ).permute(1, 0, 2) / math.sqrt(float(self.antenna_count))
        self.register_buffer("codebook_ri", codebook)

    def logits(self, channels_ri: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(torch.mean(channels_ri**2, dim=(1, 2), keepdim=True)).clamp_min(1e-6)
        return self.classifier((channels_ri / scale).flatten(1))

    def forward(self, channels_ri: torch.Tensor) -> torch.Tensor:
        indices = torch.argmax(self.logits(channels_ri), dim=1)
        return self.codebook_ri[indices]

    def oracle_labels(self, channels_ri: torch.Tensor) -> torch.Tensor:
        channel_real = channels_ri[..., 0]
        channel_imag = channels_ri[..., 1]
        beam_real = self.codebook_ri[..., 0]
        beam_imag = self.codebook_ri[..., 1]
        gain_real = channel_real @ beam_real.T + channel_imag @ beam_imag.T
        gain_imag = channel_imag @ beam_real.T - channel_real @ beam_imag.T
        return torch.argmax(gain_real**2 + gain_imag**2, dim=1)


def build_model(features: np.ndarray) -> nn.Module:
    if features.ndim != 3 or features.shape[-1] != 2:
        raise ValueError("beam channels must have shape [record, tx_antenna, 2]")
    return BeamPolicy(int(features.shape[1]))


def training_loss(model: BeamPolicy, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    del targets
    labels = model.oracle_labels(features)
    return nn.functional.cross_entropy(model.logits(features), labels)


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
    labels = torch.argmax(oracle_real**2 + oracle_imag**2, dim=1)
    similarity = torch.sum(
        weights[:, None, :, :] * codebook[None, :, :, :], dim=(2, 3)
    )
    prediction = torch.argmax(similarity, dim=1)
    weight_real, weight_imag = weights[..., 0], weights[..., 1]
    gain_real = torch.sum(channel_real * weight_real + channel_imag * weight_imag, dim=1)
    gain_imag = torch.sum(channel_imag * weight_real - channel_real * weight_imag, dim=1)
    gain = gain_real**2 + gain_imag**2
    return {
        "codebook_accuracy": float(torch.mean((prediction == labels).float()).cpu()),
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
            "Exhaustive DFT-codebook oracle",
            "baseline",
            {"mode": "codebook_sweep_reference"},
        ),
        (
            "learned_beam_policy",
            "Learned beam policy",
            "candidate",
            {
                "mode": "learned_artifact",
                "artifact_manifest_path": artifact_path,
                "artifact_entrypoint": ENTRYPOINT_ID,
                "artifact_package_sha256": package_sha256,
            },
        ),
    )
