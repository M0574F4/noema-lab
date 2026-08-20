from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict

import torch


# This is a literature-inspired demonstration model, not a reproduction of a named
# published model and not part of Noema's normative training contract.
ARCHITECTURE = "klt_initialized_angular_delay_quantization_residual_v3"


def _unitary_dft(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(int(size), dtype=torch.float32)
    phase = 2.0 * math.pi * indices[:, None] * indices[None, :] / float(size)
    scale = math.sqrt(float(size))
    return torch.cos(phase) / scale, -torch.sin(phase) / scale


def _complex_right_multiply(
    real: torch.Tensor,
    imag: torch.Tensor,
    matrix_real: torch.Tensor,
    matrix_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex right multiplication expressed entirely with real operations."""

    return (
        torch.matmul(real, matrix_real) - torch.matmul(imag, matrix_imag),
        torch.matmul(real, matrix_imag) + torch.matmul(imag, matrix_real),
    )


class FixedAngularDelayTransform(torch.nn.Module):
    """Fixed unitary antenna-DFT/frequency-IDFT pair with a real-valued graph."""

    def __init__(
        self,
        *,
        tx_antennas: int,
        subcarrier_count: int,
        enabled: bool = True,
    ):
        super().__init__()
        self.tx_antennas = int(tx_antennas)
        self.subcarrier_count = int(subcarrier_count)
        self.enabled = bool(enabled)
        antenna_real, antenna_imag = _unitary_dft(self.tx_antennas)
        frequency_real, frequency_imag = _unitary_dft(self.subcarrier_count)
        self.register_buffer("antenna_dft_real", antenna_real)
        self.register_buffer("antenna_dft_imag", antenna_imag)
        self.register_buffer("frequency_dft_real", frequency_real)
        self.register_buffer("frequency_dft_imag", frequency_imag)

    def analysis(self, csi_ri: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return csi_ri
        real, imag = csi_ri[:, 0], csi_ri[:, 1]
        real, imag = _complex_right_multiply(
            real.transpose(1, 2),
            imag.transpose(1, 2),
            self.antenna_dft_real,
            self.antenna_dft_imag,
        )
        real, imag = real.transpose(1, 2), imag.transpose(1, 2)
        # Frequency -> delay is a unitary inverse DFT.
        real, imag = _complex_right_multiply(
            real,
            imag,
            self.frequency_dft_real.transpose(0, 1),
            -self.frequency_dft_imag.transpose(0, 1),
        )
        return torch.stack((real, imag), dim=1)

    def synthesis(self, angular_delay_ri: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return angular_delay_ri
        real, imag = angular_delay_ri[:, 0], angular_delay_ri[:, 1]
        real, imag = _complex_right_multiply(
            real,
            imag,
            self.frequency_dft_real,
            self.frequency_dft_imag,
        )
        real, imag = _complex_right_multiply(
            real.transpose(1, 2),
            imag.transpose(1, 2),
            self.antenna_dft_real.transpose(0, 1),
            -self.antenna_dft_imag.transpose(0, 1),
        )
        return torch.stack((real.transpose(1, 2), imag.transpose(1, 2)), dim=1)

    def round_trip(self, csi_ri: torch.Tensor) -> torch.Tensor:
        """Apply analysis then synthesis; useful for contract and export tests."""

        return self.synthesis(self.analysis(csi_ri))

    def forward(self, csi_ri: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        return self.synthesis(csi_ri) if inverse else self.analysis(csi_ri)


class ResidualRefinementBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=False),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            inputs + self.net(inputs),
            negative_slope=0.1,
        )


class PerLatentAdaptor(torch.nn.Module):
    """Optional element-wise affine adaptor for quantizer-range utilization."""

    def __init__(self, feedback_dimension: int):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(int(feedback_dimension)))
        self.bias = torch.nn.Parameter(torch.zeros(int(feedback_dimension)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.gain + self.bias


class CsiFeedbackEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        tx_antennas: int,
        subcarrier_count: int,
        feedback_dimension: int,
        base_channels: int = 24,
        refinement_blocks: int = 2,
        clip_value: float = 1.0,
        latent_activation: str = "tanh",
        global_linear_domain: str = "angular_delay",
        use_angular_delay_transform: bool = True,
        use_per_latent_adaptor: bool = True,
    ):
        super().__init__()
        self.tx_antennas = int(tx_antennas)
        self.subcarrier_count = int(subcarrier_count)
        self.feedback_dimension = int(feedback_dimension)
        self.base_channels = int(base_channels)
        self.coarse_channels = self.base_channels * 2
        self.coarse_antennas = (self.tx_antennas + 1) // 2
        self.coarse_subcarriers = (self.subcarrier_count + 1) // 2
        self.clip_value = float(clip_value)
        self.latent_activation = str(latent_activation).strip().lower()
        if self.latent_activation not in {"identity", "hardtanh", "tanh"}:
            raise ValueError(
                "latent_activation must be identity, hardtanh, or tanh"
            )
        self.global_linear_domain = str(global_linear_domain).strip().lower()
        if self.global_linear_domain not in {"angular_delay", "frequency"}:
            raise ValueError(
                "global_linear_domain must be angular_delay or frequency"
            )
        self.transform = FixedAngularDelayTransform(
            tx_antennas=self.tx_antennas,
            subcarrier_count=self.subcarrier_count,
            enabled=use_angular_delay_transform,
        )

        real_dimension = 2 * self.tx_antennas * self.subcarrier_count
        self.global_projection = torch.nn.Linear(real_dimension, self.feedback_dimension)
        self.stem = torch.nn.Conv2d(2, self.base_channels, 3, padding=1)
        self.native_refinement = torch.nn.Sequential(
            *[
                ResidualRefinementBlock(self.base_channels)
                for _ in range(max(0, int(refinement_blocks)))
            ]
        )
        self.downsample = torch.nn.Conv2d(
            self.base_channels,
            self.coarse_channels,
            3,
            stride=2,
            padding=1,
        )
        self.coarse_refinement = torch.nn.Sequential(
            *[
                ResidualRefinementBlock(self.coarse_channels)
                for _ in range(max(0, int(refinement_blocks)))
            ]
        )
        self.nonlinear_projection = torch.nn.Linear(
            self.coarse_channels * self.coarse_antennas * self.coarse_subcarriers,
            self.feedback_dimension,
        )
        torch.nn.init.normal_(self.nonlinear_projection.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.nonlinear_projection.bias)
        self.latent_adaptor = (
            PerLatentAdaptor(self.feedback_dimension)
            if use_per_latent_adaptor
            else torch.nn.Identity()
        )

    def forward(self, csi_ri: torch.Tensor) -> torch.Tensor:
        angular_delay = self.transform.analysis(csi_ri.float())
        global_input = (
            csi_ri.float()
            if self.global_linear_domain == "frequency"
            else angular_delay
        )
        global_code = self.global_projection(global_input.flatten(start_dim=1))
        features = torch.nn.functional.leaky_relu(
            self.stem(angular_delay), negative_slope=0.1
        )
        features = self.native_refinement(features)
        features = torch.nn.functional.leaky_relu(
            self.downsample(features), negative_slope=0.1
        )
        features = self.coarse_refinement(features)
        nonlinear_code = self.nonlinear_projection(features.flatten(start_dim=1))
        code = self.latent_adaptor(global_code + nonlinear_code)
        if self.latent_activation == "identity":
            return code
        if self.latent_activation == "hardtanh":
            return torch.clamp(code, -self.clip_value, self.clip_value)
        return torch.tanh(code) * self.clip_value


class CsiFeedbackDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        tx_antennas: int,
        subcarrier_count: int,
        feedback_dimension: int,
        base_channels: int = 24,
        refinement_blocks: int = 2,
        global_linear_domain: str = "angular_delay",
        use_angular_delay_transform: bool = True,
        use_per_latent_adaptor: bool = True,
    ):
        super().__init__()
        self.tx_antennas = int(tx_antennas)
        self.subcarrier_count = int(subcarrier_count)
        self.feedback_dimension = int(feedback_dimension)
        self.base_channels = int(base_channels)
        self.coarse_channels = self.base_channels * 2
        self.coarse_antennas = (self.tx_antennas + 1) // 2
        self.coarse_subcarriers = (self.subcarrier_count + 1) // 2
        self.global_linear_domain = str(global_linear_domain).strip().lower()
        if self.global_linear_domain not in {"angular_delay", "frequency"}:
            raise ValueError(
                "global_linear_domain must be angular_delay or frequency"
            )
        self.transform = FixedAngularDelayTransform(
            tx_antennas=self.tx_antennas,
            subcarrier_count=self.subcarrier_count,
            enabled=use_angular_delay_transform,
        )
        self.latent_adaptor = (
            PerLatentAdaptor(self.feedback_dimension)
            if use_per_latent_adaptor
            else torch.nn.Identity()
        )

        real_dimension = 2 * self.tx_antennas * self.subcarrier_count
        self.global_expansion = torch.nn.Linear(self.feedback_dimension, real_dimension)
        self.nonlinear_expansion = torch.nn.Linear(
            self.feedback_dimension,
            self.coarse_channels * self.coarse_antennas * self.coarse_subcarriers,
        )
        self.coarse_refinement = torch.nn.Sequential(
            *[
                ResidualRefinementBlock(self.coarse_channels)
                for _ in range(max(0, int(refinement_blocks)))
            ]
        )
        self.upsample = torch.nn.ConvTranspose2d(
            self.coarse_channels,
            self.base_channels,
            4,
            stride=2,
            padding=1,
        )
        self.native_refinement = torch.nn.Sequential(
            *[
                ResidualRefinementBlock(self.base_channels)
                for _ in range(max(0, int(refinement_blocks)))
            ]
        )
        self.output_projection = torch.nn.Conv2d(self.base_channels, 2, 3, padding=1)

    def forward(self, feedback_code: torch.Tensor) -> torch.Tensor:
        adapted_code = self.latent_adaptor(feedback_code.float())
        global_values = self.global_expansion(adapted_code).reshape(
            feedback_code.shape[0], 2, self.tx_antennas, self.subcarrier_count
        )
        features = self.nonlinear_expansion(adapted_code).reshape(
            feedback_code.shape[0],
            self.coarse_channels,
            self.coarse_antennas,
            self.coarse_subcarriers,
        )
        features = self.coarse_refinement(
            torch.nn.functional.leaky_relu(features, negative_slope=0.1)
        )
        features = torch.nn.functional.leaky_relu(
            self.upsample(features), negative_slope=0.1
        )
        features = self.native_refinement(features)
        nonlinear_angular_delay = self.output_projection(features)
        nonlinear_angular_delay = nonlinear_angular_delay[
            :, :, : self.tx_antennas, : self.subcarrier_count
        ]
        if self.global_linear_domain == "frequency":
            return global_values + self.transform.synthesis(nonlinear_angular_delay)
        angular_delay = global_values + nonlinear_angular_delay
        return self.transform.synthesis(angular_delay)


class ReferenceCsiFeedbackAutoencoder(torch.nn.Module):
    """Replaceable literature-inspired example, not part of Noema's contract."""

    def __init__(
        self,
        *,
        tx_antennas: int,
        subcarrier_count: int,
        feedback_dimension: int,
        bits_per_latent: int,
        clip_value: float,
        feedback_mode: str = "uniform_quantized",
        base_channels: int = 24,
        refinement_blocks: int = 2,
        latent_activation: str = "tanh",
        global_linear_domain: str = "angular_delay",
        use_angular_delay_transform: bool = True,
        use_per_latent_adaptor: bool = True,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        # hidden_channels is retained as a compatibility alias for older exports.
        if hidden_channels is not None:
            base_channels = int(hidden_channels)
        self.bits_per_latent = int(bits_per_latent)
        self.clip_value = float(clip_value)
        self.feedback_mode = str(feedback_mode)
        common = {
            "tx_antennas": tx_antennas,
            "subcarrier_count": subcarrier_count,
            "feedback_dimension": feedback_dimension,
            "base_channels": base_channels,
            "refinement_blocks": refinement_blocks,
            "global_linear_domain": global_linear_domain,
            "use_angular_delay_transform": use_angular_delay_transform,
            "use_per_latent_adaptor": use_per_latent_adaptor,
        }
        self.encoder = CsiFeedbackEncoder(
            clip_value=clip_value,
            latent_activation=latent_activation,
            **common,
        )
        self.decoder = CsiFeedbackDecoder(**common)

    def initialize_klt_prior(
        self,
        *,
        mean: torch.Tensor,
        basis: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Install a training-split KLT codec as the exact zero-residual prior."""

        mean = mean.detach().float().flatten()
        basis = basis.detach().float()
        scale = scale.detach().float().flatten()
        expected_input = int(self.encoder.global_projection.in_features)
        expected_feedback = int(self.encoder.feedback_dimension)
        if tuple(basis.shape) != (expected_feedback, expected_input):
            raise ValueError(
                "KLT basis must have shape (%d,%d), got %s"
                % (expected_feedback, expected_input, tuple(basis.shape))
            )
        if tuple(mean.shape) != (expected_input,):
            raise ValueError(
                "KLT mean must have shape (%d,), got %s"
                % (expected_input, tuple(mean.shape))
            )
        if tuple(scale.shape) != (expected_feedback,) or torch.any(scale <= 0):
            raise ValueError(
                "KLT scale must contain %d positive values" % expected_feedback
            )

        encoder_weight = basis / scale[:, None]
        encoder_bias = -(mean @ encoder_weight.transpose(0, 1))
        decoder_weight = (scale[:, None] * basis).transpose(0, 1)
        with torch.no_grad():
            self.encoder.global_projection.weight.copy_(encoder_weight)
            self.encoder.global_projection.bias.copy_(encoder_bias)
            self.decoder.global_expansion.weight.copy_(decoder_weight)
            self.decoder.global_expansion.bias.copy_(mean)
            self.encoder.nonlinear_projection.weight.zero_()
            self.encoder.nonlinear_projection.bias.zero_()
            self.decoder.output_projection.weight.zero_()
            self.decoder.output_projection.bias.zero_()
            for adaptor in (
                self.encoder.latent_adaptor,
                self.decoder.latent_adaptor,
            ):
                if isinstance(adaptor, PerLatentAdaptor):
                    adaptor.gain.fill_(1.0)
                    adaptor.bias.zero_()

    def transport(
        self,
        feedback_code: torch.Tensor,
        *,
        quantize: bool = True,
    ) -> torch.Tensor:
        if not quantize or self.feedback_mode == "ideal_noiseless":
            return feedback_code
        if self.feedback_mode != "uniform_quantized":
            raise ValueError("Unsupported CSI feedback mode: %s" % self.feedback_mode)
        return uniform_quantize(
            feedback_code,
            bits_per_latent=self.bits_per_latent,
            clip_value=self.clip_value,
            straight_through=True,
        )

    def forward(
        self,
        csi_ri: torch.Tensor,
        *,
        quantize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        feedback_code = self.encoder(csi_ri)
        received_code = self.transport(feedback_code, quantize=quantize)
        reconstruction = self.decoder(received_code)
        return {
            "feedback_code": feedback_code,
            "received_code": received_code,
            "reconstruction": reconstruction,
        }


def uniform_quantize(
    values: torch.Tensor,
    *,
    bits_per_latent: int,
    clip_value: float,
    straight_through: bool,
) -> torch.Tensor:
    """Canonical symmetric uniform quantizer; STE changes backward only."""

    if int(bits_per_latent) < 1:
        raise ValueError("bits_per_latent must be at least one")
    if float(clip_value) <= 0.0:
        raise ValueError("clip_value must be positive")
    levels = float((1 << int(bits_per_latent)) - 1)
    clip = float(clip_value)
    clipped = torch.clamp(values, -clip, clip)
    indices = torch.round((clipped + clip) * levels / (2.0 * clip))
    quantized = indices * (2.0 * clip) / levels - clip
    if straight_through:
        return values + (quantized - values).detach()
    return quantized


def export_onnx_components(
    model: ReferenceCsiFeedbackAutoencoder,
    encoder_path: str | Path,
    decoder_path: str | Path,
) -> Dict[str, str]:
    encoder_destination = Path(encoder_path)
    decoder_destination = Path(decoder_path)
    encoder_destination.parent.mkdir(parents=True, exist_ok=True)
    decoder_destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example_csi = torch.zeros(
        (1, 2, model.encoder.tx_antennas, model.encoder.subcarrier_count),
        dtype=torch.float32,
    )
    with torch.no_grad():
        example_code = model.encoder(example_csi)
    torch.onnx.export(
        model.encoder,
        (example_csi,),
        str(encoder_destination),
        input_names=["csi_ri"],
        output_names=["feedback_code"],
        dynamic_axes={"csi_ri": {0: "batch"}, "feedback_code": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        model.decoder,
        (example_code,),
        str(decoder_destination),
        input_names=["feedback_code"],
        output_names=["csi_hat_ri"],
        dynamic_axes={"feedback_code": {0: "batch"}, "csi_hat_ri": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return {
        "encoder_sha256": _file_sha256(encoder_destination),
        "decoder_sha256": _file_sha256(decoder_destination),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
