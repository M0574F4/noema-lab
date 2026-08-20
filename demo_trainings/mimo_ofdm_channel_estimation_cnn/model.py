from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class CircularConv1d(nn.Module):
    """Frequency-domain convolution with OFDM wrap-around at the band edges."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError("CircularConv1d requires an odd kernel size")
        self.padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            dilation=int(dilation),
            padding=0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.padding:
            value = torch.cat(
                [
                    value[..., -self.padding :],
                    value,
                    value[..., : self.padding],
                ],
                dim=-1,
            )
        return self.conv(value)


class NoiseConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        self.filter = CircularConv1d(
            int(channels),
            int(channels),
            3,
            dilation=int(dilation),
        )
        self.project = nn.Conv1d(int(channels), int(channels), 1)
        self.condition = nn.Linear(1, 2 * int(channels))
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.activation = nn.GELU()

    def forward(
        self,
        value: torch.Tensor,
        log_noise: torch.Tensor,
    ) -> torch.Tensor:
        filtered = self.activation(self.filter(value))
        scale, bias = self.condition(log_noise).chunk(2, dim=-1)
        filtered = filtered * (1.0 + 0.25 * torch.tanh(scale[..., None]))
        filtered = filtered + bias[..., None]
        return value + self.project(self.activation(filtered))


class DelayResidualBlock(nn.Module):
    """Non-circular residual block over the finite delay axis."""

    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        padding = int(dilation)
        self.filter = nn.Conv1d(
            int(channels),
            int(channels),
            3,
            dilation=int(dilation),
            padding=padding,
        )
        self.project = nn.Conv1d(int(channels), int(channels), 1)
        self.condition = nn.Linear(1, 2 * int(channels))
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        self.activation = nn.GELU()

    def forward(
        self,
        value: torch.Tensor,
        log_noise: torch.Tensor,
    ) -> torch.Tensor:
        filtered = self.activation(self.filter(value))
        scale, bias = self.condition(log_noise).chunk(2, dim=-1)
        filtered = filtered * (1.0 + 0.25 * torch.tanh(scale[..., None]))
        filtered = filtered + bias[..., None]
        return value + self.project(self.activation(filtered))


class FrequencyResidualEstimator(nn.Module):
    """Sparse-pilot, noise-conditioned dual-domain LS refinement.

    One branch learns local and long-range structure over subcarriers. A second
    branch uses fixed real-valued DFT matrices to refine the compact
    angle-delay representation. Both branches see the divided sparse pilot
    observations; the interpolated LS estimate remains an explicit fallback.
    Zero-initialized heads make the untrained model exactly equal to LS.
    """

    def __init__(
        self,
        *,
        rx_antennas: int,
        tx_antennas: int,
        subcarriers: int,
        pilot_spacing: int,
        hidden_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.rx_antennas = int(rx_antennas)
        self.tx_antennas = int(tx_antennas)
        self.subcarriers = int(subcarriers)
        self.pilot_spacing = int(pilot_spacing)
        channels = 2 * self.rx_antennas * self.tx_antennas
        frequency = 2.0 * torch.pi * torch.arange(
            self.subcarriers,
            dtype=torch.float32,
        ) / float(self.subcarriers)
        self.register_buffer(
            "frequency_features",
            torch.stack([torch.sin(frequency), torch.cos(frequency)], dim=0)[
                None, ...
            ],
        )
        indices = torch.arange(self.subcarriers, dtype=torch.float32)
        phase = (
            2.0
            * torch.pi
            * indices[:, None]
            * indices[None, :]
            / float(self.subcarriers)
        )
        self.register_buffer("idft_cos", torch.cos(phase) / self.subcarriers)
        self.register_buffer("idft_sin", torch.sin(phase) / self.subcarriers)
        self.register_buffer("dft_cos", torch.cos(phase))
        self.register_buffer("dft_sin", torch.sin(phase))
        frequency_input_channels = 2 * channels + self.tx_antennas + 2 + 1
        self.frequency_stem = CircularConv1d(
            frequency_input_channels,
            int(hidden_channels),
            5,
        )
        dilation_schedule = (1, 2, 4, 8, 16)
        self.frequency_blocks = nn.ModuleList(
            [
                NoiseConditionedResidualBlock(
                    int(hidden_channels),
                    dilation=dilation_schedule[index % len(dilation_schedule)],
                )
                for index in range(max(1, int(depth)))
            ]
        )
        self.frequency_head = nn.Conv1d(int(hidden_channels), channels, 1)
        nn.init.zeros_(self.frequency_head.weight)
        nn.init.zeros_(self.frequency_head.bias)
        delay_hidden = max(24, int(hidden_channels) // 2)
        self.delay_stem = nn.Conv1d(
            2 * channels + 1,
            delay_hidden,
            5,
            padding=2,
        )
        self.delay_blocks = nn.ModuleList(
            [
                DelayResidualBlock(
                    delay_hidden,
                    dilation=dilation_schedule[index % 3],
                )
                for index in range(max(2, int(depth) // 2))
            ]
        )
        self.delay_head = nn.Conv1d(delay_hidden, channels, 1)
        nn.init.zeros_(self.delay_head.weight)
        nn.init.zeros_(self.delay_head.bias)
        shrinkage_hidden = max(8, int(hidden_channels) // 2)
        self.shrinkage = nn.Sequential(
            nn.Linear(1, shrinkage_hidden),
            nn.GELU(),
            nn.Linear(shrinkage_hidden, channels),
        )
        nn.init.zeros_(self.shrinkage[-1].weight)
        nn.init.zeros_(self.shrinkage[-1].bias)
        self.activation = nn.GELU()

    def forward(
        self,
        pilot_ls_ri: torch.Tensor,
        pilot_mask: torch.Tensor,
        ls_estimate_ri: torch.Tensor,
        noise_variance: torch.Tensor,
    ) -> torch.Tensor:
        # [B,R,T,K,2] -> [B,2RT,K]
        batch, rx, tx, subcarriers, _ = ls_estimate_ri.shape
        features = self._flatten_ri(ls_estimate_ri)
        pilot_features = self._flatten_ri(pilot_ls_ri)
        log_noise = torch.log10(torch.clamp(noise_variance, min=1e-12)).reshape(
            batch,
            1,
        )
        noise_plane = log_noise[..., None].expand(-1, 1, subcarriers)
        frequency = self.frequency_features.expand(batch, -1, -1)
        frequency_hidden = self.activation(
            self.frequency_stem(
                torch.cat(
                    [
                        features,
                        pilot_features,
                        pilot_mask,
                        frequency,
                        noise_plane,
                    ],
                    dim=1,
                )
            )
        )
        for block in self.frequency_blocks:
            frequency_hidden = block(frequency_hidden, log_noise)
        frequency_residual = self.frequency_head(frequency_hidden)

        delay_features = self._frequency_to_delay(ls_estimate_ri)
        delay_pilots = self._frequency_to_delay(pilot_ls_ri)
        delay_hidden = self.activation(
            self.delay_stem(
                torch.cat(
                    [delay_features, delay_pilots, noise_plane],
                    dim=1,
                )
            )
        )
        for block in self.delay_blocks:
            delay_hidden = block(delay_hidden, log_noise)
        delay_residual = self._delay_to_frequency(self.delay_head(delay_hidden))

        gain = 1.0 + torch.tanh(self.shrinkage(log_noise))[..., None]
        corrected = gain * features + frequency_residual + delay_residual
        return (
            corrected.reshape(batch, rx, tx, 2, subcarriers)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

    @staticmethod
    def _flatten_ri(value: torch.Tensor) -> torch.Tensor:
        batch, rx, tx, subcarriers, _ = value.shape
        return (
            value.permute(0, 1, 2, 4, 3)
            .reshape(batch, 2 * rx * tx, subcarriers)
        )

    def _frequency_to_delay(self, value: torch.Tensor) -> torch.Tensor:
        batch, rx, tx, subcarriers, _ = value.shape
        links = rx * tx
        real = value[..., 0].reshape(batch, links, subcarriers)
        imag = value[..., 1].reshape(batch, links, subcarriers)
        delay_real = (
            torch.matmul(real, self.idft_cos.transpose(0, 1))
            - torch.matmul(imag, self.idft_sin.transpose(0, 1))
        )
        delay_imag = (
            torch.matmul(real, self.idft_sin.transpose(0, 1))
            + torch.matmul(imag, self.idft_cos.transpose(0, 1))
        )
        return torch.stack([delay_real, delay_imag], dim=2).reshape(
            batch,
            2 * links,
            subcarriers,
        )

    def _delay_to_frequency(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, subcarriers = value.shape
        links = channels // 2
        paired = value.reshape(batch, links, 2, subcarriers)
        delay_real = paired[:, :, 0, :]
        delay_imag = paired[:, :, 1, :]
        frequency_real = (
            torch.matmul(delay_real, self.dft_cos)
            + torch.matmul(delay_imag, self.dft_sin)
        )
        frequency_imag = (
            -torch.matmul(delay_real, self.dft_sin)
            + torch.matmul(delay_imag, self.dft_cos)
        )
        return torch.stack([frequency_real, frequency_imag], dim=2).reshape(
            batch,
            channels,
            subcarriers,
        )


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    rx_antennas: int,
    tx_antennas: int,
    subcarriers: int,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example_h = torch.zeros(
        (2, int(rx_antennas), int(tx_antennas), int(subcarriers), 2),
        dtype=torch.float32,
    )
    example_pilot_h = torch.zeros_like(example_h)
    example_mask = torch.zeros(
        (2, int(tx_antennas), int(subcarriers)),
        dtype=torch.float32,
    )
    spacing = max(1, min(4, int(subcarriers)))
    for tx_index in range(int(tx_antennas)):
        example_mask[:, tx_index, tx_index::spacing] = 1.0
    example_noise = torch.ones((2, 1), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example_pilot_h, example_mask, example_h, example_noise),
        str(destination),
        input_names=[
            "pilot_ls_ri",
            "pilot_mask",
            "ls_estimate_ri",
            "noise_variance",
        ],
        output_names=["h_hat_ri"],
        dynamic_axes={
            "pilot_ls_ri": {0: "batch"},
            "pilot_mask": {0: "batch"},
            "ls_estimate_ri": {0: "batch"},
            "noise_variance": {0: "batch"},
            "h_hat_ri": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return hashlib.sha256(destination.read_bytes()).hexdigest()
