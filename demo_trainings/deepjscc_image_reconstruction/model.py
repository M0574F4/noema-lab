from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import torch


ARCHITECTURE = "nested_bandwidth_blind_csi_residual_cnn_v4"


class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.relu(values + 0.1 * self.body(values))


class _BlindComplexGainNormalizer(torch.nn.Module):
    """Estimate one complex correction per image from received-symbol statistics."""

    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.estimator = torch.nn.Sequential(
            torch.nn.Linear(6, hidden_channels),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(hidden_channels, 2),
        )
        # Start from the identity correction. Training only changes it when the
        # received codeword statistics contain useful evidence about the fade.
        torch.nn.init.zeros_(self.estimator[-1].weight)
        torch.nn.init.zeros_(self.estimator[-1].bias)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduce_dims = tuple(range(1, real.ndim))
        summary = torch.stack(
            [
                torch.mean(real, dim=reduce_dims),
                torch.mean(imag, dim=reduce_dims),
                torch.mean(real * real, dim=reduce_dims),
                torch.mean(imag * imag, dim=reduce_dims),
                torch.mean(real * imag, dim=reduce_dims),
                torch.mean(real * real + imag * imag, dim=reduce_dims),
            ],
            dim=1,
        )
        residual = self.estimator(summary)
        correction_real = 1.0 + residual[:, 0]
        correction_imag = residual[:, 1]
        view_shape = (real.shape[0],) + (1,) * (real.ndim - 1)
        correction_real = correction_real.reshape(view_shape)
        correction_imag = correction_imag.reshape(view_shape)
        corrected_real = (
            real * correction_real - imag * correction_imag
        )
        corrected_imag = (
            real * correction_imag + imag * correction_real
        )
        return corrected_real, corrected_imag


class ReferenceDeepJSCCModel(torch.nn.Module):
    """CPU-sized residual DeepJSCC model used by the exported demonstration."""

    def __init__(self, symbol_channels: int = 32):
        super().__init__()
        self.symbol_channels = int(symbol_channels)
        if self.symbol_channels < 1:
            raise ValueError("symbol_channels must be positive")
        self.blind_gain_normalizer = _BlindComplexGainNormalizer()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=False),
            _ResidualBlock(32),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=False),
            _ResidualBlock(64),
            torch.nn.Conv2d(64, self.symbol_channels * 2, 3, stride=2, padding=1),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.symbol_channels * 2, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=False),
            _ResidualBlock(64),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=False),
            _ResidualBlock(32),
            torch.nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )

    def encode(
        self,
        images: torch.Tensor,
        active_symbol_channels: int | None = None,
    ) -> torch.Tensor:
        features = self.encoder(images.float())
        real, imag = torch.chunk(features, 2, dim=1)
        active = (
            self.symbol_channels
            if active_symbol_channels is None
            else int(active_symbol_channels)
        )
        if active < 1 or active > self.symbol_channels:
            raise ValueError(
                "active_symbol_channels must be within [1, %d]"
                % self.symbol_channels
            )
        real = real[:, :active]
        imag = imag[:, :active]
        return torch.complex(real, imag)

    def decode(self, rx_symbols: torch.Tensor) -> torch.Tensor:
        if not rx_symbols.dtype.is_complex:
            raise ValueError("DeepJSCC decoder expects a complex tensor")
        active = int(rx_symbols.shape[1])
        if active < 1 or active > self.symbol_channels:
            raise ValueError(
                "received symbol channels must be within [1, %d]"
                % self.symbol_channels
            )
        if active < self.symbol_channels:
            missing = self.symbol_channels - active
            padding = torch.zeros(
                (
                    int(rx_symbols.shape[0]),
                    missing,
                    int(rx_symbols.shape[2]),
                    int(rx_symbols.shape[3]),
                ),
                dtype=rx_symbols.dtype,
                device=rx_symbols.device,
            )
            rx_symbols = torch.cat([rx_symbols, padding], dim=1)
        real, imag = self.blind_gain_normalizer(
            rx_symbols.real, rx_symbols.imag
        )
        features = torch.cat([real, imag], dim=1)
        return self.decoder(features)


class _EncoderOnnxView(torch.nn.Module):
    def __init__(
        self,
        model: ReferenceDeepJSCCModel,
        active_symbol_channels: int,
    ):
        super().__init__()
        self.model = model
        self.active_symbol_channels = int(active_symbol_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # The portable slot ABI represents C complex channels as 2C real channels.
        features = self.model.encoder(images.float())
        real, imag = torch.chunk(features, 2, dim=1)
        return torch.cat(
            [
                real[:, : self.active_symbol_channels],
                imag[:, : self.active_symbol_channels],
            ],
            dim=1,
        )


class _DecoderOnnxView(torch.nn.Module):
    def __init__(
        self,
        model: ReferenceDeepJSCCModel,
        active_symbol_channels: int,
    ):
        super().__init__()
        self.model = model
        self.active_symbol_channels = int(active_symbol_channels)

    def forward(self, symbols_ri: torch.Tensor) -> torch.Tensor:
        values = symbols_ri.float()
        real = values[:, : self.active_symbol_channels]
        imag = values[
            :,
            self.active_symbol_channels : 2 * self.active_symbol_channels,
        ]
        if self.active_symbol_channels < self.model.symbol_channels:
            padding_shape = (
                values.shape[0],
                self.model.symbol_channels - self.active_symbol_channels,
                values.shape[2],
                values.shape[3],
            )
            padding = torch.zeros(
                padding_shape,
                dtype=values.dtype,
                device=values.device,
            )
            real = torch.cat([real, padding], dim=1)
            imag = torch.cat([imag, padding], dim=1)
        real, imag = self.model.blind_gain_normalizer(real, imag)
        return self.model.decoder(torch.cat([real, imag], dim=1))


def export_onnx_components(
    model: ReferenceDeepJSCCModel,
    encoder_path: str | Path,
    decoder_path: str | Path,
    *,
    example_image_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    active_symbol_channels: int | None = None,
) -> Dict[str, str]:
    """Export complete executable encoder/decoder graphs for the neutral slot ABI."""

    encoder_destination = Path(encoder_path)
    decoder_destination = Path(decoder_path)
    encoder_destination.parent.mkdir(parents=True, exist_ok=True)
    decoder_destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    active = (
        model.symbol_channels
        if active_symbol_channels is None
        else int(active_symbol_channels)
    )
    if active < 1 or active > model.symbol_channels:
        raise ValueError("active_symbol_channels is outside the trained model")
    example_images = torch.zeros(example_image_shape, dtype=torch.float32)
    with torch.no_grad():
        example_complex_symbols = model.encode(
            example_images,
            active_symbol_channels=active,
        )
        example_symbols = torch.cat(
            [example_complex_symbols.real, example_complex_symbols.imag],
            dim=1,
        )
    torch.onnx.export(
        _EncoderOnnxView(model, active),
        (example_images,),
        str(encoder_destination),
        input_names=["images"],
        output_names=["symbols_ri"],
        dynamic_axes={
            "images": {0: "batch", 2: "image_height", 3: "image_width"},
            "symbols_ri": {0: "batch", 2: "symbol_height", 3: "symbol_width"},
        },
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        _DecoderOnnxView(model, active),
        (example_symbols,),
        str(decoder_destination),
        input_names=["symbols_ri"],
        output_names=["reconstruction"],
        dynamic_axes={
            "symbols_ri": {0: "batch", 2: "symbol_height", 3: "symbol_width"},
            "reconstruction": {0: "batch", 2: "image_height", 3: "image_width"},
        },
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
