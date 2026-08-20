from __future__ import annotations

from typing import Any, Mapping

import torch

from model import ReferenceDeepJSCCModel


class AveragePowerNormalization(torch.nn.Module):
    """Match Noema's canonical average complex-symbol power constraint."""

    def __init__(self, target_power: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.target_power = float(target_power)
        self.eps = float(eps)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, symbols.ndim))
        power = torch.mean(
            torch.abs(symbols) ** 2,
            dim=reduce_dims,
            keepdim=True,
        )
        target = torch.as_tensor(self.target_power, dtype=torch.float32, device=symbols.device)
        scale = torch.sqrt(target / torch.clamp(power.real, min=self.eps))
        return symbols * scale.to(symbols.dtype)


class DifferentiableComplexAwgn(torch.nn.Module):
    """Frozen reparameterized AWGN channel with gradients to transmitted symbols."""

    def forward(self, symbols: torch.Tensor, snr_db: float | torch.Tensor) -> torch.Tensor:
        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=symbols.device).reshape(())
        noise_variance = torch.pow(torch.as_tensor(10.0, device=symbols.device), -snr / 10.0)
        real = torch.randn(symbols.shape, dtype=torch.float32, device=symbols.device)
        imag = torch.randn(symbols.shape, dtype=torch.float32, device=symbols.device)
        noise = torch.complex(real, imag).to(symbols.dtype)
        noise = noise * torch.sqrt(noise_variance / 2.0).to(symbols.dtype)
        return symbols + noise


class DifferentiableSlowRayleigh(torch.nn.Module):
    """One unknown complex fading coefficient per image, constant over its codeword."""

    def forward(
        self,
        symbols: torch.Tensor,
        snr_db: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        snr = torch.as_tensor(
            snr_db, dtype=torch.float32, device=symbols.device
        ).reshape(())
        noise_variance = torch.pow(
            torch.as_tensor(10.0, device=symbols.device), -snr / 10.0
        )
        gain_shape = (int(symbols.shape[0]),) + (1,) * (
            symbols.ndim - 1
        )
        gain = torch.complex(
            torch.randn(
                gain_shape, dtype=torch.float32, device=symbols.device
            ),
            torch.randn(
                gain_shape, dtype=torch.float32, device=symbols.device
            ),
        ).to(symbols.dtype) / torch.sqrt(
            torch.as_tensor(2.0, device=symbols.device)
        ).to(symbols.dtype)
        noise = torch.complex(
            torch.randn(
                symbols.shape, dtype=torch.float32, device=symbols.device
            ),
            torch.randn(
                symbols.shape, dtype=torch.float32, device=symbols.device
            ),
        ).to(symbols.dtype)
        noise = noise * torch.sqrt(noise_variance / 2.0).to(symbols.dtype)
        return gain * symbols + noise, gain


class NoemaDeepJSCCScenario(torch.nn.Module):
    """Trainable endpoint slots around the frozen recipe-derived differentiable path."""

    def __init__(self, config: Mapping[str, Any], model: ReferenceDeepJSCCModel | None = None):
        super().__init__()
        model_config = dict(config.get("model") or {})
        channel_config = dict(config.get("channel") or {})
        power_config = dict(config.get("symbol_power") or {})
        self.channel_type = str(channel_config.get("type") or "awgn")
        if self.channel_type not in {"awgn", "flat_rayleigh"}:
            raise ValueError(
                "DeepJSCC training supports awgn or flat_rayleigh"
            )
        if self.channel_type == "flat_rayleigh":
            if (
                str(channel_config.get("receiver_processing") or "matched")
                != "none"
            ):
                raise ValueError(
                    "Blind slow-Rayleigh training requires receiver_processing=none"
                )
            if (
                str(channel_config.get("fading_scope") or "symbol")
                != "source_item"
            ):
                raise ValueError(
                    "Blind slow-Rayleigh training requires fading_scope=source_item"
                )
        self.model = model or ReferenceDeepJSCCModel(
            symbol_channels=int(model_config.get("symbol_channels") or 32)
        )
        self.snr_values = tuple(float(value) for value in channel_config.get("snr_db") or [12.0])
        if not self.snr_values:
            raise ValueError("channel.snr_db must contain at least one value")
        self.default_snr_db = float(channel_config.get("default_snr_db", self.snr_values[0]))
        self.use_power_normalization = bool(power_config.get("enabled", False))
        self.power_normalize = AveragePowerNormalization(
            target_power=float(power_config.get("target_power", 1.0)),
            eps=float(power_config.get("eps", 1e-8)),
        ) if self.use_power_normalization else torch.nn.Identity()
        self.channel = (
            DifferentiableComplexAwgn()
            if self.channel_type == "awgn"
            else DifferentiableSlowRayleigh()
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        snr_db: float | torch.Tensor | None = None,
        active_symbol_channels: int | None = None,
    ):
        images = batch["image"].float()
        selected_snr = self.default_snr_db if snr_db is None else snr_db
        symbols_before_power = self.model.encode(
            images,
            active_symbol_channels=active_symbol_channels,
        )
        symbols = self.power_normalize(symbols_before_power)
        if self.channel_type == "flat_rayleigh":
            rx_symbols, channel_gain = self.channel(symbols, selected_snr)
        else:
            rx_symbols = self.channel(symbols, selected_snr)
            gain_shape = (int(symbols.shape[0]),) + (1,) * (
                symbols.ndim - 1
            )
            channel_gain = torch.ones(
                gain_shape,
                dtype=symbols.dtype,
                device=symbols.device,
            )
        reconstruction = self.model.decode(rx_symbols)
        noise_variance = torch.pow(
            torch.as_tensor(10.0, dtype=torch.float32, device=images.device),
            -torch.as_tensor(selected_snr, dtype=torch.float32, device=images.device) / 10.0,
        )
        return {
            "source_image": images,
            "reconstruction": reconstruction,
            "symbols_before_power": symbols_before_power,
            "symbols": symbols,
            "rx_symbols": rx_symbols,
            # This is diagnostic evidence only. The decoder receives rx_symbols
            # and is never given channel_gain or a pilot estimate.
            "channel_gain": channel_gain,
            "snr_db": torch.as_tensor(selected_snr, dtype=torch.float32, device=images.device),
            "noise_variance": noise_variance,
            "average_symbol_power": torch.mean(torch.abs(symbols) ** 2).real,
        }


def build_scenario(config: Mapping[str, Any], model: ReferenceDeepJSCCModel | None = None):
    return NoemaDeepJSCCScenario(config, model=model)
