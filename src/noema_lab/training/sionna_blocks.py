from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Tuple

from noema_lab.training.differentiability import (
    TrainingDependencyError,
    differentiability_spec,
    require_sionna_available,
    require_torch_available,
)

JsonDict = Dict[str, Any]
_SIONNA_RNG_LOCK = threading.RLock()

try:  # Keep importing this module cheap when training dependencies are absent.
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    _torch = None
else:  # Some minimal environments expose a namespace-only torch package.
    if not (hasattr(_torch, "nn") and hasattr(_torch.nn, "Module")):
        _torch = None

_ModuleBase = _torch.nn.Module if _torch is not None else object


class ExportableBlock(_ModuleBase):
    noema_op_id = "training.exportable_block"
    trainable_params = False
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=False,
        exportable=True,
    )

    def __init__(self) -> None:
        if _torch is None:
            require_torch_available()
        super().__init__()

    @property
    def torch(self):
        return require_torch_available() if _torch is None else _torch

    def export_description(self) -> JsonDict:
        payload = {
            "block": self.__class__.__name__,
            "noema_op_id": self.noema_op_id,
            "trainable_params": bool(self.trainable_params),
            "differentiability": dict(self.differentiability),
        }
        contract = getattr(self, "materialization_contract", None)
        if isinstance(contract, dict) and contract:
            payload["materialization_contract"] = dict(contract)
        return payload


class IdentityExportBlock(ExportableBlock):
    noema_op_id = "training.identity"
    differentiability = differentiability_spec(reason="Identity edge used to preserve a typed differentiable-export path.")

    def forward(self, values):
        return values


class ReceiverIqImpairmentBlock(ExportableBlock):
    noema_op_id = "hardware.receiver_iq_imbalance"
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Frozen affine receiver I/Q distortion with an exact Torch gradient.",
    )

    def __init__(
        self,
        *,
        gain_imbalance_db: float = 5.0,
        quadrature_error_deg: float = 12.0,
        phase_offset_deg: float = 20.0,
        dc_offset_i: float = 0.18,
        dc_offset_q: float = -0.12,
    ) -> None:
        super().__init__()
        gain_ratio = 10.0 ** (float(gain_imbalance_db) / 20.0)
        i_gain = math.sqrt(gain_ratio)
        q_gain = 1.0 / i_gain
        quadrature_error = math.radians(float(quadrature_error_deg))
        phase_offset = math.radians(float(phase_offset_deg))
        rotation = self.torch.tensor(
            [
                [math.cos(phase_offset), -math.sin(phase_offset)],
                [math.sin(phase_offset), math.cos(phase_offset)],
            ],
            dtype=self.torch.float32,
        )
        imbalance = self.torch.tensor(
            [
                [i_gain, 0.0],
                [
                    q_gain * math.sin(quadrature_error),
                    q_gain * math.cos(quadrature_error),
                ],
            ],
            dtype=self.torch.float32,
        )
        self.register_buffer("forward_matrix", imbalance @ rotation)
        self.register_buffer(
            "dc_offset",
            self.torch.tensor(
                [float(dc_offset_i), float(dc_offset_q)],
                dtype=self.torch.float32,
            ),
        )
        self.materialization_contract = {
            "gain_imbalance_db": float(gain_imbalance_db),
            "quadrature_error_deg": float(quadrature_error_deg),
            "phase_offset_deg": float(phase_offset_deg),
            "dc_offset_i": float(dc_offset_i),
            "dc_offset_q": float(dc_offset_q),
        }

    def forward(self, symbols):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        features = torch.stack((symbols.real, symbols.imag), dim=-1)
        impaired = torch.matmul(
            features.to(torch.float32),
            self.forward_matrix.transpose(0, 1),
        ) + self.dc_offset
        return torch.complex(impaired[..., 0], impaired[..., 1])


class PowerNormalizationBlock(ExportableBlock):
    noema_op_id = "channel.symbol_power_normalize"
    differentiability = differentiability_spec(reason="Differentiable average-power normalization for complex channel symbols.")

    def __init__(
        self,
        target_power: float = 1.0,
        eps: float = 1e-8,
        normalization_scope: str = "source_item",
    ) -> None:
        super().__init__()
        self.target_power = float(target_power)
        self.eps = float(eps)
        self.normalization_scope = str(normalization_scope or "source_item")
        if self.normalization_scope not in {"source_item", "global"}:
            raise ValueError(
                "normalization_scope must be 'source_item' or 'global'"
            )
        self.materialization_contract = {
            "normalization_scope": self.normalization_scope,
            "target_power": self.target_power,
        }

    def forward(self, symbols):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        if symbols.numel() == 0:
            return symbols
        if self.normalization_scope == "source_item" and symbols.ndim > 1:
            reduction_axes = tuple(range(1, symbols.ndim))
            power = torch.mean(
                torch.abs(symbols) ** 2,
                dim=reduction_axes,
                keepdim=True,
            )
        else:
            power = torch.mean(torch.abs(symbols) ** 2)
        target = torch.as_tensor(
            self.target_power, dtype=torch.float32, device=symbols.device
        )
        scale = torch.sqrt(target / torch.clamp(power.real, min=self.eps))
        return symbols * scale.to(symbols.dtype)


class SymbolPowerAllocatorBlock(ExportableBlock):
    noema_op_id = "model.symbol_power_allocator"
    trainable_params = True
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=True,
        exportable=True,
        reason="Trainable continuous-symbol power allocation before the physical channel.",
    )

    def __init__(
        self,
        snr_db: float = 12.0,
        target_power: float = 1.0,
        min_power: float = 0.25,
        max_power: float = 2.0,
        midpoint_snr_db: float = 12.0,
        slope_db: float = 4.0,
        policy: str = "snr_sigmoid",
        granularity: str = "global",
        budget_mode: str = "fixed_average",
        allocation_contrast: float = 0.6,
        subcarrier_count: int = 64,
        stream_count: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        torch = self.torch
        self.default_snr_db = float(snr_db)
        self.target_power = float(max(0.0, target_power))
        self.min_power = float(max(0.0, min_power))
        self.max_power = float(max(self.min_power, max_power))
        self.policy = str(policy or "snr_sigmoid") if str(policy or "snr_sigmoid") in {"fixed", "snr_sigmoid"} else "snr_sigmoid"
        self.granularity = str(granularity or "global") if str(granularity or "global") in {"global", "per_symbol", "per_subcarrier", "per_stream"} else "global"
        self.budget_mode = str(budget_mode or "fixed_average") if str(budget_mode or "fixed_average") in {"fixed_average", "variable_average"} else "fixed_average"
        self.allocation_contrast = float(max(0.0, allocation_contrast))
        self.subcarrier_count = int(max(1, subcarrier_count))
        self.stream_count = int(max(1, stream_count))
        self.eps = float(eps)
        slope = max(1e-6, abs(float(slope_db)))
        self.snr_weight = torch.nn.Parameter(torch.tensor(1.0 / slope, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(-float(midpoint_snr_db) / slope, dtype=torch.float32))

    def forward(self, symbols, snr_db: Optional[float] = None):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        if symbols.numel() == 0:
            return symbols
        flat = symbols.reshape(-1)
        snr = torch.as_tensor(self.default_snr_db if snr_db is None else float(snr_db), dtype=torch.float32, device=symbols.device)
        alpha = torch.as_tensor(0.5, dtype=torch.float32, device=symbols.device)
        if self.policy != "fixed":
            alpha = torch.sigmoid(snr * self.snr_weight.to(symbols.device) + self.bias.to(symbols.device))
        selected_power = torch.as_tensor(self.target_power, dtype=torch.float32, device=symbols.device)
        if self.budget_mode == "variable_average" and self.policy != "fixed":
            span = torch.as_tensor(self.max_power - self.min_power, dtype=torch.float32, device=symbols.device)
            selected_power = torch.as_tensor(self.max_power, dtype=torch.float32, device=symbols.device) - alpha * span
        targets = self._group_targets(torch, flat.numel(), selected_power, alpha, flat.device)
        allocated = self._apply_group_power(torch, flat, targets).reshape(symbols.shape)
        return allocated.to(symbols.dtype)

    def _group_count(self, symbol_count: int) -> int:
        if self.granularity == "per_symbol":
            return max(1, int(symbol_count))
        if self.granularity == "per_subcarrier":
            return max(1, min(int(self.subcarrier_count), int(symbol_count)))
        if self.granularity == "per_stream":
            return max(1, min(int(self.stream_count), int(symbol_count)))
        return 1

    def _group_targets(self, torch, symbol_count: int, selected_power, alpha, device):
        group_count = self._group_count(symbol_count)
        if group_count == 1 or self.policy == "fixed" or self.allocation_contrast <= 0.0:
            return torch.full((group_count,), selected_power, dtype=torch.float32, device=device)
        positions = torch.linspace(-1.0, 1.0, group_count, dtype=torch.float32, device=device)
        tilt = (alpha - 0.5) * 2.0 * float(self.allocation_contrast)
        weights = torch.exp(tilt * positions)
        weights = weights / torch.clamp(torch.mean(weights), min=self.eps)
        return selected_power * weights

    def _group_ids(self, torch, symbol_count: int, device):
        if self.granularity == "per_subcarrier":
            group_count = self._group_count(symbol_count)
            return torch.arange(symbol_count, dtype=torch.int64, device=device) % int(group_count)
        if self.granularity == "per_stream":
            group_count = self._group_count(symbol_count)
            return torch.arange(symbol_count, dtype=torch.int64, device=device) % int(group_count)
        return torch.zeros((symbol_count,), dtype=torch.int64, device=device)

    def _apply_group_power(self, torch, flat, targets):
        if self.granularity == "per_symbol":
            powers = torch.abs(flat) ** 2
            scale = torch.sqrt(targets / torch.clamp(powers.real, min=self.eps))
            return flat * scale.to(flat.dtype)
        if targets.numel() == 1:
            current_power = torch.mean(torch.abs(flat) ** 2)
            scale = torch.sqrt(targets[0] / torch.clamp(current_power.real, min=self.eps))
            return flat * scale.to(flat.dtype)
        group_ids = self._group_ids(torch, int(flat.numel()), flat.device)
        allocated = torch.empty_like(flat)
        for group_index in range(int(targets.numel())):
            mask = group_ids == int(group_index)
            group_symbols = flat[mask]
            if group_symbols.numel() == 0:
                continue
            group_power = torch.mean(torch.abs(group_symbols) ** 2)
            scale = torch.sqrt(targets[group_index] / torch.clamp(group_power.real, min=self.eps))
            allocated[mask] = group_symbols * scale.to(group_symbols.dtype)
        return allocated


class AwgnChannelBlock(ExportableBlock):
    noema_op_id = "wireless.channel"
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Native Torch AWGN uses a differentiable reparameterized local-RNG path.",
    )

    def __init__(
        self,
        snr_db: float = 12.0,
        backend: str = "torch",
        seed: Optional[int] = None,
        receiver_processing: str = "matched",
    ) -> None:
        resolved_backend = str(backend or "torch")
        if resolved_backend != "torch":
            raise TrainingDependencyError(
                "Use SionnaAwgnChannelBlock for the Sionna 2.x/PyTorch materialization."
            )
        super().__init__()
        self.snr_db = float(snr_db)
        self.backend = resolved_backend
        self.seed = seed
        self.receiver_processing = str(receiver_processing or "matched")
        if self.receiver_processing not in {"matched", "none"}:
            raise ValueError(
                "receiver_processing must be 'matched' or 'none'"
            )
        self.materialization_contract = {
            "channel": "awgn",
            "backend": self.backend,
            "receiver_processing": self.receiver_processing,
            "seed": self.seed,
        }

    def forward(self, symbols, snr_db: Optional[float] = None):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        noise_variance = _noise_variance(torch, symbols, self.snr_db if snr_db is None else float(snr_db))
        return symbols + _complex_awgn_noise(torch, symbols, noise_variance, self.seed)


class SionnaAwgnChannelBlock(AwgnChannelBlock):
    noema_op_id = "wireless.channel"
    differentiability = differentiability_spec(
        framework="sionna",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Sionna 2.x AWGN is a PyTorch module and preserves gradients to transmitted symbols.",
    )

    def __init__(
        self,
        snr_db: float = 12.0,
        backend: str = "sionna",
        seed: Optional[int] = None,
        receiver_processing: str = "matched",
    ) -> None:
        require_sionna_available()
        ExportableBlock.__init__(self)
        self.snr_db = float(snr_db)
        self.backend = "sionna"
        self.seed = seed
        self.receiver_processing = str(receiver_processing or "matched")
        if self.receiver_processing not in {"matched", "none"}:
            raise ValueError("receiver_processing must be 'matched' or 'none'")
        self.materialization_contract = {
            "channel": "awgn",
            "backend": "sionna",
            "runtime": "sionna>=2.0.1/pytorch",
            "receiver_processing": self.receiver_processing,
            "seed": self.seed,
        }

    def forward(self, symbols, snr_db: Optional[float] = None):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        noise_variance = _noise_variance(
            torch,
            symbols,
            self.snr_db if snr_db is None else float(snr_db),
        )
        return _sionna_awgn(symbols, noise_variance, seed=self.seed)


class FlatRayleighChannelBlock(ExportableBlock):
    noema_op_id = "wireless.channel"
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Flat Rayleigh fading is differentiable with respect to the transmitted symbols for fixed sampled channel/noise.",
    )

    def __init__(
        self,
        snr_db: float = 12.0,
        receiver_processing: str = "matched",
        seed: Optional[int] = None,
        *,
        equalize: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.snr_db = float(snr_db)
        if equalize is not None:
            receiver_processing = "matched" if bool(equalize) else "none"
        self.receiver_processing = str(receiver_processing or "matched")
        if self.receiver_processing not in {"matched", "none"}:
            raise ValueError(
                "receiver_processing must be 'matched' or 'none'"
            )
        self.equalize = self.receiver_processing == "matched"
        self.seed = seed
        self.materialization_contract = {
            "channel": "flat_rayleigh",
            "backend": "torch",
            "receiver_processing": self.receiver_processing,
            "seed": self.seed,
        }

    def forward(self, symbols, snr_db: Optional[float] = None):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        noise_variance = _noise_variance(torch, symbols, self.snr_db if snr_db is None else float(snr_db))
        h = _complex_normal(torch, symbols.shape, symbols.device, self.seed, symbols.dtype) / math.sqrt(2.0)
        faded = h * symbols + _complex_awgn_noise(torch, symbols, noise_variance, None if self.seed is None else self.seed + 1)
        if self.equalize:
            return faded / torch.where(torch.abs(h) < 1e-6, torch.full_like(h, 1e-6 + 0j), h)
        return faded


class SionnaFlatFadingChannelBlock(FlatRayleighChannelBlock):
    noema_op_id = "wireless.channel"
    differentiability = differentiability_spec(
        framework="sionna",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Sionna 2.x flat fading is a PyTorch module and preserves gradients to transmitted symbols.",
    )

    def __init__(
        self,
        snr_db: float = 12.0,
        receiver_processing: str = "matched",
        seed: Optional[int] = None,
        **_unused: Any,
    ) -> None:
        require_sionna_available()
        ExportableBlock.__init__(self)
        self.snr_db = float(snr_db)
        self.receiver_processing = str(receiver_processing or "matched")
        if self.receiver_processing not in {"matched", "none"}:
            raise ValueError("receiver_processing must be 'matched' or 'none'")
        self.equalize = self.receiver_processing == "matched"
        self.seed = seed
        self.materialization_contract = {
            "channel": "flat_rayleigh",
            "backend": "sionna",
            "runtime": "sionna>=2.0.1/pytorch",
            "receiver_processing": self.receiver_processing,
            "seed": self.seed,
        }

    def forward(self, symbols, snr_db: Optional[float] = None):
        torch = self.torch
        symbols = _as_symbol_tensor(symbols, torch)
        noise_variance = _noise_variance(
            torch,
            symbols,
            self.snr_db if snr_db is None else float(snr_db),
        )
        return _sionna_flat_fading(
            symbols,
            noise_variance,
            equalize=self.equalize,
            seed=self.seed,
        )


class QamPamMapperBlock(ExportableBlock):
    noema_op_id = "modulation.digital_modulate"
    differentiability = differentiability_spec(
        framework="torch",
        gradient="stop",
        trainable_params=False,
        exportable=True,
        reason="Bit grouping and hard constellation lookup are discrete; gradients do not flow to input bits.",
    )

    def __init__(self, modulation: str = "qpsk", normalize_power: bool = True) -> None:
        super().__init__()
        self.modulation = _normalize_modulation(modulation)
        self.normalize_power = bool(normalize_power)

    def forward(self, bits):
        torch = self.torch
        constellation, labels = _constellation(torch, self.modulation, device=bits.device if hasattr(bits, "device") else None)
        bits = torch.as_tensor(bits, dtype=torch.int64, device=constellation.device).reshape(-1)
        bits_per_symbol = labels.shape[1]
        pad = (-int(bits.numel())) % int(bits_per_symbol)
        if pad:
            bits = torch.cat([bits, torch.zeros(pad, dtype=bits.dtype, device=bits.device)])
        grouped = bits.reshape(-1, bits_per_symbol)
        matches = (grouped[:, None, :] == labels[None, :, :]).all(dim=-1)
        indices = torch.argmax(matches.to(torch.int64), dim=-1)
        symbols = constellation[indices]
        if self.normalize_power:
            symbols = PowerNormalizationBlock()(symbols)
        return symbols


class SionnaMapperBlock(QamPamMapperBlock):
    noema_op_id = "modulation.digital_modulate"
    differentiability = differentiability_spec(
        framework="sionna",
        gradient="stop",
        trainable_params=False,
        exportable=True,
        reason="Sionna 2.x mapping is a PyTorch module; hard bit grouping remains discrete.",
    )

    def __init__(self, modulation: str = "qpsk", normalize_power: bool = True) -> None:
        require_sionna_available()
        ExportableBlock.__init__(self)
        self.modulation = _normalize_modulation(modulation)
        self.normalize_power = bool(normalize_power)
        self.materialization_contract = {
            "backend": "sionna",
            "runtime": "sionna>=2.0.1/pytorch",
            "modulation": self.modulation,
            "normalize_power": self.normalize_power,
        }

    def forward(self, bits):
        try:
            return _sionna_mapper(bits, self.modulation, self.normalize_power, self.torch)
        except TrainingDependencyError:
            raise
        except Exception as exc:
            raise TrainingDependencyError(
                "Sionna mapper training block could not run with the installed Sionna version: %s" % exc
            ) from exc


class SoftDemapperBlock(ExportableBlock):
    noema_op_id = "demodulation.soft_demodulate"
    differentiability = differentiability_spec(
        framework="torch",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Max-log soft demapping is differentiable with respect to received complex symbols except at distance ties.",
    )

    def __init__(self, modulation: str = "qpsk", noise_variance: float = 1.0) -> None:
        super().__init__()
        self.modulation = _normalize_modulation(modulation)
        self.noise_variance = float(noise_variance)

    def forward(self, rx_symbols, noise_variance: Optional[float] = None):
        torch = self.torch
        rx_symbols = _as_symbol_tensor(rx_symbols, torch).reshape(-1)
        constellation, labels = _constellation(torch, self.modulation, device=rx_symbols.device)
        distances = torch.abs(rx_symbols[:, None] - constellation[None, :]) ** 2
        variance = max(float(self.noise_variance if noise_variance is None else noise_variance), 1e-12)
        llrs = []
        for bit_index in range(labels.shape[1]):
            mask0 = labels[:, bit_index] == 0
            mask1 = ~mask0
            d0 = torch.min(distances[:, mask0], dim=1).values
            d1 = torch.min(distances[:, mask1], dim=1).values
            llrs.append((d1 - d0) / variance)
        return torch.stack(llrs, dim=-1).reshape(-1)


class SionnaSoftDemapperBlock(SoftDemapperBlock):
    noema_op_id = "demodulation.soft_demodulate"
    differentiability = differentiability_spec(
        framework="sionna",
        gradient="full",
        trainable_params=False,
        exportable=True,
        reason="Sionna 2.x soft demapping is differentiable with respect to received PyTorch symbols.",
    )

    def __init__(self, modulation: str = "qpsk", noise_variance: float = 1.0) -> None:
        require_sionna_available()
        ExportableBlock.__init__(self)
        self.modulation = _normalize_modulation(modulation)
        self.noise_variance = float(noise_variance)
        self.materialization_contract = {
            "backend": "sionna",
            "runtime": "sionna>=2.0.1/pytorch",
            "modulation": self.modulation,
            "noise_variance": self.noise_variance,
        }

    def forward(self, rx_symbols, noise_variance: Optional[float] = None):
        try:
            return _sionna_soft_demapper(
                rx_symbols,
                self.modulation,
                self.noise_variance if noise_variance is None else float(noise_variance),
                self.torch,
            )
        except TrainingDependencyError:
            raise
        except Exception as exc:
            raise TrainingDependencyError(
                "Sionna soft demapper training block could not run with the installed Sionna version: %s" % exc
            ) from exc


def _as_symbol_tensor(values, torch):
    tensor = values if hasattr(values, "dtype") else torch.as_tensor(values)
    if tensor.dtype.is_complex:
        return tensor
    tensor = tensor.to(torch.float32)
    if tensor.shape and tensor.shape[-1] == 2:
        return torch.view_as_complex(tensor.contiguous())
    return tensor.to(torch.complex64)


def _noise_variance(torch, symbols, snr_db: float):
    value = 10.0 ** (-float(snr_db) / 10.0)
    return torch.as_tensor(value, dtype=torch.float32, device=symbols.device)


def _complex_awgn_noise(torch, symbols, noise_variance, seed: Optional[int]):
    return _complex_normal(torch, symbols.shape, symbols.device, seed, symbols.dtype) * torch.sqrt(noise_variance / 2.0).to(symbols.dtype)


def _complex_normal(torch, shape: Tuple[int, ...], device, seed: Optional[int], dtype):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    real = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    imag = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    return (real + 1j * imag).to(dtype)


def _normalize_modulation(modulation: str) -> str:
    value = str(modulation or "qpsk").lower().replace("-", "")
    if value in {"pam2", "bpsk"}:
        return "bpsk"
    if value in {"qpsk", "qam4"}:
        return "qpsk"
    if value in {"qam16", "16qam"}:
        return "qam16"
    raise ValueError("Unsupported differentiable-export modulation: %s" % modulation)


def _constellation(torch, modulation: str, device=None):
    modulation = _normalize_modulation(modulation)
    if modulation == "bpsk":
        constellation = torch.tensor([1 + 0j, -1 + 0j], dtype=torch.complex64, device=device)
        labels = torch.tensor([[0], [1]], dtype=torch.int64, device=device)
        return constellation, labels
    if modulation == "qpsk":
        scale = 1.0 / math.sqrt(2.0)
        constellation = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex64, device=device) * scale
        labels = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64, device=device)
        return constellation, labels
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32, device=device)
    bit_pairs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64, device=device)
    points = []
    labels = []
    for i_index in range(4):
        for q_index in range(4):
            points.append(levels[i_index] + 1j * levels[q_index])
            labels.append(torch.cat([bit_pairs[i_index], bit_pairs[q_index]]))
    constellation = torch.stack(points).to(torch.complex64) / math.sqrt(10.0)
    return constellation, torch.stack(labels).to(torch.int64)


def _sionna_awgn(symbols, noise_variance, *, seed: Optional[int] = None):
    require_sionna_available()
    try:
        from sionna.phy.channel import AWGN  # type: ignore
        from sionna.phy import config as sionna_config  # type: ignore
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna 2.x is installed, but its PyTorch AWGN block could not be imported."
        ) from exc
    try:
        with _SIONNA_RNG_LOCK:
            if seed is not None:
                sionna_config.seed = int(seed)
            layer = AWGN(device=str(symbols.device))
            return layer(symbols, noise_variance)
    except Exception as exc:
        raise TrainingDependencyError("Sionna AWGN training block could not run with the installed Sionna/PyTorch stack: %s" % exc) from exc


def _sionna_constellation(modulation: str, *, device=None):
    require_sionna_available()
    torch = require_torch_available()
    try:
        from sionna.phy.mapping import Constellation  # type: ignore
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna 2.x is installed, but its mapping.Constellation class could not be imported."
        ) from exc
    # Sionna's built-in QAM uses the 3GPP Gray point order, while Noema's
    # artifact/runtime contract uses its own label-to-point order (notably for
    # 16-QAM). A custom constellation makes the implicit binary index labels
    # exactly match Noema's canonical points for every supported modulation.
    points, _labels = _constellation(torch, modulation, device=device)
    bits_per_symbol = _bits_per_symbol(modulation)
    return Constellation(
        constellation_type="custom",
        num_bits_per_symbol=bits_per_symbol,
        points=points,
        normalize=False,
        center=False,
        device=None if device is None else str(device),
    )


def _sionna_mapper(bits, modulation: str, normalize_power: bool, torch):
    require_sionna_available()
    try:
        from sionna.phy.mapping import Mapper  # type: ignore
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna 2.x is installed, but its mapping.Mapper class could not be imported."
        ) from exc
    device = bits.device if hasattr(bits, "device") else None
    constellation = _sionna_constellation(modulation, device=device)
    bits = torch.as_tensor(
        bits,
        dtype=torch.float32,
        device=constellation.points.device,
    ).reshape(-1)
    bits_per_symbol = _bits_per_symbol(modulation)
    pad = (-int(bits.numel())) % bits_per_symbol
    if pad:
        bits = torch.cat(
            [bits, torch.zeros(pad, dtype=bits.dtype, device=bits.device)]
        )
    mapper = Mapper(constellation=constellation, device=str(bits.device))
    symbols = mapper(bits)
    symbols = _torch_tensor_from_sionna(symbols, torch, require_grad_safe=False).reshape(-1).to(torch.complex64)
    if normalize_power:
        symbols = PowerNormalizationBlock()(symbols)
    return symbols


def _sionna_soft_demapper(rx_symbols, modulation: str, noise_variance: float, torch):
    require_sionna_available()
    try:
        from sionna.phy.mapping import Demapper  # type: ignore
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna 2.x is installed, but its mapping.Demapper class could not be imported."
        ) from exc
    rx_symbols = _as_symbol_tensor(rx_symbols, torch).reshape(-1)
    constellation = _sionna_constellation(
        modulation,
        device=rx_symbols.device,
    )
    no = torch.as_tensor(float(noise_variance), dtype=torch.float32, device=rx_symbols.device)
    try:
        demapper = Demapper(
            "maxlog",
            constellation=constellation,
            device=str(rx_symbols.device),
        )
        llr = demapper(rx_symbols, no)
    except Exception as exc:
        raise TrainingDependencyError("Sionna Demapper could not run: %s" % exc) from exc
    # Sionna defines LLR as log P(bit=1)/P(bit=0). Noema's modem contract,
    # including ``bits = llr < 0``, uses the opposite sign convention.
    return -_torch_tensor_from_sionna(
        llr,
        torch,
        require_grad_safe=True,
    ).reshape(-1).to(torch.float32)


def _sionna_flat_fading(
    symbols,
    noise_variance,
    *,
    equalize: bool,
    seed: Optional[int] = None,
):
    require_sionna_available()
    torch = require_torch_available()
    try:
        from sionna.phy.channel import FlatFadingChannel  # type: ignore
        from sionna.phy import config as sionna_config  # type: ignore
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna 2.x is installed, but FlatFadingChannel could not be imported."
        ) from exc
    symbols = _as_symbol_tensor(symbols, torch).reshape(-1)
    x = symbols.reshape(-1, 1)
    no = torch.as_tensor(
        noise_variance,
        dtype=torch.float32,
        device=symbols.device,
    )
    try:
        with _SIONNA_RNG_LOCK:
            if seed is not None:
                sionna_config.seed = int(seed)
            channel = FlatFadingChannel(
                num_tx_ant=1,
                num_rx_ant=1,
                return_channel=True,
                device=str(symbols.device),
            )
            y, h = channel(x, no)
    except Exception as exc:
        raise TrainingDependencyError(
            "Sionna flat-fading training block could not run: %s" % exc
        ) from exc
    y = _torch_tensor_from_sionna(y, torch, require_grad_safe=True).reshape(-1).to(torch.complex64)
    h = _torch_tensor_from_sionna(h, torch, require_grad_safe=True).reshape(-1).to(torch.complex64)
    if equalize:
        h_safe = torch.where(torch.abs(h) < 1e-6, torch.full_like(h, 1e-6 + 0j), h)
        return y / h_safe
    return y


def _torch_tensor_from_sionna(value, torch, *, require_grad_safe: bool):
    if hasattr(value, "detach"):
        return value
    if hasattr(value, "numpy"):
        if require_grad_safe:
            raise TrainingDependencyError(
                "The installed Sionna block returned a non-PyTorch tensor; this would break PyTorch gradients. "
                "Use a PyTorch-compatible Sionna version or the pure Torch backend."
            )
        return torch.as_tensor(value.numpy())
    return torch.as_tensor(value)


def _bits_per_symbol(modulation: str) -> int:
    modulation = _normalize_modulation(modulation)
    if modulation == "bpsk":
        return 1
    if modulation == "qpsk":
        return 2
    if modulation == "qam16":
        return 4
    raise ValueError("Unsupported differentiable-export modulation: %s" % modulation)
