from __future__ import annotations

import hashlib
from pathlib import Path

import torch


class ReferenceModulationCNN1D(torch.nn.Module):
    """Compact cumulant-prior residual classifier for blind-carrier AMC."""

    def __init__(
        self,
        channels: tuple[int, int] = (32, 64),
        class_count: int = 3,
        dropout: float = 0.25,
        analytic_prior_weight: float = 0.35,
    ):
        super().__init__()
        first, second = (int(channels[0]), int(channels[1]))
        self.analytic_prior_weight = float(analytic_prior_weight)
        self.register_buffer(
            "cumulant_prototypes",
            torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.1296, 1.32],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "cumulant_scales",
            torch.tensor([0.22, 0.24, 0.18], dtype=torch.float32),
        )
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(11, first, kernel_size=9, padding=4),
            torch.nn.BatchNorm1d(first),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Conv1d(first, second, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(second),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Conv1d(second, second, kernel_size=9, padding=4),
            torch.nn.BatchNorm1d(second),
            torch.nn.SiLU(),
        )
        self.residual_classifier = torch.nn.Sequential(
            torch.nn.Linear(second * 2 + 11, second),
            torch.nn.SiLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(second, int(class_count)),
        )
        final = self.residual_classifier[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def forward(self, iq_ri: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_components(iq_ri)
        return logits

    def forward_components(
        self,
        iq_ri: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if iq_ri.ndim != 3 or iq_ri.shape[-1] != 2:
            raise ValueError("iq_ri must have shape [batch, sample, 2]")
        iq = iq_ri.float()
        in_phase = iq[..., 0]
        quadrature = iq[..., 1]
        power = in_phase.square() + quadrature.square()
        rms = torch.sqrt(torch.mean(power, dim=1, keepdim=True) + 1e-7)
        in_phase = in_phase / rms
        quadrature = quadrature / rms
        power = in_phase.square() + quadrature.square()

        previous_i = in_phase[:, :-1]
        previous_q = quadrature[:, :-1]
        current_i = in_phase[:, 1:]
        current_q = quadrature[:, 1:]
        differential_real = current_i * previous_i + current_q * previous_q
        differential_imag = current_q * previous_i - current_i * previous_q
        differential_magnitude = torch.sqrt(
            differential_real.square() + differential_imag.square() + 1e-7
        )
        unit_differential_real = differential_real / differential_magnitude
        unit_differential_imag = differential_imag / differential_magnitude
        differential_magnitude = torch.nn.functional.pad(
            differential_magnitude,
            (1, 0),
        )
        unit_differential_real = torch.nn.functional.pad(
            unit_differential_real,
            (1, 0),
        )
        unit_differential_imag = torch.nn.functional.pad(
            unit_differential_imag,
            (1, 0),
        )

        differential2_real = (
            unit_differential_real.square() - unit_differential_imag.square()
        )
        differential2_imag = (
            2.0 * unit_differential_real * unit_differential_imag
        )
        differential4_real = (
            differential2_real.square() - differential2_imag.square()
        )
        differential4_imag = 2.0 * differential2_real * differential2_imag
        temporal_input = torch.stack(
            [
                in_phase,
                quadrature,
                power,
                torch.sqrt(power + 1e-7),
                differential_magnitude,
                unit_differential_real,
                unit_differential_imag,
                differential2_real,
                differential2_imag,
                differential4_real,
                differential4_imag,
            ],
            dim=1,
        )
        temporal = self.features(temporal_input)
        pooled = torch.cat(
            [
                torch.mean(temporal, dim=2),
                torch.amax(temporal, dim=2),
            ],
            dim=1,
        )
        cumulant_features = torch.stack(
            [
                torch.sqrt(
                    torch.mean(differential2_real[:, 1:], dim=1).square()
                    + torch.mean(differential2_imag[:, 1:], dim=1).square()
                    + 1e-7
                ),
                torch.sqrt(
                    torch.mean(differential4_real[:, 1:], dim=1).square()
                    + torch.mean(differential4_imag[:, 1:], dim=1).square()
                    + 1e-7
                ),
                torch.mean(power.square(), dim=1),
            ],
            dim=1,
        )
        analytic_scores = -torch.sum(
            (
                (
                    cumulant_features[:, None, :]
                    - self.cumulant_prototypes[None, :, :]
                )
                / self.cumulant_scales[None, None, :]
            ).square(),
            dim=2,
        )
        analytic_scores = analytic_scores - torch.mean(
            analytic_scores,
            dim=1,
            keepdim=True,
        )
        centered_power = power - torch.mean(power, dim=1, keepdim=True)
        summary = torch.stack(
            [
                torch.mean(power.square(), dim=1),
                torch.mean(power * power.square(), dim=1),
                torch.mean(power.square().square(), dim=1),
                torch.sqrt(
                    torch.mean(centered_power.square(), dim=1) + 1e-7
                ),
                torch.mean(torch.abs(power - 1.0), dim=1),
                *torch.unbind(cumulant_features, dim=1),
                *torch.unbind(analytic_scores, dim=1),
            ],
            dim=1,
        )
        learned_residual = self.residual_classifier(
            torch.cat([pooled, summary], dim=1)
        )
        logits = (
            self.analytic_prior_weight * analytic_scores + learned_residual
        )
        return logits, learned_residual, analytic_scores


def export_onnx_classifier(model: ReferenceModulationCNN1D, path: Path, sample_count: int) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros((4, int(sample_count), 2), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["iq_ri"],
        output_names=["class_logits"],
        dynamic_axes={
            "iq_ri": {0: "batch", 1: "sample"},
            "class_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()
