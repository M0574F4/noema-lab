from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class TrainingTap:
    step_id: str
    output_name: str
    kind: str

    def to_dict(self) -> JsonDict:
        return {"step_id": self.step_id, "output_name": self.output_name, "kind": self.kind}


@dataclass(frozen=True)
class TrainingScenario:
    name: str
    mode: str
    taps: List[TrainingTap] = field(default_factory=list)
    objective: str = ""
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "mode": self.mode,
            "objective": self.objective,
            "taps": [tap.to_dict() for tap in self.taps],
            "metadata": dict(self.metadata),
        }


def deepjscc_awgn_scenario(snr_db: float = 12.0) -> TrainingScenario:
    return TrainingScenario(
        name="deepjscc_awgn",
        mode="differentiable_export",
        objective="image_or_semantic_reconstruction_loss",
        taps=[
            TrainingTap("sender", "symbols", "channel.symbols.complex_tensor"),
            TrainingTap("wireless_channel", "rx_symbols", "channel.rx_symbols.complex_tensor"),
            TrainingTap("receiver", "reconstruction", "task.output.tensor"),
        ],
        metadata={"channel": "awgn", "snr_db": float(snr_db)},
    )


def neural_receiver_capture_scenario() -> TrainingScenario:
    return TrainingScenario(
        name="neural_receiver_capture",
        mode="dataset_capture",
        objective="rx_symbols_to_target_bits",
        taps=[
            TrainingTap("wireless_channel", "rx_symbols", "channel.rx_symbols.complex_numpy"),
            TrainingTap("tx_bit_boundary", "bits", "channel.bits.numpy"),
        ],
        metadata={"target": "transmitted_bits"},
    )
