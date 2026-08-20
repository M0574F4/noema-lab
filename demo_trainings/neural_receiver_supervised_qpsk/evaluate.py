from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from datamodule import held_out_split_integrity_report, load_capture_dataset
from frontend import compensate_receiver_iq, load_receiver_iq_calibration
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact held-out evaluation summary instead of the full JSON payload.",
    )
    args = parser.parse_args(argv)
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "trained_artifact.yaml")
    )
    manifest = load_strict_yaml_or_json(manifest_path)
    if int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("trained artifact must use schema_version: 2")
    component = dict((manifest.get("components") or [])[0])
    component_path = manifest_path.parent / str(component.get("path") or "")
    actual_sha = hashlib.sha256(component_path.read_bytes()).hexdigest()
    if actual_sha != str(component.get("sha256") or ""):
        raise ValueError("neural-receiver component SHA-256 does not match the manifest")
    test_data = load_capture_dataset(
        data.get("test_capture_dirs") or [],
        feature_tap=str(data.get("feature_tap") or "rx_symbols"),
        target_tap=str(data.get("target_tap") or "target_bits"),
        expected_split="test",
    )
    held_out_integrity = held_out_split_integrity_report(
        test_data,
        dict(dict(manifest.get("training") or {}).get("split_integrity") or {}),
    )
    session = ort.InferenceSession(
        str(component_path), providers=["CPUExecutionProvider"]
    )
    llr = np.asarray(
        session.run(
            ["bit_llr"],
            {"rx_symbols_ri": test_data.features.astype(np.float32, copy=False)},
        )[0],
        dtype=np.float32,
    )
    targets = test_data.target_bits.astype(np.float32, copy=False)
    decisions = llr < 0.0
    target_bool = targets.astype(bool)
    learned = _ber_summary(decisions, target_bool)
    uncompensated_decisions = test_data.features < 0.0
    uncompensated = _ber_summary(uncompensated_decisions, target_bool)
    frontend_matrix, frontend_offset, frontend_params = (
        load_receiver_iq_calibration(Path("noema_recipe.yaml"))
    )
    calibrated_features = compensate_receiver_iq(
        test_data.features,
        frontend_matrix,
        frontend_offset,
    )
    calibrated_decisions = calibrated_features < 0.0
    calibrated_oracle = _ber_summary(calibrated_decisions, target_bool)
    boundary_agreement = _decision_boundary_agreement(
        session,
        frontend_matrix,
        frontend_offset,
    )
    ber = float(learned["bit_error_rate"])
    logits_for_bit_one = -llr
    bce = float(
        np.mean(
            np.maximum(logits_for_bit_one, 0.0)
            - logits_for_bit_one * targets
            + np.log1p(np.exp(-np.abs(logits_for_bit_one)))
        )
    )
    metrics = {
        "split": "test",
        "bit_error_rate": ber,
        "binary_cross_entropy": bce,
        "evaluated_bits": int(targets.size),
        "learned_receiver": learned,
        "uncompensated_qpsk": uncompensated,
        "calibrated_iq_oracle": calibrated_oracle,
        "paired_ber_delta_learned_minus_uncompensated": float(
            learned["bit_error_rate"] - uncompensated["bit_error_rate"]
        ),
        "paired_ber_delta_learned_minus_calibrated_oracle": float(
            learned["bit_error_rate"] - calibrated_oracle["bit_error_rate"]
        ),
        "decision_boundary_agreement": boundary_agreement,
        "comparison_scope": {
            "channel": (
                "QPSK over AWGN followed by one stable affine receiver I/Q "
                "front-end impairment"
            ),
            "interpretation": (
                "The ordinary QPSK sign detector is deliberately mismatched. "
                "The learned receiver sees only impaired I/Q samples and should "
                "approach the diagnostic oracle that knows the simulated calibration."
            ),
            "frontend_parameters": frontend_params,
            "calibration_truth_forwarded_to_learned_runtime": False,
        },
        "split_integrity": held_out_integrity,
        "component_sha256": actual_sha,
        "test_capture_schema_sha256": list(test_data.capture_schema_sha256),
    }
    per_snr = _per_snr_metrics(
        snr_db=test_data.snr_db,
        learned_decisions=decisions,
        uncompensated_decisions=uncompensated_decisions,
        calibrated_decisions=calibrated_decisions,
        targets=target_bool,
    )
    if per_snr:
        metrics["per_snr"] = per_snr
        metrics["theoretical_qpsk_ber_weighted"] = float(
            sum(
                row["theoretical_qpsk_ber"] * row["learned_receiver"]["bit_count"]
                for row in per_snr
            )
            / sum(row["learned_receiver"]["bit_count"] for row in per_snr)
        )
    Path("evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.summary:
        print(_format_summary(metrics))
    else:
        print(json.dumps(metrics, sort_keys=True))
    return 0


def _format_summary(metrics: dict[str, object]) -> str:
    learned = dict(metrics["learned_receiver"])
    uncompensated = dict(metrics["uncompensated_qpsk"])
    oracle = dict(metrics["calibrated_iq_oracle"])
    boundary = dict(metrics["decision_boundary_agreement"])
    integrity = dict(metrics["split_integrity"])
    return "\n".join(
        (
            "held-out evaluation completed",
            "split integrity: %s; evaluated bits: %s"
            % (integrity.get("status", "unknown"), format(int(metrics["evaluated_bits"]), ",d")),
            "BER: uncompensated=%.6g; calibrated oracle=%.6g; learned=%.6g"
            % (
                float(uncompensated["bit_error_rate"]),
                float(oracle["bit_error_rate"]),
                float(learned["bit_error_rate"]),
            ),
            "learned minus uncompensated BER: %.6g"
            % float(metrics["paired_ber_delta_learned_minus_uncompensated"]),
            "learned/oracle decision-region agreement: %.4f%%"
            % (100.0 * float(boundary["symbol_region_agreement_rate"])),
            "artifact sha256: %s" % metrics["component_sha256"],
            "evidence: evaluation_metrics.json",
        )
    )


def _per_snr_metrics(
    *,
    snr_db: np.ndarray | None,
    learned_decisions: np.ndarray,
    uncompensated_decisions: np.ndarray,
    calibrated_decisions: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, object]]:
    if snr_db is None:
        return []
    values = np.asarray(snr_db, dtype=np.float64)
    if values.shape != (targets.shape[0],) or not bool(np.all(np.isfinite(values))):
        return []
    rows = []
    for snr in sorted(float(value) for value in np.unique(values)):
        mask = values == snr
        rows.append(
            {
                "snr_db": snr,
                "learned_receiver": _ber_summary(
                    learned_decisions[mask], targets[mask]
                ),
                "uncompensated_qpsk": _ber_summary(
                    uncompensated_decisions[mask], targets[mask]
                ),
                "calibrated_iq_oracle": _ber_summary(
                    calibrated_decisions[mask], targets[mask]
                ),
                "theoretical_qpsk_ber": _theoretical_qpsk_ber(snr),
            }
        )
    return rows


def _decision_boundary_agreement(
    session,
    frontend_matrix: np.ndarray,
    frontend_offset: np.ndarray,
    *,
    limit: float = 2.25,
    grid_size: int = 257,
) -> dict[str, object]:
    """Compare learned and oracle decisions over a dense, fixed I/Q plane."""

    axis = np.linspace(
        -float(limit),
        float(limit),
        max(3, int(grid_size)),
        dtype=np.float32,
    )
    i_grid, q_grid = np.meshgrid(axis, axis)
    features = np.stack((i_grid.reshape(-1), q_grid.reshape(-1)), axis=1)
    learned_logits = np.asarray(
        session.run(
            ["bit_llr"],
            {"rx_symbols_ri": features.astype(np.float32, copy=False)},
        )[0],
        dtype=np.float32,
    )
    if learned_logits.shape != features.shape:
        raise ValueError(
            "neural-receiver boundary probe returned %s, expected %s"
            % (learned_logits.shape, features.shape)
        )
    learned = learned_logits < 0.0
    calibrated = (
        compensate_receiver_iq(features, frontend_matrix, frontend_offset) < 0.0
    )
    bit_disagreement = learned != calibrated
    symbol_disagreement = np.any(bit_disagreement, axis=1)
    return {
        "coordinate_space": "impaired_received_iq",
        "i_range": [-float(limit), float(limit)],
        "q_range": [-float(limit), float(limit)],
        "grid_size": int(axis.size),
        "bit_decision_agreement_rate": float(1.0 - np.mean(bit_disagreement)),
        "symbol_region_agreement_rate": float(
            1.0 - np.mean(symbol_disagreement)
        ),
        "oracle_calibration_forwarded_to_learned_runtime": False,
    }


def _ber_summary(decisions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    errors = int(np.count_nonzero(np.asarray(decisions) != np.asarray(targets)))
    bit_count = int(np.asarray(targets).size)
    ber = float(errors / max(1, bit_count))
    lower, upper = _wilson_interval(errors, bit_count)
    return {
        "bit_errors": errors,
        "bit_count": bit_count,
        "bit_error_rate": ber,
        "ber_95_percent_wilson_interval": [lower, upper],
    }


def _wilson_interval(errors: int, count: int) -> tuple[float, float]:
    if count < 1:
        return 0.0, 1.0
    z = 1.959963984540054
    probability = float(errors / count)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _theoretical_qpsk_ber(snr_db: float) -> float:
    linear_snr = 10.0 ** (float(snr_db) / 10.0)
    return float(0.5 * math.erfc(math.sqrt(linear_snr) / math.sqrt(2.0)))


if __name__ == "__main__":
    raise SystemExit(main())
