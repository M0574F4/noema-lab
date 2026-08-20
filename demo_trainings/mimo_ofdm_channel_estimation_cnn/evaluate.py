from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from datamodule import (
    fixed_prior_lmmse_estimate,
    load_capture_dataset,
    split_integrity_report,
)

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def main() -> int:
    root = Path(__file__).resolve().parent
    config = _mapping(root / "train_config.yaml")
    data = dict(config.get("data") or {})
    training = dict(config.get("training") or {})
    test = load_capture_dataset(
        [_resolve(root, value) for value in data["test_capture_dirs"]],
        pilot_tap=str(data["pilot_tap"]),
        mask_tap=str(data["mask_tap"]),
        ls_tap=str(data["ls_tap"]),
        noise_tap=str(data["noise_tap"]),
        target_tap=str(data["target_tap"]),
        expected_split="test",
    )
    history = json.loads((root / "training_history.json").read_text(encoding="utf-8"))
    integrity = _held_out_integrity(test.record_sha256, history["split_integrity"])
    manifest_path = _resolve(root, training["artifact_manifest_path"])
    manifest = _mapping(manifest_path)
    component = dict((manifest.get("components") or [])[0])
    component_path = manifest_path.parent / str(component["path"])
    expected_sha = str(component["sha256"])
    if _sha256(component_path) != expected_sha:
        raise ValueError("Returned ONNX component SHA-256 does not match the artifact manifest")
    session = ort.InferenceSession(str(component_path), providers=["CPUExecutionProvider"])
    prediction = session.run(
        ["h_hat_ri"],
        {
            "pilot_ls_ri": test.pilot_ls_ri,
            "pilot_mask": test.pilot_mask,
            "ls_estimate_ri": test.ls_estimate_ri,
            "noise_variance": test.noise_variance,
        },
    )[0]
    learned = _metrics(prediction, test.channel_truth_ri, test.noise_variance)
    least_squares = _metrics(
        test.ls_estimate_ri,
        test.channel_truth_ri,
        test.noise_variance,
    )
    fixed_prior_lmmse = _metrics(
        fixed_prior_lmmse_estimate(test),
        test.channel_truth_ri,
        test.noise_variance,
    )
    strongest_classical_nmse_db = min(
        least_squares["nmse_db"],
        fixed_prior_lmmse["nmse_db"],
    )
    payload = {
        "schema_version": 1,
        "split": "test",
        "comparison_scope": {
            "channel": (
                "2×2 MIMO-OFDM over mixed Sionna 3GPP TDL-A/C/E profiles "
                "with sparse orthogonal comb pilots"
            ),
            "runtime_inputs": [
                "sparse divided pilot observations",
                "pilot mask",
                "operation-owned interpolated LS estimate",
                "noise variance",
            ],
            "target": "frequency-domain channel truth",
        },
        "methods": {
            "least_squares": least_squares,
            "fixed_prior_lmmse": fixed_prior_lmmse,
            "learned_estimator": learned,
        },
        "improvement": {
            "nmse_db_learned_minus_ls": learned["nmse_db"] - least_squares["nmse_db"],
            "zf_rate_retention_learned_minus_ls": (
                learned["zf_rate_retention"] - least_squares["zf_rate_retention"]
            ),
            "nmse_db_learned_minus_fixed_prior_lmmse": (
                learned["nmse_db"] - fixed_prior_lmmse["nmse_db"]
            ),
            "nmse_db_learned_minus_strongest_classical": (
                learned["nmse_db"] - strongest_classical_nmse_db
            ),
        },
        "split_integrity": integrity,
        "component_sha256": expected_sha,
        "trained_artifact_manifest_sha256": _sha256(manifest_path),
        "test_capture_schema_sha256": list(test.schema_sha256),
    }
    destination = root / "evaluation_metrics.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def _metrics(prediction_ri, truth_ri, noise_variance) -> dict[str, float]:
    prediction = _complex(prediction_ri)
    truth = _complex(truth_ri)
    error = float(np.sum(np.abs(prediction - truth) ** 2))
    power = float(np.sum(np.abs(truth) ** 2))
    nmse = error / max(power, 1e-12)
    correlation = float(
        abs(np.vdot(truth.reshape(-1), prediction.reshape(-1)))
        / max(
            math.sqrt(
                float(np.vdot(truth.reshape(-1), truth.reshape(-1)).real)
                * float(np.vdot(prediction.reshape(-1), prediction.reshape(-1)).real)
            ),
            1e-12,
        )
    )
    rate, perfect_rate = _zf_rates(
        truth,
        prediction,
        noise_variance=np.asarray(noise_variance).reshape(-1),
    )
    return {
        "nmse": nmse,
        "nmse_db": 10.0 * math.log10(max(nmse, 1e-12)),
        "complex_correlation": correlation,
        "zf_spectral_efficiency_bps_hz": rate,
        "perfect_csi_zf_spectral_efficiency_bps_hz": perfect_rate,
        "zf_rate_retention": rate / max(perfect_rate, 1e-12),
    }


def _zf_rates(truth, estimate, *, noise_variance):
    rates = []
    perfect_rates = []
    for batch in range(truth.shape[0]):
        noise = max(float(noise_variance[batch]), 1e-12)
        for subcarrier in range(truth.shape[3]):
            h_true = truth[batch, :, :, subcarrier]
            for h_design, output in (
                (estimate[batch, :, :, subcarrier], rates),
                (h_true, perfect_rates),
            ):
                equalizer = np.linalg.pinv(h_design)
                effective = equalizer @ h_true
                signal = np.abs(np.diag(effective)) ** 2
                interference = np.sum(np.abs(effective) ** 2, axis=1) - signal
                noise_enhancement = noise * np.sum(np.abs(equalizer) ** 2, axis=1)
                sinr = signal / np.maximum(interference + noise_enhancement, 1e-12)
                output.append(float(np.sum(np.log2(1.0 + sinr))))
    return float(np.mean(rates)), float(np.mean(perfect_rates))


def _held_out_integrity(test_hashes, recorded):
    train_hashes = set(str(value) for value in recorded["record_sha256"]["train"])
    validation_hashes = set(str(value) for value in recorded["record_sha256"]["validation"])
    current = set(test_hashes)
    overlaps = {
        "train__test": len(train_hashes.intersection(current)),
        "validation__test": len(validation_hashes.intersection(current)),
    }
    if any(overlaps.values()) or len(current) != len(test_hashes):
        raise ValueError("Held-out split-integrity check failed: %s" % overlaps)
    return {
        "status": "passed",
        "method": (
            "sha256(sparse pilot LS + pilot mask + interpolated LS + "
            "noise variance + channel truth)"
        ),
        "test_record_count": len(test_hashes),
        "test_unique_record_count": len(current),
        "pairwise_overlap_record_counts": overlaps,
    }


def _complex(value):
    array = np.asarray(value, dtype=np.float32)
    return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)


def _mapping(path: Path) -> dict:
    value = load_strict_yaml_or_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML/JSON object: %s" % path)
    return dict(value)


def _resolve(root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
