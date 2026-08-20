from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import onnxruntime as ort
import torch

from datamodule import build_loader, datasets_from_config, load_data_contract
from losses import csi_metrics, csi_metrics_over_snrs
try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_paths(manifest_path: Path, manifest: Mapping[str, Any]) -> Dict[str, Path]:
    result = {}
    for item in list(manifest.get("components") or []):
        component = dict(item or {})
        component_id = str(component.get("id") or "")
        path = manifest_path.parent / str(component.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError("Artifact component is missing: %s" % path)
        if str(component.get("sha256") or "") != _sha256(path):
            raise ValueError("Artifact component hash mismatch: %s" % path)
        result[component_id] = path
    if set(result) != {"encoder", "decoder"}:
        raise ValueError("CSI artifact must contain encoder and decoder components")
    return result


def _transport(code: np.ndarray, link: Mapping[str, Any]) -> np.ndarray:
    mode = str(link.get("mode") or "uniform_quantized")
    values = np.asarray(code, dtype=np.float32)
    if mode == "ideal_noiseless":
        return values.copy()
    if mode != "uniform_quantized":
        raise ValueError("Unsupported CSI feedback mode: %s" % mode)
    bits = int(link.get("bits_per_latent") or 8)
    clip = float(link.get("clip_value") or 1.0)
    levels = (1 << bits) - 1
    clipped = np.clip(values, -clip, clip)
    indices = np.rint((clipped + clip) * levels / (2.0 * clip))
    return ((indices * (2.0 * clip) / levels) - clip).astype(np.float32, copy=False)


def _aggregate_complete_metrics(
    reconstruction_batches,
    target_batches,
    *,
    downlink_snr_db: float,
    snr_db_values,
) -> Dict[str, float]:
    """Reduce nonlinear metrics once over the complete held-out population."""

    if not reconstruction_batches or not target_batches:
        raise ValueError("Held-out CSI test capture is empty")
    complete_reconstruction = torch.from_numpy(
        np.concatenate(reconstruction_batches, axis=0)
    )
    complete_target = torch.from_numpy(np.concatenate(target_batches, axis=0))
    complete_metrics = csi_metrics(
        complete_reconstruction,
        complete_target,
        downlink_snr_db=downlink_snr_db,
    )
    grid_metrics = csi_metrics_over_snrs(
        complete_reconstruction,
        complete_target,
        snr_db_values=snr_db_values,
    )
    metrics = {key: float(value) for key, value in complete_metrics.items()}
    metrics.update(
        {
            "mean_spectral_efficiency_retention": float(
                grid_metrics["mean_spectral_efficiency_retention"]
            ),
            "snr_grid_mean_spectral_efficiency_bps_hz": float(
                grid_metrics["spectral_efficiency_bps_hz"]
            ),
            "snr_grid_mean_perfect_csi_spectral_efficiency_bps_hz": float(
                grid_metrics["perfect_csi_spectral_efficiency_bps_hz"]
            ),
            "snr_grid_mean_spectral_efficiency_loss_bps_hz": float(
                grid_metrics["spectral_efficiency_loss_bps_hz"]
            ),
        }
    )
    return metrics


def main() -> int:
    config = load_strict_yaml_or_json(Path("train_config.yaml"))
    if not isinstance(config, dict):
        raise ValueError("train_config.yaml must contain a mapping")
    load_data_contract(config)
    datasets = datasets_from_config(config, include_test=True)
    training = dict(config.get("training") or {})
    evaluation = dict(config.get("evaluation") or {})
    objective = dict(config.get("objective") or {})
    link = dict(config.get("feedback_link") or {})
    manifest_path = Path(
        str(training.get("artifact_manifest_path") or "../trained_artifact.yaml")
    )
    manifest = load_strict_yaml_or_json(manifest_path)
    if not isinstance(manifest, dict) or int(manifest.get("schema_version") or 0) != 2:
        raise ValueError("Expected a schema-v2 trained CSI artifact")
    components = _component_paths(manifest_path, manifest)
    encoder = ort.InferenceSession(str(components["encoder"]), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(components["decoder"]), providers=["CPUExecutionProvider"])
    loader = build_loader(
        datasets["test"],
        batch_size=int(training.get("batch_size") or 128),
        shuffle=False,
        seed=0,
        num_workers=int(training.get("num_workers") or 0),
    )
    configured_downlink_snr_db = evaluation.get("downlink_snr_db")
    downlink_snr_db = float(
        10.0 if configured_downlink_snr_db is None else configured_downlink_snr_db
    )
    snr_db_values = [
        float(value)
        for value in (
            objective.get("snr_db_values") or [downlink_snr_db]
        )
    ]
    reconstruction_batches = []
    target_batches = []
    sample_count = 0
    quantization_mse = 0.0
    clipped_fraction = 0.0
    clip = float(link.get("clip_value") or 1.0)
    for csi_ri in loader:
        values = csi_ri.numpy().astype(np.float32, copy=False)
        code = np.asarray(encoder.run(["feedback_code"], {"csi_ri": values})[0])
        received = _transport(code, link)
        reconstruction = np.asarray(
            decoder.run(["csi_hat_ri"], {"feedback_code": received})[0],
            dtype=np.float32,
        )
        batch_size = int(values.shape[0])
        sample_count += batch_size
        reconstruction_batches.append(np.asarray(reconstruction, dtype=np.float32))
        target_batches.append(np.asarray(values, dtype=np.float32))
        quantization_mse += float(np.mean((received.astype(np.float64) - code) ** 2)) * batch_size
        clipped_fraction += float(np.mean(np.abs(code) > clip)) * batch_size
    if sample_count < 1:
        raise ValueError("Held-out CSI test capture is empty")
    metrics = _aggregate_complete_metrics(
        reconstruction_batches,
        target_batches,
        downlink_snr_db=downlink_snr_db,
        snr_db_values=snr_db_values,
    )
    metrics.update(
        {
            "feedback_bits_per_sample": int(link.get("feedback_bits_per_sample") or 0),
            "quantization_mse": quantization_mse / sample_count,
            "quantization_clipped_fraction": clipped_fraction / sample_count,
        }
    )
    output_path = Path(str(evaluation.get("metrics_path") or "evaluation_metrics.json"))
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "test",
                "records": sample_count,
                "test_capture_used_for_training_or_checkpoint_selection": False,
                "snr_db_values": snr_db_values,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "held-out test: nmse=%.7f nmse_db=%.3f mean_rate_retention=%.7f"
        % (
            metrics["nmse"],
            metrics["nmse_db"],
            metrics["mean_spectral_efficiency_retention"],
        )
    )
    print("wrote evaluation metrics: %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
