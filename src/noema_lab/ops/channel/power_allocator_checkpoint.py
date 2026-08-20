from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from noema_lab.core.operations import OperationError
from noema_lab.core.structured_input import decode_strict_json_object


CHECKPOINT_FORMAT = "noema_csi_power_deepset_npz_v1"
CHECKPOINT_KIND = "noema.csi_power_allocator_checkpoint"
INPUT_CONTRACT = "log(max(gain*average_power/noise_variance,eps))"
OUTPUT_CONTRACT = "euclidean_simplex_projection_times_fixed_sum_power"


@dataclass(frozen=True)
class CsiPowerAllocatorCheckpoint:
    path: Path
    sha256: str
    metadata: Dict[str, Any]
    feature_mean: float
    feature_scale: float
    phi_weight_0: np.ndarray
    phi_bias_0: np.ndarray
    phi_weight_1: np.ndarray
    phi_bias_1: np.ndarray
    rho_weight_0: np.ndarray
    rho_bias_0: np.ndarray
    rho_weight_out: np.ndarray
    rho_bias_out: np.ndarray

    @property
    def hidden_dim(self) -> int:
        return int(self.phi_bias_0.shape[0])


def load_csi_power_allocator_checkpoint(
    checkpoint_path: str,
    expected_sha256: str,
    *,
    strict: bool = True,
    max_bytes: int = 64 * 1024 * 1024,
) -> CsiPowerAllocatorCheckpoint:
    raw_path = str(checkpoint_path or "").strip()
    if not raw_path:
        raise OperationError("policy=learned_checkpoint requires params.checkpoint_path")
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OperationError("Learned power-allocation checkpoint is not a readable file: %s" % path) from exc
    if not resolved.is_file():
        raise OperationError("Learned power-allocation checkpoint is not a regular file: %s" % resolved)
    if resolved.suffix.lower() != ".npz":
        raise OperationError("Learned power-allocation checkpoint must use the safe .npz format")
    size = int(resolved.stat().st_size)
    if size <= 0 or size > max(1, int(max_bytes)):
        raise OperationError(
            "Learned power-allocation checkpoint size %d bytes is outside the allowed range (max %d)"
            % (size, max(1, int(max_bytes)))
        )
    declared_sha = str(expected_sha256 or "").strip().lower()
    if len(declared_sha) != 64 or any(char not in "0123456789abcdef" for char in declared_sha):
        raise OperationError("policy=learned_checkpoint requires a 64-character lowercase checkpoint_sha256")
    actual_sha = _file_sha256(resolved)
    if actual_sha != declared_sha:
        raise OperationError(
            "Learned power-allocation checkpoint SHA-256 mismatch: expected %s, got %s"
            % (declared_sha, actual_sha)
        )

    required_arrays = {
        "feature_mean",
        "feature_scale",
        "phi_weight_0",
        "phi_bias_0",
        "phi_weight_1",
        "phi_bias_1",
        "rho_weight_0",
        "rho_bias_0",
        "rho_weight_out",
        "rho_bias_out",
    }
    try:
        with np.load(str(resolved), allow_pickle=False) as payload:
            names = set(payload.files)
            missing = required_arrays - names
            if missing:
                raise OperationError(
                    "Learned power-allocation checkpoint is missing array(s): %s"
                    % ", ".join(sorted(missing))
                )
            allowed = required_arrays | {"metadata_json"}
            extra = names - allowed
            if bool(strict) and extra:
                raise OperationError(
                    "Learned power-allocation checkpoint has unexpected array(s): %s"
                    % ", ".join(sorted(extra))
                )
            if "metadata_json" not in names:
                raise OperationError("Learned power-allocation checkpoint requires metadata_json")
            metadata = _decode_metadata(payload["metadata_json"])
            arrays = {name: np.asarray(payload[name]) for name in required_arrays}
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("Could not read learned power-allocation checkpoint: %s" % resolved) from exc

    _validate_metadata(metadata)
    for name, value in arrays.items():
        if value.dtype != np.float32:
            raise OperationError("Checkpoint array %s must have dtype float32; got %s" % (name, value.dtype))
        if not np.all(np.isfinite(value)):
            raise OperationError("Checkpoint array %s contains NaN or infinite values" % name)

    feature_mean = _scalar_array(arrays["feature_mean"], "feature_mean")
    feature_scale = _scalar_array(arrays["feature_scale"], "feature_scale")
    if not math.isfinite(feature_scale) or feature_scale <= 0.0:
        raise OperationError("Checkpoint feature_scale must be finite and greater than zero")
    hidden_dim = int(arrays["phi_bias_0"].size)
    if hidden_dim < 1 or hidden_dim > 4096:
        raise OperationError("Checkpoint hidden_dim must be between 1 and 4096")
    expected_shapes = {
        "phi_weight_0": (hidden_dim, 1),
        "phi_bias_0": (hidden_dim,),
        "phi_weight_1": (hidden_dim, hidden_dim),
        "phi_bias_1": (hidden_dim,),
        "rho_weight_0": (hidden_dim, 2 * hidden_dim + 1),
        "rho_bias_0": (hidden_dim,),
        "rho_weight_out": (1, hidden_dim),
        "rho_bias_out": (1,),
    }
    for name, shape in expected_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise OperationError(
                "Checkpoint array %s must have shape %s; got %s"
                % (name, list(shape), list(arrays[name].shape))
            )
    declared_hidden = int(metadata.get("hidden_dim") or 0)
    if declared_hidden != hidden_dim:
        raise OperationError(
            "Checkpoint metadata hidden_dim=%d does not match weight shape %d"
            % (declared_hidden, hidden_dim)
        )

    return CsiPowerAllocatorCheckpoint(
        path=resolved,
        sha256=actual_sha,
        metadata=metadata,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        phi_weight_0=arrays["phi_weight_0"],
        phi_bias_0=arrays["phi_bias_0"],
        phi_weight_1=arrays["phi_weight_1"],
        phi_bias_1=arrays["phi_bias_1"],
        rho_weight_0=arrays["rho_weight_0"],
        rho_bias_0=arrays["rho_bias_0"],
        rho_weight_out=arrays["rho_weight_out"],
        rho_bias_out=arrays["rho_bias_out"],
    )


def infer_csi_power_allocation(
    checkpoint: CsiPowerAllocatorCheckpoint,
    gains: np.ndarray,
    noise_variance: float,
    average_power_budget: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    gain_rows = np.asarray(gains, dtype=np.float64)
    if gain_rows.ndim != 2 or gain_rows.shape[1] < 1:
        raise OperationError("Learned power allocator expects channel gains with shape [state, subcarrier]")
    if not np.all(np.isfinite(gain_rows)) or np.any(gain_rows < 0.0):
        raise OperationError("Learned power allocator received non-finite or negative channel gains")
    noise = float(noise_variance)
    budget = float(average_power_budget)
    if not math.isfinite(noise) or noise <= 0.0:
        raise OperationError("Learned power allocator requires finite positive noise_variance")
    if not math.isfinite(budget) or budget < 0.0:
        raise OperationError("Learned power allocator requires finite nonnegative target_power")
    if budget == 0.0:
        return np.zeros_like(gain_rows, dtype=np.float64)

    normalized = np.log(np.maximum(gain_rows * budget / noise, max(float(eps), 1e-30)))
    normalized = (normalized - checkpoint.feature_mean) / checkpoint.feature_scale
    carrier_features = normalized[..., None]
    phi = _relu(_linear(carrier_features, checkpoint.phi_weight_0, checkpoint.phi_bias_0))
    phi = _relu(_linear(phi, checkpoint.phi_weight_1, checkpoint.phi_bias_1))
    context = np.mean(phi, axis=1, keepdims=True)
    context = np.broadcast_to(context, phi.shape)
    rho_input = np.concatenate([phi, context, carrier_features], axis=-1)
    hidden = _relu(_linear(rho_input, checkpoint.rho_weight_0, checkpoint.rho_bias_0))
    scores = _linear(hidden, checkpoint.rho_weight_out, checkpoint.rho_bias_out)[..., 0]
    fractions = project_rows_to_simplex(scores)
    allocation = fractions * (float(gain_rows.shape[1]) * budget)
    if allocation.shape != gain_rows.shape or not np.all(np.isfinite(allocation)) or np.any(allocation < -1e-10):
        raise OperationError("Learned power allocator produced an invalid allocation")
    allocation = np.maximum(allocation, 0.0)
    expected_total = float(gain_rows.shape[1]) * budget
    errors = np.abs(np.sum(allocation, axis=1) - expected_total)
    tolerance = 1e-7 * max(1.0, expected_total)
    if np.any(errors > tolerance):
        raise OperationError(
            "Learned power allocator violated the fixed sum-power constraint (max error %.6g)"
            % float(np.max(errors))
        )
    return allocation.astype(np.float64, copy=False)


def project_rows_to_simplex(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        raise OperationError("Simplex projection requires a finite [batch, resource] score array")
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=np.float64)[None, :]
    active = sorted_values - cumulative / indices > 0.0
    rho = np.sum(active, axis=1) - 1
    theta = cumulative[np.arange(values.shape[0]), rho] / (rho.astype(np.float64) + 1.0)
    projected = np.maximum(values - theta[:, None], 0.0)
    totals = np.sum(projected, axis=1, keepdims=True)
    if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
        raise OperationError("Simplex projection produced an invalid row")
    return projected / totals


def _linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(values, np.asarray(weight, dtype=np.float64).T) + np.asarray(bias, dtype=np.float64)


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _decode_metadata(value: np.ndarray) -> Dict[str, Any]:
    try:
        raw = value.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = decode_strict_json_object(
            str(raw),
            label="Checkpoint metadata_json",
        )
    except Exception as exc:
        raise OperationError(
            "Checkpoint metadata_json must contain one unambiguous UTF-8 JSON "
            "object: %s" % exc
        ) from exc
    return payload


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "kind": CHECKPOINT_KIND,
        "format": CHECKPOINT_FORMAT,
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "activation": "relu",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise OperationError(
                "Checkpoint metadata %s must be %r; got %r" % (key, value, metadata.get(key))
            )
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise OperationError("Checkpoint metadata requires a training provenance object")
    if str(training.get("objective") or "") != "maximize_parallel_channel_shannon_spectral_efficiency":
        raise OperationError("Checkpoint training objective is not the supported label-free Shannon objective")
    if training.get("supervised_labels_used") is not False:
        raise OperationError("Checkpoint must declare supervised_labels_used=false")


def _scalar_array(value: np.ndarray, name: str) -> float:
    if int(value.size) != 1:
        raise OperationError("Checkpoint array %s must contain exactly one scalar" % name)
    return float(value.reshape(-1)[0])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
