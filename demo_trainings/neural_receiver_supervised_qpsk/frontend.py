from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


def load_receiver_iq_calibration(
    recipe_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    recipe = load_strict_yaml_or_json(Path(recipe_path))
    steps = list(dict(recipe).get("steps") or [])
    frontends = [
        dict(step)
        for step in steps
        if isinstance(step, Mapping)
        and str(step.get("op") or "") == "hardware.receiver_iq_imbalance"
    ]
    if len(frontends) != 1:
        raise ValueError(
            "The receiver-calibration demo requires exactly one "
            "hardware.receiver_iq_imbalance step; found %d" % len(frontends)
        )
    params = dict(frontends[0].get("params") or {})
    resolved = {
        "gain_imbalance_db": _finite(params, "gain_imbalance_db", 5.0),
        "quadrature_error_deg": _finite(
            params, "quadrature_error_deg", 12.0
        ),
        "phase_offset_deg": _finite(params, "phase_offset_deg", 20.0),
        "dc_offset_i": _finite(params, "dc_offset_i", 0.18),
        "dc_offset_q": _finite(params, "dc_offset_q", -0.12),
    }
    matrix = receiver_iq_forward_matrix(
        gain_imbalance_db=resolved["gain_imbalance_db"],
        quadrature_error_deg=resolved["quadrature_error_deg"],
        phase_offset_deg=resolved["phase_offset_deg"],
    )
    offset = np.asarray(
        [resolved["dc_offset_i"], resolved["dc_offset_q"]],
        dtype=np.float64,
    )
    return matrix, offset, resolved


def receiver_iq_forward_matrix(
    *,
    gain_imbalance_db: float,
    quadrature_error_deg: float,
    phase_offset_deg: float,
) -> np.ndarray:
    gain_ratio = 10.0 ** (float(gain_imbalance_db) / 20.0)
    i_gain = math.sqrt(gain_ratio)
    q_gain = 1.0 / i_gain
    quadrature_error = math.radians(float(quadrature_error_deg))
    phase_offset = math.radians(float(phase_offset_deg))
    rotation = np.asarray(
        [
            [math.cos(phase_offset), -math.sin(phase_offset)],
            [math.sin(phase_offset), math.cos(phase_offset)],
        ],
        dtype=np.float64,
    )
    imbalance = np.asarray(
        [
            [i_gain, 0.0],
            [
                q_gain * math.sin(quadrature_error),
                q_gain * math.cos(quadrature_error),
            ],
        ],
        dtype=np.float64,
    )
    matrix = imbalance @ rotation
    if (
        not bool(np.all(np.isfinite(matrix)))
        or abs(float(np.linalg.det(matrix))) < 1e-8
    ):
        raise ValueError("Receiver I/Q calibration matrix is not invertible")
    return matrix


def compensate_receiver_iq(
    features: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Receiver I/Q features must have shape [symbol, 2]")
    compensated = (
        values - np.asarray(offset, dtype=np.float64).reshape(1, 2)
    ) @ np.linalg.inv(np.asarray(matrix, dtype=np.float64)).T
    return np.ascontiguousarray(compensated, dtype=np.float32)


def _finite(
    params: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = float(params.get(key, default))
    if not math.isfinite(value):
        raise ValueError("Receiver I/Q parameter %s must be finite" % key)
    return value
