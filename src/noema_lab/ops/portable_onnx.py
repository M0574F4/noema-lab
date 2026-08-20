from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from noema_lab.core.operations import OperationError


ONNX_ARTIFACT_FORMAT = "onnx"
DEFAULT_MAX_ONNX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class PortableOnnxComponent:
    path: Path
    sha256: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    session: object
    session_evidence: Mapping[str, Any]


def portable_onnx_session_evidence(
    component: PortableOnnxComponent,
) -> Dict[str, Any]:
    """Return the load-time-verified deterministic execution-session identity."""

    return dict(component.session_evidence)


def load_portable_onnx_component(
    model_path: str,
    expected_sha256: str,
    *,
    expected_inputs: Sequence[str] = (),
    expected_outputs: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_ONNX_BYTES,
) -> PortableOnnxComponent:
    """Validate and load a hash-pinned ONNX graph without enabling custom code."""

    raw_path = str(model_path or "").strip()
    if not raw_path:
        raise OperationError("ONNX runtime requires params.model_path")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OperationError("ONNX model is not a readable file: %s" % raw_path) from exc
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise OperationError("Portable learned artifacts must reference a regular .onnx file")
    size = int(path.stat().st_size)
    maximum = max(1, int(max_bytes))
    if size <= 0 or size > maximum:
        raise OperationError(
            "ONNX model size %d bytes is outside the allowed range (max %d)"
            % (size, maximum)
        )
    declared_sha = _validated_sha256(expected_sha256)
    actual_sha = _file_sha256(path)
    if actual_sha != declared_sha:
        raise OperationError(
            "ONNX model SHA-256 mismatch: expected %s, got %s"
            % (declared_sha, actual_sha)
        )
    component = _load_component_cached(str(path), actual_sha, size)
    _require_names("input", component.input_names, expected_inputs)
    _require_names("output", component.output_names, expected_outputs)
    return component


def run_portable_onnx(
    component: PortableOnnxComponent,
    inputs: Mapping[str, np.ndarray],
    *,
    outputs: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    missing = [name for name in component.input_names if name not in inputs]
    extra = [name for name in inputs if name not in component.input_names]
    if missing or extra:
        details = []
        if missing:
            details.append("missing %s" % ", ".join(missing))
        if extra:
            details.append("unexpected %s" % ", ".join(extra))
        raise OperationError("ONNX input contract mismatch: %s" % "; ".join(details))
    requested = tuple(str(name) for name in outputs) or component.output_names
    unknown = [name for name in requested if name not in component.output_names]
    if unknown:
        raise OperationError("ONNX output contract has unknown output(s): %s" % ", ".join(unknown))
    feed = {
        name: np.ascontiguousarray(np.asarray(inputs[name]))
        for name in component.input_names
    }
    try:
        values = component.session.run(list(requested), feed)
    except Exception as exc:
        raise OperationError("ONNX inference failed for %s: %s" % (component.path, exc)) from exc
    result = {name: np.asarray(value) for name, value in zip(requested, values)}
    for name, value in result.items():
        if not _finite_array(value):
            raise OperationError("ONNX output %s contains NaN or infinite values" % name)
    return result


def infer_power_scores_onnx(
    component: PortableOnnxComponent,
    channel_gain: np.ndarray,
    noise_variance: float | np.ndarray,
    average_power_budget: float | np.ndarray,
) -> np.ndarray:
    gains = np.asarray(channel_gain, dtype=np.float32)
    if gains.ndim != 2 or gains.shape[0] < 1 or gains.shape[1] < 1:
        raise OperationError("Power-policy ONNX input channel_gain must have shape [batch, subcarrier]")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise OperationError("Power-policy ONNX channel_gain must be finite and nonnegative")
    batch = int(gains.shape[0])
    noise = _batch_column(noise_variance, batch, "noise_variance", positive=True)
    budget = _batch_column(average_power_budget, batch, "average_power_budget", positive=False)
    result = run_portable_onnx(
        component,
        {
            "channel_gain": gains,
            "noise_variance": noise,
            "average_power_budget": budget,
        },
        outputs=("allocation_scores",),
    )["allocation_scores"]
    scores = np.asarray(result, dtype=np.float64)
    if scores.shape != gains.shape:
        raise OperationError(
            "Power-policy ONNX output allocation_scores must have shape %s; got %s"
            % (list(gains.shape), list(scores.shape))
        )
    return scores


def project_power_scores(
    scores: np.ndarray,
    average_power_budget: float | np.ndarray,
) -> np.ndarray:
    """Noema-owned exact nonnegative fixed-sum postcondition for learned policies."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        raise OperationError("Power-policy scores must be a finite [batch, subcarrier] array")
    budget = _batch_column(
        average_power_budget,
        int(values.shape[0]),
        "average_power_budget",
        positive=False,
    )[:, 0].astype(np.float64)
    if np.any(budget < 0.0):
        raise OperationError("average_power_budget must be nonnegative")
    projected = _project_rows_to_simplex(values)
    allocation = projected * (budget[:, None] * float(values.shape[1]))
    expected = budget * float(values.shape[1])
    error = np.abs(np.sum(allocation, axis=1) - expected)
    tolerance = 1e-7 * np.maximum(1.0, expected)
    if np.any(allocation < -1e-10) or np.any(error > tolerance):
        raise OperationError("Noema power projection failed its nonnegative fixed-sum postcondition")
    return np.maximum(allocation, 0.0)


def encode_images_onnx(
    component: PortableOnnxComponent,
    images: np.ndarray,
) -> np.ndarray:
    values = np.asarray(images)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
        raise OperationError("DeepJSCC ONNX encoder expects uint8 images shaped [N,H,W,3]")
    tensor = np.ascontiguousarray(values.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    encoded = run_portable_onnx(
        component,
        {"images": tensor},
        outputs=("symbols_ri",),
    )["symbols_ri"]
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.ndim != 4 or encoded.shape[1] < 2 or encoded.shape[1] % 2:
        raise OperationError(
            "DeepJSCC ONNX encoder output symbols_ri must have shape [N,2C,H,W]"
        )
    channels = encoded.shape[1] // 2
    symbols = encoded[:, :channels] + 1j * encoded[:, channels:]
    return np.asarray(symbols, dtype=np.complex64)


def decode_images_onnx(
    component: PortableOnnxComponent,
    symbols: np.ndarray,
    *,
    image_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.ndim != 4 or not np.issubdtype(values.dtype, np.complexfloating):
        raise OperationError("DeepJSCC ONNX decoder expects complex symbols shaped [N,C,H,W]")
    symbols_ri = np.ascontiguousarray(
        np.concatenate([values.real, values.imag], axis=1),
        dtype=np.float32,
    )
    decoded = run_portable_onnx(
        component,
        {"symbols_ri": symbols_ri},
        outputs=("reconstruction",),
    )["reconstruction"]
    decoded = np.asarray(decoded, dtype=np.float32)
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise OperationError(
            "DeepJSCC ONNX decoder output reconstruction must have shape [N,3,H,W]"
        )
    decoded = decoded.transpose(0, 2, 3, 1)
    if image_shape is not None:
        shape = tuple(int(value) for value in image_shape)
        if len(shape) != 4 or shape[0] != decoded.shape[0] or shape[-1] != 3:
            raise OperationError("DeepJSCC image_shape must be [N,H,W,3]")
        if decoded.shape[1] < shape[1] or decoded.shape[2] < shape[2]:
            raise OperationError("DeepJSCC ONNX decoder output is smaller than image_shape")
        decoded = decoded[:, : shape[1], : shape[2], :]
    if not np.all(np.isfinite(decoded)):
        raise OperationError("DeepJSCC ONNX decoder produced non-finite images")
    return np.rint(np.clip(decoded, 0.0, 1.0) * 255.0).astype(np.uint8)


@lru_cache(maxsize=16)
def _load_component_cached(path_text: str, sha256: str, size: int) -> PortableOnnxComponent:
    path = Path(path_text)
    try:
        import onnx
    except Exception as exc:
        raise OperationError(
            'Portable ONNX artifacts require the `onnx` dependency. Install with '
            '`python -m pip install "noema-lab[onnx]"` in an installed environment, '
            "or `uv sync --extra onnx` in a source checkout."
        ) from exc
    try:
        model = onnx.load_model(str(path), load_external_data=False)
        external = []
        for tensor in _onnx_tensors(model):
            if int(getattr(tensor, "data_location", 0)) == int(onnx.TensorProto.EXTERNAL):
                external.append(str(getattr(tensor, "name", "tensor")))
            elif list(getattr(tensor, "external_data", []) or []):
                external.append(str(getattr(tensor, "name", "tensor")))
        if external:
            raise OperationError(
                "Portable ONNX artifacts must be self-contained; external tensor data found for %s"
                % ", ".join(external[:8])
            )
        onnx.checker.check_model(model)
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("ONNX model validation failed for %s: %s" % (path, exc)) from exc
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise OperationError(
            'Portable ONNX artifacts require `onnxruntime`. Install with `python -m '
            'pip install "noema-lab[onnx]"` in an installed environment, or '
            "`uv sync --extra onnx` in a source checkout."
        ) from exc
    try:
        options = ort.SessionOptions()
        options.enable_profiling = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_mem_pattern = False
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise OperationError("ONNX Runtime could not load %s: %s" % (path, exc)) from exc
    session_evidence = _verified_deterministic_session_evidence(session, ort)
    return PortableOnnxComponent(
        path=path,
        sha256=sha256,
        input_names=tuple(item.name for item in session.get_inputs()),
        output_names=tuple(item.name for item in session.get_outputs()),
        session=session,
        session_evidence=session_evidence,
    )


def _verified_deterministic_session_evidence(
    session: object,
    ort: object,
) -> Dict[str, Any]:
    providers = list(session.get_providers())
    options = session.get_session_options()
    checks = {
        "providers": providers == ["CPUExecutionProvider"],
        "execution_mode": options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL,
        "intra_op_num_threads": int(options.intra_op_num_threads) == 1,
        "inter_op_num_threads": int(options.inter_op_num_threads) == 1,
        "graph_optimization_level": (
            options.graph_optimization_level
            == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ),
        "enable_mem_pattern": options.enable_mem_pattern is False,
        "enable_profiling": options.enable_profiling is False,
    }
    if not all(checks.values()):
        failed = ", ".join(sorted(key for key, passed in checks.items() if not passed))
        raise OperationError(
            "ONNX Runtime deterministic session configuration drifted: %s" % failed
        )
    return {
        "providers": providers,
        "provider_fallback_permitted": False,
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "enable_mem_pattern": False,
        "enable_profiling": False,
    }


def _onnx_tensors(model: object) -> Iterable[object]:
    graph = getattr(model, "graph", None)
    if graph is None:
        return ()
    tensors = list(getattr(graph, "initializer", []) or [])
    sparse = list(getattr(graph, "sparse_initializer", []) or [])
    for value in sparse:
        tensors.append(getattr(value, "values", value))
        tensors.append(getattr(value, "indices", value))
    return tensors


def _batch_column(
    value: float | np.ndarray,
    batch: int,
    name: str,
    *,
    positive: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((batch, 1), float(array), dtype=np.float32)
    elif array.ndim == 1 and array.size == batch:
        array = array.reshape(batch, 1)
    elif array.shape == (batch, 1):
        array = np.asarray(array, dtype=np.float32)
    else:
        raise OperationError("%s must be scalar, [batch], or [batch,1]" % name)
    if not np.all(np.isfinite(array)):
        raise OperationError("%s must be finite" % name)
    if positive and np.any(array <= 0.0):
        raise OperationError("%s must be greater than zero" % name)
    if not positive and np.any(array < 0.0):
        raise OperationError("%s must be nonnegative" % name)
    return np.ascontiguousarray(array, dtype=np.float32)


def _project_rows_to_simplex(scores: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(scores, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1) - 1.0
    indices = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    active = sorted_values - cumulative / indices > 0.0
    rho = np.sum(active, axis=1) - 1
    theta = cumulative[np.arange(scores.shape[0]), rho] / (rho.astype(np.float64) + 1.0)
    projected = np.maximum(scores - theta[:, None], 0.0)
    totals = np.sum(projected, axis=1, keepdims=True)
    if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
        raise OperationError("Noema simplex projection produced an invalid row")
    return projected / totals


def _require_names(kind: str, actual: Sequence[str], expected: Sequence[str]) -> None:
    if not expected:
        return
    if tuple(actual) != tuple(str(name) for name in expected):
        raise OperationError(
            "ONNX %s signature mismatch: expected %s, got %s"
            % (kind, list(expected), list(actual))
        )


def _validated_sha256(value: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or text.lower() != text or any(char not in "0123456789abcdef" for char in text):
        raise OperationError("ONNX runtime requires a 64-character lowercase model_sha256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value: np.ndarray) -> bool:
    if np.issubdtype(value.dtype, np.number):
        return bool(np.all(np.isfinite(value)))
    return True
