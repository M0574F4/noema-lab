from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from noema_lab.core import dataplane
from noema_lab.core.artifacts import file_sha256
from noema_lab.core.operations import OperationError

JsonDict = Dict[str, Any]


BOUNDARY_CONTRACTS: Dict[str, JsonDict] = {
    "payload.bits": {
        "dtype": "uint8",
        "shape": "[N]",
        "values": [0, 1],
        "storage": "unpacked_uint8",
        "description": "Payload bits are unpacked one bit per uint8 element.",
    },
    "channel.bits": {
        "dtype": "uint8",
        "shape": "[N]",
        "values": [0, 1],
        "storage": "unpacked_uint8",
        "description": "Channel-coded, demodulated, or fixed-point bits are unpacked one bit per uint8 element.",
    },
    "channel.symbols": {
        "dtype": ["complex64", "float32"],
        "shape": "[N] or [N, D] depending on representation",
        "values": "finite",
        "description": "Continuous channel symbols. Complex baseband uses complex64; real-valued PHY surrogates may use float32.",
    },
    "image.tensor": {
        "dtype": ["uint8", "float32"],
        "shape": "[N, H, W, C]",
        "values": "uint8 in [0,255] or float32 in [0,1]",
        "description": "Batched image tensor with explicit channel dimension.",
    },
    "text.utf8_bytes": {
        "dtype": "bytes or uint8",
        "shape": "[N] for uint8 arrays",
        "values": "valid UTF-8 byte sequence",
        "description": "Text payload represented as valid UTF-8 bytes.",
    },
    "semantic.embedding": {
        "dtype": "float32",
        "shape": "[N, D]",
        "values": "finite",
        "description": "Dense semantic embedding batch.",
    },
    "metric.scalar": {
        "dtype": "number",
        "shape": "scalar",
        "values": "finite",
        "description": "Finite scalar metric value with an explicit unit and direction in metadata when reported.",
    },
    "artifact.file": {
        "dtype": "file",
        "shape": "path",
        "values": "existing readable file; hash must match if declared",
        "description": "Persistent artifact file referenced from a run, capture, benchmark, or manifest.",
    },
}

RATE_ACCOUNTING_UNITS = {"bits", "bytes", "symbols", "channel_uses", "pixels", "samples", "seconds"}


def boundary_contract(contract_id: str) -> JsonDict:
    if contract_id not in BOUNDARY_CONTRACTS:
        raise OperationError("Unknown boundary contract: %s" % contract_id)
    return dict(BOUNDARY_CONTRACTS[contract_id])


def validate_payload_bits(bits: np.ndarray, label: str = "payload.bits", backend: str = "auto") -> Tuple[np.ndarray, JsonDict]:
    """Validate canonical unpacked bit arrays and return a contiguous uint8 vector."""
    array = np.asarray(bits)
    canonical, selected_backend = dataplane.require_canonical_bits(array, label, backend)
    metadata = {
        "boundary_contract": "payload.bits",
        "bit_count": int(canonical.size),
        "byte_count": int(math.ceil(float(canonical.size) / 8.0)),
        "dtype": "uint8",
        "shape": [int(item) for item in canonical.shape],
        "storage": "unpacked_uint8",
        "bits_per_element": 1,
        "data_plane_backend": selected_backend,
    }
    return canonical, metadata


def validate_channel_bits(bits: np.ndarray, label: str = "channel.bits", backend: str = "auto") -> Tuple[np.ndarray, JsonDict]:
    canonical, metadata = validate_payload_bits(bits, label=label, backend=backend)
    metadata["boundary_contract"] = "channel.bits"
    return canonical, metadata


def validate_channel_symbols(
    symbols: np.ndarray,
    label: str = "channel.symbols",
    representation: str = "complex",
) -> Tuple[np.ndarray, JsonDict]:
    """Validate continuous channel symbols.

    `representation="complex"` enforces complex64 baseband symbols. `representation="real"`
    enforces float32 real-valued symbols. `representation="auto"` accepts either and canonicalizes
    to complex64 or float32 respectively.
    """
    representation = str(representation or "complex").strip().lower()
    if representation not in {"complex", "real", "auto"}:
        raise OperationError("Symbol boundary %s has unknown representation %r" % (label, representation))
    array = np.asarray(symbols)
    if representation == "complex" or (representation == "auto" and np.issubdtype(array.dtype, np.complexfloating)):
        if not np.issubdtype(array.dtype, np.complexfloating):
            raise OperationError("Symbol boundary %s requires complex64 symbols, got %s" % (label, array.dtype))
        canonical = array.reshape(-1).astype(np.complex64, copy=False)
        if not bool(np.all(np.isfinite(canonical.real))) or not bool(np.all(np.isfinite(canonical.imag))):
            raise OperationError("Symbol boundary %s received non-finite complex symbols" % label)
        return canonical, {
            "boundary_contract": "channel.symbols",
            "representation": "complex_baseband",
            "symbol_count": int(canonical.size),
            "channel_use_count": int(canonical.size),
            "dtype": "complex64",
            "shape": [int(item) for item in canonical.shape],
            "storage": "complex64",
        }
    if representation in {"real", "auto"}:
        if not np.issubdtype(array.dtype, np.floating):
            raise OperationError("Symbol boundary %s requires float32 real-valued symbols, got %s" % (label, array.dtype))
        canonical = array.astype(np.float32, copy=False)
        if not bool(np.all(np.isfinite(canonical))):
            raise OperationError("Symbol boundary %s received non-finite real-valued symbols" % label)
        return canonical, {
            "boundary_contract": "channel.symbols",
            "representation": "real_valued",
            "symbol_count": int(canonical.shape[0]) if canonical.ndim else int(canonical.size),
            "channel_use_count": int(canonical.shape[0]) if canonical.ndim else int(canonical.size),
            "dtype": "float32",
            "shape": [int(item) for item in canonical.shape],
            "storage": "float32",
        }
    raise OperationError("Symbol boundary %s could not validate symbols with dtype %s" % (label, array.dtype))


def validate_image_tensor(images: np.ndarray, label: str = "image.tensor") -> Tuple[np.ndarray, JsonDict]:
    array = np.asarray(images)
    if array.ndim != 4:
        raise OperationError("%s expects shape [N, H, W, C], got %s" % (label, tuple(int(item) for item in array.shape)))
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1 or array.shape[3] not in {1, 3, 4}:
        raise OperationError("%s expects non-empty [N, H, W, C] with C in {1,3,4}, got %s" % (label, tuple(int(item) for item in array.shape)))
    if array.dtype == np.dtype("uint8"):
        canonical = np.ascontiguousarray(array)
    elif array.dtype == np.dtype("float32"):
        if not bool(np.all(np.isfinite(array))):
            raise OperationError("%s received non-finite float32 image values" % label)
        if array.size and (float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0):
            raise OperationError("%s expects float32 image values in [0, 1]" % label)
        canonical = np.ascontiguousarray(array)
    else:
        raise OperationError("%s expects uint8 or float32 images, got %s" % (label, array.dtype))
    return canonical, {
        "boundary_contract": "image.tensor",
        "dtype": str(canonical.dtype),
        "shape": [int(item) for item in canonical.shape],
        "image_count": int(canonical.shape[0]),
        "height": int(canonical.shape[1]),
        "width": int(canonical.shape[2]),
        "channels": int(canonical.shape[3]),
    }


def validate_text_utf8_bytes(value: Any, label: str = "text.utf8_bytes") -> Tuple[bytes, JsonDict]:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        array = np.asarray(value)
        if array.dtype != np.dtype("uint8") or array.ndim != 1:
            raise OperationError("%s expects bytes, str, or a 1-D uint8 byte array" % label)
        raw = bytes(array.tolist())
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperationError("%s is not valid UTF-8: %s" % (label, exc)) from exc
    return raw, {"boundary_contract": "text.utf8_bytes", "byte_count": len(raw), "encoding": "utf-8"}


def validate_semantic_embedding(embedding: np.ndarray, label: str = "semantic.embedding") -> Tuple[np.ndarray, JsonDict]:
    array = np.asarray(embedding)
    if array.ndim != 2:
        raise OperationError("%s expects shape [N, D], got %s" % (label, tuple(int(item) for item in array.shape)))
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise OperationError("%s expects a non-empty embedding batch, got %s" % (label, tuple(int(item) for item in array.shape)))
    if not np.issubdtype(array.dtype, np.floating):
        raise OperationError("%s expects floating embeddings, got %s" % (label, array.dtype))
    canonical = array.astype(np.float32, copy=False)
    if not bool(np.all(np.isfinite(canonical))):
        raise OperationError("%s received non-finite embedding values" % label)
    return np.ascontiguousarray(canonical), {
        "boundary_contract": "semantic.embedding",
        "dtype": "float32",
        "shape": [int(item) for item in canonical.shape],
        "embedding_count": int(canonical.shape[0]),
        "embedding_dim": int(canonical.shape[1]),
    }


def validate_metric_scalar(value: Any, label: str = "metric.scalar", non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise OperationError("%s expects a numeric scalar, got bool" % label)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise OperationError("%s expects a numeric scalar, got %r" % (label, value)) from exc
    if not math.isfinite(numeric):
        raise OperationError("%s expects a finite numeric scalar, got %r" % (label, value))
    if non_negative and numeric < 0.0:
        raise OperationError("%s expects a non-negative scalar, got %r" % (label, value))
    return numeric


def validate_rate_accounting_point(
    name: str,
    value: Any,
    unit: str,
    integer: bool = True,
) -> int | float:
    unit = str(unit)
    if unit not in RATE_ACCOUNTING_UNITS:
        raise OperationError("Rate accounting point %s has unknown unit %s" % (name, unit))
    numeric = validate_metric_scalar(value, label="rate accounting point %s" % name, non_negative=True)
    if integer:
        rounded = int(numeric)
        if float(rounded) != numeric:
            raise OperationError("Rate accounting point %s must be an integer %s value" % (name, unit))
        return rounded
    return numeric


def validate_bits_per_pixel(bit_count: Any, pixel_count: Any, recorded_bpp: Optional[Any] = None, tolerance: float = 1e-9) -> float:
    bits = validate_rate_accounting_point("bit_count", bit_count, "bits")
    pixels = validate_rate_accounting_point("pixel_count", pixel_count, "pixels")
    if pixels <= 0:
        raise OperationError("Cannot compute bits per pixel with pixel_count=%d" % pixels)
    computed = float(bits) / float(pixels)
    if recorded_bpp is not None:
        recorded = validate_metric_scalar(recorded_bpp, label="recorded bits per pixel", non_negative=True)
        if abs(computed - recorded) > tolerance:
            raise OperationError("Recorded bpp %.12g does not match bit_count/pixel_count %.12g" % (recorded, computed))
    return computed


def validate_artifact_file(artifact: Any, label: str = "artifact.file", expected_kind: Optional[str] = None) -> JsonDict:
    kind = str(getattr(artifact, "kind", ""))
    path = Path(getattr(artifact, "path", ""))
    if expected_kind is not None and kind != expected_kind:
        raise OperationError("%s expected artifact kind %s, got %s" % (label, expected_kind, kind))
    if not path.is_file():
        raise OperationError("%s path does not exist or is not a file: %s" % (label, path))
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    recorded_hash = getattr(artifact, "sha256", None)
    if recorded_hash:
        actual = file_sha256(path)
        if actual != recorded_hash:
            raise OperationError("%s SHA-256 mismatch for %s" % (label, path))
    return {
        "boundary_contract": "artifact.file",
        "kind": kind,
        "path": str(path),
        "sha256": recorded_hash or file_sha256(path),
        "metadata_keys": sorted(str(key) for key in metadata.keys()),
    }


def validate_npz_artifact_schema(
    artifact: Any,
    required_arrays: Iterable[str],
    label: str = "artifact.file",
    expected_kind: Optional[str] = None,
) -> JsonDict:
    report = validate_artifact_file(artifact, label=label, expected_kind=expected_kind)
    path = Path(getattr(artifact, "path", ""))
    if path.suffix != ".npz":
        raise OperationError("%s expects an .npz artifact, got %s" % (label, path.name))
    with np.load(str(path), allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files if name != "metadata_json"}
        missing = [name for name in required_arrays if name not in arrays]
        if missing:
            raise OperationError("%s missing required array(s): %s" % (label, ", ".join(missing)))
        report["arrays"] = {
            name: {"dtype": str(value.dtype), "shape": [int(item) for item in value.shape]}
            for name, value in arrays.items()
        }
    return report
