from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple

import numpy as np

from noema_lab.core.operations import OperationError

BACKEND_OPTIONS = ("auto", "python_numpy", "cpp_native")


def backend_schema(default: str = "auto") -> dict:
    return {
        "type": "string",
        "default": default,
        "enum": list(BACKEND_OPTIONS),
        "description": (
            "Data-plane implementation for portable CPU kernels. auto resolves "
            "deterministically to Python / NumPy; select cpp_native explicitly "
            "to require the native extension."
        ),
        "x-noema-materialization-selector": {
            "target": "backend",
            "mapping": {
                "python_numpy": "numpy",
                "cpp_native": "cpp",
            },
            "automatic_values": ["auto"],
            "automatic_value": "python_numpy",
        },
    }


def normalize_backend(value: Any) -> str:
    backend = str(value or "auto").strip().lower()
    if backend in {"cpp", "native", "c++"}:
        backend = "cpp_native"
    if backend in {"python", "numpy"}:
        backend = "python_numpy"
    if backend not in BACKEND_OPTIONS:
        raise OperationError(
            "Unknown data_plane_backend %r; expected one of %s"
            % (value, ", ".join(BACKEND_OPTIONS))
        )
    return backend


def _kernel_backend(value: Any) -> str:
    requested = normalize_backend(value)
    return "cpp_native" if requested == "cpp_native" else "python_numpy"


def selected_backend(params: dict | None, kernel: str, prefer_cpp: bool | None = None) -> str:
    """Resolve a data-plane request without host-dependent auto switching.

    ``kernel`` and ``prefer_cpp`` remain accepted for API compatibility.  They
    no longer alter ``auto`` because a hashed execution plan must select the
    same implementation on hosts with and without the optional native module.
    """

    requested = normalize_backend((params or {}).get("data_plane_backend", "auto"))
    if requested == "python_numpy":
        return "python_numpy"
    if requested == "cpp_native":
        require_native()
        return "cpp_native"
    return "python_numpy"


@lru_cache(maxsize=1)
def _native_module():
    try:
        from noema_lab import _native_dataplane
    except Exception as exc:  # pragma: no cover - depends on local build
        return None, exc
    return _native_dataplane, None


def native_available() -> bool:
    module, _error = _native_module()
    return module is not None


def native_status() -> dict:
    module, error = _native_module()
    return {
        "available": module is not None,
        "error": "" if error is None else str(error),
    }


def require_native():
    module, error = _native_module()
    if module is None:
        raise OperationError(
            "data_plane_backend=cpp_native was requested, but noema_lab._native_dataplane is not built. "
            "Run `uv sync` or reinstall the project so the native extension is compiled. "
            "Import error: %s" % error
        )
    return module


def indices_to_bits(indices: np.ndarray, bits_per_index: int, backend: str = "auto") -> Tuple[np.ndarray, str]:
    flat = np.ascontiguousarray(indices.reshape(-1).astype(np.int64, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().indices_to_bits_i64(flat, int(bits_per_index)), selected
    shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
    return ((flat[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1), selected


def bits_to_indices(
    bits: np.ndarray,
    bits_per_index: int,
    shape: tuple,
    codebook_size: int,
    invalid_policy: str = "mod",
    backend: str = "auto",
) -> Tuple[np.ndarray, float, str]:
    bits, _validation_backend = require_canonical_bits(
        np.asarray(bits), "bits_to_indices", backend
    )
    count = int(np.prod(shape))
    needed = count * int(bits_per_index)
    if int(bits.size) != needed:
        raise OperationError(
            "bits_to_indices requires exactly %d bits for shape %s at %d "
            "bits per index; got %d"
            % (needed, tuple(int(item) for item in shape), int(bits_per_index), int(bits.size))
        )
    bits = np.ascontiguousarray(bits)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        values = require_native().bits_to_indices_i64(
            bits,
            int(bits_per_index),
            count,
            int(codebook_size),
            invalid_policy == "clamp",
        )
        if int(codebook_size) <= 0:
            invalid_fraction = 0.0
        else:
            raw_values = _bits_to_raw_index_values_numpy(bits, int(bits_per_index))
            invalid_fraction = float(np.mean(raw_values >= int(codebook_size))) if raw_values.size else 0.0
        return np.asarray(values, dtype=np.int64).reshape(shape), invalid_fraction, selected
    raw_values = _bits_to_raw_index_values_numpy(bits, int(bits_per_index))
    invalid_fraction = float(np.mean(raw_values >= int(codebook_size))) if raw_values.size else 0.0
    if invalid_policy == "clamp":
        values = np.clip(raw_values, 0, int(codebook_size) - 1)
    else:
        values = np.mod(raw_values, int(codebook_size))
    return values.reshape(shape).astype(np.int64, copy=False), invalid_fraction, selected


def _bits_to_raw_index_values_numpy(bits: np.ndarray, bits_per_index: int) -> np.ndarray:
    groups = bits.reshape(-1, bits_per_index).astype(np.int64, copy=False)
    shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
    return (groups * (1 << shifts[None, :])).sum(axis=1)


def repetition_encode(bits: np.ndarray, factor: int, backend: str = "auto") -> Tuple[np.ndarray, str]:
    bits, _validation_backend = require_canonical_bits(
        np.asarray(bits), "repetition_encode", backend
    )
    bits = np.ascontiguousarray(bits)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().repetition_encode(bits, int(factor)), selected
    return np.repeat(bits, int(factor)).astype(np.uint8, copy=False), selected


def repetition_decode(bits: np.ndarray, factor: int, payload_count: int, backend: str = "auto") -> Tuple[np.ndarray, str]:
    bits, _validation_backend = require_canonical_bits(
        np.asarray(bits), "repetition_decode", backend
    )
    bits = np.ascontiguousarray(bits)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().repetition_decode(bits, int(factor), int(payload_count)), selected
    usable_count = (bits.size // int(factor)) * int(factor)
    if usable_count == 0:
        return np.zeros((0,), dtype=np.uint8), selected
    groups = bits[:usable_count].reshape(-1, int(factor))
    return (groups.mean(axis=1) >= 0.5).astype(np.uint8, copy=False)[: int(payload_count)], selected


def bit_error_count(reference: np.ndarray, candidate: np.ndarray, backend: str = "auto") -> Tuple[int, str]:
    reference, _reference_validation_backend = require_canonical_bits(
        np.asarray(reference), "bit_error_count reference", backend
    )
    candidate, _candidate_validation_backend = require_canonical_bits(
        np.asarray(candidate), "bit_error_count candidate", backend
    )
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    count = min(int(reference.size), int(candidate.size))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return int(require_native().ber_count(reference[:count], candidate[:count])), selected
    return int(np.sum(reference[:count] != candidate[:count])), selected


def require_canonical_bits(bits: np.ndarray, label: str, backend: str = "auto") -> Tuple[np.ndarray, str]:
    if bits.dtype != np.dtype("uint8"):
        raise OperationError(
            "%s expected canonical channel bits as a 1-D np.uint8 array with values 0 or 1; got dtype %s"
            % (label, bits.dtype)
        )
    if bits.ndim != 1:
        raise OperationError(
            "%s expected canonical channel bits as a 1-D np.uint8 array; got shape %s"
            % (label, tuple(int(item) for item in bits.shape))
        )
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        try:
            return require_native().require_canonical_bits(bits), selected
        except RuntimeError as exc:
            raise OperationError("%s expected unpacked bit values 0 or 1: %s" % (label, exc)) from exc
    if bits.size:
        valid = np.logical_or(bits == 0, bits == 1)
        if not bool(np.all(valid)):
            unique = np.unique(bits[~valid])
            sample = ", ".join(str(int(item)) for item in unique[:5])
            raise OperationError(
                "%s expected unpacked bit values 0 or 1; found %s"
                % (label, sample or "non-binary values")
            )
    return np.ascontiguousarray(bits), selected


def qpsk_modulate(bits: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, str]:
    bits, selected = require_canonical_bits(
        np.asarray(bits), "qpsk_modulate", backend
    )
    bits = np.ascontiguousarray(bits)
    remainder = bits.size % 2
    padded = bits if remainder == 0 else np.pad(bits, (0, 2 - remainder), mode="constant")
    if selected == "cpp_native":
        return require_native().qpsk_modulate(bits), padded, selected
    pairs = padded.reshape(-1, 2)
    real = 1.0 - 2.0 * pairs[:, 0].astype(np.float32)
    imag = 1.0 - 2.0 * pairs[:, 1].astype(np.float32)
    symbols = (real + 1j * imag) / np.sqrt(2.0)
    return symbols.astype(np.complex64, copy=False), padded, selected


def qpsk_demodulate(symbols: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, str]:
    symbols = np.ascontiguousarray(symbols.astype(np.complex64, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().qpsk_demodulate(symbols), selected
    bits = np.empty((symbols.size, 2), dtype=np.uint8)
    bits[:, 0] = symbols.real < 0.0
    bits[:, 1] = symbols.imag < 0.0
    return bits.reshape(-1), selected


def bpsk_modulate(bits: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, str]:
    bits, selected = require_canonical_bits(
        np.asarray(bits), "bpsk_modulate", backend
    )
    bits = np.ascontiguousarray(bits)
    if selected == "cpp_native":
        return require_native().bpsk_modulate(bits), bits, selected
    symbols = (1.0 - 2.0 * bits.astype(np.float32)).astype(np.complex64)
    return symbols, bits, selected


def bpsk_demodulate(symbols: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, str]:
    symbols = np.ascontiguousarray(symbols.astype(np.complex64, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().bpsk_demodulate(symbols), selected
    return (symbols.real < 0.0).astype(np.uint8, copy=False), selected


def awgn_apply(symbols: np.ndarray, noise_real: np.ndarray, noise_imag: np.ndarray, scale: float, backend: str = "auto") -> Tuple[np.ndarray, str]:
    symbols = np.ascontiguousarray(symbols.astype(np.complex64, copy=False))
    noise_real = np.ascontiguousarray(noise_real.astype(np.float32, copy=False))
    noise_imag = np.ascontiguousarray(noise_imag.astype(np.float32, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().awgn_apply(symbols, noise_real, noise_imag, float(scale)), selected
    return symbols + (float(scale) * noise_real + 1j * float(scale) * noise_imag).astype(np.complex64), selected


def image_to_nchw(images: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, str]:
    images = np.ascontiguousarray(images.astype(np.uint8, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().image_to_nchw(images), selected
    return np.ascontiguousarray(np.transpose(images.astype(np.float32, copy=False) / 255.0, (0, 3, 1, 2))), selected


def nchw_to_image(tensor: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, str]:
    tensor = np.ascontiguousarray(tensor.astype(np.float32, copy=False))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().nchw_to_image(tensor), selected
    image = np.clip(np.rint(np.transpose(np.clip(tensor, 0.0, 1.0), (0, 2, 3, 1)) * 255.0), 0, 255)
    return image.astype(np.uint8, copy=False), selected


def float32_to_bits(values: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, str]:
    values = np.ascontiguousarray(values.astype(np.float32, copy=False).reshape(-1))
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().float32_to_bits(values), selected
    payload = values.tobytes(order="C")
    data = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(data).astype(np.uint8, copy=False), selected


def bits_to_float32(bits: np.ndarray, shape: tuple, backend: str = "auto") -> Tuple[np.ndarray, str]:
    bits, _validation_backend = require_canonical_bits(
        np.asarray(bits), "bits_to_float32", backend
    )
    bits = np.ascontiguousarray(bits)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        values = require_native().bits_to_float32(bits)
        return np.asarray(values, dtype=np.float32).reshape(shape), selected
    byte_count = int(np.prod(shape)) * np.dtype("float32").itemsize
    packed = np.packbits(bits.astype(np.uint8, copy=False))
    payload = packed[:byte_count].tobytes()
    return np.frombuffer(payload, dtype=np.float32).reshape(shape).astype(np.float32, copy=True), selected


def bytes_to_bits(payload: bytes, backend: str = "auto") -> Tuple[np.ndarray, str]:
    data = np.frombuffer(payload, dtype=np.uint8)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        return require_native().unpackbits_u8(data, int(data.size) * 8), selected
    return np.unpackbits(data).astype(np.uint8, copy=False), selected


def bits_to_bytes(bits: np.ndarray, byte_count: int, backend: str = "auto") -> Tuple[bytes, str]:
    bits, _validation_backend = require_canonical_bits(
        np.asarray(bits), "bits_to_bytes", backend
    )
    bits = np.ascontiguousarray(bits)
    selected = _kernel_backend(backend)
    if selected == "cpp_native":
        packed = require_native().packbits_u8(bits)
        return bytes(np.asarray(packed, dtype=np.uint8)[: int(byte_count)].tobytes()), selected
    packed = np.packbits(bits.astype(np.uint8, copy=False))
    return packed[: int(byte_count)].tobytes(), selected
