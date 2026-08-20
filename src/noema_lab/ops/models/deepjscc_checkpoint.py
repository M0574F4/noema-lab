from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from noema_lab.core.operations import OperationError
from noema_lab.core.structured_input import decode_strict_json_object


CHECKPOINT_FORMAT = "noema_deepjscc_reference_cnn_npz_v1"
CHECKPOINT_KIND = "noema.deepjscc_checkpoint"
CHECKPOINT_ARCHITECTURE = "reference_cnn_v1"

_WEIGHT_NAMES = {
    "encoder_0_weight",
    "encoder_0_bias",
    "encoder_2_weight",
    "encoder_2_bias",
    "decoder_0_weight",
    "decoder_0_bias",
    "decoder_2_weight",
    "decoder_2_bias",
}


@dataclass(frozen=True)
class DeepJsccReferenceCheckpoint:
    path: Path
    sha256: str
    metadata: Dict[str, Any]
    symbol_channels: int
    encoder_0_weight: np.ndarray
    encoder_0_bias: np.ndarray
    encoder_2_weight: np.ndarray
    encoder_2_bias: np.ndarray
    decoder_0_weight: np.ndarray
    decoder_0_bias: np.ndarray
    decoder_2_weight: np.ndarray
    decoder_2_bias: np.ndarray


def load_deepjscc_reference_checkpoint(
    checkpoint_path: str,
    expected_sha256: str,
    *,
    checkpoint_format: str = CHECKPOINT_FORMAT,
    strict: bool = True,
    max_bytes: int = 64 * 1024 * 1024,
) -> DeepJsccReferenceCheckpoint:
    """Load the fixed reference-CNN format without executing checkpoint code."""

    raw_path = str(checkpoint_path or "").strip()
    if not raw_path:
        raise OperationError("runtime=learned_checkpoint requires params.checkpoint_path")
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OperationError("DeepJSCC checkpoint is not a readable file: %s" % path) from exc
    if not resolved.is_file():
        raise OperationError("DeepJSCC checkpoint is not a regular file: %s" % resolved)
    if resolved.suffix.lower() != ".npz":
        raise OperationError("DeepJSCC checkpoint must use the safe .npz format")

    maximum = max(1, int(max_bytes))
    size = int(resolved.stat().st_size)
    if size <= 0 or size > maximum:
        raise OperationError(
            "DeepJSCC checkpoint size %d bytes is outside the allowed range (max %d)"
            % (size, maximum)
        )

    requested_format = str(checkpoint_format or "").strip()
    if requested_format != CHECKPOINT_FORMAT:
        raise OperationError(
            "Unsupported DeepJSCC checkpoint format %r; expected %r"
            % (requested_format, CHECKPOINT_FORMAT)
        )

    declared_sha = str(expected_sha256 or "").strip()
    if (
        len(declared_sha) != 64
        or declared_sha.lower() != declared_sha
        or any(char not in "0123456789abcdef" for char in declared_sha)
    ):
        raise OperationError(
            "runtime=learned_checkpoint requires a 64-character lowercase checkpoint_sha256"
        )
    actual_sha = _file_sha256(resolved)
    if actual_sha != declared_sha:
        raise OperationError(
            "DeepJSCC checkpoint SHA-256 mismatch: expected %s, got %s"
            % (declared_sha, actual_sha)
        )

    required = _WEIGHT_NAMES | {"metadata_json"}
    try:
        with np.load(str(resolved), allow_pickle=False) as payload:
            names = set(payload.files)
            missing = required - names
            if missing:
                raise OperationError(
                    "DeepJSCC checkpoint is missing array(s): %s"
                    % ", ".join(sorted(missing))
                )
            extra = names - required
            if bool(strict) and extra:
                raise OperationError(
                    "DeepJSCC checkpoint has unexpected array(s): %s"
                    % ", ".join(sorted(extra))
                )
            metadata = _decode_metadata(payload["metadata_json"])
            arrays = {name: np.asarray(payload[name]) for name in _WEIGHT_NAMES}
    except OperationError:
        raise
    except Exception as exc:
        raise OperationError("Could not read DeepJSCC checkpoint: %s" % resolved) from exc

    symbol_channels = _validate_metadata(metadata)
    for name, value in arrays.items():
        if value.dtype != np.float32:
            raise OperationError(
                "DeepJSCC checkpoint array %s must have dtype float32; got %s"
                % (name, value.dtype)
            )
        if not np.all(np.isfinite(value)):
            raise OperationError(
                "DeepJSCC checkpoint array %s contains NaN or infinite values" % name
            )

    expected_shapes = {
        "encoder_0_weight": (32, 3, 3, 3),
        "encoder_0_bias": (32,),
        "encoder_2_weight": (2 * symbol_channels, 32, 3, 3),
        "encoder_2_bias": (2 * symbol_channels,),
        # PyTorch ConvTranspose2d stores [in_channels, out_channels, kH, kW].
        "decoder_0_weight": (2 * symbol_channels, 32, 4, 4),
        "decoder_0_bias": (32,),
        "decoder_2_weight": (32, 3, 4, 4),
        "decoder_2_bias": (3,),
    }
    for name, shape in expected_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise OperationError(
                "DeepJSCC checkpoint array %s must have shape %s; got %s"
                % (name, list(shape), list(arrays[name].shape))
            )

    return DeepJsccReferenceCheckpoint(
        path=resolved,
        sha256=actual_sha,
        metadata=metadata,
        symbol_channels=symbol_channels,
        **arrays,
    )


def encode_deepjscc_images(
    checkpoint: DeepJsccReferenceCheckpoint,
    images: np.ndarray,
) -> np.ndarray:
    values = np.asarray(images)
    if values.dtype != np.uint8:
        raise OperationError("DeepJSCC reference encoder expects uint8 images")
    if values.ndim != 4 or values.shape[-1] != 3:
        raise OperationError("DeepJSCC reference encoder expects images shaped [N,H,W,3]")
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise OperationError("DeepJSCC reference encoder received an empty image dimension")

    torch = _require_torch()
    model = _torch_model(checkpoint, torch)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(values))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        / 255.0
    )
    with torch.inference_mode():
        symbols = model.encode(tensor).detach().cpu().numpy()
    result = np.asarray(symbols, dtype=np.complex64)
    expected_channels = int(checkpoint.symbol_channels)
    if result.ndim != 4 or result.shape[1] != expected_channels:
        raise OperationError("DeepJSCC reference encoder produced an invalid symbol tensor")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise OperationError("DeepJSCC reference encoder produced non-finite symbols")
    return result


def decode_deepjscc_symbols(
    checkpoint: DeepJsccReferenceCheckpoint,
    symbols: np.ndarray,
    *,
    image_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    values = np.asarray(symbols)
    if values.ndim != 4:
        raise OperationError(
            "DeepJSCC reference decoder expects symbols shaped [N,C,H,W]"
        )
    if values.shape[0] < 1 or values.shape[1] != checkpoint.symbol_channels:
        raise OperationError(
            "DeepJSCC reference decoder expected %d complex symbol channels; got shape %s"
            % (checkpoint.symbol_channels, list(values.shape))
        )
    if not np.issubdtype(values.dtype, np.complexfloating):
        raise OperationError("DeepJSCC reference decoder expects a complex symbol tensor")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise OperationError("DeepJSCC reference decoder received NaN or infinite symbols")

    requested_shape = _validate_image_shape(image_shape, batch_size=int(values.shape[0]))
    torch = _require_torch()
    model = _torch_model(checkpoint, torch)
    tensor = torch.from_numpy(np.ascontiguousarray(values.astype(np.complex64, copy=False)))
    with torch.inference_mode():
        decoded = model.decode(tensor).clamp(0.0, 1.0)
        decoded = decoded.permute(0, 2, 3, 1).detach().cpu().numpy()
    if requested_shape is not None:
        _batch, height, width, _channels = requested_shape
        if decoded.shape[1] < height or decoded.shape[2] < width:
            raise OperationError(
                "DeepJSCC decoder output %s is smaller than the declared image shape %s"
                % (list(decoded.shape), list(requested_shape))
            )
        decoded = decoded[:, :height, :width, :]
    if decoded.ndim != 4 or decoded.shape[-1] != 3 or not np.all(np.isfinite(decoded)):
        raise OperationError("DeepJSCC reference decoder produced an invalid image tensor")
    return np.rint(decoded * 255.0).astype(np.uint8)


def symbol_shape_from_metadata(
    metadata: Mapping[str, Any],
    symbol_count: int,
    expected_channels: int,
) -> Tuple[int, int, int, int]:
    raw_shape = metadata.get("symbol_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
        raise OperationError(
            "DeepJSCC learned-checkpoint decoding requires symbol_shape=[N,C,H,W] metadata"
        )
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise OperationError("DeepJSCC symbol_shape metadata must contain integers") from exc
    if any(value < 1 for value in shape):
        raise OperationError("DeepJSCC symbol_shape metadata must contain positive dimensions")
    if shape[1] != int(expected_channels):
        raise OperationError(
            "DeepJSCC symbol_shape declares %d channels; checkpoint expects %d"
            % (shape[1], expected_channels)
        )
    if int(np.prod(shape, dtype=np.int64)) != int(symbol_count):
        raise OperationError(
            "DeepJSCC symbol_shape %s does not match the received symbol count %d"
            % (list(shape), int(symbol_count))
        )
    return shape


def _validate_metadata(metadata: Mapping[str, Any]) -> int:
    expected = {
        "schema_version": 1,
        "kind": CHECKPOINT_KIND,
        "format": CHECKPOINT_FORMAT,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "input_channels": 3,
        "output_channels": 3,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise OperationError(
                "DeepJSCC checkpoint metadata %s must be %r; got %r"
                % (key, value, metadata.get(key))
            )
    raw_channels = metadata.get("symbol_channels")
    if isinstance(raw_channels, bool):
        raise OperationError("DeepJSCC checkpoint symbol_channels must be an integer")
    try:
        symbol_channels = int(raw_channels)
    except (TypeError, ValueError) as exc:
        raise OperationError("DeepJSCC checkpoint symbol_channels must be an integer") from exc
    if raw_channels != symbol_channels or symbol_channels < 1 or symbol_channels > 1024:
        raise OperationError("DeepJSCC checkpoint symbol_channels must be between 1 and 1024")
    return symbol_channels


def _decode_metadata(value: np.ndarray) -> Dict[str, Any]:
    try:
        if int(value.size) != 1:
            raise ValueError("metadata is not scalar")
        raw = value.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = decode_strict_json_object(
            str(raw),
            label="DeepJSCC checkpoint metadata_json",
        )
    except Exception as exc:
        raise OperationError(
            "DeepJSCC checkpoint metadata_json must contain one unambiguous "
            "UTF-8 JSON object: %s" % exc
        ) from exc
    return payload


def _validate_image_shape(
    image_shape: Optional[Sequence[int]],
    *,
    batch_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    if image_shape is None:
        return None
    if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 4:
        raise OperationError("DeepJSCC image_shape metadata must be [N,H,W,3]")
    try:
        shape = tuple(int(value) for value in image_shape)
    except (TypeError, ValueError) as exc:
        raise OperationError("DeepJSCC image_shape metadata must contain integers") from exc
    if any(value < 1 for value in shape) or shape[0] != batch_size or shape[3] != 3:
        raise OperationError(
            "DeepJSCC image_shape metadata must match the batch and have three channels"
        )
    return shape


_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _torch_model(checkpoint: DeepJsccReferenceCheckpoint, torch):
    key = (str(checkpoint.path), checkpoint.sha256)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    class ReferenceDeepJsccModel(torch.nn.Module):
        def __init__(self, symbol_channels: int):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(32, symbol_channels * 2, 3, stride=2, padding=1),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(symbol_channels * 2, 32, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                torch.nn.Sigmoid(),
            )

        def encode(self, images):
            features = self.encoder(images.float())
            real, imag = torch.chunk(features, 2, dim=1)
            return torch.complex(real, imag)

        def decode(self, rx_symbols):
            features = torch.cat([rx_symbols.real, rx_symbols.imag], dim=1)
            return self.decoder(features)

    candidate = ReferenceDeepJsccModel(checkpoint.symbol_channels)
    state = {
        "encoder.0.weight": torch.from_numpy(checkpoint.encoder_0_weight.copy()),
        "encoder.0.bias": torch.from_numpy(checkpoint.encoder_0_bias.copy()),
        "encoder.2.weight": torch.from_numpy(checkpoint.encoder_2_weight.copy()),
        "encoder.2.bias": torch.from_numpy(checkpoint.encoder_2_bias.copy()),
        "decoder.0.weight": torch.from_numpy(checkpoint.decoder_0_weight.copy()),
        "decoder.0.bias": torch.from_numpy(checkpoint.decoder_0_bias.copy()),
        "decoder.2.weight": torch.from_numpy(checkpoint.decoder_2_weight.copy()),
        "decoder.2.bias": torch.from_numpy(checkpoint.decoder_2_bias.copy()),
    }
    try:
        candidate.load_state_dict(state, strict=True)
    except Exception as exc:
        raise OperationError("Could not materialize the DeepJSCC reference CNN") from exc
    candidate.eval()
    _MODEL_CACHE[key] = candidate
    return candidate


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise OperationError(
            'DeepJSCC learned-checkpoint runtime requires PyTorch. Install with '
            '`python -m pip install "noema-lab[wireless]"` in an installed '
            "environment, or `uv sync --extra wireless` in a source checkout."
        ) from exc
    return torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
