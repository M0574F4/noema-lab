"""Bounded compact wire formats for codec payloads sent over a modeled link.

The general Noema safe-data format is appropriate for artifact interchange,
but its JSON/base64 envelope is not a representative communication wire
format.  These codec-specific formats retain only information required by the
receiver and keep independently protected source items independently
decodable.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Any


PAYLOAD_FORMAT = "noema.codec_wire.v1"
JPEG_MAGIC = b"NOEMAJP\x01"
COMPRESSAI_MAGIC = b"NOEMACA\x01"
MAX_WIRE_BYTES = 256 * 1024 * 1024
MAX_TEXT_BYTES = 4096
MAX_DIMENSIONS = 8
MAX_STREAM_GROUPS = 64
MAX_STREAMS_PER_GROUP = 1024


class CodecWireError(ValueError):
    """The compact codec payload is malformed, unsupported, or unbounded."""


class _Reader:
    def __init__(self, raw: bytes) -> None:
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise CodecWireError("codec wire payload must be bytes-like")
        self.raw = bytes(raw)
        if len(self.raw) > MAX_WIRE_BYTES:
            raise CodecWireError("codec wire payload exceeds the wire-size limit")
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.raw):
            raise CodecWireError("codec wire payload is truncated")
        value = self.raw[self.offset : self.offset + count]
        self.offset += count
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        try:
            return struct.unpack(fmt, self.take(size))
        except struct.error as exc:
            raise CodecWireError("codec wire scalar is malformed") from exc

    def finish(self) -> None:
        if self.offset != len(self.raw):
            raise CodecWireError("codec wire payload has trailing bytes")


def _bounded_text(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise CodecWireError(f"{label} must be a string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CodecWireError(f"{label} must be valid UTF-8") from exc
    if not encoded or len(encoded) > MAX_TEXT_BYTES:
        raise CodecWireError(
            f"{label} must contain between 1 and {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return encoded


def _read_text(reader: _Reader, label: str) -> str:
    (count,) = reader.unpack(">H")
    if count < 1 or count > MAX_TEXT_BYTES:
        raise CodecWireError(f"{label} length is outside the supported bounds")
    try:
        return reader.take(count).decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CodecWireError(f"{label} is not valid UTF-8") from exc


def _bounded_dimensions(value: Any, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        raise CodecWireError(f"{label} must be an integer sequence")
    dimensions = [int(item) for item in value]
    if not dimensions or len(dimensions) > MAX_DIMENSIONS:
        raise CodecWireError(f"{label} has an unsupported rank")
    if any(item < 1 or item > 0xFFFFFFFF for item in dimensions):
        raise CodecWireError(f"{label} contains an unsupported dimension")
    return dimensions


def _read_dimensions(reader: _Reader, label: str) -> list[int]:
    (rank,) = reader.unpack(">B")
    if rank < 1 or rank > MAX_DIMENSIONS:
        raise CodecWireError(f"{label} has an unsupported rank")
    dimensions = [int(reader.unpack(">I")[0]) for _ in range(rank)]
    if any(item < 1 for item in dimensions):
        raise CodecWireError(f"{label} contains a zero dimension")
    return dimensions


def _jpeg_dumps(payload: Mapping[str, Any]) -> bytes:
    entry = payload.get("entry")
    if not isinstance(entry, Mapping):
        raise CodecWireError("JPEG payload must contain an entry mapping")
    data = entry.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CodecWireError("JPEG entry data must be bytes-like")
    encoded = bytes(data)
    if len(encoded) > MAX_WIRE_BYTES - len(JPEG_MAGIC) - 4:
        raise CodecWireError("JPEG entropy stream exceeds the wire-size limit")
    return JPEG_MAGIC + struct.pack(">I", len(encoded)) + encoded


def _jpeg_loads(raw: bytes) -> dict[str, Any]:
    reader = _Reader(raw)
    if reader.take(len(JPEG_MAGIC)) != JPEG_MAGIC:
        raise CodecWireError("JPEG codec wire magic/version is invalid")
    (count,) = reader.unpack(">I")
    data = reader.take(count)
    reader.finish()
    return {
        "payload_version": 2,
        "codec": "jpeg",
        "entry": {"data": data},
    }


def _compressai_groups(value: Any) -> list[list[bytes]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        raise CodecWireError("CompressAI strings must be a sequence of groups")
    if len(value) > MAX_STREAM_GROUPS:
        raise CodecWireError("CompressAI payload has too many stream groups")
    groups: list[list[bytes]] = []
    for group in value:
        if not isinstance(group, Sequence) or isinstance(
            group, (str, bytes, bytearray, memoryview)
        ):
            raise CodecWireError(
                "each CompressAI stream group must be a sequence of byte strings"
            )
        if len(group) > MAX_STREAMS_PER_GROUP:
            raise CodecWireError("CompressAI stream group is too large")
        encoded_group: list[bytes] = []
        for stream in group:
            if not isinstance(stream, (bytes, bytearray, memoryview)):
                raise CodecWireError("CompressAI entropy streams must be bytes-like")
            encoded_group.append(bytes(stream))
        groups.append(encoded_group)
    return groups


def _compressai_dumps(payload: Mapping[str, Any]) -> bytes:
    entry = payload.get("entry")
    if not isinstance(entry, Mapping):
        raise CodecWireError("CompressAI payload must contain an entry mapping")
    model = _bounded_text(payload.get("model"), "CompressAI model")
    metric = _bounded_text(payload.get("metric"), "CompressAI metric")
    quality = int(payload.get("quality", 0))
    if quality < 1 or quality > 255:
        raise CodecWireError("CompressAI quality must be in [1, 255]")
    pretrained = payload.get("pretrained")
    if type(pretrained) is not bool:
        raise CodecWireError("CompressAI pretrained flag must be boolean")

    original_shape = _bounded_dimensions(
        entry.get("original_shape"), "CompressAI original shape"
    )
    latent_shape = _bounded_dimensions(entry.get("shape"), "CompressAI latent shape")
    groups = _compressai_groups(entry.get("strings"))
    vbr = payload.get("vbr") or {"enabled": False}
    if not isinstance(vbr, Mapping):
        raise CodecWireError("CompressAI VBR metadata must be a mapping")
    vbr_enabled = bool(vbr.get("enabled", False))
    vbr_values = [
        int(vbr.get("scale_index", 0) or 0),
        int(vbr.get("stage", 0) or 0),
        int(vbr.get("levels", 0) or 0),
    ]
    if any(item < 0 or item > 0xFFFF for item in vbr_values):
        raise CodecWireError("CompressAI VBR values exceed compact-wire bounds")

    parts = [
        COMPRESSAI_MAGIC,
        struct.pack(">H", len(model)),
        model,
        struct.pack(">B", quality),
        struct.pack(">H", len(metric)),
        metric,
        struct.pack(">B", int(pretrained)),
        struct.pack(">BHHH", int(vbr_enabled), *vbr_values),
        struct.pack(">B", len(original_shape)),
        b"".join(struct.pack(">I", item) for item in original_shape),
        struct.pack(">B", len(latent_shape)),
        b"".join(struct.pack(">I", item) for item in latent_shape),
        struct.pack(">H", len(groups)),
    ]
    for group in groups:
        parts.append(struct.pack(">H", len(group)))
        for stream in group:
            if len(stream) > 0xFFFFFFFF:
                raise CodecWireError("CompressAI entropy stream is too large")
            parts.extend((struct.pack(">I", len(stream)), stream))
    raw = b"".join(parts)
    if len(raw) > MAX_WIRE_BYTES:
        raise CodecWireError("CompressAI payload exceeds the wire-size limit")
    return raw


def _compressai_loads(raw: bytes) -> dict[str, Any]:
    reader = _Reader(raw)
    if reader.take(len(COMPRESSAI_MAGIC)) != COMPRESSAI_MAGIC:
        raise CodecWireError("CompressAI codec wire magic/version is invalid")
    model = _read_text(reader, "CompressAI model")
    (quality,) = reader.unpack(">B")
    metric = _read_text(reader, "CompressAI metric")
    (pretrained_raw,) = reader.unpack(">B")
    if pretrained_raw not in (0, 1):
        raise CodecWireError("CompressAI pretrained flag is invalid")
    vbr_enabled, scale_index, stage, levels = reader.unpack(">BHHH")
    if vbr_enabled not in (0, 1):
        raise CodecWireError("CompressAI VBR flag is invalid")
    original_shape = _read_dimensions(reader, "CompressAI original shape")
    latent_shape = _read_dimensions(reader, "CompressAI latent shape")
    (group_count,) = reader.unpack(">H")
    if group_count > MAX_STREAM_GROUPS:
        raise CodecWireError("CompressAI payload has too many stream groups")
    groups: list[list[bytes]] = []
    for _ in range(group_count):
        (stream_count,) = reader.unpack(">H")
        if stream_count > MAX_STREAMS_PER_GROUP:
            raise CodecWireError("CompressAI stream group is too large")
        group = []
        for _ in range(stream_count):
            (count,) = reader.unpack(">I")
            group.append(reader.take(count))
        groups.append(group)
    reader.finish()
    vbr: dict[str, Any]
    if vbr_enabled:
        vbr = {
            "enabled": True,
            "scale_index": int(scale_index),
            "stage": int(stage),
            "levels": int(levels),
        }
    else:
        vbr = {"enabled": False}
    return {
        "payload_version": 3,
        "codec": "compressai",
        "model": model,
        "quality": int(quality),
        "metric": metric,
        "pretrained": bool(pretrained_raw),
        "vbr": vbr,
        "entry": {
            "strings": groups,
            "shape": latent_shape,
            "original_shape": original_shape,
        },
    }


def dumps(payload: Mapping[str, Any]) -> bytes:
    """Serialize one independently decodable JPEG or CompressAI source item."""

    if not isinstance(payload, Mapping):
        raise CodecWireError("codec wire payload must be a mapping")
    codec = str(payload.get("codec") or "")
    if codec == "jpeg":
        return _jpeg_dumps(payload)
    if codec == "compressai":
        return _compressai_dumps(payload)
    raise CodecWireError(f"unsupported compact codec wire payload: {codec or '<missing>'}")


def loads(raw: bytes) -> dict[str, Any]:
    """Parse one compact codec item without importing or constructing objects."""

    wire = bytes(raw)
    if wire.startswith(JPEG_MAGIC):
        return _jpeg_loads(wire)
    if wire.startswith(COMPRESSAI_MAGIC):
        return _compressai_loads(wire)
    raise CodecWireError("codec wire magic/version is unsupported")
