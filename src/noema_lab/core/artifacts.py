from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from noema_lab.core.structured_input import decode_strict_json_object

JsonDict = Dict[str, Any]


@dataclass
class Artifact:
    kind: str
    path: Path
    metadata: JsonDict = field(default_factory=dict)
    sha256: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "metadata": dict(self.metadata),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Artifact":
        return cls(
            kind=str(data["kind"]),
            path=Path(str(data["path"])),
            metadata=dict(data.get("metadata") or {}),
            sha256=data.get("sha256"),
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(kind: str, path: Path, metadata: Optional[JsonDict] = None) -> Artifact:
    resolved_metadata = dict(metadata or {})
    _enrich_array_metadata(path, resolved_metadata)
    return Artifact(
        kind=kind,
        path=path,
        metadata=resolved_metadata,
        sha256=file_sha256(path) if path.is_file() else None,
    )


def _enrich_array_metadata(path: Path, metadata: JsonDict) -> None:
    if not path.is_file() or path.suffix != ".npz":
        return
    try:
        with np.load(str(path), allow_pickle=False) as payload:
            arrays: JsonDict = {}
            for name in payload.files:
                if name == "metadata_json":
                    embedded = decode_strict_json_object(
                        str(payload[name]),
                        label="NPZ metadata_json in %s" % path,
                    )
                    metadata.update(
                        {
                            key: value
                            for key, value in embedded.items()
                            if key not in metadata
                        }
                    )
                    continue
                value = payload[name]
                arrays[name] = {
                    "dtype": str(value.dtype),
                    "shape": [int(item) for item in value.shape],
                }
            if arrays:
                metadata.setdefault("arrays", arrays)
                if len(arrays) == 1:
                    name, info = next(iter(arrays.items()))
                    metadata.setdefault("array", name)
                    metadata.setdefault("dtype", info["dtype"])
                    metadata.setdefault("shape", info["shape"])
    except Exception as exc:
        raise ValueError("Cannot inspect NPZ artifact %s: %s" % (path, exc)) from exc
