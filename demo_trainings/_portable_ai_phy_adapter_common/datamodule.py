from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from structured_input import load_strict_yaml_or_json
except ModuleNotFoundError as exc:
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


@dataclass(frozen=True)
class CapturedTensorDataset:
    features: np.ndarray
    targets: np.ndarray
    record_sha256: tuple[str, ...]
    capture_schema_sha256: tuple[str, ...]
    source_dirs: tuple[str, ...]


def load_capture_dataset(
    directories: Sequence[str],
    *,
    feature_tap: str,
    target_tap: str,
    expected_split: str,
) -> CapturedTensorDataset:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    schema_hashes: list[str] = []
    sources: list[str] = []
    feature_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...] | None = None
    for raw in directories:
        directory = Path(str(raw)).expanduser()
        schema_path = directory / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                "Capture schema is missing for %s; run the %s capture job first"
                % (directory, expected_split)
            )
        schema = load_strict_yaml_or_json(schema_path)
        if not isinstance(schema, Mapping) or str(schema.get("kind") or "") != "noema.capture_dataset":
            raise ValueError("%s is not a Noema capture dataset" % directory)
        if str(schema.get("split") or "") != expected_split:
            raise ValueError(
                "Capture %s has split %s, expected %s"
                % (directory, schema.get("split"), expected_split)
            )
        tap_schemas = dict(schema.get("tap_schemas") or {})
        required = [feature_tap] + ([target_tap] if target_tap else [])
        missing = [tap for tap in required if tap not in tap_schemas]
        if missing:
            raise ValueError(
                "Capture %s does not contain required tap(s): %s"
                % (directory, ", ".join(missing))
            )
        for shard in list(schema.get("shards") or []):
            shard_path = directory / str(shard.get("path") or "")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                features = _portable_array(payload[feature_tap])
                targets = (
                    _portable_array(payload[target_tap])
                    if target_tap
                    else np.empty((int(features.shape[0]), 0), dtype=np.float32)
                )
            if features.ndim < 2:
                raise ValueError("%s features must include record and feature axes" % shard_path)
            if targets.shape[0] != features.shape[0]:
                raise ValueError("%s target count does not match feature count" % shard_path)
            if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
                raise ValueError("%s contains non-finite tensors" % shard_path)
            current_feature_shape = tuple(int(value) for value in features.shape[1:])
            current_target_shape = tuple(int(value) for value in targets.shape[1:])
            if feature_shape is None:
                feature_shape = current_feature_shape
                target_shape = current_target_shape
            elif current_feature_shape != feature_shape or current_target_shape != target_shape:
                raise ValueError("All capture shards must use fixed feature and target shapes")
            feature_parts.append(features)
            target_parts.append(targets)
        schema_hashes.append(hashlib.sha256(schema_path.read_bytes()).hexdigest())
        sources.append(str(directory.resolve()))
    if not feature_parts:
        raise ValueError("No capture shards were found for %s" % expected_split)
    features = np.ascontiguousarray(np.concatenate(feature_parts, axis=0), dtype=np.float32)
    targets = np.ascontiguousarray(np.concatenate(target_parts, axis=0), dtype=np.float32)
    return CapturedTensorDataset(
        features=features,
        targets=targets,
        record_sha256=tuple(
            _record_sha256(features[index], targets[index])
            for index in range(int(features.shape[0]))
        ),
        capture_schema_sha256=tuple(schema_hashes),
        source_dirs=tuple(sources),
    )


def build_loader(
    dataset: CapturedTensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(dataset.features),
            torch.from_numpy(dataset.targets),
        ),
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        generator=generator,
    )


def split_fingerprint_report(
    splits: Mapping[str, CapturedTensorDataset],
    *,
    include_record_sha256: bool = False,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    normalized = {name: tuple(dataset.record_sha256) for name, dataset in splits.items()}
    for name, fingerprints in normalized.items():
        if not fingerprints:
            raise ValueError("capture split %s has no records" % name)
        unique = sorted(set(fingerprints))
        row: dict[str, Any] = {
            "record_count": len(fingerprints),
            "unique_record_count": len(unique),
            "fingerprint_set_sha256": _fingerprint_set_sha256(unique),
        }
        if include_record_sha256:
            row["record_sha256"] = list(fingerprints)
        rows[name] = row
    pairwise = []
    for left, right in combinations(normalized, 2):
        overlap = set(normalized[left]).intersection(normalized[right])
        pairwise.append(
            {"left_split": left, "right_split": right, "overlap_count": len(overlap)}
        )
        if overlap:
            raise ValueError(
                "capture records overlap between %s and %s (%d shared)"
                % (left, right, len(overlap))
            )
    return {
        "algorithm": "sha256:noema.portable-ai-phy-capture-record@1",
        "disjoint": True,
        "splits": rows,
        "pairwise_overlap": pairwise,
    }


def _portable_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.stack([array.real, array.imag], axis=-1)
    return np.ascontiguousarray(array, dtype=np.float32)


def _record_sha256(features: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256(b"noema.portable-ai-phy-capture-record@1\0")
    for array in (features, targets):
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
        digest.update(np.asarray(canonical.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _fingerprint_set_sha256(fingerprints: Sequence[str]) -> str:
    digest = hashlib.sha256(b"noema.portable-ai-phy-capture-record-set@1\0")
    for fingerprint in fingerprints:
        digest.update(str(fingerprint).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
