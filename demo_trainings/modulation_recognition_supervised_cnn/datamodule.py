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
except ModuleNotFoundError as exc:  # In-tree inspection via runpy.
    if exc.name != "structured_input":
        raise
    from noema_lab.core.structured_input import load_strict_yaml_or_json


RECORD_FINGERPRINT_ALGORITHM = "sha256:noema.amc.capture-record@1"


@dataclass(frozen=True)
class ModulationDataset:
    iq_frames: np.ndarray
    class_ids: np.ndarray
    record_sha256: tuple[str, ...]
    capture_schema_sha256: tuple[str, ...]
    source_dirs: tuple[str, ...]


def load_capture_dataset(
    directories: Sequence[str],
    *,
    feature_tap: str,
    target_tap: str,
    expected_split: str,
) -> ModulationDataset:
    frames = []
    labels = []
    schema_hashes = []
    sources = []
    frame_length = None
    for raw in directories:
        directory = Path(str(raw)).expanduser()
        schema_path = directory / "schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                "Capture schema is missing for %s; run the %s capture job in Noema first"
                % (directory, expected_split)
            )
        schema = load_strict_yaml_or_json(schema_path)
        if str(schema.get("kind") or "") != "noema.capture_dataset":
            raise ValueError("%s is not a Noema capture dataset" % directory)
        if str(schema.get("split") or "") != str(expected_split):
            raise ValueError(
                "Capture %s has split %s, expected %s"
                % (directory, schema.get("split"), expected_split)
            )
        tap_schemas = dict(schema.get("tap_schemas") or {})
        if feature_tap not in tap_schemas or target_tap not in tap_schemas:
            raise ValueError(
                "Capture %s does not contain required taps %s and %s"
                % (directory, feature_tap, target_tap)
            )
        for shard in schema.get("shards") or []:
            shard_path = directory / str(shard.get("path") or "")
            with np.load(str(shard_path), allow_pickle=False) as payload:
                iq = np.asarray(payload[feature_tap], dtype=np.float32)
                truth = np.asarray(payload[target_tap], dtype=np.int64)
            if iq.ndim != 3 or iq.shape[-1] != 2:
                raise ValueError("%s I/Q frames must have shape [frame, sample, 2]" % shard_path)
            if truth.ndim != 1 or truth.shape[0] != iq.shape[0]:
                raise ValueError("%s must contain one class ID per I/Q frame" % shard_path)
            if np.any(truth < 0) or np.any(truth >= 3):
                raise ValueError("%s contains a class ID outside [0, 2]" % shard_path)
            if not np.all(np.isfinite(iq)):
                raise ValueError("%s contains non-finite I/Q values" % shard_path)
            if frame_length is None:
                frame_length = int(iq.shape[1])
            elif int(iq.shape[1]) != frame_length:
                raise ValueError("All capture shards must use one fixed frame length")
            frames.append(iq)
            labels.append(truth)
        schema_hashes.append(hashlib.sha256(schema_path.read_bytes()).hexdigest())
        sources.append(str(directory.resolve()))
    if not frames:
        raise ValueError("No modulation-recognition capture shards were found")
    iq_frames = np.ascontiguousarray(
        np.concatenate(frames, axis=0), dtype=np.float32
    )
    class_ids = np.ascontiguousarray(
        np.concatenate(labels, axis=0), dtype=np.int64
    )
    return ModulationDataset(
        iq_frames=iq_frames,
        class_ids=class_ids,
        record_sha256=tuple(
            _record_sha256(iq_frames[index], int(class_ids[index]))
            for index in range(int(class_ids.size))
        ),
        capture_schema_sha256=tuple(schema_hashes),
        source_dirs=tuple(sources),
    )


def split_fingerprint_report(
    splits: Mapping[str, ModulationDataset | Sequence[str]],
    *,
    include_record_sha256: bool = False,
) -> dict[str, Any]:
    """Prove that normalized capture records are byte-disjoint across splits."""

    normalized: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_records in splits.items():
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("capture split fingerprint names must be non-empty")
        values = (
            raw_records.record_sha256
            if isinstance(raw_records, ModulationDataset)
            else tuple(raw_records)
        )
        if not values:
            raise ValueError(
                "capture split %s has no record byte fingerprints" % name
            )
        fingerprints = tuple(str(value or "").strip().lower() for value in values)
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in fingerprints
        ):
            raise ValueError(
                "capture split %s contains an invalid record SHA-256" % name
            )
        normalized[name] = fingerprints

    split_rows: dict[str, dict[str, Any]] = {}
    for name, fingerprints in normalized.items():
        unique = sorted(set(fingerprints))
        row: dict[str, Any] = {
            "record_count": len(fingerprints),
            "unique_record_count": len(unique),
            "fingerprint_set_sha256": _fingerprint_set_sha256(unique),
        }
        if include_record_sha256:
            row["record_sha256"] = list(fingerprints)
        split_rows[name] = row

    pairwise = []
    for left, right in combinations(normalized, 2):
        overlap = sorted(set(normalized[left]).intersection(normalized[right]))
        pairwise.append(
            {
                "left_split": left,
                "right_split": right,
                "overlap_count": len(overlap),
            }
        )
        if overlap:
            raise ValueError(
                "capture record byte fingerprints overlap between %s and %s "
                "(%d shared record%s)"
                % (
                    left,
                    right,
                    len(overlap),
                    "" if len(overlap) == 1 else "s",
                )
            )
    return {
        "algorithm": RECORD_FINGERPRINT_ALGORITHM,
        "disjoint": True,
        "splits": split_rows,
        "pairwise_overlap": pairwise,
    }


def _record_sha256(iq_frame: np.ndarray, class_id: int) -> str:
    canonical_iq = np.ascontiguousarray(iq_frame, dtype=np.dtype("<f4"))
    digest = hashlib.sha256(b"noema.amc.capture-record@1\0")
    digest.update(
        np.asarray(canonical_iq.shape, dtype=np.dtype("<i8")).tobytes(order="C")
    )
    digest.update(canonical_iq.tobytes(order="C"))
    digest.update(int(class_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _fingerprint_set_sha256(fingerprints: Sequence[str]) -> str:
    digest = hashlib.sha256(b"noema.amc.capture-record-set@1\0")
    for fingerprint in fingerprints:
        digest.update(str(fingerprint).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_loader(
    dataset: ModulationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(dataset.iq_frames), torch.from_numpy(dataset.class_ids)),
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        generator=generator,
    )
