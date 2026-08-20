from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from noema_lab.core.structured_input import decode_strict_yaml_or_json


class CaptureSplitIntegrityError(ValueError):
    """Raised when captured dataset partitions are not independent."""


@dataclass(frozen=True)
class CaptureSplitFingerprint:
    split: str
    directory: Path
    tap_ids: Tuple[str, ...]
    record_count: int
    unique_record_count: int
    record_sha256: Tuple[str, ...]


def assert_disjoint_capture_splits(
    split_directories: Mapping[str, Path | str],
    *,
    expected_taps: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, object]:
    """Reject byte-identical complete records shared by dataset splits.

    A record fingerprint includes every selected tap for that record.  A label
    that legitimately occurs more than once therefore does not trigger this
    guard unless the corresponding captured signals are identical as well.
    """

    fingerprints = []
    for declared_split, raw_directory in split_directories.items():
        split = str(declared_split or "").strip()
        if not split:
            raise CaptureSplitIntegrityError("Captured dataset split name is empty")
        taps = tuple(
            sorted(str(item) for item in (expected_taps or {}).get(split, ()) if str(item))
        )
        fingerprints.append(
            fingerprint_capture_split(
                Path(raw_directory),
                expected_split=split,
                expected_taps=taps,
            )
        )

    if fingerprints:
        reference_taps = fingerprints[0].tap_ids
        mismatched = [
            item.split
            for item in fingerprints[1:]
            if item.tap_ids != reference_taps
        ]
        if mismatched:
            raise CaptureSplitIntegrityError(
                "Captured dataset splits do not contain the same selected taps: %s"
                % ", ".join(mismatched)
            )

    overlaps = []
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            shared = Counter(left.record_sha256) & Counter(right.record_sha256)
            shared_records = sum(shared.values())
            if shared_records:
                overlaps.append(
                    {
                        "left": left.split,
                        "right": right.split,
                        "records": shared_records,
                        "taps": list(left.tap_ids),
                    }
                )
    if overlaps:
        detail = "; ".join(
            "%s and %s share %d byte-identical complete record%s"
            % (
                item["left"],
                item["right"],
                item["records"],
                "" if item["records"] == 1 else "s",
            )
            for item in overlaps
        )
        raise CaptureSplitIntegrityError(
            "Captured dataset leakage detected: %s. The comparison fingerprints "
            "all selected taps together, so repeated labels alone are allowed. "
            "Do not train, validate, or report metrics from these captures. "
            "Overwrite all affected splits after correcting the random-seed or "
            "partitioning configuration."
            % detail
        )

    return {
        "status": "disjoint",
        "splits": {
            item.split: {
                "records": item.record_count,
                "unique_records": item.unique_record_count,
                "taps": list(item.tap_ids),
            }
            for item in fingerprints
        },
    }


def fingerprint_capture_split(
    directory: Path | str,
    *,
    expected_split: str = "",
    expected_taps: Sequence[str] = (),
) -> CaptureSplitFingerprint:
    capture_dir = Path(directory).expanduser().resolve()
    schema_path = capture_dir / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError("Noema capture schema is missing: %s" % schema_path)
    try:
        schema = decode_strict_yaml_or_json(
            schema_path.read_text(encoding="utf-8"),
            input_format="json",
        )
    except (OSError, ValueError) as exc:
        raise CaptureSplitIntegrityError(
            "Capture schema is not readable JSON %s: %s" % (schema_path, exc)
        ) from exc
    if not isinstance(schema, dict) or schema.get("kind") != "noema.capture_dataset":
        raise CaptureSplitIntegrityError(
            "Unsupported Noema capture schema: %s" % schema_path
        )
    split = str(schema.get("split") or "").strip()
    if expected_split and split != str(expected_split):
        raise CaptureSplitIntegrityError(
            "Capture %s declares split %r, expected %r"
            % (capture_dir, split, expected_split)
        )
    tap_ids = tuple(sorted(str(item) for item in expected_taps if str(item)))
    if not tap_ids:
        tap_ids = tuple(
            sorted(str(item) for item in (schema.get("tap_schemas") or {}) if str(item))
        )
    if not tap_ids:
        raise CaptureSplitIntegrityError(
            "Capture %s does not declare any tensor taps to fingerprint" % capture_dir
        )
    if len(set(tap_ids)) != len(tap_ids):
        raise CaptureSplitIntegrityError(
            "Capture %s declares duplicate tensor taps" % capture_dir
        )

    record_hashes = []
    tap_schemas = schema.get("tap_schemas") or {}
    if not isinstance(tap_schemas, Mapping):
        raise CaptureSplitIntegrityError(
            "Capture %s tap_schemas must be an object" % capture_dir
        )
    shards = list(schema.get("shards") or [])
    if not shards:
        raise CaptureSplitIntegrityError("Capture has no shards: %s" % capture_dir)
    for raw_shard in shards:
        relative = raw_shard.get("path") if isinstance(raw_shard, dict) else raw_shard
        shard_path = (capture_dir / str(relative or "")).resolve()
        try:
            shard_path.relative_to(capture_dir)
        except ValueError as exc:
            raise CaptureSplitIntegrityError(
                "Capture shard escapes its dataset directory: %s" % shard_path
            ) from exc
        if not shard_path.is_file():
            raise FileNotFoundError("Capture shard is missing: %s" % shard_path)
        try:
            with np.load(str(shard_path), allow_pickle=False) as payload:
                missing = [tap for tap in tap_ids if tap not in payload.files]
                if missing:
                    raise CaptureSplitIntegrityError(
                        "Capture shard %s omits selected tap(s): %s"
                        % (shard_path, ", ".join(missing))
                    )
                arrays = [(tap, np.asarray(payload[tap])) for tap in tap_ids]
                for tap, array in arrays:
                    declared = tap_schemas.get(tap)
                    if not isinstance(declared, Mapping):
                        raise CaptureSplitIntegrityError(
                            "Capture %s does not declare a schema for selected tap %s"
                            % (capture_dir, tap)
                        )
                    declared_dtype = declared.get("dtype")
                    if declared_dtype not in (None, ""):
                        try:
                            expected_dtype = np.dtype(str(declared_dtype))
                        except TypeError as exc:
                            raise CaptureSplitIntegrityError(
                                "Capture %s declares an invalid dtype for tap %s: %r"
                                % (capture_dir, tap, declared_dtype)
                            ) from exc
                        if array.dtype != expected_dtype:
                            raise CaptureSplitIntegrityError(
                                "Capture shard %s tap %s has dtype %s, declared %s"
                                % (
                                    shard_path,
                                    tap,
                                    array.dtype,
                                    expected_dtype,
                                )
                            )
                    declared_shape = declared.get("record_shape")
                    if declared_shape is not None:
                        try:
                            expected_shape = tuple(int(item) for item in declared_shape)
                        except (TypeError, ValueError) as exc:
                            raise CaptureSplitIntegrityError(
                                "Capture %s declares an invalid record_shape for tap %s"
                                % (capture_dir, tap)
                            ) from exc
                        actual_shape = tuple(int(item) for item in array.shape[1:])
                        if actual_shape != expected_shape:
                            raise CaptureSplitIntegrityError(
                                "Capture shard %s tap %s has record shape %s, declared %s"
                                % (
                                    shard_path,
                                    tap,
                                    list(actual_shape),
                                    list(expected_shape),
                                )
                            )
                counts = {int(array.shape[0]) for _, array in arrays if array.ndim >= 1}
                if len(counts) != 1 or any(array.ndim < 1 for _, array in arrays):
                    raise CaptureSplitIntegrityError(
                        "Capture shard %s does not align selected taps on the record axis"
                        % shard_path
                    )
                record_count = counts.pop()
                for index in range(record_count):
                    record_hashes.append(_joint_record_sha256(arrays, index))
        except CaptureSplitIntegrityError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise CaptureSplitIntegrityError(
                "Capture shard cannot be validated safely: %s" % shard_path
            ) from exc

    if not record_hashes:
        raise CaptureSplitIntegrityError("Capture contains no records: %s" % capture_dir)
    declared_count = int(schema.get("captured_samples") or 0)
    if declared_count and declared_count != len(record_hashes):
        raise CaptureSplitIntegrityError(
            "Capture %s declares %d records but its shards contain %d"
            % (capture_dir, declared_count, len(record_hashes))
        )
    unique_hashes = frozenset(record_hashes)
    return CaptureSplitFingerprint(
        split=split or str(expected_split),
        directory=capture_dir,
        tap_ids=tap_ids,
        record_count=len(record_hashes),
        unique_record_count=len(unique_hashes),
        record_sha256=tuple(record_hashes),
    )


def _joint_record_sha256(
    arrays: Sequence[tuple[str, np.ndarray]],
    index: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"noema.capture.record@1\0")
    for tap_id, array in arrays:
        record = np.ascontiguousarray(array[index])
        digest.update(tap_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(record.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(record.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()
