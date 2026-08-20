from __future__ import annotations

import errno
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.common_conditions import (
    CommonConditionError,
    materialize_common_condition_evidence,
)
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import decode_strict_yaml_or_json


JsonDict = Dict[str, Any]

RUN_EVIDENCE_DIRNAME = "run_evidence"
RUN_EVIDENCE_KIND = "noema.benchmark_run_evidence_snapshot"
RUN_EVIDENCE_MANIFEST_KIND = "noema.benchmark_run_evidence_manifest"
RUN_EVIDENCE_SCHEMA_VERSION = 2
LEGACY_RUN_EVIDENCE_SCHEMA_VERSION = 1
SUPPORTED_RUN_EVIDENCE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_RUN_EVIDENCE_SCHEMA_VERSION, RUN_EVIDENCE_SCHEMA_VERSION}
)
ARTIFACT_PROJECTION_KIND = "noema.benchmark_run_artifact_projection"
ARTIFACT_PROJECTION_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_PROJECTION_SCHEMA_VERSION = 1
SUPPORTED_ARTIFACT_PROJECTION_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_ARTIFACT_PROJECTION_SCHEMA_VERSION,
        ARTIFACT_PROJECTION_SCHEMA_VERSION,
    }
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_JSON_FILES = {
    "manifest": "manifest.json",
    "recipe": "recipe.json",
    "authored_recipe": "recipe.authored.json",
    "summary": "summary.json",
    "execution_plan": "execution-plan.json",
}
_COMPRESSED_REQUIRED_JSON_FILES = {
    role: filename + ".gz" for role, filename in _REQUIRED_JSON_FILES.items()
}
_REQUIRED_FILES = tuple(_REQUIRED_JSON_FILES)
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_METRIC_REPORT_KIND = "metrics.report"
_LINUX_FICLONE = 0x40049409
_REFLINK_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EOPNOTSUPP,
        errno.EPERM,
        errno.EXDEV,
    }
)


class BenchmarkRunEvidenceError(ValueError):
    """Raised when result-local backing-run evidence is unsafe or inconsistent."""


def snapshot_benchmark_run_evidence(
    result_dir: Path,
    *,
    entry_id: str,
    entry_index: int,
    run_dir: Path,
    run_id: str,
    semantic_recipe_sha256: str,
    metric_producer_steps: Optional[Iterable[str]] = None,
    retained_artifact_paths: Optional[Iterable[str]] = None,
) -> JsonDict:
    """Freeze the publisher-semantic projection of one completed backing run.

    Version 2 stores deterministic gzip JSON plus metric-report artifacts
    emitted by authoritative benchmark metric producers. ``None`` retains all
    metric reports for callers that do not have a benchmark metric declaration.
    Callers may additionally retain an exact, manifest-relative artifact
    allowlist with ``retained_artifact_paths``. The allowlist does not accept
    patterns and an absent requested path is an error. All declared source
    artifacts are still hash-checked before the source run may be pruned.
    """

    if not run_id or Path(run_id).name != run_id:
        raise BenchmarkRunEvidenceError("source run_id must be a directory name")
    semantic_recipe_sha256 = str(semantic_recipe_sha256 or "").lower()
    if not _DIGEST_RE.fullmatch(semantic_recipe_sha256):
        raise BenchmarkRunEvidenceError(
            "semantic_recipe_sha256 must be a lowercase SHA-256"
        )
    slug = _slugify(entry_id or run_id)
    relative_root = Path(RUN_EVIDENCE_DIRNAME) / ("%03d-%s" % (entry_index, slug))
    destination_root = result_dir / relative_root
    if destination_root.exists():
        raise BenchmarkRunEvidenceError(
            "benchmark run-evidence snapshot already exists: %s" % relative_root
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=".run-evidence-", dir=str(result_dir))
    )
    staging_root = staging_parent / relative_root.name
    staging_root.mkdir()
    try:
        files: List[JsonDict] = []
        for role, source_filename in _REQUIRED_JSON_FILES.items():
            source = run_dir / source_filename
            destination_filename = _COMPRESSED_REQUIRED_JSON_FILES[role]
            destination = staging_root / destination_filename
            files.append(
                _stable_compress_json(
                    source,
                    destination,
                    role=role,
                    result_relative=relative_root / destination.name,
                )
            )
        manifest_payload = _load_json_mapping(
            staging_root / _COMPRESSED_REQUIRED_JSON_FILES["manifest"],
            "manifest",
        )
        artifact_records, artifact_projection = _project_declared_artifacts(
            run_dir,
            staging_root,
            relative_root,
            manifest_payload,
            metric_producer_steps=metric_producer_steps,
            retained_artifact_paths=retained_artifact_paths,
        )
        files.extend(artifact_records)
        files.sort(key=lambda row: str(row["path"]))
        payloads = {
            role: _load_json_mapping(
                staging_root / _COMPRESSED_REQUIRED_JSON_FILES[role],
                role,
            )
            for role in _REQUIRED_JSON_FILES
        }
        _validate_run_semantics(
            payloads["recipe"],
            payloads["summary"],
            payloads["manifest"],
            authored_recipe=payloads["authored_recipe"],
            execution_plan=payloads["execution_plan"],
            run_id=run_id,
        )
        files_sha256 = canonical_json_sha256(files)
        snapshot_manifest: JsonDict = {
            "schema_version": RUN_EVIDENCE_SCHEMA_VERSION,
            "kind": RUN_EVIDENCE_MANIFEST_KIND,
            "source_run_id": run_id,
            "entry_id": entry_id,
            "entry_index": entry_index,
            "semantic_recipe_sha256": semantic_recipe_sha256,
            "files": files,
            "files_sha256": files_sha256,
            "artifact_projection": artifact_projection,
        }
        snapshot_manifest_path = staging_root / "snapshot.json"
        _write_json(snapshot_manifest_path, snapshot_manifest)
        snapshot_manifest_sha256 = file_sha256(snapshot_manifest_path)

        destination_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging_root), str(destination_root))
        return {
            "schema_version": RUN_EVIDENCE_SCHEMA_VERSION,
            "kind": RUN_EVIDENCE_KIND,
            "source_run_id": run_id,
            "semantic_recipe_sha256": semantic_recipe_sha256,
            "root": relative_root.as_posix(),
            "manifest": {
                "path": (relative_root / "snapshot.json").as_posix(),
                "sha256": snapshot_manifest_sha256,
            },
            "files_sha256": files_sha256,
        }
    finally:
        try:
            shutil.rmtree(staging_parent)
        except FileNotFoundError:
            pass


def copy_file_independent(source: Path, destination: Path) -> str:
    """Copy one evidence file without coupling it to the mutable source path.

    Linux filesystems that implement ``FICLONE`` receive an independent
    copy-on-write inode. Other filesystems receive a regular byte copy. Hard
    links are deliberately excluded: deleting a source hard link is harmless,
    but mutating its inode would also mutate the supposedly frozen evidence.

    The destination must not exist. The returned value is ``"reflink"`` or
    ``"copy"`` and is intended for diagnostics and focused storage tests; it
    does not enter the evidence manifest because both forms have identical
    verification semantics.
    """

    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_file():
        raise BenchmarkRunEvidenceError(
            "evidence copy source is missing or unsafe: %s" % source
        )
    if destination.exists() or destination.is_symlink():
        raise BenchmarkRunEvidenceError(
            "evidence copy destination already exists: %s" % destination
        )
    if not destination.parent.is_dir():
        raise BenchmarkRunEvidenceError(
            "evidence copy destination parent is missing: %s" % destination.parent
        )

    method = "reflink" if _try_linux_reflink(source, destination) else "copy"
    if method == "copy":
        try:
            shutil.copyfile(source, destination, follow_symlinks=False)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    try:
        if os.path.samestat(source.stat(), destination.stat()):
            raise BenchmarkRunEvidenceError(
                "evidence copy must not share a mutable inode with its source"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return method


def _try_linux_reflink(source: Path, destination: Path) -> bool:
    """Attempt an atomic Linux CoW clone, returning false when unsupported."""

    if not sys.platform.startswith("linux"):
        return False
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Linux always supplies fcntl.
        return False

    unsupported = False
    try:
        with source.open("rb") as source_file:
            with destination.open("xb") as destination_file:
                try:
                    fcntl.ioctl(
                        destination_file.fileno(),
                        _LINUX_FICLONE,
                        source_file.fileno(),
                    )
                except OSError as exc:
                    if exc.errno not in _REFLINK_UNSUPPORTED_ERRNOS:
                        raise
                    unsupported = True
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if unsupported:
        destination.unlink(missing_ok=True)
        return False

    if destination.stat().st_size != source.stat().st_size:
        destination.unlink(missing_ok=True)
        raise BenchmarkRunEvidenceError(
            "copy-on-write evidence clone has an unexpected size"
        )
    return True


def validate_benchmark_run_evidence_snapshots(
    result_dir: Path,
    result: Mapping[str, Any],
    *,
    include_payloads: bool = True,
) -> JsonDict:
    """Validate completed snapshots, retaining payloads by default for compatibility."""

    return {
        "schema_version": RUN_EVIDENCE_SCHEMA_VERSION,
        "kind": "noema.validated_benchmark_run_evidence",
        "entries": list(
            iter_benchmark_run_evidence_snapshots(
                result_dir,
                result,
                include_payloads=include_payloads,
            )
        ),
    }


def audit_benchmark_run_evidence_snapshots(
    result_dir: Path,
    result: Mapping[str, Any],
) -> JsonDict:
    """Stream-validate all snapshots and retain only compact audit metadata.

    The compatibility validator above intentionally returns decompressed recipe,
    summary, and manifest payloads because several publication surfaces consume
    them.  Executors and resume checks do not need those payloads after a row has
    verified.  This audit therefore validates one row at a time and retains only
    its authenticated descriptor, snapshot inventory, and verification report.
    """

    entries = list(
        iter_benchmark_run_evidence_snapshots(
            result_dir,
            result,
            include_payloads=False,
        )
    )
    return {
        "schema_version": RUN_EVIDENCE_SCHEMA_VERSION,
        "kind": "noema.compact_validated_benchmark_run_evidence",
        "entry_count": len(entries),
        "entries": entries,
    }


def iter_benchmark_run_evidence_snapshots(
    result_dir: Path,
    result: Mapping[str, Any],
    *,
    include_payloads: bool = True,
) -> Iterable[JsonDict]:
    """Yield validated snapshots without retaining prior decompressed payloads."""

    if not isinstance(include_payloads, bool):
        raise ValueError("include_payloads must be a boolean")

    raw_entries = result.get("recipes")
    if not isinstance(raw_entries, list):
        raise BenchmarkRunEvidenceError("benchmark result recipes must be an array")
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise BenchmarkRunEvidenceError(
                "benchmark recipe entry %d must be an object" % index
            )
        status = str(raw_entry.get("status") or "").lower()
        if status not in {"completed", "rejected_resource_budget"}:
            continue
        yield validate_benchmark_run_evidence_snapshot(
            result_dir,
            raw_entry,
            entry_index=index,
            include_payloads=include_payloads,
        )


def validate_benchmark_run_evidence_snapshot(
    result_dir: Path,
    entry: Mapping[str, Any],
    *,
    entry_index: int,
    include_payloads: bool = True,
) -> JsonDict:
    """Authenticate and load one result-local backing-run evidence projection."""

    if not isinstance(include_payloads, bool):
        raise ValueError("include_payloads must be a boolean")
    if not isinstance(entry_index, int) or isinstance(entry_index, bool) or entry_index < 0:
        raise BenchmarkRunEvidenceError("benchmark recipe entry index is invalid")
    status = str(entry.get("status") or "").lower()
    if status not in {"completed", "rejected_resource_budget"}:
        raise BenchmarkRunEvidenceError(
            "benchmark recipe entry %d is not an executed result" % entry_index
        )
    run_id = str(entry.get("run_id") or "")
    if not run_id or Path(run_id).name != run_id:
        raise BenchmarkRunEvidenceError(
            "executed benchmark recipe %d requires a valid run_id" % entry_index
        )
    descriptor = _validate_descriptor(entry.get("run_evidence_snapshot"), run_id)
    root = _confined_snapshot_directory(
        result_dir,
        descriptor["root"],
        "run-evidence snapshot root",
    )
    manifest_path = _confined_snapshot_file(
        result_dir,
        root,
        descriptor["manifest"]["path"],
        "run-evidence snapshot manifest",
    )
    if file_sha256(manifest_path) != descriptor["manifest"]["sha256"]:
        raise BenchmarkRunEvidenceError(
            "run-evidence snapshot manifest hash mismatch for %s" % run_id
        )
    snapshot_manifest = _load_json_mapping(
        manifest_path, "run-evidence snapshot manifest"
    )
    _validate_snapshot_manifest(
        snapshot_manifest,
        descriptor=descriptor,
        entry=entry,
        entry_index=entry_index,
        run_id=run_id,
    )

    files = snapshot_manifest["files"]
    payloads: Dict[str, JsonDict] = {}
    for record in files:
        candidate = _confined_snapshot_file(
            result_dir,
            root,
            record["path"],
            "run-evidence %s" % record["role"],
        )
        if candidate.stat().st_size != record["size_bytes"]:
            raise BenchmarkRunEvidenceError(
                "run-evidence snapshot size mismatch: %s" % record["path"]
            )
        if file_sha256(candidate) != record["sha256"]:
            raise BenchmarkRunEvidenceError(
                "run-evidence snapshot hash mismatch: %s" % record["path"]
            )
        if record["role"] in _REQUIRED_FILES:
            payloads[record["role"]] = _load_json_mapping(
                candidate, "run-evidence %s" % record["role"]
            )
    _validate_artifact_inventory(
        payloads["manifest"],
        files,
        schema_version=int(descriptor["schema_version"]),
        snapshot_root=str(descriptor["root"]),
        artifact_projection=snapshot_manifest.get("artifact_projection"),
        metric_provenance=entry.get("metric_provenance"),
    )
    _validate_run_semantics(
        payloads["recipe"],
        payloads["summary"],
        payloads["manifest"],
        authored_recipe=payloads["authored_recipe"],
        execution_plan=payloads["execution_plan"],
        run_id=run_id,
    )
    _validate_result_projection(entry, payloads)
    validated: JsonDict = {
        "entry_index": entry_index,
        "entry_id": str(entry.get("id") or ""),
        "run_id": run_id,
        "descriptor": descriptor,
        "snapshot_manifest": snapshot_manifest,
        "verification": _snapshot_verification_report(run_id, descriptor),
    }
    if include_payloads:
        validated.update(
            {
                "recipe": payloads["recipe"],
                "summary": payloads["summary"],
                "manifest": payloads["manifest"],
            }
        )
    return validated


def _stable_copy_json(
    source: Path,
    destination: Path,
    *,
    role: str,
    result_relative: Path,
) -> JsonDict:
    if source.is_symlink() or not source.is_file():
        raise BenchmarkRunEvidenceError("backing run %s.json is missing or unsafe" % role)
    before = _file_identity(source)
    if before[2] > _MAX_FILE_BYTES:
        raise BenchmarkRunEvidenceError(
            "backing run %s.json exceeds the snapshot limit" % role
        )
    copy_file_independent(source, destination)
    copied_sha = file_sha256(destination)
    source_sha = file_sha256(source)
    after = _file_identity(source)
    if before != after or copied_sha != source_sha:
        destination.unlink(missing_ok=True)
        raise BenchmarkRunEvidenceError(
            "backing run %s.json changed while it was being snapshotted" % role
        )
    return {
        "role": role,
        "path": result_relative.as_posix(),
        "sha256": copied_sha,
        "size_bytes": destination.stat().st_size,
    }


def _stable_compress_json(
    source: Path,
    destination: Path,
    *,
    role: str,
    result_relative: Path,
) -> JsonDict:
    """Freeze one bounded JSON document as a deterministic single gzip member."""

    if source.is_symlink() or not source.is_file():
        raise BenchmarkRunEvidenceError("backing run %s.json is missing or unsafe" % role)
    before = _file_identity(source)
    if before[2] > _MAX_FILE_BYTES:
        raise BenchmarkRunEvidenceError(
            "backing run %s.json exceeds the snapshot limit" % role
        )
    raw = source.read_bytes()
    source_sha = file_sha256(source)
    after = _file_identity(source)
    if (
        before != after
        or len(raw) != before[2]
        or hashlib.sha256(raw).hexdigest() != source_sha
    ):
        raise BenchmarkRunEvidenceError(
            "backing run %s.json changed while it was being snapshotted" % role
        )
    try:
        compressed_bytes = bytearray(gzip.compress(raw, compresslevel=6, mtime=0))
        # RFC 1952 leaves the OS byte informational. Normalize it so Python/zlib
        # platform differences cannot change evidence bytes.
        if len(compressed_bytes) < 10:
            raise BenchmarkRunEvidenceError(
                "could not create a complete gzip member for %s" % role
            )
        compressed_bytes[9] = 255
        compressed = bytes(compressed_bytes)
        if len(compressed) > _MAX_FILE_BYTES:
            raise BenchmarkRunEvidenceError(
                "compressed backing run %s.json exceeds the snapshot limit" % role
            )
        destination.write_bytes(compressed)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "role": role,
        "path": result_relative.as_posix(),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "size_bytes": len(compressed),
    }


def _project_declared_artifacts(
    run_dir: Path,
    staging_root: Path,
    relative_root: Path,
    manifest: Mapping[str, Any],
    *,
    metric_producer_steps: Optional[Iterable[str]],
    retained_artifact_paths: Optional[Iterable[str]] = None,
) -> Tuple[List[JsonDict], JsonDict]:
    """Authenticate all artifacts and retain the exact declared projection."""

    records: List[JsonDict] = []
    source_inventory: List[JsonDict] = []
    retained_inventory: List[JsonDict] = []
    seen: set[str] = set()
    retain_all_metric_reports = metric_producer_steps is None
    normalized_steps = sorted(
        {
            str(value).strip()
            for value in (metric_producer_steps or ())
            if str(value).strip()
        }
    )
    producer_steps = set(normalized_steps)
    normalized_explicit_paths = _normalize_retained_artifact_paths(
        retained_artifact_paths
    )
    explicit_paths = set(normalized_explicit_paths)
    raw_artifacts = manifest.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raise BenchmarkRunEvidenceError("backing run manifest artifacts must be an array")
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping):
            raise BenchmarkRunEvidenceError(
                "backing run manifest artifact %d must be an object" % index
            )
        declared = str(raw.get("relative_path") or "")
        _validate_relative_path(declared, "backing run artifact relative_path")
        if not declared.startswith("artifacts/") or declared in seen:
            raise BenchmarkRunEvidenceError(
                "backing run artifact path is duplicated or outside artifacts/: %s"
                % declared
            )
        seen.add(declared)
        expected_sha = str(raw.get("sha256") or "").lower()
        if not _DIGEST_RE.fullmatch(expected_sha):
            raise BenchmarkRunEvidenceError(
                "backing run artifact has no valid SHA-256: %s" % declared
            )
        source = _reject_symlinks_and_resolve(
            run_dir,
            Path(declared),
            "backing run artifact",
        )
        if not _is_relative_to(source, run_dir.resolve()) or not source.is_file():
            raise BenchmarkRunEvidenceError(
                "backing run artifact is missing or escapes the run: %s" % declared
            )
        before = _file_identity(source)
        if before[2] > _MAX_ARTIFACT_BYTES:
            raise BenchmarkRunEvidenceError(
                "backing run artifact exceeds the snapshot limit: %s" % declared
            )
        actual_sha = file_sha256(source)
        after = _file_identity(source)
        if before != after:
            raise BenchmarkRunEvidenceError(
                "backing run artifact changed while it was being snapshotted: %s"
                % declared
            )
        if actual_sha != expected_sha:
            raise BenchmarkRunEvidenceError(
                "backing run artifact hash disagrees with manifest: %s" % declared
            )
        item = {
            "path": declared,
            "sha256": expected_sha,
            "size_bytes": before[2],
            "step_id": str(raw.get("step_id") or ""),
            "output_name": str(raw.get("output_name") or ""),
            "kind": str(raw.get("kind") or ""),
        }
        source_inventory.append(item)
        should_retain = item["kind"] == _METRIC_REPORT_KIND and (
            retain_all_metric_reports or item["step_id"] in producer_steps
        )
        should_retain = should_retain or declared in explicit_paths
        if not should_retain:
            continue
        destination = staging_root / declared
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = _stable_copy_binary(
            source,
            destination,
            result_relative=relative_root / declared,
        )
        if record["sha256"] != expected_sha:
            raise BenchmarkRunEvidenceError(
                "backing run artifact hash disagrees with manifest: %s" % declared
            )
        records.append(record)
        retained_inventory.append(item)

    missing_explicit_paths = explicit_paths.difference(seen)
    if missing_explicit_paths:
        raise BenchmarkRunEvidenceError(
            "explicit retained artifact paths are absent from the backing run "
            "manifest: %s" % ", ".join(sorted(missing_explicit_paths))
        )

    source_inventory.sort(key=lambda row: str(row["path"]))
    retained_inventory.sort(key=lambda row: str(row["path"]))
    source_bytes = sum(int(row["size_bytes"]) for row in source_inventory)
    retained_bytes = sum(int(row["size_bytes"]) for row in retained_inventory)
    source_identities = [_artifact_inventory_identity(row) for row in source_inventory]
    retained_identities = [_artifact_inventory_identity(row) for row in retained_inventory]
    projection: JsonDict = {
        "schema_version": ARTIFACT_PROJECTION_SCHEMA_VERSION,
        "kind": ARTIFACT_PROJECTION_KIND,
        "selection": (
            "all_metric_reports"
            if retain_all_metric_reports
            else "declared_metric_producer_reports"
        ),
        "metric_producer_steps": normalized_steps,
        "explicit_artifact_paths": normalized_explicit_paths,
        "source_artifact_count": len(source_inventory),
        "source_artifact_bytes": source_bytes,
        "source_inventory_sha256": canonical_json_sha256(source_identities),
        "retained_artifact_count": len(retained_inventory),
        "retained_artifact_bytes": retained_bytes,
        "retained_inventory_sha256": canonical_json_sha256(retained_identities),
        "omitted_artifact_count": len(source_inventory) - len(retained_inventory),
        "omitted_artifact_bytes": source_bytes - retained_bytes,
    }
    return records, projection


def _stable_copy_binary(
    source: Path,
    destination: Path,
    *,
    result_relative: Path,
) -> JsonDict:
    if source.is_symlink() or not source.is_file():
        raise BenchmarkRunEvidenceError("backing run artifact is missing or unsafe")
    before = _file_identity(source)
    if before[2] > _MAX_ARTIFACT_BYTES:
        raise BenchmarkRunEvidenceError("backing run artifact exceeds the snapshot limit")
    copy_file_independent(source, destination)
    copied_sha = file_sha256(destination)
    source_sha = file_sha256(source)
    after = _file_identity(source)
    if before != after or copied_sha != source_sha:
        destination.unlink(missing_ok=True)
        raise BenchmarkRunEvidenceError(
            "backing run artifact changed while it was being snapshotted"
        )
    return {
        "role": "artifact",
        "path": result_relative.as_posix(),
        "sha256": copied_sha,
        "size_bytes": destination.stat().st_size,
    }


def _validate_descriptor(value: Any, run_id: str) -> JsonDict:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "source_run_id",
        "semantic_recipe_sha256",
        "root",
        "manifest",
        "files_sha256",
    }:
        raise BenchmarkRunEvidenceError(
            "completed backing run %s has no valid result-local snapshot descriptor"
            % run_id
        )
    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_RUN_EVIDENCE_SCHEMA_VERSIONS:
        raise BenchmarkRunEvidenceError("unsupported run-evidence snapshot schema")
    if value.get("kind") != RUN_EVIDENCE_KIND:
        raise BenchmarkRunEvidenceError("run-evidence snapshot kind is invalid")
    if value.get("source_run_id") != run_id:
        raise BenchmarkRunEvidenceError("run-evidence source_run_id mismatch")
    semantic_recipe_sha256 = str(value.get("semantic_recipe_sha256") or "").lower()
    if not _DIGEST_RE.fullmatch(semantic_recipe_sha256):
        raise BenchmarkRunEvidenceError(
            "run-evidence semantic_recipe_sha256 is invalid"
        )
    root = str(value.get("root") or "")
    if not root.startswith(RUN_EVIDENCE_DIRNAME + "/"):
        raise BenchmarkRunEvidenceError("run-evidence snapshot root is invalid")
    manifest = value.get("manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {"path", "sha256"}:
        raise BenchmarkRunEvidenceError("run-evidence manifest descriptor is invalid")
    manifest_path = str(manifest.get("path") or "")
    manifest_sha = str(manifest.get("sha256") or "").lower()
    files_sha = str(value.get("files_sha256") or "").lower()
    if manifest_path != root + "/snapshot.json" or not _DIGEST_RE.fullmatch(manifest_sha):
        raise BenchmarkRunEvidenceError("run-evidence manifest reference is invalid")
    if not _DIGEST_RE.fullmatch(files_sha):
        raise BenchmarkRunEvidenceError("run-evidence files_sha256 is invalid")
    return {
        "schema_version": schema_version,
        "kind": RUN_EVIDENCE_KIND,
        "source_run_id": run_id,
        "semantic_recipe_sha256": semantic_recipe_sha256,
        "root": root,
        "manifest": {"path": manifest_path, "sha256": manifest_sha},
        "files_sha256": files_sha,
    }


def _validate_snapshot_manifest(
    payload: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any],
    entry: Mapping[str, Any],
    entry_index: int,
    run_id: str,
) -> None:
    schema_version = payload.get("schema_version")
    expected_fields = {
        "schema_version",
        "kind",
        "source_run_id",
        "entry_id",
        "entry_index",
        "semantic_recipe_sha256",
        "files",
        "files_sha256",
    }
    if schema_version == RUN_EVIDENCE_SCHEMA_VERSION:
        expected_fields.add("artifact_projection")
    if set(payload) != expected_fields:
        raise BenchmarkRunEvidenceError("run-evidence snapshot manifest fields are invalid")
    if schema_version not in SUPPORTED_RUN_EVIDENCE_SCHEMA_VERSIONS:
        raise BenchmarkRunEvidenceError("unsupported run-evidence manifest schema")
    if schema_version != descriptor.get("schema_version"):
        raise BenchmarkRunEvidenceError(
            "run-evidence manifest/descriptor schema mismatch"
        )
    if payload.get("kind") != RUN_EVIDENCE_MANIFEST_KIND:
        raise BenchmarkRunEvidenceError("run-evidence manifest kind is invalid")
    if payload.get("source_run_id") != run_id:
        raise BenchmarkRunEvidenceError("run-evidence manifest source_run_id mismatch")
    if payload.get("entry_id") != str(entry.get("id") or ""):
        raise BenchmarkRunEvidenceError("run-evidence manifest entry_id mismatch")
    if payload.get("entry_index") != entry_index:
        raise BenchmarkRunEvidenceError("run-evidence manifest entry_index mismatch")
    semantic_recipe_sha256 = str(
        payload.get("semantic_recipe_sha256") or ""
    ).lower()
    if not _DIGEST_RE.fullmatch(semantic_recipe_sha256):
        raise BenchmarkRunEvidenceError(
            "run-evidence manifest semantic_recipe_sha256 is invalid"
        )
    if semantic_recipe_sha256 != descriptor.get("semantic_recipe_sha256"):
        raise BenchmarkRunEvidenceError(
            "run-evidence semantic recipe SHA descriptor mismatch"
        )
    if semantic_recipe_sha256 != entry.get("semantic_recipe_sha256"):
        raise BenchmarkRunEvidenceError(
            "benchmark semantic recipe SHA does not match its snapshot"
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or len(raw_files) < len(_REQUIRED_FILES):
        raise BenchmarkRunEvidenceError("run-evidence manifest inventory is incomplete")
    normalized = [_validate_file_record(row) for row in raw_files]
    if [row["path"] for row in normalized] != sorted(row["path"] for row in normalized):
        raise BenchmarkRunEvidenceError("run-evidence inventory must be sorted by path")
    if not set(_REQUIRED_FILES).issubset({row["role"] for row in normalized}):
        raise BenchmarkRunEvidenceError("run-evidence inventory roles are incomplete")
    expected_filenames = (
        _COMPRESSED_REQUIRED_JSON_FILES
        if schema_version == RUN_EVIDENCE_SCHEMA_VERSION
        else _REQUIRED_JSON_FILES
    )
    snapshot_root = str(descriptor.get("root") or "")
    for role, filename in expected_filenames.items():
        matches = [row for row in normalized if row["role"] == role]
        expected_path = "%s/%s" % (snapshot_root, filename)
        if len(matches) != 1 or matches[0]["path"] != expected_path:
            raise BenchmarkRunEvidenceError(
                "run-evidence required %s payload path is invalid" % role
            )
    if len({row["path"] for row in normalized}) != len(normalized):
        raise BenchmarkRunEvidenceError("run-evidence inventory paths are duplicated")
    files_sha = canonical_json_sha256(normalized)
    if payload.get("files_sha256") != files_sha or descriptor.get("files_sha256") != files_sha:
        raise BenchmarkRunEvidenceError("run-evidence inventory projection hash mismatch")


def _validate_file_record(value: Any) -> JsonDict:
    if not isinstance(value, Mapping) or set(value) != {
        "role",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise BenchmarkRunEvidenceError("run-evidence inventory record is invalid")
    role = str(value.get("role") or "")
    path = str(value.get("path") or "")
    sha = str(value.get("sha256") or "").lower()
    size = value.get("size_bytes")
    if role not in {*_REQUIRED_FILES, "artifact"}:
        raise BenchmarkRunEvidenceError("run-evidence inventory role is invalid")
    _validate_relative_path(path, "run-evidence inventory path")
    if not _DIGEST_RE.fullmatch(sha):
        raise BenchmarkRunEvidenceError("run-evidence inventory SHA-256 is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BenchmarkRunEvidenceError("run-evidence inventory size is invalid")
    return {"role": role, "path": path, "sha256": sha, "size_bytes": size}


def _validate_artifact_inventory(
    manifest: Mapping[str, Any],
    files: List[Mapping[str, Any]],
    *,
    schema_version: int,
    snapshot_root: str,
    artifact_projection: Any,
    metric_provenance: Any,
) -> None:
    if schema_version == LEGACY_RUN_EVIDENCE_SCHEMA_VERSION:
        _validate_legacy_artifact_inventory(manifest, files)
        return
    projection = _validate_artifact_projection(artifact_projection)
    if projection["selection"] == "declared_metric_producer_reports":
        if not isinstance(metric_provenance, Mapping):
            raise BenchmarkRunEvidenceError(
                "run-evidence declared metric projection has no result provenance"
            )
        expected_producer_steps = sorted(
            {
                str(raw.get("source_step") or "")
                for raw in metric_provenance.values()
                if isinstance(raw, Mapping)
                and raw.get("source_scope") == "step"
                and str(raw.get("source_step") or "")
            }
        )
        if projection["metric_producer_steps"] != expected_producer_steps:
            raise BenchmarkRunEvidenceError(
                "run-evidence metric producer projection disagrees with result provenance"
            )
    source_inventory = _manifest_artifact_inventory(manifest)
    source_identities = [_artifact_inventory_identity(row) for row in source_inventory]
    if (
        projection["source_artifact_count"] != len(source_inventory)
        or projection["source_inventory_sha256"]
        != canonical_json_sha256(source_identities)
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence source artifact projection disagrees with its manifest"
        )

    selection = projection["selection"]
    producer_steps = set(projection["metric_producer_steps"])
    explicit_paths = set(projection["explicit_artifact_paths"])
    source_paths = {str(row["path"]) for row in source_inventory}
    if not explicit_paths.issubset(source_paths):
        raise BenchmarkRunEvidenceError(
            "run-evidence explicit artifact projection names paths absent from "
            "its manifest"
        )
    expected = [
        row
        for row in source_inventory
        if (
            row["kind"] == _METRIC_REPORT_KIND
            and (
                selection == "all_metric_reports"
                or row["step_id"] in producer_steps
            )
        )
        or row["path"] in explicit_paths
    ]
    expected_identities = [_artifact_inventory_identity(row) for row in expected]
    if (
        projection["retained_artifact_count"] != len(expected)
        or projection["retained_inventory_sha256"]
        != canonical_json_sha256(expected_identities)
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence retained artifact projection is not the declared "
            "metric-report and explicit-path subset"
        )

    prefix = snapshot_root.rstrip("/") + "/"
    retained_files: Dict[str, Mapping[str, Any]] = {}
    for record in files:
        if record.get("role") != "artifact":
            continue
        declared = str(record.get("path") or "")
        if not declared.startswith(prefix):
            raise BenchmarkRunEvidenceError(
                "run-evidence artifact inventory has an inconsistent root"
            )
        relative = declared[len(prefix) :]
        if relative in retained_files:
            raise BenchmarkRunEvidenceError(
                "run-evidence artifact inventory path is duplicated"
            )
        retained_files[relative] = record
    expected_by_path = {str(row["path"]): row for row in expected}
    if set(retained_files) != set(expected_by_path):
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact files do not equal the declared retained projection"
        )
    for relative, record in retained_files.items():
        if record.get("sha256") != expected_by_path[relative]["sha256"]:
            raise BenchmarkRunEvidenceError(
                "run-evidence retained artifact hash disagrees with manifest: %s"
                % relative
            )
    retained_bytes = sum(int(row.get("size_bytes") or 0) for row in retained_files.values())
    if projection["retained_artifact_bytes"] != retained_bytes:
        raise BenchmarkRunEvidenceError(
            "run-evidence retained artifact byte count is inconsistent"
        )
    if (
        projection["source_artifact_bytes"] < retained_bytes
        or projection["omitted_artifact_count"]
        != projection["source_artifact_count"] - projection["retained_artifact_count"]
        or projection["omitted_artifact_bytes"]
        != projection["source_artifact_bytes"] - projection["retained_artifact_bytes"]
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence omitted artifact accounting is inconsistent"
        )


def _validate_legacy_artifact_inventory(
    manifest: Mapping[str, Any],
    files: List[Mapping[str, Any]],
) -> None:
    inventory = {
        str(row.get("path") or ""): str(row.get("sha256") or "")
        for row in files
        if row.get("role") == "artifact"
    }
    snapshot_roots = {
        path.split("/artifacts/", 1)[0]
        for path in inventory
        if "/artifacts/" in path
    }
    if len(snapshot_roots) > 1:
        raise BenchmarkRunEvidenceError("artifact inventory has inconsistent roots")
    prefix = next(iter(snapshot_roots), "")
    for index, raw in enumerate(manifest.get("artifacts") or []):
        if not isinstance(raw, Mapping):
            raise BenchmarkRunEvidenceError(
                "run manifest artifact %d is invalid" % index
            )
        relative = str(raw.get("relative_path") or "")
        expected_path = "%s/%s" % (prefix, relative) if prefix else relative
        if inventory.get(expected_path) != str(raw.get("sha256") or ""):
            raise BenchmarkRunEvidenceError(
                "run-evidence artifact inventory does not match manifest: %s"
                % relative
            )


def _manifest_artifact_inventory(manifest: Mapping[str, Any]) -> List[JsonDict]:
    raw_artifacts = manifest.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raise BenchmarkRunEvidenceError("run manifest artifacts must be an array")
    inventory: List[JsonDict] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping):
            raise BenchmarkRunEvidenceError(
                "run manifest artifact %d is invalid" % index
            )
        relative = str(raw.get("relative_path") or "")
        _validate_relative_path(relative, "run manifest artifact relative_path")
        if not relative.startswith("artifacts/") or relative in seen:
            raise BenchmarkRunEvidenceError(
                "run manifest artifact path is duplicated or outside artifacts/: %s"
                % relative
            )
        seen.add(relative)
        sha = str(raw.get("sha256") or "").lower()
        if not _DIGEST_RE.fullmatch(sha):
            raise BenchmarkRunEvidenceError(
                "run manifest artifact has no valid SHA-256: %s" % relative
            )
        inventory.append(
            {
                "path": relative,
                "sha256": sha,
                "step_id": str(raw.get("step_id") or ""),
                "output_name": str(raw.get("output_name") or ""),
                "kind": str(raw.get("kind") or ""),
            }
        )
    inventory.sort(key=lambda row: str(row["path"]))
    return inventory


def _artifact_inventory_identity(value: Mapping[str, Any]) -> JsonDict:
    return {
        "path": str(value.get("path") or ""),
        "sha256": str(value.get("sha256") or ""),
        "step_id": str(value.get("step_id") or ""),
        "output_name": str(value.get("output_name") or ""),
        "kind": str(value.get("kind") or ""),
    }


def _validate_artifact_projection(value: Any) -> JsonDict:
    common_fields = {
        "schema_version",
        "kind",
        "selection",
        "metric_producer_steps",
        "source_artifact_count",
        "source_artifact_bytes",
        "source_inventory_sha256",
        "retained_artifact_count",
        "retained_artifact_bytes",
        "retained_inventory_sha256",
        "omitted_artifact_count",
        "omitted_artifact_bytes",
    }
    if not isinstance(value, Mapping):
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact projection fields are invalid"
        )
    schema_version = value.get("schema_version")
    expected_fields = set(common_fields)
    if schema_version == ARTIFACT_PROJECTION_SCHEMA_VERSION:
        expected_fields.add("explicit_artifact_paths")
    if (
        schema_version not in SUPPORTED_ARTIFACT_PROJECTION_SCHEMA_VERSIONS
        or set(value) != expected_fields
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact projection fields are invalid"
        )
    if (
        value.get("kind") != ARTIFACT_PROJECTION_KIND
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact projection schema/kind is invalid"
        )
    selection = str(value.get("selection") or "")
    if selection not in {
        "all_metric_reports",
        "declared_metric_producer_reports",
    }:
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact projection selection is invalid"
        )
    raw_steps = value.get("metric_producer_steps")
    if (
        not isinstance(raw_steps, list)
        or any(not isinstance(step, str) or not step.strip() for step in raw_steps)
        or raw_steps != sorted(set(raw_steps))
        or (selection == "all_metric_reports" and raw_steps)
    ):
        raise BenchmarkRunEvidenceError(
            "run-evidence artifact projection producer steps are invalid"
        )
    if schema_version == ARTIFACT_PROJECTION_SCHEMA_VERSION:
        explicit_paths = _normalize_retained_artifact_paths(
            value.get("explicit_artifact_paths")
        )
        if value.get("explicit_artifact_paths") != explicit_paths:
            raise BenchmarkRunEvidenceError(
                "run-evidence explicit artifact paths must be sorted and unique"
            )
    else:
        explicit_paths = []
    numeric_fields = (
        "source_artifact_count",
        "source_artifact_bytes",
        "retained_artifact_count",
        "retained_artifact_bytes",
        "omitted_artifact_count",
        "omitted_artifact_bytes",
    )
    for field in numeric_fields:
        raw = value.get(field)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise BenchmarkRunEvidenceError(
                "run-evidence artifact projection %s is invalid" % field
            )
    for field in ("source_inventory_sha256", "retained_inventory_sha256"):
        if not _DIGEST_RE.fullmatch(str(value.get(field) or "").lower()):
            raise BenchmarkRunEvidenceError(
                "run-evidence artifact projection %s is invalid" % field
            )
    normalized = dict(value)
    normalized["explicit_artifact_paths"] = explicit_paths
    return normalized


def _normalize_retained_artifact_paths(
    value: Optional[Iterable[str]],
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise BenchmarkRunEvidenceError(
            "retained_artifact_paths must be an iterable of exact paths"
        )
    paths: List[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise BenchmarkRunEvidenceError(
                "retained artifact path %d must be a string" % index
            )
        _validate_relative_path(raw, "retained artifact path %d" % index)
        if not raw.startswith("artifacts/"):
            raise BenchmarkRunEvidenceError(
                "retained artifact path %d must be below artifacts/" % index
            )
        paths.append(raw)
    if len(paths) != len(set(paths)):
        raise BenchmarkRunEvidenceError(
            "retained_artifact_paths cannot contain duplicates"
        )
    return sorted(paths)


def _validate_run_semantics(
    recipe: Mapping[str, Any],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    authored_recipe: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    run_id: str,
) -> None:
    if str(summary.get("status") or "").lower() != "completed":
        raise BenchmarkRunEvidenceError("snapshotted run summary is not completed")
    if str(manifest.get("status") or "").lower() != "completed":
        raise BenchmarkRunEvidenceError("snapshotted run manifest is not completed")
    if summary.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise BenchmarkRunEvidenceError("snapshotted run_id is inconsistent")
    recipe_name = str(recipe.get("name") or "")
    if not recipe_name or summary.get("recipe_name") != recipe_name or manifest.get("recipe_name") != recipe_name:
        raise BenchmarkRunEvidenceError("snapshotted recipe identity is inconsistent")
    manifest_recipe = manifest.get("recipe")
    manifest_recipe = manifest_recipe if isinstance(manifest_recipe, Mapping) else {}
    recipe_sha = str(manifest_recipe.get("sha256") or "")
    if recipe_sha != canonical_json_sha256(recipe):
        raise BenchmarkRunEvidenceError("snapshotted recipe SHA-256 is inconsistent")
    if summary.get("recipe_sha256") and summary.get("recipe_sha256") != recipe_sha:
        raise BenchmarkRunEvidenceError("snapshotted summary recipe SHA-256 is inconsistent")
    authored_sha = canonical_json_sha256(authored_recipe)
    if str(manifest_recipe.get("authored_sha256") or "") != authored_sha:
        raise BenchmarkRunEvidenceError(
            "snapshotted authored recipe SHA-256 is inconsistent"
        )
    plan = dict(execution_plan)
    declared_plan_sha = str(plan.pop("sha256", ""))
    if not _DIGEST_RE.fullmatch(declared_plan_sha) or canonical_json_sha256(plan) != declared_plan_sha:
        raise BenchmarkRunEvidenceError("snapshotted execution plan SHA-256 is inconsistent")
    manifest_plan = manifest.get("execution_plan")
    manifest_plan = manifest_plan if isinstance(manifest_plan, Mapping) else {}
    if manifest_plan.get("sha256") != declared_plan_sha:
        raise BenchmarkRunEvidenceError(
            "snapshotted manifest does not bind the execution plan"
        )
    plan_recipe = execution_plan.get("recipe")
    plan_recipe = plan_recipe if isinstance(plan_recipe, Mapping) else {}
    if plan_recipe.get("sha256") != recipe_sha:
        raise BenchmarkRunEvidenceError(
            "snapshotted execution plan does not bind the effective recipe"
        )


def _validate_result_projection(
    entry: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    summary = payloads["summary"]
    manifest = payloads["manifest"]
    manifest_recipe = manifest.get("recipe")
    manifest_recipe = manifest_recipe if isinstance(manifest_recipe, Mapping) else {}
    if entry.get("recipe_sha256") != manifest_recipe.get("sha256"):
        raise BenchmarkRunEvidenceError("benchmark recipe SHA does not match its snapshot")
    if str(entry.get("recipe_name") or "") != str(summary.get("recipe_name") or ""):
        raise BenchmarkRunEvidenceError("benchmark recipe name does not match its snapshot")
    entry_metrics = dict(entry.get("metrics") or {})
    metric_provenance = dict(entry.get("metric_provenance") or {})
    summary_metrics = _collect_summary_metrics(summary)
    for key, value in summary_metrics.items():
        provenance = metric_provenance.get(key)
        provenance = provenance if isinstance(provenance, Mapping) else {}
        # The run summary's unqualified projection predates authoritative
        # benchmark metric binding and may be first-wins when multiple steps
        # emit the same name. Step-qualified values are always authoritative;
        # run-scoped values remain authoritative only when their provenance
        # says so. The benchmark verifier separately validates every selected
        # step producer and its definition digest.
        must_match = key.startswith("steps.") or provenance.get("source_scope") == "run"
        if must_match and (key not in entry_metrics or entry_metrics[key] != value):
            raise BenchmarkRunEvidenceError(
                "benchmark metrics do not match the run snapshot"
            )
    derived_keys = {
        "benchmark.resource_budget.admitted",
        "benchmark.resource_budget.observed",
        "benchmark.resource_budget.maximum",
        "benchmark.resource_budget.tolerance",
        "benchmark.resource_budget.excess",
    }
    unexpected = set(entry_metrics).difference(summary_metrics).difference(derived_keys)
    if unexpected:
        raise BenchmarkRunEvidenceError(
            "benchmark metrics contain unbound derived values: %s"
            % ", ".join(sorted(unexpected))
        )
    admission = entry.get("resource_admission")
    admission_metrics = set(derived_keys).intersection(entry_metrics)
    if admission is None:
        if admission_metrics or str(entry.get("status") or "") == "rejected_resource_budget":
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission is missing from an admitted/rejected projection"
            )
    else:
        if not isinstance(admission, Mapping):
            raise BenchmarkRunEvidenceError("benchmark resource admission is invalid")
        metric = str(admission.get("metric") or "")
        if not metric.startswith("steps.") or metric not in summary_metrics:
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission metric is not step-scoped in the run snapshot"
            )
        try:
            numeric_admission = {
                key: float(admission.get(key))
                for key in ("observed", "maximum", "tolerance", "excess")
            }
        except (TypeError, ValueError) as exc:
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission numeric values are invalid"
            ) from exc
        if not isinstance(admission.get("admitted"), bool) or any(
            not math.isfinite(value) or value < 0.0
            for value in numeric_admission.values()
        ):
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission values are invalid"
            )
        observed = numeric_admission["observed"]
        try:
            measured = float(summary_metrics[metric])
        except (TypeError, ValueError) as exc:
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission run metric is not numeric"
            ) from exc
        if measured != observed:
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission observed value does not match its run metric"
            )
        expected_derived = {
            "benchmark.resource_budget.admitted": 1 if admission.get("admitted") else 0,
            "benchmark.resource_budget.observed": observed,
            "benchmark.resource_budget.maximum": numeric_admission["maximum"],
            "benchmark.resource_budget.tolerance": numeric_admission["tolerance"],
            "benchmark.resource_budget.excess": numeric_admission["excess"],
        }
        if any(entry_metrics.get(key) != value for key, value in expected_derived.items()):
            raise BenchmarkRunEvidenceError(
                "benchmark resource admission derived metrics do not match its run projection"
            )

    effective_recipe = payloads["recipe"]
    recipe_metadata = effective_recipe.get("metadata")
    recipe_metadata = recipe_metadata if isinstance(recipe_metadata, Mapping) else {}
    pairing_id = next(
        (
            recipe_metadata.get(key)
            for key in ("pairing_id", "paired_seed", "benchmark_paired_seed")
            if recipe_metadata.get(key) not in (None, "")
        ),
        None,
    )
    expected_pairing_id = None if pairing_id in (None, "") else str(pairing_id)
    if entry.get("pairing_id") != expected_pairing_id:
        raise BenchmarkRunEvidenceError(
            "benchmark pairing_id does not match the effective recipe snapshot"
        )
    expected_pairing_seed = (
        pairing_id
        if isinstance(pairing_id, (int, float)) and not isinstance(pairing_id, bool)
        else None
    )
    if entry.get("pairing_seed") != expected_pairing_seed:
        raise BenchmarkRunEvidenceError(
            "benchmark pairing_seed does not match the effective recipe snapshot"
        )
    expected_cell = recipe_metadata.get("aggregation_cell_id")
    expected_cell = None if expected_cell in (None, "") else str(expected_cell)
    if entry.get("aggregation_cell_id") != expected_cell:
        raise BenchmarkRunEvidenceError(
            "benchmark aggregation_cell_id does not match the effective recipe snapshot"
        )
    expected_statistical_unit = recipe_metadata.get("statistical_unit")
    if expected_statistical_unit not in (None, "") and entry.get(
        "statistical_unit"
    ) != expected_statistical_unit:
        raise BenchmarkRunEvidenceError(
            "benchmark statistical_unit does not match the effective recipe snapshot"
        )
    snapshot_research = manifest_recipe.get("research")
    if entry.get("research") != snapshot_research:
        raise BenchmarkRunEvidenceError("benchmark research metadata does not match its snapshot")
    try:
        expected_conditions = materialize_common_condition_evidence(
            effective_recipe,
            payloads["summary"],
        )
    except CommonConditionError as exc:
        raise BenchmarkRunEvidenceError(
            "benchmark common-condition evidence cannot be reconstructed: %s" % exc
        ) from exc
    if expected_conditions:
        if entry.get("common_condition_evidence") != expected_conditions:
            raise BenchmarkRunEvidenceError(
                "benchmark common-condition evidence does not match the immutable run snapshot"
            )
    elif entry.get("common_condition_evidence") is not None:
        raise BenchmarkRunEvidenceError(
            "benchmark projects common-condition evidence absent from its run snapshot"
        )


def _collect_summary_metrics(summary: Mapping[str, Any]) -> JsonDict:
    metrics = dict(summary.get("metrics") or {})
    for step in summary.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id") or "")
        step_metrics = step.get("metrics")
        step_metrics = step_metrics if isinstance(step_metrics, Mapping) else {}
        for key, value in step_metrics.items():
            metrics.setdefault(str(key), value)
            if step_id:
                metrics["steps.%s.%s" % (step_id, key)] = value
    return metrics


def _snapshot_verification_report(
    run_id: str,
    descriptor: Mapping[str, Any],
) -> JsonDict:
    return {
        "status": "valid",
        "target_type": "benchmark_run_snapshot",
        "target_id": run_id,
        "errors": [],
        "warnings": [],
        "checks": [
            {
                "id": "benchmark.run_evidence_snapshot",
                "status": "pass",
                "message": (
                    "result-local recipes, plan, manifest, summary, and retained "
                    "artifact hashes verify"
                ),
            }
        ],
        "metadata": {
            "source_run_id": run_id,
            "files_sha256": descriptor.get("files_sha256"),
        },
    }


def _confined_snapshot_directory(result_dir: Path, declared: str, label: str) -> Path:
    _validate_relative_path(declared, label)
    candidate = _reject_symlinks_and_resolve(result_dir, Path(declared), label)
    evidence_root = (result_dir / RUN_EVIDENCE_DIRNAME).resolve()
    if not _is_relative_to(candidate, evidence_root) or not candidate.is_dir():
        raise BenchmarkRunEvidenceError("%s is missing or escapes the result" % label)
    return candidate


def _confined_snapshot_file(
    result_dir: Path,
    snapshot_root: Path,
    declared: str,
    label: str,
) -> Path:
    _validate_relative_path(declared, label)
    candidate = _reject_symlinks_and_resolve(result_dir, Path(declared), label)
    if not _is_relative_to(candidate, snapshot_root.resolve()) or not candidate.is_file():
        raise BenchmarkRunEvidenceError("%s is missing or escapes its snapshot" % label)
    return candidate


def _reject_symlinks_and_resolve(base: Path, relative: Path, label: str) -> Path:
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkRunEvidenceError("%s cannot traverse a symlink" % label)
    return cursor.resolve()


def _validate_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise BenchmarkRunEvidenceError("%s is not a safe relative path" % label)


def _load_json_mapping(path: Path, label: str) -> JsonDict:
    try:
        if path.name.endswith(".json.gz"):
            compressed = path.read_bytes()
            if len(compressed) > _MAX_FILE_BYTES:
                raise BenchmarkRunEvidenceError(
                    "%s compressed bytes exceed the snapshot limit" % label
                )
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            raw = decoder.decompress(compressed, _MAX_FILE_BYTES + 1)
            if (
                len(raw) > _MAX_FILE_BYTES
                or decoder.unconsumed_tail
                or not decoder.eof
                or decoder.unused_data
            ):
                raise BenchmarkRunEvidenceError(
                    "%s is not one bounded complete gzip member" % label
                )
            raw += decoder.flush()
            if len(raw) > _MAX_FILE_BYTES:
                raise BenchmarkRunEvidenceError(
                    "%s decompressed bytes exceed the snapshot limit" % label
                )
            text = raw.decode("utf-8")
        else:
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise BenchmarkRunEvidenceError(
                    "%s bytes exceed the snapshot limit" % label
                )
            text = path.read_text(encoding="utf-8")
        payload = decode_strict_yaml_or_json(
            text,
            input_format="json",
        )
    except Exception as exc:
        raise BenchmarkRunEvidenceError("could not parse %s: %s" % (label, exc)) from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkRunEvidenceError("%s must contain an object" % label)
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_identity(path: Path) -> Tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return slug or "run"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False
