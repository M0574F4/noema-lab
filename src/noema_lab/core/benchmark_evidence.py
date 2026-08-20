from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)


JsonDict = Dict[str, Any]

SNAPSHOT_DIRNAME = "training_evidence"
SNAPSHOT_MANIFEST_PATH = "training_evidence/manifest.json"
SNAPSHOT_PROJECTION_PATH = "training_evidence/projection.json"
SNAPSHOT_KIND = "noema.benchmark_training_evidence_snapshot"
SNAPSHOT_MANIFEST_KIND = "noema.benchmark_training_evidence_snapshot_manifest"
SNAPSHOT_PROJECTION_KIND = "noema.benchmark_training_evidence_projection"
SNAPSHOT_SCHEMA_VERSION = 1
TRAINING_EVIDENCE_FIELDS = (
    "trained_artifact_manifest",
    "training_history",
    "evaluation_metrics",
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_COMPONENT_BYTES = 256 * 1024 * 1024
_MAX_SUPPORT_FILE_BYTES = 32 * 1024 * 1024


class BenchmarkEvidenceError(ValueError):
    """Raised when benchmark training evidence is unsafe or inconsistent."""


def snapshot_benchmark_training_evidence(
    result_dir: Path,
    result: MutableMapping[str, Any],
    benchmark_source: Mapping[str, Any],
    project_root: Path,
) -> Optional[JsonDict]:
    """Copy declared training evidence into a content-verified result snapshot.

    ``result`` is rewritten only after the complete snapshot has been installed.
    Source paths remain in ``benchmark.json`` as authored provenance, while the
    completed result points exclusively at files below its own result directory.
    """

    raw_rows = _result_training_evidence(result)
    if not raw_rows:
        return None
    if (result_dir / SNAPSHOT_DIRNAME).exists():
        raise BenchmarkEvidenceError(
            "benchmark training-evidence snapshot already exists: %s"
            % (result_dir / SNAPSHOT_DIRNAME)
        )

    source_base, source_root = _source_context(benchmark_source, project_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=".training-evidence-", dir=str(result_dir))
    )
    staging_root = staging_parent / SNAPSHOT_DIRNAME
    staging_root.mkdir()
    installed = False
    try:
        entries: List[JsonDict] = []
        file_records: Dict[str, JsonDict] = {}
        used_slugs: set[str] = set()
        used_series: set[str] = set()
        rewritten_rows: List[JsonDict] = []

        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                raise BenchmarkEvidenceError(
                    "benchmark metadata.demo.training_evidence[%d] must be an object"
                    % index
                )
            series = str(raw_row.get("series") or "").strip()
            if not series:
                raise BenchmarkEvidenceError(
                    "benchmark metadata.demo.training_evidence[%d] requires series"
                    % index
                )
            unknown_fields = sorted(
                set(raw_row) - {"series", *TRAINING_EVIDENCE_FIELDS}
            )
            if unknown_fields:
                raise BenchmarkEvidenceError(
                    "benchmark training-evidence entry %d has unsupported fields: %s"
                    % (index, ", ".join(unknown_fields))
                )
            if series in used_series:
                raise BenchmarkEvidenceError(
                    "benchmark training-evidence series is duplicated: %s" % series
                )
            used_series.add(series)
            series_slug = _unique_slug(_slugify(series), used_slugs)
            used_slugs.add(series_slug)
            entry_relative = Path("entries") / ("%03d-%s" % (index, series_slug))
            entry: JsonDict = {
                "index": index,
                "series": series,
                "evidence": {},
                "artifact_references": [],
                "source_declarations": {},
            }
            rewritten: JsonDict = {"series": series}

            for field in TRAINING_EVIDENCE_FIELDS:
                if raw_row.get(field) in (None, ""):
                    continue
                declared_path, declared_sha = _normalize_source_spec(
                    raw_row[field], field
                )
                source = _safe_source_file(
                    source_base,
                    source_root,
                    declared_path,
                    field,
                    max_bytes=_MAX_EVIDENCE_BYTES,
                )
                suffix = source.suffix.lower()
                if field == "trained_artifact_manifest":
                    destination_relative = (
                        entry_relative / "artifact" / (field + suffix)
                    )
                else:
                    destination_relative = (
                        entry_relative / "evidence" / (field + suffix)
                    )
                evidence_record = _copy_snapshot_file(
                    source,
                    staging_root,
                    destination_relative,
                    expected_sha256=declared_sha or None,
                    max_bytes=_MAX_EVIDENCE_BYTES,
                )
                _record_snapshot_file(file_records, evidence_record)
                entry["evidence"][field] = dict(evidence_record)
                entry["source_declarations"][field] = {
                    "path": declared_path,
                    "declared_sha256": declared_sha or None,
                    "snapshotted_sha256": evidence_record["sha256"],
                }
                rewritten[field] = {
                    "path": evidence_record["path"],
                    "sha256": evidence_record["sha256"],
                }

                if field == "trained_artifact_manifest":
                    copied_manifest = result_dir / evidence_record["path"]
                    copied_manifest = staging_parent / copied_manifest.relative_to(
                        result_dir
                    )
                    manifest_payload = _load_mapping(
                        copied_manifest, "trained artifact manifest"
                    )
                    references = _snapshot_artifact_references(
                        manifest_payload,
                        source_manifest=source,
                        source_root=source_root,
                        staging_root=staging_root,
                        destination_manifest_relative=destination_relative,
                        file_records=file_records,
                    )
                    entry["artifact_references"] = references

            if not entry["evidence"]:
                raise BenchmarkEvidenceError(
                    "benchmark training-evidence entry %d declares no evidence files"
                    % index
                )

            entries.append(entry)
            rewritten_rows.append(rewritten)

        projection: JsonDict = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": SNAPSHOT_PROJECTION_KIND,
            "entries": entries,
            "source_declaration_sha256": canonical_json_sha256(
                _source_declaration_projection(entries)
            ),
        }
        projection_path = staging_root / "projection.json"
        _write_json(projection_path, projection)
        projection_record = _snapshot_record(
            projection_path, Path("projection.json")
        )
        _record_snapshot_file(file_records, projection_record)

        files = [file_records[path] for path in sorted(file_records)]
        files_sha256 = canonical_json_sha256(files)
        manifest: JsonDict = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": SNAPSHOT_MANIFEST_KIND,
            "projection": dict(projection_record),
            "files": files,
            "files_sha256": files_sha256,
        }
        manifest_path = staging_root / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = file_sha256(manifest_path)

        os.replace(
            str(staging_root),
            str(result_dir / SNAPSHOT_DIRNAME),
        )
        installed = True
        descriptor: JsonDict = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "kind": SNAPSHOT_KIND,
            "root": SNAPSHOT_DIRNAME,
            "manifest": {
                "path": SNAPSHOT_MANIFEST_PATH,
                "sha256": manifest_sha256,
            },
            "projection": {
                "path": SNAPSHOT_PROJECTION_PATH,
                "sha256": projection_record["sha256"],
            },
            "files_sha256": files_sha256,
        }
        _replace_result_training_evidence(result, rewritten_rows)
        result["training_evidence_snapshot"] = descriptor
        return descriptor
    except Exception:
        if installed:
            shutil.rmtree(result_dir / SNAPSHOT_DIRNAME, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def validate_benchmark_training_evidence_snapshot(
    result_dir: Path,
    result: Mapping[str, Any],
    benchmark_source: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Validate and return the immutable training-evidence projection."""

    result_rows = _result_training_evidence(result)
    descriptor = result.get("training_evidence_snapshot")
    if not result_rows:
        if descriptor not in (None, {}):
            raise BenchmarkEvidenceError(
                "benchmark result has a training-evidence snapshot but declares no training evidence"
            )
        return {"present": False}
    if not isinstance(descriptor, Mapping):
        raise BenchmarkEvidenceError(
            "benchmark result training evidence is not self-contained; snapshot descriptor is missing"
        )
    if int(descriptor.get("schema_version") or 0) != SNAPSHOT_SCHEMA_VERSION:
        raise BenchmarkEvidenceError("unsupported training-evidence snapshot schema version")
    if descriptor.get("kind") != SNAPSHOT_KIND:
        raise BenchmarkEvidenceError("training-evidence snapshot kind is invalid")
    if descriptor.get("root") != SNAPSHOT_DIRNAME:
        raise BenchmarkEvidenceError("training-evidence snapshot root is invalid")

    snapshot_root = (result_dir / SNAPSHOT_DIRNAME).resolve()
    if not snapshot_root.is_dir():
        raise BenchmarkEvidenceError("training-evidence snapshot directory is missing")
    manifest_spec = _required_snapshot_spec(
        descriptor.get("manifest"),
        expected_path=SNAPSHOT_MANIFEST_PATH,
        label="snapshot manifest",
    )
    projection_spec = _required_snapshot_spec(
        descriptor.get("projection"),
        expected_path=SNAPSHOT_PROJECTION_PATH,
        label="snapshot projection",
    )
    manifest_path = _confined_snapshot_file(
        result_dir, snapshot_root, manifest_spec["path"], "snapshot manifest"
    )
    if file_sha256(manifest_path) != manifest_spec["sha256"]:
        raise BenchmarkEvidenceError("training-evidence snapshot manifest hash mismatch")
    manifest = _load_json_mapping(manifest_path, "training-evidence snapshot manifest")
    if int(manifest.get("schema_version") or 0) != SNAPSHOT_SCHEMA_VERSION:
        raise BenchmarkEvidenceError("unsupported training-evidence manifest schema version")
    if manifest.get("kind") != SNAPSHOT_MANIFEST_KIND:
        raise BenchmarkEvidenceError("training-evidence manifest kind is invalid")
    manifest_projection = _validated_file_record(
        manifest.get("projection"), "snapshot manifest projection"
    )
    if {
        "path": manifest_projection["path"],
        "sha256": manifest_projection["sha256"],
    } != projection_spec:
        raise BenchmarkEvidenceError(
            "training-evidence manifest projection does not match result descriptor"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BenchmarkEvidenceError("training-evidence manifest files must be non-empty")
    file_records: Dict[str, JsonDict] = {}
    normalized_files: List[JsonDict] = []
    for index, raw_record in enumerate(raw_files):
        record = _validated_file_record(raw_record, "manifest files[%d]" % index)
        path = record["path"]
        if path in file_records:
            raise BenchmarkEvidenceError(
                "training-evidence manifest contains duplicate path: %s" % path
            )
        candidate = _confined_snapshot_file(
            result_dir, snapshot_root, path, "snapshot file"
        )
        if candidate.stat().st_size != record["size_bytes"]:
            raise BenchmarkEvidenceError(
                "training-evidence snapshot size mismatch: %s" % path
            )
        if file_sha256(candidate) != record["sha256"]:
            raise BenchmarkEvidenceError(
                "training-evidence snapshot hash mismatch: %s" % path
            )
        file_records[path] = record
        normalized_files.append(record)
    if [row["path"] for row in normalized_files] != sorted(file_records):
        raise BenchmarkEvidenceError(
            "training-evidence manifest files must be sorted by path"
        )
    files_sha256 = canonical_json_sha256(normalized_files)
    if manifest.get("files_sha256") != files_sha256:
        raise BenchmarkEvidenceError("training-evidence manifest file projection hash mismatch")
    if descriptor.get("files_sha256") != files_sha256:
        raise BenchmarkEvidenceError("result snapshot file projection hash mismatch")
    if file_records.get(projection_spec["path"]) != manifest_projection:
        raise BenchmarkEvidenceError(
            "training-evidence projection is missing from the snapshot manifest"
        )

    projection_path = _confined_snapshot_file(
        result_dir, snapshot_root, projection_spec["path"], "snapshot projection"
    )
    if file_sha256(projection_path) != projection_spec["sha256"]:
        raise BenchmarkEvidenceError("training-evidence snapshot projection hash mismatch")
    projection = _load_json_mapping(
        projection_path, "training-evidence snapshot projection"
    )
    if int(projection.get("schema_version") or 0) != SNAPSHOT_SCHEMA_VERSION:
        raise BenchmarkEvidenceError("unsupported training-evidence projection schema version")
    if projection.get("kind") != SNAPSHOT_PROJECTION_KIND:
        raise BenchmarkEvidenceError("training-evidence projection kind is invalid")

    expected_rows = _validate_projection_entries(
        projection,
        result_dir=result_dir,
        snapshot_root=snapshot_root,
        file_records=file_records,
        source_rows=(
            _benchmark_source_training_evidence(benchmark_source)
            if benchmark_source is not None
            else None
        ),
    )
    if result_rows != expected_rows:
        raise BenchmarkEvidenceError(
            "result training-evidence references do not match the immutable snapshot projection"
        )
    return {
        "present": True,
        "root": snapshot_root,
        "manifest": manifest,
        "projection": projection,
        "training_evidence": expected_rows,
    }


def benchmark_training_evidence_bindings(
    result_dir: Path,
    result: Mapping[str, Any],
    benchmark_source: Mapping[str, Any],
    project_root: Path,
) -> List[JsonDict]:
    """Return transient source-to-snapshot manifest bindings for execution."""

    validated = validate_benchmark_training_evidence_snapshot(
        result_dir, result, benchmark_source=benchmark_source
    )
    if not validated.get("present"):
        return []
    source_rows = _benchmark_source_training_evidence(benchmark_source)
    snapshot_rows = list(validated.get("training_evidence") or [])
    if len(source_rows) != len(snapshot_rows):
        raise BenchmarkEvidenceError(
            "source and snapshot training-evidence entry counts do not match"
        )
    source_base, source_root = _source_context(benchmark_source, project_root)
    bindings: List[JsonDict] = []
    for index, (source_row, snapshot_row) in enumerate(
        zip(source_rows, snapshot_rows)
    ):
        if not isinstance(source_row, Mapping):
            raise BenchmarkEvidenceError(
                "source training-evidence entry %d must be an object" % index
            )
        if str(source_row.get("series") or "").strip() != str(
            snapshot_row.get("series") or ""
        ):
            raise BenchmarkEvidenceError(
                "source and snapshot training-evidence series do not match"
            )
        if source_row.get("trained_artifact_manifest") in (None, ""):
            continue
        declared, declared_sha = _normalize_source_spec(
            source_row["trained_artifact_manifest"],
            "trained_artifact_manifest",
        )
        source_path = _confined_source_candidate(
            source_base,
            source_root,
            declared,
            "trained_artifact_manifest",
        )
        snapshot_spec = snapshot_row.get("trained_artifact_manifest")
        if not isinstance(snapshot_spec, Mapping):
            raise BenchmarkEvidenceError(
                "snapshot entry %d is missing trained_artifact_manifest" % index
            )
        snapshot_sha = str(snapshot_spec.get("sha256") or "").lower()
        if declared_sha and declared_sha != snapshot_sha:
            raise BenchmarkEvidenceError(
                "source and snapshot trained artifact manifest hashes do not match"
            )
        snapshot_path = _confined_snapshot_file(
            result_dir,
            Path(validated["root"]),
            str(snapshot_spec.get("path") or ""),
            "snapshotted trained artifact manifest",
        )
        bindings.append(
            {
                "series": str(snapshot_row.get("series") or ""),
                "source_path": str(source_path),
                "snapshot_path": str(snapshot_path),
                "sha256": snapshot_sha,
            }
        )
    return bindings


def _snapshot_artifact_references(
    payload: Mapping[str, Any],
    *,
    source_manifest: Path,
    source_root: Path,
    staging_root: Path,
    destination_manifest_relative: Path,
    file_records: Dict[str, JsonDict],
) -> List[JsonDict]:
    references: List[JsonDict] = []
    if int(payload.get("schema_version") or 0) != 2:
        raise BenchmarkEvidenceError(
            "benchmark training evidence requires a schema-v2 trained artifact manifest"
        )
    if payload.get("kind") != "noema.trained_block_artifact":
        raise BenchmarkEvidenceError("trained artifact manifest kind is invalid")
    components = payload.get("components") or []
    if not isinstance(components, list):
        raise BenchmarkEvidenceError(
            "trained artifact manifest components must be an array"
        )
    for index, component in enumerate(components):
        if not isinstance(component, Mapping) or not component.get("path"):
            raise BenchmarkEvidenceError(
                "trained artifact component %d requires path" % index
            )
        component_id = str(component.get("id") or index)
        reference = _copy_artifact_reference(
            source_manifest=source_manifest,
            source_root=source_root,
            declared=str(component.get("path") or ""),
            expected_sha256=str(component.get("sha256") or ""),
            label="trained artifact component %s" % component_id,
            kind="component",
            reference_id=component_id,
            staging_root=staging_root,
            destination_manifest_relative=destination_manifest_relative,
            max_bytes=_MAX_COMPONENT_BYTES,
        )
        _record_snapshot_file(file_records, reference)
        references.append(reference)

    contract = payload.get("contract")
    if isinstance(contract, Mapping) and contract.get("path"):
        contract_id = str(contract.get("id") or "contract")
        reference = _copy_artifact_reference(
            source_manifest=source_manifest,
            source_root=source_root,
            declared=str(contract.get("path") or ""),
            expected_sha256=str(contract.get("file_sha256") or ""),
            label="trained artifact contract",
            kind="contract",
            reference_id=contract_id,
            staging_root=staging_root,
            destination_manifest_relative=destination_manifest_relative,
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        _record_snapshot_file(file_records, reference)
        references.append(reference)
    source = payload.get("source")
    source = source if isinstance(source, Mapping) else {}
    data_contract = source.get("data_contract")
    if isinstance(data_contract, Mapping) and data_contract.get("path"):
        reference = _copy_artifact_reference(
            source_manifest=source_manifest,
            source_root=source_root,
            declared=str(data_contract.get("path") or ""),
            expected_sha256=str(data_contract.get("file_sha256") or ""),
            label="trained artifact data contract",
            kind="data_contract",
            reference_id="training_data",
            staging_root=staging_root,
            destination_manifest_relative=destination_manifest_relative,
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        _record_snapshot_file(file_records, reference)
        references.append(reference)
    support_files = payload.get("support_files") or []
    if not isinstance(support_files, list):
        raise BenchmarkEvidenceError("trained artifact support_files must be an array")
    for index, support_file in enumerate(support_files):
        if not isinstance(support_file, Mapping) or not support_file.get("path"):
            raise BenchmarkEvidenceError(
                "trained artifact support file %d requires path" % index
            )
        support_id = str(support_file.get("role") or index)
        reference = _copy_artifact_reference(
            source_manifest=source_manifest,
            source_root=source_root,
            declared=str(support_file.get("path") or ""),
            expected_sha256=str(support_file.get("sha256") or ""),
            label="trained artifact support file %s" % support_id,
            kind="support_file",
            reference_id=support_id,
            staging_root=staging_root,
            destination_manifest_relative=destination_manifest_relative,
            max_bytes=_MAX_SUPPORT_FILE_BYTES,
        )
        _record_snapshot_file(file_records, reference)
        references.append(reference)
    references.sort(key=lambda row: (str(row["kind"]), str(row["id"]), str(row["path"])))
    return references


def _copy_artifact_reference(
    *,
    source_manifest: Path,
    source_root: Path,
    declared: str,
    expected_sha256: str,
    label: str,
    kind: str,
    reference_id: str,
    staging_root: Path,
    destination_manifest_relative: Path,
    max_bytes: int,
) -> JsonDict:
    relative = Path(declared)
    if (
        not declared
        or relative.is_absolute()
        or _WINDOWS_ABSOLUTE_PATH_RE.match(declared)
        or ".." in relative.parts
    ):
        raise BenchmarkEvidenceError(
            "%s path must stay relative to its manifest" % label
        )
    expected = expected_sha256.strip().lower()
    if not _DIGEST_RE.fullmatch(expected):
        raise BenchmarkEvidenceError("%s requires a valid SHA-256 digest" % label)
    source_unresolved = source_manifest.parent / relative
    _assert_no_symlink_components(
        source_manifest.parent, relative, label
    )
    source = source_unresolved.resolve()
    if not _is_relative_to(source, source_root.resolve()) or not source.is_file():
        raise BenchmarkEvidenceError(
            "%s file is missing or outside the evidence root" % label
        )
    destination_relative = destination_manifest_relative.parent / relative
    if destination_relative == destination_manifest_relative:
        raise BenchmarkEvidenceError("%s cannot reference its own manifest" % label)
    record = _copy_snapshot_file(
        source,
        staging_root,
        destination_relative,
        expected_sha256=expected,
        max_bytes=max_bytes,
    )
    return {"kind": kind, "id": reference_id, **record}


def _validate_projection_entries(
    projection: Mapping[str, Any],
    *,
    result_dir: Path,
    snapshot_root: Path,
    file_records: Mapping[str, JsonDict],
    source_rows: Optional[List[Any]],
) -> List[JsonDict]:
    raw_entries = projection.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BenchmarkEvidenceError(
            "training-evidence projection entries must be non-empty"
        )
    result_rows: List[JsonDict] = []
    source_projection: List[JsonDict] = []
    if source_rows is not None and len(source_rows) != len(raw_entries):
        raise BenchmarkEvidenceError(
            "benchmark source and snapshot evidence entry counts do not match"
        )
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise BenchmarkEvidenceError(
                "training-evidence projection entry %d must be an object" % index
            )
        if raw_entry.get("index") != index:
            raise BenchmarkEvidenceError(
                "training-evidence projection entry indices are not canonical"
            )
        if set(raw_entry) != {
            "index",
            "series",
            "evidence",
            "artifact_references",
            "source_declarations",
        }:
            raise BenchmarkEvidenceError(
                "training-evidence projection entry %d has invalid fields" % index
            )
        series = str(raw_entry.get("series") or "").strip()
        if not series:
            raise BenchmarkEvidenceError(
                "training-evidence projection entry %d requires series" % index
            )
        evidence = raw_entry.get("evidence")
        if not isinstance(evidence, Mapping):
            raise BenchmarkEvidenceError(
                "training-evidence projection entry %d evidence must be an object"
                % index
            )
        source_declarations = raw_entry.get("source_declarations")
        if not isinstance(source_declarations, Mapping):
            raise BenchmarkEvidenceError(
                "training-evidence source_declarations must be an object"
            )
        if set(source_declarations) != set(evidence):
            raise BenchmarkEvidenceError(
                "training-evidence source declarations do not match evidence fields"
            )
        unknown = sorted(set(evidence) - set(TRAINING_EVIDENCE_FIELDS))
        if unknown:
            raise BenchmarkEvidenceError(
                "training-evidence projection has unsupported fields: %s"
                % ", ".join(unknown)
            )
        result_row: JsonDict = {"series": series}
        manifest_record: Optional[JsonDict] = None
        for field in TRAINING_EVIDENCE_FIELDS:
            if field not in evidence:
                continue
            record = _validated_file_record(
                evidence[field], "projection %s" % field
            )
            if file_records.get(record["path"]) != record:
                raise BenchmarkEvidenceError(
                    "projection %s is not covered by the snapshot manifest" % field
                )
            result_row[field] = {
                "path": record["path"],
                "sha256": record["sha256"],
            }
            if field == "trained_artifact_manifest":
                manifest_record = record
            declaration = _validated_source_declaration(
                source_declarations[field], "projection %s source" % field
            )
            if declaration["snapshotted_sha256"] != record["sha256"]:
                raise BenchmarkEvidenceError(
                    "projection %s source digest does not match snapshot" % field
                )

        declaration_row = {
            "series": series,
            "files": {
                field: _validated_source_declaration(
                    source_declarations[field], "projection %s source" % field
                )
                for field in TRAINING_EVIDENCE_FIELDS
                if field in source_declarations
            },
        }
        source_projection.append(declaration_row)
        if source_rows is not None:
            _compare_source_declaration(
                source_rows[index], declaration_row, index=index
            )

        raw_references = raw_entry.get("artifact_references")
        if not isinstance(raw_references, list):
            raise BenchmarkEvidenceError(
                "training-evidence artifact_references must be an array"
            )
        references = [
            _validated_reference_record(item, index=reference_index)
            for reference_index, item in enumerate(raw_references)
        ]
        for reference in references:
            file_record = {
                "path": reference["path"],
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
            }
            if file_records.get(reference["path"]) != file_record:
                raise BenchmarkEvidenceError(
                    "artifact reference is not covered by the snapshot manifest: %s"
                    % reference["path"]
                )
        if manifest_record is None and references:
            raise BenchmarkEvidenceError(
                "artifact references require a trained artifact manifest"
            )
        if manifest_record is not None:
            manifest_path = _confined_snapshot_file(
                result_dir,
                snapshot_root,
                manifest_record["path"],
                "trained artifact manifest",
            )
            artifact_payload = _load_mapping(
                manifest_path, "snapshotted trained artifact manifest"
            )
            expected_references = _projected_artifact_references(
                artifact_payload,
                manifest_path=manifest_path,
                result_dir=result_dir,
                snapshot_root=snapshot_root,
                file_records=file_records,
            )
            if references != expected_references:
                raise BenchmarkEvidenceError(
                    "artifact-reference projection does not match trained artifact manifest"
                )
        result_rows.append(result_row)
    if projection.get("source_declaration_sha256") != canonical_json_sha256(
        source_projection
    ):
        raise BenchmarkEvidenceError(
            "training-evidence source declaration projection hash mismatch"
        )
    return result_rows


def _projected_artifact_references(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    result_dir: Path,
    snapshot_root: Path,
    file_records: Mapping[str, JsonDict],
) -> List[JsonDict]:
    references: List[JsonDict] = []
    if int(payload.get("schema_version") or 0) != 2:
        raise BenchmarkEvidenceError(
            "snapshotted training evidence requires a schema-v2 trained artifact manifest"
        )
    if payload.get("kind") != "noema.trained_block_artifact":
        raise BenchmarkEvidenceError("snapshotted trained artifact kind is invalid")
    components = payload.get("components") or []
    if not isinstance(components, list):
        raise BenchmarkEvidenceError(
            "snapshotted trained artifact components must be an array"
        )
    for index, component in enumerate(components):
        if not isinstance(component, Mapping) or not component.get("path"):
            raise BenchmarkEvidenceError(
                "snapshotted trained artifact component %d requires path" % index
            )
        references.append(
            _projected_artifact_reference(
                manifest_path=manifest_path,
                result_dir=result_dir,
                snapshot_root=snapshot_root,
                file_records=file_records,
                declared=str(component.get("path") or ""),
                expected_sha256=str(component.get("sha256") or ""),
                kind="component",
                reference_id=str(component.get("id") or index),
                label="trained artifact component",
            )
        )
    contract = payload.get("contract")
    if isinstance(contract, Mapping) and contract.get("path"):
        references.append(
            _projected_artifact_reference(
                manifest_path=manifest_path,
                result_dir=result_dir,
                snapshot_root=snapshot_root,
                file_records=file_records,
                declared=str(contract.get("path") or ""),
                expected_sha256=str(contract.get("file_sha256") or ""),
                kind="contract",
                reference_id=str(contract.get("id") or "contract"),
                label="trained artifact contract",
            )
        )
    source = payload.get("source")
    source = source if isinstance(source, Mapping) else {}
    data_contract = source.get("data_contract")
    if isinstance(data_contract, Mapping) and data_contract.get("path"):
        references.append(
            _projected_artifact_reference(
                manifest_path=manifest_path,
                result_dir=result_dir,
                snapshot_root=snapshot_root,
                file_records=file_records,
                declared=str(data_contract.get("path") or ""),
                expected_sha256=str(data_contract.get("file_sha256") or ""),
                kind="data_contract",
                reference_id="training_data",
                label="trained artifact data contract",
            )
        )
    support_files = payload.get("support_files") or []
    if not isinstance(support_files, list):
        raise BenchmarkEvidenceError(
            "snapshotted trained artifact support_files must be an array"
        )
    for index, support_file in enumerate(support_files):
        if not isinstance(support_file, Mapping) or not support_file.get("path"):
            raise BenchmarkEvidenceError(
                "snapshotted trained artifact support file %d requires path" % index
            )
        references.append(
            _projected_artifact_reference(
                manifest_path=manifest_path,
                result_dir=result_dir,
                snapshot_root=snapshot_root,
                file_records=file_records,
                declared=str(support_file.get("path") or ""),
                expected_sha256=str(support_file.get("sha256") or ""),
                kind="support_file",
                reference_id=str(support_file.get("role") or index),
                label="trained artifact support file",
            )
        )
    references.sort(key=lambda row: (str(row["kind"]), str(row["id"]), str(row["path"])))
    return references


def _projected_artifact_reference(
    *,
    manifest_path: Path,
    result_dir: Path,
    snapshot_root: Path,
    file_records: Mapping[str, JsonDict],
    declared: str,
    expected_sha256: str,
    kind: str,
    reference_id: str,
    label: str,
) -> JsonDict:
    relative = Path(declared)
    if (
        not declared
        or relative.is_absolute()
        or _WINDOWS_ABSOLUTE_PATH_RE.match(declared)
        or ".." in relative.parts
    ):
        raise BenchmarkEvidenceError(
            "%s path must stay relative to its manifest" % label
        )
    expected = expected_sha256.strip().lower()
    if not _DIGEST_RE.fullmatch(expected):
        raise BenchmarkEvidenceError("%s requires a valid SHA-256 digest" % label)
    candidate = (manifest_path.parent / relative).resolve()
    if not _is_relative_to(candidate, snapshot_root) or not candidate.is_file():
        raise BenchmarkEvidenceError(
            "%s is missing from the training-evidence snapshot" % label
        )
    path = candidate.relative_to(result_dir.resolve()).as_posix()
    record = file_records.get(path)
    if record is None or record.get("sha256") != expected:
        raise BenchmarkEvidenceError(
            "%s hash is not covered by the snapshot manifest" % label
        )
    return {"kind": kind, "id": reference_id, **record}


def _result_training_evidence(result: Mapping[str, Any]) -> List[Any]:
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    metadata = benchmark.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    demo = metadata.get("demo")
    demo = demo if isinstance(demo, Mapping) else {}
    value = demo.get("training_evidence") or []
    if not isinstance(value, list):
        raise BenchmarkEvidenceError(
            "benchmark metadata.demo.training_evidence must be an array"
        )
    return list(value)


def _benchmark_source_training_evidence(
    benchmark_source: Mapping[str, Any],
) -> List[Any]:
    metadata = benchmark_source.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    demo = metadata.get("demo")
    demo = demo if isinstance(demo, Mapping) else {}
    value = demo.get("training_evidence") or []
    if not isinstance(value, list):
        raise BenchmarkEvidenceError(
            "benchmark source metadata.demo.training_evidence must be an array"
        )
    return list(value)


def _replace_result_training_evidence(
    result: MutableMapping[str, Any], rows: List[JsonDict]
) -> None:
    benchmark = dict(result.get("benchmark") or {})
    metadata = dict(benchmark.get("metadata") or {})
    demo = dict(metadata.get("demo") or {})
    demo["training_evidence"] = rows
    metadata["demo"] = demo
    benchmark["metadata"] = metadata
    result["benchmark"] = benchmark


def _source_context(
    benchmark_source: Mapping[str, Any], project_root: Path
) -> Tuple[Path, Path]:
    root = project_root.resolve()
    raw_pack_path = str(benchmark_source.get("path") or "").strip()
    if not raw_pack_path:
        return root, root
    pack_path = Path(raw_pack_path)
    if not pack_path.is_absolute():
        pack_path = root / pack_path
    pack_path = pack_path.resolve()
    if not _is_relative_to(pack_path, root):
        raise BenchmarkEvidenceError(
            "benchmark definition path is outside the project root"
        )
    base = pack_path.parent
    allowed_root = base.parent if base.name == "reference_training" else base
    return base, allowed_root.resolve()


def _normalize_source_spec(value: Any, field: str) -> Tuple[str, str]:
    if isinstance(value, str):
        path = value
        sha256 = ""
    elif isinstance(value, Mapping) and value.get("path"):
        path = str(value.get("path") or "")
        sha256 = str(value.get("sha256") or "").strip().lower()
    else:
        raise BenchmarkEvidenceError(
            "%s must be a path or an object with path" % field
        )
    _validated_declared_relative_path(path, field)
    if sha256 and not _DIGEST_RE.fullmatch(sha256):
        raise BenchmarkEvidenceError("%s.sha256 must be a SHA-256 digest" % field)
    return path, sha256


def _safe_source_file(
    base: Path,
    allowed_root: Path,
    declared: str,
    label: str,
    *,
    max_bytes: int,
) -> Path:
    candidate = _confined_source_candidate(base, allowed_root, declared, label)
    if not candidate.is_file():
        raise BenchmarkEvidenceError("declared %s file is missing: %s" % (label, declared))
    if candidate.stat().st_size > max_bytes:
        raise BenchmarkEvidenceError(
            "declared %s exceeds the %d byte snapshot limit" % (label, max_bytes)
        )
    if candidate.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise BenchmarkEvidenceError("declared %s must be JSON or YAML" % label)
    return candidate


def _confined_source_candidate(
    base: Path,
    allowed_root: Path,
    declared: str,
    label: str,
) -> Path:
    _validated_declared_relative_path(declared, label)
    path = Path(declared)
    _assert_no_symlink_components(base, path, label)
    candidate = (base / path).resolve()
    if not _is_relative_to(candidate, allowed_root.resolve()):
        raise BenchmarkEvidenceError("%s escapes the allowed evidence root" % label)
    return candidate


def _copy_snapshot_file(
    source: Path,
    staging_root: Path,
    destination_relative: Path,
    *,
    expected_sha256: Optional[str],
    max_bytes: int,
) -> JsonDict:
    if not source.is_file():
        raise BenchmarkEvidenceError("snapshot source file is missing: %s" % source)
    if source.stat().st_size > max_bytes:
        raise BenchmarkEvidenceError(
            "snapshot source exceeds the %d byte limit: %s" % (max_bytes, source)
        )
    destination = staging_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record = _snapshot_record(destination, destination_relative)
        if not expected_sha256 or record["sha256"] != expected_sha256:
            raise BenchmarkEvidenceError(
                "training-evidence snapshot path collision: %s"
                % destination_relative
            )
        return record
    source_before = _file_identity(source)
    shutil.copyfile(source, destination)
    record = _snapshot_record(destination, destination_relative)
    source_sha256 = file_sha256(source)
    source_after = _file_identity(source)
    if source_before != source_after or source_sha256 != record["sha256"]:
        destination.unlink(missing_ok=True)
        raise BenchmarkEvidenceError(
            "snapshot source changed while it was being copied: %s" % source
        )
    if expected_sha256 and record["sha256"] != expected_sha256:
        raise BenchmarkEvidenceError(
            "snapshot source hash mismatch for %s: expected %s, got %s"
            % (source, expected_sha256, record["sha256"])
        )
    return record


def _file_identity(path: Path) -> Tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _snapshot_record(path: Path, relative_to_snapshot: Path) -> JsonDict:
    return {
        "path": (Path(SNAPSHOT_DIRNAME) / relative_to_snapshot).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _record_snapshot_file(records: Dict[str, JsonDict], record: Mapping[str, Any]) -> None:
    plain = {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "size_bytes": int(record["size_bytes"]),
    }
    previous = records.get(plain["path"])
    if previous is not None and previous != plain:
        raise BenchmarkEvidenceError(
            "training-evidence snapshot path has conflicting content: %s"
            % plain["path"]
        )
    records[plain["path"]] = plain


def _required_snapshot_spec(
    value: Any, *, expected_path: str, label: str
) -> JsonDict:
    if not isinstance(value, Mapping):
        raise BenchmarkEvidenceError("%s descriptor is missing" % label)
    path = str(value.get("path") or "")
    sha256 = str(value.get("sha256") or "").lower()
    if path != expected_path:
        raise BenchmarkEvidenceError("%s path is invalid" % label)
    if not _DIGEST_RE.fullmatch(sha256):
        raise BenchmarkEvidenceError("%s SHA-256 is invalid" % label)
    return {"path": path, "sha256": sha256}


def _validated_file_record(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise BenchmarkEvidenceError("%s must be an object" % label)
    if set(value) != {"path", "sha256", "size_bytes"}:
        raise BenchmarkEvidenceError(
            "%s must contain path, sha256, and size_bytes" % label
        )
    path = str(value.get("path") or "")
    sha256 = str(value.get("sha256") or "").lower()
    size = value.get("size_bytes")
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise BenchmarkEvidenceError("%s path is invalid" % label)
    if not _DIGEST_RE.fullmatch(sha256):
        raise BenchmarkEvidenceError("%s SHA-256 is invalid" % label)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise BenchmarkEvidenceError("%s size_bytes is invalid" % label)
    return {"path": path, "sha256": sha256, "size_bytes": size}


def _validated_reference_record(value: Any, *, index: int) -> JsonDict:
    if not isinstance(value, Mapping):
        raise BenchmarkEvidenceError(
            "artifact reference %d must be an object" % index
        )
    if set(value) != {"kind", "id", "path", "sha256", "size_bytes"}:
        raise BenchmarkEvidenceError(
            "artifact reference %d has invalid fields" % index
        )
    kind = str(value.get("kind") or "")
    if kind not in {"component", "contract", "data_contract", "support_file"}:
        raise BenchmarkEvidenceError(
            "artifact reference %d kind is invalid" % index
        )
    reference_id = str(value.get("id") or "")
    if not reference_id:
        raise BenchmarkEvidenceError(
            "artifact reference %d id is missing" % index
        )
    record = _validated_file_record(
        {key: value.get(key) for key in ("path", "sha256", "size_bytes")},
        "artifact reference %d" % index,
    )
    return {"kind": kind, "id": reference_id, **record}


def _validated_source_declaration(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "declared_sha256",
        "snapshotted_sha256",
    }:
        raise BenchmarkEvidenceError("%s has invalid fields" % label)
    path = str(value.get("path") or "")
    _validated_declared_relative_path(path, label)
    declared = value.get("declared_sha256")
    if declared is not None:
        declared = str(declared).lower()
        if not _DIGEST_RE.fullmatch(declared):
            raise BenchmarkEvidenceError("%s declared SHA-256 is invalid" % label)
    snapshotted = str(value.get("snapshotted_sha256") or "").lower()
    if not _DIGEST_RE.fullmatch(snapshotted):
        raise BenchmarkEvidenceError("%s snapshot SHA-256 is invalid" % label)
    if declared and declared != snapshotted:
        raise BenchmarkEvidenceError(
            "%s declared SHA-256 does not match the snapshot" % label
        )
    return {
        "path": path,
        "declared_sha256": declared,
        "snapshotted_sha256": snapshotted,
    }


def _source_declaration_projection(entries: List[JsonDict]) -> List[JsonDict]:
    return [
        {
            "series": str(entry.get("series") or ""),
            "files": {
                field: dict((entry.get("source_declarations") or {})[field])
                for field in TRAINING_EVIDENCE_FIELDS
                if field in (entry.get("source_declarations") or {})
            },
        }
        for entry in entries
    ]


def _compare_source_declaration(
    raw_source_row: Any,
    projected: Mapping[str, Any],
    *,
    index: int,
) -> None:
    if not isinstance(raw_source_row, Mapping):
        raise BenchmarkEvidenceError(
            "benchmark source training-evidence entry %d must be an object" % index
        )
    unknown = sorted(
        set(raw_source_row) - {"series", *TRAINING_EVIDENCE_FIELDS}
    )
    if unknown:
        raise BenchmarkEvidenceError(
            "benchmark source training-evidence entry %d has unsupported fields: %s"
            % (index, ", ".join(unknown))
        )
    if str(raw_source_row.get("series") or "").strip() != projected.get("series"):
        raise BenchmarkEvidenceError(
            "benchmark source and snapshot evidence series do not match"
        )
    projected_files = projected.get("files")
    projected_files = projected_files if isinstance(projected_files, Mapping) else {}
    source_fields = {
        field
        for field in TRAINING_EVIDENCE_FIELDS
        if raw_source_row.get(field) not in (None, "")
    }
    if source_fields != set(projected_files):
        raise BenchmarkEvidenceError(
            "benchmark source and snapshot evidence fields do not match"
        )
    for field in TRAINING_EVIDENCE_FIELDS:
        if field not in source_fields:
            continue
        path, declared_sha = _normalize_source_spec(raw_source_row[field], field)
        declaration = _validated_source_declaration(
            projected_files[field], "projection %s source" % field
        )
        if declaration["path"] != path or declaration["declared_sha256"] != (
            declared_sha or None
        ):
            raise BenchmarkEvidenceError(
                "benchmark source %s declaration does not match the snapshot projection"
                % field
            )


def _confined_snapshot_file(
    result_dir: Path, snapshot_root: Path, declared: str, label: str
) -> Path:
    path = Path(declared)
    if (
        not declared
        or path.is_absolute()
        or _WINDOWS_ABSOLUTE_PATH_RE.match(declared)
        or ".." in path.parts
    ):
        raise BenchmarkEvidenceError("%s path is unsafe" % label)
    unresolved = result_dir / path
    result_resolved = result_dir.resolve()
    unresolved_root = result_dir / SNAPSHOT_DIRNAME
    if unresolved_root.is_symlink():
        raise BenchmarkEvidenceError("training-evidence snapshot root cannot be a symlink")
    root_resolved = unresolved_root.resolve()
    if not _is_relative_to(root_resolved, result_resolved):
        raise BenchmarkEvidenceError("training-evidence snapshot root escapes the result")
    cursor = result_dir
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkEvidenceError("%s cannot traverse a symlink" % label)
    candidate = unresolved.resolve()
    if not _is_relative_to(candidate, snapshot_root.resolve()):
        raise BenchmarkEvidenceError("%s escapes the benchmark result snapshot" % label)
    if not candidate.is_file() or candidate.is_symlink():
        raise BenchmarkEvidenceError("%s is missing or not a regular snapshot file" % label)
    return candidate


def _validated_declared_relative_path(declared: str, label: str) -> None:
    path = Path(declared)
    if (
        not declared
        or path.is_absolute()
        or _WINDOWS_ABSOLUTE_PATH_RE.match(declared)
        or "\\" in declared
        or any(ord(char) < 32 or ord(char) == 127 for char in declared)
    ):
        raise BenchmarkEvidenceError(
            "%s must be a normalized relative path" % label
        )


def _assert_no_symlink_components(base: Path, relative: Path, label: str) -> None:
    cursor = base
    if cursor.is_symlink():
        raise BenchmarkEvidenceError("%s base cannot be a symlink" % label)
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkEvidenceError(
                "%s cannot traverse a symlink: %s" % (label, cursor)
            )


def _load_mapping(path: Path, label: str) -> JsonDict:
    try:
        payload = load_strict_yaml_or_json(path)
    except (OSError, StructuredInputError) as exc:
        raise BenchmarkEvidenceError("could not parse %s: %s" % (label, exc)) from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkEvidenceError("%s must contain an object" % label)
    return dict(payload)


def _load_json_mapping(path: Path, label: str) -> JsonDict:
    try:
        payload = load_strict_yaml_or_json(path)
    except (OSError, StructuredInputError) as exc:
        raise BenchmarkEvidenceError("could not parse %s: %s" % (label, exc)) from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkEvidenceError("%s must contain an object" % label)
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return slug or "series"


def _unique_slug(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    counter = 2
    while "%s-%d" % (candidate, counter) in used:
        counter += 1
    return "%s-%d" % (candidate, counter)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False
