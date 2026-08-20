from __future__ import annotations

"""Exact source-tree commitments for protected local execution."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


JsonDict = Dict[str, Any]


class SourceClosureError(RuntimeError):
    """Raised when a source tree cannot be committed or verified safely."""


def build_executable_source_closure(
    source_root: Path,
    *,
    root_label: str = "src/noema_lab",
) -> JsonDict:
    """Commit every regular non-bytecode file below ``source_root``.

    Directory entries and records are sorted canonically. Symlinks and special
    filesystem nodes fail closed; the only excluded regular files are generated
    ``.pyc`` and ``.pyo`` bytecode files.
    """

    source_root = Path(source_root).absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise SourceClosureError(
            "executable source root is missing or is a symlink"
        )
    files: list[JsonDict] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceClosureError(
                "cannot enumerate executable source closure: %s" % exc
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(source_root).as_posix()
            if not relative or "\\" in relative:
                raise SourceClosureError(
                    "executable source path is not canonical"
                )
            if entry.is_symlink():
                raise SourceClosureError(
                    "executable source closure contains a symlink: %s"
                    % relative
                )
            if entry.is_dir(follow_symlinks=False):
                visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise SourceClosureError(
                    "executable source closure contains a special node: %s"
                    % relative
                )
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            try:
                before = path.stat()
                digest = _file_sha256(path)
                after = path.stat()
            except OSError as exc:
                raise SourceClosureError(
                    "cannot hash executable source file %s: %s"
                    % (relative, exc)
                ) from exc
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_mode != after.st_mode
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise SourceClosureError(
                    "executable source file changed while hashing: %s"
                    % relative
                )
            files.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": int(after.st_size),
                }
            )

    visit(source_root)
    files.sort(key=lambda record: str(record["path"]))
    if not files or [record["path"] for record in files] != sorted(
        record["path"] for record in files
    ):
        raise SourceClosureError(
            "executable source closure is empty or unordered"
        )
    python_files = [
        record for record in files if Path(record["path"]).suffix == ".py"
    ]
    if not python_files:
        raise SourceClosureError(
            "executable source closure contains no Python sources"
        )
    return {
        "schema_version": 1,
        "kind": "noema.executable_source_closure",
        "root": root_label,
        "inclusion_policy": (
            "all_recursive_regular_files_excluding_only_generated_pyc_and_pyo"
        ),
        "excluded_generated_suffixes": [".pyc", ".pyo"],
        "files": files,
        "file_count": len(files),
        "python_file_count": len(python_files),
        "total_bytes": sum(int(record["size_bytes"]) for record in files),
        "file_set_sha256": _canonical_json_sha256(files),
        "python_file_set_sha256": _canonical_json_sha256(python_files),
    }


def verify_executable_source_closure(
    declared: object,
    source_root: Path,
    *,
    root_label: str = "src/noema_lab",
) -> JsonDict:
    """Rebuild and exact-compare a declared executable source closure."""

    observed = build_executable_source_closure(
        source_root,
        root_label=root_label,
    )
    if not isinstance(declared, Mapping) or dict(declared) != observed:
        raise SourceClosureError("declared executable source closure changed")
    return observed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
