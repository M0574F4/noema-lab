from __future__ import annotations

"""Small fail-closed I/O helpers shared by publication study harnesses."""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Dict, Mapping

from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)


JsonDict = Dict[str, Any]
_SHA256_CHARS = set("0123456789abcdef")


class StudyDocumentError(ValueError):
    """Raised when a study document or binding is unsafe or malformed."""


def load_study_mapping(path: Path) -> JsonDict:
    """Load a finite JSON/YAML mapping without accepting object constructors."""

    try:
        value = load_strict_yaml_or_json(path)
    except (OSError, UnicodeError, StructuredInputError) as exc:
        raise StudyDocumentError("cannot load %s: %s" % (path, exc)) from exc
    if not isinstance(value, Mapping):
        raise StudyDocumentError("%s must contain a mapping" % path)
    _assert_finite_json_tree(value, str(path))
    return dict(value)


def write_study_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write canonical, finite JSON without following a target symlink."""

    _assert_finite_json_tree(payload, "$")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise StudyDocumentError("refusing to replace symlink: %s" % path)
    fd, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in _SHA256_CHARS for char in digest):
        raise StudyDocumentError("%s must be a lowercase SHA-256 digest" % label)
    return digest


def safe_relative_path(value: Any, label: str) -> PurePosixPath:
    """Return a portable, non-empty relative path with no traversal segments."""

    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise StudyDocumentError("%s must be a safe relative POSIX path" % label)
    return path


def resolve_bound_file(root: Path, relative: Any, expected_sha256: Any, label: str) -> Path:
    """Resolve a declared regular file below ``root`` and verify its bytes."""

    root = Path(root).resolve(strict=True)
    rel = safe_relative_path(relative, "%s.path" % label)
    lexical = root / Path(*rel.parts)
    for index in range(1, len(rel.parts) + 1):
        if (root / Path(*rel.parts[:index])).is_symlink():
            raise StudyDocumentError("%s traverses a symlink" % label)
    candidate = lexical.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StudyDocumentError("%s escapes its declared root" % label) from exc
    if not candidate.is_file():
        raise StudyDocumentError("%s must resolve to a regular file" % label)
    expected = require_sha256(expected_sha256, "%s.sha256" % label)
    actual = file_sha256(candidate)
    if actual != expected:
        raise StudyDocumentError("%s byte digest mismatch" % label)
    return candidate


def content_bound_document(payload: Mapping[str, Any]) -> JsonDict:
    """Copy a mapping and append a deterministic self hash."""

    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_json_sha256(result)
    return result


def verify_content_bound_document(payload: Mapping[str, Any], label: str) -> JsonDict:
    result = dict(payload)
    declared = require_sha256(result.pop("sha256", None), "%s.sha256" % label)
    if canonical_json_sha256(result) != declared:
        raise StudyDocumentError("%s self hash does not match its content" % label)
    result["sha256"] = declared
    return result


def _assert_finite_json_tree(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StudyDocumentError("non-finite number at %s" % path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json_tree(item, "%s[%d]" % (path, index))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StudyDocumentError("non-string object key at %s" % path)
            _assert_finite_json_tree(item, "%s.%s" % (path, key))
        return
    raise StudyDocumentError("non-JSON value %s at %s" % (type(value).__name__, path))
