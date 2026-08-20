from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from noema_lab.core.structured_input import decode_strict_yaml_or_json

JsonDict = Dict[str, Any]

_RUN_LIST_INDEX_SCHEMA_VERSION = 1
_RUN_LIST_INDEX_FILENAME = ".run-list-index.json"
_RUN_ID_RESERVATIONS_DIRNAME = ".run-id-reservations"


class LocalStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.runs_dir = workspace / "runs"
        self.benchmarks_dir = workspace / "benchmarks"
        self.run_id_reservations_dir = workspace / _RUN_ID_RESERVATIONS_DIRNAME
        self._run_list_index_entries: Optional[JsonDict] = None
        self._run_list_index_lock = threading.RLock()

    def ensure(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, recipe_name: str) -> Path:
        self.ensure()
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in recipe_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self._claim_unique_run_directory(timestamp, safe_name)
        (run_dir / "artifacts").mkdir()
        return run_dir

    def _claim_unique_run_directory(
        self,
        timestamp: str,
        safe_name: str,
    ) -> Path:
        """Persistently reserve a run ID so pruning cannot make it reusable."""

        reservations = self.run_id_reservations_dir
        if self.runs_dir.is_symlink() or reservations.is_symlink():
            raise OSError("run storage paths must not be symlinks")
        reservations.mkdir(parents=True, exist_ok=True)
        if reservations.is_symlink() or not reservations.is_dir():
            raise OSError("run ID reservation path must be a safe directory")
        counter = 1
        while True:
            suffix = "" if counter == 1 else "_%d" % counter
            name = "%s_%s%s" % (timestamp, safe_name, suffix)
            candidate = self.runs_dir / name
            reservation = reservations / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(str(reservation), flags, 0o600)
            except FileExistsError:
                counter += 1
                continue
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                # Preserve the reservation for this legacy/live directory and
                # claim the next suffix.
                counter += 1
                continue
            except Exception:
                reservation.unlink(missing_ok=True)
                raise
            reservations_descriptor = os.open(str(reservations), os.O_RDONLY)
            try:
                os.fsync(reservations_descriptor)
            finally:
                os.close(reservations_descriptor)
            return candidate

    def list_runs(self, *, read_only: bool = False) -> List[JsonDict]:
        if self.runs_dir.is_symlink():
            raise OSError("run storage path must not be a symlink")
        if not self.runs_dir.is_dir():
            if read_only:
                return []
            self.ensure()
        if read_only:
            rows: List[JsonDict] = []
            for summary_path in sorted(self.runs_dir.glob("*/summary.json")):
                if summary_path.is_symlink() or summary_path.parent.is_symlink():
                    continue
                rows.append(
                    self._run_list_row(
                        self.read_json(summary_path),
                        summary_path.parent.name,
                    )
                )
            return rows
        # summary.json is always authoritative. The index only avoids decoding a
        # potentially large summary when its source path and full filesystem
        # identity are unchanged. We still scan the directory on every call so
        # new, removed, and renamed legacy runs are observed immediately.
        with self._run_list_index_lock:
            cached_entries = self._run_list_index()
            current_entries: JsonDict = {}
            rows: List[JsonDict] = []
            for summary_path in sorted(self.runs_dir.glob("*/summary.json")):
                relative_path = summary_path.relative_to(self.runs_dir).as_posix()
                run_id = summary_path.parent.name
                row, entry = self._indexed_run_row(
                    summary_path,
                    relative_path=relative_path,
                    run_id=run_id,
                    cached_entry=cached_entries.get(relative_path),
                )
                if entry is None:
                    # The source changed repeatedly or disappeared while being
                    # inspected. Do not persist an entry for an unstable file.
                    continue
                current_entries[relative_path] = entry
                if row is not None:
                    rows.append(row)

            if current_entries != cached_entries:
                self._write_run_list_index(current_entries)
            self._run_list_index_entries = current_entries
            return rows

    def _run_list_index(self) -> JsonDict:
        if self._run_list_index_entries is not None:
            return self._run_list_index_entries
        index_path = self.runs_dir / _RUN_LIST_INDEX_FILENAME
        try:
            payload = decode_strict_yaml_or_json(
                index_path.read_text(encoding="utf-8"),
                input_format="json",
            )
        except (OSError, ValueError, TypeError):
            self._run_list_index_entries = {}
            return self._run_list_index_entries
        if not isinstance(payload, Mapping) or payload.get("schema_version") != _RUN_LIST_INDEX_SCHEMA_VERSION:
            self._run_list_index_entries = {}
            return self._run_list_index_entries
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, Mapping):
            self._run_list_index_entries = {}
            return self._run_list_index_entries
        self._run_list_index_entries = {
            str(path): dict(entry)
            for path, entry in raw_entries.items()
            if isinstance(path, str) and isinstance(entry, Mapping)
        }
        return self._run_list_index_entries

    def _indexed_run_row(
        self,
        summary_path: Path,
        *,
        relative_path: str,
        run_id: str,
        cached_entry: Any,
    ) -> Tuple[Optional[JsonDict], Optional[JsonDict]]:
        # Retry if a live run rewrites its summary while it is being read. A
        # derived entry is only attached to a stable source identity.
        for _attempt in range(3):
            identity = self._run_summary_identity(summary_path, relative_path)
            if identity is None:
                return None, None

            cache_hit, cached_row = self._cached_run_row(cached_entry, identity, run_id)
            if cache_hit:
                identity_after = self._run_summary_identity(summary_path, relative_path)
                if identity_after == identity:
                    entry: JsonDict = {"source": identity, "valid": cached_row is not None}
                    if cached_row is not None:
                        entry["row"] = self._copy_run_list_row(cached_row)
                    return cached_row, entry
                cached_entry = None
                continue

            read_error: Optional[Exception] = None
            try:
                summary = self.read_json(summary_path)
                row = self._run_list_row(summary, run_id)
                valid = True
            except Exception as exc:
                row = None
                valid = False
                read_error = exc

            identity_after = self._run_summary_identity(summary_path, relative_path)
            if identity_after != identity:
                cached_entry = None
                continue
            if read_error is not None:
                raise ValueError(
                    "Invalid run summary %s: %s" % (summary_path, read_error)
                ) from read_error
            entry: JsonDict = {"source": identity, "valid": valid}
            if row is not None:
                entry["row"] = self._copy_run_list_row(row)
            return row, entry
        return None, None

    def _run_summary_identity(self, summary_path: Path, relative_path: str) -> Optional[JsonDict]:
        if summary_path.is_symlink() or summary_path.parent.is_symlink():
            return None
        try:
            stat = summary_path.stat()
        except OSError:
            return None
        if not summary_path.is_file():
            return None
        return {
            "path": relative_path,
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "size": int(stat.st_size),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }

    def _cached_run_row(
        self,
        cached_entry: Any,
        identity: JsonDict,
        run_id: str,
    ) -> Tuple[bool, Optional[JsonDict]]:
        if not isinstance(cached_entry, Mapping) or cached_entry.get("source") != identity:
            return False, None
        if cached_entry.get("valid") is False:
            # Older indexes cached malformed authoritative summaries as
            # invisible rows. Re-read them so listing fails explicitly.
            return False, None
        if cached_entry.get("valid") is not True:
            return False, None
        row = cached_entry.get("row")
        if not isinstance(row, Mapping) or row.get("run_id") != run_id:
            return False, None
        suite = row.get("suite")
        if not isinstance(suite, Mapping):
            return False, None
        return (
            True,
            {
                "run_id": run_id,
                "status": row.get("status"),
                "recipe_name": row.get("recipe_name"),
                "suite": dict(suite),
            },
        )

    def _run_list_row(self, summary: Any, run_id: str) -> JsonDict:
        if not isinstance(summary, Mapping):
            raise ValueError("Run summary must be a JSON object")
        recipe = summary.get("recipe") or {}
        if not isinstance(recipe, Mapping):
            raise ValueError("Run summary recipe must be a JSON object")
        suite = recipe.get("suite") or {}
        if not isinstance(suite, Mapping):
            raise ValueError("Run summary recipe suite must be a JSON object")
        return {
            "run_id": run_id,
            "status": summary.get("status"),
            "recipe_name": summary.get("recipe_name"),
            "suite": dict(suite),
        }

    def _copy_run_list_row(self, row: Mapping[str, Any]) -> JsonDict:
        return {
            "run_id": row.get("run_id"),
            "status": row.get("status"),
            "recipe_name": row.get("recipe_name"),
            "suite": dict(row.get("suite") or {}),
        }

    def _write_run_list_index(self, entries: JsonDict) -> None:
        payload = {
            "schema_version": _RUN_LIST_INDEX_SCHEMA_VERSION,
            "entries": entries,
        }
        index_path = self.runs_dir / _RUN_LIST_INDEX_FILENAME
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.runs_dir),
                prefix=".%s." % _RUN_LIST_INDEX_FILENAME,
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(index_path))
        except OSError:
            # Listing runs must not depend on the availability of this derived
            # optimization. Keep the validated entries in memory and continue.
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def get_run_dir(self, run_id: str) -> Path:
        """Return a direct, non-symlink run directory below ``runs_dir``.

        Run identifiers are supplied by CLI and HTTP inspection surfaces.  A
        path-shaped identifier or a symlinked run directory must never turn a
        read-only inspection into an arbitrary filesystem read.
        """

        return self._contained_bundle_dir(self.runs_dir, run_id, "Run")

    def get_run(self, run_id: str) -> JsonDict:
        return self.read_json(
            self._contained_bundle_file(
                self.get_run_dir(run_id),
                "summary.json",
                "Run summary",
            )
        )

    def get_manifest(self, run_id: str) -> JsonDict:
        return self.read_json(
            self._contained_bundle_file(
                self.get_run_dir(run_id),
                "manifest.json",
                "Run manifest",
            )
        )

    def create_benchmark_dir(self, benchmark_id: str) -> Path:
        self.ensure()
        safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in benchmark_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self._claim_unique_directory(self.benchmarks_dir, timestamp, safe_id)

    @staticmethod
    def _claim_unique_directory(root: Path, timestamp: str, safe_name: str) -> Path:
        """Atomically claim a timestamped directory across threads/processes."""

        counter = 1
        while True:
            suffix = "" if counter == 1 else "_%d" % counter
            candidate = root / ("%s_%s%s" % (timestamp, safe_name, suffix))
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                counter += 1
                continue
            return candidate

    def list_benchmark_results(
        self,
        *,
        read_only: bool = False,
    ) -> List[JsonDict]:
        if self.benchmarks_dir.is_symlink():
            raise OSError("benchmark storage path must not be a symlink")
        if not self.benchmarks_dir.is_dir():
            if read_only:
                return []
            self.ensure()
        rows = []
        for result_path in sorted(self.benchmarks_dir.glob("*/result.json")):
            result = self.read_json(result_path)
            raw_benchmark = result.get("benchmark")
            if not isinstance(raw_benchmark, Mapping):
                raise ValueError(
                    "Benchmark result must contain an object-valued benchmark: %s"
                    % result_path
                )
            benchmark = dict(raw_benchmark)
            raw_suite = benchmark.get("suite") or {}
            if not isinstance(raw_suite, Mapping):
                raise ValueError(
                    "Benchmark result suite must be a JSON object: %s" % result_path
                )
            raw_recipes = result.get("recipes") or []
            if not isinstance(raw_recipes, list):
                raise ValueError(
                    "Benchmark result recipes must be a JSON array: %s" % result_path
                )
            rows.append(
                {
                    "result_id": result_path.parent.name,
                    "status": result.get("status"),
                    "benchmark_id": benchmark.get("id"),
                    "benchmark_version": benchmark.get("version"),
                    "benchmark_name": benchmark.get("name"),
                    "suite": dict(raw_suite),
                    "created_at_utc": result.get("created_at_utc"),
                    "completed_at_utc": result.get("completed_at_utc"),
                    "recipe_count": len(raw_recipes),
                }
            )
        return rows

    def get_benchmark_result(self, result_id: str) -> JsonDict:
        return self.read_json(
            self._contained_bundle_file(
                self.get_benchmark_result_dir(result_id),
                "result.json",
                "Benchmark result",
            )
        )

    def get_benchmark_result_dir(self, result_id: str) -> Path:
        return self._contained_bundle_dir(
            self.benchmarks_dir,
            result_id,
            "Benchmark result",
        )

    @staticmethod
    def _contained_bundle_dir(root: Path, bundle_id: str, label: str) -> Path:
        child = Path(bundle_id)
        if (
            not bundle_id
            or bundle_id in {".", ".."}
            or child.is_absolute()
            or len(child.parts) != 1
            or child.name != bundle_id
            or "\\" in bundle_id
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in bundle_id
            )
        ):
            raise FileNotFoundError(
                "%s id must be a directory name: %s" % (label, bundle_id)
            )
        if root.is_symlink():
            raise FileNotFoundError("%s storage root must not be a symlink" % label)
        candidate = root / bundle_id
        if candidate.is_symlink():
            raise FileNotFoundError(
                "%s directory must not be a symlink: %s" % (label, bundle_id)
            )
        try:
            candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(
                "%s directory escapes its storage root: %s" % (label, bundle_id)
            ) from exc
        return candidate

    @staticmethod
    def _contained_bundle_file(bundle_dir: Path, filename: str, label: str) -> Path:
        if bundle_dir.is_symlink():
            raise FileNotFoundError("%s directory must not be a symlink" % label)
        path = bundle_dir / filename
        if path.is_symlink():
            raise FileNotFoundError("%s must not be a symlink: %s" % (label, path))
        try:
            resolved_bundle = bundle_dir.resolve(strict=True)
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_bundle)
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(
                "%s is missing or outside its bundle directory: %s" % (label, path)
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError("%s is not a regular file: %s" % (label, path))
        return path

    def write_json(self, path: Path, payload: JsonDict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=".%s." % path.name,
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(path))
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def read_json(self, path: Path) -> JsonDict:
        payload = decode_strict_yaml_or_json(
            path.read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(payload, dict):
            raise ValueError("JSON document must contain an object: %s" % path)
        return payload
