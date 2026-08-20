from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Dict, Iterator, List, Mapping, Optional
from uuid import uuid4

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.reproducibility import canonical_json_sha256, utc_now_iso
from noema_lab.core.structured_input import decode_strict_yaml_or_json

JsonDict = Dict[str, Any]

ATTEMPT_LEDGER_SCHEMA_VERSION = 1
ATTEMPT_EVENT_KIND = "noema.benchmark_attempt_event"
ATTEMPT_SNAPSHOT_KIND = "noema.benchmark_attempt_ledger_snapshot"
TERMINAL_ATTEMPT_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "incomplete",
    "resource_rejected",
}


class AttemptLedgerError(ValueError):
    pass


@contextmanager
def _exclusive_file_lock(handle: BinaryIO) -> Iterator[None]:
    """Hold an advisory one-byte lock on POSIX or Windows."""

    if _fcntl is not None:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows CI
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        _msvcrt.locking(handle.fileno(), _msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)
        return
    raise AttemptLedgerError(
        "attempt ledger locking is unavailable on this platform"
    )


class BenchmarkAttemptLedger:
    """Append-only, atomic, hash-chained benchmark invocation ledger.

    The event files are authoritative; ``head.json`` is only an atomically
    replaced index. A started attempt remains in the denominator even if the
    process dies before a terminal event can be appended.
    """

    def __init__(self, benchmarks_root: Path) -> None:
        self.benchmarks_root = benchmarks_root.expanduser().resolve()
        self.ledger_dir = self.benchmarks_root / ".attempt-ledger"
        self.events_dir = self.ledger_dir / "events"
        self.head_path = self.ledger_dir / "head.json"
        self.lock_path = self.ledger_dir / "ledger.lock"

    def begin(
        self,
        *,
        benchmark_id: str,
        benchmark_version: str,
        protocol_sha256: str,
        invocation: Optional[Mapping[str, Any]] = None,
        attempt_id: Optional[str] = None,
    ) -> JsonDict:
        benchmark_id = _required_text(benchmark_id, "benchmark_id")
        benchmark_version = _required_text(benchmark_version, "benchmark_version")
        protocol_sha256 = _required_sha256(protocol_sha256, "protocol_sha256")
        resolved_attempt_id = attempt_id or uuid4().hex
        if (
            not isinstance(resolved_attempt_id, str)
            or not resolved_attempt_id
            or Path(resolved_attempt_id).name != resolved_attempt_id
        ):
            raise AttemptLedgerError("attempt_id must be a safe directory-name token")
        payload: JsonDict = {
            "benchmark_id": benchmark_id,
            "benchmark_version": benchmark_version,
            "protocol_sha256": protocol_sha256,
            "invocation": dict(invocation or {}),
        }
        canonical_json_sha256(payload)
        return self._append("started", resolved_attempt_id, payload)

    def finalize(
        self,
        attempt_id: str,
        *,
        status: str,
        result_dir: Optional[Path] = None,
        recipe_outcomes: Optional[List[Mapping[str, Any]]] = None,
        error: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> JsonDict:
        status = str(status or "").strip().lower()
        if status not in TERMINAL_ATTEMPT_STATUSES:
            raise AttemptLedgerError(
                "attempt status must be one of %s"
                % ", ".join(sorted(TERMINAL_ATTEMPT_STATUSES))
            )
        payload: JsonDict = {
            "status": status,
            "recipe_outcomes": [dict(row) for row in (recipe_outcomes or [])],
            "details": dict(details or {}),
        }
        if error:
            payload["error"] = str(error)
        if result_dir is not None:
            payload["result"] = self._result_identity(result_dir)
        canonical_json_sha256(payload)
        return self._append("finalized", attempt_id, payload)

    def supersede(
        self,
        attempt_id: str,
        *,
        superseded_by_attempt_id: str,
        reason: str,
    ) -> JsonDict:
        payload = {
            "superseded_by_attempt_id": _required_text(
                superseded_by_attempt_id,
                "superseded_by_attempt_id",
            ),
            "reason": _required_text(reason, "reason"),
        }
        return self._append("superseded", attempt_id, payload)

    def verify(self) -> JsonDict:
        try:
            events = self._load_verified_events()
            attempts = _project_attempts(events)
        except (AttemptLedgerError, OSError, ValueError, TypeError) as exc:
            return {
                "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
                "kind": ATTEMPT_SNAPSHOT_KIND,
                "status": "invalid",
                "ledger": str(self.ledger_dir),
                "errors": [str(exc)],
                "events": [],
                "attempts": [],
            }
        return _snapshot_payload(self.ledger_dir, events, attempts)

    def registration_snapshot(self, destination: Optional[Path] = None) -> JsonDict:
        """Return, and optionally atomically write, a paper-registerable view."""

        report = self.verify()
        if report["status"] != "valid":
            raise AttemptLedgerError(
                "cannot register an invalid attempt ledger: %s"
                % "; ".join(report.get("errors") or [])
            )
        snapshot = {
            key: value
            for key, value in report.items()
            if key not in {"ledger"}
        }
        snapshot["sha256"] = canonical_json_sha256(snapshot)
        if destination is not None:
            destination = destination.expanduser()
            if destination.is_symlink():
                raise AttemptLedgerError("registration snapshot destination must not be a symlink")
            absolute_destination = Path(os.path.abspath(str(destination)))
            absolute_events = Path(os.path.abspath(str(self.events_dir)))
            try:
                absolute_destination.relative_to(absolute_events)
            except ValueError:
                pass
            else:
                raise AttemptLedgerError(
                    "registration snapshot cannot overwrite immutable attempt events"
                )
            if absolute_destination in {
                Path(os.path.abspath(str(self.head_path))),
                Path(os.path.abspath(str(self.lock_path))),
            }:
                raise AttemptLedgerError(
                    "registration snapshot cannot overwrite ledger control files"
                )
            _atomic_write_json(absolute_destination, snapshot)
            snapshot["path"] = str(absolute_destination.resolve())
            snapshot["file_sha256"] = file_sha256(absolute_destination.resolve())
        return snapshot

    def _append(self, event_type: str, attempt_id: str, payload: JsonDict) -> JsonDict:
        attempt_id = _required_text(attempt_id, "attempt_id")
        if self.ledger_dir.is_symlink():
            raise AttemptLedgerError("attempt ledger directory must not be a symlink")
        self.events_dir.mkdir(parents=True, exist_ok=True)
        if self.ledger_dir.is_symlink() or self.events_dir.is_symlink():
            raise AttemptLedgerError("attempt ledger paths must not be symlinks")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise AttemptLedgerError("attempt ledger lock must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        lock_fd = os.open(str(self.lock_path), flags, 0o600)
        with os.fdopen(lock_fd, "a+b") as lock_handle:
            with _exclusive_file_lock(lock_handle):
                events = self._load_verified_events()
                attempts = _project_attempts(events)
                payload = dict(payload)
                if event_type == "started":
                    payload = _reserve_test_access(payload, attempts)
                _validate_transition(event_type, attempt_id, payload, attempts)
                sequence = len(events) + 1
                previous_sha256 = events[-1]["event_sha256"] if events else None
                event_without_digest: JsonDict = {
                    "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
                    "kind": ATTEMPT_EVENT_KIND,
                    "sequence": sequence,
                    "event_type": event_type,
                    "attempt_id": attempt_id,
                    "recorded_at_utc": utc_now_iso(),
                    "previous_event_sha256": previous_sha256,
                    "payload": payload,
                }
                event = dict(event_without_digest)
                event["event_sha256"] = canonical_json_sha256(event_without_digest)
                event_path = self.events_dir / (
                    "%012d-%s.json" % (sequence, event["event_sha256"])
                )
                _atomic_write_json(event_path, event, require_absent=True)
                _fsync_directory(self.events_dir)
                head = {
                    "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
                    "kind": "noema.benchmark_attempt_ledger_head",
                    "event_count": sequence,
                    "head_event_sha256": event["event_sha256"],
                    "head_event": event_path.name,
                }
                _atomic_write_json(self.head_path, head)
                _fsync_directory(self.ledger_dir)
                return dict(event)

    def _load_verified_events(self) -> List[JsonDict]:
        if not self.events_dir.exists():
            return []
        if self.events_dir.is_symlink() or not self.events_dir.is_dir():
            raise AttemptLedgerError("attempt ledger events path is unsafe")
        paths = sorted(self.events_dir.glob("*.json"), key=lambda item: item.name)
        events: List[JsonDict] = []
        previous: Optional[str] = None
        for expected_sequence, path in enumerate(paths, start=1):
            if path.is_symlink() or not path.is_file():
                raise AttemptLedgerError("attempt event is unsafe: %s" % path)
            try:
                raw = decode_strict_yaml_or_json(
                    path.read_text(encoding="utf-8"),
                    input_format="json",
                )
            except (OSError, ValueError) as exc:
                raise AttemptLedgerError("cannot decode attempt event %s: %s" % (path.name, exc)) from exc
            if not isinstance(raw, Mapping):
                raise AttemptLedgerError("attempt event is not an object: %s" % path.name)
            event = dict(raw)
            digest = str(event.pop("event_sha256", ""))
            expected_digest = canonical_json_sha256(event)
            if digest != expected_digest:
                raise AttemptLedgerError("attempt event digest mismatch: %s" % path.name)
            if event.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION or event.get("kind") != ATTEMPT_EVENT_KIND:
                raise AttemptLedgerError("attempt event schema/kind is invalid: %s" % path.name)
            if event.get("sequence") != expected_sequence:
                raise AttemptLedgerError("attempt event sequence is not contiguous: %s" % path.name)
            if event.get("previous_event_sha256") != previous:
                raise AttemptLedgerError("attempt event hash chain is broken: %s" % path.name)
            expected_name = "%012d-%s.json" % (expected_sequence, digest)
            if path.name != expected_name:
                raise AttemptLedgerError("attempt event filename does not bind its digest: %s" % path.name)
            event["event_sha256"] = digest
            events.append(event)
            previous = digest
        # Validate the complete state machine, not only individual hashes.
        _project_attempts(events)
        # head.json is an optimization and may lag after a crash between the
        # immutable event commit and index replacement. The event chain is
        # authoritative and the next append repairs the head. A head claiming
        # more events than remain, or disagreeing at the same length, is still
        # evidence of truncation/tampering and fails closed.
        if self.head_path.is_symlink():
            raise AttemptLedgerError("attempt ledger head is unsafe")
        if self.head_path.exists():
            if not self.head_path.is_file():
                raise AttemptLedgerError("attempt ledger head is unsafe")
            try:
                head = decode_strict_yaml_or_json(
                    self.head_path.read_text(encoding="utf-8"),
                    input_format="json",
                )
            except (OSError, ValueError) as exc:
                raise AttemptLedgerError("attempt ledger head is unreadable: %s" % exc) from exc
            if not isinstance(head, Mapping):
                raise AttemptLedgerError("attempt ledger head is not an object")
            head_count = head.get("event_count")
            if not isinstance(head_count, int) or isinstance(head_count, bool) or head_count < 0:
                raise AttemptLedgerError("attempt ledger head count is invalid")
            if head_count > len(events):
                raise AttemptLedgerError("attempt event chain was truncated below its recorded head")
            if head_count == len(events) and head.get("head_event_sha256") != previous:
                raise AttemptLedgerError("attempt ledger head digest does not match the event chain")
        return events

    def _result_identity(self, result_dir: Path) -> JsonDict:
        candidate = result_dir.expanduser()
        if candidate.is_symlink():
            raise AttemptLedgerError("result_dir must not be a symlink")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.benchmarks_root)
        except ValueError as exc:
            raise AttemptLedgerError("result_dir is outside the benchmark store") from exc
        if relative == Path(".") or len(relative.parts) != 1 or not resolved.is_dir():
            raise AttemptLedgerError("result_dir must be one direct benchmark-result directory")
        result_json = resolved / "result.json"
        if result_json.is_symlink() or not result_json.is_file():
            raise AttemptLedgerError("result_dir does not contain a safe result.json")
        return {
            "result_id": resolved.name,
            "result_json_sha256": file_sha256(result_json),
            "result_json_size_bytes": int(result_json.stat().st_size),
        }


def _validate_transition(
    event_type: str,
    attempt_id: str,
    payload: Mapping[str, Any],
    attempts: Mapping[str, JsonDict],
) -> None:
    if event_type == "started":
        if attempt_id in attempts:
            raise AttemptLedgerError("attempt_id is already recorded: %s" % attempt_id)
        return
    if attempt_id not in attempts:
        raise AttemptLedgerError("attempt_id has no started event: %s" % attempt_id)
    attempt = attempts[attempt_id]
    if event_type == "finalized":
        if attempt.get("terminal_status") is not None:
            raise AttemptLedgerError("attempt is already finalized: %s" % attempt_id)
        if payload.get("status") not in TERMINAL_ATTEMPT_STATUSES:
            raise AttemptLedgerError("attempt terminal status is invalid")
        return
    if event_type == "superseded":
        if attempt.get("terminal_status") is None:
            raise AttemptLedgerError("only a finalized attempt can be superseded")
        if attempt.get("superseded_by_attempt_id") is not None:
            raise AttemptLedgerError("attempt is already superseded")
        replacement = str(payload.get("superseded_by_attempt_id") or "")
        if replacement == attempt_id or replacement not in attempts:
            raise AttemptLedgerError("superseding attempt must be another recorded attempt")
        return
    raise AttemptLedgerError("unknown attempt event type: %s" % event_type)


def _reserve_test_access(
    payload: JsonDict,
    attempts: Mapping[str, JsonDict],
) -> JsonDict:
    invocation = payload.get("invocation")
    if not isinstance(invocation, Mapping) or invocation.get("test_access") is not True:
        return payload
    reserved_invocation = dict(invocation)
    token = _required_sha256(
        reserved_invocation.get("access_token_id"),
        "invocation.access_token_id",
    )
    budget = reserved_invocation.get("access_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise AttemptLedgerError(
            "invocation.access_budget must be a positive integer"
        )
    resume = reserved_invocation.get("resume")
    if isinstance(resume, Mapping) and resume.get("explicit") is True:
        source_result_id = _required_text(
            resume.get("source_result_id"),
            "invocation.resume.source_result_id",
        )
        sources = [
            row
            for row in attempts.values()
            if isinstance(row.get("result"), Mapping)
            and row["result"].get("result_id") == source_result_id
        ]
        if len(sources) != 1:
            raise AttemptLedgerError(
                "sealed-test resume source is absent or duplicated in the attempt ledger"
            )
        source = sources[0]
        source_invocation = source.get("invocation")
        if (
            source.get("terminal_status") != "failed"
            or not isinstance(source_invocation, Mapping)
            or source_invocation.get("test_access") is not True
            or source_invocation.get("access_token_id") != token
            or source_invocation.get("access_budget") != budget
            or source_invocation.get("access_granted") is not True
            or source.get("benchmark_id") != payload.get("benchmark_id")
            or source.get("benchmark_version") != payload.get("benchmark_version")
            or source.get("protocol_sha256") != payload.get("protocol_sha256")
        ):
            raise AttemptLedgerError(
                "sealed-test resume source does not match the granted failed attempt"
            )
        sequence = source_invocation.get("access_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise AttemptLedgerError(
                "sealed-test resume source has an invalid access sequence"
            )
        reserved_invocation.update(
            {
                "access_token_id": token,
                "access_sequence": sequence,
                "access_granted": True,
                "access_continuation": True,
                "continued_attempt_id": source.get("attempt_id"),
            }
        )
        reserved_payload = dict(payload)
        reserved_payload["invocation"] = reserved_invocation
        return reserved_payload
    prior = [
        row
        for row in attempts.values()
        if isinstance(row.get("invocation"), Mapping)
        and row["invocation"].get("test_access") is True
        and row["invocation"].get("access_token_id") == token
    ]
    prior_budgets = {
        row["invocation"].get("access_budget")
        for row in prior
    }
    if prior_budgets and prior_budgets != {budget}:
        raise AttemptLedgerError(
            "invocation.access_budget changed for an existing sealed-test token"
        )
    sequence = len(prior) + 1
    reserved_invocation["access_token_id"] = token
    reserved_invocation["access_sequence"] = sequence
    reserved_invocation["access_granted"] = sequence <= budget
    reserved_payload = dict(payload)
    reserved_payload["invocation"] = reserved_invocation
    return reserved_payload


def _project_attempts(events: List[JsonDict]) -> JsonDict:
    attempts: JsonDict = {}
    for event in events:
        event_type = str(event.get("event_type") or "")
        attempt_id = str(event.get("attempt_id") or "")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise AttemptLedgerError("attempt event payload must be an object")
        _validate_transition(event_type, attempt_id, payload, attempts)
        if event_type == "started":
            attempts[attempt_id] = {
                "attempt_id": attempt_id,
                "started_at_utc": event["recorded_at_utc"],
                "start_event_sha256": event["event_sha256"],
                **dict(payload),
                "terminal_status": None,
            }
        elif event_type == "finalized":
            attempts[attempt_id].update(
                {
                    "terminal_status": payload["status"],
                    "finalized_at_utc": event["recorded_at_utc"],
                    "final_event_sha256": event["event_sha256"],
                    "recipe_outcomes": list(payload.get("recipe_outcomes") or []),
                    "details": dict(payload.get("details") or {}),
                }
            )
            if payload.get("error"):
                attempts[attempt_id]["error"] = str(payload["error"])
            if isinstance(payload.get("result"), Mapping):
                attempts[attempt_id]["result"] = dict(payload["result"])
        else:
            attempts[attempt_id].update(
                {
                    "superseded_by_attempt_id": payload["superseded_by_attempt_id"],
                    "superseded_at_utc": event["recorded_at_utc"],
                    "supersede_event_sha256": event["event_sha256"],
                    "supersede_reason": payload["reason"],
                }
            )
    return attempts


def _snapshot_payload(
    ledger_dir: Path,
    events: List[JsonDict],
    attempts: Mapping[str, JsonDict],
) -> JsonDict:
    rows = [dict(attempts[key]) for key in attempts]
    status_counts: JsonDict = {}
    for row in rows:
        status = row.get("terminal_status") or "started_without_terminal_event"
        if row.get("superseded_by_attempt_id"):
            status = "superseded"
        status_counts[str(status)] = int(status_counts.get(str(status), 0)) + 1
    payload: JsonDict = {
        "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "kind": ATTEMPT_SNAPSHOT_KIND,
        "status": "valid",
        "ledger": str(ledger_dir),
        "threat_model": "internal_hash_chain_without_external_trust_anchor",
        "event_count": len(events),
        "attempt_count": len(rows),
        "head_event_sha256": events[-1]["event_sha256"] if events else None,
        "status_counts": status_counts,
        "events": [dict(event) for event in events],
        "attempts": rows,
        "errors": [],
    }
    return payload


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    require_absent: bool = False,
) -> None:
    canonical_json_sha256(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if require_absent:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError as exc:
                raise AttemptLedgerError(
                    "attempt event path already exists: %s" % path
                ) from exc
            temporary.unlink()
            temporary = None
        else:
            os.replace(str(temporary), str(path))
            temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Python cannot open directory handles with os.open on Windows. The
        # event and head files themselves have already been flushed above.
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttemptLedgerError("%s must be a non-empty string" % label)
    return value.strip()


def _required_sha256(value: Any, label: str) -> str:
    text = _required_text(value, label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AttemptLedgerError("%s must be a lowercase SHA-256" % label)
    return text
