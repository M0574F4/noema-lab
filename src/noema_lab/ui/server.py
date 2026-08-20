from __future__ import annotations

import gzip
import hashlib
import errno
import json
import math
import mimetypes
import io
import os
import shlex
import tempfile
import threading
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmarks import (
    BenchmarkPack,
    finalize_resource_exhausted_benchmark,
    load_benchmark_pack,
    resolve_benchmark_recipe_path,
    validate_benchmark_pack,
    write_benchmark_resource_guard_evidence,
)
from noema_lab.core.benchmark_plots import load_benchmark_plot_records
from noema_lab.core.benchmark_run_evidence import (
    validate_benchmark_run_evidence_snapshot,
)
from noema_lab.core.execution_controls import (
    DEFAULT_EXECUTION_CONTROLS,
    execution_controls_contract,
    normalize_execution_controls,
    require_strict_lint,
)
from noema_lab.core.plan_cache import ExecutionPlanCache
from noema_lab.core.capture import (
    DatasetCaptureError,
    validate_dataset_capture_contract,
)
from noema_lab.core.execution_profiles import execution_profile_catalog, inspect_execution_profile
from noema_lab.core.graph import recipe_graph
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.recipes import compile_recipe, recipe_from_dict, load_recipe
from noema_lab.core.recipe_templates import (
    RecipeTemplateInstantiationError,
    inspect_recipe_template_catalog,
    instantiate_recipe_template,
)
from noema_lab.core.reproducibility import canonical_json_sha256, recipe_fingerprint
from noema_lab.core.reproducibility import finalize_manifest, utc_now_iso
from noema_lab.core.resource_guard import (
    ExecutionDeadlineExceeded,
    IsolatedJobError,
    IsolatedJobSupervisor,
    ResourceExhausted,
)
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import load_research_catalog
from noema_lab.core.runtime_readiness import inspect_recipe_run_readiness
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_json_object,
    decode_strict_yaml_or_json,
    load_strict_yaml_or_json,
)
from noema_lab.core.suites_catalog import load_suites_catalog
from noema_lab.core.training import inspect_training_feasibility
from noema_lab.core.training_plans import apply_training_plan, training_plan_from_dict
from noema_lab.core.trained_artifacts import (
    MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES,
    MAX_IMPORTED_CHECKPOINT_BYTES,
    TrainedArtifactError,
    discover_trained_artifacts,
    import_external_trained_artifact,
    inspect_trained_artifact,
    trained_artifact_recipe_compatibility_issues,
)
from noema_lab.core.verification import verify_benchmark_result, verify_run_bundle
from noema_lab.core.variants import plan_recipe_variants, prepare_single_run_recipe
from noema_lab.ops import build_registry
from noema_lab.ops.source.kodak import KODAK_FILENAMES, download_kodak_dataset
from noema_lab.training.differentiability import TrainingDependencyError
from noema_lab.training.capture_integrity import (
    CaptureSplitIntegrityError,
    assert_disjoint_capture_splits,
)
from noema_lab.training.export_graph import build_export_graph_from_recipe, export_differentiable_graph_bundle
from noema_lab.training.exporter import (
    DifferentiableExportError,
    export_differentiable_scenario,
    inspect_training_capture,
)

JsonDict = Dict[str, Any]

_GZIP_MINIMUM_BYTES = 1024
_STATIC_IMMUTABLE_MAX_AGE_SECONDS = 31536000
_DEFAULT_EXECUTION_OPTIONS = dict(DEFAULT_EXECUTION_CONTROLS)
_DEFAULT_RUN_JOB_WAIT_SECONDS = 30.0
_MAX_RUN_JOB_WAIT_SECONDS = 60.0
_MAX_JSON_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_ACTIVE_JOBS_PER_KIND = 4
_MAX_RETAINED_JOBS_PER_KIND = 128
_MAX_RETAINED_JOB_EVENTS = 512
_MAX_JOB_EVENT_QUERY_LIMIT = 500
_DEFAULT_JOB_EVENT_QUERY_LIMIT = 500
_LEGACY_SYNC_WAIT_SECONDS = 60.0

_TERMINAL_JOB_STATUSES = {
    "completed",
    "failed",
    "canceled",
    "resource_exhausted",
    "timed_out",
    "incomplete",
}


class RequestBodyTooLarge(ValueError):
    pass


class JobCapacityError(RuntimeError):
    pass


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (BrokenPipeError, ConnectionResetError, ConnectionAbortedError),
    ):
        return True
    return isinstance(exc, OSError) and exc.errno in {
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ESHUTDOWN,
    }


def _training_request_recipe(payload: Mapping[str, Any]):
    """Compose a transient export/capture recipe without mutating the authored recipe."""

    recipe_payload = dict(payload.get("recipe") or {})
    recipe = recipe_from_dict(recipe_payload)
    raw_plan = payload.get("training_plan")
    if raw_plan is None:
        return recipe
    if not isinstance(raw_plan, Mapping):
        raise ValueError("training_plan must be a JSON object")
    return apply_training_plan(recipe, training_plan_from_dict(raw_plan))


def _replacement_steps_from_request(
    payload: Mapping[str, Any],
    fallback: Any = None,
):
    selected = (
        payload.get("replacement")
        or payload.get("replacement_steps")
        or payload.get("optimizable")
        or payload.get("optimizable_steps")
    )
    if selected:
        return selected
    raw_plan = payload.get("training_plan")
    if isinstance(raw_plan, Mapping) and raw_plan.get("selected_steps"):
        return raw_plan.get("selected_steps")
    return fallback


def _route_loss_steps_from_request(payload: Mapping[str, Any]):
    """Read recipe loss/evaluation boundaries, never the demo objective."""

    selected = payload.get("route_loss_steps") or payload.get("loss_steps")
    if selected:
        return selected
    raw_plan = payload.get("training_plan")
    if isinstance(raw_plan, Mapping) and raw_plan.get("loss_steps"):
        return raw_plan.get("loss_steps")
    return None


def _execution_options_from_request(payload: Any) -> JsonDict:
    """Validate and normalize the optional execution controls in an API body."""

    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    if "execution" not in payload:
        return normalize_execution_controls()
    raw_execution = payload.get("execution")
    if not isinstance(raw_execution, Mapping):
        raise ValueError("execution must be a JSON object")
    return normalize_execution_controls(raw_execution)


def _require_requested_strict_lint(
    recipe: Any,
    execution: Mapping[str, Any],
    registry: Any,
) -> None:
    if execution.get("strict_lint"):
        require_strict_lint(recipe, registry, context=recipe.name)


def _run_job_wait_seconds(query: str) -> float:
    """Return a bounded timeout for the run-job terminal-state wait API."""

    values = urllib.parse.parse_qs(query).get("timeout_seconds") or []
    if len(values) > 1:
        raise ValueError("timeout_seconds must be provided at most once")
    raw_value = values[0] if values else str(_DEFAULT_RUN_JOB_WAIT_SECONDS)
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be a number") from exc
    if not math.isfinite(timeout) or timeout <= 0 or timeout > _MAX_RUN_JOB_WAIT_SECONDS:
        raise ValueError(
            "timeout_seconds must be greater than zero and at most %g"
            % _MAX_RUN_JOB_WAIT_SECONDS
        )
    return timeout


def _validated_authored_recipe_payload(
    recipe_payload: Any,
    registry,
):
    """Strictly validate and normalize an authored recipe without running it."""

    if not isinstance(recipe_payload, Mapping):
        raise ValueError("recipe must be a JSON object")
    authored_payload = dict(recipe_payload)
    compilation = compile_recipe(
        authored_payload,
        mode="strict",
        registry=registry,
    )
    compilation.require_recipe()
    variant_plan = plan_recipe_variants(authored_payload, registry)
    recipe = variant_plan.recipe
    matrix = variant_plan.canonical_matrix
    if recipe.dataset_capture:
        validate_dataset_capture_contract(recipe, registry)
    normalized_payload = recipe.to_dict()
    normalized_metadata = dict(normalized_payload.get("metadata") or {})
    normalized_metadata.pop("sweeps", None)
    normalized_metadata.pop("ui_sweeps", None)
    if matrix.enabled:
        normalized_metadata["matrix"] = matrix.definition
    else:
        normalized_metadata.pop("matrix", None)
    normalized_payload["metadata"] = normalized_metadata
    return (
        recipe_from_dict(normalized_payload),
        compilation,
        matrix,
    )


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace a text file atomically using a sibling temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    staged_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            staged_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged_path.chmod(existing_mode)
        staged_path.replace(path)
        staged_path = None
    finally:
        if staged_path is not None and staged_path.exists():
            staged_path.unlink()


def _compact_json_bytes(payload: Any) -> bytes:
    """Serialize API payloads without development-only indentation overhead."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _accepts_gzip(value: str) -> bool:
    """Return whether an Accept-Encoding value permits gzip.

    An explicit gzip quality overrides a wildcard, including ``gzip;q=0``.
    """

    gzip_quality: Optional[float] = None
    wildcard_quality: Optional[float] = None
    for raw_item in str(value or "").split(","):
        parts = [part.strip() for part in raw_item.split(";")]
        coding = parts[0].lower()
        if not coding:
            continue
        quality = 1.0
        for parameter in parts[1:]:
            key, separator, raw_quality = parameter.partition("=")
            if separator and key.strip().lower() == "q":
                try:
                    quality = min(1.0, max(0.0, float(raw_quality.strip())))
                except ValueError:
                    quality = 0.0
        if coding == "gzip":
            gzip_quality = quality
        elif coding == "*":
            wildcard_quality = quality
    selected = gzip_quality if gzip_quality is not None else wildcard_quality
    return bool(selected is not None and selected > 0.0)


def _weak_content_etag(data: bytes) -> str:
    return 'W/"sha256-%s"' % hashlib.sha256(data).hexdigest()


def _etag_matches(if_none_match: str, etag: str) -> bool:
    """Apply weak comparison for If-None-Match on a safe GET response."""

    expected = str(etag or "").strip()
    if not expected:
        return False

    def normalize(value: str) -> str:
        value = value.strip()
        return value[2:].strip() if value.startswith("W/") else value

    expected = normalize(expected)
    return any(
        candidate.strip() == "*" or normalize(candidate) == expected
        for candidate in str(if_none_match or "").split(",")
        if candidate.strip()
    )


def _content_addressed_static_version(query: str, data: bytes) -> bool:
    """Recognize only full content hashes as immutable static versions.

    Human-readable ``?v=...`` labels still use ETag revalidation. This prevents a
    forgotten version bump from serving stale JavaScript or CSS indefinitely.
    """

    values = urllib.parse.parse_qs(str(query or ""), keep_blank_values=True)
    digest = hashlib.sha256(data).hexdigest()
    accepted = {digest, "sha256-%s" % digest}
    return any(
        str(candidate).strip().lower() in accepted
        for key in ("v", "sha256")
        for candidate in values.get(key, [])
    )


def _compressible_content_type(content_type: str) -> bool:
    normalized = str(content_type or "").split(";", 1)[0].strip().lower()
    return (
        normalized.startswith("text/")
        or normalized in {
            "application/json",
            "application/javascript",
            "application/xml",
            "image/svg+xml",
        }
    )


def _finalize_interrupted_run(
    store: LocalStore,
    run_id: Optional[str],
    status: str,
    error: str,
    resource_guard: Mapping[str, Any],
    failure_kind: Optional[str] = None,
) -> None:
    if not run_id:
        return
    run_dir = store.runs_dir / str(run_id)
    try:
        summary = store.read_json(run_dir / "summary.json")
    except (OSError, ValueError, TypeError):
        summary = {"schema_version": 1, "run_id": str(run_id), "metrics": {}, "steps": []}
    if str(summary.get("status") or "") not in {"completed", "failed", "canceled"}:
        summary["status"] = status
        summary["error"] = error
        summary["completed_at_utc"] = utc_now_iso()
        summary["failure"] = {"kind": failure_kind or ("resource_exhausted" if status == "failed" else "canceled")}
        summary["resource_guard"] = dict(resource_guard)
        store.write_json(run_dir / "summary.json", summary)
    try:
        manifest = store.read_json(run_dir / "manifest.json")
    except (OSError, ValueError, TypeError):
        manifest = {"schema_version": 1, "run_id": str(run_id)}
    if str(manifest.get("status") or "") not in {"completed", "failed", "canceled"}:
        finalize_manifest(manifest, status, error)
        manifest["failure"] = {"kind": failure_kind or ("resource_exhausted" if status == "failed" else "canceled")}
        manifest["resource_guard"] = dict(resource_guard)
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not summary_path.is_symlink():
            manifest["summary"] = {
                "kind": "noema.run_summary",
                "relative_path": "summary.json",
                "sha256": file_sha256(summary_path),
                "size_bytes": int(summary_path.stat().st_size),
            }
        store.write_json(run_dir / "manifest.json", manifest)


def _finalize_interrupted_benchmark(
    store: LocalStore,
    result_id: Optional[str],
    error: str,
    resource_guard: Mapping[str, Any],
    failure_kind: str = "resource_exhausted",
) -> None:
    finalize_resource_exhausted_benchmark(
        store,
        str(result_id or ""),
        error,
        resource_guard,
        failure_kind,
    )


def _annotate_completed_run_resource_guard(
    store: LocalStore, run_id: str, resource_guard: Mapping[str, Any]
) -> None:
    run_dir = store.runs_dir / run_id
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    payload: JsonDict = {
        "schema_version": 1,
        "kind": "noema.run_resource_guard_evidence",
        "run_id": run_id,
        "summary": {
            "sha256": file_sha256(summary_path),
            "size_bytes": int(summary_path.stat().st_size),
        },
        "manifest": {
            "sha256": file_sha256(manifest_path),
            "size_bytes": int(manifest_path.stat().st_size),
        },
        "resource_guard": dict(resource_guard),
        "recorded_at_utc": utc_now_iso(),
    }
    payload["sha256"] = canonical_json_sha256(payload)
    store.write_json(run_dir / "resource-guard.json", payload)


def _annotate_completed_benchmark_resource_guard(
    store: LocalStore, result_id: str, resource_guard: Mapping[str, Any]
) -> None:
    write_benchmark_resource_guard_evidence(
        store,
        result_id,
        resource_guard,
    )


def _admit_bounded_job(jobs: Dict[str, Any], job: Any) -> None:
    """Insert one job while bounding live work and retained history."""

    terminal_ids = [
        job_id
        for job_id, retained in jobs.items()
        if str(retained.status) in _TERMINAL_JOB_STATUSES
    ]
    while len(jobs) >= _MAX_RETAINED_JOBS_PER_KIND and terminal_ids:
        jobs.pop(terminal_ids.pop(0), None)
    if len(jobs) >= _MAX_RETAINED_JOBS_PER_KIND:
        raise JobCapacityError(
            "Too many retained jobs; wait for an active job to finish before submitting another"
        )
    active = sum(
        1
        for retained in jobs.values()
        if str(retained.status) not in _TERMINAL_JOB_STATUSES
    )
    if active >= _MAX_ACTIVE_JOBS_PER_KIND:
        raise JobCapacityError(
            "At most %d %s jobs may be queued or running"
            % (_MAX_ACTIVE_JOBS_PER_KIND, type(job).__name__.replace("Job", "").lower())
        )
    jobs[job.job_id] = job


def _job_event_cursor(query: str) -> Optional[Tuple[int, int]]:
    params = urllib.parse.parse_qs(query, keep_blank_values=True)
    if "after_seq" not in params and "event_limit" not in params:
        return None
    after_values = params.get("after_seq") or ["0"]
    limit_values = params.get("event_limit") or [
        str(_DEFAULT_JOB_EVENT_QUERY_LIMIT)
    ]
    if len(after_values) != 1 or len(limit_values) != 1:
        raise ValueError("after_seq and event_limit must be provided at most once")
    try:
        after_seq = int(after_values[0])
        event_limit = int(limit_values[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("after_seq and event_limit must be integers") from exc
    if after_seq < 0:
        raise ValueError("after_seq must be non-negative")
    if event_limit < 1 or event_limit > _MAX_JOB_EVENT_QUERY_LIMIT:
        raise ValueError(
            "event_limit must be between 1 and %d" % _MAX_JOB_EVENT_QUERY_LIMIT
        )
    return after_seq, event_limit


class RunJob:
    def __init__(
        self,
        recipe,
        registry,
        store: LocalStore,
        plan_cache: ExecutionPlanCache,
        execution: Mapping[str, Any],
        project_root: Path,
        adapter_paths: Optional[List[str]] = None,
    ) -> None:
        self.job_id = uuid.uuid4().hex
        self.recipe = recipe
        self.registry = registry
        self.store = store
        self.plan_cache = plan_cache
        self.execution = normalize_execution_controls(execution)
        self.project_root = Path(project_root).resolve()
        self.adapter_paths = list(adapter_paths or [])
        self.recipe_name = recipe.name
        self.status = "queued"
        self.run_id: Optional[str] = None
        self.summary_path: Optional[str] = None
        self.error: Optional[str] = None
        self.resource_guard: JsonDict = {}
        self.supervisor: Optional[IsolatedJobSupervisor] = None
        self.events: List[JsonDict] = []
        self._next_event_seq = 1
        self._events_dropped_through = 0
        self.progress: Dict[str, JsonDict] = {}
        self.cancel_event = threading.Event()
        self.terminal_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="noema-run-%s" % self.job_id[:8], daemon=True)

    def start(self) -> None:
        self.add_event("queued", "Queued %s" % self.recipe_name)
        self.thread.start()

    def cancel(self) -> JsonDict:
        self.cancel_event.set()
        supervisor = self.supervisor
        if supervisor is not None:
            supervisor.cancel()
        self.add_event("stop_requested", "Stop requested")
        return self.snapshot()

    def wait(self, timeout: Optional[float] = None) -> JsonDict:
        self.terminal_event.wait(timeout)
        return self.snapshot()

    def snapshot(
        self, event_cursor: Optional[Tuple[int, int]] = None
    ) -> JsonDict:
        with self.lock:
            retained_events = list(self.events)
            if event_cursor is None:
                response_events = retained_events
                next_event_seq = (
                    int(retained_events[-1]["seq"])
                    if retained_events
                    else self._events_dropped_through
                )
                events_truncated = self._events_dropped_through > 0
            else:
                after_seq, event_limit = event_cursor
                matching = [
                    event
                    for event in retained_events
                    if int(event.get("seq") or 0) > after_seq
                ]
                response_events = matching[:event_limit]
                next_event_seq = (
                    int(response_events[-1]["seq"])
                    if response_events
                    else max(after_seq, self._events_dropped_through)
                )
                events_truncated = (
                    after_seq < self._events_dropped_through
                    or len(matching) > len(response_events)
                )
            return {
                "job_id": self.job_id,
                "recipe_name": self.recipe_name,
                "status": self.status,
                "run_id": self.run_id,
                "summary_path": self.summary_path,
                "error": self.error,
                "execution": dict(self.execution),
                "resource_guard": dict(self.resource_guard),
                "events": response_events,
                "next_event_seq": next_event_seq,
                "events_truncated": events_truncated,
                "progress": {"steps": dict(self.progress)},
            }

    def add_event(self, kind: str, message: str, level: str = "info", **fields: Any) -> None:
        with self.lock:
            event = {
                **fields,
                "seq": self._next_event_seq,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "kind": kind,
                "level": level,
                "message": message,
            }
            self._next_event_seq += 1
            self.events.append(event)
            if len(self.events) > _MAX_RETAINED_JOB_EVENTS:
                dropped = self.events.pop(0)
                self._events_dropped_through = int(dropped["seq"])

    def _executor_event(self, event: JsonDict) -> None:
        kind = str(event.get("kind") or "event")
        run_id = event.get("run_id")
        if run_id:
            with self.lock:
                self.run_id = str(run_id)
        level = "error" if kind in ("run_failed", "run_canceled") else "info"
        fields = {key: value for key, value in event.items() if key not in ("kind", "message")}
        self._update_progress(kind, event)
        self.add_event(kind, str(event.get("message") or kind), level=level, **fields)

    def _update_progress(self, kind: str, event: JsonDict) -> None:
        step_id = event.get("step_id")
        if not step_id:
            if kind in ("run_failed", "run_canceled"):
                with self.lock:
                    for item in self.progress.values():
                        if item.get("status") == "running":
                            item["status"] = "failed" if kind == "run_failed" else "canceled"
                            item["percent"] = item.get("percent", 50)
                            item["message"] = str(event.get("message") or kind)
            return
        step_key = str(step_id)
        with self.lock:
            current = dict(self.progress.get(step_key) or {})
            if kind == "step_started":
                current.update(
                    {
                        "step_id": step_key,
                        "op": str(event.get("op") or ""),
                        "status": "running",
                        "phase": "running",
                        "percent": 0,
                        "completed": 0,
                        "total": 1,
                        "unit": "step",
                        "message": str(event.get("message") or "running"),
                    }
                )
            elif kind == "step_completed":
                current.update(
                    {
                        "step_id": step_key,
                        "op": str(event.get("op") or current.get("op") or ""),
                        "status": "completed",
                        "phase": "completed",
                        "percent": 100,
                        "completed": 1,
                        "total": 1,
                        "unit": "step",
                        "message": str(event.get("message") or "completed"),
                    }
                )
            elif kind == "step_progress":
                current.update({key: value for key, value in event.items() if key not in ("kind", "message")})
                current.setdefault("step_id", step_key)
                current.setdefault("status", "running")
                current.setdefault("phase", "running")
                current.setdefault("message", str(event.get("message") or "progress"))
            self.progress[step_key] = current

    def has_event(self, kind: str, message: str) -> bool:
        with self.lock:
            return any(event.get("kind") == kind and event.get("message") == message for event in self.events)

    def _fail_active_progress(self, message: str) -> None:
        with self.lock:
            for item in self.progress.values():
                if item.get("status") == "running":
                    item["status"] = "failed"
                    item["phase"] = "resource_exhausted"
                    item["message"] = message

    def _run(self) -> None:
        with self.lock:
            self.status = "running"
        self.add_event("job_started", "Started %s" % self.recipe_name)
        try:
            supervisor = IsolatedJobSupervisor(
                job_id=self.job_id,
                request={
                    "kind": "recipe",
                    "workspace": str(self.store.workspace),
                    "project_root": str(self.project_root),
                    "adapter_paths": list(self.adapter_paths),
                    "recipe": self.recipe.to_dict(),
                    "execution": dict(self.execution),
                },
                workspace=self.store.workspace,
                project_root=self.project_root,
                event_sink=self._executor_event,
            )
            self.supervisor = supervisor
            if self.cancel_event.is_set():
                supervisor.cancel()
            outcome = supervisor.run()
            self.resource_guard = dict(outcome.evidence)
            if outcome.status == "canceled":
                if not self.run_id and outcome.payload.get("run_id"):
                    self.run_id = str(outcome.payload["run_id"])
                _finalize_interrupted_run(self.store, self.run_id, "canceled", "Run canceled", outcome.evidence)
                with self.lock:
                    self.status = "canceled"
                    self.error = "Run canceled"
                self.add_event("job_canceled", "Run canceled", level="warning")
                return
            run_id = str(outcome.payload.get("run_id") or self.run_id or "")
            run_dir = self.store.runs_dir / run_id
            _annotate_completed_run_resource_guard(self.store, run_id, outcome.evidence)
            with self.lock:
                self.run_id = run_id
                self.summary_path = str(run_dir / "summary.json")
                self.status = "completed"
            self.add_event("job_completed", "Completed %s" % self.recipe_name, run_id=run_id)
        except ResourceExhausted as exc:
            self.resource_guard = dict(exc.evidence)
            if not self.run_id and exc.payload.get("run_id"):
                self.run_id = str(exc.payload["run_id"])
            _finalize_interrupted_run(self.store, self.run_id, "failed", str(exc), exc.evidence)
            with self.lock:
                self.status = "resource_exhausted"
                self.error = str(exc)
            self._fail_active_progress(str(exc))
            self.add_event("job_resource_exhausted", str(exc), level="error")
        except ExecutionDeadlineExceeded as exc:
            self.resource_guard = dict(exc.evidence)
            if not self.run_id and exc.payload.get("run_id"):
                self.run_id = str(exc.payload["run_id"])
            _finalize_interrupted_run(
                self.store,
                self.run_id,
                "failed",
                str(exc),
                exc.evidence,
                "execution_deadline",
            )
            with self.lock:
                self.status = "timed_out"
                self.error = str(exc)
            self._fail_active_progress(str(exc))
            self.add_event("job_timed_out", str(exc), level="error")
        except (IsolatedJobError, Exception) as exc:
            _finalize_interrupted_run(
                self.store,
                self.run_id,
                "failed",
                str(exc),
                self.resource_guard,
                "worker_crash",
            )
            with self.lock:
                self.status = "failed"
                self.error = str(exc)
            if not self.has_event("run_failed", str(exc)):
                self.add_event("job_failed", str(exc), level="error")
        finally:
            with self.lock:
                if self.status not in {
                    "completed",
                    "failed",
                    "canceled",
                    "resource_exhausted",
                    "timed_out",
                }:
                    self.status = "failed"
                    self.error = self.error or (
                        "Run job exited without recording a terminal status"
                    )
            self.terminal_event.set()


class RunJobManager:
    def __init__(
        self,
        registry,
        store: LocalStore,
        plan_cache: ExecutionPlanCache,
        project_root: Path,
        adapter_paths: Optional[List[str]] = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.plan_cache = plan_cache
        self.project_root = Path(project_root).resolve()
        self.adapter_paths = list(adapter_paths or [])
        self.jobs: Dict[str, RunJob] = {}
        self.lock = threading.Lock()

    def start(self, recipe, execution: Mapping[str, Any]) -> JsonDict:
        job = RunJob(
            recipe,
            self.registry,
            self.store,
            self.plan_cache,
            execution,
            self.project_root,
            self.adapter_paths,
        )
        with self.lock:
            _admit_bounded_job(self.jobs, job)
        job.start()
        return job.snapshot()

    def get(
        self,
        job_id: str,
        event_cursor: Optional[Tuple[int, int]] = None,
    ) -> JsonDict:
        job = self._job(job_id)
        return job.snapshot(event_cursor)

    def wait(self, job_id: str, timeout: float) -> JsonDict:
        job = self._job(job_id)
        return job.wait(timeout)

    def cancel(self, job_id: str) -> JsonDict:
        job = self._job(job_id)
        return job.cancel()

    def _job(self, job_id: str) -> RunJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if not job:
            raise KeyError("unknown run job: %s" % job_id)
        return job


class BenchmarkJob:
    def __init__(
        self,
        pack: BenchmarkPack,
        registry,
        store: LocalStore,
        project_root: Path,
        execution: Mapping[str, Any],
        adapter_paths: Optional[List[str]] = None,
    ) -> None:
        self.job_id = uuid.uuid4().hex
        self.pack = pack
        self.registry = registry
        self.store = store
        self.project_root = project_root
        self.execution = normalize_execution_controls(execution)
        self.adapter_paths = list(adapter_paths or [])
        self.status = "queued"
        self.result_id: Optional[str] = None
        self.error: Optional[str] = None
        self.resource_guard: JsonDict = {}
        self.supervisor: Optional[IsolatedJobSupervisor] = None
        self.cancel_event = threading.Event()
        self.terminal_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run,
            name="noema-benchmark-%s" % self.job_id[:8],
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> JsonDict:
        self.cancel_event.set()
        supervisor = self.supervisor
        if supervisor is not None:
            supervisor.cancel()
        return self.snapshot()

    def wait(self, timeout: Optional[float] = None) -> JsonDict:
        self.terminal_event.wait(timeout)
        return self.snapshot()

    def snapshot(self) -> JsonDict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "benchmark_id": self.pack.id,
                "benchmark_name": self.pack.name or self.pack.id,
                "status": self.status,
                "result_id": self.result_id,
                "error": self.error,
                "execution": dict(self.execution),
                "resource_guard": dict(self.resource_guard),
            }

    def _worker_event(self, event: JsonDict) -> None:
        if event.get("kind") == "benchmark_created" and event.get("result_id"):
            with self.lock:
                self.result_id = str(event["result_id"])

    def _run(self) -> None:
        with self.lock:
            self.status = "running"
        try:
            if self.pack.path is None:
                raise IsolatedJobError("Isolated benchmark execution requires a benchmark-pack path")
            supervisor = IsolatedJobSupervisor(
                job_id=self.job_id,
                request={
                    "kind": "benchmark",
                    "workspace": str(self.store.workspace),
                    "project_root": str(self.project_root),
                    "adapter_paths": list(self.adapter_paths),
                    "pack_path": str(Path(self.pack.path).resolve()),
                    "execution": dict(self.execution),
                },
                workspace=self.store.workspace,
                project_root=self.project_root,
                event_sink=self._worker_event,
            )
            self.supervisor = supervisor
            if self.cancel_event.is_set():
                supervisor.cancel()
            outcome = supervisor.run()
            self.resource_guard = dict(outcome.evidence)
            if outcome.status == "canceled":
                if not self.result_id and outcome.payload.get("result_id"):
                    self.result_id = str(outcome.payload["result_id"])
                _finalize_interrupted_benchmark(
                    self.store,
                    self.result_id,
                    "Benchmark canceled",
                    outcome.evidence,
                    "user_canceled",
                )
                with self.lock:
                    self.status = "canceled"
                    self.error = "Benchmark canceled"
                return
            result_id = str(outcome.payload.get("result_id") or self.result_id or "")
            _annotate_completed_benchmark_resource_guard(self.store, result_id, outcome.evidence)
            result_status = str(
                outcome.payload.get("result_status") or "completed"
            ).lower()
            with self.lock:
                self.result_id = result_id
                self.status = (
                    "completed"
                    if result_status == "completed"
                    else "incomplete"
                )
                if self.status == "incomplete":
                    self.error = (
                        "Benchmark contains skipped or otherwise incomplete methods"
                    )
        except ResourceExhausted as exc:
            self.resource_guard = dict(exc.evidence)
            if not self.result_id and exc.payload.get("result_id"):
                self.result_id = str(exc.payload["result_id"])
            _finalize_interrupted_benchmark(self.store, self.result_id, str(exc), exc.evidence)
            with self.lock:
                self.error = str(exc)
                self.status = "resource_exhausted"
        except ExecutionDeadlineExceeded as exc:
            self.resource_guard = dict(exc.evidence)
            if not self.result_id and exc.payload.get("result_id"):
                self.result_id = str(exc.payload["result_id"])
            _finalize_interrupted_benchmark(
                self.store,
                self.result_id,
                str(exc),
                exc.evidence,
                "execution_deadline",
            )
            with self.lock:
                self.error = str(exc)
                self.status = "timed_out"
        except (IsolatedJobError, Exception) as exc:
            _finalize_interrupted_benchmark(
                self.store,
                self.result_id,
                str(exc),
                self.resource_guard,
                "worker_crash",
            )
            with self.lock:
                self.error = str(exc)
                self.status = "failed"
        finally:
            with self.lock:
                if self.status not in _TERMINAL_JOB_STATUSES:
                    self.status = "failed"
                    self.error = self.error or (
                        "Benchmark job exited without recording a terminal status"
                    )
            self.terminal_event.set()


class BenchmarkJobManager:
    def __init__(self, registry, store: LocalStore, project_root: Path, adapter_paths: Optional[List[str]] = None) -> None:
        self.registry = registry
        self.store = store
        self.project_root = project_root
        self.adapter_paths = list(adapter_paths or [])
        self.jobs: Dict[str, BenchmarkJob] = {}
        self.lock = threading.Lock()

    def start(
        self,
        pack: BenchmarkPack,
        execution: Mapping[str, Any],
    ) -> JsonDict:
        job = BenchmarkJob(
            pack,
            self.registry,
            self.store,
            self.project_root,
            execution,
            self.adapter_paths,
        )
        with self.lock:
            _admit_bounded_job(self.jobs, job)
        job.start()
        return job.snapshot()

    def get(self, job_id: str) -> JsonDict:
        return self._job(job_id).snapshot()

    def wait(self, job_id: str, timeout: Optional[float]) -> JsonDict:
        return self._job(job_id).wait(timeout)

    def cancel(self, job_id: str) -> JsonDict:
        return self._job(job_id).cancel()

    def _job(self, job_id: str) -> BenchmarkJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if not job:
            raise KeyError("unknown benchmark job: %s" % job_id)
        return job


class DatasetCaptureJob:
    def __init__(
        self,
        recipe,
        registry,
        store: LocalStore,
        out_dir: Path,
        project_root: Path,
        adapter_paths: Optional[List[str]] = None,
        *,
        force: bool = False,
    ) -> None:
        self.job_id = uuid.uuid4().hex
        self.recipe = recipe
        self.registry = registry
        self.store = store
        self.out_dir = Path(out_dir)
        self.project_root = Path(project_root).resolve()
        self.adapter_paths = list(adapter_paths or [])
        self.force = bool(force)
        self.recipe_name = recipe.name
        self.split = str((recipe.dataset_capture or {}).get("split") or "train")
        self.status = "queued"
        self.dataset_capture: Optional[JsonDict] = None
        self.error: Optional[str] = None
        self.resource_guard: JsonDict = {}
        self.supervisor: Optional[IsolatedJobSupervisor] = None
        self.cancel_event = threading.Event()
        self.terminal_event = threading.Event()
        self.events: List[JsonDict] = []
        self._next_event_seq = 1
        self.progress: JsonDict = {
            "percent": 0.0,
            "phase": "queued",
            "message": "Dataset capture queued",
            "completed_samples": 0,
            "total_samples": (recipe.dataset_capture or {}).get("samples"),
            "unit": "samples",
        }
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run,
            name="noema-dataset-capture-%s" % self.job_id[:8],
            daemon=True,
        )

    def start(self) -> None:
        self._add_event(
            {
                "kind": "job_queued",
                "message": "Queued dataset capture for %s" % self.recipe_name,
            }
        )
        self.thread.start()

    def cancel(self) -> JsonDict:
        self.cancel_event.set()
        supervisor = self.supervisor
        if supervisor is not None:
            supervisor.cancel()
        self._add_event(
            {"kind": "stop_requested", "message": "Stop requested"}
        )
        return self.snapshot()

    def wait(self, timeout: Optional[float] = None) -> JsonDict:
        self.terminal_event.wait(timeout)
        return self.snapshot()

    def snapshot(self) -> JsonDict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "recipe_name": self.recipe_name,
                "split": self.split,
                "out_dir": str(self.out_dir),
                "status": self.status,
                "dataset_capture": dict(self.dataset_capture) if self.dataset_capture is not None else None,
                "error": self.error,
                "resource_guard": dict(self.resource_guard),
                "progress": dict(self.progress),
                "events": list(self.events),
            }

    def _add_event(self, event: JsonDict) -> None:
        item = dict(event)
        kind = str(item.get("kind") or "event")
        item.setdefault("message", kind)
        item.setdefault("level", "error" if kind in {"run_failed", "job_failed"} else "info")
        with self.lock:
            item["seq"] = self._next_event_seq
            self._next_event_seq += 1
            item["time"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self.events.append(item)
            if len(self.events) > _MAX_RETAINED_JOB_EVENTS:
                self.events.pop(0)

    def _update_progress(self, progress: JsonDict) -> None:
        item = dict(progress)
        with self.lock:
            previous_percent = float(self.progress.get("percent") or 0.0)
            item["percent"] = max(previous_percent, min(100.0, float(item.get("percent") or 0.0)))
            self.progress = item

    def _worker_event(self, event: JsonDict) -> None:
        item = dict(event)
        if str(item.get("kind") or "") == "capture_progress":
            item.pop("kind", None)
            self._update_progress(item)
            return
        self._add_event(item)

    def _run(self) -> None:
        with self.lock:
            self.status = "running"
            self.progress.update({"phase": "starting", "message": "Starting dataset capture"})
        self._add_event(
            {
                "kind": "job_started",
                "message": "Started dataset capture for %s" % self.recipe_name,
            }
        )
        try:
            supervisor = IsolatedJobSupervisor(
                job_id=self.job_id,
                request={
                    "kind": "dataset_capture",
                    "workspace": str(self.store.workspace),
                    "project_root": str(self.project_root),
                    "adapter_paths": list(self.adapter_paths),
                    "recipe": self.recipe.to_dict(),
                    "out_dir": str(self.out_dir),
                    "force": self.force,
                },
                workspace=self.store.workspace,
                project_root=self.project_root,
                event_sink=self._worker_event,
            )
            self.supervisor = supervisor
            if self.cancel_event.is_set():
                supervisor.cancel()
            outcome = supervisor.run()
            self.resource_guard = dict(outcome.evidence)
            if outcome.status == "canceled":
                _finalize_interrupted_run(
                    self.store,
                    (
                        str(outcome.payload["run_id"])
                        if outcome.payload.get("run_id")
                        else None
                    ),
                    "canceled",
                    "Dataset capture canceled",
                    outcome.evidence,
                    "user_canceled",
                )
                with self.lock:
                    self.status = "canceled"
                    self.error = "Dataset capture canceled"
                    self.progress.update(
                        {"phase": "canceled", "message": self.error}
                    )
                self._add_event(
                    {
                        "kind": "job_canceled",
                        "message": self.error,
                        "level": "warning",
                    }
                )
                return
            result = dict(outcome.payload.get("dataset_capture") or {})
            if not result:
                raise IsolatedJobError(
                    "Dataset capture worker completed without a result"
                )
            with self.lock:
                self.dataset_capture = result
                self.status = "completed"
                self.progress.update(
                    {
                        "percent": 100.0,
                        "phase": "completed",
                        "message": "Dataset capture completed",
                    }
                )
            self._add_event(
                {
                    "kind": "job_completed",
                    "message": "Completed dataset capture for %s" % self.recipe_name,
                }
            )
        except ResourceExhausted as exc:
            self.resource_guard = dict(exc.evidence)
            _finalize_interrupted_run(
                self.store,
                str(exc.payload["run_id"])
                if exc.payload.get("run_id")
                else None,
                "failed",
                str(exc),
                exc.evidence,
                "resource_exhausted",
            )
            with self.lock:
                self.status = "resource_exhausted"
                self.error = str(exc)
                self.progress.update(
                    {"phase": "resource_exhausted", "message": str(exc)}
                )
            self._add_event(
                {"kind": "job_resource_exhausted", "message": str(exc), "level": "error"}
            )
        except ExecutionDeadlineExceeded as exc:
            self.resource_guard = dict(exc.evidence)
            _finalize_interrupted_run(
                self.store,
                str(exc.payload["run_id"])
                if exc.payload.get("run_id")
                else None,
                "failed",
                str(exc),
                exc.evidence,
                "execution_deadline",
            )
            with self.lock:
                self.status = "timed_out"
                self.error = str(exc)
                self.progress.update(
                    {"phase": "timed_out", "message": str(exc)}
                )
            self._add_event(
                {"kind": "job_timed_out", "message": str(exc), "level": "error"}
            )
        except Exception as exc:
            with self.lock:
                self.status = "failed"
                self.error = str(exc)
                self.progress.update({"phase": "failed", "message": str(exc)})
            self._add_event({"kind": "job_failed", "message": str(exc), "level": "error"})
        finally:
            with self.lock:
                if self.status not in _TERMINAL_JOB_STATUSES:
                    self.status = "failed"
                    self.error = self.error or (
                        "Dataset capture job exited without recording a terminal status"
                    )
            self.terminal_event.set()


class DatasetCaptureJobManager:
    def __init__(
        self,
        registry,
        store: LocalStore,
        project_root: Path,
        adapter_paths: Optional[List[str]] = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.adapter_paths = list(adapter_paths or [])
        self.jobs: Dict[str, DatasetCaptureJob] = {}
        self.lock = threading.Lock()

    def start(self, recipe, out_dir: Path, *, force: bool = False) -> JsonDict:
        job = DatasetCaptureJob(
            recipe,
            self.registry,
            self.store,
            out_dir,
            self.project_root,
            self.adapter_paths,
            force=force,
        )
        with self.lock:
            _admit_bounded_job(self.jobs, job)
        job.start()
        return job.snapshot()

    def get(self, job_id: str) -> JsonDict:
        return self._job(job_id).snapshot()

    def wait(self, job_id: str, timeout: Optional[float]) -> JsonDict:
        return self._job(job_id).wait(timeout)

    def cancel(self, job_id: str) -> JsonDict:
        return self._job(job_id).cancel()

    def _job(self, job_id: str) -> DatasetCaptureJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if not job:
            raise KeyError("unknown dataset capture job: %s" % job_id)
        return job


def serve_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: Optional[Path] = None,
    project_root: Optional[Path] = None,
    adapter_paths: Optional[List[str]] = None,
) -> None:
    workspace_path = Path(workspace or ".noema").resolve()
    project_root_path = Path(project_root or ".").resolve()
    handler = _make_handler(workspace_path, project_root_path, adapter_paths=adapter_paths)
    server = ThreadingHTTPServer((host, int(port)), handler)
    print("noema ui: http://%s:%d" % (host, int(port)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _validated_recipe_matrix_expansion(
    recipe_payload: Any,
    registry: Any,
) -> JsonDict:
    """Return the shared strict, registry-validated matrix expansion."""

    if not isinstance(recipe_payload, Mapping):
        raise ValueError("recipe matrix expansion requires a recipe object")
    return plan_recipe_variants(recipe_payload, registry).to_expansion_dict()


def _make_handler(workspace: Path, project_root: Path, adapter_paths: Optional[List[str]] = None):
    registry = build_registry(adapter_paths)
    store = LocalStore(workspace)
    execution_plan_cache = ExecutionPlanCache()
    job_manager = RunJobManager(
        registry,
        store,
        execution_plan_cache,
        project_root,
        adapter_paths,
    )
    benchmark_job_manager = BenchmarkJobManager(
        registry,
        store,
        project_root,
        adapter_paths,
    )
    dataset_capture_job_manager = DatasetCaptureJobManager(
        registry,
        store,
        project_root,
        adapter_paths,
    )
    static_root = Path(__file__).resolve().parent / "static"
    dataset_lock = threading.Lock()
    response_cache_lock = threading.Lock()
    compressed_response_cache: Dict[str, bytes] = {}

    # These descriptions are authoritative for the lifetime of this handler.
    # Recipe-template inspection is intentionally excluded: the catalog is
    # process-cached package data, but its project recipe files are mutable.
    stable_json_responses = {
        "/api/ops": _compact_json_bytes({"operations": registry.describe()}),
        "/api/research/catalog": _compact_json_bytes(load_research_catalog().to_dict()),
        "/api/execution-profiles": _compact_json_bytes(execution_profile_catalog().to_dict()),
        "/api/suites": _compact_json_bytes(load_suites_catalog().to_dict()),
    }

    def cached_gzip(data: bytes) -> bytes:
        cache_key = hashlib.sha256(data).hexdigest()
        with response_cache_lock:
            cached = compressed_response_cache.get(cache_key)
        if cached is not None:
            return cached
        compressed = gzip.compress(data, compresslevel=5, mtime=0)
        with response_cache_lock:
            # The cache key is the complete content digest, so concurrent
            # computation is harmless and cannot return a stale representation.
            compressed_response_cache[cache_key] = compressed
        return compressed

    class NoemaUiHandler(BaseHTTPRequestHandler):
        server_version = "NoemaUi/0.1"

        def log_message(self, format, *args):  # type: ignore[override]
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/":
                    return self._serve_file(static_root / "index.html")
                if parsed.path.startswith("/static/"):
                    relative = parsed.path[len("/static/") :]
                    return self._serve_file(
                        _safe_child(static_root, relative),
                        query=parsed.query,
                        revalidate=True,
                    )
                if parsed.path == "/api/health":
                    return self._json(
                        {
                            "status": "ok",
                            "workspace": str(workspace),
                            "execution_controls": execution_controls_contract(),
                        }
                    )
                if parsed.path == "/api/ops":
                    return self._stable_json(stable_json_responses[parsed.path])
                if parsed.path == "/api/research/catalog":
                    return self._stable_json(stable_json_responses[parsed.path])
                if parsed.path == "/api/execution-profiles":
                    return self._stable_json(stable_json_responses[parsed.path])
                if parsed.path == "/api/suites":
                    return self._stable_json(stable_json_responses[parsed.path])
                if parsed.path == "/api/recipe-templates":
                    return self._json(
                        inspect_recipe_template_catalog(project_root, registry).to_dict()
                    )
                if parsed.path == "/api/datasets/images":
                    return self._json(_dataset_images_payload(parsed.query, project_root, dataset_lock))
                if parsed.path == "/api/recipes":
                    return self._json({"recipes": _list_recipes(project_root, registry)})
                if parsed.path == "/api/workbench/exported-project":
                    project_path = _query_path(parsed.query, "path", project_root)
                    return self._json(_exported_project_payload(project_path, project_root, registry=registry))
                if parsed.path == "/api/workbench/exported-project-watch":
                    project_path = _query_path(parsed.query, "path", project_root)
                    return self._json(_exported_project_watch_payload(project_path, project_root))
                if parsed.path == "/api/trained-artifacts":
                    params = urllib.parse.parse_qs(parsed.query)
                    operation = str((params.get("operation") or [""])[0])
                    artifacts = discover_trained_artifacts(
                        project_root,
                        registry=registry,
                        operation=operation,
                    )
                    return self._json(
                        {
                            "status": "ok",
                            "operation": operation,
                            "artifacts": artifacts,
                        }
                    )
                if parsed.path == "/api/recipe":
                    try:
                        recipe_path = _query_path(parsed.query, "path", project_root)
                        recipe = load_recipe(
                            recipe_path,
                            mode="strict",
                            registry=registry,
                            effective=False,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe is invalid: %s" % exc)
                    return self._json({"recipe": recipe.to_dict()})
                if parsed.path == "/api/recipe/graph":
                    try:
                        recipe_path = _query_path(parsed.query, "path", project_root)
                        recipe = load_recipe(
                            recipe_path,
                            mode="strict",
                            registry=registry,
                            effective=False,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe is invalid: %s" % exc)
                    return self._json(recipe_graph(recipe, registry))
                if parsed.path == "/api/recipe/matrix":
                    try:
                        recipe_path = _query_path(parsed.query, "path", project_root)
                        recipe_payload = load_strict_yaml_or_json(recipe_path)
                        expansion = _validated_recipe_matrix_expansion(
                            recipe_payload,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe matrix is invalid: %s" % exc)
                    return self._json(expansion)
                if parsed.path == "/api/runs":
                    return self._json({"runs": store.list_runs(read_only=True)})
                if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/verify"):
                    run_id = _run_id_from_visual_path(parsed.path, "verify")
                    return self._json(verify_run_bundle(store, run_id, registry=registry))
                if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/segmentation-overlay"):
                    run_id = _run_id_from_visual_path(parsed.path, "segmentation-overlay")
                    params = urllib.parse.parse_qs(parsed.query)
                    kind = str((params.get("kind") or ["prediction"])[0])
                    index = int((params.get("index") or ["0"])[0])
                    return self._bytes(_segmentation_overlay_png(store, project_root, workspace, run_id, kind, index), "image/png")
                if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/detection-overlay"):
                    run_id = _run_id_from_visual_path(parsed.path, "detection-overlay")
                    params = urllib.parse.parse_qs(parsed.query)
                    kind = str((params.get("kind") or ["prediction"])[0])
                    index = int((params.get("index") or ["0"])[0])
                    return self._bytes(_detection_overlay_png(store, project_root, workspace, run_id, kind, index), "image/png")
                if parsed.path.startswith("/api/runs/"):
                    run_id = urllib.parse.unquote(parsed.path[len("/api/runs/") :])
                    return self._json(_run_payload(store, run_id))
                if parsed.path == "/api/benchmarks/results":
                    return self._json(
                        {"results": store.list_benchmark_results(read_only=True)}
                    )
                if parsed.path.startswith("/api/benchmarks/results/") and parsed.path.endswith("/verify"):
                    result_id = _benchmark_result_id_from_path(parsed.path, "verify")
                    return self._json(verify_benchmark_result(store, result_id, registry=registry))
                if (
                    parsed.path.startswith("/api/benchmarks/results/")
                    and "/runs/" in parsed.path
                ):
                    result_id, run_id = _benchmark_result_run_ids_from_path(parsed.path)
                    return self._json(
                        _benchmark_result_run_payload(store, result_id, run_id)
                    )
                if parsed.path.startswith("/api/benchmarks/results/"):
                    result_id = urllib.parse.unquote(parsed.path[len("/api/benchmarks/results/") :])
                    return self._json(_benchmark_result_payload(store, result_id))
                if parsed.path.startswith("/api/run-jobs/") and parsed.path.endswith("/wait"):
                    job_id = urllib.parse.unquote(
                        parsed.path[len("/api/run-jobs/") : -len("/wait")]
                    )
                    try:
                        timeout = _run_job_wait_seconds(parsed.query)
                    except ValueError as exc:
                        return self._error(400, str(exc))
                    return self._json(job_manager.wait(job_id, timeout))
                if parsed.path.startswith("/api/run-jobs/"):
                    job_id = urllib.parse.unquote(parsed.path[len("/api/run-jobs/") :])
                    try:
                        event_cursor = _job_event_cursor(parsed.query)
                    except ValueError as exc:
                        return self._error(400, str(exc))
                    return self._json(job_manager.get(job_id, event_cursor))
                if parsed.path.startswith("/api/benchmark-jobs/"):
                    job_id = urllib.parse.unquote(parsed.path[len("/api/benchmark-jobs/") :])
                    return self._json(benchmark_job_manager.get(job_id))
                if parsed.path.startswith("/api/dataset-capture-jobs/"):
                    job_id = urllib.parse.unquote(parsed.path[len("/api/dataset-capture-jobs/") :])
                    return self._json(dataset_capture_job_manager.get(job_id))
                if parsed.path == "/api/artifact":
                    artifact_path = _query_managed_artifact_path(
                        parsed.query,
                        "path",
                        project_root,
                        workspace,
                        store,
                    )
                    return self._json(_artifact_preview(artifact_path))
                if parsed.path == "/api/image":
                    artifact_path = _query_managed_artifact_path(
                        parsed.query,
                        "path",
                        project_root,
                        workspace,
                        store,
                    )
                    params = urllib.parse.parse_qs(parsed.query)
                    index = int((params.get("index") or ["0"])[0])
                    array_name = str((params.get("array") or ["images"])[0])
                    return self._bytes(_image_bmp(artifact_path, array_name, index), "image/bmp")
                return self._error(404, "not found")
            except (StructuredInputError, UnicodeDecodeError) as exc:
                return self._error(400, "Invalid structured input: %s" % exc)
            except ValueError as exc:
                return self._error(400, str(exc))
            except KeyError as exc:
                return self._error(404, str(exc).strip("'"))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    self.close_connection = True
                    return None
                return self._exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/api/recipe-templates/instantiate":
                    payload = self._read_json()
                    if not isinstance(payload, Mapping):
                        return self._error(
                            400,
                            "Recipe template instantiation requires a JSON object",
                        )
                    try:
                        result = instantiate_recipe_template(
                            payload.get("template_id"),
                            project_root,
                            registry,
                            overrides=payload.get("overrides"),
                        )
                    except (RecipeTemplateInstantiationError, ValueError) as exc:
                        return self._error(
                            400,
                            "Recipe template cannot be instantiated: %s" % exc,
                        )
                    return self._json(
                        {"status": "instantiated", **result.to_dict()},
                        status=201,
                    )
                if parsed.path == "/api/recipe/matrix/expand":
                    try:
                        payload = self._read_json()
                        recipe_payload = (
                            payload.get("recipe")
                            if isinstance(payload, Mapping)
                            else None
                        )
                        expansion = _validated_recipe_matrix_expansion(
                            recipe_payload,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe matrix is invalid: %s" % exc)
                    return self._json(expansion)
                if parsed.path == "/api/trained-artifacts/import":
                    content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                    if content_type != "application/octet-stream":
                        return self._error(415, "trained artifact import requires application/octet-stream")
                    try:
                        content_length = int(self.headers.get("Content-Length") or 0)
                    except (TypeError, ValueError):
                        return self._error(400, "trained artifact import requires a valid Content-Length")
                    if content_length <= 0:
                        return self._error(400, "trained artifact upload is empty")
                    if content_length > MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES:
                        return self._error(
                            413,
                            "trained artifact upload exceeds the %d-byte limit"
                            % MAX_IMPORTED_ARTIFACT_PACKAGE_BYTES,
                        )
                    params = urllib.parse.parse_qs(parsed.query)
                    operation = str((params.get("operation") or [""])[0]).strip()
                    label = str((params.get("label") or [""])[0]).strip()
                    filename = str(
                        (params.get("filename") or [self.headers.get("X-Noema-Filename") or ""])[0]
                    ).strip()
                    if not filename:
                        return self._error(400, "trained artifact import requires a filename")
                    safe_filename = Path(filename).name
                    suffix = Path(safe_filename).suffix.lower()
                    paired_deepjscc_operation = operation in {
                        "model.deepjscc_external_encode",
                        "model.deepjscc_external_decode",
                    }
                    if paired_deepjscc_operation and suffix == ".onnx":
                        return self._error(
                            400,
                            "a DeepJSCC encoder or decoder ONNX file is not a complete artifact; "
                            "upload one .zip/.noema-artifact package containing trained_artifact.yaml, "
                            "the contract, and both encoder and decoder components",
                        )
                    if paired_deepjscc_operation and suffix in {".yaml", ".yml", ".json"}:
                        return self._error(
                            400,
                            "a browser-uploaded DeepJSCC manifest cannot include its paired component files; "
                            "use the automatically discovered local artifact or upload the complete "
                            ".zip/.noema-artifact package",
                        )
                    allowed_suffixes = {
                        ".npz",
                        ".zip",
                        ".noema-artifact",
                        ".yaml",
                        ".yml",
                        ".json",
                    }
                    if suffix not in allowed_suffixes:
                        return self._error(
                            400,
                            "choose a safe .npz checkpoint or a .zip/.noema-artifact portable package",
                        )
                    staged_path: Optional[Path] = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            prefix="noema-artifact-upload-",
                            suffix=suffix,
                            delete=False,
                        ) as handle:
                            staged_path = Path(handle.name)
                            remaining = content_length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise TrainedArtifactError("trained artifact upload ended before Content-Length")
                                handle.write(chunk)
                                remaining -= len(chunk)
                        artifact = import_external_trained_artifact(
                            project_root,
                            staged_path,
                            operation=operation,
                            original_filename=filename,
                            label=label,
                            registry=registry,
                        )
                    except TrainedArtifactError as exc:
                        return self._error(400, str(exc))
                    finally:
                        if staged_path is not None and staged_path.exists():
                            staged_path.unlink()
                    return self._json({"status": "imported", "artifact": artifact}, status=201)
                if parsed.path == "/api/recipe/run":
                    payload = self._read_json()
                    try:
                        execution = _execution_options_from_request(payload)
                        recipe_path = _resolve_project_path(project_root, str(payload.get("path") or ""))
                        recipe = prepare_single_run_recipe(
                            load_recipe(recipe_path, mode="strict", registry=registry),
                            registry,
                        )
                        _require_requested_strict_lint(
                            recipe,
                            execution,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe cannot run: %s" % exc)
                    run_job = job_manager.start(recipe, execution)
                    run_job = job_manager.wait(
                        str(run_job["job_id"]), _LEGACY_SYNC_WAIT_SECONDS
                    )
                    if run_job["status"] in {"queued", "running"}:
                        return self._json(run_job, status=202)
                    if run_job["status"] != "completed":
                        return self._error(
                            422,
                            str(run_job.get("error") or "Recipe execution failed"),
                        )
                    return self._json(
                        {
                            "status": "completed",
                            "run_id": run_job["run_id"],
                            "summary_path": run_job["summary_path"],
                            "execution": execution,
                            "resource_guard": run_job["resource_guard"],
                        }
                    )
                if parsed.path == "/api/recipe/run-payload":
                    payload = self._read_json()
                    try:
                        execution = _execution_options_from_request(payload)
                        recipe = prepare_single_run_recipe(
                            payload.get("recipe") if isinstance(payload, Mapping) else None,
                            registry,
                        )
                        _require_requested_strict_lint(
                            recipe,
                            execution,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe cannot run: %s" % exc)
                    blockers = _runtime_artifact_recipe_blockers(
                        recipe, project_root, registry
                    )
                    if blockers:
                        return self._error(
                            400,
                            "Recipe artifact is not compatible: %s"
                            % "; ".join(blockers),
                        )
                    run_job = job_manager.start(recipe, execution)
                    run_job = job_manager.wait(
                        str(run_job["job_id"]), _LEGACY_SYNC_WAIT_SECONDS
                    )
                    if run_job["status"] in {"queued", "running"}:
                        return self._json(run_job, status=202)
                    if run_job["status"] != "completed":
                        return self._error(
                            422,
                            str(run_job.get("error") or "Recipe execution failed"),
                        )
                    return self._json(
                        {
                            "status": "completed",
                            "run_id": run_job["run_id"],
                            "summary_path": run_job["summary_path"],
                            "execution": execution,
                            "resource_guard": run_job["resource_guard"],
                        }
                    )
                if parsed.path == "/api/run-jobs":
                    payload = self._read_json()
                    try:
                        execution = _execution_options_from_request(payload)
                        recipe = prepare_single_run_recipe(
                            payload.get("recipe") if isinstance(payload, Mapping) else None,
                            registry,
                        )
                        _require_requested_strict_lint(
                            recipe,
                            execution,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe cannot run: %s" % exc)
                    blockers = _runtime_artifact_recipe_blockers(
                        recipe, project_root, registry
                    )
                    if blockers:
                        return self._error(
                            400,
                            "Recipe artifact is not compatible: %s"
                            % "; ".join(blockers),
                        )
                    return self._json(job_manager.start(recipe, execution))
                if parsed.path == "/api/benchmark-jobs":
                    payload = self._read_json()
                    try:
                        execution = _execution_options_from_request(payload)
                        pack_path = _resolve_project_path(
                            project_root, str(payload.get("path") or "")
                        )
                        pack = load_benchmark_pack(pack_path)
                        validate_benchmark_pack(
                            pack,
                            registry,
                            project_root,
                            strict_lint=bool(execution["strict_lint"]),
                        )
                        blockers = _benchmark_checkpoint_blockers(
                            pack, project_root
                        )
                    except Exception as exc:
                        return self._error(
                            400, "Benchmark cannot run: %s" % exc
                        )
                    if blockers:
                        return self._error(400, "Benchmark is not ready: %s" % "; ".join(blockers))
                    return self._json(
                        benchmark_job_manager.start(pack, execution)
                    )
                if parsed.path == "/api/dataset-capture-jobs":
                    payload = self._read_json()
                    try:
                        recipe, out_dir, force = _dataset_capture_request(
                            payload,
                            project_root,
                            workspace,
                            registry,
                        )
                    except (DatasetCaptureError, ValueError) as exc:
                        return self._error(400, str(exc))
                    return self._json(
                        dataset_capture_job_manager.start(
                            recipe,
                            out_dir,
                            force=force,
                        )
                    )
                if parsed.path.startswith("/api/run-jobs/") and parsed.path.endswith("/cancel"):
                    job_id = urllib.parse.unquote(parsed.path[len("/api/run-jobs/") : -len("/cancel")])
                    return self._json(job_manager.cancel(job_id))
                if parsed.path.startswith("/api/benchmark-jobs/") and parsed.path.endswith("/cancel"):
                    job_id = urllib.parse.unquote(
                        parsed.path[
                            len("/api/benchmark-jobs/") : -len("/cancel")
                        ]
                    )
                    return self._json(benchmark_job_manager.cancel(job_id))
                if parsed.path.startswith("/api/dataset-capture-jobs/") and parsed.path.endswith("/cancel"):
                    job_id = urllib.parse.unquote(
                        parsed.path[
                            len("/api/dataset-capture-jobs/") : -len("/cancel")
                        ]
                    )
                    return self._json(dataset_capture_job_manager.cancel(job_id))
                if parsed.path == "/api/recipe/validate":
                    payload = self._read_json()
                    try:
                        recipe, compilation, matrix = _validated_authored_recipe_payload(
                            payload.get("recipe") if isinstance(payload, Mapping) else None,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe is invalid: %s" % exc)
                    return self._json(
                        {
                            "status": "valid",
                            "recipe": recipe.to_dict(),
                            "validation": {
                                "status": "valid",
                                "mode": compilation.mode,
                                "defaults_materialized": compilation.defaults_materialized,
                                "diagnostics": [
                                    item.to_dict() for item in compilation.diagnostics
                                ]
                                + [item.to_dict() for item in matrix.diagnostics],
                            },
                        }
                    )
                if parsed.path == "/api/recipe/save":
                    payload = self._read_json()
                    try:
                        recipe, compilation, matrix = _validated_authored_recipe_payload(
                            payload.get("recipe") if isinstance(payload, Mapping) else None,
                            registry,
                        )
                    except Exception as exc:
                        return self._error(400, "Recipe is invalid: %s" % exc)
                    safe_name = _safe_recipe_filename(str(payload.get("filename") or recipe.name))
                    recipes_dir = project_root / "recipes"
                    recipes_dir.mkdir(parents=True, exist_ok=True)
                    recipe_path = recipes_dir / safe_name
                    if not recipe_path.suffix:
                        recipe_path = recipe_path.with_suffix(".yaml")
                    _atomic_write_text(
                        recipe_path,
                        yaml.safe_dump(recipe.to_dict(), sort_keys=False),
                    )
                    return self._json(
                        {
                            "status": "saved",
                            "path": str(recipe_path.relative_to(project_root)),
                            "recipe": recipe.to_dict(),
                            "validation": {
                                "status": "valid",
                                "mode": compilation.mode,
                                "defaults_materialized": compilation.defaults_materialized,
                                "diagnostics": [
                                    item.to_dict() for item in compilation.diagnostics
                                ]
                                + [item.to_dict() for item in matrix.diagnostics],
                            },
                        }
                    )
                if parsed.path == "/api/recipe/differentiable-inspect":
                    payload = self._read_json()
                    backend = str(payload.get("backend") or "torch")
                    include_power_normalization = bool(payload.get("include_power_normalization", False))
                    recipe = _training_request_recipe(payload)
                    inspection = inspect_training_feasibility(
                        recipe,
                        registry,
                        optimizable_steps=_replacement_steps_from_request(payload),
                        loss=_route_loss_steps_from_request(payload),
                    )
                    response: JsonDict = {
                        "status": "ok",
                        "inspection": inspection,
                        "capture_contract": inspect_training_capture(
                            recipe,
                            registry,
                            optimizable_steps=(
                                _replacement_steps_from_request(payload)
                                or inspection.get("selected_replacement_steps")
                                or inspection.get("selected_optimizable_steps")
                            ),
                            route_loss_steps=_route_loss_steps_from_request(payload),
                            project_root=project_root,
                        ),
                        "export_graph": None,
                        "export_graph_error": "",
                    }
                    try:
                        export_graph = build_export_graph_from_recipe(
                            recipe,
                            registry,
                            backend=backend,
                            include_power_normalization=include_power_normalization,
                            replacement_steps=inspection.get("selected_replacement_steps") or [],
                            loss_steps=inspection.get("selected_loss_steps") or [],
                        )
                        response["export_graph"] = export_graph.to_dict()
                    except (TrainingDependencyError, ValueError) as exc:
                        response["export_graph_error"] = str(exc)
                    return self._json(response)
                if parsed.path == "/api/recipe/differentiable-export":
                    payload = self._read_json()
                    recipe = _training_request_recipe(payload)
                    out_value = str(payload.get("out") or "differentiable_exports/%s" % _safe_path_stem(recipe.name))
                    raw_training_plan = (
                        payload.get("training_plan")
                        if isinstance(payload.get("training_plan"), Mapping)
                        else {}
                    )
                    training_objective = str(raw_training_plan.get("objective") or "")
                    try:
                        export_payload = export_differentiable_scenario(
                            recipe,
                            registry,
                            optimizable_steps=_replacement_steps_from_request(payload, "sender,receiver"),
                            route_loss_steps=_route_loss_steps_from_request(payload),
                            loss="",
                            framework=str(payload.get("framework") or "torch-sionna"),
                            out_dir=_resolve_differentiable_export_dir(project_root, out_value),
                            project_root=project_root,
                            force=bool(payload.get("force", False)),
                            exporter="auto",
                            include_starter=False,
                            training_objective=training_objective,
                        )
                    except (DifferentiableExportError, ValueError) as exc:
                        return self._error(400, str(exc))
                    return self._json({"status": "exported", "export": export_payload})
                if parsed.path == "/api/recipe/differentiable-export-graph":
                    payload = self._read_json()
                    recipe = _training_request_recipe(payload)
                    out_value = str(payload.get("out") or "differentiable_exports/%s_graph" % _safe_path_stem(recipe.name))
                    try:
                        export_payload = export_differentiable_graph_bundle(
                            recipe,
                            registry,
                            backend=str(payload.get("backend") or "torch"),
                            include_power_normalization=bool(payload.get("include_power_normalization", False)),
                            replacement_steps=_replacement_steps_from_request(payload),
                            loss_steps=_route_loss_steps_from_request(payload),
                            out_dir=_resolve_differentiable_export_dir(project_root, out_value),
                            force=bool(payload.get("force", False)),
                        )
                    except (TrainingDependencyError, ValueError) as exc:
                        return self._error(400, str(exc))
                    return self._json({"status": "exported", "export": export_payload})
                if parsed.path == "/api/recipe/dataset-capture-run":
                    payload = self._read_json()
                    try:
                        recipe, out_dir, force = _dataset_capture_request(
                            payload,
                            project_root,
                            workspace,
                            registry,
                        )
                        capture_job = dataset_capture_job_manager.start(
                            recipe,
                            out_dir,
                            force=force,
                        )
                        capture_job = dataset_capture_job_manager.wait(
                            str(capture_job["job_id"]),
                            _LEGACY_SYNC_WAIT_SECONDS,
                        )
                    except (DatasetCaptureError, ValueError) as exc:
                        return self._error(400, str(exc))
                    if capture_job["status"] in {"queued", "running"}:
                        return self._json(capture_job, status=202)
                    if capture_job["status"] != "completed":
                        return self._error(
                            422,
                            str(
                                capture_job.get("error")
                                or "Dataset capture failed"
                            ),
                        )
                    capture_payload = capture_job["dataset_capture"]
                    return self._json({"status": "captured", "dataset_capture": capture_payload})
                return self._error(404, "not found")
            except (StructuredInputError, UnicodeDecodeError) as exc:
                return self._error(400, "Invalid JSON request: %s" % exc)
            except RequestBodyTooLarge as exc:
                return self._error(413, str(exc))
            except JobCapacityError as exc:
                return self._error(429, str(exc))
            except KeyError as exc:
                return self._error(404, str(exc).strip("'"))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    self.close_connection = True
                    return None
                return self._exception(exc)

        def _read_json(self) -> JsonDict:
            if str(self.headers.get("Transfer-Encoding") or "").strip():
                raise StructuredInputError(
                    "chunked request bodies are not supported; provide Content-Length"
                )
            raw_length = str(self.headers.get("Content-Length") or "0").strip()
            try:
                length = int(raw_length)
            except (TypeError, ValueError) as exc:
                raise StructuredInputError(
                    "Content-Length must be a non-negative integer"
                ) from exc
            if length < 0:
                raise StructuredInputError(
                    "Content-Length must be a non-negative integer"
                )
            if length > _MAX_JSON_REQUEST_BYTES:
                raise RequestBodyTooLarge(
                    "JSON request body exceeds the %d-byte limit"
                    % _MAX_JSON_REQUEST_BYTES
                )
            raw = self.rfile.read(length) if length else b"{}"
            if len(raw) != length:
                raise StructuredInputError(
                    "JSON request body ended before Content-Length"
                )
            if not raw:
                return {}
            payload = decode_strict_yaml_or_json(
                raw.decode("utf-8"),
                input_format="json",
            )
            if not isinstance(payload, dict):
                raise StructuredInputError("JSON request body must contain an object")
            return payload

        def _serve_file(
            self,
            path: Path,
            *,
            query: str = "",
            revalidate: bool = False,
        ) -> None:
            if not path.is_file():
                return self._error(404, "not found")
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            if not revalidate:
                # index.html is deliberately never retained: it is the source of
                # the current static-asset URLs and must always be freshly read.
                return self._response_bytes(
                    data,
                    content_type,
                    cache_control="no-store, max-age=0",
                    cache_compression=False,
                )
            immutable = _content_addressed_static_version(query, data)
            cache_control = (
                "public, max-age=%d, immutable" % _STATIC_IMMUTABLE_MAX_AGE_SECONDS
                if immutable
                else "private, no-cache, must-revalidate"
            )
            return self._response_bytes(
                data,
                content_type,
                cache_control=cache_control,
                etag=_weak_content_etag(data),
                cache_compression=True,
            )

        def _stable_json(self, data: bytes) -> None:
            return self._response_bytes(
                data,
                "application/json; charset=utf-8",
                cache_control="private, no-cache, must-revalidate",
                etag=_weak_content_etag(data),
                cache_compression=True,
            )

        def _json(self, payload: Any, status: int = 200) -> None:
            try:
                return self._response_bytes(
                    _compact_json_bytes(payload),
                    "application/json; charset=utf-8",
                    status=status,
                    cache_control="no-store, max-age=0",
                    cache_compression=False,
                )
            except Exception as exc:
                if _is_client_disconnect(exc):
                    self.close_connection = True
                    return None
                raise

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            return self._response_bytes(
                data,
                content_type,
                status=status,
                cache_control="no-store, max-age=0",
                allow_gzip=False,
            )

        def _response_bytes(
            self,
            data: bytes,
            content_type: str,
            *,
            status: int = 200,
            cache_control: str,
            etag: str = "",
            allow_gzip: bool = True,
            cache_compression: bool = False,
        ) -> None:
            if (
                status == 200
                and etag
                and _etag_matches(str(self.headers.get("If-None-Match") or ""), etag)
            ):
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", cache_control)
                if allow_gzip:
                    self.send_header("Vary", "Accept-Encoding")
                self.end_headers()
                return

            response_data = data
            use_gzip = (
                allow_gzip
                and len(data) >= _GZIP_MINIMUM_BYTES
                and _compressible_content_type(content_type)
                and _accepts_gzip(str(self.headers.get("Accept-Encoding") or ""))
            )
            if use_gzip:
                response_data = (
                    cached_gzip(data)
                    if cache_compression
                    else gzip.compress(data, compresslevel=3, mtime=0)
                )

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response_data)))
            self.send_header("Cache-Control", cache_control)
            if etag:
                self.send_header("ETag", etag)
            if allow_gzip:
                self.send_header("Vary", "Accept-Encoding")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(response_data)

        def _error(self, status: int, message: str) -> None:
            return self._json({"status": "error", "error": message}, status=status)

        def _exception(self, exc: Exception) -> None:
            return self._json(
                {
                    "status": "error",
                    "error": "Internal server error",
                },
                status=500,
            )

    return NoemaUiHandler


def start_ui_server_in_thread(
    host: str,
    port: int,
    workspace: Path,
    project_root: Path,
    adapter_paths: Optional[List[str]] = None,
) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    handler = _make_handler(workspace.resolve(), project_root.resolve(), adapter_paths=adapter_paths)
    server = ThreadingHTTPServer((host, int(port)), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server, thread


def _recipe_purpose_summary(recipe) -> JsonDict:
    """Classify a recipe without making purpose metadata a loading requirement."""

    try:
        research_specs = research_specs_from_recipe(recipe)
        purpose = dict(research_specs.get("task") or {})
        if purpose:
            catalog_validation = dict(research_specs.get("catalog_validation") or {})
            errors = [str(item) for item in catalog_validation.get("errors") or []]
            warnings = [str(item) for item in catalog_validation.get("warnings") or []]
            purpose["source"] = str(research_specs.get("source") or "")
            purpose["status"] = "invalid" if errors else "warning" if warnings else "valid"
            if errors or warnings:
                purpose["error"] = "; ".join(errors or warnings)
        return purpose
    except Exception as exc:
        metadata = dict(recipe.metadata or {})
        research = metadata.get("research")
        research_task = (
            research.get("task")
            if isinstance(research, Mapping) and isinstance(research.get("task"), Mapping)
            else {}
        )
        purpose_id = str(research_task.get("id") or metadata.get("task_id") or "").strip()
        return {
            "id": purpose_id,
            "source": "invalid recipe research contract",
            "status": "invalid",
            "error": str(exc),
        }


def _list_recipes(project_root: Path, registry=None) -> list:
    registry = registry or build_registry()
    recipe_dir = project_root / "recipes"
    rows = []
    for path in sorted(list(recipe_dir.glob("*.yaml")) + list(recipe_dir.glob("*.json"))):
        try:
            recipe = load_recipe(
                path,
                mode="strict",
                registry=registry,
                effective=False,
            )
            rows.append(
                {
                    "name": recipe.name,
                    "path": str(path.relative_to(project_root)),
                    "description": recipe.description,
                    "step_count": len(recipe.steps),
                    "metadata": dict(recipe.metadata),
                    "purpose": _recipe_purpose_summary(recipe),
                    "execution_profile": recipe.execution_profile.to_dict(),
                    "execution_profile_report": inspect_execution_profile(recipe).to_dict(),
                    "run_readiness": inspect_recipe_run_readiness(recipe, registry),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": path.stem,
                    "path": str(path.relative_to(project_root)),
                    "description": str(exc),
                    "step_count": 0,
                    "metadata": {},
                    "purpose": {},
                    "execution_profile": None,
                    "execution_profile_report": None,
                    "status": "error",
                }
            )
    return rows


def _exported_project_payload(
    project_path: Path,
    project_root: Path,
    *,
    registry=None,
) -> JsonDict:
    project_dir, manifest_path = _exported_project_locations(project_path)
    if not manifest_path.is_file():
        raise ValueError("No exported-project manifest exists at %s" % manifest_path)
    manifest = load_strict_yaml_or_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("Exported-project manifest must be a mapping: %s" % manifest_path)
    manifest = dict(manifest)
    if str(manifest.get("kind") or "") not in {
        "noema.standalone_training_project",
        "noema.training_interface_bundle@1",
    }:
        raise ValueError("Unsupported exported-project manifest kind: %s" % manifest.get("kind"))

    captures = []
    capture_directories: Dict[str, Path] = {}
    capture_taps: Dict[str, List[str]] = {}
    for raw_job in list(manifest.get("capture_jobs") or []):
        if not isinstance(raw_job, Mapping):
            continue
        job = dict(raw_job)
        split = str(job.get("split") or "")
        recipe_path = _manifest_project_path(project_root, job.get("recipe_path"))
        output_dir = _manifest_project_path(project_root, job.get("output_dir"))
        capture_directories[split] = output_dir
        capture_taps[split] = [
            str(item.get("id") or "")
            for item in list(job.get("expected_taps") or [])
            if isinstance(item, Mapping) and str(item.get("id") or "")
        ]
        requested_samples = int(job.get("requested_samples") or 0)
        row: JsonDict = {
            **job,
            "recipe_path": _project_relative_or_absolute(recipe_path, project_root),
            "output_dir": _project_relative_or_absolute(output_dir, project_root),
            "status": "pending",
            "captured_samples": 0,
            "schema_valid": False,
            "issues": [],
        }
        try:
            capture_recipe = load_recipe(recipe_path)
            expected_sha = recipe_fingerprint(capture_recipe)
            schema_path = output_dir / "schema.json"
            if schema_path.is_file():
                schema = decode_strict_yaml_or_json(
                    schema_path.read_text(encoding="utf-8"),
                    input_format="json",
                )
                captured = int(schema.get("captured_samples") or 0)
                issues = []
                if str(schema.get("split") or "") != split:
                    issues.append("captured split does not match")
                if str(schema.get("recipe_sha256") or "") != expected_sha:
                    issues.append("captured recipe fingerprint does not match")
                if int(schema.get("requested_samples") or 0) != requested_samples:
                    issues.append("requested sample count does not match")
                if captured < requested_samples:
                    issues.append("capture is incomplete")
                row.update(
                    {
                        "status": "complete" if not issues else "invalid",
                        "captured_samples": captured,
                        "schema_valid": not issues,
                        "issues": issues,
                    }
                )
        except Exception as exc:
            row.update({"status": "invalid", "issues": [str(exc)]})
        captures.append(row)

    capture_integrity: JsonDict = {"status": "pending"}
    if captures and all(row.get("status") == "complete" for row in captures):
        try:
            capture_integrity = assert_disjoint_capture_splits(
                capture_directories,
                expected_taps=capture_taps,
            )
        except (CaptureSplitIntegrityError, FileNotFoundError) as exc:
            message = str(exc)
            capture_integrity = {"status": "invalid", "issues": [message]}
            for row in captures:
                row["status"] = "invalid"
                row["schema_valid"] = False
                row["issues"] = [*list(row.get("issues") or []), message]

    trained_artifacts = []
    for raw_artifact in list(manifest.get("trained_artifacts") or []):
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact_config = dict(raw_artifact)
        try:
            artifact_manifest_path = _manifest_project_path(
                project_root,
                artifact_config.get("manifest_path"),
            )
            trained_artifacts.append(
                inspect_trained_artifact(
                    artifact_manifest_path,
                    project_root=project_root,
                    registry=registry,
                )
            )
        except Exception as exc:
            trained_artifacts.append(
                {
                    **artifact_config,
                    "status": "pending" if "missing" in str(exc).lower() else "invalid",
                    "ready": False,
                    "issues": [str(exc)],
                    "compatible_operations": [],
                    "artifact": {},
                }
            )

    training = dict(manifest.get("training") or {})
    training_cwd = _manifest_project_path(project_root, training.get("working_directory") or project_dir)
    history_path = _manifest_project_path(project_root, training.get("history_path") or training_cwd / "training_history.json")
    evaluation = dict(manifest.get("evaluation") or {})
    metrics_path = _manifest_project_path(project_root, evaluation.get("metrics_path") or training_cwd / "evaluation_metrics.json")
    external_training = dict(manifest.get("external_training") or {})
    raw_demo_starter = external_training.get("optional_demo_scaffold") or {}
    demo_starter: JsonDict = {}
    if isinstance(raw_demo_starter, Mapping) and bool(raw_demo_starter.get("included")):
        demo_starter = dict(raw_demo_starter)
        demo_training = _demo_command_for_checkout(
            demo_starter.get("training"),
            project_root=project_root,
        )
        demo_cwd = _manifest_project_path(
            project_root,
            demo_training.get("working_directory") or project_dir / "reference_training",
        )
        demo_history = _manifest_project_path(
            project_root,
            demo_training.get("history_path") or demo_cwd / "training_history.json",
        )
        demo_evaluation = _demo_command_for_checkout(
            demo_starter.get("evaluation"),
            project_root=project_root,
        )
        demo_metrics = _manifest_project_path(
            project_root,
            demo_evaluation.get("metrics_path") or demo_cwd / "evaluation_metrics.json",
        )
        root_handoff, root_handoff_issues = _inspect_demo_root_handoff(
            project_dir,
            demo_starter.get("root_handoff"),
            project_root=project_root,
        )
        if root_handoff:
            root_training = dict(root_handoff.get("training") or {})
            root_evaluation = dict(root_handoff.get("evaluation") or {})
            # The launch commands move to the discoverable bundle root, while
            # evidence paths remain those declared by the nested demo project.
            demo_training.update(root_training)
            demo_evaluation.update(root_evaluation)
            demo_cwd = Path(str(root_training["working_directory"])).resolve()
            demo_evaluation["working_directory"] = _project_relative_or_absolute(
                Path(str(root_evaluation["working_directory"])).resolve(),
                project_root,
            )
        demo_starter["training"] = {
            **demo_training,
            "working_directory": _project_relative_or_absolute(demo_cwd, project_root),
            "history_path": _project_relative_or_absolute(demo_history, project_root),
            "history_exists": demo_history.is_file(),
        }
        demo_starter["evaluation"] = {
            **demo_evaluation,
            "metrics_path": _project_relative_or_absolute(demo_metrics, project_root),
            "metrics_exist": demo_metrics.is_file(),
        }
        rendered_root_handoff = demo_starter.get("root_handoff") or {}
        demo_starter["root_handoff"] = {
            **(
                dict(rendered_root_handoff)
                if isinstance(rendered_root_handoff, Mapping)
                else {}
            ),
            "active": bool(root_handoff),
            "issues": root_handoff_issues,
        }
    capture_ready = all(row.get("status") == "complete" for row in captures)
    return {
        "status": "ok",
        "revision": _exported_project_watch_payload(project_path, project_root)["revision"],
        "manifest": manifest,
        "manifest_path": _project_relative_or_absolute(manifest_path, project_root),
        "out_dir": _project_relative_or_absolute(project_dir, project_root),
        "exporter": str(manifest.get("exporter") or ""),
        "training_template": str(manifest.get("training_template") or ""),
        "demo_starter": demo_starter,
        "captures": captures,
        "capture_integrity": capture_integrity,
        "ready_for_external_training": capture_ready,
        "trained_artifacts": trained_artifacts,
        "artifacts_ready": bool(trained_artifacts) and all(
            bool(item.get("ready")) for item in trained_artifacts
        ),
        "training": {
            **training,
            "working_directory": _project_relative_or_absolute(training_cwd, project_root),
            "history_path": _project_relative_or_absolute(history_path, project_root),
            "history_exists": history_path.is_file(),
        },
        "evaluation": {
            **evaluation,
            "metrics_path": _project_relative_or_absolute(metrics_path, project_root),
            "metrics_exist": metrics_path.is_file(),
        },
    }


def _exported_project_locations(project_path: Path) -> Tuple[Path, Path]:
    """Resolve either a bundle directory or its explicit project manifest.

    A not-yet-created output directory must remain the intended project
    directory.  Treating every nonexistent path as a manifest used to inspect
    its parent directory instead, which made quiet bundle discovery unreliable.
    """

    path = Path(project_path)
    if path.is_file() or path.name == "project_manifest.yaml":
        return path.parent, path
    return path, path / "project_manifest.yaml"


def _exported_project_watch_payload(
    project_path: Path,
    project_root: Path,
) -> JsonDict:
    """Return a cheap revision token for externally modified training bundles.

    Training and demo helpers intentionally run outside the browser.  The token
    covers only files that can change Workbench state: the project manifest,
    capture completion schemas, returned-artifact manifests/components, demo
    handoff launchers, and the presence of declared history/metric evidence.
    Large captured tensor shards are deliberately excluded.
    """

    project_dir, manifest_path = _exported_project_locations(project_path)
    if not manifest_path.is_file():
        return {
            "status": "ok",
            "exists": False,
            "revision": "",
            "out_dir": _project_relative_or_absolute(project_dir, project_root),
        }

    digest = hashlib.sha256()
    seen = set()
    issues: List[str] = []

    def add_path(path: Path, *, existence_only: bool = False) -> None:
        resolved = Path(path).resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        try:
            relative = str(resolved.relative_to(Path(project_root).resolve()))
        except ValueError:
            relative = str(resolved)
        digest.update(relative.encode("utf-8", errors="surrogatepass"))
        try:
            stat = resolved.stat()
        except OSError:
            digest.update(b"\0missing")
            return
        digest.update(b"\0present")
        if existence_only:
            return
        digest.update(("\0%d\0%d" % (stat.st_size, stat.st_mtime_ns)).encode("ascii"))

    add_path(manifest_path)
    try:
        raw_manifest = load_strict_yaml_or_json(manifest_path)
    except Exception as exc:
        issues.append("Cannot decode project manifest %s: %s" % (manifest_path, exc))
        raw_manifest = {}
    if not isinstance(raw_manifest, Mapping):
        issues.append("Project manifest must contain an object: %s" % manifest_path)
        manifest: JsonDict = {}
    else:
        manifest = dict(raw_manifest)

    raw_capture_jobs = manifest.get("capture_jobs") or []
    if not isinstance(raw_capture_jobs, list):
        issues.append("Project manifest capture_jobs must be an array")
        raw_capture_jobs = []
    for index, raw_job in enumerate(raw_capture_jobs):
        if not isinstance(raw_job, Mapping):
            issues.append("Project manifest capture_jobs[%d] must be an object" % index)
            continue
        try:
            output_dir = _manifest_project_path(project_root, raw_job.get("output_dir"))
        except Exception as exc:
            issues.append("Invalid capture_jobs[%d].output_dir: %s" % (index, exc))
            continue
        add_path(output_dir / "schema.json")

    raw_trained_artifacts = manifest.get("trained_artifacts") or []
    if not isinstance(raw_trained_artifacts, list):
        issues.append("Project manifest trained_artifacts must be an array")
        raw_trained_artifacts = []
    for index, raw_artifact in enumerate(raw_trained_artifacts):
        if not isinstance(raw_artifact, Mapping):
            issues.append("Project manifest trained_artifacts[%d] must be an object" % index)
            continue
        try:
            artifact_manifest = _manifest_project_path(
                project_root,
                raw_artifact.get("manifest_path"),
            )
        except Exception as exc:
            issues.append("Invalid trained_artifacts[%d].manifest_path: %s" % (index, exc))
            continue
        add_path(artifact_manifest)
        if not artifact_manifest.is_file():
            continue
        try:
            raw_returned = load_strict_yaml_or_json(artifact_manifest)
        except Exception as exc:
            issues.append("Cannot decode trained artifact manifest %s: %s" % (artifact_manifest, exc))
            raw_returned = {}
        if not isinstance(raw_returned, Mapping):
            issues.append("Trained artifact manifest must contain an object: %s" % artifact_manifest)
            returned: JsonDict = {}
        else:
            returned = dict(raw_returned)

        def add_artifact_relative(value: Any) -> None:
            text = str(value or "").strip()
            if not text:
                return
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = artifact_manifest.parent / candidate
            candidate = candidate.resolve()
            if _is_relative_to(candidate, Path(project_root).resolve()):
                add_path(candidate)

        contract = returned.get("contract") or {}
        if isinstance(contract, Mapping):
            add_artifact_relative(contract.get("path"))
        for collection_name in ("components", "support_files"):
            for raw_file in list(returned.get(collection_name) or []):
                if isinstance(raw_file, Mapping):
                    add_artifact_relative(raw_file.get("path"))

    training = manifest.get("training") or {}
    if isinstance(training, Mapping):
        try:
            training_cwd = _manifest_project_path(
                project_root,
                training.get("working_directory") or project_dir,
            )
            history_path = _manifest_project_path(
                project_root,
                training.get("history_path") or training_cwd / "training_history.json",
            )
            add_path(history_path, existence_only=True)
        except Exception as exc:
            issues.append("Invalid training evidence path: %s" % exc)
    evaluation = manifest.get("evaluation") or {}
    if isinstance(evaluation, Mapping):
        try:
            metrics_path = _manifest_project_path(
                project_root,
                evaluation.get("metrics_path") or project_dir / "evaluation_metrics.json",
            )
            add_path(metrics_path, existence_only=True)
        except Exception as exc:
            issues.append("Invalid evaluation evidence path: %s" % exc)

    external_training = manifest.get("external_training") or {}
    scaffold = (
        external_training.get("optional_demo_scaffold") or {}
        if isinstance(external_training, Mapping)
        else {}
    )
    if isinstance(scaffold, Mapping) and bool(scaffold.get("included")):
        for section_name, evidence_name in (
            ("training", "history_path"),
            ("evaluation", "metrics_path"),
        ):
            section = scaffold.get(section_name) or {}
            if not isinstance(section, Mapping):
                continue
            value = section.get(evidence_name)
            if value:
                try:
                    add_path(
                        _manifest_project_path(project_root, value),
                        existence_only=True,
                    )
                except Exception as exc:
                    issues.append(
                        "Invalid optional demo %s.%s: %s"
                        % (section_name, evidence_name, exc)
                    )
        handoff = scaffold.get("root_handoff") or {}
        if isinstance(handoff, Mapping):
            for raw_file in list(handoff.get("managed_files") or []):
                if not isinstance(raw_file, Mapping):
                    continue
                relative = str(raw_file.get("path") or "").strip()
                if not relative:
                    continue
                candidate = (project_dir / relative).resolve()
                if _is_relative_to(candidate, project_dir.resolve()):
                    add_path(candidate)

    return {
        "status": "invalid" if issues else "ok",
        "exists": True,
        "revision": digest.hexdigest(),
        "out_dir": _project_relative_or_absolute(project_dir, project_root),
        "issues": issues,
    }


def _manifest_project_path(project_root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("exported-project manifest contains an empty path")
    return _resolve_project_path(project_root, text)


def _inspect_demo_root_handoff(
    project_dir: Path,
    raw_handoff: Any,
    *,
    project_root: Path,
) -> Tuple[JsonDict, List[str]]:
    """Validate helper-owned launchers before advertising their commands."""

    if not isinstance(raw_handoff, Mapping):
        return {}, []
    handoff = dict(raw_handoff)
    issues: List[str] = []
    if str(handoff.get("ownership") or "") != "noema_demo_helper":
        issues.append("root demo handoff has an unknown owner")
    if bool(handoff.get("normative", True)):
        issues.append("root demo handoff is not marked non-normative")

    managed: Dict[str, str] = {}
    for raw_file in list(handoff.get("managed_files") or []):
        if not isinstance(raw_file, Mapping):
            continue
        relative = str(raw_file.get("path") or "").strip()
        expected_sha = str(raw_file.get("sha256") or "").strip().lower()
        if relative:
            managed[relative] = expected_sha

    working_directory = _confined_demo_handoff_path(
        project_dir,
        handoff.get("working_directory") or ".",
        "root demo working directory",
        issues,
    )
    normalized: JsonDict = {**handoff}
    for section_name in ("training", "evaluation"):
        section = dict(handoff.get(section_name) or {})
        relative = str(section.get("path") or "").strip()
        launcher = _confined_demo_handoff_path(
            project_dir,
            relative,
            "%s launcher" % section_name,
            issues,
        )
        if relative not in managed:
            issues.append("%s launcher is not helper-managed" % section_name)
        elif launcher is not None:
            if not launcher.is_file():
                issues.append("%s launcher is missing" % section_name)
            else:
                actual_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()
                if not managed[relative] or actual_sha != managed[relative]:
                    issues.append("%s launcher hash does not match" % section_name)
        section["working_directory"] = str(working_directory or project_dir.resolve())
        checkout_root = Path(project_root).resolve()
        if (
            launcher is not None
            and (checkout_root / "pyproject.toml").is_file()
            and (checkout_root / "uv.lock").is_file()
        ):
            section["command"] = (
                "uv run --project %s --extra onnx python %s"
                % (shlex.quote(str(checkout_root)), shlex.quote(relative))
            )
            section["environment"] = "noema_uv_checkout"
        normalized[section_name] = section

    quickstart = str(handoff.get("quickstart") or "").strip()
    if quickstart:
        quickstart_path = _confined_demo_handoff_path(
            project_dir,
            quickstart,
            "demo quick-start",
            issues,
        )
        if quickstart not in managed:
            issues.append("demo quick-start is not helper-managed")
        elif quickstart_path is not None and quickstart_path.is_file():
            actual_sha = hashlib.sha256(quickstart_path.read_bytes()).hexdigest()
            if not managed[quickstart] or actual_sha != managed[quickstart]:
                issues.append("demo quick-start hash does not match")
        else:
            issues.append("demo quick-start is missing")

    if issues or working_directory is None:
        return {}, issues
    normalized["working_directory"] = _project_relative_or_absolute(
        working_directory,
        project_root,
    )
    return normalized, []


def _demo_command_for_checkout(raw_section: Any, *, project_root: Path) -> JsonDict:
    """Use Noema's supported interpreter for an attached repository demo."""

    section = dict(raw_section) if isinstance(raw_section, Mapping) else {}
    command = str(section.get("command") or "").strip()
    checkout_root = Path(project_root).resolve()
    if (
        command
        and not command.startswith("uv run ")
        and (checkout_root / "pyproject.toml").is_file()
        and (checkout_root / "uv.lock").is_file()
    ):
        section["command"] = "uv run --project %s --extra onnx %s" % (
            shlex.quote(str(checkout_root)),
            command,
        )
        section["environment"] = "noema_uv_checkout"
    return section


def _confined_demo_handoff_path(
    project_dir: Path,
    value: Any,
    label: str,
    issues: List[str],
) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        issues.append("%s is empty" % label)
        return None
    raw = Path(text)
    if raw.is_absolute():
        issues.append("%s must be bundle-relative" % label)
        return None
    resolved_root = project_dir.resolve()
    resolved = (resolved_root / raw).resolve()
    if not _is_relative_to(resolved, resolved_root):
        issues.append("%s escapes the training bundle" % label)
        return None
    return resolved


def _project_relative_or_absolute(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _step_trained_artifact_kind(step: Any, registry) -> str:
    """Return the active operation-owned artifact mode for a recipe step.

    Portable artifacts do not share one controller parameter: current operations
    use ``policy``, ``runtime``, or ``mode``.  The operation ABI is the authority
    for the value that activates its portable artifact binding.  Legacy checkpoint
    controls are discovered from the operation schema instead of another list of
    hard-coded parameter names.
    """

    params = dict(getattr(step, "params", None) or {})
    operation_id = str(getattr(step, "op", None) or "").strip()
    try:
        operation = registry.get(operation_id).describe()
    except Exception as exc:
        raise RuntimeError(
            "Cannot inspect operation %s for trained-artifact controls: %s"
            % (operation_id or "<empty>", exc)
        ) from exc
    schema = dict(operation.get("params_schema") or {})
    properties = dict(schema.get("properties") or {})

    def effective_value(name: str) -> Any:
        if name in params:
            return params[name]
        property_schema = properties.get(name)
        if isinstance(property_schema, Mapping) and "default" in property_schema:
            return property_schema.get("default")
        return None

    artifact_abi = dict(operation.get("trained_artifact_abi") or {})
    binding_params = dict(artifact_abi.get("binding_params") or {})
    portable_selectors = {
        str(name): expected
        for name, expected in binding_params.items()
        if str(expected) == "learned_artifact"
    }
    if portable_selectors and all(
        effective_value(name) == expected
        for name, expected in portable_selectors.items()
    ):
        return "portable"

    for param_name, raw_property in properties.items():
        if not isinstance(raw_property, Mapping):
            continue
        ui = raw_property.get("x-noema-ui") or {}
        if not isinstance(ui, Mapping) or ui.get("control") != "trained_artifact":
            continue
        visible_when = ui.get("visible_when") or {}
        if not isinstance(visible_when, Mapping):
            continue
        if not all(
            effective_value(str(name)) == expected
            for name, expected in visible_when.items()
        ):
            continue
        return "portable" if str(param_name) == "artifact_manifest_path" else "checkpoint"
    return ""


def _step_trained_artifact_entrypoint(step: Any, registry) -> str:
    params = dict(getattr(step, "params", None) or {})
    configured = str(params.get("artifact_entrypoint") or "").strip()
    if configured:
        return configured
    operation_id = str(getattr(step, "op", None) or "").strip()
    try:
        operation = registry.get(operation_id).describe()
    except Exception as exc:
        raise RuntimeError(
            "Cannot inspect operation %s for a trained-artifact entrypoint: %s"
            % (operation_id or "<empty>", exc)
        ) from exc
    return str(
        (operation.get("trained_artifact_abi") or {}).get("entrypoint_id") or ""
    ).strip()


def _runtime_artifact_recipe_blockers(recipe, project_root: Path, registry) -> List[str]:
    blockers: List[str] = []
    inspected_by_path: Dict[Path, JsonDict] = {}
    for step in recipe.steps:
        params = dict(step.params or {})
        try:
            artifact_kind = _step_trained_artifact_kind(step, registry)
        except Exception as exc:
            blockers.append("%s: %s" % (step.id, exc))
            continue
        if artifact_kind != "portable":
            continue
        raw_manifest = str(params.get("artifact_manifest_path") or "").strip()
        if not raw_manifest:
            blockers.append("%s: trained artifact manifest is not selected" % step.id)
            continue
        manifest_path = Path(raw_manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (Path(project_root).resolve() / manifest_path).resolve()
        else:
            manifest_path = manifest_path.resolve()
        inspected = inspected_by_path.get(manifest_path)
        if inspected is None:
            if not manifest_path.is_file():
                blockers.append("%s: trained artifact manifest is missing" % step.id)
                continue
            try:
                inspected = inspect_trained_artifact(
                    manifest_path,
                    project_root=project_root,
                    registry=registry,
                )
            except Exception as exc:
                blockers.append("%s: %s" % (step.id, exc))
                continue
            inspected_by_path[manifest_path] = inspected
            if not inspected.get("ready"):
                blockers.append(
                    "%s: trained artifact is not ready: %s"
                    % (
                        step.id,
                        "; ".join(
                            str(item)
                            for item in inspected.get("issues")
                            or (inspected.get("runtime") or {}).get(
                                "unavailable_reasons"
                            )
                            or ["runtime unavailable"]
                        ),
                    )
                )
        if inspected is not None:
            try:
                entrypoint = _step_trained_artifact_entrypoint(step, registry)
            except Exception as exc:
                blockers.append("%s: %s" % (step.id, exc))
                continue
            matching_bindings = [
                binding
                for binding in inspected.get("compatible_operations") or []
                if str(binding.get("operation") or "") == str(step.op)
                and (
                    not entrypoint
                    or str(binding.get("runtime_entrypoint") or "") == entrypoint
                )
            ]
            if not matching_bindings:
                blockers.append(
                    "%s: trained artifact has no compatible binding for %s%s"
                    % (
                        step.id,
                        step.op,
                        " entrypoint %s" % entrypoint if entrypoint else "",
                    )
                )
            blockers.extend(
                "%s: %s" % (step.id, issue)
                for issue in trained_artifact_recipe_compatibility_issues(
                    inspected,
                    recipe,
                )
            )
    return list(dict.fromkeys(blockers))


def _learned_checkpoint_readiness(
    recipe_path: Path,
    project_root: Optional[Path] = None,
) -> JsonDict:
    """Preflight every frozen learned artifact referenced by a recipe.

    Learned artifacts are a generic operation concern.  The first implementation
    checked only ``model.symbol_power_allocator``; that made a returned DeepJSCC
    sender/receiver pair invisible to benchmark preflight.  Runtime loaders still
    perform their own format-specific validation, while this check provides a
    consistent early error for missing training evidence, files, or hashes.
    """

    blockers: List[str] = []
    checkpoints: List[JsonDict] = []
    training_performed = False
    try:
        recipe_path = Path(recipe_path).resolve()
        recipe = load_recipe(recipe_path)
        registry = build_registry()
        training_performed = bool((recipe.metadata or {}).get("training_performed"))
        learned_steps = [
            (step, kind)
            for step in recipe.steps
            for kind in [_step_trained_artifact_kind(step, registry)]
            if kind
        ]
        if not learned_steps:
            blockers.append("recipe has no learned artifact operation")
        elif not training_performed and any(
            kind == "checkpoint" for _, kind in learned_steps
        ):
            blockers.append("external training has not marked the benchmark recipe complete")

        verified_by_identity: Dict[tuple[str, str], JsonDict] = {}
        for step, artifact_kind in learned_steps:
            params = dict(step.params or {})
            if artifact_kind == "portable":
                raw_manifest = str(params.get("artifact_manifest_path") or "").strip()
                manifest_path = _resolve_recipe_checkpoint_path(
                    raw_manifest,
                    recipe_path=recipe_path,
                    project_root=project_root,
                )
                entrypoint = _step_trained_artifact_entrypoint(step, registry)
                identity = (
                    str(manifest_path) if manifest_path is not None else raw_manifest,
                    "portable-v2:%s:%s" % (step.op, entrypoint),
                )
                existing = verified_by_identity.get(identity)
                if existing is not None:
                    existing.setdefault("steps", []).append(step.id)
                    continue
                row = {
                    "steps": [step.id],
                    "path": str(manifest_path) if manifest_path is not None else "",
                    "exists": bool(manifest_path and manifest_path.is_file()),
                    "expected_sha256": "",
                    "actual_sha256": "",
                    "format": "noema_trained_artifact_manifest_v2",
                    "valid": True,
                    "blockers": [],
                }
                if manifest_path is None or not manifest_path.is_file():
                    row["blockers"].append("trained artifact manifest is missing")
                else:
                    try:
                        inspected = inspect_trained_artifact(
                            manifest_path,
                            project_root=Path(project_root or recipe_path.parent),
                            registry=registry,
                        )
                        row["actual_sha256"] = str(
                            (inspected.get("artifact") or {}).get("sha256") or ""
                        )
                        matching = [
                            binding
                            for binding in inspected.get("compatible_operations") or []
                            if str(binding.get("operation") or "") == step.op
                            and (
                                not entrypoint
                                or str(binding.get("runtime_entrypoint") or "")
                                == entrypoint
                            )
                        ]
                        if not inspected.get("ready"):
                            row["blockers"].append(
                                "trained artifact is not ready: %s"
                                % "; ".join(
                                    str(item)
                                    for item in inspected.get("issues")
                                    or (inspected.get("runtime") or {}).get(
                                        "unavailable_reasons"
                                    )
                                    or ["runtime unavailable"]
                                )
                            )
                        if not matching:
                            row["blockers"].append(
                                "trained artifact has no compatible binding for %s%s"
                                % (
                                    step.op,
                                    " entrypoint %s" % entrypoint if entrypoint else "",
                                )
                            )
                        row["blockers"].extend(
                            trained_artifact_recipe_compatibility_issues(
                                inspected,
                                recipe,
                            )
                        )
                    except Exception as exc:
                        row["blockers"].append(str(exc))
                row["valid"] = not row["blockers"]
                checkpoints.append(row)
                verified_by_identity[identity] = row
                continue

            raw_checkpoint = str(params.get("checkpoint_path") or "").strip()
            expected_sha = str(params.get("checkpoint_sha256") or "").strip().lower()
            checkpoint_path = _resolve_recipe_checkpoint_path(
                raw_checkpoint,
                recipe_path=recipe_path,
                project_root=project_root,
            )
            identity = (
                str(checkpoint_path) if checkpoint_path is not None else raw_checkpoint,
                expected_sha,
            )
            existing = verified_by_identity.get(identity)
            if existing is not None:
                existing.setdefault("steps", []).append(step.id)
                continue

            row: JsonDict = {
                "steps": [step.id],
                "path": str(checkpoint_path) if checkpoint_path is not None else "",
                "exists": bool(checkpoint_path and checkpoint_path.is_file()),
                "expected_sha256": expected_sha,
                "actual_sha256": "",
                "format": str(params.get("checkpoint_format") or ""),
                "valid": True,
                "blockers": [],
            }
            if len(expected_sha) != 64 or any(
                char not in "0123456789abcdef" for char in expected_sha
            ):
                row["blockers"].append("checkpoint SHA-256 is missing or invalid")
            if checkpoint_path is None or not checkpoint_path.is_file():
                row["blockers"].append("trained checkpoint file is missing")
            else:
                actual_sha = _file_sha256(checkpoint_path)
                row["actual_sha256"] = actual_sha
                if expected_sha and actual_sha != expected_sha:
                    row["blockers"].append(
                        "trained checkpoint SHA-256 does not match the benchmark recipe"
                    )
            row["valid"] = not row["blockers"]
            checkpoints.append(row)
            verified_by_identity[identity] = row

        for row in checkpoints:
            step_label = ", ".join(str(item) for item in row.get("steps") or [])
            blockers.extend(
                "%s: %s" % (step_label, item)
                for item in row.get("blockers") or []
            )
    except Exception as exc:
        blockers.append(str(exc))

    primary = checkpoints[0] if checkpoints else {}
    return {
        # Keep the original single-checkpoint fields for API/tests while exposing
        # all unique checkpoints for paired or multi-model recipes.
        "path": str(primary.get("path") or ""),
        "exists": bool(primary.get("exists")),
        "expected_sha256": str(primary.get("expected_sha256") or ""),
        "actual_sha256": str(primary.get("actual_sha256") or ""),
        "checkpoints": checkpoints,
        "training_performed": training_performed,
        "valid": not blockers,
        "blockers": blockers,
    }


def _resolve_recipe_checkpoint_path(
    raw_checkpoint: str,
    *,
    recipe_path: Path,
    project_root: Optional[Path],
) -> Optional[Path]:
    raw = str(raw_checkpoint or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = []
    if project_root is not None:
        candidates.append((Path(project_root).resolve() / path).resolve())
    candidates.append((Path(recipe_path).resolve().parent / path).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _benchmark_checkpoint_blockers(pack: BenchmarkPack, project_root: Path) -> List[str]:
    blockers = []
    checked = set()
    registry = build_registry()
    for entry in pack.recipes:
        recipe_path = resolve_benchmark_recipe_path(pack, entry, project_root).resolve()
        if recipe_path in checked:
            continue
        checked.add(recipe_path)
        recipe = load_recipe(recipe_path)
        learned = any(
            _step_trained_artifact_kind(step, registry)
            for step in recipe.steps
        )
        if not learned:
            continue
        readiness = _learned_checkpoint_readiness(recipe_path, project_root)
        blockers.extend("%s: %s" % (entry.id, item) for item in readiness.get("blockers") or [])
    return blockers


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_images_payload(query: str, project_root: Path, dataset_lock: threading.Lock) -> JsonDict:
    params = urllib.parse.parse_qs(query)
    dataset = str((params.get("dataset") or ["kodak"])[0])
    if dataset != "kodak":
        raise ValueError("unsupported dataset: %s" % dataset)
    dataset_dir_value = str((params.get("dataset_dir") or [".noema/datasets/kodak"])[0])
    dataset_dir = _resolve_dataset_dir(project_root, dataset_dir_value)
    ensure = str((params.get("ensure") or ["0"])[0]).lower() in ("1", "true", "yes")
    if ensure:
        with dataset_lock:
            download_kodak_dataset(dataset_dir)
    expected = list(KODAK_FILENAMES)
    images = []
    if dataset_dir.is_dir():
        expected_names = set(expected)
        for path in sorted(dataset_dir.glob("*.png")):
            if path.name not in expected_names:
                continue
            images.append(
                {
                    "id": path.stem,
                    "filename": path.name,
                    "path": str(path),
                }
            )
    missing = [name for name in expected if not (dataset_dir / name).is_file()]
    return {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "images": images,
        "count": len(images),
        "expected_count": len(expected),
        "complete": not missing,
        "missing": missing,
    }


def _query_path(query: str, name: str, project_root: Path) -> Path:
    params = urllib.parse.parse_qs(query)
    values = params.get(name) or []
    if not values:
        raise ValueError("missing query parameter: %s" % name)
    return _resolve_project_path(project_root, values[0])


def _query_managed_artifact_path(
    query: str,
    name: str,
    project_root: Path,
    workspace: Path,
    store: LocalStore,
) -> Path:
    params = urllib.parse.parse_qs(query)
    values = params.get(name) or []
    if len(values) != 1:
        raise ValueError("missing query parameter: %s" % name)
    raw = str(values[0] or "").strip()
    if not raw:
        raise ValueError("missing query parameter: %s" % name)
    lexical_path = Path(raw).expanduser()
    if not lexical_path.is_absolute():
        lexical_path = Path(project_root) / lexical_path
    try:
        path = lexical_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("artifact path does not exist") from exc
    allowed_roots = {
        Path(store.runs_dir).resolve(),
        Path(store.benchmarks_dir).resolve(),
        (Path(workspace).resolve() / "dataset_captures").resolve(),
        (
            Path(project_root).resolve()
            / ".noema"
            / "dataset_captures"
        ).resolve(),
    }
    if not any(path != root and _is_relative_to(path, root) for root in allowed_roots):
        raise ValueError(
            "artifact path is outside Noema-managed run, benchmark, and capture storage"
        )
    if not path.is_file():
        raise ValueError("artifact path must identify a file")
    return path


def _resolve_differentiable_export_dir(project_root: Path, value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("differentiable export output directory is required")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    path = (project_root / path).resolve()
    try:
        path.relative_to(project_root)
    except ValueError:
        raise ValueError("differentiable export path escapes project root: %s" % value)
    return path


def _dataset_capture_request(
    payload: Mapping[str, Any],
    project_root: Path,
    workspace: Path,
    registry,
):
    project_path_value = str(payload.get("project_path") or "").strip()
    if project_path_value:
        if (
            payload.get("recipe") is not None
            or str(payload.get("path") or "").strip()
            or str(payload.get("out") or "").strip()
        ):
            raise ValueError(
                "manifest-backed dataset capture accepts project_path and split, not recipe, path, or out"
            )
        recipe, out_dir = _manifest_dataset_capture_request(
            project_path_value,
            str(payload.get("split") or "").strip(),
            project_root,
        )
        validate_recipe_against_registry(recipe, registry)
        force = bool(payload.get("force", False))
        _validate_ui_capture_target(out_dir, force=force)
        return recipe, out_dir, force

    recipe_value = payload.get("recipe")
    recipe_path_value = str(payload.get("path") or "").strip()
    if bool(recipe_value) == bool(recipe_path_value):
        raise ValueError("dataset capture requires exactly one of recipe or path")
    if recipe_path_value:
        recipe_path = _resolve_project_path(project_root, recipe_path_value)
        recipe = load_recipe(recipe_path)
        raw_plan = payload.get("training_plan")
        if raw_plan is not None:
            if not isinstance(raw_plan, Mapping):
                raise ValueError("training_plan must be a JSON object")
            recipe = apply_training_plan(recipe, training_plan_from_dict(raw_plan))
    else:
        recipe = _training_request_recipe(payload)
    validate_recipe_against_registry(recipe, registry)
    out_value = str(payload.get("out") or ".noema/dataset_captures/%s" % _safe_path_stem(recipe.name))
    out_dir = _resolve_ui_managed_capture_dir(
        project_root, workspace, out_value
    )
    force = bool(payload.get("force", False))
    _validate_ui_capture_target(out_dir, force=force)
    return recipe, out_dir, force


def _manifest_dataset_capture_request(
    project_path_value: str,
    split: str,
    project_root: Path,
):
    if not split:
        raise ValueError("manifest-backed dataset capture requires split")
    project_path = _resolve_project_path(project_root, project_path_value)
    project_dir = project_path if project_path.is_dir() else project_path.parent
    manifest_path = (
        project_path
        if project_path.is_file()
        else project_dir / "project_manifest.yaml"
    )
    if manifest_path.name != "project_manifest.yaml" or not manifest_path.is_file():
        raise ValueError(
            "No exported-project manifest exists at %s" % manifest_path
        )
    raw_manifest = load_strict_yaml_or_json(manifest_path)
    if not isinstance(raw_manifest, Mapping):
        raise ValueError(
            "Exported-project manifest must be a mapping: %s" % manifest_path
        )
    manifest = dict(raw_manifest)
    if str(manifest.get("kind") or "") not in {
        "noema.training_interface_bundle@1",
        "noema.standalone_training_project",
    }:
        raise ValueError(
            "Unsupported exported-project manifest kind: %s"
            % manifest.get("kind")
        )

    matches = [
        dict(item)
        for item in list(manifest.get("capture_jobs") or [])
        if isinstance(item, Mapping) and str(item.get("split") or "") == split
    ]
    if len(matches) != 1:
        raise ValueError(
            "Exported project must declare exactly one dataset capture for split %r"
            % split
        )
    job = matches[0]
    recipe_path = _manifest_project_path(project_root, job.get("recipe_path"))
    out_dir = _manifest_project_path(project_root, job.get("output_dir"))
    for label, path in (("capture recipe", recipe_path), ("capture output", out_dir)):
        if path == project_dir or not _is_relative_to(path, project_dir):
            raise ValueError(
                "Manifest-declared %s must be inside exported project %s"
                % (label, project_dir)
            )
    if not recipe_path.is_file():
        raise ValueError("Manifest-declared capture recipe is missing: %s" % recipe_path)

    expected_file_sha = str(job.get("recipe_file_sha256") or "").strip()
    if expected_file_sha:
        actual_file_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        if actual_file_sha != expected_file_sha:
            raise ValueError(
                "Manifest-declared capture recipe file hash mismatch for split %r"
                % split
            )
    recipe = load_recipe(recipe_path)
    capture = dict(recipe.dataset_capture or {})
    if str(capture.get("split") or "") != split:
        raise ValueError(
            "Manifest capture split %r does not match its recipe" % split
        )
    requested_samples = int(job.get("requested_samples") or 0)
    if requested_samples and int(capture.get("samples") or 0) != requested_samples:
        raise ValueError(
            "Manifest capture sample count does not match its recipe for split %r"
            % split
        )
    expected_taps = job.get("expected_taps")
    if expected_taps is not None and list(capture.get("taps") or []) != list(
        expected_taps
    ):
        raise ValueError(
            "Manifest capture taps do not match its recipe for split %r" % split
        )
    return recipe, out_dir


def _resolve_ui_managed_capture_dir(project_root: Path, workspace: Path, value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("dataset capture output directory is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    allowed_roots = {
        Path(workspace).resolve(),
        (Path(project_root).resolve() / ".noema" / "dataset_captures").resolve(),
        (Path(workspace).resolve() / "dataset_captures").resolve(),
    }
    if not any(path != root and _is_relative_to(path, root) for root in allowed_roots):
        raise ValueError(
            "UI-managed dataset capture output must be inside a Noema dataset_captures directory: %s"
            % value
        )
    return path


def _validate_ui_capture_target(out_dir: Path, *, force: bool) -> None:
    """Allow force replacement only for a completed Noema capture dataset."""

    path = Path(out_dir)
    if not path.exists() or not force:
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(
            "Dataset Capture force target must be a real Noema capture directory: %s"
            % path
        )
    schema_path = path / "schema.json"
    if schema_path.is_symlink() or not schema_path.is_file():
        raise ValueError(
            "Dataset Capture force will not replace an unowned directory: %s"
            % path
        )
    try:
        schema = load_strict_yaml_or_json(schema_path)
    except (OSError, StructuredInputError, ValueError) as exc:
        raise ValueError(
            "Dataset Capture force target has an invalid ownership schema: %s"
            % path
        ) from exc
    if not isinstance(schema, Mapping) or str(schema.get("kind") or "") != "noema.capture_dataset":
        raise ValueError(
            "Dataset Capture force will not replace an unowned directory: %s"
            % path
        )


def _safe_path_stem(value: str) -> str:
    return Path(_safe_recipe_filename(value)).stem or "recipe"


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / value
    path = path.resolve()
    try:
        path.relative_to(project_root)
    except ValueError:
        raise ValueError("path escapes project root: %s" % value)
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_dataset_dir(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / value
    return path.resolve()


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("path escapes static root")
    return path


def _safe_recipe_filename(value: str) -> str:
    stem = Path(value).stem or "recipe"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return safe.strip("._") or "recipe"


def _run_payload(store: LocalStore, run_id: str) -> JsonDict:
    payload = store.get_run(run_id)
    resource_guard_path = store.runs_dir / run_id / "resource-guard.json"
    if resource_guard_path.is_file() and not resource_guard_path.is_symlink():
        evidence = store.read_json(resource_guard_path)
        digest_payload = dict(evidence)
        declared_sha256 = digest_payload.pop("sha256", None)
        summary_path = store.runs_dir / run_id / "summary.json"
        manifest_path = store.runs_dir / run_id / "manifest.json"
        expected_summary = {
            "sha256": file_sha256(summary_path),
            "size_bytes": int(summary_path.stat().st_size),
        }
        expected_manifest = {
            "sha256": file_sha256(manifest_path),
            "size_bytes": int(manifest_path.stat().st_size),
        }
        if (
            evidence.get("schema_version") != 1
            or evidence.get("kind") != "noema.run_resource_guard_evidence"
            or evidence.get("run_id") != run_id
            or evidence.get("summary") != expected_summary
            or evidence.get("manifest") != expected_manifest
            or declared_sha256 != canonical_json_sha256(digest_payload)
            or not isinstance(evidence.get("resource_guard"), Mapping)
        ):
            raise ValueError(
                "run resource-guard evidence is invalid or bound to another run"
            )
        payload = dict(payload)
        payload["resource_guard"] = dict(evidence["resource_guard"])
        payload["resource_guard_evidence"] = {
            "path": "resource-guard.json",
            "sha256": file_sha256(resource_guard_path),
        }
    _enrich_run_payload_artifacts(payload)
    recipe_path = store.runs_dir / run_id / "recipe.json"
    if recipe_path.is_file():
        payload = dict(payload)
        payload["recipe"] = store.read_json(recipe_path)
    return payload


def _run_id_from_visual_path(path: str, endpoint: str) -> str:
    prefix = "/api/runs/"
    suffix = "/" + endpoint
    if not path.startswith(prefix) or not path.endswith(suffix):
        raise ValueError("invalid visual endpoint path")
    return urllib.parse.unquote(path[len(prefix) : -len(suffix)])


def _benchmark_result_id_from_path(path: str, endpoint: str) -> str:
    prefix = "/api/benchmarks/results/"
    suffix = "/" + endpoint
    if not path.startswith(prefix) or not path.endswith(suffix):
        raise ValueError("invalid benchmark result endpoint path")
    return urllib.parse.unquote(path[len(prefix) : -len(suffix)])


def _benchmark_result_run_ids_from_path(path: str) -> Tuple[str, str]:
    prefix = "/api/benchmarks/results/"
    if not path.startswith(prefix):
        raise ValueError("invalid benchmark result run endpoint path")
    remainder = path[len(prefix) :]
    raw_result_id, separator, raw_run_id = remainder.partition("/runs/")
    if not separator or not raw_result_id or not raw_run_id or "/" in raw_run_id:
        raise ValueError("invalid benchmark result run endpoint path")
    result_id = urllib.parse.unquote(raw_result_id)
    run_id = urllib.parse.unquote(raw_run_id)
    if Path(result_id).name != result_id or Path(run_id).name != run_id:
        raise ValueError("invalid benchmark result or run id")
    return result_id, run_id


def _segmentation_overlay_png(store: LocalStore, project_root: Path, workspace: Path, run_id: str, kind: str, index: int) -> bytes:
    image = _run_image_array(store, project_root, workspace, run_id, index)
    summary = store.get_run(run_id)
    step_id = "data" if str(kind).lower() in ("reference", "ground_truth", "gt") else "receiver"
    artifact_payload = _summary_artifact(summary, step_id, "segmentation")
    mask_path = _resolved_artifact_file(project_root, workspace, artifact_payload.get("path"))
    with np.load(str(mask_path), allow_pickle=False) as payload:
        masks = payload["masks"]
        if index < 0 or index >= int(masks.shape[0]):
            raise ValueError("segmentation index out of range: %s" % index)
        mask = np.asarray(masks[index])
    return _png_bytes(_segmentation_overlay_image(image, mask))


def _detection_overlay_png(store: LocalStore, project_root: Path, workspace: Path, run_id: str, kind: str, index: int) -> bytes:
    image = _run_image_array(store, project_root, workspace, run_id, index)
    summary = store.get_run(run_id)
    step_id = "data" if str(kind).lower() in ("reference", "ground_truth", "gt") else "receiver"
    artifact_payload = _summary_artifact(summary, step_id, "detections")
    detection_path = _resolved_artifact_file(project_root, workspace, artifact_payload.get("path"))
    payload = decode_strict_yaml_or_json(
        detection_path.read_text(encoding="utf-8"),
        input_format="json",
    )
    examples = payload.get("examples") or []
    if index < 0 or index >= len(examples):
        raise ValueError("detection index out of range: %s" % index)
    detections = examples[index].get("detections") or []
    return _png_bytes(_detection_overlay_image(image, detections))


def _run_image_array(store: LocalStore, project_root: Path, workspace: Path, run_id: str, index: int) -> np.ndarray:
    summary = store.get_run(run_id)
    artifact_payload = _summary_artifact(summary, "data", "images")
    image_path = _resolved_artifact_file(project_root, workspace, artifact_payload.get("path"))
    with np.load(str(image_path), allow_pickle=False) as payload:
        images = payload["images"]
        metadata = {}
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Run image metadata_json",
                )
            )
        if index < 0 or index >= int(images.shape[0]):
            raise ValueError("image index out of range: %s" % index)
        image = np.asarray(images[index])
    shape = _single_original_image_shape(metadata, index)
    if shape:
        _count, height, width, _channels = shape
        image = image[:height, :width, :]
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return image


def _summary_artifact(summary: JsonDict, step_id: str, name: str) -> JsonDict:
    for step in summary.get("steps") or []:
        if not isinstance(step, dict) or step.get("id") != step_id:
            continue
        outputs = step.get("outputs") or {}
        artifact_payload = outputs.get(name)
        if isinstance(artifact_payload, dict):
            return artifact_payload
    raise FileNotFoundError("%s.%s artifact not found" % (step_id, name))


def _resolved_artifact_file(project_root: Path, workspace: Path, value: Any) -> Path:
    if not value:
        raise ValueError("missing artifact path")
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    roots = [project_root.resolve(), workspace.resolve()]
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("artifact path escapes workspace: %s" % value)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _segmentation_overlay_image(image: np.ndarray, mask: np.ndarray):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Install Pillow to render segmentation overlays") from exc
    base = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").convert("RGBA")
    if mask.shape != image.shape[:2]:
        mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint16), mode="I;16")
        mask = np.asarray(mask_image.resize(base.size, resample=Image.NEAREST), dtype=np.uint16)
    mask = np.asarray(mask, dtype=np.int64)
    colors = _class_palette()
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    active = mask > 0
    if np.any(active):
        overlay[active, :3] = colors[(mask[active] - 1) % len(colors)]
        overlay[active, 3] = 118
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def _detection_overlay_image(image: np.ndarray, detections: List[JsonDict]):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError("Install Pillow to render detection overlays") from exc
    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = _class_palette()
    width, height = canvas.size
    for item in detections:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        class_id = int(item.get("class_id") or 0)
        color = tuple(int(value) for value in colors[class_id % len(colors)])
        x1, y1, x2, y2 = [float(value) for value in bbox]
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width - 1), x2))
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height - 1), y2))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(item.get("label") or ("class_%d" % class_id))
        score = item.get("score")
        if isinstance(score, (int, float)):
            label = "%s %.2f" % (label, float(score))
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        text_height = bottom - top + 4
        text_width = right - left + 6
        text_y = max(0.0, y1 - text_height)
        draw.rectangle([x1, text_y, min(float(width), x1 + text_width), text_y + text_height], fill=color)
        draw.text((x1 + 3, text_y + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def _class_palette() -> np.ndarray:
    return np.asarray(
        [
            (37, 99, 235),
            (16, 185, 129),
            (245, 158, 11),
            (239, 68, 68),
            (139, 92, 246),
            (20, 184, 166),
            (236, 72, 153),
            (132, 204, 22),
            (249, 115, 22),
            (14, 165, 233),
            (168, 85, 247),
            (234, 179, 8),
        ],
        dtype=np.uint8,
    )


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _benchmark_result_payload(store: LocalStore, result_id: str) -> JsonDict:
    result = store.get_benchmark_result(result_id)
    plots = load_benchmark_plot_records(
        store.get_benchmark_result_dir(result_id)
    )
    if plots:
        result["plots"] = plots
    return {"result_id": result_id, "result": result}


def _benchmark_result_run_payload(
    store: LocalStore,
    result_id: str,
    run_id: str,
) -> JsonDict:
    """Return a run-shaped payload from one authenticated result-local snapshot."""

    result = store.get_benchmark_result(result_id)
    raw_entries = result.get("recipes")
    if not isinstance(raw_entries, list):
        raise ValueError("Benchmark result recipes must be a JSON array")
    matches = [
        (index, entry)
        for index, entry in enumerate(raw_entries)
        if isinstance(entry, Mapping) and str(entry.get("run_id") or "") == run_id
    ]
    if not matches:
        raise KeyError(
            "Benchmark result %s has no backing run %s" % (result_id, run_id)
        )
    if len(matches) != 1:
        raise ValueError(
            "Benchmark result %s contains duplicate backing run id %s"
            % (result_id, run_id)
        )
    entry_index, entry = matches[0]
    validated = validate_benchmark_run_evidence_snapshot(
        store.get_benchmark_result_dir(result_id),
        entry,
        entry_index=entry_index,
    )
    payload = dict(validated["summary"])
    payload["recipe"] = dict(validated["recipe"])
    _localize_benchmark_snapshot_artifact_paths(
        payload,
        validated,
        store.get_benchmark_result_dir(result_id),
    )
    payload["benchmark_run_evidence"] = {
        "source": "result_local_snapshot",
        "result_id": result_id,
        "entry_id": str(validated.get("entry_id") or ""),
        "entry_index": entry_index,
        "verification": dict(validated.get("verification") or {}),
        "descriptor": dict(validated.get("descriptor") or {}),
    }
    return payload


def _localize_benchmark_snapshot_artifact_paths(
    payload: JsonDict,
    validated: Mapping[str, Any],
    result_dir: Path,
) -> None:
    """Point retained summary outputs at their authenticated result-local copies."""

    descriptor = validated.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    relative_root = str(descriptor.get("root") or "").rstrip("/")
    snapshot_manifest = validated.get("snapshot_manifest")
    snapshot_manifest = (
        snapshot_manifest if isinstance(snapshot_manifest, Mapping) else {}
    )
    retained_by_relative: Dict[str, Path] = {}
    prefix = relative_root + "/" if relative_root else ""
    for record in snapshot_manifest.get("files") or []:
        if not isinstance(record, Mapping) or record.get("role") != "artifact":
            continue
        path = str(record.get("path") or "")
        if not prefix or not path.startswith(prefix):
            continue
        retained_by_relative[path[len(prefix) :]] = result_dir / path

    manifest = validated.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    retained_by_output: Dict[Tuple[str, str], Path] = {}
    for artifact in manifest.get("artifacts") or []:
        if not isinstance(artifact, Mapping):
            continue
        retained = retained_by_relative.get(
            str(artifact.get("relative_path") or "")
        )
        if retained is not None:
            retained_by_output[
                (
                    str(artifact.get("step_id") or ""),
                    str(artifact.get("output_name") or ""),
                )
            ] = retained

    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        outputs = step.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for output_name, artifact in outputs.items():
            retained = retained_by_output.get((step_id, str(output_name)))
            if retained is not None and isinstance(artifact, dict):
                artifact["path"] = str(retained)


def _enrich_run_payload_artifacts(payload: JsonDict) -> None:
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        outputs = step.get("outputs") or {}
        if not isinstance(outputs, dict):
            continue
        for artifact_payload in outputs.values():
            if not isinstance(artifact_payload, dict):
                continue
            path = Path(str(artifact_payload.get("path") or ""))
            if not path.is_file() or path.suffix != ".npz":
                continue
            metadata = dict(artifact_payload.get("metadata") or {})
            arrays = dict(metadata.get("arrays") or {})
            try:
                with np.load(str(path), allow_pickle=False) as npz:
                    for name in npz.files:
                        if name == "metadata_json":
                            continue
                        value = npz[name]
                        arrays[name] = {
                            "shape": [int(item) for item in value.shape],
                            "dtype": str(value.dtype),
                        }
            except Exception as exc:
                artifact_payload["inspection"] = {
                    "status": "invalid",
                    "error": "Cannot inspect NPZ artifact %s: %s" % (path, exc),
                }
                continue
            artifact_payload["inspection"] = {"status": "ok"}
            if arrays:
                metadata["arrays"] = arrays
                if len(arrays) == 1:
                    name, info = next(iter(arrays.items()))
                    metadata.setdefault("array", name)
                    metadata.setdefault("dtype", info["dtype"])
                    metadata.setdefault("shape", info["shape"])
                artifact_payload["metadata"] = metadata


def _artifact_preview(path: Path) -> JsonDict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix == ".json":
        return {
            "kind": "json",
            "path": str(path),
            "payload": decode_strict_yaml_or_json(
                path.read_text(encoding="utf-8"),
                input_format="json",
            ),
        }
    if path.suffix == ".npz":
        preview = {"kind": "npz", "path": str(path), "arrays": {}}
        with np.load(str(path), allow_pickle=False) as payload:
            for name in payload.files:
                value = payload[name]
                if name == "metadata_json":
                    preview["metadata"] = decode_strict_json_object(
                        str(value.item()),
                        label="Artifact preview metadata_json",
                    )
                else:
                    preview["arrays"][name] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
        return preview
    return {"kind": "file", "path": str(path), "size_bytes": path.stat().st_size}


def _image_bmp(path: Path, array_name: str, index: int) -> bytes:
    with np.load(str(path), allow_pickle=False) as payload:
        images = payload[array_name]
        metadata = {}
        if "metadata_json" in payload:
            metadata.update(
                decode_strict_json_object(
                    str(payload["metadata_json"]),
                    label="Image metadata_json",
                )
            )
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("expected image array [N,H,W,3], got %s" % (images.shape,))
    if index < 0 or index >= images.shape[0]:
        raise ValueError("image index out of range: %s" % index)
    image = images[index]
    shape = _single_original_image_shape(metadata, index)
    if shape:
        _count, height, width, _channels = shape
        image = image[:height, :width, :]
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    height, width, _channels = image.shape
    row_stride = ((width * 3 + 3) // 4) * 4
    padding = b"\x00" * (row_stride - width * 3)
    rows = []
    for row in image[::-1]:
        bgr = row[:, ::-1].tobytes()
        rows.append(bgr + padding)
    pixel_data = b"".join(rows)
    file_size = 54 + len(pixel_data)
    header = (
        b"BM"
        + int(file_size).to_bytes(4, "little")
        + b"\x00\x00\x00\x00"
        + int(54).to_bytes(4, "little")
        + int(40).to_bytes(4, "little")
        + int(width).to_bytes(4, "little")
        + int(height).to_bytes(4, "little")
        + int(1).to_bytes(2, "little")
        + int(24).to_bytes(2, "little")
        + int(0).to_bytes(4, "little")
        + int(len(pixel_data)).to_bytes(4, "little")
        + int(2835).to_bytes(4, "little")
        + int(2835).to_bytes(4, "little")
        + int(0).to_bytes(4, "little")
        + int(0).to_bytes(4, "little")
    )
    return header + pixel_data


def _single_original_image_shape(metadata: JsonDict, index: int):
    shapes = metadata.get("original_shapes")
    if isinstance(shapes, (list, tuple)) and index < len(shapes):
        value = shapes[index]
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [1, int(value[1]), int(value[2]), int(value[3])]
    original = metadata.get("original_shape") or metadata.get("shape")
    if isinstance(original, (list, tuple)) and len(original) == 4:
        return [1, int(original[1]), int(original[2]), int(original[3])]
    return None
