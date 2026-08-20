from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from noema_lab.core.structured_input import (
    StructuredInputError,
    decode_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]
MIB = 1024 * 1024
GIB = 1024 * MIB


class IsolatedJobError(RuntimeError):
    pass


class ResourceExhausted(IsolatedJobError):
    def __init__(self, message: str, evidence: Mapping[str, Any], payload: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)
        self.payload = dict(payload or {})


class ExecutionDeadlineExceeded(IsolatedJobError):
    def __init__(
        self,
        message: str,
        evidence: Mapping[str, Any],
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)
        self.payload = dict(payload or {})


class _AdmissionCanceled(RuntimeError):
    pass


class _AdmissionDeadline(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryGuardConfig:
    reserve_bytes: int
    poll_hz: float = 10.0
    high_ratio: float = 1.0
    swap_max_bytes: int = GIB
    explicit_max_bytes: Optional[int] = None
    timeout_seconds: float = 3600.0

    @classmethod
    def from_environment(cls) -> "MemoryGuardConfig":
        total = _memory_snapshot_bytes().get("MemTotal", 0)
        default_reserve = max(2 * GIB, int(math.ceil(total * 0.10)))
        reserve = _environment_mib("NOEMA_EXECUTION_MEMORY_RESERVE_MB", default_reserve)
        explicit = _optional_environment_mib("NOEMA_EXECUTION_MEMORY_MAX_MB")
        swap_max = _environment_mib("NOEMA_EXECUTION_MEMORY_SWAP_MAX_MB", GIB)
        poll_hz = _environment_float("NOEMA_EXECUTION_MEMORY_POLL_HZ", 10.0, 1.0, 50.0)
        high_ratio = _environment_float("NOEMA_EXECUTION_MEMORY_HIGH_RATIO", 1.0, 0.50, 1.0)
        timeout_seconds = _environment_float(
            "NOEMA_EXECUTION_TIMEOUT_SECONDS", 3600.0, 1.0, 86400.0
        )
        return cls(
            reserve_bytes=reserve,
            poll_hz=poll_hz,
            high_ratio=high_ratio,
            swap_max_bytes=swap_max,
            explicit_max_bytes=explicit,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class IsolatedJobResult:
    status: str
    payload: JsonDict
    evidence: JsonDict


class IsolatedJobSupervisor:
    """Run one Noema job outside the server and protect the host from OOM.

    Linux cgroup v2 limits are requested through a transient user systemd
    service when available. A process-group and MemAvailable watchdog remains
    active on every platform as the portable safety/failure boundary.
    """

    def __init__(
        self,
        *,
        job_id: str,
        request: Mapping[str, Any],
        workspace: Path,
        project_root: Path,
        config: Optional[MemoryGuardConfig] = None,
        event_sink: Optional[Callable[[JsonDict], None]] = None,
        environment_set: Optional[Mapping[str, str]] = None,
        environment_unset: Optional[Iterable[str]] = None,
    ) -> None:
        self.job_id = str(job_id)
        self.request = dict(request)
        self.workspace = Path(workspace).resolve()
        self.project_root = Path(project_root).resolve()
        self.config = config or MemoryGuardConfig.from_environment()
        self.event_sink = event_sink
        self.environment_set = _validated_environment_set(environment_set or {})
        self.environment_unset = _validated_environment_unset(
            environment_unset or (),
            forbidden_names=self.environment_set,
        )
        self.process: Optional[subprocess.Popen] = None
        self._cancel_requested = threading.Event()
        self._event_offset = 0
        self._event_remainder = ""
        self._cgroup_path: Optional[Path] = None
        self._cgroup_memory_sample_count = 0
        self._cgroup_kernel_peak_bytes = 0
        self._cgroup_sampled_current_peak_bytes = 0
        self._cgroup_memory_events: JsonDict = {}
        self.control_dir = self.workspace / "job-control" / self.job_id
        self.request_path = self.control_dir / "request.json"
        self.status_path = self.control_dir / "status.json"
        self.events_path = self.control_dir / "events.jsonl"
        self.log_path = self.control_dir / "worker.log"
        self.bytecode_cache_path = self.control_dir / "bytecode-cache"
        self.evidence: JsonDict = {}

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._terminate_process_group(signal.SIGTERM)

    def run(self) -> IsolatedJobResult:
        deadline = time.monotonic() + self.config.timeout_seconds
        try:
            with _execution_admission(
                self.workspace,
                cancel_event=self._cancel_requested,
                deadline=deadline,
            ):
                return self._run_admitted(deadline=deadline)
        except _AdmissionCanceled:
            available = _available_memory_bytes()
            evidence = self._base_evidence(
                available_at_start=available,
                hard_limit=0,
                high_limit=0,
                backend="not_started",
            )
            evidence["termination_reason"] = "user_canceled"
            return IsolatedJobResult("canceled", {}, evidence)
        except _AdmissionDeadline:
            available = _available_memory_bytes()
            evidence = self._base_evidence(
                available_at_start=available,
                hard_limit=0,
                high_limit=0,
                backend="not_started",
            )
            evidence["termination_reason"] = "execution_deadline"
            raise ExecutionDeadlineExceeded(
                "Execution exceeded the %.0f-second deadline while waiting for admission."
                % self.config.timeout_seconds,
                evidence,
            )

    def _run_admitted(self, *, deadline: Optional[float] = None) -> IsolatedJobResult:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_bytecode_cache_path()
        _atomic_json_write(self.request_path, self.request)
        available_at_start = _available_memory_bytes()
        if self._cancel_requested.is_set():
            evidence = self._base_evidence(
                available_at_start=available_at_start,
                hard_limit=0,
                high_limit=0,
                backend="not_started",
            )
            evidence["termination_reason"] = "user_canceled"
            return IsolatedJobResult("canceled", {}, evidence)
        reserve_limited_max = available_at_start - self.config.reserve_bytes
        hard_limit = (
            min(self.config.explicit_max_bytes, reserve_limited_max)
            if self.config.explicit_max_bytes is not None
            else reserve_limited_max
        )
        if hard_limit < 256 * MIB:
            evidence = self._base_evidence(
                available_at_start=available_at_start,
                hard_limit=max(0, hard_limit),
                high_limit=0,
                backend="not_started",
            )
            evidence["termination_reason"] = "insufficient_memory_at_launch"
            raise ResourceExhausted(
                "Execution was not started because the protected host-memory reserve is unavailable.",
                evidence,
            )
        high_limit = max(128 * MIB, int(hard_limit * self.config.high_ratio))
        backend = "systemd_cgroup_v2" if _systemd_user_scope_available() else "process_group_watchdog"
        self.evidence = self._base_evidence(
            available_at_start=available_at_start,
            hard_limit=hard_limit,
            high_limit=high_limit,
            backend=backend,
        )
        environment = self._child_environment()
        command = self._worker_command()
        if backend == "systemd_cgroup_v2":
            command = self._systemd_scope_command(
                command,
                high_limit,
                hard_limit,
                environment=environment,
            )
        self.evidence["worker_log"] = str(self.log_path)
        minimum_available = available_at_start
        termination_reason = ""
        started = time.monotonic()
        try:
            with self.log_path.open("ab", buffering=0) as log_handle:
                self.process = subprocess.Popen(
                    command,
                    cwd=str(self.project_root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
                self.evidence["launcher_pid"] = int(self.process.pid)
                while self.process.poll() is None:
                    self._drain_events()
                    if backend == "systemd_cgroup_v2":
                        self._sample_systemd_cgroup_memory()
                    available = _available_memory_bytes()
                    minimum_available = min(minimum_available, available)
                    if self._cancel_requested.is_set():
                        termination_reason = "user_canceled"
                        self._terminate_with_grace()
                        break
                    if (
                        time.monotonic()
                        >= (
                            deadline
                            if deadline is not None
                            else started + self.config.timeout_seconds
                        )
                    ):
                        termination_reason = "execution_deadline"
                        self._terminate_with_grace()
                        break
                    if available <= self.config.reserve_bytes:
                        termination_reason = "system_memory_reserve"
                        self._terminate_with_grace()
                        break
                    time.sleep(1.0 / self.config.poll_hz)
                if backend == "systemd_cgroup_v2":
                    self._sample_systemd_cgroup_memory()
                return_code = self.process.wait()
        except BaseException:
            self._terminate_with_grace()
            raise
        finally:
            if self.process is not None and self.process.poll() is None:
                self._terminate_with_grace()
        self._drain_events(final=True)
        self._verify_bytecode_cache_after_exit()
        available_at_end = _available_memory_bytes()
        minimum_available = min(minimum_available, available_at_end)
        if backend == "systemd_cgroup_v2":
            self.evidence.update(self._systemd_unit_evidence())
        self.evidence.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "minimum_available_bytes": int(minimum_available),
                "available_at_end_bytes": int(available_at_end),
                "return_code": int(return_code),
                "termination_reason": termination_reason or None,
            }
        )
        status_payload = _read_json_if_present(self.status_path)
        if termination_reason == "user_canceled":
            return IsolatedJobResult("canceled", status_payload, dict(self.evidence))
        if termination_reason == "execution_deadline":
            raise ExecutionDeadlineExceeded(
                "Execution exceeded the %.0f-second deadline."
                % self.config.timeout_seconds,
                self.evidence,
                status_payload,
            )
        if termination_reason:
            raise ResourceExhausted(
                "Execution stopped before system memory fell below the protected reserve.",
                self.evidence,
                status_payload,
            )
        worker_status = str(status_payload.get("status") or "")
        if worker_status == "completed" and return_code == 0:
            return IsolatedJobResult("completed", status_payload, dict(self.evidence))
        log_tail = _tail_text(self.log_path, 8192).lower()
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        if worker_status == "resource_exhausted" or return_code in {75, 137, -sigkill} or (
            backend == "systemd_cgroup_v2" and "oom-kill" in log_tail
        ):
            self.evidence["termination_reason"] = "worker_memory_limit"
            raise ResourceExhausted(
                str(status_payload.get("error") or "Execution exceeded its isolated memory limit."),
                self.evidence,
                status_payload,
            )
        raise IsolatedJobError(
            str(status_payload.get("error") or "Isolated execution worker exited with code %d; see %s" % (return_code, self.log_path))
        )

    def _prepare_bytecode_cache_path(self) -> None:
        """Create the one empty, private bytecode lookup root for this worker."""

        path = self.bytecode_cache_path
        if path.is_symlink():
            raise IsolatedJobError("isolated worker bytecode cache path is a symlink")
        if path.exists():
            if not path.is_dir():
                raise IsolatedJobError(
                    "isolated worker bytecode cache path is not a directory"
                )
            try:
                if any(path.iterdir()):
                    raise IsolatedJobError(
                        "isolated worker bytecode cache path is not empty"
                    )
            except OSError as exc:
                raise IsolatedJobError(
                    "isolated worker bytecode cache path cannot be inspected"
                ) from exc
        else:
            path.mkdir(mode=0o700)
        resolved_control = self.control_dir.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_control)
        except ValueError as exc:  # pragma: no cover - defensive containment
            raise IsolatedJobError(
                "isolated worker bytecode cache path escapes job control"
            ) from exc

    def _verify_bytecode_cache_after_exit(self) -> None:
        path = self.bytecode_cache_path
        try:
            safe_and_empty = (
                path.is_dir()
                and not path.is_symlink()
                and not any(path.iterdir())
            )
        except OSError as exc:
            raise IsolatedJobError(
                "isolated worker bytecode cache path cannot be inspected after exit"
            ) from exc
        if not safe_and_empty:
            raise IsolatedJobError(
                "isolated worker bytecode cache path changed during execution"
            )
        self.evidence["bytecode_cache_empty_after_exit"] = True

    def _worker_command(self):
        return [
            sys.executable,
            "-I",
            "-B",
            "-X",
            "pycache_prefix=%s" % self.bytecode_cache_path,
            "-m",
            "noema_lab.core.job_worker",
            "--request",
            str(self.request_path),
            "--status",
            str(self.status_path),
            "--events",
            str(self.events_path),
        ]

    def _child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        # The worker is launched with Python's isolated mode below.  Do not
        # forward an ambient PYTHONPATH as a second, unauthenticated import
        # surface.  The selected interpreter must resolve the installed
        # noema_lab package through its own site-packages configuration.
        environment.pop("PYTHONPATH", None)
        for name in self.environment_unset:
            environment.pop(name, None)
        environment.update(self.environment_set)
        return environment

    def _systemd_scope_command(
        self,
        command,
        high_limit: int,
        hard_limit: int,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ):
        unit = "noema-job-%s.service" % "".join(c for c in self.job_id.lower() if c.isalnum())[:24]
        self.evidence["cgroup_unit"] = unit
        return [
            "systemd-run",
            "--user",
            "--wait",
            "--pipe",
            "--unit",
            unit,
            "--property",
            "MemoryAccounting=yes",
            "--property",
            "MemoryHigh=%d" % high_limit,
            "--property",
            "MemoryMax=%d" % hard_limit,
            "--property",
            "MemorySwapMax=%d" % self.config.swap_max_bytes,
            "--property",
            "OOMPolicy=kill",
            "--property",
            "KillMode=control-group",
            "--working-directory",
            str(self.project_root),
            *self._systemd_environment_arguments(environment=environment),
            "--",
            *command,
        ]

    def _systemd_environment_arguments(
        self,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ):
        environment = dict(environment or self._child_environment())
        default_names = (
            "PATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TF_CPP_MIN_LOG_LEVEL",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "XLA_FLAGS",
            "JAX_PLATFORM_NAME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "TORCH_HOME",
        )
        names = list(default_names)
        names.extend(
            name
            for name in sorted(self.environment_set)
            if name not in default_names
        )
        names.extend(
            name
            for name in sorted(self.environment_unset)
            if name not in names
        )
        arguments: list[str] = []
        for name in names:
            if name in self.environment_unset:
                # A transient user service can inherit the user manager's
                # environment.  An explicit empty value prevents a variable
                # removed from the Popen environment from reappearing there.
                arguments.append("--setenv=%s=" % name)
            elif name in environment:
                arguments.append("--setenv=%s=%s" % (name, environment[name]))
        return arguments

    def _base_evidence(self, *, available_at_start: int, hard_limit: int, high_limit: int, backend: str) -> JsonDict:
        return {
            "schema_version": 1,
            "kind": "noema.execution_resource_guard",
            "isolation": "subprocess_process_group",
            "backend": backend,
            "hard_limit_enforced": backend == "systemd_cgroup_v2",
            "host_total_memory_bytes": int(_memory_snapshot_bytes().get("MemTotal", 0)),
            "available_at_start_bytes": int(available_at_start),
            "protected_reserve_bytes": int(self.config.reserve_bytes),
            "memory_high_bytes": int(high_limit),
            "memory_max_bytes": int(hard_limit),
            "memory_swap_max_bytes": int(self.config.swap_max_bytes),
            "poll_hz": float(self.config.poll_hz),
            "timeout_seconds": float(self.config.timeout_seconds),
        }

    def _discover_systemd_cgroup_path(self) -> Optional[Path]:
        if self._cgroup_path is not None:
            return self._cgroup_path
        unit = str(self.evidence.get("cgroup_unit") or "")
        if not unit:
            return None
        try:
            completed = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=ControlGroup",
                    "--value",
                ],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        control_group = completed.stdout.strip() if completed.returncode == 0 else ""
        if not control_group:
            return None
        root = Path("/sys/fs/cgroup").resolve()
        candidate = (root / control_group.lstrip("/")).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if not candidate.is_dir():
            return None
        self._cgroup_path = candidate
        return candidate

    def _sample_systemd_cgroup_memory(self) -> None:
        cgroup_path = self._discover_systemd_cgroup_path()
        if cgroup_path is None:
            return
        self._cgroup_memory_sample_count += 1
        kernel_peak = _nonnegative_integer_file(cgroup_path / "memory.peak")
        if kernel_peak is not None:
            self._cgroup_kernel_peak_bytes = max(
                self._cgroup_kernel_peak_bytes,
                kernel_peak,
            )
        current = _nonnegative_integer_file(cgroup_path / "memory.current")
        if current is not None:
            self._cgroup_sampled_current_peak_bytes = max(
                self._cgroup_sampled_current_peak_bytes,
                current,
            )
        for key, value in _key_value_file(cgroup_path / "memory.events").items():
            if isinstance(value, int) and not isinstance(value, bool):
                self._cgroup_memory_events[key] = max(
                    int(self._cgroup_memory_events.get(key) or 0),
                    value,
                )

    def _systemd_unit_evidence(self) -> JsonDict:
        unit = str(self.evidence.get("cgroup_unit") or "")
        if not unit:
            return {}
        try:
            completed = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=Result,MemoryPeak,ControlGroup"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "systemd_inspection_status": "failed",
                "systemd_inspection_error": "%s: %s" % (type(exc).__name__, exc),
            }
        if completed.returncode != 0:
            return {
                "systemd_inspection_status": "failed",
                "systemd_inspection_error": (
                    "systemctl show exited with code %d: %s"
                    % (completed.returncode, completed.stderr.strip() or "no diagnostic")
                ),
            }
        values = {
            name: value
            for line in completed.stdout.splitlines()
            for name, separator, value in [line.partition("=")]
            if separator
        }
        payload: JsonDict = {"systemd_inspection_status": "ok"}
        payload["systemd_result"] = values.get("Result") or None
        post_exit_peak = (
            int(values["MemoryPeak"])
            if str(values.get("MemoryPeak") or "").isdigit()
            else 0
        )
        exact_peak = max(post_exit_peak, self._cgroup_kernel_peak_bytes)
        if exact_peak > 0:
            payload["peak_memory_bytes"] = exact_peak
            payload["peak_memory_source"] = (
                "cgroup_v2_memory.peak_live"
                if self._cgroup_kernel_peak_bytes >= post_exit_peak
                else "systemd_MemoryPeak_post_exit"
            )
        elif self._cgroup_sampled_current_peak_bytes > 0:
            payload["peak_memory_bytes"] = self._cgroup_sampled_current_peak_bytes
            payload["peak_memory_source"] = (
                "cgroup_v2_memory.current_sampled_live_fallback"
            )
        payload["cgroup_memory_sample_count"] = int(
            self._cgroup_memory_sample_count
        )
        payload["cgroup_control_path_observed_live"] = self._cgroup_path is not None
        memory_events = dict(self._cgroup_memory_events)
        if values.get("ControlGroup"):
            events_path = Path("/sys/fs/cgroup") / values["ControlGroup"].strip().lstrip("/") / "memory.events"
            for key, value in _key_value_file(events_path).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    memory_events[key] = max(
                        int(memory_events.get(key) or 0),
                        value,
                    )
        if memory_events:
            payload["memory_events"] = memory_events
        if payload.get("systemd_result") == "success":
            # A successful transient service has no failed state to reset and
            # may already have unloaded by this point. Calling reset-failed on
            # that absent/non-failed unit returns 1 on supported systemd
            # versions despite there being nothing left to clean.
            payload["systemd_cleanup_status"] = "ok"
        else:
            try:
                reset = subprocess.run(
                    ["systemctl", "--user", "reset-failed", unit],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.0,
                    check=False,
                )
                if reset.returncode != 0:
                    payload["systemd_cleanup_status"] = "failed"
                    payload["systemd_cleanup_error"] = (
                        "systemctl reset-failed exited with code %d"
                        % reset.returncode
                    )
                else:
                    payload["systemd_cleanup_status"] = "ok"
            except (OSError, subprocess.TimeoutExpired) as exc:
                payload["systemd_cleanup_status"] = "failed"
                payload["systemd_cleanup_error"] = "%s: %s" % (
                    type(exc).__name__,
                    exc,
                )
        return payload

    def _drain_events(self, final: bool = False) -> None:
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._event_offset)
            chunk = handle.read()
            self._event_offset = handle.tell()
        text = self._event_remainder + chunk
        lines = text.split("\n")
        self._event_remainder = "" if text.endswith("\n") else lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                event = decode_strict_yaml_or_json(line, input_format="json")
            except StructuredInputError as exc:
                raise IsolatedJobError(
                    "Isolated worker emitted malformed event JSON: %s" % exc
                ) from exc
            if not isinstance(event, dict):
                raise IsolatedJobError(
                    "Isolated worker event must contain a JSON object"
                )
            if self.event_sink is not None:
                self.event_sink(event)
        if final and self._event_remainder.strip():
            try:
                event = decode_strict_yaml_or_json(
                    self._event_remainder,
                    input_format="json",
                )
            except StructuredInputError as exc:
                raise IsolatedJobError(
                    "Isolated worker emitted malformed final event JSON: %s" % exc
                ) from exc
            if not isinstance(event, dict):
                raise IsolatedJobError(
                    "Isolated worker final event must contain a JSON object"
                )
            if self.event_sink is not None:
                self.event_sink(event)
            self._event_remainder = ""

    def _terminate_with_grace(self) -> None:
        self._terminate_process_group(signal.SIGTERM)
        if self.process is None:
            return
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._terminate_process_group(getattr(signal, "SIGKILL", signal.SIGTERM))

    def _terminate_process_group(self, sig: signal.Signals) -> None:
        process = self.process
        unit = str(self.evidence.get("cgroup_unit") or "")
        if unit and shutil.which("systemctl"):
            try:
                completed = subprocess.run(
                    ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=%s" % sig.name, unit],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.0,
                    check=False,
                )
                if completed.returncode != 0:
                    self.evidence.setdefault("termination_attempt_errors", []).append(
                        {
                            "backend": "systemd",
                            "signal": sig.name,
                            "error": "systemctl kill exited with code %d"
                            % completed.returncode,
                        }
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.evidence.setdefault("termination_attempt_errors", []).append(
                    {
                        "backend": "systemd",
                        "signal": sig.name,
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    }
                )
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP gives the worker its own console group,
            # but Python cannot reliably deliver SIGTERM to all descendants on
            # Windows. taskkill is the platform process-tree boundary.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.terminate()
                except (ProcessLookupError, OSError):
                    pass
            return
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError, AttributeError):
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json_if_present(path: Path) -> JsonDict:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise IsolatedJobError(
            "Could not read isolated worker status %s: %s" % (path, exc)
        ) from exc
    try:
        value = decode_strict_yaml_or_json(raw_text, input_format="json")
    except StructuredInputError as exc:
        raise IsolatedJobError(
            "Isolated worker status contains invalid JSON: %s" % exc
        ) from exc
    if not isinstance(value, Mapping):
        raise IsolatedJobError(
            "Isolated worker status must contain a JSON object: %s" % path
        )
    return dict(value)


def _meminfo_bytes() -> JsonDict:
    values: JsonDict = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.strip().split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0]) * 1024
    return values


def _memory_snapshot_bytes() -> JsonDict:
    """Return total and available memory using a platform-appropriate source.

    `/proc/meminfo` remains the most accurate Linux source.  psutil, native
    Windows APIs, macOS vm_stat, and POSIX sysconf provide portable fallbacks.
    A missing reading is not guessed: callers fail closed instead of launching
    an unguarded worker.
    """

    values = _meminfo_bytes()
    if int(values.get("MemTotal") or 0) > 0 and int(
        values.get("MemAvailable") or values.get("MemFree") or 0
    ) > 0:
        return values

    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        if int(memory.total) > 0 and int(memory.available) > 0:
            return {
                "MemTotal": int(memory.total),
                "MemAvailable": int(memory.available),
                "source": "psutil",
            }
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return {
                    "MemTotal": int(status.total_phys),
                    "MemAvailable": int(status.avail_phys),
                    "source": "windows_global_memory_status",
                }
        except (AttributeError, OSError, ValueError):
            pass

    if sys.platform == "darwin":
        snapshot = _macos_memory_snapshot_bytes()
        if snapshot:
            return snapshot

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and total_pages > 0 and available_pages > 0:
            return {
                "MemTotal": page_size * total_pages,
                "MemAvailable": page_size * available_pages,
                "source": "posix_sysconf",
            }
    except (AttributeError, OSError, ValueError):
        pass
    return {}


def _macos_memory_snapshot_bytes() -> JsonDict:
    try:
        total_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        vm_result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if total_result.returncode or vm_result.returncode:
        return {}
    try:
        total = int(total_result.stdout.strip())
        first_line = vm_result.stdout.splitlines()[0]
        page_size = int(first_line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
        pages: Dict[str, int] = {}
        for line in vm_result.stdout.splitlines()[1:]:
            key, separator, raw_value = line.partition(":")
            if separator:
                pages[key.strip()] = int(raw_value.strip().rstrip("."))
        available_pages = sum(
            pages.get(name, 0)
            for name in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            )
        )
        available = available_pages * page_size
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    if total <= 0 or available <= 0:
        return {}
    return {
        "MemTotal": total,
        "MemAvailable": min(total, available),
        "source": "macos_vm_stat",
    }


def _available_memory_bytes() -> int:
    values = _memory_snapshot_bytes()
    available = int(values.get("MemAvailable") or values.get("MemFree") or 0)
    if available <= 0:
        raise IsolatedJobError(
            "Could not determine available system memory on %s; install psutil "
            "or configure a supported native memory provider before running jobs"
            % platform.system()
        )
    return available


def _tail_text(path: Path, maximum_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _nonnegative_integer_file(path: Path) -> Optional[int]:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if not value.isdigit():
        return None
    return int(value)


def _key_value_file(path: Path) -> JsonDict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    payload: JsonDict = {}
    for line in lines:
        key, separator, value = line.partition(" ")
        if separator and value.strip().isdigit():
            payload[key] = int(value.strip())
    return payload


def _valid_environment_name(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    first, remainder = value[0], value[1:]
    return (first == "_" or first.isalpha()) and all(
        character == "_" or character.isalnum()
        for character in remainder
    )


def _validated_environment_set(
    value: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise IsolatedJobError("environment_set must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if not _valid_environment_name(raw_name):
            raise IsolatedJobError("environment_set contains an invalid name")
        if not isinstance(raw_value, str) or "\0" in raw_value:
            raise IsolatedJobError(
                "environment_set values must be NUL-free strings"
            )
        result[raw_name] = raw_value
    return result


def _validated_environment_unset(
    value: Iterable[str],
    *,
    forbidden_names: Mapping[str, str],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise IsolatedJobError("environment_unset must be an iterable of names")
    result: list[str] = []
    seen: set[str] = set()
    for raw_name in value:
        if not _valid_environment_name(raw_name):
            raise IsolatedJobError("environment_unset contains an invalid name")
        if raw_name in seen:
            raise IsolatedJobError("environment_unset contains a duplicate name")
        if raw_name in forbidden_names:
            raise IsolatedJobError(
                "an environment variable cannot be both set and unset"
            )
        seen.add(raw_name)
        result.append(raw_name)
    return tuple(result)


def _environment_mib(name: str, default_bytes: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return int(default_bytes)
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise IsolatedJobError("%s must be a finite non-negative MiB value" % name)
    return int(parsed * MIB)


def _optional_environment_mib(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return _environment_mib(name, 0)


def _environment_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise IsolatedJobError("%s must be between %s and %s" % (name, minimum, maximum))
    return value


_EXECUTION_ADMISSION_LOCK = threading.Lock()


@contextmanager
def _execution_admission(
    workspace: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
):
    """Serialize protected workers across UI and CLI processes in one workspace."""

    acquired_thread_lock = False
    while not acquired_thread_lock:
        if cancel_event is not None and cancel_event.is_set():
            raise _AdmissionCanceled()
        if deadline is not None and time.monotonic() >= deadline:
            raise _AdmissionDeadline()
        acquired_thread_lock = _EXECUTION_ADMISSION_LOCK.acquire(timeout=0.1)
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        lock_path = workspace / ".execution-resource-guard.lock"
        with lock_path.open("a+") as handle:
            unlock: Optional[Callable[[], None]] = None
            try:
                import fcntl

                while True:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                        unlock = lambda: fcntl.flock(
                            handle.fileno(), fcntl.LOCK_UN
                        )
                        break
                    except BlockingIOError:
                        _check_admission_wait(cancel_event, deadline)
                        time.sleep(0.1)
            except (ImportError, OSError):
                if os.name == "nt":
                    try:
                        import msvcrt

                        handle.seek(0, os.SEEK_END)
                        if handle.tell() == 0:
                            handle.write("\0")
                            handle.flush()
                        while True:
                            handle.seek(0)
                            try:
                                msvcrt.locking(
                                    handle.fileno(), msvcrt.LK_NBLCK, 1
                                )
                                unlock = lambda: _unlock_windows_file(
                                    handle, msvcrt
                                )
                                break
                            except OSError:
                                _check_admission_wait(cancel_event, deadline)
                                time.sleep(0.1)
                    except (ImportError, OSError):
                        # The process-local lock still provides a safe boundary
                        # for the UI. Cross-process locking is unavailable on
                        # this unusual platform and is recorded by the worker
                        # evidence's portable watchdog backend.
                        unlock = None
            try:
                yield
            finally:
                if unlock is not None:
                    try:
                        unlock()
                    except OSError:
                        pass
    finally:
        _EXECUTION_ADMISSION_LOCK.release()


def _check_admission_wait(
    cancel_event: Optional[threading.Event], deadline: Optional[float]
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _AdmissionCanceled()
    if deadline is not None and time.monotonic() >= deadline:
        raise _AdmissionDeadline()


def _unlock_windows_file(handle: Any, msvcrt: Any) -> None:
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _systemd_user_scope_available() -> bool:
    if shutil.which("systemd-run") is None or not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        return False
    unit = "noema-resource-probe-%d.service" % os.getpid()
    try:
        completed = subprocess.run(
            [
                "systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
                "--unit", unit,
                "--property", "MemoryAccounting=yes",
                "--property", "MemoryHigh=48M",
                "--property", "MemoryMax=64M",
                "--property", "MemorySwapMax=0",
                "--property", "OOMPolicy=kill",
                "--property", "KillMode=control-group",
                "/bin/true",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
