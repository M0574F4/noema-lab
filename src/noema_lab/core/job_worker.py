from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Mapping

from noema_lab.core.benchmarks import load_benchmark_pack, run_benchmark_pack
from noema_lab.core.capture import run_dataset_capture_recipe
from noema_lab.core.execution_controls import (
    executor_options,
    normalize_execution_controls,
    require_strict_lint,
)
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.resource_guard import _atomic_json_write
from noema_lab.core.runtime_policy import (
    apply_deterministic_cpu_runtime_policy,
    canonical_json_sha256,
    observe_deterministic_cpu_runtime_policy,
    runtime_policy_lifecycle_evidence,
    scientific_runtime_identity,
    worker_runtime_identity_evidence,
)
from noema_lab.core.source_closure import verify_executable_source_closure
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import decode_strict_yaml_or_json


JsonDict = Dict[str, Any]
MAX_WORKER_EVENT_LINE_BYTES = 32 * 1024
MAX_WORKER_EVENT_STREAM_BYTES = 1024 * 1024

ISOLATED_BYTECODE_POLICY_V1: JsonDict = {
    "schema_version": 1,
    "kind": "noema.isolated_python_bytecode_policy",
    "policy_id": "isolated-fresh-job-control-pycache-prefix-v1",
    "source_adjacent_bytecode_lookup_redirected": True,
    "bytecode_writes_disabled": True,
}


def isolated_python_bytecode_policy_v1() -> JsonDict:
    return dict(ISOLATED_BYTECODE_POLICY_V1)


def build_registry(adapter_paths):
    """Import the operation registry lazily after runtime-policy bootstrap."""

    from noema_lab.ops import build_registry as _build_registry

    return _build_registry(adapter_paths)


class WorkerEventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._written_bytes = self.path.stat().st_size if self.path.exists() else 0
        self._truncated = False
        self._lock = threading.Lock()

    def __call__(self, event: Mapping[str, Any]) -> None:
        line = (json.dumps(dict(event), sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            if self._truncated:
                return
            if (
                len(line) > MAX_WORKER_EVENT_LINE_BYTES
                or self._written_bytes + len(line) > MAX_WORKER_EVENT_STREAM_BYTES
            ):
                line = (
                    json.dumps(
                        {
                            "kind": "event_stream_truncated",
                            "level": "warning",
                            "message": (
                                "Worker event stream reached its bounded retention limit"
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                self._truncated = True
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
            self._written_bytes += len(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected_file_sha256(
    path: Path,
    expected: object,
    *,
    label: str,
) -> str | None:
    """Optionally verify a parent-prepared input again inside the worker.

    The check is deliberately opt-in so existing worker requests retain their
    behavior.  Guarded benchmark launchers can use it to close the interval
    between parent preflight and worker-side parsing.
    """

    if expected in (None, ""):
        return None
    expected_text = str(expected)
    if len(expected_text) != 64 or any(
        character not in "0123456789abcdef" for character in expected_text
    ):
        raise ValueError("%s expected SHA-256 is malformed" % label)
    observed = _sha256(path)
    if observed != expected_text:
        raise ValueError("%s changed after parent preflight" % label)
    return observed


def _raise_oom_preference() -> None:
    try:
        Path("/proc/self/oom_score_adj").write_text("500\n", encoding="ascii")
    except OSError:
        pass


def _verify_dependency_bindings(
    declared: object,
    *,
    project_root: Path,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("execution dependency bindings must be a nonempty object")
    verified: Dict[str, Dict[str, Any]] = {}
    for raw_name in sorted(declared):
        name = str(raw_name)
        record = declared[raw_name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "size_bytes"}
            or record.get("path") != name
        ):
            raise ValueError("execution dependency binding schema changed")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("execution dependency binding path is unsafe")
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("execution dependency binding is missing or unsafe")
        observed = {
            "path": name,
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        if dict(record) != observed:
            raise ValueError("execution dependency binding changed")
        verified[name] = observed
    return verified


def _bytecode_isolation_observation(
    *,
    project_root: Path,
    status_path: Path,
) -> JsonDict:
    import noema_lab

    expected_prefix = (status_path.parent / "bytecode-cache").resolve()
    observed_prefix_raw = sys.pycache_prefix
    observed_prefix = (
        Path(observed_prefix_raw).resolve()
        if isinstance(observed_prefix_raw, str) and observed_prefix_raw
        else None
    )
    package_source = Path(str(noema_lab.__file__ or "")).resolve()
    expected_source_root = (project_root / "src/noema_lab").resolve()
    try:
        package_source.relative_to(expected_source_root)
        package_source_within_expected_root = True
    except ValueError:
        package_source_within_expected_root = False
    prefix_empty = (
        observed_prefix is not None
        and observed_prefix.is_dir()
        and not observed_prefix.is_symlink()
        and not any(observed_prefix.iterdir())
    )
    observation: JsonDict = {
        "isolated_mode": int(sys.flags.isolated) == 1,
        "ignore_environment": int(sys.flags.ignore_environment) == 1,
        "no_user_site": int(sys.flags.no_user_site) == 1,
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "pycache_prefix_matches_job_control": observed_prefix == expected_prefix,
        "pycache_prefix_relative_to_job_control": (
            "bytecode-cache" if observed_prefix == expected_prefix else None
        ),
        "expected_pycache_prefix_relative_to_job_control": "bytecode-cache",
        "pycache_prefix_confined_to_job_control": (
            observed_prefix == expected_prefix
        ),
        "pycache_prefix_empty": prefix_empty,
        "package_source_within_expected_project_root": (
            package_source_within_expected_root
        ),
    }
    if (
        any(
            observation.get(name) is not True
            for name in (
                "isolated_mode",
                "ignore_environment",
                "no_user_site",
                "dont_write_bytecode",
                "pycache_prefix_matches_job_control",
                "pycache_prefix_confined_to_job_control",
                "pycache_prefix_empty",
                "package_source_within_expected_project_root",
            )
        )
        or observation["pycache_prefix_relative_to_job_control"]
        != "bytecode-cache"
        or observation["expected_pycache_prefix_relative_to_job_control"]
        != "bytecode-cache"
    ):
        raise RuntimeError("isolated worker bytecode policy verification failed")
    return observation


def _bytecode_isolation_evidence(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> JsonDict:
    evidence: JsonDict = {
        "schema_version": 1,
        "kind": "noema.isolated_python_bytecode_policy_evidence",
        "status": "passed",
        "policy_id": "isolated-fresh-job-control-pycache-prefix-v1",
        "pycache_prefix_command_line_bound": True,
        "source_adjacent_bytecode_lookup_redirected": True,
        "prefix_empty_before": bool(initial.get("pycache_prefix_empty")),
        "prefix_empty_after": bool(final.get("pycache_prefix_empty")),
        "initial": dict(initial),
        "final": dict(final),
    }
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def _source_closure_evidence(
    *,
    declared: Mapping[str, Any],
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
    dependencies_pre: Mapping[str, Any],
    dependencies_post: Mapping[str, Any],
) -> JsonDict:
    dependency_expected_sha256 = canonical_json_sha256(dependencies_pre)
    if dependencies_pre != dependencies_post:
        raise RuntimeError("execution dependencies changed during worker execution")
    evidence: JsonDict = {
        "schema_version": 1,
        "kind": "noema.worker_executable_source_closure_evidence",
        "status": "passed",
        "expected_file_set_sha256": str(declared["file_set_sha256"]),
        "pre_registry_file_set_sha256": str(pre["file_set_sha256"]),
        "post_execution_file_set_sha256": str(post["file_set_sha256"]),
        "expected_python_file_set_sha256": str(
            declared["python_file_set_sha256"]
        ),
        "pre_registry_python_file_set_sha256": str(
            pre["python_file_set_sha256"]
        ),
        "post_execution_python_file_set_sha256": str(
            post["python_file_set_sha256"]
        ),
        "expected_dependency_bindings_sha256": dependency_expected_sha256,
        "pre_registry_dependency_bindings_sha256": canonical_json_sha256(
            dependencies_pre
        ),
        "post_execution_dependency_bindings_sha256": canonical_json_sha256(
            dependencies_post
        ),
    }
    if (
        evidence["expected_file_set_sha256"]
        != evidence["pre_registry_file_set_sha256"]
        or evidence["expected_file_set_sha256"]
        != evidence["post_execution_file_set_sha256"]
        or evidence["expected_python_file_set_sha256"]
        != evidence["pre_registry_python_file_set_sha256"]
        or evidence["expected_python_file_set_sha256"]
        != evidence["post_execution_python_file_set_sha256"]
    ):
        raise RuntimeError("worker source closure evidence changed")
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    return evidence


def validate_bytecode_isolation_evidence(value: Mapping[str, Any]) -> JsonDict:
    evidence = dict(value)
    commitment = evidence.pop("evidence_sha256", None)
    if commitment != canonical_json_sha256(evidence):
        raise ValueError("bytecode isolation evidence commitment changed")
    observation_keys = {
        "isolated_mode",
        "ignore_environment",
        "no_user_site",
        "dont_write_bytecode",
        "pycache_prefix_matches_job_control",
        "pycache_prefix_relative_to_job_control",
        "expected_pycache_prefix_relative_to_job_control",
        "pycache_prefix_confined_to_job_control",
        "pycache_prefix_empty",
        "package_source_within_expected_project_root",
    }
    initial = evidence.get("initial")
    final = evidence.get("final")
    if (
        set(evidence)
        != {
            "schema_version",
            "kind",
            "status",
            "policy_id",
            "pycache_prefix_command_line_bound",
            "source_adjacent_bytecode_lookup_redirected",
            "prefix_empty_before",
            "prefix_empty_after",
            "initial",
            "final",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("kind")
        != "noema.isolated_python_bytecode_policy_evidence"
        or evidence.get("status") != "passed"
        or evidence.get("policy_id") != ISOLATED_BYTECODE_POLICY_V1["policy_id"]
        or any(
            evidence.get(name) is not True
            for name in (
                "pycache_prefix_command_line_bound",
                "source_adjacent_bytecode_lookup_redirected",
                "prefix_empty_before",
                "prefix_empty_after",
            )
        )
        or not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or set(initial) != observation_keys
        or set(final) != observation_keys
        or any(
            initial.get(name) is not True or final.get(name) is not True
            for name in observation_keys
            if name
            not in {
                "pycache_prefix_relative_to_job_control",
                "expected_pycache_prefix_relative_to_job_control",
            }
        )
        or initial.get("pycache_prefix_relative_to_job_control")
        != "bytecode-cache"
        or final.get("pycache_prefix_relative_to_job_control")
        != "bytecode-cache"
        or initial.get("expected_pycache_prefix_relative_to_job_control")
        != "bytecode-cache"
        or final.get("expected_pycache_prefix_relative_to_job_control")
        != "bytecode-cache"
    ):
        raise ValueError("bytecode isolation evidence changed")
    return {**evidence, "evidence_sha256": commitment}


def validate_source_closure_evidence(
    value: Mapping[str, Any],
    *,
    declared: Mapping[str, Any],
    dependency_bindings: Mapping[str, Any],
) -> JsonDict:
    evidence = dict(value)
    commitment = evidence.pop("evidence_sha256", None)
    if commitment != canonical_json_sha256(evidence):
        raise ValueError("source closure evidence commitment changed")
    source_digest = str(declared.get("file_set_sha256") or "")
    python_digest = str(declared.get("python_file_set_sha256") or "")
    dependency_digest = canonical_json_sha256(dependency_bindings)
    if (
        set(evidence)
        != {
            "schema_version",
            "kind",
            "status",
            "expected_file_set_sha256",
            "pre_registry_file_set_sha256",
            "post_execution_file_set_sha256",
            "expected_python_file_set_sha256",
            "pre_registry_python_file_set_sha256",
            "post_execution_python_file_set_sha256",
            "expected_dependency_bindings_sha256",
            "pre_registry_dependency_bindings_sha256",
            "post_execution_dependency_bindings_sha256",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("kind")
        != "noema.worker_executable_source_closure_evidence"
        or evidence.get("status") != "passed"
        or any(
            evidence.get(name) != source_digest
            for name in (
                "expected_file_set_sha256",
                "pre_registry_file_set_sha256",
                "post_execution_file_set_sha256",
            )
        )
        or any(
            evidence.get(name) != python_digest
            for name in (
                "expected_python_file_set_sha256",
                "pre_registry_python_file_set_sha256",
                "post_execution_python_file_set_sha256",
            )
        )
        or any(
            evidence.get(name) != dependency_digest
            for name in (
                "expected_dependency_bindings_sha256",
                "pre_registry_dependency_bindings_sha256",
                "post_execution_dependency_bindings_sha256",
            )
        )
    ):
        raise ValueError("source closure evidence changed")
    return {**evidence, "evidence_sha256": commitment}


def run_worker(request_path: Path, status_path: Path, events_path: Path) -> int:
    kind = "unknown"
    try:
        request = decode_strict_yaml_or_json(
            request_path.read_text(encoding="utf-8"),
            input_format="json",
        )
        if not isinstance(request, Mapping):
            raise ValueError("isolated job request must be an object")
        kind = str(request.get("kind") or "")
        workspace = Path(str(request["workspace"])).resolve()
        project_root = Path(str(request["project_root"])).resolve()
        bytecode_policy = request.get("bytecode_isolation_policy")
        runtime_policy = request.get("runtime_policy")
        declared_source_closure = request.get("executable_source_closure")
        declared_dependency_bindings = request.get(
            "execution_dependency_bindings"
        )
        expected_runtime_identity = request.get("expected_runtime_identity")
        protected_values = (
            bytecode_policy,
            runtime_policy,
            declared_source_closure,
            declared_dependency_bindings,
            expected_runtime_identity,
        )
        protected_execution = any(value is not None for value in protected_values)
        if protected_execution and not all(value is not None for value in protected_values):
            raise ValueError("protected worker controls must be supplied as one closed bundle")
        if protected_execution:
            if bytecode_policy != ISOLATED_BYTECODE_POLICY_V1:
                raise ValueError("isolated worker bytecode policy is not the exact v1 policy")
            bytecode_initial = _bytecode_isolation_observation(
                project_root=project_root,
                status_path=status_path,
            )
        else:
            bytecode_initial = None
        if not protected_execution:
            source_closure_pre = None
            dependencies_pre = None
        elif isinstance(declared_source_closure, Mapping):
            source_closure_pre = verify_executable_source_closure(
                declared_source_closure,
                project_root / "src/noema_lab",
            )
            dependencies_pre = _verify_dependency_bindings(
                declared_dependency_bindings,
                project_root=project_root,
            )
        else:
            raise ValueError("isolated job executable source closure is invalid")
        if runtime_policy is None:
            runtime_policy_evidence = None
        elif isinstance(runtime_policy, Mapping):
            runtime_policy_evidence = apply_deterministic_cpu_runtime_policy(
                runtime_policy,
                workspace=workspace,
            )
        else:
            raise ValueError("isolated job runtime_policy must be an object")
        if protected_execution:
            if not isinstance(expected_runtime_identity, Mapping):
                raise ValueError("expected worker runtime identity is invalid")
            runtime_identity_pre = scientific_runtime_identity()
            if runtime_identity_pre != dict(expected_runtime_identity):
                raise RuntimeError("pre-registry worker runtime identity changed")
        else:
            runtime_identity_pre = None
        _raise_oom_preference()
        adapter_paths = [str(value) for value in request.get("adapter_paths") or []]
        if kind == "resource_accounting_probe":
            registry = None
        else:
            # Runtime policy setup deliberately precedes registry import.  The
            # registry imports the optional PHY modules whose Torch/Sionna
            # defaults the policy freezes.
            registry = build_registry(adapter_paths)
        store = None if kind == "resource_accounting_probe" else LocalStore(workspace)
        events = WorkerEventWriter(events_path)
        running_status: JsonDict = {
            "status": "running",
            "kind": kind,
            "pid": os.getpid(),
        }
        if runtime_policy_evidence is not None:
            running_status["runtime_policy_evidence"] = runtime_policy_evidence
        _atomic_json_write(status_path, running_status)
        if kind == "resource_accounting_probe":
            allocation_bytes = int(request.get("allocation_bytes") or 0)
            if allocation_bytes < 1024 * 1024 or allocation_bytes > 32 * 1024 * 1024:
                raise ValueError("resource-accounting probe allocation is out of bounds")
            hold_seconds = float(request.get("hold_seconds") or 0.0)
            if hold_seconds < 0.0 or hold_seconds > 5.0:
                raise ValueError("resource-accounting probe hold is out of bounds")
            allocation = bytearray(allocation_bytes)
            for offset in range(0, allocation_bytes, 4096):
                allocation[offset] = 1
            if hold_seconds:
                time.sleep(hold_seconds)
            payload = {
                "status": "completed",
                "kind": kind,
                "allocation_bytes": allocation_bytes,
                "pages_touched": (allocation_bytes + 4095) // 4096,
                "hold_seconds": hold_seconds,
            }
            del allocation
        elif kind == "recipe":
            recipe = recipe_from_dict(dict(request["recipe"]))
            execution = normalize_execution_controls(
                request.get("execution")
            )
            if execution["strict_lint"]:
                require_strict_lint(recipe, registry, context=recipe.name)

            def recipe_event(event: JsonDict) -> None:
                events(event)
                if event.get("kind") == "run_created" and event.get("run_id"):
                    _atomic_json_write(
                        status_path,
                        {
                            "status": "running",
                            "kind": kind,
                            "pid": os.getpid(),
                            "run_id": str(event["run_id"]),
                        },
                    )

            run_dir = LocalExecutor(registry, store).run(
                recipe,
                event_sink=recipe_event,
                **executor_options(execution),
            )
            payload = {
                "status": "completed",
                "kind": kind,
                "run_id": run_dir.name,
                "summary_path": str(run_dir / "summary.json"),
            }
        elif kind == "benchmark":
            pack_path = Path(str(request["pack_path"])).resolve()
            verified_pack_sha256 = _verify_expected_file_sha256(
                pack_path,
                request.get("expected_pack_file_sha256"),
                label="benchmark pack",
            )
            pack = load_benchmark_pack(pack_path)
            execution = normalize_execution_controls(
                request.get("execution")
            )

            def benchmark_event(event: JsonDict) -> None:
                if protected_execution:
                    if event.get("kind") == "benchmark_backing_run_pruned":
                        events(
                            {
                                "kind": "benchmark_backing_run_pruned",
                                "entry_id": str(event.get("entry_id") or ""),
                            }
                        )
                    elif event.get("kind") == "benchmark_created":
                        events(
                            {
                                "kind": "benchmark_created",
                                "result_id": str(event.get("result_id") or ""),
                            }
                        )
                else:
                    events(event)
                if event.get("kind") == "benchmark_created" and event.get("result_id"):
                    _atomic_json_write(
                        status_path,
                        {
                            "status": "running",
                            "kind": kind,
                            "pid": os.getpid(),
                            "result_id": str(event["result_id"]),
                        },
                    )

            result_dir = run_benchmark_pack(
                pack,
                registry,
                store,
                project_root,
                event_sink=benchmark_event,
                resume_result_id=(
                    str(request["resume_result_id"])
                    if request.get("resume_result_id") not in (None, "")
                    else None
                ),
                retain_backing_runs=bool(
                    request.get("retain_backing_runs", False)
                ),
                **execution,
            )
            result = store.get_benchmark_result(result_dir.name)
            recipe_rows = result.get("recipes")
            if not isinstance(recipe_rows, list):
                raise ValueError("benchmark result lacks a structural recipe roster")
            structural_recipes = []
            for row in recipe_rows:
                if not isinstance(row, Mapping):
                    raise ValueError("benchmark result structural roster is invalid")
                structural_recipes.append(
                    {
                        "id": str(row.get("id") or ""),
                        "status": str(row.get("status") or ""),
                    }
                )
            payload = {
                "status": "completed",
                "kind": kind,
                "result_id": result_dir.name,
                "result_path": str(result_dir / "result.json"),
                "result_status": str(result.get("status") or ""),
                "execution": dict(execution),
                "structural_recipes": structural_recipes,
            }
            if verified_pack_sha256 is not None:
                payload["verified_pack_file_sha256"] = verified_pack_sha256
        elif kind == "dataset_capture":
            recipe = recipe_from_dict(dict(request["recipe"]))
            out_dir = Path(str(request["out_dir"])).resolve()

            def capture_progress(progress: JsonDict) -> None:
                events({"kind": "capture_progress", **dict(progress)})

            def capture_event(event: JsonDict) -> None:
                events(event)
                if event.get("kind") == "run_created" and event.get("run_id"):
                    _atomic_json_write(
                        status_path,
                        {
                            "status": "running",
                            "kind": kind,
                            "pid": os.getpid(),
                            "run_id": str(event["run_id"]),
                        },
                    )

            capture = run_dataset_capture_recipe(
                recipe,
                registry,
                store,
                out_dir,
                force=bool(request.get("force", False)),
                event_sink=capture_event,
                progress_sink=capture_progress,
            )
            payload = {
                "status": "completed",
                "kind": kind,
                "dataset_capture": dict(capture),
            }
        else:
            raise ValueError("unknown isolated job kind: %s" % kind)
        if runtime_policy_evidence is not None:
            runtime_policy_final_evidence = (
                observe_deterministic_cpu_runtime_policy(
                    runtime_policy,
                    workspace=workspace,
                    observation_phase="worker_final",
                    prior_evidence_sha256=runtime_policy_evidence[
                        "evidence_sha256"
                    ],
                )
            )
            payload["runtime_policy_evidence"] = (
                runtime_policy_lifecycle_evidence(
                    runtime_policy_evidence,
                    runtime_policy_final_evidence,
                )
            )
        if bytecode_initial is not None:
            bytecode_final = _bytecode_isolation_observation(
                project_root=project_root,
                status_path=status_path,
            )
            payload["bytecode_isolation_evidence"] = _bytecode_isolation_evidence(
                bytecode_initial,
                bytecode_final,
            )
        if source_closure_pre is not None and dependencies_pre is not None:
            source_closure_post = verify_executable_source_closure(
                declared_source_closure,
                project_root / "src/noema_lab",
            )
            dependencies_post = _verify_dependency_bindings(
                declared_dependency_bindings,
                project_root=project_root,
            )
            payload["source_closure_evidence"] = _source_closure_evidence(
                declared=declared_source_closure,
                pre=source_closure_pre,
                post=source_closure_post,
                dependencies_pre=dependencies_pre,
                dependencies_post=dependencies_post,
            )
            runtime_identity_post = scientific_runtime_identity()
            payload["runtime_identity_evidence"] = (
                worker_runtime_identity_evidence(
                    expected_runtime_identity,
                    pre_registry=runtime_identity_pre,
                    post_execution=runtime_identity_post,
                )
            )
        _atomic_json_write(status_path, payload)
        return 0
    except MemoryError as exc:
        _atomic_json_write(status_path, {"status": "resource_exhausted", "kind": kind, "error": str(exc) or "Memory allocation failed"})
        return 75
    except BaseException as exc:
        _atomic_json_write(status_path, {"status": "failed", "kind": kind, "error": "%s: %s" % (type(exc).__name__, exc)})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    return run_worker(Path(args.request), Path(args.status), Path(args.events))


if __name__ == "__main__":
    raise SystemExit(main())
