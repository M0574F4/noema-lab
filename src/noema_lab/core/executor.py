from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

from noema_lab.core.artifacts import Artifact, file_sha256
from noema_lab.core.materialization import normalize_materialization_runner
from noema_lab.core.operations import (
    ExecutionCancelled,
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
    normalize_input_metadata_requirements,
    normalize_output_metadata_guarantees,
)
from noema_lab.core.plan_cache import ExecutionPlanCache
from noema_lab.core.planner import (
    DEFAULT_EXECUTION_RUNNER,
    PlannedStep,
    RecipePlanningError,
    validate_recipe_execution_profile,
)
from noema_lab.core.recipes import Recipe, compile_recipe
from noema_lab.core.reproducibility import (
    append_step_to_manifest,
    canonical_json_sha256,
    finalize_manifest,
    initial_run_manifest,
    master_seed_from_recipe,
    seed_namespace_from_recipe,
    utc_now_iso,
)
from noema_lab.core.storage import LocalStore
from noema_lab.core.variants import prepare_compiled_single_run_recipe

JsonDict = Dict[str, object]

MAX_PARALLEL_WORKERS = 32
_DEFAULT_EXECUTION_PLAN_CACHE = ExecutionPlanCache()
_RSS_MEASUREMENT_KEY = "memory.run.rss"
_RSS_MEASUREMENT_BACKEND = "linux_procfs_statm"
_RSS_MEASUREMENT_SOURCE = "/proc/self/statm"


def compile_effective_recipe_for_runner(
    authored_recipe: Recipe,
    registry: OperationRegistry,
    *,
    runner: str,
) -> Recipe:
    """Compile an executable recipe while preserving runner-owned RNG seeds.

    Operation schemas expose numeric seed defaults for UI discoverability and
    standalone calls. During Dataset Capture, however, an omitted operation
    seed means "inherit the capture run's master seed." Default expansion must
    not turn that omission into an explicit, fixed ``seed: 0`` override.
    """

    compilation = compile_recipe(
        authored_recipe,
        mode="strict",
        registry=registry,
    )
    effective_recipe = compilation.require_recipe(effective=True)
    if runner == "dataset_capture":
        _remove_schema_derived_seed_defaults(
            authored_recipe,
            effective_recipe,
            registry,
        )
    return effective_recipe


def _remove_schema_derived_seed_defaults(
    authored_recipe: Recipe,
    effective_recipe: Recipe,
    registry: OperationRegistry,
) -> None:
    authored_steps = {step.id: step for step in authored_recipe.steps}
    for step in effective_recipe.steps:
        authored_step = authored_steps.get(step.id)
        if authored_step is None:
            continue
        if authored_step.params.get("seed") is not None:
            continue
        operation = registry.get(step.op)
        properties = dict(operation.params_schema.get("properties") or {})
        seed_schema = properties.get("seed")
        if isinstance(seed_schema, Mapping) and "default" in seed_schema:
            step.params.pop("seed", None)


class CancellationToken:
    """Thread-safe cooperative cancellation shared by a run and its operations.

    An existing :class:`threading.Event` can be wrapped for compatibility with
    older callers. Cancellation is cooperative: the executor checks the token
    at scheduling boundaries, while long-running operations can call
    ``ctx.raise_if_cancelled()`` at safe interruption points.
    """

    def __init__(self, event: Optional[Event] = None) -> None:
        self._event = event if event is not None else Event()
        self._lock = Lock()
        self._reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> bool:
        """Request cancellation and return whether this was the first request."""

        with self._lock:
            first_request = not self._event.is_set()
            if first_request and reason:
                self._reason = str(reason)
            self._event.set()
            return first_request

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        """Expose the Event spelling for compatibility with existing code."""

        return self.is_cancelled()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason or "Run cancellation requested"

    def raise_if_cancelled(self, message: Optional[str] = None) -> None:
        if self.is_cancelled():
            reason = self.reason
            if message and reason != "Run cancellation requested":
                raise ExecutionCancelled("%s: %s" % (message, reason))
            raise ExecutionCancelled(str(message or reason))


class LocalExecutor:
    def __init__(
        self,
        registry: OperationRegistry,
        store: LocalStore,
        *,
        plan_cache: Optional[ExecutionPlanCache] = None,
    ) -> None:
        if plan_cache is not None and not isinstance(plan_cache, ExecutionPlanCache):
            raise TypeError("plan_cache must be an ExecutionPlanCache")
        self.registry = registry
        self.store = store
        self.plan_cache = plan_cache or _DEFAULT_EXECUTION_PLAN_CACHE

    def run(
        self,
        recipe: Recipe,
        event_sink: Optional[Callable[[JsonDict], None]] = None,
        cancel_event: Optional[Event] = None,
        *,
        cancellation_token: Optional[CancellationToken] = None,
        parallel_workers: int = 1,
        use_plan_cache: bool = True,
        runner: str = DEFAULT_EXECUTION_RUNNER,
        backend: Optional[str] = None,
        implementation: Optional[str] = None,
    ) -> Path:
        worker_count = _validate_parallel_workers(parallel_workers)
        if not isinstance(use_plan_cache, bool):
            raise TypeError("use_plan_cache must be a boolean")
        token = _resolve_cancellation_token(cancel_event, cancellation_token)
        event_sink = _synchronized_event_sink(event_sink)
        execution_config: JsonDict = {
            "mode": "parallel" if worker_count > 1 else "sequential",
            "parallel_workers": worker_count,
        }
        runner_id = normalize_materialization_runner(runner, "runner")
        if runner_id not in {"benchmark_run", "dataset_capture"}:
            raise RecipePlanningError(
                "LocalExecutor cannot execute runner %s through Operation.run; "
                "use the differentiable export compiler for runner=differentiable_export"
                % runner_id
            )
        authored_recipe = recipe
        recipe = compile_effective_recipe_for_runner(
            authored_recipe,
            self.registry,
            runner=runner_id,
        )
        # A declared standard profile is a structural execution contract. If
        # an authored matrix is also unresolved, report the structural defect
        # before the single-run gate so the matrix error does not mask it.
        profile_prevalidated = any(
            field in recipe.metadata for field in ("matrix", "sweeps", "ui_sweeps")
        )
        if profile_prevalidated:
            validate_recipe_execution_profile(recipe)
        recipe = prepare_compiled_single_run_recipe(recipe)
        plan_cache_result = self.plan_cache.plan(
            recipe,
            self.registry,
            runner=runner_id,
            backend=backend,
            implementation=implementation,
            enforce_execution_profile=not profile_prevalidated,
            authored_recipe=authored_recipe,
            use_cache=use_plan_cache,
        )
        execution_plan = plan_cache_result.plan
        plan_cache_evidence = plan_cache_result.evidence.to_dict()
        authored_recipe_sha256 = canonical_json_sha256(authored_recipe.to_dict())
        run_dir = self.store.create_run_dir(recipe.name)
        self.store.write_json(run_dir / "recipe.authored.json", authored_recipe.to_dict())
        self.store.write_json(run_dir / "recipe.json", recipe.to_dict())
        execution_plan_payload = execution_plan.to_dict()
        self.store.write_json(run_dir / "execution-plan.json", execution_plan_payload)
        operation_contracts = execution_plan.operation_contracts_to_dict()
        manifest = initial_run_manifest(recipe, operation_contracts, run_dir.name)
        manifest["recipe"]["authored_sha256"] = authored_recipe_sha256
        manifest["recipe"]["effective_sha256"] = execution_plan.recipe_sha256
        manifest["execution_plan"] = {
            "schema_version": execution_plan.schema_version,
            "kind": execution_plan.kind,
            "path": "execution-plan.json",
            "sha256": execution_plan.sha256,
            "runner": execution_plan.runner,
            "authored_recipe_sha256": authored_recipe_sha256,
            "effective_recipe_sha256": execution_plan.recipe_sha256,
            "steps": [step.to_dict() for step in execution_plan.steps],
            "cache": plan_cache_evidence,
        }
        manifest["execution"] = dict(execution_config)
        master_seed = master_seed_from_recipe(recipe)
        seed_namespace = seed_namespace_from_recipe(recipe)
        produced: Dict[str, Dict[str, Artifact]] = {}
        memory_sampler = _ProcessMemorySampler()
        summary = {
            "schema_version": 1,
            "kind": "noema.run_summary",
            "run_id": run_dir.name,
            "recipe_name": recipe.name,
            "status": "running",
            "created_at_utc": manifest["created_at_utc"],
            "manifest": "manifest.json",
            "recipe_sha256": manifest["recipe"]["sha256"],
            "authored_recipe_sha256": authored_recipe_sha256,
            "effective_recipe_sha256": execution_plan.recipe_sha256,
            "execution_plan": {
                "schema_version": execution_plan.schema_version,
                "path": "execution-plan.json",
                "sha256": execution_plan.sha256,
                "runner": execution_plan.runner,
                "cache": plan_cache_evidence,
            },
            "execution": dict(execution_config),
            "seed_policy": manifest["seed_policy"],
            "steps": [],
            "metrics": {},
            "measurement_evidence": {
                _RSS_MEASUREMENT_KEY: memory_sampler.measurement_record(),
            },
        }
        _write_summary_and_manifest(
            self.store,
            run_dir,
            summary,
            manifest,
        )
        memory_sampler.start()

        try:
            _emit(
                event_sink,
                "run_created",
                "Created run directory",
                run_id=run_dir.name,
                recipe_name=recipe.name,
            )
            _emit(
                event_sink,
                "execution_plan_cache",
                "Execution plan cache %s" % plan_cache_result.evidence.outcome,
                run_id=run_dir.name,
                outcome=plan_cache_result.evidence.outcome,
                enabled=plan_cache_result.evidence.enabled,
                key_sha256=plan_cache_result.evidence.key_sha256,
                plan_sha256=execution_plan.sha256,
            )
            _emit(
                event_sink,
                "run_started",
                "Started recipe execution",
                run_id=run_dir.name,
                execution_mode=execution_config["mode"],
                parallel_workers=worker_count,
            )
            recipe_steps = {step.id: step for step in recipe.steps}
            completed_steps: Dict[str, JsonDict] = {}
            execution_args = {
                "recipe": recipe,
                "recipe_steps": recipe_steps,
                "planned_steps": execution_plan.steps,
                "run_dir": run_dir,
                "master_seed": master_seed,
                "seed_namespace": seed_namespace,
                "token": token,
                "event_sink": event_sink,
                "produced": produced,
                "completed_steps": completed_steps,
                "summary": summary,
                "manifest": manifest,
            }
            if worker_count == 1:
                self._run_sequential(**execution_args)
            else:
                self._run_parallel(parallel_workers=worker_count, **execution_args)
            token.raise_if_cancelled("Run stopped before successful finalization")
            _finalize_memory_measurement(summary, memory_sampler)
            summary["status"] = "completed"
            summary["completed_at_utc"] = utc_now_iso()
            finalize_manifest(manifest, "completed")
            _write_summary_and_manifest(
                self.store,
                run_dir,
                summary,
                manifest,
            )
            _emit(event_sink, "run_completed", "Completed recipe execution", run_id=run_dir.name)
            return run_dir
        except ExecutionCancelled as exc:
            _finalize_memory_measurement(summary, memory_sampler)
            summary["status"] = "canceled"
            summary["error"] = str(exc)
            summary["completed_at_utc"] = utc_now_iso()
            finalize_manifest(manifest, "canceled", str(exc))
            _write_summary_and_manifest(
                self.store,
                run_dir,
                summary,
                manifest,
            )
            _attach_run_failure_identity(exc, run_dir)
            _emit(event_sink, "run_canceled", str(exc), run_id=run_dir.name)
            raise
        except Exception as exc:
            _finalize_memory_measurement(summary, memory_sampler)
            summary["status"] = "failed"
            summary["error"] = str(exc)
            summary["completed_at_utc"] = utc_now_iso()
            finalize_manifest(manifest, "failed", str(exc))
            _write_summary_and_manifest(
                self.store,
                run_dir,
                summary,
                manifest,
            )
            _attach_run_failure_identity(exc, run_dir)
            _emit(event_sink, "run_failed", str(exc), run_id=run_dir.name)
            raise

    def _run_sequential(
        self,
        *,
        recipe: Recipe,
        recipe_steps: Mapping[str, Any],
        planned_steps: Tuple[PlannedStep, ...],
        run_dir: Path,
        master_seed: Optional[int],
        seed_namespace: str,
        token: CancellationToken,
        event_sink: Optional[Callable[[JsonDict], None]],
        produced: Dict[str, Dict[str, Artifact]],
        completed_steps: Dict[str, JsonDict],
        summary: JsonDict,
        manifest: JsonDict,
    ) -> None:
        for planned_step in planned_steps:
            step = recipe_steps[planned_step.step_id]
            token.raise_if_cancelled("Run stopped before step %s" % step.id)
            step_inputs = _resolve_step_inputs(step.inputs, produced)
            try:
                step_summary, outputs, metrics = _execute_planned_step(
                    recipe,
                    step,
                    planned_step,
                    step_inputs,
                    run_dir,
                    master_seed,
                    seed_namespace,
                    token,
                    event_sink,
                )
            except Exception as exc:
                completed_steps[step.id] = _terminal_step_summary(
                    step,
                    planned_step,
                    exc,
                )
                _persist_completed_steps(
                    self.store,
                    run_dir,
                    summary,
                    manifest,
                    planned_steps,
                    completed_steps,
                )
                raise
            produced[step.id] = outputs
            completed_steps[step.id] = step_summary
            _persist_completed_steps(
                self.store,
                run_dir,
                summary,
                manifest,
                planned_steps,
                completed_steps,
            )
            _emit_step_completed(event_sink, run_dir, step, planned_step, metrics)
            token.raise_if_cancelled("Run stopped after step %s" % step.id)

    def _run_parallel(
        self,
        *,
        recipe: Recipe,
        recipe_steps: Mapping[str, Any],
        planned_steps: Tuple[PlannedStep, ...],
        run_dir: Path,
        master_seed: Optional[int],
        seed_namespace: str,
        token: CancellationToken,
        event_sink: Optional[Callable[[JsonDict], None]],
        produced: Dict[str, Dict[str, Artifact]],
        completed_steps: Dict[str, JsonDict],
        summary: JsonDict,
        manifest: JsonDict,
        parallel_workers: int,
    ) -> None:
        plan_order = {
            planned_step.step_id: index
            for index, planned_step in enumerate(planned_steps)
        }
        planned_by_id = {
            planned_step.step_id: planned_step for planned_step in planned_steps
        }
        dependencies = {
            step_id: {
                reference.split(".", 1)[0]
                for reference in recipe_steps[step_id].inputs.values()
            }
            for step_id in plan_order
        }
        unscheduled: Set[str] = set(plan_order)
        completed: Set[str] = set()
        running: Dict[Future[Tuple[JsonDict, Dict[str, Artifact], JsonDict]], str] = {}
        primary_error: Optional[Exception] = None
        cancellation_error: Optional[ExecutionCancelled] = None

        pool = ThreadPoolExecutor(
            max_workers=parallel_workers,
            thread_name_prefix="noema-step",
        )
        try:
            while unscheduled or running:
                if token.is_cancelled() and primary_error is None:
                    cancellation_error = cancellation_error or ExecutionCancelled(
                        token.reason
                    )

                if primary_error is None and cancellation_error is None:
                    ready = sorted(
                        (
                            step_id
                            for step_id in unscheduled
                            if dependencies[step_id].issubset(completed)
                        ),
                        key=plan_order.__getitem__,
                    )
                    capacity = parallel_workers - len(running)
                    admitted = _parallel_admission(
                        ready,
                        running_step_ids=set(running.values()),
                        planned_by_id=planned_by_id,
                        capacity=capacity,
                    )
                    for step_id in admitted:
                        token.raise_if_cancelled(
                            "Run stopped before step %s" % step_id
                        )
                        step = recipe_steps[step_id]
                        step_inputs = _resolve_step_inputs(step.inputs, produced)
                        future = pool.submit(
                            _execute_planned_step,
                            recipe,
                            step,
                            planned_by_id[step_id],
                            step_inputs,
                            run_dir,
                            master_seed,
                            seed_namespace,
                            token,
                            event_sink,
                        )
                        running[future] = step_id
                        unscheduled.remove(step_id)

                if primary_error is not None or cancellation_error is not None:
                    for future in running:
                        future.cancel()

                if not running:
                    if primary_error is not None:
                        raise primary_error
                    if cancellation_error is not None:
                        raise cancellation_error
                    if unscheduled:
                        blocked = ", ".join(sorted(unscheduled, key=plan_order.__getitem__))
                        raise RuntimeError(
                            "Parallel scheduler cannot resolve dependencies for: %s"
                            % blocked
                        )
                    break

                done, _pending = wait(
                    tuple(running),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue

                newly_completed = []
                completed_events = []
                evidence_changed = False
                for future in sorted(done, key=lambda item: plan_order[running[item]]):
                    step_id = running.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        step_summary, outputs, metrics = future.result()
                    except ExecutionCancelled as exc:
                        token.cancel(str(exc))
                        cancellation_error = cancellation_error or exc
                    except Exception as exc:
                        completed_steps[step_id] = _terminal_step_summary(
                            recipe_steps[step_id],
                            planned_by_id[step_id],
                            exc,
                        )
                        evidence_changed = True
                        if primary_error is None:
                            primary_error = exc
                            token.cancel(
                                "Run stopped after step %s failed: %s"
                                % (step_id, exc)
                            )
                    else:
                        produced[step_id] = outputs
                        completed_steps[step_id] = step_summary
                        completed.add(step_id)
                        newly_completed.append(step_id)
                        evidence_changed = True
                        completed_events.append((step_id, metrics))

                if evidence_changed:
                    _persist_completed_steps(
                        self.store,
                        run_dir,
                        summary,
                        manifest,
                        planned_steps,
                        completed_steps,
                    )
                    for step_id, metrics in completed_events:
                        _emit_step_completed(
                            event_sink,
                            run_dir,
                            recipe_steps[step_id],
                            planned_by_id[step_id],
                            metrics,
                        )
        except BaseException as exc:
            # Failures in main-thread evidence persistence or event delivery
            # must also release cooperative workers. Without this boundary,
            # shutdown(wait=True) could wait forever on a peer that has no
            # reason to stop.
            token.cancel("Parallel scheduler stopped: %s" % exc)
            for future in running:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if primary_error is not None:
            raise primary_error
        if cancellation_error is not None:
            raise cancellation_error
        token.raise_if_cancelled("Run stopped before successful finalization")


def _emit(event_sink: Optional[Callable[[JsonDict], None]], kind: str, message: str, **fields: object) -> None:
    if event_sink is not None:
        event_sink({"kind": kind, "message": message, **fields})


def _validate_parallel_workers(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("parallel_workers must be an integer")
    if value < 1 or value > MAX_PARALLEL_WORKERS:
        raise ValueError(
            "parallel_workers must be between 1 and %d" % MAX_PARALLEL_WORKERS
        )
    return value


def _resolve_cancellation_token(
    cancel_event: Optional[Event],
    cancellation_token: Optional[CancellationToken],
) -> CancellationToken:
    if cancel_event is not None and cancellation_token is not None:
        raise ValueError("Pass either cancel_event or cancellation_token, not both")
    if cancellation_token is not None:
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken")
        return cancellation_token
    return CancellationToken(cancel_event)


def _synchronized_event_sink(
    event_sink: Optional[Callable[[JsonDict], None]],
) -> Optional[Callable[[JsonDict], None]]:
    if event_sink is None:
        return None
    lock = Lock()

    def synchronized(payload: JsonDict) -> None:
        with lock:
            event_sink(payload)

    return synchronized


def _step_binding_evidence(planned_step: PlannedStep) -> JsonDict:
    return planned_step.to_dict()


def _resolve_step_inputs(
    references: Mapping[str, str],
    produced: Mapping[str, Mapping[str, Artifact]],
) -> Dict[str, Artifact]:
    step_inputs = {}
    for input_name, reference in references.items():
        source_step, source_output = reference.split(".", 1)
        step_inputs[input_name] = produced[source_step][source_output]
    return step_inputs


def _execute_planned_step(
    recipe: Recipe,
    step: Any,
    planned_step: PlannedStep,
    step_inputs: Mapping[str, Artifact],
    run_dir: Path,
    master_seed: Optional[int],
    seed_namespace: str,
    token: CancellationToken,
    event_sink: Optional[Callable[[JsonDict], None]],
) -> Tuple[JsonDict, Dict[str, Artifact], JsonDict]:
    token.raise_if_cancelled("Run stopped before step %s" % step.id)
    _emit(
        event_sink,
        "step_started",
        "Started %s" % step.id,
        run_id=run_dir.name,
        step_id=step.id,
        op=step.op,
        runner=planned_step.runner,
        backend=planned_step.backend,
        implementation=planned_step.implementation,
    )
    if Path(str(step.id)).name != str(step.id) or str(step.id) in {"", ".", ".."}:
        raise OperationError("Step id is unsafe for artifact storage: %r" % step.id)
    artifacts_dir = run_dir / "artifacts"
    staging_root = artifacts_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    step_dir = Path(
        tempfile.mkdtemp(prefix="%s-" % step.id, dir=str(staging_root))
    )
    input_dir = Path(
        tempfile.mkdtemp(prefix=".inputs-%s-" % step.id, dir=str(staging_root))
    )
    committed = False
    try:
        isolated_inputs = _isolate_step_inputs(
            step_inputs,
            input_dir=input_dir,
            step_id=str(step.id),
        )
    except Exception:
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(step_dir, ignore_errors=True)
        raise
    ctx = OperationContext(
        recipe_name=recipe.name,
        step_id=step.id,
        params=planned_step.params_for_execution(step.params),
        inputs=isolated_inputs,
        run_dir=run_dir,
        step_dir=step_dir,
        progress_sink=event_sink,
        master_seed=master_seed,
        seed_namespace=seed_namespace,
        cancellation_token=token,
    )
    try:
        # A step-started event can itself trigger cancellation. Recheck at the
        # actual operation dispatch boundary so that cancellation never crosses
        # from scheduling/evidence setup into user code unnoticed.
        token.raise_if_cancelled("Run stopped before operation %s" % step.id)
        planned_step.assert_implementation_unchanged()
        _validate_operation_input_metadata(
            isolated_inputs,
            operation=planned_step.operation,
            step_id=str(step.id),
        )
        execution_operation = planned_step.operation.execution_instance()
        try:
            result, timing = _run_operation_with_timing(execution_operation, ctx)
        finally:
            _assert_artifacts_unchanged(
                step_inputs,
                step_id=str(step.id),
                phase="after downstream dispatch",
            )
        validated_outputs = _validate_operation_result(
            result,
            operation=planned_step.operation,
            step_id=str(step.id),
            staging_dir=step_dir,
        )
        committed_outputs = _commit_step_outputs(
            validated_outputs,
            staging_dir=step_dir,
            final_dir=artifacts_dir / str(step.id),
        )
        committed = True
        metrics = dict(result.metrics)
        metrics["timing.step.wall_time_s"] = timing["wall_time_s"]
        metrics.update(timing.get("metrics") or {})
        metadata = dict(result.metadata)
        if timing.get("metadata"):
            metadata["timing"] = timing["metadata"]
        step_summary: JsonDict = {
            "id": step.id,
            "op": step.op,
            "status": "completed",
            "execution_binding": _step_binding_evidence(planned_step),
            "outputs": {
                name: output.to_dict() for name, output in committed_outputs.items()
            },
            "metrics": metrics,
            "metadata": metadata,
        }
        return step_summary, committed_outputs, metrics
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
        if not committed:
            shutil.rmtree(step_dir, ignore_errors=True)


def _terminal_step_summary(
    step: Any,
    planned_step: PlannedStep,
    exc: BaseException,
) -> JsonDict:
    return {
        "id": step.id,
        "op": step.op,
        "status": "canceled" if isinstance(exc, ExecutionCancelled) else "failed",
        "execution_binding": _step_binding_evidence(planned_step),
        "outputs": {},
        "metrics": {},
        "metadata": {},
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _isolate_step_inputs(
    inputs: Mapping[str, Artifact],
    *,
    input_dir: Path,
    step_id: str,
) -> Dict[str, Artifact]:
    """Give a consumer private input files and bind them to committed hashes."""

    _assert_artifacts_unchanged(inputs, step_id=step_id, phase="before dispatch")
    isolated: Dict[str, Artifact] = {}
    for index, (name, source) in enumerate(sorted(inputs.items())):
        slot = input_dir / ("%03d" % index)
        slot.mkdir(parents=True, exist_ok=False)
        destination = slot / source.path.name
        shutil.copy2(source.path, destination)
        copied_sha256 = file_sha256(destination)
        if copied_sha256 != source.sha256:
            raise OperationError(
                "Step %s input %s changed while creating its private dispatch copy"
                % (step_id, name)
            )
        isolated[name] = Artifact(
            kind=source.kind,
            path=destination,
            metadata=dict(source.metadata),
            sha256=copied_sha256,
        )
    return isolated


def _assert_artifacts_unchanged(
    inputs: Mapping[str, Artifact],
    *,
    step_id: str,
    phase: str,
) -> None:
    for name, value in inputs.items():
        try:
            actual_sha256 = file_sha256(value.path)
        except OSError as exc:
            raise OperationError(
                "Step %s input %s is unavailable %s" % (step_id, name, phase)
            ) from exc
        if actual_sha256 != value.sha256:
            raise OperationError(
                "Step %s detected mutation of committed input %s %s"
                % (step_id, name, phase)
            )


def _validate_operation_input_metadata(
    inputs: Mapping[str, Artifact],
    *,
    operation: Operation,
    step_id: str,
) -> None:
    """Enforce consumer metadata requirements against runtime artifacts."""

    declared_inputs = {
        *operation.input_kinds,
        *dict(getattr(operation, "optional_input_kinds", {}) or {}),
    }
    requirements = normalize_input_metadata_requirements(
        getattr(operation, "input_metadata_requirements", None),
        operation.id,
        input_names=declared_inputs,
    )
    for input_name, requirement in requirements.items():
        # An unsupplied optional input has no runtime contract to check.
        if input_name not in inputs:
            continue
        metadata = inputs[input_name].metadata
        all_of = list(requirement.get("all_of") or [])
        missing = [
            path
            for path in all_of
            if not _metadata_path_present(metadata, path)
        ]
        if missing:
            raise OperationError(
                "Step %s input %s is missing required runtime metadata: %s"
                % (
                    step_id,
                    input_name,
                    ", ".join("metadata.%s" % path for path in missing),
                )
            )
        any_of = list(requirement.get("any_of") or [])
        if any_of and not any(
            _metadata_path_present(metadata, path) for path in any_of
        ):
            raise OperationError(
                "Step %s input %s is missing every accepted runtime metadata "
                "alternative: %s"
                % (
                    step_id,
                    input_name,
                    ", ".join("metadata.%s" % path for path in any_of),
                )
            )


def _metadata_path_present(metadata: Mapping[str, Any], path: str) -> bool:
    current: Any = metadata
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return False
        current = current[component]
    return True


def _persist_completed_steps(
    store: LocalStore,
    run_dir: Path,
    summary: JsonDict,
    manifest: JsonDict,
    planned_steps: Tuple[PlannedStep, ...],
    completed_steps: Mapping[str, JsonDict],
) -> None:
    ordered = [
        completed_steps[planned_step.step_id]
        for planned_step in planned_steps
        if planned_step.step_id in completed_steps
    ]
    summary["steps"] = ordered
    manifest["steps"] = []
    manifest["artifacts"] = []
    for step_summary in ordered:
        append_step_to_manifest(manifest, step_summary, run_dir)
        manifest["steps"][-1]["execution_binding"] = step_summary[
            "execution_binding"
        ]
    _write_summary_and_manifest(store, run_dir, summary, manifest)


def _write_summary_and_manifest(
    store: LocalStore,
    run_dir: Path,
    summary: JsonDict,
    manifest: JsonDict,
) -> None:
    """Persist an acyclic content binding from manifest to summary bytes."""

    summary_path = run_dir / "summary.json"
    store.write_json(summary_path, summary)
    summary_metrics = dict(summary.get("metrics") or {})
    measurement_evidence = dict(summary.get("measurement_evidence") or {})
    manifest["summary_evidence"] = {
        "schema_version": 1,
        "kind": "noema.run_summary_evidence",
        "metrics": summary_metrics,
        "metrics_sha256": canonical_json_sha256(summary_metrics),
        "measurement_evidence": measurement_evidence,
        "measurement_evidence_sha256": canonical_json_sha256(
            measurement_evidence
        ),
    }
    manifest["summary"] = {
        "kind": "noema.run_summary",
        "relative_path": "summary.json",
        "sha256": file_sha256(summary_path),
        "size_bytes": int(summary_path.stat().st_size),
    }
    store.write_json(run_dir / "manifest.json", manifest)


def _attach_run_failure_identity(exc: BaseException, run_dir: Path) -> None:
    """Make already-persisted failure evidence discoverable to CLI callers."""

    for name, value in (
        ("noema_run_id", run_dir.name),
        ("noema_run_dir", str(run_dir)),
    ):
        try:
            setattr(exc, name, value)
        except Exception:
            pass


def _emit_step_completed(
    event_sink: Optional[Callable[[JsonDict], None]],
    run_dir: Path,
    step: Any,
    planned_step: PlannedStep,
    metrics: JsonDict,
) -> None:
    _emit(
        event_sink,
        "step_completed",
        "Completed %s" % step.id,
        run_id=run_dir.name,
        step_id=step.id,
        op=step.op,
        runner=planned_step.runner,
        backend=planned_step.backend,
        implementation=planned_step.implementation,
        metrics=metrics,
    )


def _run_operation_with_timing(
    operation: Operation,
    ctx: OperationContext,
) -> Tuple[OperationResult, JsonDict]:
    wall_time_s, result = _timed_run(operation, ctx)
    timing: JsonDict = {
        "wall_time_s": wall_time_s,
        "metadata": {
            "step_wall_time_s": wall_time_s,
            "clock": "perf_counter",
        },
    }
    return result, timing


def _parallel_admission(
    ready: list[str],
    *,
    running_step_ids: Set[str],
    planned_by_id: Mapping[str, PlannedStep],
    capacity: int,
) -> list[str]:
    """Admit thread-safe work in parallel and unsafe work exclusively.

    The first ready unsafe step is a scheduling barrier. This preserves plan
    order for global-RNG/non-reentrant operations instead of relying on lock
    acquisition order inside worker threads.
    """

    if capacity <= 0:
        return []
    running_is_safe = all(
        bool(getattr(planned_by_id[step_id].operation, "thread_safe", False))
        for step_id in running_step_ids
    )
    selected: list[str] = []
    for step_id in ready:
        thread_safe = bool(
            getattr(planned_by_id[step_id].operation, "thread_safe", False)
        )
        if not thread_safe:
            if not running_step_ids and not selected:
                selected.append(step_id)
            break
        if not running_is_safe:
            break
        selected.append(step_id)
        if len(selected) >= capacity:
            break
    return selected


def _validate_operation_result(
    result: Any,
    *,
    operation: Operation,
    step_id: str,
    staging_dir: Path,
) -> Dict[str, Artifact]:
    if not isinstance(result, OperationResult):
        raise OperationError(
            "Step %s operation %s returned %s instead of OperationResult"
            % (step_id, operation.id, type(result).__name__)
        )
    if not isinstance(result.outputs, Mapping):
        raise OperationError("Step %s outputs must be an object" % step_id)
    expected = set(operation.output_kinds)
    actual = set(result.outputs)
    if any(not isinstance(name, str) for name in result.outputs):
        raise OperationError("Step %s output names must be strings" % step_id)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: %s" % ", ".join(missing))
        if extra:
            details.append("unexpected: %s" % ", ".join(extra))
        raise OperationError(
            "Step %s outputs violate operation %s contract (%s)"
            % (step_id, operation.id, "; ".join(details))
        )
    if not isinstance(result.metrics, Mapping) or not isinstance(result.metadata, Mapping):
        raise OperationError("Step %s metrics and metadata must be objects" % step_id)
    # Strict canonical validation rejects non-string keys, non-finite values,
    # and implicit Path/datetime coercion before durable JSON is written.
    canonical_json_sha256(dict(result.metrics))
    canonical_json_sha256(dict(result.metadata))
    output_guarantees = normalize_output_metadata_guarantees(
        getattr(operation, "output_metadata_guarantees", None),
        operation.id,
        output_names=expected,
    )

    root = staging_dir.resolve(strict=True)
    validated: Dict[str, Artifact] = {}
    seen_paths: Set[Path] = set()
    for name in sorted(actual):
        output = result.outputs[name]
        if not isinstance(output, Artifact):
            raise OperationError(
                "Step %s output %s is not an Artifact" % (step_id, name)
            )
        expected_kind = operation.output_kinds[name]
        if output.kind != expected_kind:
            raise OperationError(
                "Step %s output %s kind mismatch: expected %s, got %s"
                % (step_id, name, expected_kind, output.kind)
            )
        if not isinstance(output.path, Path):
            raise OperationError("Step %s output %s path must be a Path" % (step_id, name))
        if output.path.is_symlink():
            raise OperationError("Step %s output %s path must not be a symlink" % (step_id, name))
        try:
            resolved = output.path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise OperationError(
                "Step %s output %s is missing or outside its staging directory: %s"
                % (step_id, name, output.path)
            ) from exc
        if relative == Path(".") or not resolved.is_file():
            raise OperationError("Step %s output %s is not a regular file" % (step_id, name))
        _reject_symlink_components(output.path, root, step_id=step_id, output_name=name)
        if resolved in seen_paths:
            raise OperationError(
                "Step %s outputs alias the same file: %s" % (step_id, resolved)
            )
        seen_paths.add(resolved)
        recorded = str(output.sha256 or "").lower()
        if len(recorded) != 64 or any(char not in "0123456789abcdef" for char in recorded):
            raise OperationError(
                "Step %s output %s is missing a valid SHA-256" % (step_id, name)
            )
        actual_sha256 = file_sha256(resolved)
        if recorded != actual_sha256:
            raise OperationError(
                "Step %s output %s SHA-256 mismatch" % (step_id, name)
            )
        if not isinstance(output.metadata, Mapping):
            raise OperationError(
                "Step %s output %s metadata must be an object" % (step_id, name)
            )
        canonical_json_sha256(dict(output.metadata))
        missing_guarantees = [
            path
            for path in output_guarantees.get(name, [])
            if not _metadata_path_present(output.metadata, path)
        ]
        if missing_guarantees:
            raise OperationError(
                "Step %s output %s violates operation %s metadata guarantees; "
                "missing: %s"
                % (
                    step_id,
                    name,
                    operation.id,
                    ", ".join(
                        "metadata.%s" % path for path in missing_guarantees
                    ),
                )
            )
        validated[name] = Artifact(
            kind=output.kind,
            path=resolved,
            metadata=dict(output.metadata),
            sha256=actual_sha256,
        )
    return validated


def _reject_symlink_components(
    path: Path,
    root: Path,
    *,
    step_id: str,
    output_name: str,
) -> None:
    candidate = Path(os.path.abspath(str(path)))
    while True:
        if candidate.is_symlink():
            raise OperationError(
                "Step %s output %s traverses a symlink: %s"
                % (step_id, output_name, candidate)
            )
        if candidate == root:
            return
        if candidate.parent == candidate:
            raise OperationError(
                "Step %s output %s is not rooted in its staging directory"
                % (step_id, output_name)
            )
        candidate = candidate.parent


def _commit_step_outputs(
    outputs: Mapping[str, Artifact],
    *,
    staging_dir: Path,
    final_dir: Path,
) -> Dict[str, Artifact]:
    if final_dir.exists() or final_dir.is_symlink():
        raise OperationError("Step artifact destination already exists: %s" % final_dir)
    relative_paths = {
        name: output.path.relative_to(staging_dir.resolve(strict=True))
        for name, output in outputs.items()
    }
    os.replace(str(staging_dir), str(final_dir))
    try:
        committed: Dict[str, Artifact] = {}
        for name, output in outputs.items():
            path = (final_dir / relative_paths[name]).resolve(strict=True)
            actual_sha256 = file_sha256(path)
            if actual_sha256 != output.sha256:
                raise OperationError(
                    "Step output %s changed during atomic commit" % name
                )
            committed[name] = Artifact(
                kind=output.kind,
                path=path,
                metadata=dict(output.metadata),
                sha256=actual_sha256,
            )
        return committed
    except Exception:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _timed_run(operation: Operation, ctx: OperationContext) -> Tuple[float, OperationResult]:
    start = time.perf_counter()
    result = operation.run(ctx)
    return time.perf_counter() - start, result


class _ProcessMemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = float(interval_s)
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._stopped = False
        self.start_rss_bytes: Optional[int] = None
        self.peak_rss_bytes: Optional[int] = None
        self.end_rss_bytes: Optional[int] = None
        self.successful_samples = 0
        self.failed_samples = 0
        self.last_error: Optional[JsonDict] = None

    def start(self) -> None:
        self._sample()
        self._thread = Thread(target=self._run, name="noema-memory-sampler", daemon=True)
        self._thread.start()

    def stop_evidence(self) -> Tuple[JsonDict, JsonDict]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval_s * 4))
        self._sample()
        self._stopped = True
        return self.metrics(), self.measurement_record()

    def metrics(self) -> JsonDict:
        if (
            self.start_rss_bytes is None
            or self.peak_rss_bytes is None
            or self.end_rss_bytes is None
        ):
            return {}
        delta = max(0, self.peak_rss_bytes - self.start_rss_bytes)
        return {
            "memory.run.start_rss_bytes": int(self.start_rss_bytes),
            "memory.run.peak_rss_bytes": int(self.peak_rss_bytes),
            "memory.run.end_rss_bytes": int(self.end_rss_bytes),
            "memory.run.peak_rss_delta_bytes": int(delta),
            "memory.run.sampler_interval_s": float(self.interval_s),
        }

    def measurement_record(self) -> JsonDict:
        if not self._stopped:
            status = "pending"
            available: object = None
        elif self.successful_samples == 0:
            status = "unavailable"
            available = False
        elif self.failed_samples:
            status = "partially_measured"
            available = True
        else:
            status = "measured"
            available = True
        return {
            "schema_version": 1,
            "status": status,
            "available": available,
            "backend": _RSS_MEASUREMENT_BACKEND,
            "source": _RSS_MEASUREMENT_SOURCE,
            "sampler_interval_s": float(self.interval_s),
            "successful_samples": int(self.successful_samples),
            "failed_samples": int(self.failed_samples),
            "error": dict(self.last_error) if self.last_error is not None else None,
        }

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        try:
            rss = _current_rss_bytes()
        except Exception as exc:
            self.failed_samples += 1
            self.last_error = {
                "type": type(exc).__name__,
                "message": str(exc) or "RSS measurement failed without an error message",
            }
            return
        self.successful_samples += 1
        if self.start_rss_bytes is None:
            self.start_rss_bytes = rss
            self.peak_rss_bytes = rss
        self.end_rss_bytes = rss
        if self.peak_rss_bytes is None or rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss


def _current_rss_bytes() -> int:
    with open(_RSS_MEASUREMENT_SOURCE, "r", encoding="utf-8") as handle:
        parts = handle.read().strip().split()
    if len(parts) < 2:
        raise ValueError(
            "%s does not contain the resident-page field" % _RSS_MEASUREMENT_SOURCE
        )
    resident_pages = int(parts[1])
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if resident_pages <= 0:
        raise ValueError(
            "%s reports a non-positive resident-page count" % _RSS_MEASUREMENT_SOURCE
        )
    if page_size <= 0:
        raise ValueError("SC_PAGE_SIZE must be positive")
    return resident_pages * page_size


def _finalize_memory_measurement(
    summary: JsonDict,
    sampler: _ProcessMemorySampler,
) -> None:
    metrics, measurement = sampler.stop_evidence()
    summary_metrics = summary.get("metrics")
    if not isinstance(summary_metrics, dict):
        raise TypeError("run summary metrics must be an object")
    summary_metrics.update(metrics)
    measurement_evidence = summary.get("measurement_evidence")
    if not isinstance(measurement_evidence, dict):
        raise TypeError("run summary measurement_evidence must be an object")
    measurement_evidence[_RSS_MEASUREMENT_KEY] = measurement
