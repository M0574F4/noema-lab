from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from noema_lab.core.artifacts import artifact
from noema_lab.core.executor import (
    MAX_PARALLEL_WORKERS,
    CancellationToken,
    ExecutionCancelled,
    LocalExecutor,
)
from noema_lab.core.external_adapters import (
    ExternalAdapterManifest,
    ExternalAdapterOperationSpec,
    ManifestWrappedOperation,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationRegistry,
    OperationResult,
)
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.storage import LocalStore


class _CallbackSourceOperation(Operation):
    id = "test.concurrent_source"
    name = "Synthetic concurrent source"
    output_kinds = {"value": "test.concurrent_value"}
    thread_safe = True

    def __init__(self, callback):
        self.callback = callback

    def run(self, ctx: OperationContext) -> OperationResult:
        self.callback(ctx)
        path = ctx.output_path("value", ".txt")
        path.write_text(ctx.step_id, encoding="utf-8")
        return OperationResult(outputs={"value": artifact("test.concurrent_value", path)})


class _JoinOperation(Operation):
    id = "test.concurrent_join"
    name = "Synthetic concurrent join"
    input_kinds = {
        "left": ["test.concurrent_value"],
        "right": ["test.concurrent_value"],
    }

    def __init__(self, callback=None):
        self.callback = callback

    def run(self, ctx: OperationContext) -> OperationResult:
        if self.callback is not None:
            self.callback(ctx)
        return OperationResult(metadata={"joined": sorted(ctx.inputs)})


def _registry(source_callback, join_callback=None) -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(_CallbackSourceOperation(source_callback))
    registry.register(_JoinOperation(join_callback))
    return registry


def _independent_recipe(step_ids=("left", "right"), *, include_join=False):
    steps = [
        {"id": step_id, "op": _CallbackSourceOperation.id}
        for step_id in step_ids
    ]
    if include_join:
        steps.append(
            {
                "id": "join",
                "op": _JoinOperation.id,
                "inputs": {
                    "left": "left.value",
                    "right": "right.value",
                },
            }
        )
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": "executor_concurrency_test",
            "steps": steps,
        }
    )


class ExecutorConcurrencyTests(unittest.TestCase):
    def test_parallel_branches_overlap_and_join_waits_for_dependencies(self):
        lock = threading.Lock()
        both_started = threading.Event()
        source_finished = set()
        active = set()

        def source_callback(ctx):
            with lock:
                active.add(ctx.step_id)
                if active == {"left", "right"}:
                    both_started.set()
            if not both_started.wait(2):
                raise AssertionError("independent branches did not overlap")
            if ctx.step_id == "left":
                time.sleep(0.04)
            with lock:
                source_finished.add(ctx.step_id)

        def join_callback(_ctx):
            with lock:
                if source_finished != {"left", "right"}:
                    raise AssertionError("dependent join ran before its inputs completed")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(
                _registry(source_callback, join_callback),
                LocalStore(Path(tmp)),
            ).run(
                _independent_recipe(include_join=True),
                parallel_workers=2,
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([step["id"] for step in summary["steps"]], ["left", "right", "join"])
        self.assertEqual([step["id"] for step in manifest["steps"]], ["left", "right", "join"])
        self.assertEqual(
            summary["execution"],
            {"mode": "parallel", "parallel_workers": 2},
        )
        self.assertEqual(manifest["execution"], summary["execution"])

    def test_parallel_cancellation_stops_admission_and_marks_terminal_evidence(self):
        lock = threading.Lock()
        started = set()
        both_started = threading.Event()
        token = CancellationToken()

        def source_callback(ctx):
            with lock:
                started.add(ctx.step_id)
                if len(started) == 2:
                    both_started.set()
            while True:
                ctx.raise_if_cancelled()
                time.sleep(0.005)

        def cancel_when_running():
            if not both_started.wait(2):
                return
            token.cancel("test cancellation requested")

        cancel_thread = threading.Thread(target=cancel_when_running, daemon=True)
        cancel_thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(ExecutionCancelled, "test cancellation requested"):
                LocalExecutor(_registry(source_callback), store).run(
                    _independent_recipe(("first", "second", "never")),
                    cancellation_token=token,
                    parallel_workers=2,
                )
            cancel_thread.join(timeout=2)
            rows = store.list_runs()
            self.assertEqual(len(rows), 1)
            run_dir = store.runs_dir / rows[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(started, {"first", "second"})
        self.assertEqual(summary["status"], "canceled")
        self.assertEqual(manifest["status"], "canceled")
        self.assertEqual(summary["steps"], [])
        self.assertEqual(summary["error"], "test cancellation requested")

    def test_parallel_failure_cancels_running_and_pending_work_but_stays_failed(self):
        lock = threading.Lock()
        started = set()
        both_started = threading.Event()

        def source_callback(ctx):
            with lock:
                started.add(ctx.step_id)
                if len(started) == 2:
                    both_started.set()
            if not both_started.wait(2):
                raise AssertionError("workers did not start")
            if ctx.step_id == "first":
                raise RuntimeError("synthetic branch failure")
            while True:
                ctx.raise_if_cancelled()
                time.sleep(0.005)

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "synthetic branch failure"):
                LocalExecutor(_registry(source_callback), store).run(
                    _independent_recipe(("first", "second", "never")),
                    parallel_workers=2,
                )
            rows = store.list_runs()
            run_dir = store.runs_dir / rows[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(started, {"first", "second"})
        self.assertEqual(summary["status"], "failed")
        self.assertIn("synthetic branch failure", summary["error"])
        failed = [row for row in summary["steps"] if row["status"] == "failed"]
        self.assertEqual([row["id"] for row in failed], ["first"])
        self.assertIn("synthetic branch failure", failed[0]["error"])

    def test_run_started_event_failure_finalizes_failed_evidence(self):
        def event_sink(event):
            if event.get("kind") == "run_started":
                raise RuntimeError("run-start observer failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "run-start observer failed"):
                LocalExecutor(_registry(lambda _ctx: None), store).run(
                    _independent_recipe(("work",)),
                    event_sink=event_sink,
                )
            run_dir = store.runs_dir / store.list_runs()[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("run-start observer failed", summary["error"])

    def test_scheduler_callback_failure_cancels_cooperative_peer(self):
        both_started = threading.Event()
        cancellation_seen = threading.Event()
        started = set()
        lock = threading.Lock()

        def source_callback(ctx):
            with lock:
                started.add(ctx.step_id)
                if len(started) == 2:
                    both_started.set()
            self.assertTrue(both_started.wait(2))
            if ctx.step_id == "fast":
                return
            try:
                while True:
                    ctx.raise_if_cancelled()
                    time.sleep(0.005)
            except ExecutionCancelled:
                cancellation_seen.set()
                raise

        def event_sink(event):
            if event.get("kind") == "step_completed":
                raise RuntimeError("synthetic event sink failure")

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            started_at = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "synthetic event sink failure"):
                LocalExecutor(_registry(source_callback), store).run(
                    _independent_recipe(("fast", "slow")),
                    event_sink=event_sink,
                    parallel_workers=2,
                )
            elapsed = time.monotonic() - started_at
            run_dir = store.runs_dir / store.list_runs()[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertLess(elapsed, 2)
        self.assertTrue(cancellation_seen.is_set())
        self.assertEqual(summary["status"], "failed")

    def test_default_execution_is_sequential_and_records_one_worker(self):
        calls = []

        def source_callback(ctx):
            calls.append(ctx.step_id)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(
                _registry(source_callback),
                LocalStore(Path(tmp)),
            ).run(_independent_recipe())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(calls, ["left", "right"])
        self.assertEqual(
            summary["execution"],
            {"mode": "sequential", "parallel_workers": 1},
        )

    def test_sequential_cancellation_between_steps_records_completed_work_only(self):
        token = CancellationToken()
        calls = []

        def source_callback(ctx):
            calls.append(ctx.step_id)
            if ctx.step_id == "left":
                token.cancel("stop between steps")

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(ExecutionCancelled, "stop between steps"):
                LocalExecutor(_registry(source_callback), store).run(
                    _independent_recipe(),
                    cancellation_token=token,
                )
            rows = store.list_runs()
            run_dir = store.runs_dir / rows[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(calls, ["left"])
        self.assertEqual(summary["status"], "canceled")
        self.assertEqual([step["id"] for step in summary["steps"]], ["left"])

    def test_cancellation_from_step_started_event_prevents_operation_dispatch(self):
        token = CancellationToken()
        calls = []

        def event_sink(event):
            if event.get("kind") == "step_started":
                token.cancel("cancel at dispatch boundary")

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(ExecutionCancelled, "cancel at dispatch boundary"):
                LocalExecutor(_registry(lambda ctx: calls.append(ctx.step_id)), store).run(
                    _independent_recipe(("work",)),
                    event_sink=event_sink,
                    cancellation_token=token,
                )
            run_dir = store.runs_dir / store.list_runs()[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(calls, [])
        self.assertEqual(summary["status"], "canceled")

    def test_cancellation_token_first_request_is_thread_safe_and_idempotent(self):
        token = CancellationToken()
        barrier = threading.Barrier(8)
        outcomes = []
        lock = threading.Lock()

        def request(index):
            barrier.wait(timeout=2)
            outcome = token.cancel("request-%d" % index)
            with lock:
                outcomes.append((index, outcome))

        threads = [threading.Thread(target=request, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        winners = [index for index, first in outcomes if first]
        self.assertEqual(len(winners), 1)
        self.assertTrue(token.is_cancelled())
        self.assertEqual(token.reason, "request-%d" % winners[0])

    def test_operation_progress_is_a_cooperative_cancellation_checkpoint(self):
        token = CancellationToken()
        events = []
        token.cancel("stop batched operation")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = OperationContext(
                recipe_name="cooperative-cancellation",
                step_id="work",
                params={},
                inputs={},
                run_dir=root,
                step_dir=root / "artifacts" / "work",
                progress_sink=events.append,
                cancellation_token=token,
            )

            with self.assertRaisesRegex(ExecutionCancelled, "stop batched operation"):
                context.report_progress("batch complete", batch=1)

        self.assertEqual(events, [])

    def test_raw_event_context_cancellation_uses_canceled_classification(self):
        cancel_event = threading.Event()
        cancel_event.set()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = OperationContext(
                recipe_name="raw-event-cancellation",
                step_id="work",
                params={},
                inputs={},
                run_dir=root,
                step_dir=root / "artifacts" / "work",
                cancellation_token=cancel_event,
            )

            with self.assertRaises(ExecutionCancelled):
                context.raise_if_cancelled()

    def test_external_adapter_wrapper_propagates_cancellation_token(self):
        observed_tokens = []

        class _Wrapped(Operation):
            id = "test.wrapped_cancellation"
            name = "Wrapped cancellation observer"

            def run(self, ctx):
                observed_tokens.append(ctx.cancellation_token)
                return OperationResult()

        token = CancellationToken()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text(
                "schema_version: 1\nname: cancellation-test\n",
                encoding="utf-8",
            )
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "def observe(value, params):\n    return value\n",
                encoding="utf-8",
            )
            manifest = ExternalAdapterManifest(
                path=manifest_path,
                schema_version=1,
                name="cancellation-test",
            )
            spec = ExternalAdapterOperationSpec(
                id="test.external_cancellation",
                name="External cancellation observer",
                wraps=_Wrapped.id,
                adapter_params={
                    "path": str(adapter_path),
                    "callable": "observe",
                },
            )
            operation = ManifestWrappedOperation(manifest, spec, _Wrapped())
            operation.run(
                OperationContext(
                    recipe_name="external-cancellation",
                    step_id="work",
                    params={},
                    inputs={},
                    run_dir=root,
                    step_dir=root / "artifacts" / "work",
                    cancellation_token=token,
                )
            )

        self.assertEqual(observed_tokens, [token])

    def test_parallel_worker_bounds_fail_before_run_side_effects(self):
        recipe = _independent_recipe(("work",))
        for invalid in (0, MAX_PARALLEL_WORKERS + 1, True, 1.5):
            with self.subTest(parallel_workers=invalid), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                with self.assertRaises((TypeError, ValueError)):
                    LocalExecutor(
                        _registry(lambda _ctx: None),
                        LocalStore(workspace),
                    ).run(recipe, parallel_workers=invalid)
                self.assertEqual(list(workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
