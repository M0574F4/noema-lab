from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from noema_lab.core.artifacts import Artifact, artifact
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.external_adapters import (
    ExternalAdapterManifest,
    ExternalAdapterOperationSpec,
    ManifestWrappedOperation,
)
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationError,
    OperationRegistry,
    OperationResult,
    object_schema,
)
from noema_lab.core.planner import RecipePlanningError, plan_recipe
from noema_lab.core.recipes import recipe_from_dict
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.submissions import validate_submission_bundle


class _RuntimeOutputFixture(Operation):
    id = "test.runtime_output_fixture"
    name = "Runtime output fixture"
    output_kinds = {"value": "test.value"}

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def run(self, ctx: OperationContext) -> OperationResult:
        if self.mode == "missing":
            return OperationResult()
        if self.mode == "non_result":
            return {}  # type: ignore[return-value]
        if self.mode == "outside":
            path = ctx.run_dir / "escaped.txt"
        else:
            path = ctx.output_path("value", ".txt")
        if self.mode == "symlink":
            target = ctx.output_path("target", ".txt")
            target.write_text("evidence", encoding="utf-8")
            path.symlink_to(target.name)
        if self.mode != "nonexistent":
            if self.mode != "symlink":
                path.write_text("evidence", encoding="utf-8")
        output = artifact("test.value", path) if path.is_file() else Artifact(
            "test.value", path, sha256="0" * 64
        )
        if self.mode == "wrong_kind":
            output.kind = "test.wrong"
        if self.mode == "stale_digest":
            output.sha256 = "0" * 64
        outputs = {"value": output}
        if self.mode == "extra":
            extra_path = ctx.output_path("extra", ".txt")
            extra_path.write_text("extra", encoding="utf-8")
            outputs["extra"] = artifact("test.extra", extra_path)
        if self.mode == "mutate_self":
            self.output_kinds = {}
        return OperationResult(outputs=outputs)


class _RuntimeSink(Operation):
    id = "test.runtime_sink"
    name = "Runtime sink"
    input_kinds = {"value": ["test.value"]}

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, ctx: OperationContext) -> OperationResult:
        self.calls.append(ctx.step_id)
        return OperationResult()


class _MutatingRuntimeSink(Operation):
    id = "test.mutating_runtime_sink"
    name = "Mutating runtime sink"
    input_kinds = {"value": ["test.value"]}

    def __init__(self, *, bypass_context: bool = False) -> None:
        self.bypass_context = bypass_context

    def run(self, ctx: OperationContext) -> OperationResult:
        observed_input_path = ctx.inputs["value"].path
        target = (
            ctx.run_dir / "artifacts" / "producer" / "value.txt"
            if self.bypass_context
            else ctx.inputs["value"].path
        )
        target.write_text("mutated", encoding="utf-8")
        return OperationResult(metadata={"observed_input_path": str(observed_input_path)})


class _UnsafeOverlapFixture(Operation):
    id = "test.unsafe_overlap_fixture"
    name = "Unsafe overlap fixture"

    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    def run(self, ctx: OperationContext) -> OperationResult:
        lock = self.state["lock"]
        assert isinstance(lock, type(threading.Lock()))
        with lock:
            active = int(self.state["active"])
            self.state["active"] = active + 1
            self.state["peak"] = max(int(self.state["peak"]), active + 1)
        time.sleep(0.025)
        with lock:
            self.state["active"] = int(self.state["active"]) - 1
        return OperationResult()


class _WrappedFixture(Operation):
    id = "test.wrapped_fixture"
    name = "Wrapped fixture"

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


class CoreContractHardeningTests(unittest.TestCase):
    def test_registration_rejects_malformed_and_ambiguous_contracts(self):
        class BadId(_WrappedFixture):
            id = "../bad"

        class BadInputs(_WrappedFixture):
            id = "test.bad_inputs"
            input_kinds = {"value": {"test.value"}}

        class BadOutputs(_WrappedFixture):
            id = "test.bad_outputs"
            output_kinds = {"value": ""}

        class BadSchema(_WrappedFixture):
            id = "test.bad_schema"
            params_schema = {"type": "array"}

        class UnorderedBackends(_WrappedFixture):
            id = "test.unordered_backends"
            backends = {"benchmark_run": {"numpy", "torch"}}

        class Ambiguous(_WrappedFixture):
            id = "test.ambiguous_materialization"
            params_schema = object_schema(
                {"left": {"type": "string"}, "right": {"type": "string"}},
                additional=False,
            )
            materializations = [
                {
                    "runner": "benchmark_run",
                    "backend": "numpy",
                    "implementation": "left",
                    "parameter_bindings": {"left": "x"},
                },
                {
                    "runner": "benchmark_run",
                    "backend": "numpy",
                    "implementation": "right",
                    "parameter_bindings": {"right": "y"},
                },
            ]

        class Nondeterministic(_WrappedFixture):
            id = "test.nondeterministic_contract"

            def __init__(self) -> None:
                self.calls = 0

            def describe(self):
                payload = super().describe()
                self.calls += 1
                payload["nonce"] = self.calls
                return payload

        for operation in (
            BadId(),
            BadInputs(),
            BadOutputs(),
            BadSchema(),
            UnorderedBackends(),
            Ambiguous(),
            Nondeterministic(),
        ):
            with self.subTest(operation=operation.__class__.__name__):
                with self.assertRaises(OperationError):
                    OperationRegistry().register(operation)

    def test_bad_runtime_outputs_fail_atomically_before_downstream_dispatch(self):
        for mode in (
            "missing",
            "extra",
            "wrong_kind",
            "outside",
            "nonexistent",
            "stale_digest",
            "symlink",
            "non_result",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                calls: list[str] = []
                registry = OperationRegistry()
                registry.register(_RuntimeOutputFixture(mode))
                registry.register(_RuntimeSink(calls))
                recipe = recipe_from_dict(
                    {
                        "schema_version": 1,
                        "name": "runtime-output-%s" % mode,
                        "steps": [
                            {"id": "producer", "op": _RuntimeOutputFixture.id},
                            {
                                "id": "consumer",
                                "op": _RuntimeSink.id,
                                "inputs": {"value": "producer.value"},
                            },
                        ],
                    }
                )
                store = LocalStore(Path(tmp))
                with self.assertRaises((OperationError, TypeError, ValueError)):
                    LocalExecutor(registry, store).run(recipe)
                run_dir = store.runs_dir / store.list_runs()[0]["run_id"]
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(calls, [])
                self.assertFalse((run_dir / "artifacts" / "producer").exists())

    def test_dispatch_copy_contains_operation_instance_mutation(self):
        registry = OperationRegistry()
        operation = _RuntimeOutputFixture("mutate_self")
        registry.register(operation)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "dispatch-copy",
                "steps": [{"id": "producer", "op": operation.id}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(recipe)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(operation.output_kinds, {"value": "test.value"})

    def test_downstream_receives_private_input_copy(self):
        registry = OperationRegistry()
        registry.register(_RuntimeOutputFixture("valid"))
        sink = _MutatingRuntimeSink()
        registry.register(sink)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "private-input-copy",
                "steps": [
                    {"id": "producer", "op": _RuntimeOutputFixture.id},
                    {
                        "id": "consumer",
                        "op": sink.id,
                        "inputs": {"value": "producer.value"},
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = LocalExecutor(registry, LocalStore(Path(tmp))).run(recipe)
            committed = run_dir / "artifacts" / "producer" / "value.txt"
            self.assertEqual(committed.read_text(encoding="utf-8"), "evidence")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            observed = Path(summary["steps"][1]["metadata"]["observed_input_path"])
            self.assertNotEqual(observed, committed)
            self.assertFalse(observed.exists())

    def test_direct_committed_input_mutation_fails_the_run(self):
        registry = OperationRegistry()
        registry.register(_RuntimeOutputFixture("valid"))
        registry.register(_MutatingRuntimeSink(bypass_context=True))
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "committed-input-mutation",
                "steps": [
                    {"id": "producer", "op": _RuntimeOutputFixture.id},
                    {
                        "id": "consumer",
                        "op": _MutatingRuntimeSink.id,
                        "inputs": {"value": "producer.value"},
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            with self.assertRaisesRegex(
                OperationError,
                "detected mutation of committed input value",
            ):
                LocalExecutor(registry, store).run(recipe)
            run_dir = store.runs_dir / store.list_runs()[0]["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            failed = [row for row in summary["steps"] if row["status"] == "failed"]
            self.assertEqual([row["id"] for row in failed], ["consumer"])

    def test_planned_contract_mutation_is_rejected_before_dispatch(self):
        registry = OperationRegistry()
        operation = _RuntimeOutputFixture("valid")
        registry.register(operation)
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "planned-contract-mutation",
                "steps": [{"id": "producer", "op": operation.id}],
            }
        )
        planned = plan_recipe(recipe, registry).steps[0]
        operation.name = "Mutated after planning"
        with self.assertRaisesRegex(RecipePlanningError, "contract changed"):
            planned.assert_implementation_unchanged()

    def test_unsafe_operations_are_exclusive_under_parallel_scheduler(self):
        state: dict[str, object] = {
            "lock": threading.Lock(),
            "active": 0,
            "peak": 0,
        }
        registry = OperationRegistry()
        registry.register(_UnsafeOverlapFixture(state))
        recipe = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "unsafe-exclusive",
                "steps": [
                    {"id": "left", "op": _UnsafeOverlapFixture.id},
                    {"id": "right", "op": _UnsafeOverlapFixture.id},
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            LocalExecutor(registry, LocalStore(Path(tmp))).run(
                recipe,
                parallel_workers=2,
            )
        self.assertEqual(state["peak"], 1)

    def test_canonical_json_hashing_rejects_implicit_or_non_json_values(self):
        invalid = (
            {"value": float("nan")},
            {"value": Path("same")},
            {"value": datetime.now(timezone.utc)},
            {1: "integer-key"},
            {"value": (1, 2)},
        )
        for payload in invalid:
            with self.subTest(payload=repr(payload)):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_json_sha256(payload)
        self.assertNotEqual(
            canonical_json_sha256({"value": "2026-01-01"}),
            canonical_json_sha256({"value": "2026-01-02"}),
        )

    def test_external_adapter_callable_is_preflighted_at_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "noema_adapter.yaml"
            manifest_path.write_text("schema_version: 1\nname: bad\n", encoding="utf-8")
            module_path = root / "adapter.py"
            module_path.write_text("def available():\n    return None\n", encoding="utf-8")
            manifest = ExternalAdapterManifest(
                path=manifest_path,
                schema_version=1,
                name="bad",
            )
            spec = ExternalAdapterOperationSpec(
                id="test.bad_adapter",
                name="Bad adapter",
                wraps=_WrappedFixture.id,
                adapter_params={"path": str(module_path), "callable": "missing"},
            )
            with self.assertRaisesRegex(OperationError, "cannot resolve callable"):
                ManifestWrappedOperation(manifest, spec, _WrappedFixture())

    def test_forged_submission_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "bundle"
            result_dir.mkdir()
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "noema.benchmark_result",
                        "benchmark": {"id": "b", "version": "1"},
                        "status": "completed",
                        "recipes": [],
                    }
                ),
                encoding="utf-8",
            )
            submission = {
                "schema_version": 1,
                "comparison_division": "closed",
                "result_bundle": "bundle",
                "method": {"name": "forged", "authors": ["Mallory"]},
                "benchmark": {"id": "b", "version": "1"},
                "runtime": {"hardware": "unknown", "software": "unknown"},
                "reproducibility": {
                    "recipe_sha256": "0" * 64,
                    "seed_policy": "0" * 64,
                    "seeds": [1],
                },
            }
            submission_path = root / "submission.json"
            submission_path.write_text(json.dumps(submission), encoding="utf-8")
            report = validate_submission_bundle(submission_path)
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(report["verdict"], "rejected")
            self.assertEqual(report["verification"]["status"], "invalid")

            submission["result_bundle"] = "../outside"
            submission_path.write_text(json.dumps(submission), encoding="utf-8")
            traversal = validate_submission_bundle(submission_path)
            self.assertEqual(traversal["status"], "invalid")
            self.assertTrue(any("confined" in item for item in traversal["errors"]))


if __name__ == "__main__":
    unittest.main()
