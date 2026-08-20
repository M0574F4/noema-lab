from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import gc
from threading import Barrier, Event, Lock
import time
import unittest
from unittest import mock
import weakref

import noema_lab.core.plan_cache as plan_cache_module
from noema_lab.core.operations import (
    Operation,
    OperationContext,
    OperationRegistry,
    OperationResult,
    object_schema,
)
from noema_lab.core.plan_cache import (
    EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION,
    ExecutionPlanCache,
    build_execution_plan_cache_key,
)
from noema_lab.core.planner import RecipePlanningError
from noema_lab.core.recipes import Recipe, recipe_from_dict


class _CacheOperation(Operation):
    id = "test.plan_cache"
    name = "Plan cache operation"
    backends = {
        "benchmark_run": ["numpy", "python"],
        "dataset_capture": ["numpy"],
        "differentiable_export": [],
    }
    materializations = [
        {
            "runner": "benchmark_run",
            "backend": "numpy",
            "implementation": "numpy_reference",
        },
        {
            "runner": "benchmark_run",
            "backend": "python",
            "implementation": "python_reference",
        },
        {
            "runner": "dataset_capture",
            "backend": "numpy",
            "implementation": "numpy_capture",
        },
    ]
    params_schema = object_schema(
        {"count": {"type": "integer", "minimum": 1, "default": 1}}
    )

    def __init__(self, instance_name: str = "default") -> None:
        self.instance_name = instance_name

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult(metadata={"instance_name": self.instance_name})


class _UnreferencedOperation(Operation):
    id = "test.plan_cache_unreferenced"
    name = "Unreferenced cache operation"

    def run(self, ctx: OperationContext) -> OperationResult:
        return OperationResult()


def _registry(operation: Operation | None = None) -> OperationRegistry:
    registry = OperationRegistry()
    registry.register(operation or _CacheOperation())
    return registry


def _recipe(count: int = 1, *, name: str = "plan_cache") -> Recipe:
    return recipe_from_dict(
        {
            "schema_version": 1,
            "name": name,
            "steps": [
                {
                    "id": "work",
                    "op": _CacheOperation.id,
                    "params": {"count": count},
                }
            ],
        }
    )


class ExecutionPlanCacheTests(unittest.TestCase):
    def test_repeated_plan_is_a_hit_and_returns_immutable_evidence(self):
        cache = ExecutionPlanCache(max_entries=4)
        registry = _registry()
        recipe = _recipe()

        first = cache.plan(recipe, registry)
        second = cache.plan(recipe, registry)

        self.assertEqual(first.evidence.outcome, "miss")
        self.assertEqual(second.evidence.outcome, "hit")
        self.assertIs(first.plan, second.plan)
        self.assertEqual(first.evidence.key_sha256, second.evidence.key_sha256)
        self.assertEqual(cache.stats().hits, 1)
        self.assertEqual(cache.stats().misses, 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            second.evidence.outcome = "miss"
        with self.assertRaises(TypeError):
            second.plan.steps[0].implementation_metadata["status"] = "changed"

    def test_key_covers_every_planner_selector_and_schema_contract(self):
        registry = _registry()
        recipe = _recipe()
        base = build_execution_plan_cache_key(recipe, registry)

        keys = {
            build_execution_plan_cache_key(
                recipe, registry, runner="dataset_capture"
            ).sha256,
            build_execution_plan_cache_key(
                recipe, registry, backend="python"
            ).sha256,
            build_execution_plan_cache_key(
                recipe, registry, implementation="numpy_reference"
            ).sha256,
            build_execution_plan_cache_key(
                recipe, registry, enforce_execution_profile=False
            ).sha256,
            build_execution_plan_cache_key(_recipe(2), registry).sha256,
        }

        self.assertEqual(base.schema_version, EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION)
        self.assertNotIn(base.sha256, keys)
        self.assertEqual(len(keys), 5)
        self.assertEqual(base.runner, "benchmark_run")

    def test_contract_and_unreferenced_registry_changes_invalidate_key(self):
        operation = _CacheOperation()
        registry = _registry(operation)
        recipe = _recipe()
        original = build_execution_plan_cache_key(recipe, registry)

        operation.name = "Changed operation contract"
        changed_contract = build_execution_plan_cache_key(recipe, registry)
        registry.register(_UnreferencedOperation())
        changed_registry = build_execution_plan_cache_key(recipe, registry)

        self.assertNotEqual(original.operation_registry_sha256, changed_contract.operation_registry_sha256)
        self.assertNotEqual(original.sha256, changed_contract.sha256)
        self.assertNotEqual(changed_contract.sha256, changed_registry.sha256)

    def test_implementation_source_drift_forces_a_cache_miss(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        recipe = _recipe()
        real_identity = plan_cache_module._operation_source_identity
        revision = {"value": "first"}

        def source_identity(operation):
            payload = dict(real_identity(operation))
            payload["source_sha256"] = revision["value"]
            return payload

        with mock.patch.object(
            plan_cache_module,
            "_operation_source_identity",
            side_effect=source_identity,
        ):
            first = cache.plan(recipe, registry)
            revision["value"] = "second"
            second = cache.plan(recipe, registry)

        self.assertEqual(first.evidence.outcome, "miss")
        self.assertEqual(second.evidence.outcome, "miss")
        self.assertNotEqual(
            first.evidence.operation_registry_sha256,
            second.evidence.operation_registry_sha256,
        )
        self.assertIsNot(first.plan, second.plan)

    def test_validation_catalog_changes_invalidate_key(self):
        class _Catalog:
            def __init__(self, revision):
                self.revision = revision

            def to_dict(self):
                return {"schema_version": 1, "revision": self.revision}

        registry = _registry()
        recipe = _recipe()
        with mock.patch.object(
            plan_cache_module,
            "load_research_catalog",
            return_value=_Catalog("first"),
        ):
            first = build_execution_plan_cache_key(recipe, registry)
        with mock.patch.object(
            plan_cache_module,
            "load_research_catalog",
            return_value=_Catalog("second"),
        ):
            second = build_execution_plan_cache_key(recipe, registry)

        self.assertNotEqual(
            first.planner_validation_sha256,
            second.planner_validation_sha256,
        )
        self.assertNotEqual(first.sha256, second.sha256)

    def test_authored_provenance_does_not_reduce_effective_recipe_hits(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        effective = _recipe()
        authored_a = _recipe(name="authored_a")
        authored_b = _recipe(name="authored_b")

        first = cache.plan(
            effective,
            registry,
            authored_recipe=authored_a,
        )
        second = cache.plan(
            effective,
            registry,
            authored_recipe=authored_b,
        )

        self.assertEqual(first.evidence.outcome, "miss")
        self.assertEqual(second.evidence.outcome, "hit")
        self.assertEqual(first.evidence.key_sha256, second.evidence.key_sha256)
        self.assertNotEqual(
            first.evidence.authored_recipe_sha256,
            second.evidence.authored_recipe_sha256,
        )

    def test_cache_is_privately_partitioned_by_registry_instance(self):
        first_operation = _CacheOperation("registry_one")
        second_operation = _CacheOperation("registry_two")
        first_registry = _registry(first_operation)
        second_registry = _registry(second_operation)
        recipe = _recipe()
        first_key = build_execution_plan_cache_key(recipe, first_registry)
        second_key = build_execution_plan_cache_key(recipe, second_registry)
        cache = ExecutionPlanCache()

        first = cache.plan(recipe, first_registry)
        second = cache.plan(recipe, second_registry)

        self.assertEqual(first_key.sha256, second_key.sha256)
        self.assertEqual(first.evidence.outcome, "miss")
        self.assertEqual(second.evidence.outcome, "miss")
        self.assertIs(first.plan.steps[0].operation, first_operation)
        self.assertIs(second.plan.steps[0].operation, second_operation)

    def test_cache_does_not_retain_the_whole_registry(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        registry_reference = weakref.ref(registry)
        cache.plan(_recipe(), registry)

        del registry
        gc.collect()

        self.assertIsNone(registry_reference())

    def test_capacity_is_bounded_and_uses_lru_eviction(self):
        cache = ExecutionPlanCache(max_entries=2)
        registry = _registry()

        cache.plan(_recipe(1), registry)
        cache.plan(_recipe(2), registry)
        cache.plan(_recipe(2), registry)  # Make count=2 the most recent entry.
        cache.plan(_recipe(3), registry)
        replay = cache.plan(_recipe(1), registry)

        stats = cache.stats()
        self.assertEqual(replay.evidence.outcome, "miss")
        self.assertEqual(stats.entry_count, 2)
        self.assertEqual(stats.max_entries, 2)
        self.assertEqual(stats.evictions, 2)

    def test_bypass_and_invalidation_are_explicit(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        recipe = _recipe()

        bypassed = cache.plan(recipe, registry, use_cache=False)
        self.assertEqual(bypassed.evidence.outcome, "bypass")
        self.assertFalse(bypassed.evidence.enabled)
        self.assertEqual(cache.stats().entry_count, 0)

        stored = cache.plan(recipe, registry)
        self.assertEqual(stored.evidence.outcome, "miss")
        self.assertEqual(
            cache.invalidate(key_sha256=stored.evidence.key_sha256),
            1,
        )
        self.assertEqual(cache.plan(recipe, registry).evidence.outcome, "miss")
        self.assertEqual(cache.clear(), 1)
        self.assertEqual(cache.stats().entry_count, 0)

    def test_planning_failures_are_not_cached(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        invalid = recipe_from_dict(
            {
                "schema_version": 1,
                "name": "invalid_cache_recipe",
                "steps": [{"id": "missing", "op": "test.unknown"}],
            }
        )

        for _ in range(2):
            with self.assertRaises(RecipePlanningError):
                cache.plan(invalid, registry)

        self.assertEqual(cache.stats().misses, 2)
        self.assertEqual(cache.stats().entry_count, 0)

    def test_concurrent_same_key_callers_share_one_planning_operation(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        recipe = _recipe()
        started = Event()
        release = Event()
        count_lock = Lock()
        call_count = 0
        real_plan_recipe = plan_cache_module.plan_recipe

        def controlled_plan(*args, **kwargs):
            nonlocal call_count
            with count_lock:
                call_count += 1
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return real_plan_recipe(*args, **kwargs)

        with mock.patch.object(plan_cache_module, "plan_recipe", controlled_plan):
            with ThreadPoolExecutor(max_workers=8) as pool:
                first = pool.submit(cache.plan, recipe, registry)
                self.assertTrue(started.wait(timeout=5))
                remaining = [
                    pool.submit(cache.plan, recipe, registry) for _ in range(7)
                ]
                release.set()
                results = [first.result(timeout=5)] + [
                    item.result(timeout=5) for item in remaining
                ]

        self.assertEqual(call_count, 1)
        self.assertEqual(
            sorted(result.evidence.outcome for result in results),
            ["hit"] * 7 + ["miss"],
        )
        self.assertTrue(all(result.plan is results[0].plan for result in results))

    def test_concurrent_same_key_failure_is_shared_but_not_cached(self):
        cache = ExecutionPlanCache()
        registry = _registry()
        recipe = _recipe()
        started = Event()
        release = Event()
        barrier = Barrier(8)
        count_lock = Lock()
        call_count = 0

        def controlled_failure(*_args, **_kwargs):
            nonlocal call_count
            with count_lock:
                call_count += 1
            started.set()
            self.assertTrue(release.wait(timeout=5))
            raise RecipePlanningError("synthetic shared planning failure")

        def request_plan():
            barrier.wait(timeout=5)
            return cache.plan(recipe, registry)

        with mock.patch.object(plan_cache_module, "plan_recipe", controlled_failure):
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(request_plan) for _ in range(8)]
                self.assertTrue(started.wait(timeout=5))
                time.sleep(0.05)
                release.set()
                for future in futures:
                    with self.assertRaisesRegex(
                        RecipePlanningError,
                        "synthetic shared planning failure",
                    ):
                        future.result(timeout=5)

        self.assertEqual(call_count, 1)
        self.assertEqual(cache.stats().entry_count, 0)

    def test_configuration_and_input_types_are_checked(self):
        with self.assertRaises(TypeError):
            ExecutionPlanCache(max_entries=True)
        with self.assertRaises(ValueError):
            ExecutionPlanCache(max_entries=0)

        cache = ExecutionPlanCache()
        with self.assertRaises(TypeError):
            cache.plan(_recipe(), _registry(), use_cache=1)
        with self.assertRaises(TypeError):
            build_execution_plan_cache_key(
                _recipe(), _registry(), implementation=42
            )


if __name__ == "__main__":
    unittest.main()
