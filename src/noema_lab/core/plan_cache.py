"""Bounded, deterministic caching for immutable execution plans.

The public cache key is semantic and reproducible.  The private lookup slot is
also partitioned by the concrete :class:`OperationRegistry` instance because
an :class:`ExecutionPlan` holds executable ``Operation`` objects in addition
to its serialized contract.  This prevents an otherwise identical registry
from receiving operation objects owned by an older registry.

Cache-key schema versions must be incremented whenever planner resolution
semantics change without a corresponding execution-plan schema change.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, RLock
from typing import Any, Dict, Literal, Optional, Tuple
from weakref import ReferenceType, ref

from noema_lab.core.execution_profiles import execution_profile_catalog
from noema_lab.core.materialization import (
    normalize_materialization_backend,
    normalize_materialization_runner,
)
from noema_lab.core.operations import OperationError, OperationRegistry
from noema_lab.core.planner import (
    DEFAULT_EXECUTION_RUNNER,
    EXECUTION_PLAN_KIND,
    EXECUTION_PLAN_SCHEMA_VERSION,
    PLANNED_STEP_SCHEMA_VERSION,
    ExecutionPlan,
    RecipePlanningError,
    _operation_source_identity,
    plan_recipe,
    validate_execution_plan_runtime_availability,
)
from noema_lab.core.recipes import Recipe
from noema_lab.core.research_catalog import load_research_catalog
from noema_lab.core.reproducibility import canonical_json_sha256, recipe_fingerprint

JsonDict = Dict[str, Any]
CacheOutcome = Literal["hit", "miss", "bypass"]

EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION = 2
EXECUTION_PLAN_CACHE_EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_EXECUTION_PLAN_CACHE_MAX_ENTRIES = 128


@dataclass(frozen=True)
class ExecutionPlanCacheKey:
    """Deterministic identity of every semantic input to plan resolution."""

    sha256: str
    effective_recipe_sha256: str
    operation_registry_sha256: str
    planner_validation_sha256: str
    runner: str
    backend: Optional[str]
    implementation: Optional[str]
    enforce_execution_profile: bool
    schema_version: int = EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "effective_recipe_sha256": self.effective_recipe_sha256,
            "operation_registry_sha256": self.operation_registry_sha256,
            "planner_validation_sha256": self.planner_validation_sha256,
            "runner": self.runner,
            "backend": self.backend,
            "implementation": self.implementation,
            "enforce_execution_profile": self.enforce_execution_profile,
        }


@dataclass(frozen=True)
class ExecutionPlanCacheEvidence:
    """Small serializable record suitable for run manifests and events."""

    outcome: CacheOutcome
    key_sha256: str
    key_schema_version: int
    plan_sha256: str
    effective_recipe_sha256: str
    authored_recipe_sha256: Optional[str]
    operation_registry_sha256: str
    planner_validation_sha256: str
    entry_count: int
    max_entries: int
    schema_version: int = EXECUTION_PLAN_CACHE_EVIDENCE_SCHEMA_VERSION

    @property
    def enabled(self) -> bool:
        return self.outcome != "bypass"

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "outcome": self.outcome,
            "key_sha256": self.key_sha256,
            "key_schema_version": self.key_schema_version,
            "plan_sha256": self.plan_sha256,
            "effective_recipe_sha256": self.effective_recipe_sha256,
            "authored_recipe_sha256": self.authored_recipe_sha256,
            "operation_registry_sha256": self.operation_registry_sha256,
            "planner_validation_sha256": self.planner_validation_sha256,
            "entry_count": self.entry_count,
            "max_entries": self.max_entries,
        }


@dataclass(frozen=True)
class ExecutionPlanCacheResult:
    plan: ExecutionPlan
    evidence: ExecutionPlanCacheEvidence


@dataclass(frozen=True)
class ExecutionPlanCacheStats:
    hits: int
    misses: int
    bypasses: int
    evictions: int
    entry_count: int
    inflight_count: int
    max_entries: int

    def to_dict(self) -> JsonDict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "entry_count": self.entry_count,
            "inflight_count": self.inflight_count,
            "max_entries": self.max_entries,
        }


@dataclass(frozen=True)
class _CacheEntry:
    registry_ref: ReferenceType[OperationRegistry]
    plan: ExecutionPlan


@dataclass
class _InflightPlan:
    registry: OperationRegistry
    completed: Event
    generation: int
    failure_type: Optional[type] = None
    failure_args: Tuple[Any, ...] = ()
    failure_message: str = ""


_LookupSlot = Tuple[int, str]


class ExecutionPlanCache:
    """Thread-safe, bounded LRU cache for successful execution plans.

    Concurrent callers for the same registry and semantic key share one
    planning operation.  Planning failures are never retained. ``clear`` and
    ``invalidate`` advance a generation counter so a plan already in flight
    cannot repopulate an explicitly invalidated cache.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_EXECUTION_PLAN_CACHE_MAX_ENTRIES,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: "OrderedDict[_LookupSlot, _CacheEntry]" = OrderedDict()
        self._inflight: Dict[_LookupSlot, _InflightPlan] = {}
        self._lock = RLock()
        self._generation = 0
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._evictions = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def plan(
        self,
        recipe: Recipe,
        registry: OperationRegistry,
        *,
        runner: str = DEFAULT_EXECUTION_RUNNER,
        backend: Optional[str] = None,
        implementation: Optional[str] = None,
        enforce_execution_profile: bool = True,
        authored_recipe: Optional[Recipe] = None,
        use_cache: bool = True,
    ) -> ExecutionPlanCacheResult:
        """Resolve ``recipe`` and return cache evidence with the plan.

        ``recipe`` is the effective recipe submitted to the planner.
        ``authored_recipe`` is optional because it does not change planner
        output. Callers that preserve authored/effective provenance should
        supply it so the run-specific authored digest is included in evidence
        without reducing hits for identical effective recipes. Set
        ``use_cache=False`` for an explicit one-shot bypass.
        """

        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be a boolean")
        if authored_recipe is not None and not isinstance(authored_recipe, Recipe):
            raise TypeError("authored_recipe must be a Recipe")
        key = build_execution_plan_cache_key(
            recipe,
            registry,
            runner=runner,
            backend=backend,
            implementation=implementation,
            enforce_execution_profile=enforce_execution_profile,
        )
        authored_recipe_sha256 = (
            recipe_fingerprint(authored_recipe)
            if authored_recipe is not None
            else None
        )
        if not use_cache:
            with self._lock:
                self._bypasses += 1
            plan = plan_recipe(
                recipe,
                registry,
                runner=key.runner,
                backend=key.backend,
                implementation=key.implementation,
                enforce_execution_profile=key.enforce_execution_profile,
            )
            return ExecutionPlanCacheResult(
                plan=plan,
                evidence=self._evidence(
                    "bypass",
                    key,
                    plan,
                    authored_recipe_sha256=authored_recipe_sha256,
                ),
            )

        slot = (id(registry), key.sha256)
        while True:
            with self._lock:
                entry = self._entries.get(slot)
                if entry is not None and entry.registry_ref() is registry:
                    # Dependency/import state is intentionally not part of the
                    # immutable semantic cache key. Recheck it before every
                    # hit so a stale plan cannot bypass execution preflight.
                    validate_execution_plan_runtime_availability(
                        recipe,
                        entry.plan,
                    )
                    self._entries.move_to_end(slot)
                    self._hits += 1
                    return ExecutionPlanCacheResult(
                        plan=entry.plan,
                        evidence=self._evidence_locked(
                            "hit",
                            key,
                            entry.plan,
                            authored_recipe_sha256=authored_recipe_sha256,
                        ),
                    )
                if entry is not None:
                    # Defensive handling for theoretical Python object-id reuse.
                    del self._entries[slot]

                inflight = self._inflight.get(slot)
                if inflight is None or inflight.registry is not registry:
                    inflight = _InflightPlan(
                        registry=registry,
                        completed=Event(),
                        generation=self._generation,
                    )
                    self._inflight[slot] = inflight
                    self._misses += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            inflight.completed.wait()
            if inflight.failure_type is not None:
                try:
                    failure = inflight.failure_type(*inflight.failure_args)
                except Exception:
                    failure = RecipePlanningError(inflight.failure_message)
                raise failure

        try:
            plan = plan_recipe(
                recipe,
                registry,
                runner=key.runner,
                backend=key.backend,
                implementation=key.implementation,
                enforce_execution_profile=key.enforce_execution_profile,
            )
        except BaseException as exc:
            self._complete_inflight(slot, inflight, failure=exc)
            raise

        with self._lock:
            current = self._inflight.get(slot)
            may_store = (
                current is inflight
                and inflight.generation == self._generation
            )
            if may_store:
                self._entries[slot] = _CacheEntry(registry_ref=ref(registry), plan=plan)
                self._entries.move_to_end(slot)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
                    self._evictions += 1
            if current is inflight:
                del self._inflight[slot]
            inflight.completed.set()
            evidence = self._evidence_locked(
                "miss",
                key,
                plan,
                authored_recipe_sha256=authored_recipe_sha256,
            )
        return ExecutionPlanCacheResult(plan=plan, evidence=evidence)

    def invalidate(
        self,
        *,
        key_sha256: Optional[str] = None,
        registry: Optional[OperationRegistry] = None,
    ) -> int:
        """Remove matching entries and prevent in-flight repopulation.

        With no filters this is equivalent to :meth:`clear`.  A semantic key
        invalidates all registry partitions for that key unless ``registry``
        is supplied as an additional filter.  Returns the number removed.
        """

        if key_sha256 is not None and not isinstance(key_sha256, str):
            raise TypeError("key_sha256 must be a string")
        if registry is not None and not isinstance(registry, OperationRegistry):
            raise TypeError("registry must be an OperationRegistry")
        with self._lock:
            matches = [
                slot
                for slot, entry in self._entries.items()
                if (key_sha256 is None or slot[1] == key_sha256)
                and (registry is None or entry.registry_ref() is registry)
            ]
            for slot in matches:
                del self._entries[slot]
            # This conservatively suppresses every currently in-flight insert.
            # It keeps invalidation semantics simple and deterministic.
            self._generation += 1
            return len(matches)

    def clear(self) -> int:
        return self.invalidate()

    def stats(self) -> ExecutionPlanCacheStats:
        with self._lock:
            return ExecutionPlanCacheStats(
                hits=self._hits,
                misses=self._misses,
                bypasses=self._bypasses,
                evictions=self._evictions,
                entry_count=len(self._entries),
                inflight_count=len(self._inflight),
                max_entries=self._max_entries,
            )

    def _complete_inflight(
        self,
        slot: _LookupSlot,
        inflight: _InflightPlan,
        *,
        failure: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            if failure is not None:
                inflight.failure_type = type(failure)
                inflight.failure_args = tuple(failure.args)
                inflight.failure_message = str(failure)
            if self._inflight.get(slot) is inflight:
                del self._inflight[slot]
            inflight.completed.set()

    def _evidence(
        self,
        outcome: CacheOutcome,
        key: ExecutionPlanCacheKey,
        plan: ExecutionPlan,
        *,
        authored_recipe_sha256: Optional[str],
    ) -> ExecutionPlanCacheEvidence:
        with self._lock:
            return self._evidence_locked(
                outcome,
                key,
                plan,
                authored_recipe_sha256=authored_recipe_sha256,
            )

    def _evidence_locked(
        self,
        outcome: CacheOutcome,
        key: ExecutionPlanCacheKey,
        plan: ExecutionPlan,
        *,
        authored_recipe_sha256: Optional[str],
    ) -> ExecutionPlanCacheEvidence:
        return ExecutionPlanCacheEvidence(
            outcome=outcome,
            key_sha256=key.sha256,
            key_schema_version=key.schema_version,
            plan_sha256=plan.sha256,
            effective_recipe_sha256=key.effective_recipe_sha256,
            authored_recipe_sha256=authored_recipe_sha256,
            operation_registry_sha256=key.operation_registry_sha256,
            planner_validation_sha256=key.planner_validation_sha256,
            entry_count=len(self._entries),
            max_entries=self._max_entries,
        )


def build_execution_plan_cache_key(
    recipe: Recipe,
    registry: OperationRegistry,
    *,
    runner: str = DEFAULT_EXECUTION_RUNNER,
    backend: Optional[str] = None,
    implementation: Optional[str] = None,
    enforce_execution_profile: bool = True,
) -> ExecutionPlanCacheKey:
    """Build the stable semantic key used by :class:`ExecutionPlanCache`."""

    if not isinstance(recipe, Recipe):
        raise TypeError("recipe must be a Recipe")
    if not isinstance(registry, OperationRegistry):
        raise TypeError("registry must be an OperationRegistry")
    if not isinstance(enforce_execution_profile, bool):
        raise TypeError("enforce_execution_profile must be a boolean")
    if implementation is not None and not isinstance(implementation, str):
        raise TypeError("implementation must be a string")
    try:
        runner_id = normalize_materialization_runner(runner, "runner")
        backend_id = (
            normalize_materialization_backend(backend, "backend")
            if backend is not None
            else None
        )
        operation_contracts = []
        for operation in registry.list():
            contract = operation.describe()
            operation_contracts.append(
                {
                    "operation_id": operation.id,
                    "implementation_identity": "%s.%s"
                    % (
                        operation.__class__.__module__,
                        operation.__class__.__qualname__,
                    ),
                    "implementation_source": _operation_source_identity(operation),
                    "contract": contract,
                }
            )
    except OperationError as exc:
        raise RecipePlanningError(
            "Cannot build execution-plan cache key: %s" % exc
        ) from exc

    effective_sha256 = recipe_fingerprint(recipe)
    registry_contract = {
        "registry_identity": "%s.%s"
        % (
            registry.__class__.__module__,
            registry.__class__.__qualname__,
        ),
        "operations": operation_contracts,
    }
    registry_sha256 = canonical_json_sha256(registry_contract)
    planner_validation_sha256 = canonical_json_sha256(
        {
            "research_catalog": load_research_catalog().to_dict(),
            "execution_profile_catalog": execution_profile_catalog().to_dict(),
        }
    )
    payload = {
        "schema_version": EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION,
        "planner_contract": {
            "execution_plan_kind": EXECUTION_PLAN_KIND,
            "execution_plan_schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "planned_step_schema_version": PLANNED_STEP_SCHEMA_VERSION,
            "operation_contract_set_schema_version": 1,
        },
        "effective_recipe_sha256": effective_sha256,
        "runner": runner_id,
        "backend": backend_id,
        "implementation": implementation,
        "enforce_execution_profile": enforce_execution_profile,
        "operation_registry_sha256": registry_sha256,
        "planner_validation_sha256": planner_validation_sha256,
    }
    return ExecutionPlanCacheKey(
        sha256=canonical_json_sha256(payload),
        effective_recipe_sha256=effective_sha256,
        operation_registry_sha256=registry_sha256,
        planner_validation_sha256=planner_validation_sha256,
        runner=runner_id,
        backend=backend_id,
        implementation=implementation,
        enforce_execution_profile=enforce_execution_profile,
    )
