# Execution-Plan Cache

Execution-plan caching is an optional optimization around the authoritative
`plan_recipe(...)` contract. It does not cache run results, operation outputs,
or planning failures.

`ExecutionPlanCache` is an in-memory, bounded LRU cache. Its public key is a
canonical SHA-256 digest over every semantic input to planner resolution:

- the complete effective recipe;
- normalized runner and backend selectors, plus implementation selection;
- execution-profile enforcement policy;
- the research and execution-profile validation catalogs;
- execution-plan, planned-step, operation-contract-set, and cache-key schema
  versions;
- every operation contract in the supplied registry, including parameter
  schemas and materialization declarations;
- the registry's Python class identity;
- each operation's Python class identity.

The authored recipe hash is run provenance rather than a planner input. It is
therefore included in cache evidence when supplied, but it does not reduce
hits between authored recipes that compile to the same effective recipe.

## Safety And Concurrency

An execution plan embeds concrete `Operation` objects. Internally, cache slots
are therefore partitioned by `OperationRegistry` object identity even when two
registries have the same deterministic semantic key. This guarantees that a
plan requested with a new registry cannot execute an operation instance from
an older registry. Entries weakly reference the registry itself, so the global
bounded cache does not keep an otherwise-unused registry and all of its
unreferenced operations alive.

Plans and cache evidence are immutable. Concurrent requests for the same key
and registry coalesce around one planning operation; requests for distinct
keys can plan concurrently. Callers already waiting on a failed planning
attempt observe that same failure, but the failure is never retained for a
later request.

The cache defaults to 128 entries and evicts the least-recently-used entry.
Callers can:

- set `use_cache=False` for a one-shot bypass;
- call `invalidate(key_sha256=...)` for semantic-key invalidation;
- optionally restrict invalidation to one registry;
- call `clear()` to remove all entries;
- call `stats()` to inspect hits, misses, bypasses, evictions, entries, and
  in-flight planning operations.

Both `clear()` and `invalidate()` also suppress insertion of plans that were
already in flight when invalidation occurred.

## Evidence

Each lookup returns `ExecutionPlanCacheResult(plan, evidence)`. Evidence is a
small serializable object with `outcome` (`hit`, `miss`, or `bypass`), the
deterministic key schema/digest, effective-recipe, registry, validation-catalog, and plan digests,
optional authored-recipe provenance, and the cache capacity/current size.
Executors can persist `evidence.to_dict()` in their manifest and summary
without making cache state part of the execution plan digest.

Run-bundle verification checks the evidence schema and links that can be established from the
bundle. The process-local hit/miss outcome and opaque cache-key/registry digests are performance
telemetry, not an independently attestable scientific-result claim.

`EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION` must be incremented when resolution
semantics change without an execution-plan schema-version change. This makes
invalidation explicit instead of relying on process restarts.
