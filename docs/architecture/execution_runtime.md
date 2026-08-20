# Execution Runtime

Noema plans every recipe before creating a run bundle, then executes the planned DAG locally.
Sequential execution remains the default. Parallel execution is an explicit runtime choice and does
not alter the recipe, execution profile, or execution-plan digest.

## Controls

CLI recipe runs accept bounded worker counts from 1 through 32:

```bash
noema recipe run recipes/example.yaml --parallel-workers 4
noema recipe run-matrix recipes/example.yaml --parallel-workers 4
noema recipe run recipes/example.yaml --no-plan-cache
```

`--parallel-workers 1` is the default. `run-matrix` applies the worker limit independently to each
variant; variants themselves still run in matrix order. The dashboard exposes Sequential, 2, 4,
and 8 workers and keeps Sequential as the initial choice.

The synchronous recipe endpoints and background run-job endpoint accept the same nested object:

```json
{
  "execution": {
    "parallel_workers": 4,
    "use_plan_cache": true
  }
}
```

This object is supported by `POST /api/recipe/run`, `POST /api/recipe/run-payload`, and
`POST /api/run-jobs`. Unknown execution fields, booleans used as worker counts, values outside
1–32, and non-boolean cache flags are rejected. Omitting the object preserves the sequential,
cache-enabled defaults.

## DAG Scheduling

With more than one worker, the scheduler submits only steps whose declared input dependencies have
completed. Independent ready steps may overlap up to the selected bound only when their operations
explicitly declare `thread_safe = true`; undeclared operations execute exclusively and form a
plan-order scheduling barrier. Join steps wait for all of their inputs. The main scheduler alone
updates produced-artifact maps and run evidence. Summary and manifest steps are persisted in
execution-plan order, so completion timing does not reorder the durable bundle.

The dashboard's worker selection applies to every recipe job it starts, but **Run All** still runs
recipe tabs sequentially and runs matrix variants sequentially. Workers are used only for
dependency-independent DAG branches inside the current recipe; a linear pipeline therefore gains
no concurrency. Step and model timings are elapsed wall-clock measurements. Concurrent branches
can contend for CPU, GPU, memory bandwidth, and I/O, so use one worker for latency comparisons.
Parallel timing is appropriate only when concurrency or throughput is part of the declared protocol,
and comparisons must use the same recorded execution mode and worker count.

The registry retains one planned operation, but dispatch uses a shallow per-step copy. This contains
ordinary scalar instance mutation; it does not isolate deliberately shared models, callbacks, nested
containers, native libraries, or process-global RNG state. Consequently `thread_safe = true` is an
explicit implementation assertion covering the operation and everything it calls. External adapters
remain exclusive under the current manifest schema.

Sionna 2.x is a narrower exception because its random configuration is process-global. Noema wraps
seed assignment and each stochastic Sionna artifact call in a process-local reentrant lock. This
preserves seeded serial/parallel equivalence inside one executor process, at the cost of serializing
Sionna stochastic sections. Native NumPy and Torch channel materializations use local random
generators and remain parallel-safe.

Every operation writes into a private staging directory. Before a step can complete or release an
artifact downstream, the executor requires an `OperationResult` whose output names exactly match the
registered contract. Each artifact kind must match, its path must be a regular non-symlink file inside
the staging directory, its SHA-256 must match its bytes, and metadata/metrics must be strict finite
JSON. The executor then atomically renames the staged directory into `artifacts/<step-id>` and verifies
the hashes again. Missing, extra, wrong-kind, aliased, stale, or out-of-run outputs fail the run and do
not dispatch dependent steps.

## Cooperative Cancellation

`CancellationToken` is thread-safe and idempotent. The executor checks it before scheduling, at the
operation dispatch boundary, after sequential steps, and before successful finalization. Every
`OperationContext` receives the same token and provides:

- `ctx.is_cancelled()` for non-throwing polling;
- `ctx.raise_if_cancelled()` for a cancellation checkpoint;
- `ctx.report_progress(...)`, which checks cancellation before emitting progress.

Long-running operations should poll at natural batch, epoch, or external-I/O boundaries. External
adapter wrappers propagate the token to the wrapped operation. Background jobs use the existing
`POST /api/run-jobs/<job_id>/cancel` endpoint.

Cancellation is cooperative: Python threads and native/model calls cannot be force-killed safely.
An operation that never returns or checks its context can delay cancellation and executor shutdown.
On a branch failure, Noema stops admitting new work, requests cancellation from running peers, and
preserves the original error as the run's failed status. A user cancellation without another
failure produces a canceled run.

## Plan Cache

The process-local execution-plan cache is a bounded optimization around the authoritative planner.
It never caches operation outputs or failed plans. Its semantic key covers the effective recipe,
planner selectors, operation registry contracts and implementation identities, planner validation
catalogs, and schema versions. See [Execution-Plan Cache](execution_plan_cache.md) for key,
invalidation, and concurrency details.

## Run Evidence

New run summaries and manifests contain matching scheduler evidence:

```json
{
  "execution": {
    "mode": "parallel",
    "parallel_workers": 4
  }
}
```

Their `execution_plan.cache` object records `hit`, `miss`, or `bypass`, relevant digests, and cache
capacity. Bundle verification checks its schema and internal links to the plan and authored/effective
recipes. Cache outcome and opaque process-local key/registry digests are performance provenance;
they are not an independently attestable claim that changes scientific results.
