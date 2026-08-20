# Run and Result Verification

Noema separates an experiment plan from the evidence produced by running it.

- A **recipe** is the plan before execution: a typed DAG of data, codec, channel, receiver, and metric operations.
- A **run bundle** is execution evidence under `.noema/runs/<run_id>/`, usually including `summary.json`, `manifest.json`, `recipe.json`, and artifacts.
- A **benchmark result bundle** is benchmark-level evidence under `.noema/benchmarks/<result_id>/`, usually centered on `result.json`.

The verifier does not prove that a scientific claim is true. It checks that a bundle is complete,
internally consistent, sufficiently documented to inspect, and compliant with the benchmark/protocol
rules that Noema can verify locally. It never reruns the full experiment.

## CLI

List stored runs:

```bash
noema runs list
```

Verify one run bundle:

```bash
noema runs verify <run_id>
noema runs verify <run_id> --json
```

Verify one benchmark result bundle:

```bash
noema benchmark verify <result_id>
noema benchmark verify <result_id> --json
```

Human output starts with one status line and then lists warnings and errors:

```text
valid run: 20260709T000000Z_example
warning: ...
error: ...
```

JSON output contains `status`, `target_type`, `target_id`, `path`, `errors`, `warnings`, and detailed `checks`.
Benchmark-result output also contains a `certification` record. It always reports:

- `verdict_schema_revision` and the benchmark-result schema revision;
- benchmark tier;
- whether the strongest traceability profile was requested;
- claimed and required profile identifiers and digests;
- profile-binding status;
- applicable I1--I6 predicates;
- one unambiguous verdict class; and
- separate `externally_conformant`, `archived`, `independently_reproduced`, and
  `publication_candidate` fields.

The four local verdict classes are `current_profile_pass`,
`current_profile_fail`, `nonpublication_evidence_pass`, and
`nonpublication_evidence_fail`. A legacy or experimental result can therefore never be reported as
a current-profile pass merely because its declared internal verifier succeeds.

## What Is Checked

For local run bundles, Noema verifies:

- required files: run directory, `summary.json`, `manifest.json`, and `recipe.json`;
- JSON parsing and expected top-level field types;
- run status, where completed runs are comparable and failed/canceled/incomplete runs are non-comparable;
- recipe identity by recomputing the canonical `recipe.json` SHA-256 and comparing it with the manifest;
- manifest integrity: run id, created time, status, recipe info, operation contracts, seed policy,
  steps, artifacts, and the SHA-256 and size of the exact `summary.json` bytes;
- artifact integrity: manifest artifact paths exist, hashes match when recorded, and metadata is present enough to interpret outputs;
- metric plausibility: numeric metrics are finite and rates, latencies, bit counts, counts, and durations are non-negative where appropriate;
- simple accounting consistency, such as recomputing bpp from transmitted bits and source pixels when both are available;
- strict recipe lint compatibility when the operation registry is available.

For benchmark result bundles, Noema verifies:

- `result.json` structure and benchmark id/version;
- completed benchmark status and the separately declared `benchmark_tier` and
  `traceability_profile_requested` state;
- for a strongest-profile pack, the exact identifier and canonical digest of the
  `noema.traceability.v2` minimum verification profile in both the frozen protocol and result;
- each recipe entry status, including an intentionally skipped placeholder or explicit
  resource-budget rejection as a preserved non-observation;
- required benchmark metrics declared by the benchmark protocol;
- dataset/task protocol identifiers when available;
- result-local backing-run snapshots, effective recipe identity, pairing/aggregation/statistical-unit
  declarations, and step-scoped resource-admission decisions bound to the protocol hash;
- required training-evidence snapshots and content-disjoint training/evaluation lineage when the
  frozen benchmark flag requires it;
- terminal attempt-ledger binding of the immutable `result.json` SHA-256 and size;
- exact hashes, byte sizes, and semantic regeneration of `metrics.csv`, `recipes.csv`, and
  `summary.md`;
- resource-guard sidecar binding to the sealed result identity;
- plot-sidecar binding, renderer-independent semantic-projection identity when present, image/data
  hashes, and regeneration from the stored result and render specification, with
  renderer implementation, Python/Matplotlib/FreeType, and font identity checked before exact-byte
  comparison; and
- safe, non-colliding result-relative evidence paths. Verification does not consult a mutable
  external run directory when a result-local snapshot is required.

## Dashboard

Open the dashboard and use the Results source selector:

```bash
noema ui serve --port 8766
```

In the Results view, the **Open** selector lists:

- current recipe tabs;
- past runs from `.noema/runs`;
- benchmark result bundles from `.noema/benchmarks`.

Select a past run to load its stored summary, recipe, metrics, artifacts, and manifest-derived result displays without rerunning the experiment. Click **Verify run** to run the same local verifier from the UI. The dashboard shows PASS, WARNING, or FAIL plus the first actionable message.

## Interpreting Results

- **valid** means Noema found no local consistency or compliance problems.
- **warning** means the bundle is usable but has missing optional evidence or limitations that should be reported.
- **invalid** means the bundle is incomplete, corrupted, internally inconsistent, non-comparable, or missing required protocol evidence.

An incomplete placeholder campaign terminates as `incomplete`, even though its skipped row remains in
the denominator. It is non-comparable, unplottable, and unpublishable rather than a zero or missing
measurement.

A valid verifier report is not a claim that a model is scientifically superior or that a result is
publishable. A current-profile claim requires both `benchmark_tier: canonical` and
`traceability_profile_requested: true`, plus the exact `verification_profile` binding returned by
`publication_verification_profile_binding()`. The machine-readable profile in
`noema_lab.core.publication_profile` fixes the non-optional predicate classes, their conditional
applicability, dependency edges, validator modules, and conformance-test modules. Authors may add
study-specific checks but cannot remove that minimum while retaining the current-profile claim.
Verification then establishes local consistency of the declared evidence and emits
`current_profile_pass`; it does not establish external FEC/PHY conformance, immutable archive
availability, independent reproduction, fairness, authenticity, producer honesty, or venue
acceptance. Those are separate status-vector dimensions and remain `not_evaluated` or
`not_determined` unless another authorized process supplies them. Smoke and experimental results
remain non-publication evidence even when valid.

`publication_ready` remains accepted in benchmark packs and results as a deprecated compatibility
alias. When both request fields occur they must agree, and disagreement fails closed. This alias is
not the similarly named returned-artifact or dataset-rights state. See
[Traceability Profile Governance](traceability_profiles.md) for ownership, immutable profile
versioning, migration, and conformance levels.

The verifier's public tests exercise returned artifacts, strict accounting, attempt and failure
denominators, metric provenance, report and plot projections, and reconstruction from copied result
bundles without mutable backing runs. These are local consistency controls, not independent
reproduction or publication certification.
