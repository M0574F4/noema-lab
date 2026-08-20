# Benchmarking

Noema separates plumbing checks from frozen benchmark protocols.

## Benchmark Tiers

- `smoke`: fast checks for CI, UI plumbing, adapter wiring, and artifact contracts. Do not cite
  smoke numbers as research comparisons.
- `canonical`: frozen benchmark protocol intended for publication and method comparison. Canonical
  packs must declare dataset split, sample IDs, channel settings, rate accounting, expected output
  files, protocol ID/version, and `frozen: true`.
- `experimental`: useful research prototypes that are not yet frozen enough for leaderboard-style
  comparison.

Tier and strongest-profile selection are separate machine-readable states. A benchmark requests
Noema's strongest local traceability checks with
`traceability_profile_requested: true`; that request is admitted only for the `canonical` tier and
must carry the exact current `verification_profile` binding. A successful run or valid verifier
report does not promote a smoke or experimental pack, and a current-profile pass does not by itself
establish scientific validity, fair comparison, legal clearance, independent reproduction, or
publishability. See [Traceability Profile Governance](traceability_profiles.md) for versioning,
deprecation, and conformance policy. The historical `publication_ready` benchmark field is accepted
only as a deprecated alias for `traceability_profile_requested`.

The current Kodak v1-shaped pack is a frozen-name development regression protocol, not a canonical
publication pack. Its reusable split and image-rights evidence remain unresolved, so it is kept in
the `experimental` tier:

```bash
uv run noema benchmark validate benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
uv run noema benchmark run benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
```

## Experimental Kodak Image Reconstruction Development Protocol

Protocol: `noema.benchmark_v1.image_reconstruction.kodak`

- Dataset: Kodak 24-image test set.
- Samples: `kodim01` through `kodim24`.
- Crop: no crop, full images.
- Channel: disabled physical channel represented by the identity channel path.
- Channel code: identity.
- Required fixed points: payload bit boundary and transmitted bit boundary.
- Metrics: PSNR, MSE, MAE, codec bits, payload bits, transmitted bits.
- Baselines: JPEG q75 and CompressAI `bmshj2018_hyperprior` q3.
- Output files: `result.json`, `metrics.csv`, `recipes.csv`, `summary.md`.

Image reconstruction reports retain one row per sample. MSE, MAE, and PSNR are arithmetic means of
the per-image values rather than a pooled-pixel estimate. Inputs must have identical counts, declared
IDs/order (when IDs exist), and original shapes. PSNR uses the canonical range inferred from a shared
`uint8` (`[0,255]`) or `float32` (`[0,1]`) representation unless an explicit range is supplied.

The development pack requires every recipe to pass strict lint. This ensures that method results use
the same bit/channel accounting anchors.

Benchmark metrics are required for every executable recipe by default. A genuinely method-specific
quantity, such as native JPEG/entropy-stream bpp in a mixed digital/analog-symbol comparison, must
declare an exact `applicable_roles` list. The runner and result verifier then require and
producer-bind that metric only for those roles; this is not an escape hatch for a common outcome or
resource metric. Step-qualified values remain in the bundle even when they are not promoted to a
cross-method summary metric.

## Manifest-backed image populations

`source.image_dataset` keeps the historical Kodak shortcut, but any other image population must be
provided through a hash-pinned JSON or YAML manifest. The manifest contract is deliberately stricter
than a directory glob; its machine-readable definition is
[`schemas/image_dataset_manifest.schema.json`](../schemas/image_dataset_manifest.schema.json):

```yaml
schema_version: 1
kind: noema.image_dataset_manifest
id: licensed_second_dataset
version: 2026-07-23
root: images
splits:
  publication_test: [sample-002, sample-001]
samples:
  - sample_id: sample-001
    path: 001.png
    sha256: <file-sha256>
    source_id: capture-001
    group_id: scene-001
    source_sha256: <root-source-sha256>
    ancestry_ids: [capture-session-a, capture-001]
    transform: {name: identity}
    transform_fingerprint_sha256: <canonical-transform-sha256>
```

Paths are confined beneath the manifest root, symlinks are rejected, and bytes are checked before
decoding. The source operation requires both `manifest_path` and its independently recorded
`manifest_sha256`; recording a digest only after a run is not treated as a frozen input. Each result
records the selected order, source/group ancestry, transform fingerprint,
post-transform shape/dtype/hash, repetition index, and one digest over the ordered materialized
population. Cropped, renamed, recompressed, or re-noised derivatives remain related only when the
data curator truthfully retains their common `source_id`, `group_id`, `source_sha256`, and
`ancestry_ids`; Noema rejects that overlap against a returned artifact's fitting-data contract.

Publication packs declare `dataset.source_bindings` instead of relying on an operation name embedded
in the runner. A binding names an operation, its selection parameter/encoding, and the complete fixed
parameter map. During materialization Noema replaces the authored source parameters with this closed
map, so a recipe-local resize, crop, repeat count, or alternate manifest cannot survive silently.

The repository does not ship or license a second publication dataset. Adding the manifest loader and
lineage checks makes that population executable and auditable; acquiring the bytes and documenting
authoritative use/redistribution terms remain external release tasks.

## Enforced paired common conditions

A strongest-profile pack's `metadata.common_conditions` is an executable contract, not only prose.
It declares the power coordinate/scope/target, paired stochastic operation IDs and seed derivation,
CSI and receiver processing, plus the outage, decode-failure, and denominator policies. Every recipe
must also carry an `aggregation_cell_id` and `pairing_id`.

The benchmark materializer derives the same explicit channel seed for every method in a paired cell.
After execution it reconstructs evidence from the immutable run snapshot and verifies:

- identical ordered post-transform items, count, order, and tensor identities;
- identical per-item channel seeds, identity keys, and channel realization contract;
- the declared measured power or energy coordinate and source-item normalization scope;
- identical channel/CSI/backend and receiver-processing ownership; and
- a per-item outage vector whose denominator includes every attempted source item.

The comparison fails closed when any section is missing or differs across paired methods. The
result-local evidence and its hashes are recomputed during `benchmark verify`, so editing
`result.json` cannot manufacture common-condition compliance.

## Memory isolation

Dashboard Run All jobs and `noema benchmark run` execute in a worker process rather than in the UI
server. On Linux with cgroup v2 and a user systemd manager, Noema gives that worker a transient
memory-controlled service; other environments use an isolated process group and the same host-memory
watchdog as a best-effort fallback. If the protected reserve is reached, Noema terminates the worker
tree, keeps the server alive, and records `failure.kind: resource_exhausted` plus the resource-guard
settings in the run or benchmark evidence.

The default protected reserve is the greater of 2 GiB or 10% of physical RAM. These environment
variables tune the guard before starting Noema:

```bash
NOEMA_EXECUTION_MEMORY_RESERVE_MB=2048 \
NOEMA_EXECUTION_MEMORY_POLL_HZ=10 \
uv run noema ui serve
```

- `NOEMA_EXECUTION_MEMORY_RESERVE_MB`: RAM kept available for the OS and UI server.
- `NOEMA_EXECUTION_MEMORY_MAX_MB`: optional additional cap on the worker; it can never override the
  protected reserve.
- `NOEMA_EXECUTION_MEMORY_SWAP_MAX_MB`: worker swap allowance, 1024 MiB by default.
- `NOEMA_EXECUTION_MEMORY_POLL_HZ`: host-memory sampling rate from 1 through 50 Hz.
- `NOEMA_EXECUTION_MEMORY_HIGH_RATIO`: cgroup soft threshold relative to the hard limit. It defaults
  to 1.0 so benchmark timing is not distorted by early memory throttling.

Only one protected worker is admitted at a time. Parallel graph branches still use the configured
execution worker count inside that isolated process.

Learned power-allocation policies also expose `Model batch size` (default `1024`). This limits only
the number of independent channel-state rows passed through the returned model in one inference
call. Noema preserves the recipe data batch, complete CSI population, output order, and fixed-sum
power projection while concatenating the chunked model outputs.

## Resuming a failed benchmark

A failed terminal benchmark can be resumed explicitly without rerunning recipe rows whose frozen
run evidence is still valid:

```bash
uv run noema benchmark run benchmarks/<suite>/<pack>.yaml \
  --resume <failed_result_id>
```

Resume is never selected automatically. The source must have terminal status `failed`, and the
benchmark protocol hash and all execution controls (`--strict-lint`, backend, implementation,
parallel workers, and plan-cache setting) must exactly match the original attempt. Noema verifies
the source result against the append-only attempt ledger, rehashes every completed run-evidence
snapshot, and checks each prepared recipe and bound trained-artifact identity before reusing it. The
failed recipe row is retried and undeclared work after that row is rejected; subsequent recipe rows
then run normally.

A resume creates a new result bundle and a new attempt-ledger entry. The failed source remains
immutable and stays in the attempt history. Reused rows record their source result and attempt, and
their evidence is cloned into the new result with the same independent-file guarantees as an
ordinary benchmark snapshot. A completed, incomplete, cancelled, modified, or protocol-mismatched
result cannot be used as a resume source.

## Project-Local Discovery

`benchmarks/` and `recipes/` are project protocol files. They are not hidden inside the installed
Python package. In a fresh clone:

```bash
uv run noema benchmark list
uv run noema benchmark validate benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
```

If Noema is installed as a package elsewhere, run benchmark commands from a directory that contains
the benchmark and recipe files, or pass explicit paths.

## Result Bundle

Every benchmark run writes:

- `result.json`: immutable, canonical machine-readable result bundle;
- `metrics.csv`: long metric table for analysis, hash- and size-bound from `result.json`;
- `recipes.csv`: one row per method/recipe with selected metrics, hash- and size-bound from
  `result.json`;
- `summary.md`: human-readable result summary, hash- and size-bound from `result.json`; and
- optional post-run sidecars such as `resource-guard.json` and `*.plot.json`.

Noema creates the reports before sealing the terminal `result.json`. The terminal attempt-ledger
event binds the exact result bytes by SHA-256 and size, and later commands never rewrite those bytes.
Each referenced run snapshot similarly binds the exact stored payload bytes and validates their
decompressed JSON semantics against the run manifest. The verifier checks the stored report bytes
and regenerates their semantic content from the sealed result.

Benchmark run-evidence stores a strict projection so verification never depends on a
mutable or eventually deleted `.noema/runs/` directory without duplicating every intermediate
tensor. Each recipe snapshot contains the effective and authored recipes, execution plan, manifest,
and summary as deterministic gzip JSON. It retains artifact bytes only for `metrics.report` outputs
emitted by the authoritative producer steps of metrics declared for that recipe role. Demo plots use
the sealed result metrics, while publication observation tables use those retained reports.

When an analysis also needs a compact non-metric artifact, one recipe entry can declare exact
manifest-relative paths. This opt-in is part of the frozen pack identity and does not broaden
retention for heterogeneous methods:

```yaml
recipes:
  - id: protected_digital
    path: ../../recipes/protected_digital.yaml
    params:
      run_evidence:
        retained_artifact_paths:
          - artifacts/channel_encoder/coded_bits.npz
```

The list is non-empty, unique, and limited to canonical `artifacts/...` paths. It is an exact
allowlist, not a glob. Every named path must appear in that run's manifest, and the snapshot binds
the sorted allowlist, retained bytes, and source inventory. Entries without the declaration keep
the metric-only default.

Before pruning the source run, Noema resolves the role-filtered metric producers and SHA-256 checks
every artifact declared by the run manifest, including artifacts omitted from the compact snapshot.
The snapshot manifest binds the complete source inventory digest, the exact retained subset, and
source/retained/omitted counts and byte totals. Later verification rehashes every retained byte,
recomputes the projection from the frozen run manifest, and revalidates recipe, plan, metric
producer, common-condition, and result-row semantics. Tampered JSON, reports, projection metadata,
or result bindings therefore fail closed after the source run is gone. Concatenated, truncated, or
oversized gzip payloads are rejected.

Legacy schema-v1 results that contain full artifact snapshots remain readable and are never migrated
or rewritten. For retained files, Linux filesystems with copy-on-write support use independent
reflink inodes; other filesystems receive an ordinary byte copy. Hard links are never used, and
failed snapshot staging trees are removed.

After each result-local snapshot has been revalidated and its recipe row and reports have been
written atomically, benchmark runs prune the now-redundant `.noema/runs/<run_id>` directory by
default. A run whose snapshot failed remains available for diagnosis. Use
`--retain-backing-runs` when you need undeclared intermediate tensors for exploratory analysis or
debugging after the benchmark:

```bash
uv run noema benchmark run benchmarks/<suite>/<pack>.yaml --retain-backing-runs
```

Resume reads completed work exclusively from the failed result's frozen `run_evidence/` snapshots,
so it continues to work after successful backing runs have been pruned. The Results UI likewise
falls back to the authenticated result-local summary and recipe when a live backing run is absent.

An intentionally skipped placeholder is retained as a recipe row with `status: skipped`, while the
benchmark and attempt terminate with `status: incomplete`. This preserves the attempted denominator
without representing a missing method as an observation. Incomplete results cannot be plotted,
published, or used as comparable benchmark evidence.

Benchmark results can also export paper-ready figures from the stored `result.json` evidence:

```bash
uv run noema benchmark plot <result_id> \
  --plot graceful-degradation \
  --x channel.snr_db \
  --y quality.psnr_db \
  --group method \
  --method-order "JPEG + protected digital link,CompressAI q3 + protected digital link,DeepJSCC baseline,User method" \
  --packet-success-panel \
  --style paper \
  --out figures/graceful_degradation.png

uv run noema benchmark plot <result_id> \
  --plot packet-success \
  --x channel.snr_db \
  --out figures/packet_success.png
```

Relative `--out` paths are written inside `.noema/benchmarks/<result_id>/`, keeping the benchmark
bundle self-contained. Each plot writes a sibling CSV with the exact plotted points and an immutable
`*.plot.json` sidecar. The sidecar binds the sealed result ID, result SHA-256 and size, render
specification, figure, and data bytes without mutating `result.json`. New sidecars also bind
`semantic_projection_sha256`, a renderer-independent digest over the plot type, selection
declaration, axes, grouping and method order, display-selection switches, and reproduced ordered
rows. This semantic identity is separate from—not a replacement for—the raw CSV/image hashes and
named renderer environment used for exact-byte verification. Use `.svg` or `.pdf` as the `--out`
suffix to export vector figures. Repeated runs with the same method and x value are aggregated into
one plotted point. Two-sided 95% intervals use Student's t distribution and are left unreported for
a single observation. When exactly one baseline series and complete pairing IDs exist, the CSV also
records paired candidate-minus-baseline contrasts; mismatched pairing sets are identified instead of
partially joined. Resource-budget-rejected rows cannot enter curves. Outage markers make digital
cliff points visible instead of silently treating gray-image fallback PSNR as ordinary reconstruction
quality.

`resource-guard.json` is also a post-run sidecar. It binds its own content to the sealed result
identity and records the resource-guard outcome without reopening the terminal result. Verification
checks both resource-guard and plot sidecars against the exact sealed result bytes.

Each recipe row links to a run manifest containing recipe SHA-256, operation contracts, environment,
git state, seed policy, and artifact hashes.

## Attempt denominator ledger

Publication campaigns must retain every benchmark invocation, not only successful result bundles.
`BenchmarkAttemptLedger` stores immutable events under
`.noema/benchmarks/.attempt-ledger/events/`. A `started` event is committed before validation or
execution. A later event records `completed`, `failed`, `cancelled`, or `resource_rejected`, including
per-recipe outcomes and an optional result identity. If a process dies, the unmatched start remains
visible as `started_without_terminal_event`. Corrected reruns append a `superseded` event rather than
rewriting or deleting the earlier attempt.

Event filenames bind a monotonic sequence and SHA-256. Every event binds its predecessor; writes use
a temporary file, fsync, an atomic no-overwrite link, and a process lock. `head.json` is only a
replaceable index—the immutable event chain is authoritative after a crash. Use
`registration_snapshot(path)` to write a self-hashed, paper-registerable projection containing the
full attempt denominator and status counts.

Sealed publication-test access is reserved inside the same process lock. For a `started` invocation
with `test_access: true`, the ledger validates the seal digest and positive budget, counts prior starts
for that seal, and overwrites `access_sequence` with the authoritative next value. The event records
`access_granted`; starts beyond budget are still appended with `false` so concurrent denied queries
cannot disappear from the access denominator.

This is an internal integrity control, not an external timestamp or signature. The snapshot reports
`internal_hash_chain_without_external_trust_anchor`; archive release should pin or sign its final
digest outside the workspace.
