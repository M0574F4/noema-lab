# Benchmark Submissions

This document defines the local submission rules for Noema benchmark results.

## Submission route

1. Validate the complete bundle locally as described below.
2. Open a proposal at <https://github.com/M0574F4/noema-lab/issues/new> with the benchmark ID,
   comparison division, bundle size, and a digest. Do not attach private data or security-sensitive
   details.
3. Submit small metadata/fixture changes through a pull request. For large result bundles, wait for
   the acting maintainer to approve an immutable artifact location before upload.

The current repository does not operate a private leaderboard or automatic intake service.

## Allowed Modifications

Closed division submissions may change only the method contribution:

- codec, semantic encoder, receiver, channel model, channel code, modulation block, metric, or
  dataset adapter when the benchmark allows that contribution type;
- runtime/backend choices declared in the recipe and manifest;
- method-specific hyperparameters declared in the recipe.

Closed division submissions must not change:

- canonical benchmark dataset/split/sample IDs;
- metric definitions or reductions;
- rate/channel fixed points;
- reference data or ground truth;
- benchmark pack ID/version;
- seed policy, except where the protocol explicitly defines a seed grid.

Open division submissions may change more of the system, but must declare every difference in the
submission metadata and should not be compared directly with closed-division results.

## Required Metadata

Submissions must include:

- method name and authors;
- comparison division: `closed` or `open`;
- adapter manifest path/name/version when an adapter is used;
- benchmark ID and version;
- result bundle path;
- recipe SHA-256;
- hardware, runtime, software dependencies, and optional accelerator details;
- seed policy and concrete seeds;
- any external data, checkpoint, or pretrained artifact identifiers.

## Validate Locally

```bash
uv run noema submission validate path/to/submission.yaml
```

The validator requires a normalized relative `result_bundle` path confined to the submission
directory and rejects traversal, absolute paths, and symlinks. It invokes the full benchmark verifier,
including result-local run evidence, protocol identity, artifacts, reports, plots, resource admission,
and training-lineage checks. It also cross-checks every recipe digest, concrete master seed, recorded
seed-policy digest, and the comparison division frozen in `benchmark.json`.

The default successful verdict is `verified_internal_consistency_untrusted_producer`. That verdict is
appropriate for an honest producer and accidental-drift detection; it does not authenticate hostile
input. API callers can pass an independently obtained `trusted_bundle_root_sha256` to receive
`verified_pinned_bundle_integrity` after the exact bundle inventory matches. A pin proves byte identity,
not author identity or scientific truth; signatures and producer identity remain release-policy
responsibilities outside this local validator.

For multi-recipe results, `reproducibility.recipe_sha256s` must map every benchmark recipe ID to the
recorded run recipe digest. `reproducibility.seed_policy_sha256s` must list the distinct SHA-256
identities of the full recorded seed-policy objects. A single-recipe submission may put the one
seed-policy digest directly in `seed_policy`; the human-readable field remains available when the
digest list is supplied. The frozen benchmark must declare either
`metadata.comparison_division` or `metadata.allowed_comparison_divisions`.

## Minimal Submission YAML

```yaml
schema_version: 1
comparison_division: closed
result_bundle: .noema/benchmarks/20260708T000000Z_benchmark_v1.image_reconstruction.kodak
method:
  name: My Semantic Codec
  authors:
    - A. Researcher
adapter:
  manifest: adapters/my_codec/noema_adapter.yaml
  name: my_codec
  version: 0.1.0
benchmark:
  id: benchmark_v1.image_reconstruction.kodak
  version: "1.0.0"
runtime:
  hardware: CPU
  software: noema 0.2.0.dev0
  dependencies:
    - numpy
reproducibility:
  recipe_sha256: "<sha256 from manifest>"
  seed_policy: noema deterministic step-derived seeds
  seed_policy_sha256s:
    - "<sha256 of the complete recorded seed_policy object>"
  seeds:
    - 23
```
