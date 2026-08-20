# Publication readiness for trained artifacts

Noema reports two deliberately different verdicts for a returned trained artifact:

- `ready` / `status` answer whether its files, hashes, ABI, operation bindings, and runtime are valid
  for ordinary execution. Existing recipes and UI clients continue to use this verdict.
- `publication_ready` / `publication_status` add a model-selection provenance gate. A runnable
  development artifact can therefore be `ready: true` and `publication_ready: false`.

A publication-frozen benchmark rejects every learned artifact that does not pass the stronger gate.
It also requires disjoint training lineage and a sealed publication-test population. This prevents a
runtime-valid checkpoint from silently standing in for a model whose search history or held-out-test
isolation is unknown.

## Required package evidence

Publication readiness requires a schema-v2 artifact and this manifest reference:

```yaml
publication:
  selection_history:
    path: provenance/model_selection_history.yaml
    sha256: <sha256-of-the-exact-history-file>
```

The history file must also appear in `support_files` with role `model_selection_history`. The frozen
search program and every candidate configuration must be package files with roles
`model_selection_search_program` and `model_selection_configuration`, respectively. All paths are
package-relative, regular files, and hash checked.

The history document uses `kind: noema.model_selection_history`, `schema_version: 1`, and records:

- `all_candidates_disclosed: true` and `publication_test_accessed: false`;
- a content-identified `selection_population` whose role is `development_only` or
  `adaptation_validation`, never a publication test;
- the exact objective, direction, deterministic selection/tie-breaking rule, search-program argv,
  and selected candidate ID;
- every attempted candidate, including failed and resource-rejected attempts;
- for each candidate, `configuration_path` plus the exact-file
  `configuration_identity_sha256`, validation-population hash, explicit
  `publication_test_accessed: false`, status, and either a finite validation objective plus
  `runtime_component_set_sha256` or a failure reason.

The selected candidate's `runtime_component_set_sha256` is not an arbitrary checkpoint label. It
must equal `publication_readiness.runtime_component_set_sha256`, the canonical digest of the
deployed component IDs, roles, formats, and verified hashes. This is distinct from the registered
release object's `package_file_sha256` (exact archive bytes or deterministic directory-tree
digest) and from `runtime_identity_sha256` (manifest, contract, components, and support files).
The latter includes the history, so putting it inside that history would create a hash cycle.
Inspect the completed runtime package once to obtain both runtime identities, then freeze the
history and register its files in the release manifest. Legacy runtime clients still receive
`package_sha256` as an alias of `runtime_identity_sha256` and
`runtime_artifact_sha256` as an alias of `runtime_component_set_sha256`.

Use `validate_trained_artifact_publication_readiness(...)` for a hard gate. Ordinary
`inspect_trained_artifact(...)` never changes runtime compatibility because publication evidence is
missing; it returns the blockers in `publication_issues` instead.

During publication release, artifact registration is also cross-checked against each result bundle. The
checker validates the result-local training-evidence snapshot, re-inspects every copied schema-v2
package, and requires its `runtime_component_set_sha256` plus `runtime_identity_sha256` to match the
exact `artifact:*` objects named by that result. Artifact IDs alone cannot authorize swapped bytes,
and an unregistered local package or a registered package absent from the result both fail closed.

The publication observation producer uses the exact result-local
`trained_artifact_manifest.sha256` as the `training_replication` hierarchy value. The statistical
plan freezes the allowed independent-replication manifest digests, and each completed result must
match exactly one. Re-evaluating one checkpoint under several attempt IDs therefore cannot inflate
the number of independent training replications.

## What this gate does not prove

The package makes the disclosure content-addressed and machine-checkable, but an author can still lie
about whether every off-record trial was disclosed or whether an external test set was viewed. Final
publication release therefore still needs the repository's append-only attempt ledger, independent
reproduction/red-team review, rights clearance for data and weights, and a signed archived release.
Those are external evidence obligations, not facts that model files can prove by themselves.
