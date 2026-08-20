# Traceability Profile Governance

Noema's strongest local verifier is selected by the benchmark metadata field
`traceability_profile_requested`. The field asks Noema to apply a fixed,
content-identified minimum set of traceability checks. It does **not** claim that a result is
scientifically valid, fairly compared, independently reproduced, legally cleared, archived, or
accepted for publication.

The current profile is `noema.traceability.v2`. A canonical benchmark that requests it must bind
both that identifier and the exact digest returned by
`publication_verification_profile_binding()`. The digest covers the normative I1--I6 definition and
the verifier's check-to-predicate semantics.

The current local binding is content-digest based, not a signature. An officially distributed
profile should additionally be anchored by a repository or release signature, or an equivalent
transparency and custody mechanism; local digest conformance alone does not establish who authorized
or signed the profile.

## Ownership and change control

The Noema maintainers own the profile definition in
`noema_lab.core.publication_profile`, its public schema bindings, and its conformance tests. A
released profile identifier is immutable:

- editorial clarification that does not change machine semantics may retain the identifier only
  when the canonical digest is unchanged;
- adding, removing, weakening, or reinterpreting a required predicate creates a new profile
  identifier and digest;
- study authors may add stricter checks, but may not remove profile requirements while reporting a
  current-profile pass; and
- a profile change is reviewed together with schemas, verifier behavior, migration notes, and
  positive and negative conformance tests.

## Request-field migration

`publication_ready` is a deprecated compatibility alias for
`traceability_profile_requested` in benchmark packs and benchmark results. Stored documents with
only the old field remain readable. If both fields are present, both must be booleans and must
agree; disagreement fails closed. New readers and result writers use the reader-accurate field as
canonical; writers do not add the deprecated alias to new results.

This migration applies only to the benchmark profile trigger. Dataset-rights records and returned
model packages have separate publication-readiness fields with different meanings and are not
renamed by this policy.

Removal of the alias requires a separately announced schema-breaking release after shipped packs,
stored results, static-demo consumers, and external integrations have had a documented migration
window.

## Conformance levels

Noema reports local conformance without collapsing external obligations into one badge:

| State | Meaning |
|---|---|
| Profile not requested | Local evidence may still verify, but it is not evaluated against the strongest profile. |
| Requested, `current_profile_fail` | The exact current profile was missing, substituted, or at least one applicable local predicate failed. |
| Requested, `current_profile_pass` | The canonical tier, exact profile binding, and every applicable local I1--I6 predicate passed. |
| External status | Archival custody, independent reproduction, scientific fairness, legal clearance, and venue acceptance remain separate and are not established by a local pass. |

The verifier report preserves these dimensions explicitly. Consumers should use
`traceability_profile_requested`, `verdict_class`, `profile_binding_status`, and the individual
status-vector fields instead of inventing a generic “publication ready” state.
