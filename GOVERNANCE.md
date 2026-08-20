# Governance

Noema uses maintainer-led governance until the project has a broader contributor base.

The acting maintainer is the repository owner, [M0574F4](https://github.com/M0574F4). Maintainer
changes are recorded in this file through a reviewed pull request. If the acting maintainer is
inactive for 90 days, an established contributor may propose succession publicly in the issue
tracker; security and conduct reports continue to use their private channels.

## Maintainer Responsibilities

Maintainers are responsible for:

- preserving benchmark comparability and reproducibility;
- reviewing changes to stable operation, artifact, adapter, and benchmark contracts;
- deciding when a benchmark protocol is frozen or superseded;
- coordinating releases, changelog entries, and security responses;
- labeling experimental functionality clearly.

## Decision Process

Routine fixes can be merged after normal review. Changes to public benchmark protocols, stable
artifact kinds, adapter manifest schema, or metric definitions require explicit maintainer approval
and must document compatibility impact.

If a benchmark changes in a way that affects comparability, create a new benchmark version instead
of silently changing the old one.

## External Benchmark Submissions

Benchmark submissions are accepted only when they include the required metadata, recipe hash,
result bundle, and reproducibility manifest. Maintainers may reject submissions that modify
benchmark data, metrics, channel/rate accounting, or closed-division rules.

Open a submission proposal at <https://github.com/M0574F4/noema-lab/issues/new> before transferring
large artifacts. The issue must contain no private data or embargoed results. After local validation,
submit reviewable metadata and small fixtures by pull request; use a maintainer-approved immutable
artifact location for large bundles. A rejection must identify the failed protocol or evidence gate,
and the submitter may request reconsideration with corrected evidence.
