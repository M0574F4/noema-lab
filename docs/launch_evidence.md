# Canonical launch evidence

Noema has one machine-readable source for launch-facing quantitative material:
[`launch_evidence.json`](https://github.com/M0574F4/noema-lab/blob/main/launch_evidence.json).
README claims, the documentation landing page, the public webpage, experiment figures, result
tables, and video overlays must derive their numbers from that projection instead of copying values
from tutorial prose.

The current projection selects the synthetic QPSK I/Q-calibration demonstration. It contains all
63 per-run integer error counts and identities, their deterministic seven-cell aggregation, the
headline selection rule, figure/table/video data contracts, and the warning and limitations
inherited from the retained benchmark manifest. The selected CSV bytes and the embedded semantic
observation grid are both frozen by digest. Because the observations are embedded, the arithmetic
remains verifiable when the private authoring inputs are absent from a public-source export.

Scientific status and distribution clearance are deliberately independent:

- `scientific_status` remains `completed_experimental_benchmark`, `warning`, and
  `publication_ready: false`. Completing a rights review cannot upgrade those claims.
- `distribution_clearance` records the candidate state attached to the frozen evidence in the
  private authoring repository. It is provenance metadata, not a scientific-quality claim.
- Observed minima and maxima are display ranges across three paired held-out seeds. They are not
  confidence intervals or population guarantees.
- The calibrated I/Q oracle is a diagnostic reference with calibration knowledge, not a deployable
  same-information competitor.

## Verify the public projection

The public repository verifies the self-contained projection and all generated launch assets:

```bash
uv run --frozen python tools/generate_qpsk_iq_calibration_demo_assets.py --check
uv run --frozen python tools/generate_launch_evidence.py --verify
uv run --frozen python tools/generate_launch_assets.py --check
```

`--verify` validates the standalone JSON's schema, self-hash, frozen observation-grid digest,
coordinate closure, arithmetic, embedded distribution state, and consumer contract. Regeneration
of the projection remains in the private authoring repository because it also consumes private
release-control inputs; the public repository contains the frozen benchmark CSV and manifest needed
to inspect the demonstration itself.

The strict contract is
[`schemas/launch_evidence.schema.json`](https://github.com/M0574F4/noema-lab/blob/main/schemas/launch_evidence.schema.json),
and the deterministic producer is
[`tools/generate_launch_evidence.py`](https://github.com/M0574F4/noema-lab/blob/main/tools/generate_launch_evidence.py).
The canonical F3 figure and T0 table have generated paths in the projection. The full
[F0–F5 and T0–T2 launch set](launch_assets.md) is rendered by
[`tools/generate_launch_assets.py`](https://github.com/M0574F4/noema-lab/blob/main/tools/generate_launch_assets.py)
and closed by a digest manifest. The presentation contract also records the stable identity and
external URLs of the [learned QPSK I/Q calibration walkthrough](https://www.youtube.com/watch?v=bKNXS_vHLHc);
the recording itself remains outside the source repository.
