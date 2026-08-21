# Launch figures and tables

This is the generated visual system for Noema's launch demonstration. Figures F0–F5 and tables
T0–T2 derive from the repository-root [`launch_evidence.json`](launch_evidence.md); none is a
second source of experimental truth. They are documentation and launch assets only. Selecting any
of them for the paper remains a later editorial decision after manuscript review.

Regenerate and verify the complete set with:

```bash
uv run --frozen python tools/generate_launch_evidence.py --check
uv run --frozen python tools/generate_launch_assets.py
uv run --frozen python tools/generate_launch_assets.py --check
```

The machine-readable asset inventory is
[`docs/_static/launch/manifest.json`](_static/launch/manifest.json). Every quantitative figure and
table carries the experimental disclosure or an adjacent note, uses observed min/max terminology,
and preserves the calibrated oracle's diagnostic-reference role.

## Figures required — F0–F5

### F0 · Launch hero

```{figure} _static/launch/f0-launch-hero.svg
:alt: Noema launch hero stating Make the comparison traceable, with the canonical experimental receiver headline and its non-publication-ready disclosure
:class: noema-launch-figure

The repository and webpage hero. It combines the product thesis with a bounded evidence card.
```

### F1 · Comparison anatomy

```{figure} _static/launch/f1-comparison-anatomy.svg
:alt: Five comparison boundaries—condition, pairing, aggregation, metric, and role—feeding a claim that survives the audit
:class: noema-launch-figure

The minimum visual explanation of why the surrounding contract matters more than a percentage.
```

### F2 · Contract-to-evidence chain

```{figure} _static/launch/f2-contract-to-evidence.svg
:alt: Noema chain from typed protocol through execution, accounting, retained evidence, and launch projection
:class: noema-launch-figure

The system figure for documentation, talks, and the future launch video.
```

### F3 · Canonical experiment figure

```{figure} _static/launch/f3-receiver-ber-vs-snr.svg
:alt: Log-scale pre-decoder BER versus SNR for uncompensated QPSK, a calibrated diagnostic oracle, and the learned receiver; the learned curve includes an observed min-max band over paired seeds
:class: noema-launch-figure

The canonical launch result. Its band is the observed minimum and maximum across three paired
held-out seeds—not a confidence interval.
```

### F4 · Break the comparison

```{figure} _static/launch/f4-break-the-comparison.svg
:alt: Four contract failures around a broken percentage claim: condition mismatch, seed cherry-picking, metric mismatch, and hidden oracle information
:class: noema-launch-figure

The static companion to the [interactive flagship demo](break_the_comparison.md).
```

### F5 · One source, many public outputs

```{figure} _static/launch/f5-one-source-many-surfaces.svg
:alt: Launch evidence JSON at the center supplying the README, documentation, web demo, experiment figure, tables, and recorded launch video
:class: noema-launch-figure

The launch-production rule: every displayed number and warning comes from one generated source.
```

## Tables required — T0–T2

CSV is the transport format; Markdown is the human-readable projection rendered below.

```{include} _static/launch/tables/t0-result-summary.md
```

[Open T0 as CSV](_static/launch/tables/t0-result-summary.csv).

```{include} _static/launch/tables/t1-experiment-contract.md
```

[Open T1 as CSV](_static/launch/tables/t1-experiment-contract.csv).

```{include} _static/launch/tables/t2-claim-guardrails.md
```

[Open T2 as CSV](_static/launch/tables/t2-claim-guardrails.csv).

## Launch video

The [recorded learned QPSK I/Q calibration walkthrough](https://www.youtube.com/watch?v=bKNXS_vHLHc)
shows the clean installation, training-contract export, dataset capture, external training,
returned-model validation, and UI comparison against both declared baselines. The reviewed video
is hosted externally; no local recording or edit file is committed to the source repository.

The versioned [recording runbook](launch_video.md) preserves the demonstrated steps, while
`launch_evidence.json` records the stable video identity and URLs alongside the numerical data
paths used by the public README, documentation, figures, tables, and demo.
