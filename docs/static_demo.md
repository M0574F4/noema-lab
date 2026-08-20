# Static Demo Pages

GitHub Pages cannot run the Noema API server, execute recipes, or inspect local artifacts on a user's
machine. It can still host useful read-only demos: completed experiment summaries, figures, recipe
metadata, and links to retained bundle files.

Noema's static demo assets live under `docs/demo/` and are copied into the built documentation site by
Sphinx. In the published site, the demo is available at `demo/index.html` relative to the docs root.
The hosted explorer uses the dashboard Results stylesheet and the same interactive chart runtime,
chart-data scripts, CSV tables, and snapshot manifests as the chart-bearing documentation tutorials.
`tools/generate_hosted_demo_catalog.py` discovers those tutorials and generates
`docs/demo/catalog.json`, so the landing page does not contain copied metric values or a manually
maintained experiment list.

## What Static Demos Can Do

- Show already-generated benchmark curves and result cards.
- Explain which files a real run bundle contains, such as `manifest.json`, `summary.json`,
  `recipe.json`, `metrics.csv`, and plotted-data CSV files.
- Demonstrate how Noema visualizes a completed experiment before a researcher installs the tool.
- Link back to tutorials, benchmark packs, and adapter documentation.

## Read-Only Limitations

- Run recipes.
- Edit recipes.
- Open local `.noema/` artifacts from the visitor's machine.
- Verify a visitor's local run bundle.
- Claim demo data is a verified scientific result unless it comes from a real verified bundle.

When a visitor clicks a live-only control, the demo should show a clear message:

> This is a static GitHub Pages demo. Install Noema or run the local UI to execute recipes, edit configurations, or inspect live artifacts.

## Publish A Stored Benchmark

The publication input is a completed benchmark `result_id`, not a recipe and not a live UI state.
Noema verifies the benchmark and every referenced run before copying any public evidence:

```bash
noema benchmark verify <result_id>
noema benchmark publish <result_id> \
  --slug <metadata.demo.slug> \
  --out docs/demo/experiments/<metadata.demo.slug>
```

When a benchmark declares `metadata.demo`, that specification selects the research question,
method grouping, table metrics, plots, and compact training evidence. The publisher reads only
stored metrics and integrity-bound files. It does not execute recipes, capture data, evaluate a
checkpoint, or train a model.

Each generated directory contains:

- `index.html`, a self-contained result page that displays verification status, benchmark tier, and
  the separate strongest-traceability-profile request;
- `data/demo.json` and `data/metrics.csv`, the machine-readable publication payload and all stored
  scalar metrics;
- `data/plots/*.csv` and `figures/*.svg`, so every plotted point remains inspectable;
- compact recipe, run, verification, and declared training evidence under `evidence/`;
- `publication-manifest.json`, with SHA-256 values for every generated file.

Publishing under `docs/demo/experiments/<slug>` also updates the deterministic
`docs/demo/experiments/index.json` registry, which the hosted demo landing page discovers
automatically. Re-publishing unchanged stored evidence produces the same page payload and file
manifest, but an existing output directory is never overwritten implicitly. A deterministic
re-publish therefore requires repeating the command with `--force`; use it only when intentionally
replacing the existing publication for that slug.
Verifier warnings are rejected unless the researcher explicitly passes `--allow-warnings`; invalid
or incomplete bundles are never publishable.

Verification and external publication status are intentionally distinct. A valid page does not
imply publishable evidence. Only a `canonical` benchmark with
`traceability_profile_requested: true` and the exact profile binding can receive a local
current-profile pass, and the page exposes the tier, request, and verifier status rather than
collapsing them into one badge. The generated JSON retains `publication_ready` only as a deprecated
projection alias during the compatibility window.

## Local Preview

Build the docs site, then open the generated demo page:

```bash
uv run sphinx-build -W -b html docs docs/_build/html
xdg-open docs/_build/html/demo/index.html
```

The generated `catalog.js` packages the discovered catalog and CSV table text for browsers that
block `fetch()` on `file://` pages. `catalog.json` and the original CSV files remain available as
the canonical machine-readable and downloadable evidence. The demo remains static HTML, CSS,
JavaScript, CSV, and JSON, so the same build works locally and on GitHub Pages without a backend.
