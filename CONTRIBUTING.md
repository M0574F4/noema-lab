# Contributing

Noema is intended to be a benchmark platform, so contributions must preserve comparability as well
as functionality.

## Development Setup

```bash
uv sync
uv run python -m unittest discover -s tests
uv run noema recipe lint recipes/compressai_kodak_default.yaml
uv build --sdist --wheel
```

Optional model families live behind extras such as `compressai`, `foundation`, `vision`,
`wireless`, and `upstream-lic`. Core tests should not require heavyweight optional downloads.

## Pull Request Shape

Keep changes reviewable:

- release metadata, CI, packaging, docs, benchmark protocols, and model adapters should usually be
  separate PRs;
- every new operation needs typed `input_kinds`, `output_kinds`, a params schema, tests, and docs;
- every publishable benchmark change needs a benchmark version update or a new benchmark ID;
- every new external adapter contract needs a scaffold example and a validation test;
- benchmark recipes should pass `noema recipe lint` unless they are explicitly marked as smoke or
  experimental plumbing.

## Stable v1 Contracts

The stable public surface for benchmark submissions is:

- operation kind strings and recipe `step.output` wiring;
- canonical bit boundary contract: flat `np.uint8`, values `0` or `1`, one array element per bit;
- canonical symbol boundary contract: flat `np.complex64` channel symbols;
- benchmark pack ID/version, metric IDs, and generated result bundle files;
- run manifests with recipe hashes, operation contracts, environment, seed policy, and artifacts;
- external adapter manifest schema version `1`.

Experimental areas are marked in docs and metadata. They can change before a formal benchmark v1
freeze.

## Tests Expected

Before asking for review, run:

```bash
uv run python -m unittest discover -s tests
uv run noema recipe lint recipes/text_semantic_utf8_clean.yaml
uv run noema benchmark validate benchmarks/benchmark_v1/kodak_image_reconstruction_v1.yaml
```

For external adapters, also run the adapter manifest validator and at least one recipe or benchmark
that uses the adapter without editing Noema core source.
