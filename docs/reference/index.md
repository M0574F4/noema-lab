# Reference

This section is where implementation-adjacent source-of-truth metadata is exposed as researcher-facing documentation.

```{toctree}
:maxdepth: 2
:titlesonly:

generated/cli
generated/operations
api
schemas
```

## Generated References

`generated/cli.md` is produced from CLI help. `generated/operations.md` is produced from the operation registry. Do not edit those files by hand; run:

```bash
uv run python tools/generate_docs_reference.py
```

CI checks that the generated files are up to date.
