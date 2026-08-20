# Security Policy

## Supported Versions

No tagged software release is supported yet. Security fixes are applied to the `main` development
branch and will be backported only after the first supported release is published.

## Reporting a Vulnerability

Do not open a public issue for security-sensitive reports. Use GitHub's private vulnerability
reporting form for this repository:

<https://github.com/M0574F4/noema-lab/security/advisories/new>

If that form is unavailable, contact the acting maintainer through
<https://github.com/M0574F4> without posting exploit details publicly. The project aims to
acknowledge a private report within seven calendar days and provide an initial disposition within
fourteen days.

Please include:

- affected Noema version or commit;
- operating system and Python version;
- exact command, recipe, adapter, or benchmark involved;
- whether untrusted adapter code, datasets, archives, or model files are required to reproduce;
- a minimal reproduction when possible.

Noema executes local Python adapter code by design. Treat third-party adapters, model checkpoints,
datasets, and benchmark bundles as code/data from their authors. Run untrusted contributions in an
isolated environment.

Network-fetched executable model code is disabled by default. A user who explicitly enables an
upstream integration must verify its pinned revision and digest before use.
