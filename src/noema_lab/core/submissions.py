from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)
from noema_lab.core.verification import verify_benchmark_result

JsonDict = Dict[str, Any]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SubmissionValidationError(ValueError):
    pass


def validate_submission_bundle(
    path: Path,
    *,
    registry: Optional[OperationRegistry] = None,
    trusted_bundle_root_sha256: Optional[str] = None,
) -> JsonDict:
    """Verify a submission and label the trust boundary explicitly.

    Without an out-of-band ``trusted_bundle_root_sha256`` this function proves
    internal consistency against Noema's full benchmark verifier. It does not
    authenticate a malicious producer. Supplying a trusted pin additionally
    proves that the inspected bytes match that external anchor.
    """

    raw_submission_path = path.expanduser()
    if raw_submission_path.is_symlink():
        raise SubmissionValidationError("submission file must not be a symlink")
    submission_path = raw_submission_path.resolve()
    data = _load_mapping(submission_path)
    errors: List[str] = []
    warnings: List[str] = []
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        errors.append("schema_version must be 1")
    method = _mapping(data.get("method"))
    benchmark = _mapping(data.get("benchmark"))
    runtime = _mapping(data.get("runtime"))
    reproducibility = _mapping(data.get("reproducibility"))
    raw_result_bundle = data.get("result_bundle")

    _require_string(errors, method, "method.name")
    authors = method.get("authors")
    if (
        not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, str) or not author.strip() for author in authors)
    ):
        errors.append("method.authors must be a non-empty list of names")
    _require_string(errors, benchmark, "benchmark.id")
    _require_string(errors, benchmark, "benchmark.version")
    _require_string(errors, runtime, "runtime.hardware")
    _require_string(errors, runtime, "runtime.software")
    _require_string(errors, reproducibility, "reproducibility.recipe_sha256")
    _require_string(errors, reproducibility, "reproducibility.seed_policy")
    if not isinstance(reproducibility.get("seeds"), list):
        errors.append("reproducibility.seeds must be a list")
    if not isinstance(raw_result_bundle, str) or not raw_result_bundle.strip():
        errors.append("result_bundle is required and must be a relative path")

    division = data.get("comparison_division")
    if division not in {"closed", "open"}:
        errors.append("comparison_division must be `closed` or `open`")

    result: Optional[JsonDict] = None
    result_path: Optional[Path] = None
    result_dir: Optional[Path] = None
    verification: Optional[JsonDict] = None
    if isinstance(raw_result_bundle, str) and raw_result_bundle.strip():
        try:
            result_dir, result_path = _resolve_result_bundle(
                submission_path.parent,
                raw_result_bundle,
            )
        except SubmissionValidationError as exc:
            errors.append(str(exc))
        else:
            try:
                loaded = load_strict_yaml_or_json(result_path)
                if not isinstance(loaded, Mapping):
                    raise ValueError("result.json must contain an object")
                result = dict(loaded)
            except (OSError, ValueError, TypeError) as exc:
                errors.append("cannot read result bundle: %s" % exc)

    if result is not None and result_dir is not None:
        result_benchmark = _mapping(result.get("benchmark"))
        if benchmark.get("id") and result_benchmark.get("id") != benchmark.get("id"):
            errors.append(
                "benchmark.id does not match result bundle: %s != %s"
                % (benchmark.get("id"), result_benchmark.get("id"))
            )
        if benchmark.get("version") and str(result_benchmark.get("version") or "") != str(
            benchmark.get("version")
        ):
            errors.append(
                "benchmark.version does not match result bundle: %s != %s"
                % (benchmark.get("version"), result_benchmark.get("version"))
            )
        if result.get("status") != "completed":
            errors.append("result bundle status must be completed")

        try:
            verifier_store = LocalStore(result_dir.parent)
            # verify_benchmark_result only requires a benchmark-root + result
            # directory when deep_backing_runs=False; result-local snapshots
            # remain authoritative.
            verifier_store.benchmarks_dir = result_dir.parent
            verification = verify_benchmark_result(
                verifier_store,
                result_dir.name,
                registry=registry,
                deep_backing_runs=False,
            )
        except Exception as exc:
            errors.append("full benchmark verification failed to run: %s" % exc)
        else:
            if verification.get("status") != "valid":
                messages = list(verification.get("errors") or []) + list(
                    verification.get("warnings") or []
                )
                errors.append(
                    "full benchmark verifier did not return valid: %s"
                    % ("; ".join(str(item) for item in messages) or verification.get("status"))
                )

        _cross_check_recipe_hashes(errors, reproducibility, result)
        _cross_check_seed_evidence(errors, reproducibility, result_dir, result)
        _cross_check_division(errors, str(division or ""), result_dir)

    adapter = _mapping(data.get("adapter"))
    if adapter and not adapter.get("version"):
        warnings.append("adapter.version is recommended for external method submissions")

    bundle_root_sha256: Optional[str] = None
    if result_dir is not None and result_dir.is_dir():
        try:
            bundle_root_sha256 = _bundle_root_sha256(result_dir)
        except (OSError, SubmissionValidationError, TypeError, ValueError) as exc:
            errors.append("cannot identify result bundle bytes: %s" % exc)

    pinned = trusted_bundle_root_sha256 is not None
    if pinned:
        trusted = str(trusted_bundle_root_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(trusted):
            errors.append("trusted_bundle_root_sha256 must be a lowercase SHA-256")
        elif bundle_root_sha256 != trusted:
            errors.append(
                "result bundle does not match the out-of-band trusted root: %s != %s"
                % (bundle_root_sha256, trusted)
            )

    valid = not errors
    if not valid:
        verdict = "rejected"
    elif pinned:
        verdict = "verified_pinned_bundle_integrity"
    else:
        verdict = "verified_internal_consistency_untrusted_producer"
        warnings.append(
            "no out-of-band bundle-root pin was supplied; this verdict does not authenticate an adversarial producer"
        )

    return {
        "schema_version": 1,
        "status": "valid" if valid else "invalid",
        "verdict": verdict,
        "threat_model": (
            "adversarial_transport_with_out_of_band_pin"
            if pinned
            else "honest_producer_or_accidental_drift"
        ),
        "producer_authenticated": False,
        "submission": str(submission_path),
        "result_json": str(result_path) if result_path else None,
        "result_bundle_root_sha256": bundle_root_sha256,
        "errors": errors,
        "warnings": warnings,
        "benchmark": dict(benchmark),
        "method": dict(method),
        "verification": verification,
    }


def _resolve_result_bundle(base: Path, value: str) -> Tuple[Path, Path]:
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SubmissionValidationError(
            "result_bundle must be a normalized relative path confined to the submission directory"
        )
    root = base.resolve(strict=True)
    candidate = root / relative
    _reject_symlink_path(root, candidate)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SubmissionValidationError(
            "result_bundle is missing or escapes the submission directory: %s" % value
        ) from exc
    result_dir = resolved if resolved.is_dir() else resolved.parent
    result_path = resolved / "result.json" if resolved.is_dir() else resolved
    if result_path.name != "result.json" or result_path.is_symlink() or not result_path.is_file():
        raise SubmissionValidationError(
            "result_bundle does not contain a safe result.json: %s" % result_path
        )
    return result_dir, result_path


def _reject_symlink_path(root: Path, candidate: Path) -> None:
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise SubmissionValidationError(
                "result_bundle path traverses a symlink: %s" % current
            )


def _cross_check_recipe_hashes(
    errors: List[str],
    reproducibility: JsonDict,
    result: JsonDict,
) -> None:
    rows = result.get("recipes")
    if not isinstance(rows, list) or not rows:
        errors.append("verified submission requires at least one benchmark recipe result")
        return
    actual: JsonDict = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append("result recipe %d is not an object" % index)
            continue
        row_id = str(row.get("id") or index)
        digest = str(row.get("recipe_sha256") or "").lower()
        if not _SHA256_RE.fullmatch(digest):
            errors.append("result recipe %s has no valid recipe_sha256" % row_id)
            continue
        if row_id in actual:
            errors.append("result recipe id is duplicated: %s" % row_id)
            continue
        actual[row_id] = digest
    if len(actual) == 1:
        declared = str(reproducibility.get("recipe_sha256") or "").lower()
        expected = next(iter(actual.values()))
        if declared != expected:
            errors.append(
                "reproducibility.recipe_sha256 does not match result evidence: %s != %s"
                % (declared, expected)
            )
        return
    declared_map = reproducibility.get("recipe_sha256s")
    if (
        not isinstance(declared_map, Mapping)
        or any(not isinstance(key, str) for key in declared_map)
        or {key: str(value).lower() for key, value in declared_map.items()} != actual
    ):
        errors.append(
            "multi-recipe submissions require reproducibility.recipe_sha256s matching every result recipe"
        )


def _cross_check_seed_evidence(
    errors: List[str],
    reproducibility: JsonDict,
    result_dir: Path,
    result: JsonDict,
) -> None:
    policies: List[JsonDict] = []
    for index, row in enumerate(result.get("recipes") or []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status") or "") not in {"completed", "rejected_resource_budget"}:
            continue
        descriptor = row.get("run_evidence_snapshot")
        if not isinstance(descriptor, Mapping):
            errors.append("result recipe %d is missing run_evidence_snapshot" % index)
            continue
        root_value = descriptor.get("root")
        if not isinstance(root_value, str):
            errors.append("result recipe %d has invalid run-evidence root" % index)
            continue
        try:
            evidence_root, manifest_path = _resolve_confined_file(
                result_dir,
                root_value,
                "manifest.json",
            )
            del evidence_root
            manifest = load_strict_yaml_or_json(manifest_path)
        except (OSError, ValueError, SubmissionValidationError) as exc:
            errors.append("cannot inspect seed evidence for recipe %d: %s" % (index, exc))
            continue
        policy = manifest.get("seed_policy") if isinstance(manifest, Mapping) else None
        if not isinstance(policy, Mapping):
            errors.append("result recipe %d has no recorded seed_policy" % index)
            continue
        policies.append(dict(policy))
    if not policies:
        return
    raw_actual_seeds = {policy.get("master_seed") for policy in policies}
    invalid_actual_seeds = any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in raw_actual_seeds
    )
    if invalid_actual_seeds:
        errors.append("run evidence must record concrete non-negative integer seeds")
        actual_seeds: List[int] = []
    else:
        actual_seeds = sorted(raw_actual_seeds)
    declared_seeds = reproducibility.get("seeds")
    if (
        not isinstance(declared_seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in declared_seeds)
        or len(declared_seeds) != len(set(declared_seeds))
        or sorted(declared_seeds) != actual_seeds
    ):
        errors.append(
            "reproducibility.seeds does not exactly match run-evidence master seeds"
        )
    actual_policy_hashes = sorted({canonical_json_sha256(policy) for policy in policies})
    declared_hashes = reproducibility.get("seed_policy_sha256s")
    human_policy = str(reproducibility.get("seed_policy") or "").lower()
    if len(actual_policy_hashes) == 1 and human_policy == actual_policy_hashes[0]:
        return
    if (
        not isinstance(declared_hashes, list)
        or sorted(str(item).lower() for item in declared_hashes) != actual_policy_hashes
    ):
        errors.append(
            "reproducibility.seed_policy_sha256s must match every recorded seed policy"
        )


def _resolve_confined_file(
    root: Path,
    relative_root: str,
    filename: str,
) -> Tuple[Path, Path]:
    relative = Path(relative_root)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SubmissionValidationError("run-evidence root is not a safe relative path")
    candidate_root = root / relative
    _reject_symlink_path(root.resolve(strict=True), candidate_root)
    resolved_root = candidate_root.resolve(strict=True)
    resolved_root.relative_to(root.resolve(strict=True))
    path = resolved_root / filename
    if path.is_symlink() or not path.is_file():
        raise SubmissionValidationError("run-evidence file is missing or unsafe: %s" % path)
    return resolved_root, path


def _cross_check_division(errors: List[str], division: str, result_dir: Path) -> None:
    benchmark_path = result_dir / "benchmark.json"
    if benchmark_path.is_symlink() or not benchmark_path.is_file():
        errors.append("result bundle requires a safe benchmark.json for division verification")
        return
    try:
        benchmark = load_strict_yaml_or_json(benchmark_path)
    except (OSError, ValueError) as exc:
        errors.append("cannot read benchmark.json for division verification: %s" % exc)
        return
    metadata = benchmark.get("metadata") if isinstance(benchmark, Mapping) else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    declared = metadata.get("comparison_division")
    allowed = metadata.get("allowed_comparison_divisions")
    if declared in {"closed", "open"}:
        allowed_values = [declared]
    elif isinstance(allowed, list) and allowed and all(item in {"closed", "open"} for item in allowed):
        allowed_values = list(allowed)
    else:
        errors.append(
            "benchmark.json must freeze metadata.comparison_division or allowed_comparison_divisions"
        )
        return
    if division not in allowed_values:
        errors.append(
            "comparison_division %s is not allowed by frozen benchmark.json" % division
        )


def _bundle_root_sha256(root: Path) -> str:
    resolved_root = root.resolve(strict=True)
    inventory: List[JsonDict] = []
    paths = sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix())
    for path in paths:
        if path.is_symlink():
            raise SubmissionValidationError("bundle contains a symlink: %s" % path)
        if not path.is_file():
            continue
        before = _stable_file_identity(path)
        digest = file_sha256(path)
        after = _stable_file_identity(path)
        if before != after:
            raise SubmissionValidationError("bundle file changed while hashing: %s" % path)
        inventory.append(
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "size_bytes": before[2],
                "sha256": digest,
            }
        )
    paths_after = sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix())
    if [path.as_posix() for path in paths] != [path.as_posix() for path in paths_after]:
        raise SubmissionValidationError("bundle inventory changed while hashing")
    if not inventory:
        raise SubmissionValidationError("result bundle contains no files")
    return canonical_json_sha256(inventory)


def _stable_file_identity(path: Path) -> Tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _load_mapping(path: Path) -> JsonDict:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("submission file does not exist or is unsafe: %s" % path)
    try:
        data = load_strict_yaml_or_json(path)
    except StructuredInputError as exc:
        raise SubmissionValidationError(
            "invalid submission YAML/JSON input %s: %s" % (path, exc)
        ) from exc
    if not isinstance(data, Mapping):
        raise SubmissionValidationError("submission must be a mapping")
    return dict(data)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _require_string(errors: List[str], data: JsonDict, dotted_key: str) -> None:
    key = dotted_key.split(".")[-1]
    if not isinstance(data.get(key), str) or not str(data.get(key)).strip():
        errors.append("%s is required" % dotted_key)
