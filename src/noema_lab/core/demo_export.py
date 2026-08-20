from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmark_evidence import (
    BenchmarkEvidenceError,
    validate_benchmark_training_evidence_snapshot,
)
from noema_lab.core.benchmark_run_evidence import (
    BenchmarkRunEvidenceError,
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.benchmark_plots import _mean_std_ci95
from noema_lab.core.operations import OperationRegistry
from noema_lab.core.publication_profile import traceability_profile_requested
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)
from noema_lab.core.verification import verify_benchmark_result

JsonDict = Dict[str, Any]

DEMO_SCHEMA_VERSION = 1
PUBLICATION_MANIFEST_SCHEMA_VERSION = 1
DEMO_KIND = "noema.static_benchmark_demo"
PUBLICATION_MANIFEST_KIND = "noema.demo_publication_manifest"

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be185d",
    "#4f46e5",
)
_LIKELY_X_METRICS = (
    "channel.snr_db",
    "snr_db",
    "rate.payload_bpp",
    "rate.framed_bpp",
    "rate.coded_bpp",
    "rate.padded_bpp",
    "rate.bpp",
    "quality.rate_bpp",
    "channel.bandwidth_ratio",
    "channel.channel_uses_per_source_pixel",
    "resource.average_power_budget",
    "average_power_budget",
)
_TRAINING_EVIDENCE_FIELDS = (
    "trained_artifact_manifest",
    "training_history",
    "evaluation_metrics",
)
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_VOLATILE_VERIFICATION_KEYS = {
    "checked_at",
    "checked_at_utc",
    "generated_at",
    "generated_at_utc",
    "verified_at",
    "verified_at_utc",
}


class DemoExportError(RuntimeError):
    """Raised when a stored benchmark cannot be safely published."""


def publish_benchmark_demo(
    store: LocalStore,
    result_id: str,
    out_dir: Path,
    slug: str,
    *,
    registry: Optional[OperationRegistry] = None,
    project_root: Optional[Path] = None,
    force: bool = False,
    allow_warnings: bool = False,
) -> JsonDict:
    """Publish a verified benchmark result as deterministic static files.

    This function only reads completed benchmark/run evidence. It never invokes
    recipe execution, dataset capture, model export, or training.
    """

    slug = _validated_slug(slug, "demo slug")
    destination = Path(out_dir)
    project_root = (project_root or Path.cwd()).resolve()
    result_dir = store.get_benchmark_result_dir(result_id)
    _validate_destination(destination, store)
    _require_file(result_dir / "result.json", "benchmark result")
    _require_file(result_dir / "benchmark.json", "benchmark definition")

    benchmark_verification = verify_benchmark_result(store, result_id, registry=registry)
    _require_publishable_verification(
        benchmark_verification,
        "benchmark result %s" % result_id,
        allow_warnings=allow_warnings,
    )

    result = store.get_benchmark_result(result_id)
    benchmark_source = store.read_json(result_dir / "benchmark.json")
    _require_completed_result(result, result_id)
    benchmark = dict(result.get("benchmark") or {})
    demo_spec = _normalize_demo_spec(benchmark, slug)
    try:
        run_evidence = validate_benchmark_run_evidence_snapshots(
            result_dir,
            result,
        )
    except BenchmarkRunEvidenceError as exc:
        raise DemoExportError(
            "benchmark run-evidence snapshot is invalid: %s" % exc
        ) from exc
    snapshot_entries = {
        int(item["entry_index"]): item
        for item in run_evidence.get("entries") or []
        if isinstance(item, Mapping) and isinstance(item.get("entry_index"), int)
    }
    redaction_roots = _redaction_roots(store, project_root)
    if demo_spec.get("training_evidence"):
        try:
            snapshot = validate_benchmark_training_evidence_snapshot(
                result_dir,
                result,
                benchmark_source=benchmark_source,
            )
        except BenchmarkEvidenceError as exc:
            raise DemoExportError(
                "benchmark training-evidence snapshot is invalid: %s" % exc
            ) from exc
        if not snapshot.get("present"):
            raise DemoExportError(
                "benchmark training evidence requires a result-local snapshot"
            )
        evidence_base, evidence_root = result_dir, Path(snapshot["root"])
    else:
        evidence_base, evidence_root = project_root, project_root

    run_records: List[JsonDict] = []
    run_verifications: List[JsonDict] = []
    excluded_methods: List[JsonDict] = []
    used_method_ids: set[str] = set()
    benchmark_entries = {
        str(item.get("id")): dict(item)
        for item in (benchmark_source.get("recipes") or [])
        if isinstance(item, Mapping) and item.get("id")
    }
    for index, raw_entry in enumerate(result.get("recipes") or []):
        if not isinstance(raw_entry, Mapping):
            raise DemoExportError("benchmark recipe entry %d is not an object" % index)
        entry = dict(raw_entry)
        entry_status = str(entry.get("status") or "").strip().lower()
        run_id = _safe_directory_id(str(entry.get("run_id") or ""), "run")
        if entry_status not in {"completed", "rejected_resource_budget"}:
            raise DemoExportError("benchmark method %s is incomplete" % (entry.get("id") or index))

        snapshot_entry = snapshot_entries.get(index)
        if not isinstance(snapshot_entry, Mapping) or snapshot_entry.get("run_id") != run_id:
            raise DemoExportError(
                "benchmark method %s has no matching result-local run snapshot"
                % (entry.get("id") or index)
            )
        run_verification = dict(snapshot_entry.get("verification") or {})
        _require_publishable_verification(
            run_verification,
            "backing run %s" % run_id,
            allow_warnings=allow_warnings,
        )
        summary = dict(snapshot_entry.get("summary") or {})
        manifest = dict(snapshot_entry.get("manifest") or {})
        recipe = dict(snapshot_entry.get("recipe") or {})
        if str(summary.get("status") or "").lower() != "completed":
            raise DemoExportError("backing run %s is incomplete" % run_id)

        if entry_status == "rejected_resource_budget":
            admission = entry.get("resource_admission")
            if not isinstance(admission, Mapping) or admission.get("admitted") is not False:
                raise DemoExportError(
                    "benchmark method %s claims resource rejection without a "
                    "fail-closed admission record" % (entry.get("id") or index)
                )
            exclusion_id = _unique_slug(
                _slugify(str(entry.get("id") or "excluded-%d" % (index + 1))),
                used_method_ids,
            )
            used_method_ids.add(exclusion_id)
            excluded_methods.append(
                {
                    "id": exclusion_id,
                    "entry_id": str(entry.get("id") or exclusion_id),
                    "label": str(entry.get("label") or entry.get("id") or exclusion_id),
                    "role": str(entry.get("role") or "candidate"),
                    "run_id": run_id,
                    "status": "rejected_resource_budget",
                    "reason": "excluded_from_public_comparison",
                    "resource_admission": _safe_scalar_or_list(dict(admission)),
                    "metrics": _scalar_metrics(dict(entry.get("metrics") or {})),
                }
            )
            run_verifications.append(
                {
                    "method_id": exclusion_id,
                    "run_id": run_id,
                    "publication_status": "excluded_resource_budget",
                    "report": _redact_payload(
                        _strip_volatile_verification(run_verification),
                        redaction_roots,
                    ),
                }
            )
            continue

        method_id = _unique_slug(
            _slugify(str(entry.get("id") or entry.get("label") or "method-%d" % (index + 1))),
            used_method_ids,
        )
        used_method_ids.add(method_id)
        record = _run_record(
            entry,
            benchmark_entries.get(str(entry.get("id") or ""), {}),
            summary,
            manifest,
            recipe,
            run_id=run_id,
            method_id=method_id,
            index=index,
        )
        run_records.append(record)
        run_verifications.append(
            {
                "method_id": method_id,
                "run_id": run_id,
                "report": _redact_payload(
                    _strip_volatile_verification(run_verification),
                    redaction_roots,
                ),
            }
        )

    if not run_records:
        raise DemoExportError(
            "benchmark result %s contains no completed, resource-admitted methods"
            % result_id
        )

    excluded_methods = _redact_payload(excluded_methods, redaction_roots)

    methods = _redact_payload(_public_methods(run_records, demo_spec), redaction_roots)
    plots = _redact_payload(_plot_definitions(demo_spec, benchmark, methods), redaction_roots)
    training_evidence, training_payloads = _collect_training_evidence(
        demo_spec,
        run_records=run_records,
        evidence_base=evidence_base,
        evidence_root=evidence_root,
        redaction_roots=redaction_roots,
    )
    source_projection = _redact_payload({
        "schema_version": 1,
        "result_id": result_id,
        "benchmark": _source_benchmark_projection(benchmark, benchmark_source),
        "completed_at_utc": result.get("completed_at_utc"),
        "methods": [_source_method_projection(record) for record in run_records],
        "excluded_methods": excluded_methods,
        "demo": demo_spec,
        "training_evidence": training_evidence,
    }, redaction_roots)
    # The tutorial has already passed the URL/path allowlist. Preserve an
    # approved root-relative web path instead of treating it as a filesystem
    # path during generic evidence redaction.
    source_projection["demo"]["tutorial"] = demo_spec["tutorial"]
    source_bundle_sha256 = canonical_json_sha256(source_projection)

    verification_payload: JsonDict = {
        "schema_version": 1,
        "kind": "noema.demo_verification_evidence",
        "source_result_id": result_id,
        "benchmark": _redact_payload(
            _strip_volatile_verification(benchmark_verification),
            redaction_roots,
        ),
        "runs": run_verifications,
        "excluded_methods": excluded_methods,
    }
    key_metrics = _redact_payload(
        _key_metric_definitions(demo_spec, benchmark, methods),
        redaction_roots,
    )
    series_summaries = _series_summaries(methods, key_metrics)
    public_demo = _redact_payload(_public_demo_spec(demo_spec), redaction_roots)
    public_demo["tutorial"] = demo_spec["tutorial"]
    demo_payload: JsonDict = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "kind": DEMO_KIND,
        "slug": slug,
        "source_result_id": result_id,
        "source_bundle_sha256": source_bundle_sha256,
        "benchmark": _redact_payload(
            _public_benchmark(benchmark, benchmark_source),
            redaction_roots,
        ),
        "created_at_utc": result.get("created_at_utc"),
        "completed_at_utc": result.get("completed_at_utc"),
        "demo": public_demo,
        "verification": {
            "status": benchmark_verification.get("status"),
            "warning_count": len(benchmark_verification.get("warnings") or []),
            "evidence": "evidence/verification.json",
        },
        "key_metrics": key_metrics,
        "series": series_summaries,
        "runs": methods,
        "excluded_methods": excluded_methods,
        "plots": [],
        "training_evidence": training_evidence,
        "exports": {
            "metrics_csv": "data/metrics.csv",
            "excluded_methods_csv": "data/excluded_methods.csv",
        },
    }
    publication_sha256 = canonical_json_sha256(demo_payload)
    demo_payload["publication_sha256"] = publication_sha256

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s." % destination.name, dir=str(parent)))
    try:
        _write_json(staging / "evidence" / "verification.json", verification_payload)
        for record in run_records:
            method_id = str(record["method_id"])
            _write_json(
                staging / "evidence" / "recipes" / (method_id + ".json"),
                _redact_payload(record["recipe"], redaction_roots),
            )
            _write_json(
                staging / "evidence" / "runs" / (method_id + ".json"),
                _redact_payload(_compact_run_evidence(record), redaction_roots),
            )
        for relative_path, payload in sorted(training_payloads.items()):
            _write_json(staging / relative_path, payload)

        _write_metrics_csv(staging / "data" / "metrics.csv", methods)
        _write_excluded_methods_csv(
            staging / "data" / "excluded_methods.csv", excluded_methods
        )
        rendered_plots = []
        for plot in plots:
            rendered_plots.append(_write_plot(staging, plot, methods))
        demo_payload["plots"] = rendered_plots
        # Plot files are part of the publication model; recompute after their
        # deterministic relative paths and metric selections are known.
        demo_payload.pop("publication_sha256", None)
        publication_sha256 = canonical_json_sha256(demo_payload)
        demo_payload["publication_sha256"] = publication_sha256
        _write_json(staging / "data" / "demo.json", demo_payload)
        (staging / "index.html").write_text(_render_index_html(demo_payload), encoding="utf-8")

        manifest = _publication_manifest(
            staging,
            slug=slug,
            result_id=result_id,
            source_bundle_sha256=source_bundle_sha256,
            publication_sha256=publication_sha256,
        )
        _write_json(staging / "publication-manifest.json", manifest)
        _assert_tree_has_no_absolute_roots(staging, redaction_roots)
        _install_staging_directory(staging, destination, force=force)
        _maybe_update_demo_registry(destination, demo_payload)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "schema_version": 1,
        "kind": "noema.demo_publication",
        "slug": slug,
        "result_id": result_id,
        "out_dir": str(destination),
        "index": str(destination / "index.html"),
        "demo": str(destination / "data" / "demo.json"),
        "manifest": str(destination / "publication-manifest.json"),
        "source_bundle_sha256": source_bundle_sha256,
        "publication_sha256": publication_sha256,
        "verification_status": benchmark_verification.get("status"),
    }


def _validated_slug(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not value or value in {".", ".."} or not _SLUG_RE.fullmatch(value):
        raise DemoExportError("%s must contain only letters, digits, dots, underscores, and hyphens" % label)
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_").lower()
    return slug or "method"


def _unique_slug(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    index = 2
    while "%s-%d" % (candidate, index) in used:
        index += 1
    return "%s-%d" % (candidate, index)


def _safe_directory_id(value: str, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise DemoExportError("%s id must be a directory name: %s" % (label, value or "<missing>"))
    return value


def _validate_destination(destination: Path, store: LocalStore) -> None:
    if not destination.name or destination.name in {".", ".."}:
        raise DemoExportError("publication output must be a named directory")
    if destination.is_symlink():
        raise DemoExportError("publication output cannot be a symbolic link")
    resolved = destination.resolve(strict=False)
    for protected in (store.runs_dir.resolve(strict=False), store.benchmarks_dir.resolve(strict=False)):
        if resolved == protected or _is_relative_to(resolved, protected):
            raise DemoExportError("publication output cannot be inside stored run or benchmark bundles")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise DemoExportError("%s is missing: %s" % (label, path.name))


def _require_publishable_verification(report: Mapping[str, Any], label: str, *, allow_warnings: bool) -> None:
    status = str(report.get("status") or "invalid")
    if status == "invalid":
        errors = "; ".join(str(item) for item in (report.get("errors") or []))
        raise DemoExportError("%s failed verification%s" % (label, ": " + errors if errors else ""))
    if status == "warning" and not allow_warnings:
        warnings = "; ".join(str(item) for item in (report.get("warnings") or []))
        raise DemoExportError(
            "%s has verifier warnings%s; pass --allow-warnings to publish them visibly"
            % (label, ": " + warnings if warnings else "")
        )


def _require_completed_result(result: Mapping[str, Any], result_id: str) -> None:
    if str(result.get("kind") or "") != "noema.benchmark_result":
        raise DemoExportError("benchmark result %s has an unsupported kind" % result_id)
    if str(result.get("status") or "").lower() != "completed":
        raise DemoExportError("benchmark result %s is incomplete" % result_id)
    recipes = result.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise DemoExportError("benchmark result %s contains no completed methods" % result_id)


def _normalize_demo_spec(benchmark: Mapping[str, Any], slug: str) -> JsonDict:
    metadata = benchmark.get("metadata") if isinstance(benchmark.get("metadata"), Mapping) else {}
    raw_demo = metadata.get("demo") if isinstance(metadata, Mapping) else None
    if raw_demo is None:
        raw_demo = {}
    if not isinstance(raw_demo, Mapping):
        raise DemoExportError("benchmark.metadata.demo must be an object")
    demo = dict(raw_demo)
    if demo and int(demo.get("schema_version") or 1) != 1:
        raise DemoExportError("benchmark.metadata.demo schema_version must be 1")
    declared_slug = str(demo.get("slug") or "").strip()
    if declared_slug:
        declared_slug = _validated_slug(declared_slug, "benchmark.metadata.demo.slug")
        if declared_slug != slug:
            raise DemoExportError(
                "publication slug %s does not match benchmark.metadata.demo.slug %s" % (slug, declared_slug)
            )

    normalized: JsonDict = {
        "schema_version": 1,
        "slug": slug,
        "title": _plain_text(demo.get("title") or benchmark.get("name") or benchmark.get("id") or slug),
        "summary": _plain_text(demo.get("summary") or benchmark.get("description") or "Stored benchmark comparison."),
        "question": _plain_text(demo.get("question") or ""),
        "tutorial": _validated_tutorial_url(demo.get("tutorial") or ""),
        "held_constant": _safe_list(demo.get("held_constant") or []),
        "changed": _safe_list(demo.get("changed") or []),
        "primary_metric": _normalize_metric_spec(demo.get("primary_metric")),
        "comparison_axis": _normalize_metric_spec(demo.get("comparison_axis")),
        "series": _safe_list(demo.get("series") or []),
        "plots": _normalize_plot_specs(demo.get("plots") or []),
        "table_metrics": _normalize_metric_specs(demo.get("table_metrics") or []),
        "training_evidence": _normalize_training_evidence(demo.get("training_evidence") or []),
    }
    return normalized


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    raise DemoExportError("demo narrative values must be scalar text")


def _validated_tutorial_url(value: Any) -> str:
    tutorial = _plain_text(value).strip()
    if not tutorial:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 for char in tutorial):
        raise DemoExportError("benchmark.metadata.demo.tutorial contains control characters")
    if "\\" in tutorial:
        raise DemoExportError("benchmark.metadata.demo.tutorial must use URL-style forward slashes")
    parsed = urllib.parse.urlsplit(tutorial)
    scheme = parsed.scheme.lower()
    if scheme:
        if scheme not in {"http", "https"}:
            raise DemoExportError(
                "benchmark.metadata.demo.tutorial uses an unsafe URL scheme: %s" % scheme
            )
        if not parsed.netloc:
            raise DemoExportError("benchmark.metadata.demo.tutorial HTTP URL requires a host")
        if parsed.username is not None or parsed.password is not None:
            raise DemoExportError("benchmark.metadata.demo.tutorial HTTP URL cannot contain credentials")
        return tutorial
    if tutorial.startswith("//") or parsed.netloc:
        raise DemoExportError("benchmark.metadata.demo.tutorial cannot be a scheme-relative URL")
    return tutorial


def _safe_list(value: Any) -> List[Any]:
    if not isinstance(value, list):
        raise DemoExportError("demo list fields must be arrays")
    rows: List[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append({str(key): _safe_scalar_or_list(child) for key, child in sorted(item.items())})
        elif isinstance(item, (str, int, float, bool)) or item is None:
            rows.append(item)
        else:
            raise DemoExportError("demo list entries must be scalars or objects")
    return rows


def _safe_scalar_or_list(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_scalar_or_list(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe_scalar_or_list(child) for key, child in sorted(value.items())}
    raise DemoExportError("demo metadata contains an unsupported value")


def _normalize_metric_spec(value: Any) -> Optional[JsonDict]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return {"id": value}
    if not isinstance(value, Mapping) or not value.get("id"):
        raise DemoExportError("metric specs must be metric ids or objects with an id")
    return {
        key: _safe_scalar_or_list(child)
        for key, child in sorted(value.items())
        if key in {"id", "label", "unit", "direction", "format"}
    }


def _normalize_metric_specs(value: Any) -> List[JsonDict]:
    if not isinstance(value, list):
        raise DemoExportError("demo.table_metrics must be an array")
    return [spec for spec in (_normalize_metric_spec(item) for item in value) if spec]


def _normalize_plot_specs(value: Any) -> List[JsonDict]:
    if not isinstance(value, list):
        raise DemoExportError("demo.plots must be an array")
    plots: List[JsonDict] = []
    used: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DemoExportError("demo.plots entries must be objects")
        plot_id = _unique_slug(_slugify(str(item.get("id") or "plot-%d" % (index + 1))), used)
        used.add(plot_id)
        kind = str(item.get("kind") or "line").lower()
        if kind not in {"line", "scatter", "bar"}:
            raise DemoExportError("demo plot %s has unsupported kind %s" % (plot_id, kind))
        method_order = item.get("method_order") or []
        if isinstance(method_order, str):
            method_order = [part.strip() for part in method_order.split(",") if part.strip()]
        if not isinstance(method_order, list) or not all(isinstance(part, str) for part in method_order):
            raise DemoExportError("demo plot %s method_order must be an array of strings" % plot_id)
        method_order = [part.strip() for part in method_order]
        if any(not part for part in method_order):
            raise DemoExportError(
                "demo plot %s method_order labels cannot be empty" % plot_id
            )
        if len(set(method_order)) != len(method_order):
            raise DemoExportError(
                "demo plot %s method_order labels must be unique" % plot_id
            )
        x_metric = str(item.get("x") or "").strip()
        if kind == "bar" and x_metric:
            raise DemoExportError(
                "demo bar plot %s cannot declare x; use a line/scatter plot or "
                "one categorical bar per series" % plot_id
            )
        plot: JsonDict = {
            "id": plot_id,
            "title": _plain_text(item.get("title") or plot_id.replace("-", " ").title()),
            "kind": kind,
            "x": x_metric,
            "y": str(item.get("y") or "").strip(),
            "group": str(item.get("group") or "method").strip(),
            "method_order": method_order,
            "style": _normalize_plot_style(item.get("style")),
        }
        plots.append(plot)
    return plots


def _normalize_plot_style(value: Any) -> JsonDict:
    if value in (None, ""):
        return {"aggregation": "mean_ci", "y_scale": "linear"}
    if isinstance(value, str):
        style = {"aggregation": value, "y_scale": "linear"}
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - {"aggregation", "y_scale"})
        if unknown:
            raise DemoExportError("unsupported demo plot style fields: %s" % ", ".join(unknown))
        style = {
            "aggregation": str(value.get("aggregation") or "mean_ci"),
            "y_scale": str(value.get("y_scale") or "linear"),
        }
    else:
        raise DemoExportError("demo plot style must be an object or aggregation name")
    if style["aggregation"] not in {"mean", "mean_ci"}:
        raise DemoExportError("demo plot aggregation must be mean or mean_ci")
    if style["y_scale"] not in {"linear", "log"}:
        raise DemoExportError("demo plot y_scale must be linear or log")
    return style


def _normalize_training_evidence(value: Any) -> List[JsonDict]:
    if not isinstance(value, list):
        raise DemoExportError("demo.training_evidence must be an array")
    rows: List[JsonDict] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DemoExportError("demo.training_evidence entries must be objects")
        series = str(item.get("series") or "").strip()
        if not series:
            raise DemoExportError("demo.training_evidence entry %d requires series" % index)
        row: JsonDict = {"series": series}
        for field in _TRAINING_EVIDENCE_FIELDS:
            if item.get(field) not in (None, ""):
                row[field] = _normalize_evidence_file_spec(item[field], field)
        rows.append(row)
    return rows


def _normalize_evidence_file_spec(value: Any, field: str) -> JsonDict:
    if isinstance(value, str):
        return {"path": value}
    if not isinstance(value, Mapping) or not value.get("path"):
        raise DemoExportError("%s must be a path or an object with path" % field)
    payload = {"path": str(value["path"])}
    if value.get("sha256"):
        digest = str(value["sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DemoExportError("%s.sha256 must be a SHA-256 hex digest" % field)
        payload["sha256"] = digest
    return payload


def _run_record(
    entry: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    summary: JsonDict,
    manifest: JsonDict,
    recipe: JsonDict,
    *,
    run_id: str,
    method_id: str,
    index: int,
) -> JsonDict:
    manifest_recipe = manifest.get("recipe") if isinstance(manifest.get("recipe"), Mapping) else {}
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else {}
    params = benchmark_entry.get("params") if isinstance(benchmark_entry.get("params"), Mapping) else {}
    method_metadata = params.get("metadata") if isinstance(params.get("metadata"), Mapping) else {}
    benchmark_method = str(
        method_metadata.get("benchmark_method")
        or method_metadata.get("method")
        or entry.get("benchmark_method")
        or ""
    )
    series = str(
        method_metadata.get("series")
        or benchmark_method
        or entry.get("series")
        or entry.get("label")
        or entry.get("recipe_name")
        or method_id
    )
    return {
        "method_id": method_id,
        "index": index,
        "entry_id": str(entry.get("id") or method_id),
        "label": str(entry.get("label") or entry.get("recipe_name") or method_id),
        "role": str(entry.get("role") or "candidate"),
        "series": series,
        "benchmark_method": benchmark_method or series,
        "paired_seed": (
            entry.get("pairing_id")
            if entry.get("pairing_id") is not None
            else method_metadata.get(
                "pairing_id",
                method_metadata.get("benchmark_paired_seed", method_metadata.get("paired_seed")),
            )
        ),
        "aggregation_cell_id": str(
            entry.get("aggregation_cell_id")
            or method_metadata.get("aggregation_cell_id")
            or ""
        ),
        "statistical_unit": entry.get("statistical_unit")
        or method_metadata.get("statistical_unit")
        or "",
        "benchmark_method_metadata": _safe_scalar_or_list(dict(method_metadata)),
        "run_id": run_id,
        "recipe_name": str(entry.get("recipe_name") or summary.get("recipe_name") or recipe.get("name") or ""),
        "status": "completed",
        "metrics": _scalar_metrics(metrics),
        "recipe_sha256": entry.get("recipe_sha256") or manifest_recipe.get("sha256") or summary.get("recipe_sha256"),
        "semantic_recipe_sha256": entry.get("semantic_recipe_sha256"),
        "authored_recipe_sha256": manifest_recipe.get("authored_sha256") or summary.get("authored_recipe_sha256"),
        "effective_recipe_sha256": manifest_recipe.get("effective_sha256") or summary.get("effective_recipe_sha256"),
        "created_at_utc": summary.get("created_at_utc") or manifest.get("created_at_utc"),
        "completed_at_utc": summary.get("completed_at_utc") or manifest.get("completed_at_utc"),
        "summary": summary,
        "manifest": manifest,
        "recipe": recipe,
    }


def _scalar_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    payload: JsonDict = {}
    for key, value in sorted(metrics.items(), key=lambda item: str(item[0])):
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise DemoExportError("metric %s is not finite" % key)
            payload[str(key)] = value
    return payload


def _public_methods(records: Sequence[JsonDict], demo_spec: Mapping[str, Any]) -> List[JsonDict]:
    order = _series_order(demo_spec.get("series") or [])
    catalog = _series_catalog(demo_spec.get("series") or [])
    indexed = list(records)
    if order:
        positions = {value: index for index, value in enumerate(order)}
        indexed.sort(
            key=lambda row: (
                positions.get(str(row.get("series")), positions.get(str(row.get("entry_id")), len(positions))),
                int(row.get("index") or 0),
            )
        )
    methods: List[JsonDict] = []
    for record in indexed:
        method_id = str(record["method_id"])
        series_id = str(record["series"])
        series_spec = catalog.get(series_id, {})
        methods.append(
            {
                "id": method_id,
                "entry_id": record["entry_id"],
                "label": record["label"],
                "role": record["role"],
                "series": series_id,
                "series_label": series_spec.get("label") or series_id,
                "series_role": series_spec.get("role") or record["role"],
                "benchmark_method": record["benchmark_method"],
                "paired_seed": record.get("paired_seed"),
                "aggregation_cell_id": record.get("aggregation_cell_id"),
                "statistical_unit": record.get("statistical_unit"),
                "benchmark_method_metadata": dict(record.get("benchmark_method_metadata") or {}),
                "run_id": record["run_id"],
                "recipe_name": record["recipe_name"],
                "status": "completed",
                "recipe_sha256": record.get("recipe_sha256"),
                "semantic_recipe_sha256": record.get("semantic_recipe_sha256"),
                "authored_recipe_sha256": record.get("authored_recipe_sha256"),
                "effective_recipe_sha256": record.get("effective_recipe_sha256"),
                "metrics": dict(record["metrics"]),
                "evidence": {
                    "recipe": "evidence/recipes/%s.json" % method_id,
                    "run": "evidence/runs/%s.json" % method_id,
                },
            }
        )
    return methods


def _series_order(series: Sequence[Any]) -> List[str]:
    order: List[str] = []
    for item in series:
        if isinstance(item, str):
            order.append(item)
        elif isinstance(item, Mapping):
            value = item.get("id") or item.get("label") or item.get("series")
            if value:
                order.append(str(value))
    return order


def _series_catalog(series: Sequence[Any]) -> Dict[str, JsonDict]:
    catalog: Dict[str, JsonDict] = {}
    for item in series:
        if isinstance(item, str):
            catalog[item] = {"id": item, "label": item}
            continue
        if not isinstance(item, Mapping):
            continue
        series_id = str(item.get("id") or item.get("series") or item.get("label") or "")
        if not series_id:
            continue
        catalog[series_id] = {
            "id": series_id,
            "label": str(item.get("label") or item.get("name") or series_id),
            "role": str(item.get("role") or ""),
        }
    return catalog


def _traceability_profile_requested(benchmark: Mapping[str, Any]) -> bool:
    """Return the externally displayable strongest-profile request.

    A caller-provided boolean is not sufficient: only a canonical benchmark
    can request the strongest local traceability profile.
    """

    return (
        str(benchmark.get("benchmark_tier") or "").strip().lower() == "canonical"
        and traceability_profile_requested(
            benchmark,
            context="benchmark result",
        )
    )


def _source_benchmark_projection(benchmark: Mapping[str, Any], benchmark_source: Mapping[str, Any]) -> JsonDict:
    return {
        "id": benchmark.get("id"),
        "version": benchmark.get("version"),
        "sha256": benchmark.get("sha256"),
        "dataset": _safe_scalar_or_list(dict(benchmark.get("dataset") or {})),
        "task": _safe_scalar_or_list(dict(benchmark.get("task") or {})),
        "metrics": _safe_scalar_or_list(list(benchmark.get("metrics") or [])),
        "suite": _safe_scalar_or_list(dict(benchmark.get("suite") or {})),
        "benchmark_tier": benchmark.get("benchmark_tier"),
        "traceability_profile_requested": _traceability_profile_requested(
            benchmark
        ),
        # Deprecated projection alias for existing static-demo consumers.
        "publication_ready": _traceability_profile_requested(benchmark),
        "definition_sha256": canonical_json_sha256(_portable_benchmark_definition(benchmark_source)),
    }


def _portable_benchmark_definition(payload: Mapping[str, Any]) -> JsonDict:
    copied = dict(payload)
    copied.pop("path", None)
    recipes = []
    for entry in copied.get("recipes") or []:
        if not isinstance(entry, Mapping):
            continue
        row = dict(entry)
        raw_path = str(row.get("path") or "")
        if raw_path and (_is_absolute_string(raw_path) or ".." in Path(raw_path).parts):
            row["path"] = "<declared-recipe>/%s" % (Path(raw_path).name or "recipe")
        recipes.append(row)
    copied["recipes"] = recipes
    return copied


def _source_method_projection(record: Mapping[str, Any]) -> JsonDict:
    manifest = record.get("manifest") if isinstance(record.get("manifest"), Mapping) else {}
    execution_plan = manifest.get("execution_plan") if isinstance(manifest.get("execution_plan"), Mapping) else {}
    artifacts = []
    for item in manifest.get("artifacts") or []:
        if not isinstance(item, Mapping):
            continue
        artifacts.append(
            {
                "step_id": item.get("step_id"),
                "output_name": item.get("output_name"),
                "kind": item.get("kind"),
                "relative_path": item.get("relative_path"),
                "sha256": item.get("sha256"),
            }
        )
    artifacts.sort(key=lambda item: (str(item.get("step_id") or ""), str(item.get("output_name") or "")))
    return {
        "method_id": record.get("method_id"),
        "entry_id": record.get("entry_id"),
        "benchmark_method": record.get("benchmark_method"),
        "paired_seed": record.get("paired_seed"),
        "benchmark_method_metadata": record.get("benchmark_method_metadata") or {},
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "recipe_sha256": record.get("recipe_sha256"),
        "semantic_recipe_sha256": record.get("semantic_recipe_sha256"),
        "authored_recipe_sha256": record.get("authored_recipe_sha256"),
        "effective_recipe_sha256": record.get("effective_recipe_sha256"),
        "execution_plan_sha256": execution_plan.get("sha256"),
        "completed_at_utc": record.get("completed_at_utc"),
        "metrics": dict(record.get("metrics") or {}),
        "artifacts": artifacts,
    }


def _public_benchmark(
    benchmark: Mapping[str, Any],
    benchmark_source: Mapping[str, Any],
) -> JsonDict:
    return {
        "id": benchmark.get("id"),
        "version": benchmark.get("version"),
        "name": benchmark.get("name") or benchmark.get("id"),
        "sha256": benchmark.get("sha256"),
        "definition_sha256": canonical_json_sha256(
            _portable_benchmark_definition(benchmark_source)
        ),
        "dataset": _safe_scalar_or_list(dict(benchmark.get("dataset") or {})),
        "task": _safe_scalar_or_list(dict(benchmark.get("task") or {})),
        "suite": _safe_scalar_or_list(dict(benchmark.get("suite") or {})),
        "benchmark_tier": benchmark.get("benchmark_tier"),
        "traceability_profile_requested": _traceability_profile_requested(
            benchmark
        ),
        # Deprecated projection alias for existing static-demo consumers.
        "publication_ready": _traceability_profile_requested(benchmark),
    }


def _public_demo_spec(demo: Mapping[str, Any]) -> JsonDict:
    return {
        key: demo.get(key)
        for key in (
            "schema_version",
            "slug",
            "title",
            "summary",
            "question",
            "tutorial",
            "held_constant",
            "changed",
            "primary_metric",
            "comparison_axis",
            "series",
            "table_metrics",
        )
    }


def _compact_run_evidence(record: Mapping[str, Any]) -> JsonDict:
    manifest = record.get("manifest") if isinstance(record.get("manifest"), Mapping) else {}
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    execution_plan = manifest.get("execution_plan") if isinstance(manifest.get("execution_plan"), Mapping) else {}
    artifacts = []
    for item in manifest.get("artifacts") or []:
        if not isinstance(item, Mapping):
            continue
        artifacts.append(
            {
                "step_id": item.get("step_id"),
                "output_name": item.get("output_name"),
                "kind": item.get("kind"),
                "relative_path": item.get("relative_path"),
                "sha256": item.get("sha256"),
                "dtype": item.get("dtype"),
                "shape": item.get("shape"),
            }
        )
    artifacts.sort(key=lambda item: (str(item.get("step_id") or ""), str(item.get("output_name") or "")))
    return {
        "schema_version": 1,
        "kind": "noema.compact_run_evidence",
        "run_id": record.get("run_id"),
        "status": summary.get("status"),
        "recipe_name": record.get("recipe_name"),
        "benchmark_method": record.get("benchmark_method"),
        "paired_seed": record.get("paired_seed"),
        "benchmark_method_metadata": record.get("benchmark_method_metadata") or {},
        "created_at_utc": record.get("created_at_utc"),
        "completed_at_utc": record.get("completed_at_utc"),
        "recipe": {
            "sha256": record.get("recipe_sha256"),
            "semantic_sha256": record.get("semantic_recipe_sha256"),
            "authored_sha256": record.get("authored_recipe_sha256"),
            "effective_sha256": record.get("effective_recipe_sha256"),
        },
        "execution_plan": {
            "schema_version": execution_plan.get("schema_version"),
            "sha256": execution_plan.get("sha256"),
            "runner": execution_plan.get("runner"),
        },
        "seed_policy": manifest.get("seed_policy") or summary.get("seed_policy") or {},
        "metrics": dict(record.get("metrics") or {}),
        "artifacts": artifacts,
    }


def _key_metric_definitions(
    demo_spec: Mapping[str, Any], benchmark: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]
) -> List[JsonDict]:
    specs = list(demo_spec.get("table_metrics") or [])
    primary = demo_spec.get("primary_metric")
    if primary and all(str(item.get("id")) != str(primary.get("id")) for item in specs):
        specs.insert(0, dict(primary))
    if not specs:
        specs = [
            _normalize_metric_spec(item) or {}
            for item in (benchmark.get("metrics") or [])
            if isinstance(item, (str, Mapping))
        ]
        specs = [item for item in specs if item.get("id")]
    if not specs:
        numeric_keys = _numeric_metric_keys(methods)
        specs = [{"id": metric_id} for metric_id in numeric_keys[:6]]
    return [_metric_definition(spec, benchmark) for spec in specs[:12]]


def _metric_definition(spec: Mapping[str, Any], benchmark: Mapping[str, Any]) -> JsonDict:
    metric_id = str(spec.get("id") or "")
    declared = {}
    for item in benchmark.get("metrics") or []:
        if isinstance(item, Mapping) and str(item.get("id") or "") == metric_id:
            declared = dict(item)
            break
    return {
        "id": metric_id,
        "label": spec.get("label") or declared.get("name") or declared.get("label") or _human_metric(metric_id),
        "unit": spec.get("unit") or declared.get("unit") or "",
        "direction": spec.get("direction") or declared.get("direction") or "",
        "format": spec.get("format") or declared.get("format") or "",
    }


def _series_summaries(
    methods: Sequence[Mapping[str, Any]], key_metrics: Sequence[Mapping[str, Any]]
) -> List[JsonDict]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    order: List[str] = []
    for method in methods:
        series = str(method.get("series") or method.get("label") or method.get("id"))
        if series not in grouped:
            grouped[series] = []
            order.append(series)
        grouped[series].append(method)
    summaries: List[JsonDict] = []
    for series in order:
        rows = grouped[series]
        roles = []
        recipe_hashes = []
        semantic_recipe_hashes = []
        run_ids = []
        for row in rows:
            role = str(row.get("role") or "candidate")
            if role not in roles:
                roles.append(role)
            recipe_hash = str(row.get("recipe_sha256") or "")
            if recipe_hash and recipe_hash not in recipe_hashes:
                recipe_hashes.append(recipe_hash)
            semantic_recipe_hash = str(row.get("semantic_recipe_sha256") or "")
            if (
                semantic_recipe_hash
                and semantic_recipe_hash not in semantic_recipe_hashes
            ):
                semantic_recipe_hashes.append(semantic_recipe_hash)
            run_ids.append(str(row.get("run_id") or ""))
        declared_role = str(rows[0].get("series_role") or "")
        if declared_role and declared_role not in roles:
            roles.insert(0, declared_role)
        cell_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
        for row in rows:
            cell_id = str(row.get("aggregation_cell_id") or "").strip()
            unit = _demo_statistical_unit(row.get("statistical_unit"))
            if len(rows) > 1 and (not cell_id or not unit):
                raise DemoExportError(
                    "series %s has multiple runs but no explicit aggregation_cell_id "
                    "and statistical_unit" % series
                )
            cell_groups.setdefault((cell_id, unit), []).append(row)
        metrics_by_cell: List[JsonDict] = []
        for (cell_id, unit), cell_rows in cell_groups.items():
            if len(cell_rows) > 1:
                _validate_demo_aggregation_group(
                    [dict(row) for row in cell_rows],
                    series_id=series,
                    x_value=0.0,
                    y_metric="series_summary",
                )
            cell_metrics: JsonDict = {}
            for metric in key_metrics:
                metric_id = str(metric.get("id") or "")
                values = [
                    _number(dict(row.get("metrics") or {}).get(metric_id))
                    for row in cell_rows
                ]
                numeric = [value for value in values if value is not None]
                if numeric:
                    cell_metrics[metric_id] = {
                        "mean": sum(numeric) / len(numeric),
                        "min": min(numeric),
                        "max": max(numeric),
                        "count": len(numeric),
                    }
            metrics_by_cell.append(
                {
                    "aggregation_cell_id": cell_id,
                    "statistical_unit": unit,
                    "run_ids": [str(row.get("run_id") or "") for row in cell_rows],
                    "metrics": cell_metrics,
                }
            )
        metrics_by_cell.sort(key=lambda item: item["aggregation_cell_id"])
        aggregate_metrics = (
            dict(metrics_by_cell[0]["metrics"])
            if len(metrics_by_cell) == 1
            else {}
        )
        representative = rows[0]
        summaries.append(
            {
                "id": series,
                "slug": _slugify(series),
                "label": representative.get("series_label") or series,
                "declared_role": representative.get("series_role") or "",
                "roles": roles,
                "run_count": len(rows),
                "run_ids": run_ids,
                "recipe_sha256": recipe_hashes,
                "semantic_recipe_sha256": semantic_recipe_hashes,
                "metrics": aggregate_metrics,
                "metrics_by_aggregation_cell": metrics_by_cell,
                "representative_evidence": (
                    dict(representative.get("evidence") or {})
                    if len(rows) == 1
                    else {}
                ),
                "evidence_records": [
                    {
                        "run_id": str(row.get("run_id") or ""),
                        "aggregation_cell_id": str(
                            row.get("aggregation_cell_id") or ""
                        ),
                        "evidence": dict(row.get("evidence") or {}),
                    }
                    for row in rows
                ],
            }
        )
    return summaries


def _plot_definitions(
    demo_spec: Mapping[str, Any], benchmark: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]
) -> List[JsonDict]:
    configured = list(demo_spec.get("plots") or [])
    if configured:
        output = []
        for raw_plot in configured:
            plot = _validate_plot_metrics(dict(raw_plot), methods)
            plot["selection_status"] = "benchmark_protocol"
            plot["selection_protocol_sha256"] = canonical_json_sha256(raw_plot)
            output.append(plot)
        return output

    numeric_keys = _common_numeric_metric_keys(methods)
    comparison = demo_spec.get("comparison_axis") or {}
    primary = demo_spec.get("primary_metric") or {}
    x_metric = str(comparison.get("id") or "")
    y_metric = str(primary.get("id") or "")
    if not x_metric:
        x_metric = next((key for key in _LIKELY_X_METRICS if key in numeric_keys), "")
    if not y_metric:
        declared_ids = [
            str(item.get("id"))
            for item in benchmark.get("metrics") or []
            if isinstance(item, Mapping) and item.get("id")
        ]
        y_metric = next((key for key in declared_ids if key in numeric_keys and key != x_metric), "")
    if not y_metric:
        y_metric = next((key for key in numeric_keys if key != x_metric), "")
    if not y_metric:
        return []
    if x_metric and _distinct_numeric_count(methods, x_metric) > 1:
        return [
            {
                "id": "primary-comparison",
                "title": "%s versus %s" % (_human_metric(y_metric), _human_metric(x_metric)),
                "kind": "line",
                "x": x_metric,
                "y": y_metric,
                "group": "method",
                "method_order": [],
                "style": {"aggregation": "mean_ci", "y_scale": "linear"},
                "selection_status": "exploratory_default",
                "selection_protocol_sha256": None,
            }
        ]
    return [
        {
            "id": "primary-comparison",
            "title": _human_metric(y_metric),
            "kind": "bar",
            "x": "",
            "y": y_metric,
            "group": "method",
            "method_order": [],
            "style": {"aggregation": "mean_ci", "y_scale": "linear"},
            "selection_status": "exploratory_default",
            "selection_protocol_sha256": None,
        }
    ]


def _validate_plot_metrics(plot: JsonDict, methods: Sequence[Mapping[str, Any]]) -> JsonDict:
    y_metric = str(plot.get("y") or "")
    x_metric = str(plot.get("x") or "")
    if not y_metric:
        raise DemoExportError("demo plot %s requires y" % plot.get("id"))
    if plot.get("kind") == "bar" and x_metric:
        raise DemoExportError(
            "demo bar plot %s cannot declare x" % plot.get("id")
        )
    missing_y = _methods_missing_numeric_metric(methods, y_metric)
    if missing_y:
        raise DemoExportError(
            "demo plot %s y metric %s is missing/non-numeric for methods: %s"
            % (plot.get("id"), y_metric, ", ".join(missing_y))
        )
    if plot.get("kind") != "bar":
        if not x_metric:
            raise DemoExportError("demo plot %s requires x for %s plots" % (plot.get("id"), plot.get("kind")))
        missing_x = _methods_missing_numeric_metric(methods, x_metric)
        if missing_x:
            raise DemoExportError(
                "demo plot %s x metric %s is missing/non-numeric for methods: %s"
                % (plot.get("id"), x_metric, ", ".join(missing_x))
            )
    return plot


def _common_numeric_metric_keys(
    methods: Sequence[Mapping[str, Any]],
) -> List[str]:
    if not methods:
        return []
    common: Optional[set[str]] = None
    for method in methods:
        keys = {
            str(key)
            for key, value in dict(method.get("metrics") or {}).items()
            if _number(value) is not None
        }
        common = keys if common is None else common & keys
    return sorted(common or set())


def _numeric_metric_keys(methods: Sequence[Mapping[str, Any]]) -> List[str]:
    keys = set()
    for method in methods:
        for key, value in dict(method.get("metrics") or {}).items():
            if _number(value) is not None:
                keys.add(str(key))
    return sorted(keys)


def _methods_missing_numeric_metric(
    methods: Sequence[Mapping[str, Any]], metric_id: str
) -> List[str]:
    return [
        str(method.get("id") or method.get("label") or index)
        for index, method in enumerate(methods)
        if _number(dict(method.get("metrics") or {}).get(metric_id)) is None
    ]


def _distinct_numeric_count(methods: Sequence[Mapping[str, Any]], metric_id: str) -> int:
    return len(
        {
            _number(dict(method.get("metrics") or {}).get(metric_id))
            for method in methods
            if _number(dict(method.get("metrics") or {}).get(metric_id)) is not None
        }
    )


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _collect_training_evidence(
    demo_spec: Mapping[str, Any],
    *,
    run_records: Sequence[Mapping[str, Any]],
    evidence_base: Path,
    evidence_root: Path,
    redaction_roots: Sequence[Tuple[str, str]],
) -> Tuple[List[JsonDict], Dict[str, Any]]:
    published: List[JsonDict] = []
    payloads: Dict[str, Any] = {}
    used_series: set[str] = set()
    for row in demo_spec.get("training_evidence") or []:
        series = str(row.get("series") or "")
        series_slug = _unique_slug(_slugify(series), used_series)
        used_series.add(series_slug)
        public_row: JsonDict = {"series": series, "files": {}}
        for field in _TRAINING_EVIDENCE_FIELDS:
            spec = row.get(field)
            if not isinstance(spec, Mapping):
                continue
            source = _safe_declared_file(
                evidence_base,
                evidence_root,
                str(spec.get("path") or ""),
                field,
            )
            source_sha256 = file_sha256(source)
            declared_sha256 = str(spec.get("sha256") or "").lower()
            if declared_sha256 and declared_sha256 != source_sha256:
                raise DemoExportError(
                    "%s hash mismatch for series %s: expected %s, got %s"
                    % (field, series, declared_sha256, source_sha256)
                )
            parsed = _read_structured_evidence(source, field)
            if field == "trained_artifact_manifest":
                _verify_trained_artifact_references(parsed, source.parent, evidence_root)
            relative = "evidence/training/%s/%s.json" % (series_slug, field)
            payloads[relative] = _redact_payload(parsed, redaction_roots)
            public_row["files"][field] = {
                "path": relative,
                "source_sha256": source_sha256,
                "declared_sha256": declared_sha256 or None,
            }
            if field == "trained_artifact_manifest":
                public_row["runtime_provenance"] = _verify_series_runtime_manifest_sha(
                    series,
                    source_sha256,
                    run_records,
                )
        if public_row["files"]:
            published.append(public_row)
    return published, payloads


def _safe_declared_file(base: Path, allowed_root: Path, declared: str, label: str) -> Path:
    path = Path(declared)
    if not declared or path.is_absolute() or _WINDOWS_ABSOLUTE_PATH_RE.match(declared):
        raise DemoExportError("%s must be a path relative to the benchmark pack" % label)
    candidate = (base / path).resolve()
    if not _is_relative_to(candidate, allowed_root.resolve()):
        raise DemoExportError("%s escapes the allowed local evidence root" % label)
    if not candidate.is_file():
        raise DemoExportError("declared %s file is missing: %s" % (label, declared))
    size = candidate.stat().st_size
    if size > _MAX_EVIDENCE_BYTES:
        raise DemoExportError("declared %s file exceeds the %d byte publication limit" % (label, _MAX_EVIDENCE_BYTES))
    if candidate.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise DemoExportError("declared %s must be JSON or YAML" % label)
    return candidate


def _verify_series_runtime_manifest_sha(
    series: str,
    published_manifest_sha256: str,
    run_records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    matching = [
        record
        for record in run_records
        if series
        in {
            str(record.get("series") or ""),
            str(record.get("benchmark_method") or ""),
        }
    ]
    if not matching:
        raise DemoExportError(
            "training evidence series %s has no matching benchmark runs" % series
        )
    runs = []
    for record in matching:
        run_id = str(record.get("run_id") or "")
        artifact_hashes = _runtime_metadata_hashes(record, "artifact_manifest_sha256")
        checkpoint_hashes = _runtime_metadata_hashes(record, "checkpoint_sha256")
        if artifact_hashes:
            field = "artifact_manifest_sha256"
            recorded = artifact_hashes
        else:
            field = "checkpoint_sha256"
            recorded = checkpoint_hashes
        if not recorded:
            raise DemoExportError(
                "run %s in training evidence series %s has no runtime-recorded artifact manifest SHA"
                % (run_id, series)
            )
        mismatched = [digest for digest in recorded if digest != published_manifest_sha256]
        if mismatched:
            raise DemoExportError(
                "run %s runtime %s does not match published trained artifact manifest SHA %s: %s"
                % (run_id, field, published_manifest_sha256, ", ".join(mismatched))
            )
        runs.append(
            {
                "run_id": run_id,
                "field": field,
                "recorded_sha256": published_manifest_sha256,
                "observation_count": len(recorded),
            }
        )
    runs.sort(key=lambda row: str(row["run_id"]))
    return {
        "published_manifest_sha256": published_manifest_sha256,
        "run_count": len(runs),
        "runs": runs,
    }


def _runtime_metadata_hashes(record: Mapping[str, Any], key: str) -> List[str]:
    values: set[str] = set()
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    for step in summary.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        _collect_named_hashes(step.get("metadata"), key, values)
        outputs = step.get("outputs")
        if isinstance(outputs, Mapping):
            for output in outputs.values():
                if isinstance(output, Mapping):
                    _collect_named_hashes(output.get("metadata"), key, values)
    manifest = record.get("manifest") if isinstance(record.get("manifest"), Mapping) else {}
    for artifact in manifest.get("artifacts") or []:
        if isinstance(artifact, Mapping):
            _collect_named_hashes(artifact.get("metadata"), key, values)
    return sorted(values)


def _collect_named_hashes(value: Any, key: str, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if str(child_key) == key and isinstance(child, str) and child.strip():
                output.add(child.strip().lower())
            elif isinstance(child, (Mapping, list, tuple)):
                _collect_named_hashes(child, key, output)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_named_hashes(child, key, output)


def _read_structured_evidence(path: Path, label: str) -> Any:
    try:
        payload = load_strict_yaml_or_json(path)
    except (OSError, StructuredInputError) as exc:
        raise DemoExportError("could not parse %s: %s" % (label, exc)) from exc
    if label == "training_history":
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, list) and payload:
            return list(payload)
        raise DemoExportError(
            "training_history must contain a non-empty array or an object"
        )
    if not isinstance(payload, Mapping):
        raise DemoExportError("%s must contain an object" % label)
    return dict(payload)


def _verify_trained_artifact_references(payload: Mapping[str, Any], base: Path, evidence_root: Path) -> None:
    components = payload.get("components") or []
    if not isinstance(components, list):
        raise DemoExportError("trained artifact manifest components must be an array")
    for index, component in enumerate(components):
        if not isinstance(component, Mapping) or not component.get("path"):
            raise DemoExportError("trained artifact component %d requires path" % index)
        _verify_relative_hash_reference(
            base,
            evidence_root,
            str(component.get("path")),
            str(component.get("sha256") or ""),
            "trained artifact component %s" % (component.get("id") or index),
        )
    contract = payload.get("contract")
    if isinstance(contract, Mapping) and contract.get("path") and contract.get("file_sha256"):
        _verify_relative_hash_reference(
            base,
            evidence_root,
            str(contract.get("path")),
            str(contract.get("file_sha256")),
            "trained artifact contract",
        )


def _verify_relative_hash_reference(base: Path, evidence_root: Path, declared: str, expected: str, label: str) -> None:
    relative = Path(declared)
    if relative.is_absolute() or _WINDOWS_ABSOLUTE_PATH_RE.match(declared) or ".." in relative.parts:
        raise DemoExportError("%s path must stay relative to its manifest" % label)
    candidate = (base / relative).resolve()
    if not _is_relative_to(candidate, evidence_root.resolve()) or not candidate.is_file():
        raise DemoExportError("%s file is missing or outside the evidence root" % label)
    expected = expected.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise DemoExportError("%s requires a valid SHA-256 digest" % label)
    actual = file_sha256(candidate)
    if actual != expected:
        raise DemoExportError("%s hash mismatch: expected %s, got %s" % (label, expected, actual))


def _write_metrics_csv(path: Path, methods: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for method in methods:
        for metric_id, value in sorted(dict(method.get("metrics") or {}).items()):
            rows.append(
                {
                    "method_id": method.get("id"),
                    "entry_id": method.get("entry_id"),
                    "label": method.get("label"),
                    "series": method.get("series"),
                    "series_label": method.get("series_label"),
                    "benchmark_method": method.get("benchmark_method"),
                    "paired_seed": method.get("paired_seed"),
                    "aggregation_cell_id": method.get("aggregation_cell_id"),
                    "statistical_unit": _csv_value(method.get("statistical_unit")),
                    "role": method.get("role"),
                    "run_id": method.get("run_id"),
                    "metric_id": metric_id,
                    "value": _csv_value(value),
                }
            )
    _write_csv(
        path,
        rows,
        (
            "method_id",
            "entry_id",
            "label",
            "series",
            "series_label",
            "benchmark_method",
            "paired_seed",
            "aggregation_cell_id",
            "statistical_unit",
            "role",
            "run_id",
            "metric_id",
            "value",
        ),
    )


def _write_excluded_methods_csv(
    path: Path, methods: Sequence[Mapping[str, Any]]
) -> None:
    rows = [
        {
            "id": method.get("id"),
            "entry_id": method.get("entry_id"),
            "label": method.get("label"),
            "role": method.get("role"),
            "run_id": method.get("run_id"),
            "status": method.get("status"),
            "reason": method.get("reason"),
            "resource_admission": json.dumps(
                method.get("resource_admission") or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for method in methods
    ]
    _write_csv(
        path,
        rows,
        (
            "id",
            "entry_id",
            "label",
            "role",
            "run_id",
            "status",
            "reason",
            "resource_admission",
        ),
    )


def _write_plot(staging: Path, plot: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = _plot_rows(plot, methods)
    if not rows:
        raise DemoExportError("demo plot %s has no plottable stored metrics" % plot.get("id"))
    plot_id = str(plot["id"])
    csv_relative = "data/plots/%s.csv" % plot_id
    svg_relative = "figures/%s.svg" % plot_id
    _write_csv(
        staging / csv_relative,
        rows,
        (
            "method_id",
            "label",
            "series",
            "series_id",
            "role",
            "run_id",
            "paired_seed",
            "aggregation_cell_id",
            "statistical_unit",
            "x_metric",
            "x_value",
            "y_metric",
            "y_value",
            "sample_count",
            "aggregation_status",
            "sampling_unit_ids",
            "ci_scope",
            "y_stddev",
            "y_ci95",
            "y_ci95_method",
            "paired_reference_series",
            "paired_difference_mean",
            "paired_difference_stddev",
            "paired_difference_ci95",
            "paired_difference_ci95_method",
            "paired_sample_count",
            "paired_ids",
            "paired_contrast_status",
            "plot_y_value",
            "zero_floor",
        ),
    )
    (staging / svg_relative).parent.mkdir(parents=True, exist_ok=True)
    (staging / svg_relative).write_text(_render_plot_svg(plot, rows), encoding="utf-8")
    return {
        "id": plot_id,
        "title": plot.get("title"),
        "kind": plot.get("kind"),
        "x": plot.get("x"),
        "y": plot.get("y"),
        "group": plot.get("group"),
        "method_order": list(plot.get("method_order") or []),
        "style": dict(plot.get("style") or {}),
        "selection_status": plot.get("selection_status") or "unbound_internal",
        "selection_protocol_sha256": plot.get("selection_protocol_sha256"),
        "figure": svg_relative,
        "data_csv": csv_relative,
        "point_count": len(rows),
    }


def _plot_rows(plot: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]) -> List[JsonDict]:
    raw_rows = []
    x_metric = str(plot.get("x") or "")
    y_metric = str(plot.get("y") or "")
    group = str(plot.get("group") or "method")
    if plot.get("kind") == "bar" and x_metric:
        raise DemoExportError(
            "demo bar plot %s cannot declare x" % plot.get("id")
        )
    for index, method in enumerate(methods):
        metrics = dict(method.get("metrics") or {})
        y_value = _number(metrics.get(y_metric))
        if y_value is None:
            raise DemoExportError(
                "demo plot %s y metric %s is missing/non-numeric for method %s"
                % (
                    plot.get("id"),
                    y_metric,
                    method.get("id") or method.get("label") or index,
                )
            )
        x_value = _number(metrics.get(x_metric)) if x_metric else 0.0
        if x_value is None:
            raise DemoExportError(
                "demo plot %s x metric %s is missing/non-numeric for method %s"
                % (
                    plot.get("id"),
                    x_metric,
                    method.get("id") or method.get("label") or index,
                )
            )
        if group in {"method", "series", "benchmark_method"}:
            if group == "benchmark_method":
                series_id = str(method.get("benchmark_method") or method.get("series") or method.get("label") or method.get("id"))
            else:
                series_id = str(method.get("series") or method.get("label") or method.get("id"))
            series = str(method.get("series_label") or series_id)
        elif group == "role":
            series = str(method.get("role") or "candidate")
            series_id = series
        elif group == "recipe":
            series = str(method.get("recipe_name") or method.get("label") or method.get("id"))
            series_id = series
        elif group.startswith("metric:"):
            series = str(metrics.get(group.split(":", 1)[1], "unspecified"))
            series_id = series
        else:
            series_id = str(method.get("series") or method.get("label") or method.get("id"))
            series = str(method.get("series_label") or series_id)
        raw_rows.append(
            {
                "method_id": method.get("id"),
                "label": method.get("label"),
                "series": series,
                "series_id": series_id,
                "role": method.get("role"),
                "run_id": method.get("run_id"),
                "paired_seed": method.get("paired_seed"),
                "aggregation_cell_id": str(
                    method.get("aggregation_cell_id") or ""
                ),
                "statistical_unit": _demo_statistical_unit(
                    method.get("statistical_unit")
                ),
                "x_metric": x_metric or "method_index",
                "x_value": x_value,
                "y_metric": y_metric,
                "y_value": y_value,
            }
        )
    grouped: Dict[Tuple[str, float], List[JsonDict]] = {}
    for row in raw_rows:
        key = (str(row["series_id"]), float(row["x_value"]))
        grouped.setdefault(key, []).append(row)
    rows: List[JsonDict] = []
    for (series_id, x_value), samples in grouped.items():
        aggregation = _validate_demo_aggregation_group(
            samples,
            series_id=series_id,
            x_value=x_value,
            y_metric=y_metric,
        )
        y_values = [float(sample["y_value"]) for sample in samples]
        mean, stddev, ci95 = _mean_std_ci95(y_values)
        rows.append(
            {
                "method_id": ";".join(str(sample["method_id"]) for sample in samples),
                "label": str(samples[0]["series"]),
                "series": str(samples[0]["series"]),
                "series_id": series_id,
                "role": str(samples[0]["role"]),
                "run_id": ";".join(str(sample["run_id"]) for sample in samples),
                "paired_seed": ";".join(
                    str(sample["paired_seed"])
                    for sample in samples
                    if sample.get("paired_seed") is not None
                ),
                "aggregation_cell_id": aggregation["aggregation_cell_id"],
                "statistical_unit": aggregation["statistical_unit"],
                "x_metric": str(samples[0]["x_metric"]),
                "x_value": x_value,
                "y_metric": str(samples[0]["y_metric"]),
                "y_value": mean,
                "sample_count": len(samples),
                "aggregation_status": aggregation["status"],
                "sampling_unit_ids": ";".join(
                    aggregation["sampling_unit_ids"]
                ),
                "ci_scope": (
                    "between-statistical-unit"
                    if len(samples) >= 2
                    else "not_estimable_single_observation"
                ),
                "y_stddev": stddev,
                "y_ci95": ci95,
                "y_ci95_method": _demo_ci95_method(len(samples)),
            }
        )
    _attach_demo_paired_contrasts(rows, raw_rows)
    if plot.get("kind") == "bar":
        series_order = [str(item) for item in (plot.get("method_order") or [])]
        ordered_ids = []
        for requested in series_order:
            match = next(
                (str(row["series_id"]) for row in rows if requested in {str(row["series_id"]), str(row["series"])}),
                requested,
            )
            if match not in ordered_ids:
                ordered_ids.append(match)
        remaining = sorted({str(row["series_id"]) for row in rows if str(row["series_id"]) not in ordered_ids})
        positions = {name: float(index) for index, name in enumerate(ordered_ids + remaining)}
        for row in rows:
            row["x_value"] = positions[str(row["series_id"])]
    style = plot.get("style") if isinstance(plot.get("style"), Mapping) else {}
    if str(style.get("y_scale") or "linear") == "log":
        if any(float(row["y_value"]) < 0 for row in rows):
            raise DemoExportError("demo plot %s cannot use log y_scale with negative values" % plot.get("id"))
        positives = [float(row["y_value"]) for row in rows if float(row["y_value"]) > 0]
        if not positives:
            raise DemoExportError("demo plot %s cannot use log y_scale without a positive value" % plot.get("id"))
        zero_floor = min(positives) / 10.0
        for row in rows:
            value = float(row["y_value"])
            row["plot_y_value"] = value if value > 0 else zero_floor
            row["zero_floor"] = zero_floor if value == 0 else ""
    else:
        for row in rows:
            row["plot_y_value"] = row["y_value"]
            row["zero_floor"] = ""
    method_order = [str(item) for item in (plot.get("method_order") or [])]
    positions = {label: index for index, label in enumerate(method_order)}
    rows.sort(
        key=lambda row: (
            min(
                positions.get(str(row["series_id"]), len(positions)),
                positions.get(str(row["series"]), len(positions)),
            ),
            str(row["series"]),
            float(row["x_value"]),
            str(row["method_id"]),
        )
    )
    return rows


def _demo_statistical_unit(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise DemoExportError(
            "demo statistical_unit must be JSON-compatible"
        ) from exc


def _validate_demo_aggregation_group(
    samples: List[JsonDict],
    *,
    series_id: str,
    x_value: float,
    y_metric: str,
) -> JsonDict:
    if len(samples) == 1:
        paired_seed = samples[0].get("paired_seed")
        return {
            "aggregation_cell_id": str(
                samples[0].get("aggregation_cell_id") or ""
            ),
            "statistical_unit": str(samples[0].get("statistical_unit") or ""),
            "sampling_unit_ids": (
                [str(paired_seed)] if paired_seed not in (None, "") else []
            ),
            "status": "single_observation",
        }
    context = "%s at x=%s for %s" % (series_id, x_value, y_metric)
    cell_ids = {
        str(sample.get("aggregation_cell_id") or "").strip()
        for sample in samples
    }
    if "" in cell_ids or len(cell_ids) != 1:
        raise DemoExportError(
            "Cannot aggregate demo %s: repeated observations require one shared, "
            "explicit aggregation_cell_id" % context
        )
    units = {
        str(sample.get("statistical_unit") or "").strip() for sample in samples
    }
    if "" in units or len(units) != 1:
        raise DemoExportError(
            "Cannot aggregate demo %s: repeated observations require one shared, "
            "explicit statistical_unit declaration" % context
        )
    paired_ids = [str(sample.get("paired_seed") or "").strip() for sample in samples]
    if any(not value for value in paired_ids) or len(set(paired_ids)) != len(paired_ids):
        raise DemoExportError(
            "Cannot aggregate demo %s: each observation requires a unique paired_seed "
            "identifying its statistical unit" % context
        )
    run_ids = [str(sample.get("run_id") or "").strip() for sample in samples]
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise DemoExportError(
            "Cannot aggregate demo %s: each observation must come from a distinct run_id"
            % context
        )
    roles = {str(sample.get("role") or "").strip().lower() for sample in samples}
    if len(roles) != 1:
        raise DemoExportError(
            "Cannot aggregate demo %s: observations have heterogeneous roles" % context
        )
    return {
        "aggregation_cell_id": next(iter(cell_ids)),
        "statistical_unit": next(iter(units)),
        "sampling_unit_ids": sorted(paired_ids),
        "status": "declared_unique_statistical_units",
    }


def _demo_ci95_method(sample_count: int) -> str:
    return (
        "student_t_95_two_sided_tabulated_conservative_approximation"
        if sample_count >= 2
        else "not_estimable_n_lt_2"
    )


def _attach_demo_paired_contrasts(
    rows: List[JsonDict], raw_rows: List[JsonDict]
) -> None:
    for row in rows:
        row.update(
            {
                "paired_reference_series": None,
                "paired_difference_mean": None,
                "paired_difference_stddev": None,
                "paired_difference_ci95": None,
                "paired_difference_ci95_method": "not_available",
                "paired_sample_count": 0,
                "paired_ids": "",
                "paired_contrast_status": "no_declared_reference",
            }
        )
    cells: Dict[Tuple[float, str], List[JsonDict]] = {}
    for raw in raw_rows:
        cells.setdefault(
            (float(raw["x_value"]), str(raw["y_metric"])), []
        ).append(raw)
    output_index = {
        (str(row["series_id"]), float(row["x_value"]), str(row["y_metric"])): row
        for row in rows
    }
    reference_roles = {"baseline", "reference", "control"}
    for (x_value, y_metric), samples in cells.items():
        references = {
            str(sample["series_id"])
            for sample in samples
            if str(sample.get("role") or "").strip().lower() in reference_roles
        }
        if len(references) != 1:
            status = "ambiguous_reference" if references else "no_declared_reference"
            for sample in samples:
                target = output_index.get(
                    (str(sample["series_id"]), x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = status
            continue
        reference_id = next(iter(references))
        by_series: Dict[str, List[JsonDict]] = {}
        for sample in samples:
            by_series.setdefault(str(sample["series_id"]), []).append(sample)
        reference_output = output_index.get((reference_id, x_value, y_metric))
        if len(by_series) == 1:
            if reference_output is not None:
                reference_output["paired_reference_series"] = reference_id
                reference_output["paired_contrast_status"] = "reference"
            continue
        comparison_cells = {
            (
                str(sample.get("aggregation_cell_id") or "").strip(),
                str(sample.get("statistical_unit") or "").strip(),
            )
            for sample in samples
        }
        if any(not cell_id or not unit for cell_id, unit in comparison_cells):
            for sample in samples:
                target = output_index.get(
                    (str(sample["series_id"]), x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = "missing_comparison_cell"
            continue
        if len(comparison_cells) != 1:
            for sample in samples:
                target = output_index.get(
                    (str(sample["series_id"]), x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = "comparison_cell_mismatch"
            continue
        reference_values = _demo_unique_pair_values(by_series[reference_id])
        if reference_output is not None:
            reference_output["paired_reference_series"] = reference_id
            reference_output["paired_contrast_status"] = "reference"
        if reference_values is None or not reference_values:
            continue
        for series_id, series_samples in by_series.items():
            if series_id == reference_id:
                continue
            target = output_index.get((series_id, x_value, y_metric))
            if target is None:
                continue
            target["paired_reference_series"] = reference_id
            candidate_values = _demo_unique_pair_values(series_samples)
            if candidate_values is None or not candidate_values:
                target["paired_contrast_status"] = "missing_or_duplicate_pairing_ids"
                continue
            if set(candidate_values) != set(reference_values):
                target["paired_contrast_status"] = "pairing_set_mismatch"
                continue
            paired_ids = sorted(reference_values)
            differences = [
                candidate_values[pair_id] - reference_values[pair_id]
                for pair_id in paired_ids
            ]
            mean, stddev, ci95 = _mean_std_ci95(differences)
            target.update(
                {
                    "paired_difference_mean": mean,
                    "paired_difference_stddev": stddev,
                    "paired_difference_ci95": ci95,
                    "paired_difference_ci95_method": _demo_ci95_method(len(differences)),
                    "paired_sample_count": len(differences),
                    "paired_ids": ";".join(paired_ids),
                    "paired_contrast_status": "computed",
                }
            )


def _demo_unique_pair_values(
    samples: List[JsonDict],
) -> Optional[Dict[str, float]]:
    output: Dict[str, float] = {}
    for sample in samples:
        paired_seed = sample.get("paired_seed")
        if paired_seed is None or str(paired_seed) == "":
            return None
        key = str(paired_seed)
        if key in output:
            return None
        output[key] = float(sample["y_value"])
    return output


def _render_plot_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 920, 520
    left, right, top, bottom = 88, 32, 62, 82
    chart_width = width - left - right
    chart_height = height - top - bottom
    x_values = [float(row["x_value"]) for row in rows]
    style = plot.get("style") if isinstance(plot.get("style"), Mapping) else {}
    log_y = str(style.get("y_scale") or "linear") == "log"
    show_ci = str(style.get("aggregation") or "mean_ci") == "mean_ci"
    plotted_y_values = [float(row.get("plot_y_value", row["y_value"])) for row in rows]
    extent_values = list(plotted_y_values)
    if show_ci:
        for row in rows:
            mean = float(row["y_value"])
            ci95 = float(row.get("y_ci95") or 0.0)
            if log_y:
                floor = float(row.get("zero_floor") or min(plotted_y_values))
                extent_values.extend([max(floor, mean - ci95), max(floor, mean + ci95)])
            else:
                extent_values.extend([mean - ci95, mean + ci95])
    scaled_y_values = [math.log10(value) for value in extent_values] if log_y else extent_values
    x_min, x_max = _padded_extent(x_values)
    y_min, y_max = _padded_extent(
        scaled_y_values,
        include_zero=plot.get("kind") == "bar" and not log_y,
    )

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_width

    def sy(value: float) -> float:
        scaled = math.log10(value) if log_y else value
        return top + chart_height - (scaled - y_min) / (y_max - y_min) * chart_height

    series_names = []
    for row in rows:
        name = str(row["series"])
        if name not in series_names:
            series_names.append(name)
    colors = {name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(series_names)}
    fragments = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="title desc">'
        % (width, height, width, height),
        "<title id=\"title\">%s</title>" % html.escape(str(plot.get("title") or "Benchmark plot")),
        "<desc id=\"desc\">Generated deterministically from stored benchmark metrics. "
        "Each plotted point is the arithmetic mean of the observations represented by its row%s.</desc>"
        % ("; 95 percent confidence intervals are shown where estimable" if show_ci else ""),
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        "<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#172033}.axis{stroke:#697386;stroke-width:1}.grid{stroke:#e5e9f0;stroke-width:1}.label{font-size:13px}.tick{font-size:12px}.title{font-size:19px;font-weight:650}.legend{font-size:12px}</style>",
        '<text class="title" x="%d" y="32">%s</text>' % (left, html.escape(str(plot.get("title") or "Benchmark plot"))),
    ]
    for index in range(6):
        fraction = index / 5.0
        y = top + chart_height - fraction * chart_height
        scaled_value = y_min + fraction * (y_max - y_min)
        value = 10 ** scaled_value if log_y else scaled_value
        fragments.append('<line class="grid" x1="%s" y1="%s" x2="%s" y2="%s"/>' % (_fmt(left), _fmt(y), _fmt(left + chart_width), _fmt(y)))
        fragments.append('<text class="tick" x="%s" y="%s" text-anchor="end">%s</text>' % (_fmt(left - 10), _fmt(y + 4), html.escape(_metric_number(value))))
    fragments.extend(
        [
            '<line class="axis" x1="%s" y1="%s" x2="%s" y2="%s"/>' % (_fmt(left), _fmt(top + chart_height), _fmt(left + chart_width), _fmt(top + chart_height)),
            '<line class="axis" x1="%s" y1="%s" x2="%s" y2="%s"/>' % (_fmt(left), _fmt(top), _fmt(left), _fmt(top + chart_height)),
        ]
    )

    if plot.get("kind") == "bar":
        bar_width = max(12.0, min(64.0, chart_width / max(1, len(rows)) * 0.62))
        zero_y = sy(10 ** y_min) if log_y else (sy(0.0) if y_min <= 0.0 <= y_max else top + chart_height)
        for index, row in enumerate(rows):
            x = sx(float(row["x_value"]))
            y = sy(float(row.get("plot_y_value", row["y_value"])))
            top_y = min(y, zero_y)
            bar_height = max(1.0, abs(zero_y - y))
            fragments.append(
                '<rect x="%s" y="%s" width="%s" height="%s" rx="3" fill="%s"><title>%s: %s</title></rect>'
                % (
                    _fmt(x - bar_width / 2),
                    _fmt(top_y),
                    _fmt(bar_width),
                    _fmt(bar_height),
                    colors[str(row["series"])],
                    html.escape(str(row["label"])),
                    html.escape(_metric_number(float(row["y_value"]))),
                )
            )
            fragments.append('<text class="tick" x="%s" y="%s" text-anchor="middle">%d</text>' % (_fmt(x), _fmt(top + chart_height + 20), index + 1))
            ci95 = float(row.get("y_ci95") or 0.0)
            if show_ci and ci95 > 0:
                mean = float(row["y_value"])
                if log_y:
                    floor = float(row.get("zero_floor") or min(plotted_y_values))
                    low, high = max(floor, mean - ci95), max(floor, mean + ci95)
                else:
                    low, high = mean - ci95, mean + ci95
                color = colors[str(row["series"])]
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x), _fmt(sy(low)), _fmt(x), _fmt(sy(high)), color))
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x - 4), _fmt(sy(low)), _fmt(x + 4), _fmt(sy(low)), color))
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x - 4), _fmt(sy(high)), _fmt(x + 4), _fmt(sy(high)), color))
    else:
        grouped = {name: [] for name in series_names}
        for row in rows:
            grouped[str(row["series"])].append(row)
        for name in series_names:
            points = sorted(grouped[name], key=lambda row: (float(row["x_value"]), str(row["method_id"])))
            if plot.get("kind") == "line" and len(points) > 1:
                coordinates = " ".join(
                    "%s,%s"
                    % (
                        _fmt(sx(float(row["x_value"]))),
                        _fmt(sy(float(row.get("plot_y_value", row["y_value"])))),
                    )
                    for row in points
                )
                fragments.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.5"/>' % (coordinates, colors[name]))
            for row in points:
                fragments.append(
                    '<circle cx="%s" cy="%s" r="5" fill="%s" stroke="#ffffff" stroke-width="1.5"><title>%s · x=%s · y=%s</title></circle>'
                    % (
                        _fmt(sx(float(row["x_value"]))),
                        _fmt(sy(float(row.get("plot_y_value", row["y_value"])))),
                        colors[name],
                        html.escape(str(row["label"])),
                        html.escape(_metric_number(float(row["x_value"]))),
                        html.escape(_metric_number(float(row["y_value"]))),
                    )
                )
        if show_ci:
            for row in rows:
                ci95 = float(row.get("y_ci95") or 0.0)
                if ci95 <= 0:
                    continue
                x = sx(float(row["x_value"]))
                mean = float(row["y_value"])
                if log_y:
                    floor = float(row.get("zero_floor") or min(plotted_y_values))
                    low, high = max(floor, mean - ci95), max(floor, mean + ci95)
                else:
                    low, high = mean - ci95, mean + ci95
                color = colors[str(row["series"])]
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x), _fmt(sy(low)), _fmt(x), _fmt(sy(high)), color))
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x - 4), _fmt(sy(low)), _fmt(x + 4), _fmt(sy(low)), color))
                fragments.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4"/>' % (_fmt(x - 4), _fmt(sy(high)), _fmt(x + 4), _fmt(sy(high)), color))
        for index in range(6):
            fraction = index / 5.0
            x = left + fraction * chart_width
            value = x_min + fraction * (x_max - x_min)
            fragments.append('<text class="tick" x="%s" y="%s" text-anchor="middle">%s</text>' % (_fmt(x), _fmt(top + chart_height + 20), html.escape(_metric_number(value))))

    x_label = "Methods" if plot.get("kind") == "bar" else _human_metric(str(plot.get("x") or ""))
    y_label = _human_metric(str(plot.get("y") or "")) + (" (log scale)" if log_y else "")
    fragments.append('<text class="label" x="%s" y="%s" text-anchor="middle">%s</text>' % (_fmt(left + chart_width / 2), _fmt(height - 24), html.escape(x_label)))
    fragments.append('<text class="label" x="18" y="%s" text-anchor="middle" transform="rotate(-90 18 %s)">%s</text>' % (_fmt(top + chart_height / 2), _fmt(top + chart_height / 2), html.escape(y_label)))
    if log_y and any(float(row["y_value"]) == 0 for row in rows):
        zero_floor = min(float(row["plot_y_value"]) for row in rows if float(row["y_value"]) == 0)
        fragments.append(
            '<text class="tick" x="%s" y="%s">Zero values are displayed at the explicit floor %s.</text>'
            % (_fmt(left), _fmt(height - 5), html.escape(_metric_number(zero_floor)))
        )
    legend_x = left + 8
    legend_y = top + 12
    for index, name in enumerate(series_names):
        x = legend_x + (index % 3) * 255
        y = legend_y + (index // 3) * 22
        fragments.append('<circle cx="%s" cy="%s" r="4" fill="%s"/>' % (_fmt(x), _fmt(y), colors[name]))
        fragments.append('<text class="legend" x="%s" y="%s">%s</text>' % (_fmt(x + 9), _fmt(y + 4), html.escape(name[:34])))
    fragments.append("</svg>\n")
    return "\n".join(fragments)


def _padded_extent(values: Sequence[float], include_zero: bool = False) -> Tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    if include_zero:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    if minimum == maximum:
        padding = max(1.0, abs(minimum) * 0.08)
    else:
        padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def _render_index_html(demo: Mapping[str, Any]) -> str:
    demo_info = dict(demo.get("demo") or {})
    benchmark_info = dict(demo.get("benchmark") or {})
    series_rows = list(demo.get("series") or [])
    key_metrics = list(demo.get("key_metrics") or [])
    plots = list(demo.get("plots") or [])
    training = list(demo.get("training_evidence") or [])
    excluded = list(demo.get("excluded_methods") or [])
    verification_status = str(
        dict(demo.get("verification") or {}).get("status") or "unknown"
    )
    benchmark_tier = str(benchmark_info.get("benchmark_tier") or "unspecified")
    profile_requested = _traceability_profile_requested(benchmark_info)
    verification_label = (
        "verification: %s · tier: %s · strongest traceability profile requested: %s"
        % (
            verification_status,
            benchmark_tier,
            "yes" if profile_requested else "no",
        )
    )
    series_cards = []
    for series in series_rows:
        cells = []
        metrics = dict(series.get("metrics") or {})
        for metric in key_metrics:
            metric_id = str(metric.get("id") or "")
            if metric_id not in metrics:
                continue
            cells.append(
                "<div><dt>%s</dt><dd>%s%s</dd></div>"
                % (
                    html.escape(str(metric.get("label") or _human_metric(metric_id))),
                    html.escape(_display_aggregate(dict(metrics[metric_id]))),
                    (" " + html.escape(str(metric.get("unit")))) if metric.get("unit") else "",
                )
            )
        evidence = dict(series.get("representative_evidence") or {})
        recipe_hashes = list(series.get("recipe_sha256") or [])
        if evidence.get("recipe") and evidence.get("run"):
            evidence_html = (
                '<a href="%s">single-run recipe</a> · '
                '<a href="%s">single-run evidence</a> · '
                '<a href="data/metrics.csv">all runs</a>'
                % (
                    html.escape(str(evidence["recipe"]), quote=True),
                    html.escape(str(evidence["run"]), quote=True),
                )
            )
        else:
            evidence_html = '<a href="data/metrics.csv">all runs and declared cells</a>'
        series_cards.append(
            "<article class=\"method\"><div class=\"method-head\"><h3>%s</h3><span>%s</span></div><p class=\"muted\">%d stored run%s · %d recipe hash%s</p><dl>%s</dl><p class=\"evidence\">%s</p></article>"
            % (
                html.escape(str(series.get("label") or series.get("id"))),
                html.escape(", ".join(str(role) for role in (series.get("roles") or ["candidate"]))),
                int(series.get("run_count") or 0),
                "s" if int(series.get("run_count") or 0) != 1 else "",
                len(recipe_hashes),
                "es" if len(recipe_hashes) != 1 else "",
                "".join(cells) or "<div><dt>Metrics</dt><dd>See CSV</dd></div>",
                evidence_html,
            )
        )
    plot_cards = "".join(
        "<figure><img src=\"%s\" alt=\"%s\"><figcaption>%s · <a href=\"%s\">plotted data</a></figcaption></figure>"
        % (
            html.escape(str(plot.get("figure") or ""), quote=True),
            html.escape(str(plot.get("title") or "Benchmark plot"), quote=True),
            html.escape(str(plot.get("title") or "Benchmark plot")),
            html.escape(str(plot.get("data_csv") or ""), quote=True),
        )
        for plot in plots
    )
    held = _render_fact_list("Held constant", demo_info.get("held_constant") or [])
    changed = _render_fact_list("Changed", demo_info.get("changed") or [])
    tutorial_url = str(demo_info.get("tutorial") or "")
    tutorial_html = (
        '<p class="tutorial-link"><a href="%s"%s>Open the experiment tutorial</a></p>'
        % (
            html.escape(tutorial_url, quote=True),
            ' target="_blank" rel="noopener noreferrer"'
            if urllib.parse.urlsplit(tutorial_url).scheme in {"http", "https"}
            else "",
        )
        if tutorial_url
        else ""
    )
    training_html = ""
    if training:
        links = []
        for row in training:
            file_links = " · ".join(
                '<a href="%s">%s</a>' % (html.escape(str(info.get("path")), quote=True), html.escape(field.replace("_", " ")))
                for field, info in sorted(dict(row.get("files") or {}).items())
            )
            links.append("<li><strong>%s:</strong> %s</li>" % (html.escape(str(row.get("series"))), file_links))
        training_html = "<section><h2>Training evidence</h2><ul>%s</ul></section>" % "".join(links)
    exclusion_html = ""
    if excluded:
        items = "".join(
            "<li><strong>%s</strong> (%s, run <code>%s</code>)</li>"
            % (
                html.escape(str(row.get("label") or row.get("entry_id") or row.get("id"))),
                html.escape(str(row.get("status") or "excluded")),
                html.escape(str(row.get("run_id") or "")),
            )
            for row in excluded
        )
        exclusion_html = (
            "<section><h2>Excluded by frozen resource admission</h2>"
            "<p>These stored runs are preserved in the evidence ledger and are not "
            "included in comparison tables or figures.</p><ul>%s</ul>"
            '<p><a href="data/excluded_methods.csv">exclusion ledger</a></p></section>'
            % items
        )
    embedded = _json_text(demo).replace("<", "\\u003c")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · Noema</title>
  <style>
    :root{{--ink:#162033;--muted:#64748b;--line:#dbe3ee;--surface:#f6f8fb;--accent:#2563eb}}
    *{{box-sizing:border-box}} body{{margin:0;font:15px/1.55 system-ui,-apple-system,sans-serif;color:var(--ink);background:#fff}}
    main{{width:min(1120px,calc(100% - 32px));margin:0 auto;padding:42px 0 72px}} h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.05;margin:.25rem 0 1rem}}
    h2{{margin-top:2.25rem}} .eyebrow,.muted{{color:var(--muted)}} .eyebrow{{letter-spacing:.08em;text-transform:uppercase;font-weight:700;font-size:.78rem}}
    .verify{{display:inline-flex;padding:.28rem .65rem;border-radius:999px;background:#e2e8f0;color:#334155;font-weight:700}}
    .verify.profile-requested{{background:#dcfce7;color:#166534}}
    .facts,.methods{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}} .fact,.method,figure{{border:1px solid var(--line);border-radius:14px;background:var(--surface);padding:18px}}
    .method-head{{display:flex;justify-content:space-between;gap:16px;align-items:start}} .method h3{{margin:0}} .method-head span{{font-size:.75rem;text-transform:uppercase;color:var(--muted)}}
    dl{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:16px 0}} dl div{{background:#fff;border-radius:9px;padding:9px}} dt{{font-size:.74rem;color:var(--muted)}} dd{{margin:2px 0 0;font-weight:700}}
    figure{{margin:16px 0;background:#fff}} figure img{{display:block;width:100%;height:auto}} figcaption{{color:var(--muted);margin-top:8px}} a{{color:var(--accent)}} .tutorial-link a{{display:inline-flex;padding:.55rem .85rem;border:1px solid var(--accent);border-radius:9px;font-weight:700;text-decoration:none}} code{{overflow-wrap:anywhere}} .provenance{{border-top:1px solid var(--line);margin-top:38px;padding-top:20px}}
  </style>
</head>
<body><main>
  <div class="eyebrow">Noema stored benchmark demo</div>
  <h1>{title}</h1>
  <p>{summary}</p>
  {question}
  {tutorial}
  <p><span class="verify {verification_class}">{verification}</span></p>
  <div class="facts">{facts}</div>
  <section><h2>Comparison</h2><div class="methods">{methods}</div></section>
  {exclusions}
  {plots}
  {training}
  <section class="provenance"><h2>Reproducibility evidence</h2>
    <p>Source result <code>{result_id}</code></p>
    <p>Portable benchmark definition SHA-256 <code>{definition_sha}</code></p>
    <p>Publication SHA-256 <code>{publication_sha}</code></p>
    <p><a href="data/demo.json">demo data</a> · <a href="data/metrics.csv">admitted stored metrics</a> · <a href="data/excluded_methods.csv">resource-exclusion ledger</a> · <a href="evidence/verification.json">verification report</a> · <a href="publication-manifest.json">file manifest</a></p>
  </section>
  <script type="application/json" id="noema-demo-data">{embedded}</script>
</main></body></html>
""".format(
        title=html.escape(str(demo_info.get("title") or "Noema benchmark")),
        summary=html.escape(str(demo_info.get("summary") or "")),
        question=("<p><strong>Research question:</strong> %s</p>" % html.escape(str(demo_info.get("question")))) if demo_info.get("question") else "",
        tutorial=tutorial_html,
        verification=html.escape(verification_label),
        verification_class="profile-requested" if profile_requested else "internal-only",
        facts=held + changed,
        methods="".join(series_cards),
        exclusions=exclusion_html,
        plots=("<section><h2>Stored-metric figures</h2>%s</section>" % plot_cards) if plot_cards else "",
        training=training_html,
        result_id=html.escape(str(demo.get("source_result_id") or "")),
        definition_sha=html.escape(
            str(benchmark_info.get("definition_sha256") or "")
        ),
        publication_sha=html.escape(str(demo.get("publication_sha256") or "")),
        embedded=embedded,
    )


def _render_fact_list(title: str, values: Sequence[Any]) -> str:
    if not values:
        return ""
    items = "".join("<li>%s</li>" % html.escape(_fact_text(item)) for item in values)
    return '<section class="fact"><strong>%s</strong><ul>%s</ul></section>' % (html.escape(title), items)


def _fact_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return "; ".join("%s: %s" % (key, child) for key, child in sorted(value.items()))
    return str(value)


def _publication_manifest(
    root: Path, *, slug: str, result_id: str, source_bundle_sha256: str, publication_sha256: str
) -> JsonDict:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    files_sha256 = canonical_json_sha256(files)
    return {
        "schema_version": PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "kind": PUBLICATION_MANIFEST_KIND,
        "slug": slug,
        "source_result_id": result_id,
        "source_bundle_sha256": source_bundle_sha256,
        "publication_sha256": publication_sha256,
        "hash_algorithm": "sha256",
        "scope": "all publication files except this manifest",
        "files_sha256": files_sha256,
        "files": files,
    }


def _maybe_update_demo_registry(destination: Path, demo: Mapping[str, Any]) -> None:
    experiments_dir = destination.parent
    if (
        destination.name != str(demo.get("slug") or "")
        or experiments_dir.name != "experiments"
        or experiments_dir.parent.name != "demo"
        or experiments_dir.parent.parent.name != "docs"
    ):
        return
    registry_path = experiments_dir / "index.json"
    if registry_path.is_symlink():
        raise DemoExportError("demo registry cannot be a symbolic link")
    if registry_path.exists():
        try:
            current = load_strict_yaml_or_json(registry_path)
        except (OSError, StructuredInputError) as exc:
            raise DemoExportError("could not parse existing demo registry: %s" % exc) from exc
        if not isinstance(current, Mapping) or not isinstance(current.get("demos") or [], list):
            raise DemoExportError("existing demo registry has an unsupported schema")
        entries = [dict(item) for item in current.get("demos") or [] if isinstance(item, Mapping)]
    else:
        entries = []
    slug = str(demo.get("slug") or "")
    entries = [entry for entry in entries if str(entry.get("slug") or "") != slug]
    demo_info = demo.get("demo") if isinstance(demo.get("demo"), Mapping) else {}
    entries.append(
        {
            "slug": slug,
            "title": demo_info.get("title") or slug,
            "result_id": demo.get("source_result_id"),
            "source_bundle_sha256": demo.get("source_bundle_sha256"),
            "path": "%s/index.html" % slug,
        }
    )
    entries.sort(key=lambda item: str(item.get("slug") or ""))
    payload = {
        "schema_version": 1,
        "kind": "noema.demo_registry",
        "demos": entries,
    }
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(experiments_dir),
            prefix=".index.json.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_json_text(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(registry_path))
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise


def _strip_volatile_verification(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile_verification(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).lower() not in _VOLATILE_VERIFICATION_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile_verification(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload) + "\n", encoding="utf-8")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return _fmt(value)
    return value


def _install_staging_directory(staging: Path, destination: Path, *, force: bool) -> None:
    if destination.exists():
        if not force:
            raise DemoExportError("publication output already exists; pass --force to replace it: %s" % destination)
        if destination.is_symlink() or not destination.is_dir():
            raise DemoExportError("publication output must be a regular directory: %s" % destination)
        shutil.rmtree(destination)
    os.replace(str(staging), str(destination))


def _redaction_roots(store: LocalStore, project_root: Path) -> List[Tuple[str, str]]:
    roots = [
        (str(store.workspace.resolve()), "<workspace>"),
        (str(project_root.resolve()), "<project-root>"),
    ]
    deduplicated: Dict[str, str] = {}
    for source, replacement in roots:
        if source and source != os.sep:
            deduplicated[source] = replacement
    return sorted(deduplicated.items(), key=lambda item: len(item[0]), reverse=True)


def _redact_payload(value: Any, roots: Sequence[Tuple[str, str]]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_payload(child, roots) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_redact_payload(item, roots) for item in value]
    if isinstance(value, tuple):
        return [_redact_payload(item, roots) for item in value]
    if isinstance(value, str):
        return _redact_string(value, roots)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DemoExportError("publication evidence contains a non-finite number")
        return value
    return str(value)


def _redact_string(value: str, roots: Sequence[Tuple[str, str]]) -> str:
    redacted = value
    for source, replacement in roots:
        redacted = redacted.replace(source, replacement)
        redacted = redacted.replace(source.replace("/", "\\"), replacement)
    if _is_absolute_string(redacted):
        name = Path(redacted.replace("\\", "/")).name
        return "<absolute-path>/%s" % (name or "path")
    # Verification messages can contain a path after punctuation. Known roots
    # are handled above; redact any remaining whitespace-delimited absolute path.
    redacted = re.sub(
        r"(?<![:/A-Za-z0-9>])/(?:[^\s\"'<>]+)",
        lambda match: "<absolute-path>/%s" % (Path(match.group(0).rstrip(".,;:)" )).name or "path"),
        redacted,
    )
    redacted = re.sub(
        r"\b[A-Za-z]:[\\/][^\s\"'<>]+",
        lambda match: "<absolute-path>/%s" % (Path(match.group(0).replace("\\", "/")).name or "path"),
        redacted,
    )
    return redacted


def _is_absolute_string(value: str) -> bool:
    if value.startswith(("http://", "https://", "data:")):
        return False
    return Path(value).is_absolute() or bool(_WINDOWS_ABSOLUTE_PATH_RE.match(value))


def _assert_tree_has_no_absolute_roots(root: Path, roots: Sequence[Tuple[str, str]]) -> None:
    needles = [source.encode("utf-8") for source, _replacement in roots]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_bytes()
        for needle in needles:
            if needle and needle in content:
                raise DemoExportError("publication leaked an absolute source path in %s" % path.relative_to(root))


def _human_metric(metric_id: str) -> str:
    tail = metric_id.split(".")[-1] if metric_id else "Value"
    return tail.replace("_", " ").strip().title()


def _display_value(value: Any) -> str:
    numeric = _number(value)
    if numeric is not None:
        return _metric_number(numeric)
    return str(value)


def _display_aggregate(value: Mapping[str, Any]) -> str:
    mean = _number(value.get("mean"))
    minimum = _number(value.get("min"))
    maximum = _number(value.get("max"))
    if mean is None:
        return "n/a"
    if minimum is None or maximum is None or minimum == maximum:
        return _metric_number(mean)
    return "%s [%s–%s]" % (
        _metric_number(mean),
        _metric_number(minimum),
        _metric_number(maximum),
    )


def _metric_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.001:
        return "%.3e" % value
    return "%.5g" % value


def _fmt(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded == 0:
        return "0"
    return ("%.6f" % rounded).rstrip("0").rstrip(".")
