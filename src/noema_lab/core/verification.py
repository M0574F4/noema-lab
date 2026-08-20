from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.attempt_ledger import BenchmarkAttemptLedger
from noema_lab.core.benchmark_evidence import (
    BenchmarkEvidenceError,
    validate_benchmark_training_evidence_snapshot,
)
from noema_lab.core.benchmark_run_evidence import (
    BenchmarkRunEvidenceError,
    validate_benchmark_run_evidence_snapshots,
)
from noema_lab.core.benchmarks import benchmark_protocol_sha256_matches
from noema_lab.core.common_conditions import validate_common_condition_evidence_set
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.operations import (
    MATERIALIZATION_RUNNERS,
    OperationError,
    OperationRegistry,
)
from noema_lab.core.plan_cache import (
    EXECUTION_PLAN_CACHE_EVIDENCE_SCHEMA_VERSION,
    EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION,
)
from noema_lab.core.planner import (
    EXECUTION_PLAN_KIND,
    EXECUTION_PLAN_SCHEMA_VERSION,
    PLANNED_STEP_SCHEMA_VERSION,
    validate_recipe_against_registry,
)
from noema_lab.core.publication_profile import (
    LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD,
    LEGACY_PUBLICATION_VERIFICATION_PROFILE_IDS,
    TRACEABILITY_PROFILE_REQUEST_FIELD,
    publication_predicate_check_semantics,
    publication_predicate_check_semantics_binding,
    publication_verification_profile,
    publication_verification_profile_binding,
    traceability_profile_requested,
)
from noema_lab.core.recipes import RecipeValidationError, recipe_from_dict
from noema_lab.core.reproducibility import canonical_json_sha256
from noema_lab.core.resource_units import (
    ExecutedCodedModulationBindings,
    IdealizedNativePayloadUseProxy,
    ResourceQuantity,
    ResourceUnitError,
    evaluate_resource_admission as evaluate_typed_resource_admission,
)
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import decode_strict_yaml_or_json

JsonDict = Dict[str, Any]

_RUN_SUMMARY_SCHEMA_VERSION = 1
_RUN_SUMMARY_KIND = "noema.run_summary"
_RUN_MANIFEST_SCHEMA_VERSION = 1
_RUN_MANIFEST_KIND = "noema.run_manifest"
_BENCHMARK_RESULT_SCHEMA_VERSION = 1
_BENCHMARK_RESULT_KIND = "noema.benchmark_result"
_TERMINAL_BAD_STATUSES = {"failed", "canceled", "cancelled", "running", "queued", "incomplete"}
_NONNEGATIVE_TOKENS = (
    "bpp",
    "bit_count",
    "bits",
    "byte_count",
    "bytes",
    "channel_use",
    "count",
    "duration",
    "latency",
    "memory",
    "pixel",
    "rate",
    "rss",
    "time",
    "wall_time_s",
)
_UNIT_INTERVAL_TOKENS = (
    ".ber",
    ".bler",
    "_probability",
    ".probability",
    "_fraction",
    ".fraction",
    "_utilization",
    ".utilization",
    "_success_rate",
    ".success_rate",
    "_outage_rate",
    ".outage_rate",
    "_delivery_success",
    ".delivery_success",
)
_TX_BIT_KEYS = (
    "tx_bit_boundary.channel.fixed.modulator_input.bit_count",
    "channel.fixed.modulator_input.bit_count",
    "channel.transmitted_bit_count",
    "channel.coded_bit_count",
    "modulator.channel.transmitted_bit_count",
    "codec.bit_count",
)
_BPP_KEYS = ("rate_bpp", "rate.bpp", "quality.rate_bpp")
_PIXEL_KEYS = ("source_pixels", "source_pixel_count", "image.source_pixel_count")


@dataclass
class CheckResult:
    id: str
    status: str
    message: str
    details: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class _CheckRecorder:
    def __init__(self) -> None:
        self.checks: List[CheckResult] = []

    def pass_(self, check_id: str, message: str, **details: Any) -> None:
        self.checks.append(CheckResult(check_id, "pass", message, _clean_details(details)))

    def warning(self, check_id: str, message: str, **details: Any) -> None:
        self.checks.append(CheckResult(check_id, "warning", message, _clean_details(details)))

    def error(self, check_id: str, message: str, **details: Any) -> None:
        self.checks.append(CheckResult(check_id, "error", message, _clean_details(details)))

    def report(
        self,
        *,
        target_type: str,
        target_id: str,
        path: Path,
        metadata: Optional[JsonDict] = None,
    ) -> JsonDict:
        errors = [check.message for check in self.checks if check.status == "error"]
        warnings = [check.message for check in self.checks if check.status == "warning"]
        status = "invalid" if errors else "warning" if warnings else "valid"
        payload: JsonDict = {
            "status": status,
            "target_type": target_type,
            "target_id": target_id,
            "path": str(path),
            "errors": errors,
            "warnings": warnings,
            "checks": [check.to_dict() for check in self.checks],
        }
        if metadata:
            payload.update(metadata)
        return payload


def verify_run_bundle(
    store: LocalStore,
    run_id: str,
    registry: Optional[OperationRegistry] = None,
) -> JsonDict:
    """Verify one local run bundle without rerunning the experiment."""

    recorder = _CheckRecorder()
    run_dir = store.get_run_dir(run_id)
    summary: Optional[JsonDict] = None
    manifest: Optional[JsonDict] = None
    recipe: Optional[JsonDict] = None

    if run_dir.exists() and run_dir.is_dir():
        recorder.pass_("structure.run_dir", "run directory exists")
    else:
        recorder.error("structure.run_dir", "run directory does not exist: %s" % run_dir)
        return recorder.report(target_type="run", target_id=run_id, path=run_dir)

    summary = _load_json_file(run_dir / "summary.json", recorder, "summary")
    manifest = _load_json_file(run_dir / "manifest.json", recorder, "manifest")
    recipe = _load_json_file(run_dir / "recipe.json", recorder, "recipe")

    if summary is not None:
        _check_run_summary_schema(summary, run_id, recorder)
    if manifest is not None:
        _check_manifest_schema(manifest, run_id, recorder)
    if summary is not None and manifest is not None:
        _check_summary_manifest_consistency(summary, manifest, recorder)
        _check_summary_content_binding(run_dir, manifest, recorder)
        _check_execution_runtime_evidence(summary, manifest, recorder)
    if recipe is not None and manifest is not None:
        _check_recipe_identity(recipe, manifest, summary, recorder)
    if recipe is not None and registry is not None:
        _check_recipe_lint(recipe, registry, recorder)
    if manifest is not None:
        _check_execution_plan_evidence(
            run_dir,
            manifest,
            summary,
            recipe,
            recorder,
            registry=registry,
        )
        _check_manifest_artifacts(run_dir, manifest, recorder)
    if summary is not None:
        _check_run_status(summary, manifest, recorder)
        flat_metrics = flatten_summary_metrics(summary)
        _check_metric_plausibility(flat_metrics, recorder, prefix="metrics")
        rate_summary = dict(summary)
        if recipe is not None:
            rate_summary["recipe"] = recipe
        _check_rate_accounting(rate_summary, flat_metrics, recorder)

    metadata = _run_report_metadata(summary, manifest)
    return recorder.report(target_type="run", target_id=run_id, path=run_dir, metadata=metadata)


def verify_benchmark_result(
    store: LocalStore,
    result_id: str,
    registry: Optional[OperationRegistry] = None,
    *,
    deep_backing_runs: bool = False,
) -> JsonDict:
    """Verify one local benchmark result bundle without rerunning recipes."""

    recorder = _CheckRecorder()
    result_dir = store.get_benchmark_result_dir(result_id)
    if result_dir.exists() and result_dir.is_dir():
        recorder.pass_("structure.benchmark_dir", "benchmark result directory exists")
    else:
        recorder.error("structure.benchmark_dir", "benchmark result directory does not exist: %s" % result_dir)
        return recorder.report(target_type="benchmark_result", target_id=result_id, path=result_dir)

    result = _load_json_file(result_dir / "result.json", recorder, "benchmark result")
    benchmark_json = _load_optional_json_file(result_dir / "benchmark.json", recorder, "benchmark pack")
    if result is None:
        return recorder.report(target_type="benchmark_result", target_id=result_id, path=result_dir)

    _check_benchmark_schema(result, recorder)
    _check_benchmark_attempt_ledger(
        store,
        result_dir,
        result,
        recorder,
    )
    _check_benchmark_status(result, recorder)
    _check_metric_plausibility(_benchmark_flat_metrics(result), recorder, prefix="benchmark.metrics")
    validated_run_evidence = _check_benchmark_run_evidence_snapshots(
        result_dir, result, recorder
    )
    _check_benchmark_required_metrics(
        result,
        recorder,
        validated_run_evidence=validated_run_evidence,
    )
    _check_benchmark_resource_admission(result, benchmark_json, recorder)
    _check_benchmark_protocol(result, benchmark_json, recorder)
    _check_benchmark_common_conditions(result, benchmark_json, recorder)
    _check_benchmark_training_evidence_snapshot(
        result_dir,
        result,
        benchmark_json,
        recorder,
    )
    _check_benchmark_report_artifacts(result_dir, result, recorder)
    declared_outputs = _check_benchmark_expected_outputs(
        result_dir,
        result,
        benchmark_json,
        recorder,
    )
    _check_benchmark_plot_artifacts(result_dir, result, recorder)
    _check_benchmark_plot_sidecars(
        result_dir,
        result,
        recorder,
        declared_plot_outputs=declared_outputs["plot_artifacts"],
        required_plot_sidecars=declared_outputs["plot_sidecars"],
    )
    _check_benchmark_resource_guard_sidecar(result_dir, result, recorder)
    _check_benchmark_backing_runs(
        store,
        result,
        registry,
        recorder,
        deep=deep_backing_runs,
    )

    benchmark = dict(result.get("benchmark") or {})
    metadata: JsonDict = {
        "benchmark_id": benchmark.get("id"),
        "benchmark_version": benchmark.get("version"),
        "created_time": result.get("created_at_utc"),
        "completed_time": result.get("completed_at_utc"),
        "certification": _benchmark_certification_verdict(
            result,
            benchmark_json,
            recorder,
        ),
    }
    return recorder.report(
        target_type="benchmark_result",
        target_id=result_id,
        path=result_dir,
        metadata=metadata,
    )


def format_verification_human(report: JsonDict) -> str:
    target_type = str(report.get("target_type") or "target").replace("_", " ")
    target_id = str(report.get("target_id") or "")
    status = str(report.get("status") or "invalid")
    if target_type == "benchmark result":
        noun = "benchmark result"
    else:
        noun = target_type
    lines = ["%s %s: %s" % (status, noun, target_id)]
    certification = report.get("certification")
    if isinstance(certification, Mapping):
        profile = certification.get("claimed_profile")
        if isinstance(profile, Mapping):
            profile_label = "%s@%s" % (
                profile.get("id") or "<missing-id>",
                profile.get("sha256") or "<missing-digest>",
            )
        else:
            profile_label = "none"
        lines.append(
            "traceability verdict: %s (%s); tier: %s; schema revision: %s; "
            "claimed profile: %s"
            % (
                certification.get("traceability_verdict") or "invalid",
                certification.get("verdict_class") or "invalid",
                certification.get("benchmark_tier") or "unknown",
                certification.get("benchmark_schema_revision") or "unknown",
                profile_label,
            )
        )
        predicates = certification.get("applicable_predicates")
        if isinstance(predicates, list):
            lines.append(
                "applicable predicates: %s"
                % (", ".join(str(value) for value in predicates) or "none")
            )
        invariants = certification.get("invariants")
        if isinstance(invariants, list):
            lines.append(
                "invariant statuses: %s"
                % ", ".join(
                    "%s=%s"
                    % (
                        row.get("id") or "?",
                        row.get("status") or "missing",
                    )
                    for row in invariants
                    if isinstance(row, Mapping)
                )
            )
    for message in report.get("errors") or []:
        lines.append("error: %s" % message)
    for message in report.get("warnings") or []:
        lines.append("warning: %s" % message)
    return "\n".join(lines)


def _resolve_traceability_profile_request(
    state: Mapping[str, Any],
    recorder: _CheckRecorder,
    *,
    context: str,
    check_id: str,
) -> bool:
    """Resolve the profile trigger while turning malformed aliases into evidence."""

    try:
        return traceability_profile_requested(state, context=context)
    except ValueError as exc:
        recorder.error(check_id, str(exc))
        # A contradictory document must fail, but a true value on either side
        # still requests fail-closed strongest-profile treatment.
        return any(
            state.get(field) is True
            for field in (
                TRACEABILITY_PROFILE_REQUEST_FIELD,
                LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD,
            )
        )


def _benchmark_certification_verdict(
    result: Mapping[str, Any],
    benchmark_json: Optional[Mapping[str, Any]],
    recorder: _CheckRecorder,
) -> JsonDict:
    """Return a fail-closed, non-publishability verifier verdict.

    ``status=valid`` historically meant that a result was internally
    consistent with whatever tier it declared.  This record prevents a legacy
    or experimental success from being presented as a pass under the current
    publication profile.  The other status-vector fields are intentionally
    ``not_evaluated``: local verification does not establish external
    conformance, archival availability, independent reproduction, or venue
    acceptance.
    """

    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    frozen_metadata = (
        benchmark_json.get("metadata")
        if isinstance(benchmark_json, Mapping)
        and isinstance(benchmark_json.get("metadata"), Mapping)
        else {}
    )
    result_metadata = benchmark.get("metadata")
    result_metadata = (
        result_metadata if isinstance(result_metadata, Mapping) else {}
    )
    tier = str(
        benchmark.get("benchmark_tier")
        or frozen_metadata.get("benchmark_tier")
        or frozen_metadata.get("tier")
        or "smoke"
    ).strip().lower()
    result_profile_requested = _resolve_traceability_profile_request(
        benchmark,
        recorder,
        context="result benchmark",
        check_id="benchmark.protocol.result_traceability_profile_request",
    )
    frozen_profile_requested = _resolve_traceability_profile_request(
        frozen_metadata,
        recorder,
        context="benchmark.json metadata",
        check_id="benchmark.protocol.traceability_profile_request",
    )
    if (
        isinstance(benchmark_json, Mapping)
        and result_profile_requested is not frozen_profile_requested
    ):
        recorder.error(
            "benchmark.protocol.traceability_profile_request_identity",
            "result does not preserve the frozen traceability-profile request",
        )
    publication_claimed = bool(
        result_profile_requested or frozen_profile_requested
    )
    claimed_profile = result_metadata.get("verification_profile")
    if claimed_profile is None:
        claimed_profile = frozen_metadata.get("verification_profile")
    claimed_profile = (
        dict(claimed_profile) if isinstance(claimed_profile, Mapping) else None
    )
    required_profile = publication_verification_profile_binding()
    exact_profile = claimed_profile == required_profile
    claimed_profile_id = (
        str(claimed_profile.get("id") or "")
        if isinstance(claimed_profile, Mapping)
        else ""
    )
    legacy_profile = (
        claimed_profile_id in LEGACY_PUBLICATION_VERIFICATION_PROFILE_IDS
    )
    invariant_results = _evaluate_publication_invariants(
        result,
        benchmark_json,
        recorder,
    )
    applicable = [
        row["id"]
        for row in invariant_results
        if row["applicability"] == "applicable"
    ]
    failed_invariants = [
        row["id"]
        for row in invariant_results
        if row["applicability"] == "applicable" and row["status"] != "pass"
    ]
    all_applicable_pass = not failed_invariants
    has_errors = any(check.status == "error" for check in recorder.checks)
    if not publication_claimed:
        verdict_class = (
            "nonpublication_evidence_fail"
            if has_errors
            else "nonpublication_evidence_pass"
        )
        profile_binding_status = (
            "not_claimed"
            if claimed_profile is None
            else "profile_present_without_publication_claim"
        )
        traceability_verdict = (
            "nonpublication_evidence_invalid"
            if has_errors
            else "nonpublication_evidence_valid"
        )
    else:
        verdict_class = (
            "current_profile_pass"
            if exact_profile and tier == "canonical" and all_applicable_pass
            else "current_profile_fail"
        )
        if claimed_profile is None:
            profile_binding_status = "missing"
        elif exact_profile:
            profile_binding_status = "matched"
        elif legacy_profile:
            profile_binding_status = "legacy_noncurrent"
        else:
            profile_binding_status = "substituted"
        if verdict_class == "current_profile_pass":
            traceability_verdict = "traceability_contract_pass"
        elif legacy_profile:
            traceability_verdict = "legacy_profile_noncurrent"
        else:
            traceability_verdict = "traceability_contract_fail"

    return {
        "verdict_schema_revision": 2,
        "verdict_class": verdict_class,
        "traceability_verdict": traceability_verdict,
        "benchmark_schema_revision": result.get("schema_version"),
        "benchmark_tier": tier,
        "traceability_profile_requested": publication_claimed,
        # Deprecated report alias retained while stored verification consumers
        # migrate to the reader-accurate field above.
        "publication_profile_claimed": publication_claimed,
        "claimed_profile": claimed_profile,
        "required_current_profile": required_profile,
        "predicate_check_semantics": (
            publication_predicate_check_semantics_binding()
        ),
        "profile_binding_status": profile_binding_status,
        "applicable_predicates": applicable,
        "invariants": invariant_results,
        "verdict_basis": {
            "current_profile_exactly_bound": exact_profile,
            "canonical_benchmark_tier": tier == "canonical",
            "all_applicable_invariants_pass": all_applicable_pass,
            "failed_invariants": failed_invariants,
            "verifier_error_count": sum(
                check.status == "error" for check in recorder.checks
            ),
        },
        "contract_compliant": verdict_class == "current_profile_pass",
        "externally_conformant": "not_evaluated",
        "archived": "not_evaluated",
        "independently_reproduced": "not_evaluated",
        "publication_candidate": "not_determined",
    }


def _evaluate_publication_invariants(
    result: Mapping[str, Any],
    benchmark_json: Optional[Mapping[str, Any]],
    recorder: _CheckRecorder,
) -> List[JsonDict]:
    """Map concrete verifier checks to evidence-bearing I1--I6 verdicts."""

    profile = publication_verification_profile()
    semantics = publication_predicate_check_semantics()
    raw_semantics = semantics.get("predicates")
    predicate_semantics = (
        raw_semantics if isinstance(raw_semantics, Mapping) else {}
    )
    trained_artifact_declared = _contains_trained_artifact_declaration(
        result
    ) or (
        isinstance(benchmark_json, Mapping)
        and _contains_trained_artifact_declaration(benchmark_json)
    )
    declared_plot_outputs = _contains_declared_plot_outputs(
        result,
        benchmark_json,
    )
    context = {
        "trained_artifact_declared": trained_artifact_declared,
        "declared_plot_outputs": declared_plot_outputs,
    }
    predicate_definitions = profile.get("predicates")
    predicate_definitions = (
        predicate_definitions
        if isinstance(predicate_definitions, Mapping)
        else {}
    )

    rows: List[JsonDict] = []
    by_id: Dict[str, JsonDict] = {}
    for predicate_id in ("I1", "I2", "I3", "I4", "I5", "I6"):
        raw = predicate_semantics.get(predicate_id)
        rule = raw if isinstance(raw, Mapping) else {}
        applicability_rule = str(rule.get("applicability") or "always")
        is_applicable = (
            applicability_rule == "always"
            or bool(context.get(applicability_rule))
        )
        definition = predicate_definitions.get(predicate_id)
        definition = definition if isinstance(definition, Mapping) else {}
        dependencies = [
            str(value)
            for value in definition.get("depends_on") or []
            if str(value)
        ]
        if not is_applicable:
            row = {
                "id": predicate_id,
                "name": rule.get("name") or definition.get("name"),
                "applicability": "not_applicable",
                "applicability_reason": (
                    "no imported or returned trained artifact is declared"
                    if applicability_rule == "trained_artifact_declared"
                    else "the predicate applicability condition is false"
                ),
                "status": "not_applicable",
                "local_status": "not_applicable",
                "dependencies": dependencies,
                "check_ids": [],
                "witness_references": [],
                "required_pass_groups": [],
                "missing_required_pass_groups": [],
            }
            rows.append(row)
            by_id[predicate_id] = row
            continue

        exact_ids = {
            str(value)
            for value in rule.get("check_ids") or []
            if str(value)
        }
        prefixes = tuple(
            str(value)
            for value in rule.get("check_id_prefixes") or []
            if str(value)
        )
        error_scope = str(rule.get("error_scope") or "")
        selected: List[Tuple[int, CheckResult]] = []
        for index, check in enumerate(recorder.checks):
            if (
                error_scope == "all_verifier_checks"
                or check.id in exact_ids
                or check.id.startswith(prefixes)
            ):
                selected.append((index, check))

        groups: List[Mapping[str, Any]] = [
            group
            for group in rule.get("required_pass_groups") or []
            if isinstance(group, Mapping)
        ]
        for group in rule.get("conditional_required_pass_groups") or []:
            if not isinstance(group, Mapping):
                continue
            condition = str(group.get("when") or "")
            if bool(context.get(condition)):
                groups.append(group)
        missing_groups: List[str] = []
        group_records: List[JsonDict] = []
        for group in groups:
            group_id = str(group.get("id") or "unnamed")
            alternatives = [
                str(value)
                for value in group.get("any_of") or []
                if str(value)
            ]
            passing = sorted(
                {
                    check.id
                    for check in recorder.checks
                    if check.id in alternatives and check.status == "pass"
                }
            )
            satisfied = bool(passing)
            if not satisfied:
                missing_groups.append(group_id)
            group_records.append(
                {
                    "id": group_id,
                    "any_of": alternatives,
                    "satisfied": satisfied,
                    "passing_check_ids": passing,
                }
            )
        selected_errors = [
            check.id for _, check in selected if check.status == "error"
        ]
        status = (
            "fail"
            if selected_errors or missing_groups
            else "pass"
        )
        row = {
            "id": predicate_id,
            "name": rule.get("name") or definition.get("name"),
            "applicability": "applicable",
            "applicability_reason": (
                "an imported or returned trained artifact is declared"
                if applicability_rule == "trained_artifact_declared"
                else "required by the current traceability profile"
            ),
            "status": status,
            "local_status": status,
            "dependencies": dependencies,
            "check_ids": sorted({check.id for _, check in selected}),
            "witness_references": [
                {
                    "check_index": index,
                    "check_id": check.id,
                    "status": check.status,
                }
                for index, check in selected
            ],
            "required_pass_groups": group_records,
            "missing_required_pass_groups": missing_groups,
            "failing_check_ids": sorted(set(selected_errors)),
        }
        rows.append(row)
        by_id[predicate_id] = row

    # Dependency closure is evaluated after all base statuses so the profile
    # need not be serialized in topological order (I2, for example, names I3).
    for row in rows:
        if row["applicability"] != "applicable":
            continue
        failed_dependencies: List[str] = []
        for dependency in row.get("dependencies") or []:
            dependency_id = dependency
            conditional = dependency.endswith("_if_applicable")
            if conditional:
                dependency_id = dependency[: -len("_if_applicable")]
            dependency_row = by_id.get(dependency_id)
            if dependency_row is None:
                failed_dependencies.append(dependency)
                continue
            if (
                conditional
                and dependency_row["applicability"] == "not_applicable"
            ):
                continue
            if dependency_row["status"] != "pass":
                failed_dependencies.append(dependency)
        row["failed_dependencies"] = failed_dependencies
        if failed_dependencies:
            row["status"] = "fail"
    return rows


def _contains_trained_artifact_declaration(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {
                "artifact_manifest_path",
                "trained_artifact_manifest",
                "training_evidence_snapshot",
            } and child not in (None, "", {}, []):
                return True
            if _contains_trained_artifact_declaration(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_trained_artifact_declaration(child) for child in value)
    return False


def _contains_declared_plot_outputs(
    result: Mapping[str, Any],
    benchmark_json: Optional[Mapping[str, Any]],
) -> bool:
    raw = _benchmark_expected_outputs(result, benchmark_json)
    if not isinstance(raw, list):
        return False
    for value in raw:
        if not isinstance(value, str):
            continue
        path = Path(value)
        if (
            len(path.parts) > 1
            and path.parts[0] == "plots"
            and (
                path.suffix.lower() in {".csv", ".pdf", ".png", ".svg"}
                or value.endswith(".plot.json")
            )
        ):
            return True
    return False


def flatten_summary_metrics(summary: Mapping[str, Any]) -> JsonDict:
    metrics: JsonDict = {}
    if isinstance(summary.get("metrics"), Mapping):
        metrics.update(dict(summary.get("metrics") or {}))
    for step in summary.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id") or "")
        step_metrics = step.get("metrics") or {}
        if not isinstance(step_metrics, Mapping):
            continue
        for key, value in dict(step_metrics).items():
            metrics.setdefault(str(key), value)
            if step_id:
                metrics["steps.%s.%s" % (step_id, key)] = value
    return metrics


def _load_json_file(path: Path, recorder: _CheckRecorder, label: str) -> Optional[JsonDict]:
    if path.is_symlink():
        recorder.error(
            "structure.%s" % _check_key(label),
            "%s.json must not be a symlink" % label,
        )
        return None
    if not path.is_file():
        recorder.error("structure.%s" % _check_key(label), "%s.json is missing" % label)
        return None
    try:
        payload = decode_strict_yaml_or_json(
            path.read_text(encoding="utf-8"),
            input_format="json",
        )
    except Exception as exc:
        recorder.error("schema.%s_json" % _check_key(label), "%s.json could not be parsed: %s" % (label, exc))
        return None
    if not isinstance(payload, dict):
        recorder.error("schema.%s_mapping" % _check_key(label), "%s.json must contain a JSON object" % label)
        return None
    recorder.pass_("schema.%s_json" % _check_key(label), "%s.json parses" % label)
    return payload


def _load_optional_json_file(path: Path, recorder: _CheckRecorder, label: str) -> Optional[JsonDict]:
    if not path.is_file():
        recorder.warning("structure.%s_optional" % _check_key(label), "%s.json is not present" % label)
        return None
    return _load_json_file(path, recorder, label)


def _check_run_summary_schema(summary: JsonDict, run_id: str, recorder: _CheckRecorder) -> None:
    _require_type(summary, "schema_version", int, recorder, "summary")
    _require_type(summary, "kind", str, recorder, "summary")
    _require_type(summary, "run_id", str, recorder, "summary")
    _require_type(summary, "recipe_name", str, recorder, "summary")
    _require_type(summary, "status", str, recorder, "summary")
    _require_type(summary, "created_at_utc", str, recorder, "summary")
    _require_type(summary, "steps", list, recorder, "summary")
    _require_type(summary, "metrics", dict, recorder, "summary")
    _require_type(
        summary,
        "measurement_evidence",
        dict,
        recorder,
        "summary",
    )
    _check_exact_version(
        summary.get("schema_version"),
        _RUN_SUMMARY_SCHEMA_VERSION,
        recorder,
        "schema.summary.schema_version_supported",
        "run-summary schema version",
    )
    if summary.get("kind") != _RUN_SUMMARY_KIND:
        recorder.error(
            "schema.summary.kind_value",
            "summary kind is missing or unsupported",
            expected=_RUN_SUMMARY_KIND,
            actual=summary.get("kind"),
        )
    else:
        recorder.pass_("schema.summary.kind_value", "summary kind is supported")
    if summary.get("run_id") != run_id:
        recorder.error("schema.summary.run_id", "summary run_id does not match directory name")


def _check_manifest_schema(manifest: JsonDict, run_id: str, recorder: _CheckRecorder) -> None:
    required = {
        "schema_version": int,
        "kind": str,
        "run_id": str,
        "created_at_utc": str,
        "status": str,
        "recipe": dict,
        "operation_contracts": dict,
        "execution_plan": dict,
        "environment": dict,
        "seed_policy": dict,
        "steps": list,
        "artifacts": list,
        "summary_evidence": dict,
    }
    for key, expected in required.items():
        _require_type(manifest, key, expected, recorder, "manifest")
    if manifest.get("run_id") != run_id:
        recorder.error("schema.manifest.run_id", "manifest run_id does not match directory name")
    _check_exact_version(
        manifest.get("schema_version"),
        _RUN_MANIFEST_SCHEMA_VERSION,
        recorder,
        "schema.manifest.schema_version_supported",
        "run-manifest schema version",
    )
    if manifest.get("kind") != _RUN_MANIFEST_KIND:
        recorder.error(
            "schema.manifest.kind_value",
            "manifest kind is missing or unsupported",
            expected=_RUN_MANIFEST_KIND,
            actual=manifest.get("kind"),
        )
    else:
        recorder.pass_("schema.manifest.kind_value", "manifest kind is supported")
    recipe = manifest.get("recipe") if isinstance(manifest.get("recipe"), Mapping) else {}
    if not recipe.get("sha256"):
        recorder.error("manifest.recipe_sha", "manifest recipe SHA is missing")
    else:
        recorder.pass_("manifest.recipe_sha", "manifest recipe SHA is present")
    execution_plan = (
        manifest.get("execution_plan")
        if isinstance(manifest.get("execution_plan"), Mapping)
        else {}
    )
    if not _is_sha256(execution_plan.get("sha256")):
        recorder.error(
            "manifest.execution_plan_sha",
            "manifest execution-plan SHA-256 is missing or malformed",
        )
    environment = (
        manifest.get("environment")
        if isinstance(manifest.get("environment"), Mapping)
        else {}
    )
    git = environment.get("git") if isinstance(environment.get("git"), Mapping) else {}
    commit = str(git.get("commit") or "")
    if not git.get("available") or len(commit) != 40:
        recorder.error(
            "manifest.source_commit",
            "manifest does not identify the executed Git commit",
        )
    project_files = (
        environment.get("project_files")
        if isinstance(environment.get("project_files"), Mapping)
        else {}
    )
    if not project_files or any(
        not isinstance(row, Mapping) or not _is_sha256(row.get("sha256"))
        for row in project_files.values()
    ):
        recorder.error(
            "manifest.environment_files",
            "manifest environment does not content-identify project files",
        )


def _check_summary_manifest_consistency(summary: JsonDict, manifest: JsonDict, recorder: _CheckRecorder) -> None:
    if summary.get("status") != manifest.get("status"):
        recorder.error("manifest.status", "summary status and manifest status differ")
    else:
        recorder.pass_("manifest.status", "summary status matches manifest status")
    summary_steps = summary.get("steps") if isinstance(summary.get("steps"), list) else []
    manifest_steps = manifest.get("steps") if isinstance(manifest.get("steps"), list) else []
    if len(summary_steps) != len(manifest_steps):
        recorder.error(
            "manifest.steps.count",
            "summary and manifest step counts differ",
            summary_steps=len(summary_steps),
            manifest_steps=len(manifest_steps),
        )
    else:
        recorder.pass_("manifest.steps.count", "summary and manifest step counts match")
    _check_run_level_summary_evidence(summary, manifest, recorder)
    _check_step_summary_evidence(summary_steps, manifest_steps, manifest, recorder)


def _check_run_level_summary_evidence(
    summary: JsonDict,
    manifest: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    evidence = manifest.get("summary_evidence")
    if not isinstance(evidence, Mapping):
        recorder.error(
            "manifest.summary_evidence",
            "manifest does not independently preserve run-summary evidence",
        )
        return
    _check_exact_version(
        evidence.get("schema_version"),
        1,
        recorder,
        "manifest.summary_evidence.schema_version",
        "run-summary evidence schema version",
    )
    if evidence.get("kind") != "noema.run_summary_evidence":
        recorder.error(
            "manifest.summary_evidence.kind",
            "run-summary evidence kind is missing or unsupported",
        )
    else:
        recorder.pass_(
            "manifest.summary_evidence.kind",
            "run-summary evidence kind is supported",
        )
    for field_name in ("metrics", "measurement_evidence"):
        preserved = evidence.get(field_name)
        current = summary.get(field_name)
        digest_field = "%s_sha256" % field_name
        if not isinstance(preserved, Mapping) or not isinstance(current, Mapping):
            recorder.error(
                "manifest.summary_evidence.%s" % field_name,
                "summary and manifest %s evidence must be objects" % field_name,
            )
            continue
        try:
            preserved_digest = canonical_json_sha256(dict(preserved))
            current_digest = canonical_json_sha256(dict(current))
        except (TypeError, ValueError) as exc:
            recorder.error(
                "manifest.summary_evidence.%s" % field_name,
                "%s evidence is not canonical JSON: %s" % (field_name, exc),
            )
            continue
        if evidence.get(digest_field) != preserved_digest:
            recorder.error(
                "manifest.summary_evidence.%s_digest" % field_name,
                "manifest %s digest does not match its preserved values"
                % field_name,
            )
        elif current_digest != preserved_digest:
            recorder.error(
                "manifest.summary_evidence.%s_binding" % field_name,
                "summary %s values differ from manifest evidence" % field_name,
            )
        else:
            recorder.pass_(
                "manifest.summary_evidence.%s_binding" % field_name,
                "summary %s values match manifest evidence" % field_name,
            )


def _check_step_summary_evidence(
    summary_steps: Sequence[Any],
    manifest_steps: Sequence[Any],
    manifest: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    summary_by_id = _unique_steps_by_id(summary_steps, "summary", recorder)
    manifest_by_id = _unique_steps_by_id(manifest_steps, "manifest", recorder)
    if set(summary_by_id) != set(manifest_by_id):
        recorder.error(
            "manifest.steps.identity",
            "summary and manifest step identities differ",
            summary=sorted(summary_by_id),
            manifest=sorted(manifest_by_id),
        )
    else:
        recorder.pass_(
            "manifest.steps.identity",
            "summary and manifest step identities match",
        )

    artifact_records: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for index, raw_record in enumerate(manifest.get("artifacts") or []):
        if not isinstance(raw_record, Mapping):
            continue
        identity = (
            str(raw_record.get("step_id") or ""),
            str(raw_record.get("output_name") or ""),
        )
        if not all(identity) or identity in artifact_records:
            recorder.error(
                "manifest.artifacts.identity",
                "manifest artifact identities must be unique and non-empty",
                index=index,
            )
            continue
        artifact_records[identity] = raw_record

    for step_id in sorted(set(summary_by_id).intersection(manifest_by_id)):
        summary_step = summary_by_id[step_id]
        manifest_step = manifest_by_id[step_id]
        if (
            summary_step.get("op") != manifest_step.get("op")
            or summary_step.get("status") != manifest_step.get("status")
        ):
            recorder.error(
                "manifest.step.identity",
                "summary and manifest operation/status differ for step %s"
                % step_id,
            )
        _check_step_metric_evidence(step_id, summary_step, manifest_step, recorder)
        _check_step_output_evidence(
            step_id,
            summary_step,
            manifest_step,
            artifact_records,
            recorder,
        )


def _unique_steps_by_id(
    raw_steps: Sequence[Any],
    scope: str,
    recorder: _CheckRecorder,
) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            recorder.error(
                "manifest.%s_step" % scope,
                "%s step %d is not an object" % (scope, index),
            )
            continue
        step_id = str(raw_step.get("id") or "")
        if not step_id or step_id in rows:
            recorder.error(
                "manifest.%s_step_id" % scope,
                "%s step ids must be unique and non-empty" % scope,
            )
            continue
        rows[step_id] = raw_step
    return rows


def _check_step_metric_evidence(
    step_id: str,
    summary_step: Mapping[str, Any],
    manifest_step: Mapping[str, Any],
    recorder: _CheckRecorder,
) -> None:
    summary_metrics = summary_step.get("metrics")
    manifest_metrics = manifest_step.get("metrics")
    if not isinstance(summary_metrics, Mapping) or not isinstance(
        manifest_metrics, Mapping
    ):
        recorder.error(
            "manifest.step_metrics",
            "step %s metrics are not independently preserved" % step_id,
        )
        return
    try:
        summary_digest = canonical_json_sha256(dict(summary_metrics))
        manifest_digest = canonical_json_sha256(dict(manifest_metrics))
    except (TypeError, ValueError) as exc:
        recorder.error(
            "manifest.step_metrics",
            "step %s metrics are not canonical JSON: %s" % (step_id, exc),
        )
        return
    expected_keys = sorted(str(key) for key in manifest_metrics)
    if manifest_step.get("metrics_keys") != expected_keys:
        recorder.error(
            "manifest.step_metric_keys",
            "step %s metric keys do not match preserved values" % step_id,
        )
    if manifest_step.get("metrics_sha256") != manifest_digest:
        recorder.error(
            "manifest.step_metrics_digest",
            "step %s manifest metric digest is missing or mismatched" % step_id,
        )
    elif summary_digest != manifest_digest:
        recorder.error(
            "manifest.step_metrics_binding",
            "step %s summary metric values differ from manifest evidence"
            % step_id,
        )
    else:
        recorder.pass_(
            "manifest.step_metrics_binding",
            "step %s metric values match manifest evidence" % step_id,
        )


def _check_step_output_evidence(
    step_id: str,
    summary_step: Mapping[str, Any],
    manifest_step: Mapping[str, Any],
    artifact_records: Mapping[Tuple[str, str], Mapping[str, Any]],
    recorder: _CheckRecorder,
) -> None:
    summary_outputs = summary_step.get("outputs")
    manifest_outputs = manifest_step.get("outputs")
    if not isinstance(summary_outputs, Mapping) or not isinstance(
        manifest_outputs, Mapping
    ):
        recorder.error(
            "manifest.step_outputs",
            "step %s outputs must be objects in summary and manifest" % step_id,
        )
        return
    if set(summary_outputs) != set(manifest_outputs):
        recorder.error(
            "manifest.step_outputs.identity",
            "step %s summary and manifest output identities differ" % step_id,
        )
    for output_name in sorted(set(summary_outputs).intersection(manifest_outputs)):
        summary_output = summary_outputs[output_name]
        manifest_output = manifest_outputs[output_name]
        if not isinstance(summary_output, Mapping) or not isinstance(
            manifest_output, Mapping
        ):
            recorder.error(
                "manifest.output_schema",
                "step %s output %s evidence must be objects"
                % (step_id, output_name),
            )
            continue
        artifact_record = artifact_records.get((step_id, str(output_name)))
        if artifact_record is None or dict(artifact_record) != dict(manifest_output):
            recorder.error(
                "manifest.output_artifact_link",
                "step %s output %s does not match the top-level artifact record"
                % (step_id, output_name),
            )
        summary_metadata = summary_output.get("metadata")
        manifest_metadata = manifest_output.get("metadata")
        if not isinstance(summary_metadata, Mapping) or not isinstance(
            manifest_metadata, Mapping
        ):
            recorder.error(
                "manifest.output_metadata",
                "step %s output %s metadata must be objects"
                % (step_id, output_name),
            )
            continue
        try:
            summary_digest = canonical_json_sha256(dict(summary_metadata))
            manifest_digest = canonical_json_sha256(dict(manifest_metadata))
        except (TypeError, ValueError) as exc:
            recorder.error(
                "manifest.output_metadata",
                "step %s output %s metadata is not canonical JSON: %s"
                % (step_id, output_name, exc),
            )
            continue
        if manifest_output.get("metadata_sha256") != manifest_digest:
            recorder.error(
                "manifest.output_metadata_digest",
                "step %s output %s metadata digest is missing or mismatched"
                % (step_id, output_name),
            )
        elif manifest_output.get("producer_metrics_sha256") != manifest_step.get(
            "metrics_sha256"
        ):
            recorder.error(
                "manifest.output_metric_binding",
                "step %s output %s does not bind the producer metric evidence"
                % (step_id, output_name),
            )
        elif summary_digest != manifest_digest:
            recorder.error(
                "manifest.output_metadata_binding",
                "step %s output %s summary metadata differs from artifact evidence"
                % (step_id, output_name),
            )
        elif (
            summary_output.get("kind") != manifest_output.get("kind")
            or summary_output.get("sha256") != manifest_output.get("sha256")
        ):
            recorder.error(
                "manifest.output_identity_binding",
                "step %s output %s kind or SHA differs from artifact evidence"
                % (step_id, output_name),
            )
        else:
            recorder.pass_(
                "manifest.output_metadata_binding",
                "step %s output %s metadata matches artifact evidence"
                % (step_id, output_name),
            )


def _check_summary_content_binding(
    run_dir: Path,
    manifest: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    descriptor = manifest.get("summary")
    if not isinstance(descriptor, Mapping):
        recorder.error(
            "manifest.summary_binding",
            "manifest does not content-bind summary.json",
        )
        return
    expected_keys = {"kind", "relative_path", "sha256", "size_bytes"}
    if set(descriptor) != expected_keys or (
        descriptor.get("kind") != "noema.run_summary"
        or descriptor.get("relative_path") != "summary.json"
        or not _is_sha256(descriptor.get("sha256"))
        or isinstance(descriptor.get("size_bytes"), bool)
        or not isinstance(descriptor.get("size_bytes"), int)
        or int(descriptor.get("size_bytes")) < 0
    ):
        recorder.error(
            "manifest.summary_binding",
            "manifest summary descriptor is malformed",
        )
        return
    summary_path = run_dir / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        recorder.error(
            "manifest.summary_binding",
            "bound summary.json is missing or unsafe",
        )
        return
    actual_size = int(summary_path.stat().st_size)
    actual_sha = file_sha256(summary_path)
    if (
        descriptor.get("size_bytes") != actual_size
        or descriptor.get("sha256") != actual_sha
    ):
        recorder.error(
            "manifest.summary_binding",
            "summary.json does not match its manifest content binding",
            expected_sha256=descriptor.get("sha256"),
            actual_sha256=actual_sha,
            expected_size_bytes=descriptor.get("size_bytes"),
            actual_size_bytes=actual_size,
        )
        return
    recorder.pass_(
        "manifest.summary_binding",
        "summary.json matches its manifest content binding",
    )


def _check_recipe_identity(
    recipe: JsonDict,
    manifest: JsonDict,
    summary: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    manifest_recipe = dict(manifest.get("recipe") or {})
    manifest_sha = str(manifest_recipe.get("sha256") or "")
    computed_sha = canonical_json_sha256(recipe)
    if not manifest_sha:
        return
    if computed_sha != manifest_sha:
        recorder.error(
            "recipe.sha256",
            "recipe.json SHA does not match manifest recipe SHA",
            expected=manifest_sha,
            actual=computed_sha,
        )
    else:
        recorder.pass_("recipe.sha256", "recipe.json SHA matches manifest recipe SHA")
    if summary is not None and summary.get("recipe_sha256") and summary.get("recipe_sha256") != manifest_sha:
        recorder.error("recipe.summary_sha", "summary recipe SHA does not match manifest recipe SHA")


def _check_recipe_lint(recipe_json: JsonDict, registry: OperationRegistry, recorder: _CheckRecorder) -> None:
    try:
        recipe = recipe_from_dict(recipe_json)
        validate_recipe_against_registry(recipe, registry)
        report = lint_recipe_invariants(recipe, registry, strict=True)
    except Exception as exc:
        recorder.error("recipe.lint", "recipe strict lint could not run cleanly: %s" % exc)
        return
    for issue in report.get("issues") or []:
        message = str(issue.get("message") or issue)
        if issue.get("severity") == "error":
            recorder.error("recipe.lint.%s" % (issue.get("code") or "error"), message)
        else:
            recorder.warning("recipe.lint.%s" % (issue.get("code") or "warning"), message)
    if not report.get("issues"):
        recorder.pass_("recipe.lint", "recipe passes strict lint")


def _check_execution_plan_evidence(
    run_dir: Path,
    manifest: JsonDict,
    summary: Optional[JsonDict],
    recipe: Optional[JsonDict],
    recorder: _CheckRecorder,
    *,
    registry: Optional[OperationRegistry] = None,
) -> None:
    """Validate the self-contained execution binding evidence for a run.

    Publication verification is fail closed: a bundle without an execution
    plan cannot establish what implementation was bound before execution and
    is therefore invalid.  Every digest and cross-file link is required so a
    partial or edited plan cannot silently verify.  Without a registry this
    establishes internal consistency only.  Supplying a registry additionally
    anchors every used embedded operation contract to the currently registered
    contract.
    """

    manifest_evidence = manifest.get("execution_plan")
    plan_path = run_dir / "execution-plan.json"
    if manifest_evidence is None:
        recorder.error(
            "execution_plan.required",
            "manifest does not bind required execution-plan evidence",
            execution_plan_file_present=plan_path.exists(),
        )
        return
    if not isinstance(manifest_evidence, Mapping):
        recorder.error(
            "execution_plan.manifest_schema",
            "manifest.execution_plan must be an object",
        )
        return

    _check_exact_version(
        manifest_evidence.get("schema_version"),
        EXECUTION_PLAN_SCHEMA_VERSION,
        recorder,
        "execution_plan.manifest_schema_version",
        "manifest execution-plan schema version",
    )
    if manifest_evidence.get("kind") != EXECUTION_PLAN_KIND:
        recorder.error(
            "execution_plan.manifest_kind",
            "manifest execution-plan kind is invalid",
            expected=EXECUTION_PLAN_KIND,
            actual=manifest_evidence.get("kind"),
        )
    else:
        recorder.pass_(
            "execution_plan.manifest_kind",
            "manifest execution-plan kind is valid",
        )
    if manifest_evidence.get("path") != "execution-plan.json":
        recorder.error(
            "execution_plan.manifest_path",
            "manifest execution-plan path must be execution-plan.json",
            actual=manifest_evidence.get("path"),
        )
    else:
        recorder.pass_(
            "execution_plan.manifest_path",
            "manifest references execution-plan.json",
        )

    plan = _load_json_file(plan_path, recorder, "execution-plan")
    if plan is None:
        return
    authored_recipe = _load_json_file(
        run_dir / "recipe.authored.json",
        recorder,
        "recipe.authored",
    )
    _check_exact_version(
        plan.get("schema_version"),
        EXECUTION_PLAN_SCHEMA_VERSION,
        recorder,
        "execution_plan.schema_version",
        "execution-plan schema version",
    )
    if plan.get("kind") != EXECUTION_PLAN_KIND:
        recorder.error(
            "execution_plan.kind",
            "execution-plan kind is invalid",
            expected=EXECUTION_PLAN_KIND,
            actual=plan.get("kind"),
        )
    else:
        recorder.pass_("execution_plan.kind", "execution-plan kind is valid")

    declared_plan_sha = plan.get("sha256")
    computed_plan_sha = _canonical_digest_without(plan, "sha256")
    if not _is_sha256(declared_plan_sha):
        recorder.error(
            "execution_plan.sha256_schema",
            "execution-plan SHA-256 is missing or malformed",
        )
    elif declared_plan_sha != computed_plan_sha:
        recorder.error(
            "execution_plan.sha256",
            "execution-plan SHA-256 does not match its canonical payload",
            expected=declared_plan_sha,
            actual=computed_plan_sha,
        )
    else:
        recorder.pass_(
            "execution_plan.sha256",
            "execution-plan SHA-256 matches its canonical payload",
        )
    _check_digest_reference(
        manifest_evidence.get("sha256"),
        declared_plan_sha,
        recorder,
        "execution_plan.manifest_sha256",
        "manifest execution-plan SHA",
    )

    summary_evidence = summary.get("execution_plan") if isinstance(summary, Mapping) else None
    if not isinstance(summary_evidence, Mapping):
        recorder.error(
            "execution_plan.summary_schema",
            "summary.execution_plan is required when the manifest references a plan",
        )
    else:
        _check_exact_version(
            summary_evidence.get("schema_version"),
            EXECUTION_PLAN_SCHEMA_VERSION,
            recorder,
            "execution_plan.summary_schema_version",
            "summary execution-plan schema version",
        )
        if summary_evidence.get("path") != "execution-plan.json":
            recorder.error(
                "execution_plan.summary_path",
                "summary execution-plan path must be execution-plan.json",
                actual=summary_evidence.get("path"),
            )
        else:
            recorder.pass_(
                "execution_plan.summary_path",
                "summary references execution-plan.json",
            )
        _check_digest_reference(
            summary_evidence.get("sha256"),
            declared_plan_sha,
            recorder,
            "execution_plan.summary_sha256",
            "summary execution-plan SHA",
        )

    runner = plan.get("runner")
    if not isinstance(runner, str) or runner not in MATERIALIZATION_RUNNERS:
        recorder.error(
            "execution_plan.runner",
            "execution-plan runner is missing, unsupported, or invalid",
            expected=sorted(MATERIALIZATION_RUNNERS),
            actual=runner,
        )
    else:
        recorder.pass_("execution_plan.runner", "execution-plan runner is present")
        for scope, evidence in (
            ("manifest", manifest_evidence),
            ("summary", summary_evidence),
        ):
            if not isinstance(evidence, Mapping):
                continue
            if evidence.get("runner") != runner:
                recorder.error(
                    "execution_plan.%s_runner" % scope,
                    "%s execution-plan runner does not match execution-plan.json" % scope,
                    expected=runner,
                    actual=evidence.get("runner"),
                )
            else:
                recorder.pass_(
                    "execution_plan.%s_runner" % scope,
                    "%s execution-plan runner matches execution-plan.json" % scope,
                )

    _check_execution_plan_recipe_links(
        plan,
        manifest_evidence,
        manifest,
        summary,
        recipe,
        authored_recipe,
        recorder,
    )
    _check_execution_plan_cache_evidence(
        manifest_evidence,
        summary_evidence,
        plan,
        manifest,
        recorder,
    )
    plan_steps, operation_digests = _check_execution_plan_contracts_and_steps(
        plan,
        manifest,
        recorder,
        registry=registry,
    )
    _check_plan_recipe_steps(plan, recipe, recorder)
    _check_plan_step_copy(
        manifest_evidence.get("steps"),
        plan_steps,
        operation_digests,
        recorder,
    )
    completed = str(manifest.get("status") or "").lower() == "completed"
    _check_runtime_step_bindings(
        manifest.get("steps"),
        plan_steps,
        operation_digests,
        recorder,
        scope="manifest",
        require_all=completed,
    )
    _check_runtime_step_bindings(
        summary.get("steps") if isinstance(summary, Mapping) else None,
        plan_steps,
        operation_digests,
        recorder,
        scope="summary",
        require_all=completed,
    )
    _check_runtime_backend_namespaces(
        summary.get("steps") if isinstance(summary, Mapping) else None,
        plan_steps,
        recorder,
        require_all=completed,
    )


def _check_execution_runtime_evidence(
    summary: JsonDict,
    manifest: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    """Validate optional P3 scheduling evidence without breaking legacy runs."""

    summary_execution = summary.get("execution")
    manifest_execution = manifest.get("execution")
    if summary_execution is None and manifest_execution is None:
        return
    if not isinstance(manifest_execution, Mapping):
        recorder.error(
            "execution.manifest_schema",
            "manifest.execution must be an object when execution evidence is present",
        )
        return
    if not isinstance(summary_execution, Mapping):
        recorder.error(
            "execution.summary_schema",
            "summary.execution must be an object when execution evidence is present",
        )
        return
    if dict(manifest_execution) != dict(summary_execution):
        recorder.error(
            "execution.summary_manifest_link",
            "summary and manifest execution configuration differ",
        )
    else:
        recorder.pass_(
            "execution.summary_manifest_link",
            "summary and manifest execution configuration match",
        )

    workers = manifest_execution.get("parallel_workers")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or workers > 32
    ):
        recorder.error(
            "execution.parallel_workers",
            "execution parallel_workers must be an integer from 1 through 32",
            actual=workers,
        )
        return
    recorder.pass_(
        "execution.parallel_workers",
        "execution parallel_workers is valid",
    )
    expected_mode = "parallel" if workers > 1 else "sequential"
    mode = manifest_execution.get("mode")
    if mode != expected_mode:
        recorder.error(
            "execution.mode",
            "execution mode does not match parallel_workers",
            expected=expected_mode,
            actual=mode,
        )
    else:
        recorder.pass_("execution.mode", "execution mode matches parallel_workers")


def _check_execution_plan_cache_evidence(
    manifest_evidence: Mapping[str, Any],
    summary_evidence: Optional[Mapping[str, Any]],
    plan: JsonDict,
    manifest: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    """Validate optional cache provenance and all links available in a run."""

    manifest_cache = manifest_evidence.get("cache")
    summary_cache = (
        summary_evidence.get("cache")
        if isinstance(summary_evidence, Mapping)
        else None
    )
    if manifest_cache is None and summary_cache is None:
        return
    if not isinstance(manifest_cache, Mapping):
        recorder.error(
            "execution_plan.cache_manifest_schema",
            "manifest execution-plan cache evidence must be an object",
        )
        return
    if not isinstance(summary_cache, Mapping):
        recorder.error(
            "execution_plan.cache_summary_schema",
            "summary execution-plan cache evidence must be an object",
        )
        return
    if dict(manifest_cache) != dict(summary_cache):
        recorder.error(
            "execution_plan.cache_summary_link",
            "summary and manifest execution-plan cache evidence differ",
        )
    else:
        recorder.pass_(
            "execution_plan.cache_summary_link",
            "summary and manifest execution-plan cache evidence match",
        )

    _check_exact_version(
        manifest_cache.get("schema_version"),
        EXECUTION_PLAN_CACHE_EVIDENCE_SCHEMA_VERSION,
        recorder,
        "execution_plan.cache_schema_version",
        "execution-plan cache evidence schema version",
    )
    _check_exact_version(
        manifest_cache.get("key_schema_version"),
        EXECUTION_PLAN_CACHE_KEY_SCHEMA_VERSION,
        recorder,
        "execution_plan.cache_key_schema_version",
        "execution-plan cache key schema version",
    )

    outcome = manifest_cache.get("outcome")
    enabled = manifest_cache.get("enabled")
    if outcome not in {"hit", "miss", "bypass"}:
        recorder.error(
            "execution_plan.cache_outcome",
            "execution-plan cache outcome is invalid",
            actual=outcome,
        )
    elif not isinstance(enabled, bool) or enabled != (outcome != "bypass"):
        recorder.error(
            "execution_plan.cache_enabled",
            "execution-plan cache enabled flag does not match its outcome",
            expected=outcome != "bypass",
            actual=enabled,
        )
    else:
        recorder.pass_(
            "execution_plan.cache_outcome",
            "execution-plan cache outcome and enabled flag are consistent",
        )

    _check_digest_reference(
        manifest_cache.get("plan_sha256"),
        plan.get("sha256"),
        recorder,
        "execution_plan.cache_plan_sha256",
        "execution-plan cache plan SHA",
    )
    plan_recipe = plan.get("recipe")
    effective_sha = (
        plan_recipe.get("sha256") if isinstance(plan_recipe, Mapping) else None
    )
    _check_digest_reference(
        manifest_cache.get("effective_recipe_sha256"),
        effective_sha,
        recorder,
        "execution_plan.cache_effective_recipe_sha256",
        "execution-plan cache effective recipe SHA",
    )
    authored_sha = manifest.get("recipe", {}).get("authored_sha256") if isinstance(
        manifest.get("recipe"), Mapping
    ) else None
    _check_digest_reference(
        manifest_cache.get("authored_recipe_sha256"),
        authored_sha,
        recorder,
        "execution_plan.cache_authored_recipe_sha256",
        "execution-plan cache authored recipe SHA",
    )
    for field, label in (
        ("key_sha256", "execution-plan cache key SHA"),
        ("operation_registry_sha256", "execution-plan cache registry SHA"),
        ("planner_validation_sha256", "execution-plan cache validation-catalog SHA"),
    ):
        if _is_sha256(manifest_cache.get(field)):
            recorder.pass_("execution_plan.cache_%s" % field, "%s is valid" % label)
        else:
            recorder.error(
                "execution_plan.cache_%s" % field,
                "%s is missing or malformed" % label,
            )

    entry_count = manifest_cache.get("entry_count")
    max_entries = manifest_cache.get("max_entries")
    valid_entry_count = (
        isinstance(entry_count, int)
        and not isinstance(entry_count, bool)
        and entry_count >= 0
    )
    valid_max_entries = (
        isinstance(max_entries, int)
        and not isinstance(max_entries, bool)
        and max_entries >= 1
    )
    if not valid_entry_count or not valid_max_entries or entry_count > max_entries:
        recorder.error(
            "execution_plan.cache_capacity",
            "execution-plan cache entry counts are malformed or exceed capacity",
            entry_count=entry_count,
            max_entries=max_entries,
        )
    else:
        recorder.pass_(
            "execution_plan.cache_capacity",
            "execution-plan cache entry counts are within capacity",
        )
def _check_execution_plan_recipe_links(
    plan: JsonDict,
    manifest_evidence: Mapping[str, Any],
    manifest: JsonDict,
    summary: Optional[JsonDict],
    recipe: Optional[JsonDict],
    authored_recipe: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    plan_recipe = plan.get("recipe")
    if not isinstance(plan_recipe, Mapping):
        recorder.error(
            "execution_plan.recipe_schema",
            "execution-plan recipe identity must be an object",
        )
        return
    effective_sha = plan_recipe.get("sha256")
    if not _is_sha256(effective_sha):
        recorder.error(
            "execution_plan.recipe_sha256_schema",
            "execution-plan effective recipe SHA-256 is missing or malformed",
        )
    if recipe is not None:
        _check_digest_reference(
            effective_sha,
            canonical_json_sha256(recipe),
            recorder,
            "execution_plan.recipe_sha256",
            "execution-plan effective recipe SHA",
        )
        recipe_steps = recipe.get("steps") if isinstance(recipe.get("steps"), list) else []
        plan_step_count = plan_recipe.get("step_count")
        if not isinstance(plan_step_count, int) or isinstance(plan_step_count, bool):
            recorder.error(
                "execution_plan.recipe_step_count",
                "execution-plan recipe step_count is missing or invalid",
            )
        elif plan_step_count != len(recipe_steps):
            recorder.error(
                "execution_plan.recipe_step_count",
                "execution-plan recipe step_count does not match recipe.json",
                expected=len(recipe_steps),
                actual=plan_step_count,
            )
        else:
            recorder.pass_(
                "execution_plan.recipe_step_count",
                "execution-plan recipe step_count matches recipe.json",
            )
        if plan_recipe.get("name") != recipe.get("name"):
            recorder.error(
                "execution_plan.recipe_name",
                "execution-plan recipe name does not match recipe.json",
            )
        else:
            recorder.pass_(
                "execution_plan.recipe_name",
                "execution-plan recipe name matches recipe.json",
            )

    manifest_recipe = manifest.get("recipe") if isinstance(manifest.get("recipe"), Mapping) else {}
    links = (
        (
            manifest_evidence.get("effective_recipe_sha256"),
            "execution_plan.manifest_effective_recipe_sha256",
            "manifest execution-plan effective recipe SHA",
        ),
        (
            manifest_recipe.get("effective_sha256"),
            "execution_plan.manifest_recipe_sha256",
            "manifest effective recipe SHA",
        ),
    )
    for value, check_id, label in links:
        _check_digest_reference(value, effective_sha, recorder, check_id, label)
    if isinstance(summary, Mapping):
        _check_digest_reference(
            summary.get("effective_recipe_sha256"),
            effective_sha,
            recorder,
            "execution_plan.summary_recipe_sha256",
            "summary effective recipe SHA",
        )

    if authored_recipe is not None:
        authored_sha = canonical_json_sha256(authored_recipe)
        _check_digest_reference(
            manifest_evidence.get("authored_recipe_sha256"),
            authored_sha,
            recorder,
            "execution_plan.manifest_authored_recipe_sha256",
            "manifest execution-plan authored recipe SHA",
        )
        _check_digest_reference(
            manifest_recipe.get("authored_sha256"),
            authored_sha,
            recorder,
            "execution_plan.manifest_recipe_authored_sha256",
            "manifest authored recipe SHA",
        )
        if isinstance(summary, Mapping):
            _check_digest_reference(
                summary.get("authored_recipe_sha256"),
                authored_sha,
                recorder,
                "execution_plan.summary_authored_recipe_sha256",
                "summary authored recipe SHA",
            )


def _check_execution_plan_contracts_and_steps(
    plan: JsonDict,
    manifest: JsonDict,
    recorder: _CheckRecorder,
    *,
    registry: Optional[OperationRegistry] = None,
) -> Tuple[Dict[str, JsonDict], Dict[str, str]]:
    plan_contracts = plan.get("operation_contracts")
    if not isinstance(plan_contracts, Mapping):
        recorder.error(
            "execution_plan.operation_contracts_schema",
            "execution-plan operation_contracts must be an object",
        )
        operations: Mapping[str, Any] = {}
        declared_contracts_sha = None
    else:
        _check_exact_version(
            plan_contracts.get("schema_version"),
            1,
            recorder,
            "execution_plan.operation_contracts_schema_version",
            "execution-plan operation-contract schema version",
        )
        operations_value = plan_contracts.get("operations")
        if not isinstance(operations_value, Mapping):
            recorder.error(
                "execution_plan.operation_contracts_operations",
                "execution-plan operation contracts must be an object",
            )
            operations = {}
        else:
            operations = operations_value
        declared_contracts_sha = plan_contracts.get("sha256")
        computed_contracts_sha = canonical_json_sha256(operations)
        _check_digest_reference(
            declared_contracts_sha,
            computed_contracts_sha,
            recorder,
            "execution_plan.operation_contracts_sha256",
            "execution-plan operation-contract SHA",
        )

    operation_digests: Dict[str, str] = {}
    operation_contracts: Dict[str, Mapping[str, Any]] = {}
    for operation_id, contract in operations.items():
        if not isinstance(operation_id, str) or not isinstance(contract, Mapping):
            recorder.error(
                "execution_plan.operation_contract",
                "execution-plan operation contract entries must map string ids to objects",
            )
            continue
        if contract.get("id") != operation_id:
            recorder.error(
                "execution_plan.operation_contract_id",
                "embedded operation contract id differs from its operation_contracts key",
                expected=operation_id,
                actual=contract.get("id"),
            )
        else:
            recorder.pass_(
                "execution_plan.operation_contract_id",
                "embedded operation contract id matches its operation_contracts key",
                operation_id=operation_id,
            )
        operation_contracts[operation_id] = contract
        operation_digests[operation_id] = canonical_json_sha256(contract)

    manifest_contracts = manifest.get("operation_contracts")
    if not isinstance(manifest_contracts, Mapping):
        recorder.error(
            "execution_plan.manifest_operation_contracts",
            "manifest operation_contracts must be an object",
        )
    else:
        manifest_operations = manifest_contracts.get("operations")
        if not isinstance(manifest_operations, Mapping):
            recorder.error(
                "execution_plan.manifest_operation_contracts",
                "manifest operation contracts must be an object",
            )
        else:
            manifest_computed_sha = canonical_json_sha256(manifest_operations)
            _check_digest_reference(
                manifest_contracts.get("sha256"),
                manifest_computed_sha,
                recorder,
                "execution_plan.manifest_operation_contracts_sha256",
                "manifest operation-contract SHA",
            )
            _check_digest_reference(
                manifest_contracts.get("sha256"),
                declared_contracts_sha,
                recorder,
                "execution_plan.operation_contracts_link",
                "manifest operation-contract SHA link",
            )
            if canonical_json_sha256(manifest_operations) != canonical_json_sha256(operations):
                recorder.error(
                    "execution_plan.operation_contracts_payload",
                    "manifest operation contracts differ from execution-plan.json",
                )
            else:
                recorder.pass_(
                    "execution_plan.operation_contracts_payload",
                    "manifest operation contracts match execution-plan.json",
                )

    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list):
        recorder.error("execution_plan.steps_schema", "execution-plan steps must be a list")
        return {}, operation_digests
    if registry is not None:
        _check_operation_contracts_against_registry(
            raw_steps,
            operation_contracts,
            registry,
            recorder,
        )
    plan_steps: Dict[str, JsonDict] = {}
    plan_runner = plan.get("runner")
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            recorder.error(
                "execution_plan.step_schema",
                "execution-plan step %d must be an object" % index,
            )
            continue
        step = dict(raw_step)
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            recorder.error(
                "execution_plan.step_id",
                "execution-plan step %d has no valid step_id" % index,
            )
            continue
        if step_id in plan_steps:
            recorder.error(
                "execution_plan.step_id",
                "execution-plan has duplicate step_id %s" % step_id,
            )
            continue
        if isinstance(plan_runner, str):
            if step.get("runner") != plan_runner:
                recorder.error(
                    "execution_plan.step_runner",
                    "execution-plan step %s runner differs from the plan runner"
                    % step_id,
                    expected=plan_runner,
                    actual=step.get("runner"),
                )
            else:
                recorder.pass_(
                    "execution_plan.step_runner",
                    "execution-plan step %s runner matches the plan runner" % step_id,
                )
        _check_planned_step(
            step,
            operation_digests,
            recorder,
            "execution-plan",
            operation_contracts=operation_contracts,
        )
        plan_steps[step_id] = step

    plan_recipe = plan.get("recipe") if isinstance(plan.get("recipe"), Mapping) else {}
    if isinstance(plan_recipe.get("step_count"), int) and plan_recipe.get("step_count") != len(raw_steps):
        recorder.error(
            "execution_plan.steps_count",
            "execution-plan step count differs from its recipe identity",
        )
    else:
        recorder.pass_(
            "execution_plan.steps_count",
            "execution-plan step count matches its recipe identity",
        )
    return plan_steps, operation_digests


def _check_operation_contracts_against_registry(
    raw_steps: Sequence[Any],
    operation_contracts: Mapping[str, Mapping[str, Any]],
    registry: OperationRegistry,
    recorder: _CheckRecorder,
) -> None:
    operation_ids = sorted(
        {
            step.get("operation_id")
            for step in raw_steps
            if isinstance(step, Mapping)
            and isinstance(step.get("operation_id"), str)
            and step.get("operation_id")
        }
    )
    for operation_id in operation_ids:
        embedded = operation_contracts.get(operation_id)
        if not isinstance(embedded, Mapping):
            recorder.error(
                "execution_plan.operation_contract_registry",
                "execution-plan is missing the embedded operation contract required "
                "for registry verification",
                operation_id=operation_id,
            )
            continue
        try:
            registered = registry.get(operation_id).describe()
        except OperationError as exc:
            recorder.error(
                "execution_plan.operation_contract_registry",
                "execution-plan operation contract cannot be resolved in the supplied registry",
                operation_id=operation_id,
                error=str(exc),
            )
            continue
        if canonical_json_sha256(embedded) != canonical_json_sha256(registered):
            recorder.error(
                "execution_plan.operation_contract_registry",
                "embedded operation contract differs from the supplied registry",
                operation_id=operation_id,
            )
        else:
            recorder.pass_(
                "execution_plan.operation_contract_registry",
                "embedded operation contract matches the supplied registry",
                operation_id=operation_id,
            )


def _check_plan_recipe_steps(
    plan: JsonDict,
    recipe: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    if recipe is None:
        return
    plan_steps = plan.get("steps")
    recipe_steps = recipe.get("steps")
    if not isinstance(plan_steps, list) or not isinstance(recipe_steps, list):
        return
    if len(plan_steps) != len(recipe_steps):
        recorder.error(
            "execution_plan.recipe_steps",
            "execution-plan steps do not match the effective recipe step count",
        )
        return
    for index, (planned, recipe_step) in enumerate(zip(plan_steps, recipe_steps)):
        if not isinstance(planned, Mapping) or not isinstance(recipe_step, Mapping):
            recorder.error(
                "execution_plan.recipe_steps",
                "execution-plan and effective recipe steps must be objects",
            )
            continue
        if (
            planned.get("step_id") != recipe_step.get("id")
            or planned.get("operation_id") != recipe_step.get("op")
        ):
            recorder.error(
                "execution_plan.recipe_steps",
                "execution-plan step %d does not bind the corresponding effective recipe step"
                % index,
            )
        else:
            recorder.pass_(
                "execution_plan.recipe_steps",
                "execution-plan step %s binds the corresponding effective recipe step"
                % planned.get("step_id"),
            )


def _check_plan_step_copy(
    raw_steps: Any,
    plan_steps: Mapping[str, JsonDict],
    operation_digests: Mapping[str, str],
    recorder: _CheckRecorder,
) -> None:
    if not isinstance(raw_steps, list):
        recorder.error(
            "execution_plan.manifest_steps_schema",
            "manifest.execution_plan.steps must be a list",
        )
        return
    copied: Dict[str, Mapping[str, Any]] = {}
    for index, step in enumerate(raw_steps):
        if not isinstance(step, Mapping):
            recorder.error(
                "execution_plan.manifest_step_schema",
                "manifest execution-plan step %d must be an object" % index,
            )
            continue
        _check_planned_step(step, operation_digests, recorder, "manifest execution-plan")
        step_id = step.get("step_id")
        if isinstance(step_id, str):
            if step_id in copied:
                recorder.error(
                    "execution_plan.manifest_step_id",
                    "manifest execution-plan has duplicate step_id %s" % step_id,
                )
            copied[step_id] = step
    if canonical_json_sha256(raw_steps) != canonical_json_sha256(list(plan_steps.values())):
        recorder.error(
            "execution_plan.manifest_steps_link",
            "manifest execution-plan steps differ from execution-plan.json",
        )
    else:
        recorder.pass_(
            "execution_plan.manifest_steps_link",
            "manifest execution-plan steps match execution-plan.json",
        )


def _check_runtime_step_bindings(
    raw_steps: Any,
    plan_steps: Mapping[str, JsonDict],
    operation_digests: Mapping[str, str],
    recorder: _CheckRecorder,
    *,
    scope: str,
    require_all: bool,
) -> None:
    if not isinstance(raw_steps, list):
        recorder.error(
            "execution_plan.%s_steps_schema" % scope,
            "%s steps must be a list" % scope,
        )
        return
    seen = set()
    for index, record in enumerate(raw_steps):
        if not isinstance(record, Mapping):
            continue
        step_id = record.get("id")
        if not isinstance(step_id, str) or step_id not in plan_steps:
            recorder.error(
                "execution_plan.%s_step_id" % scope,
                "%s step %d does not correspond to an execution-plan step" % (scope, index),
            )
            continue
        if step_id in seen:
            recorder.error(
                "execution_plan.%s_step_id" % scope,
                "%s has duplicate runtime step %s" % (scope, step_id),
            )
            continue
        seen.add(step_id)
        binding = record.get("execution_binding")
        if not isinstance(binding, Mapping):
            recorder.error(
                "execution_plan.%s_binding" % scope,
                "%s step %s is missing execution_binding" % (scope, step_id),
            )
            continue
        _check_planned_step(binding, operation_digests, recorder, "%s runtime" % scope)
        if canonical_json_sha256(binding) != canonical_json_sha256(plan_steps[step_id]):
            recorder.error(
                "execution_plan.%s_binding_link" % scope,
                "%s step %s binding differs from execution-plan.json" % (scope, step_id),
            )
        else:
            recorder.pass_(
                "execution_plan.%s_binding_link" % scope,
                "%s step %s binding matches execution-plan.json" % (scope, step_id),
            )
        if record.get("op") != plan_steps[step_id].get("operation_id"):
            recorder.error(
                "execution_plan.%s_operation" % scope,
                "%s step %s operation differs from its planned binding" % (scope, step_id),
            )
    missing = sorted(set(plan_steps).difference(seen))
    if missing and require_all:
        recorder.error(
            "execution_plan.%s_bindings_complete" % scope,
            "%s is missing completed execution bindings: %s" % (scope, ", ".join(missing)),
        )
    elif missing:
        recorder.warning(
            "execution_plan.%s_bindings_partial" % scope,
            "%s has no runtime binding for unexecuted plan steps: %s" % (scope, ", ".join(missing)),
        )
    else:
        recorder.pass_(
            "execution_plan.%s_bindings_complete" % scope,
            "%s records every planned step binding" % scope,
        )


def _check_runtime_backend_namespaces(
    raw_steps: Any,
    plan_steps: Mapping[str, JsonDict],
    recorder: _CheckRecorder,
    *,
    require_all: bool,
) -> None:
    """Keep operation backends distinct from low-level data-plane engines."""

    if not isinstance(raw_steps, list):
        return
    wireless_operation_ids = {
        "wireless.channel",
        "wireless.digital_link",
        "wireless.pilot_observation",
        "source.ai_phy_pilot_channel",
        "wireless.miso_ofdm_csi",
    }
    data_plane_required_operation_ids = {
        "wireless.channel",
        "wireless.digital_link",
    }
    runtime = {
        str(row.get("id") or ""): row
        for row in raw_steps
        if isinstance(row, Mapping)
    }
    for step_id, plan_step in plan_steps.items():
        operation_id = str(plan_step.get("operation_id") or "")
        if operation_id not in wireless_operation_ids:
            continue
        record = runtime.get(step_id)
        if not isinstance(record, Mapping):
            if require_all:
                recorder.error(
                    "execution_plan.runtime_backend",
                    "wireless step %s has no runtime backend evidence" % step_id,
                )
            continue
        evidence_rows: List[Mapping[str, Any]] = []
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            evidence_rows.append(metadata)
        outputs = record.get("outputs")
        if isinstance(outputs, Mapping):
            for output in outputs.values():
                if isinstance(output, Mapping) and isinstance(
                    output.get("metadata"), Mapping
                ):
                    evidence_rows.append(output["metadata"])
        wireless_values = {
            str(row.get("wireless_backend") or "")
            for row in evidence_rows
        }
        data_plane_values = {
            str(row.get("data_plane_backend") or "")
            for row in evidence_rows
        }
        planned_backend = str(plan_step.get("backend") or "")
        if (
            not evidence_rows
            or wireless_values != {planned_backend}
            or planned_backend not in {"numpy", "sionna"}
        ):
            recorder.error(
                "execution_plan.runtime_wireless_backend",
                "wireless step %s runtime backend does not match its planned "
                "wireless materialization" % step_id,
                planned=planned_backend,
                runtime=sorted(wireless_values),
            )
            continue
        if operation_id not in data_plane_required_operation_ids:
            recorder.pass_(
                "execution_plan.runtime_wireless_backend",
                "wireless step %s runtime backend matches the plan" % step_id,
            )
            continue
        if (
            "" in data_plane_values
            or data_plane_values.intersection({"numpy", "sionna"})
            or len(data_plane_values) != 1
        ):
            recorder.error(
                "execution_plan.runtime_data_plane_backend",
                "wireless step %s data-plane backend is missing, inconsistent, "
                "or uses the wireless-backend namespace" % step_id,
                runtime=sorted(data_plane_values),
            )
            continue
        recorder.pass_(
            "execution_plan.runtime_wireless_backend",
            "wireless step %s runtime wireless/data-plane backends are "
            "namespaced and match the plan" % step_id,
        )


def _check_planned_step(
    step: Mapping[str, Any],
    operation_digests: Mapping[str, str],
    recorder: _CheckRecorder,
    scope: str,
    *,
    operation_contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    scope_id = _check_key(scope)
    _check_exact_version(
        step.get("schema_version"),
        PLANNED_STEP_SCHEMA_VERSION,
        recorder,
        "execution_plan.%s_step_schema_version" % scope_id,
        "%s planned-step schema version" % scope,
    )
    required_strings = (
        "step_id",
        "operation_id",
        "runner",
        "backend",
        "implementation",
        "materialization_id",
        "implementation_identity",
    )
    for key in required_strings:
        if not isinstance(step.get(key), str) or not step.get(key):
            recorder.error(
                "execution_plan.%s_step_%s" % (scope_id, key),
                "%s planned step has missing or invalid %s" % (scope, key),
            )
    if not isinstance(step.get("implementation_metadata"), Mapping):
        recorder.error(
            "execution_plan.%s_step_implementation_metadata" % scope_id,
            "%s planned step implementation_metadata must be an object" % scope,
        )

    declared_binding_sha = step.get("binding_sha256")
    computed_binding_sha = _canonical_digest_without(step, "binding_sha256")
    _check_digest_reference(
        declared_binding_sha,
        computed_binding_sha,
        recorder,
        "execution_plan.%s_binding_sha256" % scope_id,
        "%s planned-step binding SHA" % scope,
    )
    operation_id = step.get("operation_id")
    expected_contract_sha = operation_digests.get(operation_id) if isinstance(operation_id, str) else None
    if expected_contract_sha is None:
        recorder.error(
            "execution_plan.%s_operation_contract" % scope_id,
            "%s planned step references an absent operation contract" % scope,
            operation_id=operation_id,
        )
    else:
        _check_digest_reference(
            step.get("operation_contract_sha256"),
            expected_contract_sha,
            recorder,
            "execution_plan.%s_operation_contract_sha256" % scope_id,
            "%s planned-step operation-contract SHA" % scope,
        )
    if operation_contracts is not None:
        _check_planned_step_materialization(
            step,
            operation_contracts,
            recorder,
            scope,
        )


def _check_planned_step_materialization(
    step: Mapping[str, Any],
    operation_contracts: Mapping[str, Mapping[str, Any]],
    recorder: _CheckRecorder,
    scope: str,
) -> None:
    scope_id = _check_key(scope)
    operation_id = step.get("operation_id")
    runner = step.get("runner")
    backend = step.get("backend")
    implementation = step.get("implementation")
    identity_fields = (operation_id, runner, backend, implementation)
    if not all(isinstance(value, str) and value for value in identity_fields):
        return

    expected_identity = "%s@%s/%s/%s" % identity_fields
    if step.get("materialization_id") != expected_identity:
        recorder.error(
            "execution_plan.%s_materialization_id" % scope_id,
            "%s planned-step materialization_id does not match its binding tuple"
            % scope,
            expected=expected_identity,
            actual=step.get("materialization_id"),
        )
    else:
        recorder.pass_(
            "execution_plan.%s_materialization_id" % scope_id,
            "%s planned-step materialization_id matches its binding tuple" % scope,
        )

    operation_contract = operation_contracts.get(operation_id)
    if not isinstance(operation_contract, Mapping):
        return
    raw_materializations = operation_contract.get("materializations")
    if not isinstance(raw_materializations, list):
        recorder.error(
            "execution_plan.%s_materialization_contract" % scope_id,
            "%s operation contract has no valid materializations list" % scope,
            operation_id=operation_id,
        )
        return
    matches = [
        item
        for item in raw_materializations
        if isinstance(item, Mapping)
        and item.get("runner") == runner
        and item.get("backend") == backend
        and item.get("implementation") == implementation
    ]
    if not matches:
        recorder.error(
            "execution_plan.%s_materialization_contract" % scope_id,
            "%s planned-step binding is absent from the embedded operation contract"
            % scope,
            operation_id=operation_id,
            runner=runner,
            backend=backend,
            implementation=implementation,
        )
        return
    if len(matches) > 1:
        recorder.error(
            "execution_plan.%s_materialization_contract" % scope_id,
            "%s embedded operation contract has an ambiguous materialization identity"
            % scope,
            operation_id=operation_id,
            runner=runner,
            backend=backend,
            implementation=implementation,
        )
        return

    declared = matches[0]
    if declared.get("status") != "implemented":
        recorder.error(
            "execution_plan.%s_materialization_status" % scope_id,
            "%s planned-step materialization is not declared implemented" % scope,
            actual=declared.get("status"),
        )
    else:
        recorder.pass_(
            "execution_plan.%s_materialization_status" % scope_id,
            "%s planned-step materialization is declared implemented" % scope,
        )

    metadata = step.get("implementation_metadata")
    selection = metadata.get("selection") if isinstance(metadata, Mapping) else None
    metadata_mismatches = []
    if isinstance(metadata, Mapping):
        if metadata.get("status") != declared.get("status"):
            metadata_mismatches.append("status")
        if metadata.get("operation_class") != step.get("implementation_identity"):
            metadata_mismatches.append("operation_class")
        if metadata.get("operation_name") != operation_contract.get("name"):
            metadata_mismatches.append("operation_name")
    if not isinstance(selection, Mapping):
        metadata_mismatches.append("selection")
    else:
        if selection.get("resolved_backend") != backend:
            metadata_mismatches.append("selection.resolved_backend")
        if declared.get("status") == "implemented":
            implemented_for_runner = [
                item
                for item in raw_materializations
                if isinstance(item, Mapping)
                and item.get("runner") == runner
                and item.get("status") == "implemented"
            ]
            expected_candidate_index = implemented_for_runner.index(declared)
            candidate_index = selection.get("candidate_index")
            if (
                not isinstance(candidate_index, int)
                or isinstance(candidate_index, bool)
                or candidate_index != expected_candidate_index
            ):
                metadata_mismatches.append("selection.candidate_index")
    if metadata_mismatches:
        recorder.error(
            "execution_plan.%s_materialization_metadata" % scope_id,
            "%s planned materialization metadata is inconsistent with its binding"
            % scope,
            fields=metadata_mismatches,
        )
    else:
        recorder.pass_(
            "execution_plan.%s_materialization_metadata" % scope_id,
            "%s planned materialization metadata matches its binding" % scope,
        )

    declared_bindings = declared.get("parameter_bindings")
    if declared_bindings is None:
        declared_bindings = {}
    planned_bindings = (
        selection.get("parameter_bindings")
        if isinstance(selection, Mapping)
        else None
    )
    if planned_bindings is None:
        planned_bindings = {}
    if not isinstance(declared_bindings, Mapping):
        recorder.error(
            "execution_plan.%s_materialization_parameter_bindings" % scope_id,
            "%s embedded materialization parameter_bindings must be an object"
            % scope,
        )
        return
    if not isinstance(planned_bindings, Mapping):
        recorder.error(
            "execution_plan.%s_materialization_parameter_bindings" % scope_id,
            "%s planned materialization parameter_bindings must be an object" % scope,
        )
        return
    if canonical_json_sha256(declared_bindings) != canonical_json_sha256(
        planned_bindings
    ):
        recorder.error(
            "execution_plan.%s_materialization_parameter_bindings" % scope_id,
            "%s planned parameter bindings do not match the embedded materialization contract"
            % scope,
            expected=dict(declared_bindings),
            actual=dict(planned_bindings),
        )
        return

    overrides = (
        metadata.get("parameter_overrides")
        if isinstance(metadata, Mapping)
        else None
    )
    if declared_bindings and not isinstance(overrides, Mapping):
        recorder.error(
            "execution_plan.%s_materialization_parameter_overrides" % scope_id,
            "%s planned materialization does not apply its declared parameter bindings"
            % scope,
        )
        return
    if isinstance(overrides, Mapping):
        missing_or_changed = [
            name
            for name, value in declared_bindings.items()
            if name not in overrides
            or canonical_json_sha256({"value": overrides[name]})
            != canonical_json_sha256({"value": value})
        ]
        if missing_or_changed:
            recorder.error(
                "execution_plan.%s_materialization_parameter_overrides" % scope_id,
                "%s planned parameter overrides do not activate the declared materialization"
                % scope,
                parameters=sorted(str(name) for name in missing_or_changed),
            )
            return
    if declared.get("status") == "implemented" and not metadata_mismatches:
        recorder.pass_(
            "execution_plan.%s_materialization_contract" % scope_id,
            "%s planned-step binding matches an implemented embedded materialization"
            % scope,
        )


def _canonical_digest_without(payload: Mapping[str, Any], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return canonical_json_sha256(value)


def _check_digest_reference(
    value: Any,
    expected: Any,
    recorder: _CheckRecorder,
    check_id: str,
    label: str,
) -> None:
    if not _is_sha256(value):
        recorder.error(check_id, "%s is missing or malformed" % label)
    elif not _is_sha256(expected):
        recorder.error(check_id, "%s cannot be linked to a valid SHA-256" % label)
    elif value != expected:
        recorder.error(
            check_id,
            "%s does not match" % label,
            expected=expected,
            actual=value,
        )
    else:
        recorder.pass_(check_id, "%s matches" % label)


def _check_exact_version(
    value: Any,
    expected: int,
    recorder: _CheckRecorder,
    check_id: str,
    label: str,
) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value == expected:
        recorder.pass_(check_id, "%s is supported" % label)
    else:
        recorder.error(
            check_id,
            "%s is unsupported or malformed" % label,
            expected=expected,
            actual=value,
        )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _check_manifest_artifacts(run_dir: Path, manifest: JsonDict, recorder: _CheckRecorder) -> None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    if not artifacts:
        recorder.warning("artifacts.present", "manifest records no artifacts")
        return
    checked = 0
    for index, record in enumerate(artifacts):
        if not isinstance(record, Mapping):
            recorder.error("artifact.record", "manifest artifact %d is not an object" % index)
            continue
        label = _artifact_label(record, index)
        try:
            artifact_path = _resolve_artifact_path(run_dir, record)
        except ValueError as exc:
            recorder.error("artifact.location", "%s: %s" % (label, exc))
            continue
        if not artifact_path.is_file():
            recorder.error("artifact.exists", "artifact is missing: %s" % label, path=str(artifact_path))
            continue
        checked += 1
        recorded_hash = str(record.get("sha256") or "")
        if _is_sha256(recorded_hash):
            actual_hash = file_sha256(artifact_path)
            if actual_hash != recorded_hash:
                recorder.error(
                    "artifact.sha256",
                    "artifact hash mismatch: %s" % label,
                    expected=recorded_hash,
                    actual=actual_hash,
                )
            else:
                recorder.pass_("artifact.sha256", "artifact hash matches: %s" % label)
        else:
            recorder.error(
                "artifact.sha256_missing",
                "artifact SHA-256 is missing or malformed: %s" % label,
            )
        if not record.get("kind"):
            recorder.warning("artifact.kind_missing", "artifact kind is missing: %s" % label)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            recorder.error("artifact.metadata_missing", "artifact metadata is missing or not an object: %s" % label)
            continue
        try:
            metadata_sha256 = canonical_json_sha256(dict(metadata))
        except (TypeError, ValueError) as exc:
            recorder.error(
                "artifact.metadata",
                "artifact metadata is not canonical JSON: %s (%s)" % (label, exc),
            )
            continue
        if record.get("metadata_sha256") != metadata_sha256:
            recorder.error(
                "artifact.metadata_sha256",
                "artifact metadata digest is missing or mismatched: %s" % label,
            )
        if not _is_sha256(record.get("producer_metrics_sha256")):
            recorder.error(
                "artifact.producer_metrics_sha256",
                "artifact producer metric binding is missing or malformed: %s"
                % label,
            )
        for field_name in ("dtype", "shape", "array"):
            if record.get(field_name) != metadata.get(field_name):
                recorder.error(
                    "artifact.metadata_projection",
                    "artifact %s projection differs from metadata: %s"
                    % (field_name, label),
                )
    if checked:
        recorder.pass_("artifacts.present", "%d manifest artifacts exist" % checked)


def _check_run_status(summary: JsonDict, manifest: Optional[JsonDict], recorder: _CheckRecorder) -> None:
    status = str(summary.get("status") or "").lower()
    if status == "completed":
        recorder.pass_("status.completed", "run completed")
    elif status in _TERMINAL_BAD_STATUSES or not status:
        recorder.error("status.comparable", "run status is %s; bundle is non-comparable" % (status or "missing"))
    else:
        recorder.error("status.unknown", "run status is not recognized: %s" % status)
    if status == "completed" and not summary.get("completed_at_utc"):
        recorder.warning("status.completed_time", "completed run is missing completed_at_utc")
    if manifest is not None and status == "completed" and not manifest.get("completed_at_utc"):
        recorder.warning("status.manifest_completed_time", "completed manifest is missing completed_at_utc")


def _check_metric_plausibility(metrics: Mapping[str, Any], recorder: _CheckRecorder, *, prefix: str) -> None:
    numeric_count = 0
    invalid_count = 0
    for name, value in metrics.items():
        if not _is_json_number(value):
            invalid_count += 1
            recorder.error(
                "%s.type" % prefix,
                "metric must be a JSON number: %s (got %s)"
                % (name, type(value).__name__),
            )
            continue
        number = _as_number(value)
        if number is None:
            invalid_count += 1
            recorder.error(
                "%s.finite" % prefix,
                "numeric metric is not finite: %s" % name,
            )
            continue
        numeric_count += 1
        if _expects_nonnegative(str(name)) and number < 0:
            invalid_count += 1
            recorder.error("%s.nonnegative" % prefix, "metric must be non-negative: %s" % name)
        if _expects_unit_interval(str(name)) and not 0.0 <= number <= 1.0:
            invalid_count += 1
            recorder.error(
                "%s.unit_interval" % prefix,
                "metric must be in [0, 1]: %s" % name,
            )
    if numeric_count and not invalid_count:
        recorder.pass_("%s.plausibility" % prefix, "%d numeric metrics are finite/plausible" % numeric_count)
    elif not metrics:
        recorder.warning("%s.present" % prefix, "no numeric metrics found")


def _check_rate_accounting(summary: JsonDict, metrics: Mapping[str, Any], recorder: _CheckRecorder) -> None:
    explicit_boundaries = (
        (
            "native_codec",
            ("codec.native_bit_count",),
            ("rate.native_codec_bpp",),
        ),
        (
            "serialized_payload",
            ("codec.serialized_payload_bit_count", "channel.payload_bit_count"),
            ("rate.serialized_payload_bpp", "rate.payload_bpp"),
        ),
        (
            "framed",
            ("channel.framed_bit_count",),
            ("rate.framed_bpp",),
        ),
        (
            "coded",
            ("channel.coded_bit_count",),
            ("rate.coded_bpp",),
        ),
        (
            "transmitted_padded",
            ("channel.transmitted_bit_count",),
            ("rate.padded_bpp",),
        ),
    )
    accounting_keys = tuple(
        dict.fromkeys(
            _TX_BIT_KEYS
            + _PIXEL_KEYS
            + _BPP_KEYS
            + tuple(
                key
                for _, bit_keys, bpp_keys in explicit_boundaries
                for key in bit_keys + bpp_keys
            )
        )
    )
    invalid_claims = [
        str(name)
        for name, value in metrics.items()
        if (
            str(name) in accounting_keys
            or any(str(name).endswith("." + key) for key in accounting_keys)
            or str(name).endswith(".rate_bpp")
            or str(name).endswith(".bpp")
        )
        and _as_number(value) is None
    ]
    if invalid_claims:
        recorder.error(
            "accounting.metric_values",
            "present rate-accounting metrics must be finite JSON numbers: %s"
            % ", ".join(sorted(invalid_claims)),
        )
        return
    tx_bits = _first_metric(metrics, _TX_BIT_KEYS)
    image_rate_run = _is_image_rate_run(summary)
    source_pixels = _first_metric(metrics, _PIXEL_KEYS)
    if source_pixels is None and image_rate_run:
        source_pixels = _source_pixels_from_artifacts(summary)
    checked_explicit = 0
    if source_pixels is not None and source_pixels > 0:
        for label, bit_keys, bpp_keys in explicit_boundaries:
            bit_count = _first_metric(metrics, bit_keys)
            boundary_bpp = _first_metric(metrics, bpp_keys)
            if bit_count is None or boundary_bpp is None:
                continue
            checked_explicit += 1
            recomputed_boundary = float(bit_count) / float(source_pixels)
            tolerance = max(1e-6, abs(float(boundary_bpp)) * 1e-4)
            if abs(float(boundary_bpp) - recomputed_boundary) > tolerance:
                recorder.error(
                    "accounting.%s_bpp" % label,
                    "%s bpp does not match its named bit boundary / source pixels"
                    % label.replace("_", " "),
                    recorded=boundary_bpp,
                    recomputed=recomputed_boundary,
                    bit_count=bit_count,
                    source_pixels=source_pixels,
                )
            else:
                recorder.pass_(
                    "accounting.%s_bpp" % label,
                    "%s bpp matches its named bit boundary / source pixels"
                    % label.replace("_", " "),
                )
    if checked_explicit and _first_metric(metrics, _BPP_KEYS) is None:
        # Do not collapse a correctly named multi-boundary accounting chain
        # back to the legacy ambiguous bpp heuristic below. If a legacy
        # ``rate_bpp`` is also present, however, it remains a claimed
        # transmitted-rate coordinate and must still agree with the recipe's
        # fixed transmitted-bit boundary; named metrics may not mask a forged
        # legacy value.
        return
    recorded_bpp = _first_metric(metrics, _BPP_KEYS)
    if recorded_bpp is None:
        for key, value in metrics.items():
            if str(key).endswith(".rate_bpp") or str(key).endswith(".bpp"):
                recorded_bpp = _as_number(value)
                break
    if (
        tx_bits is None
        and source_pixels is None
        and recorded_bpp is None
        and not image_rate_run
    ):
        recorder.pass_(
            "accounting.rate_bpp",
            "image bpp accounting is not applicable to this task-direct run",
        )
        return
    if source_pixels is None and tx_bits is not None:
        recorder.pass_(
            "accounting.rate_bpp",
            "image bpp accounting is not applicable to this non-image bit transport run",
        )
        return
    if tx_bits is None or source_pixels is None or source_pixels <= 0:
        recorder.warning("accounting.rate_bpp", "not enough information to recompute bpp")
        return
    recomputed = float(tx_bits) / float(source_pixels)
    if recorded_bpp is None:
        recorder.pass_("accounting.rate_bpp", "bpp can be recomputed from transmitted bits and source pixels", bpp=recomputed)
        return
    tolerance = max(1e-6, abs(float(recorded_bpp)) * 1e-4)
    if abs(float(recorded_bpp) - recomputed) > tolerance:
        recorder.error(
            "accounting.rate_bpp",
            "recorded bpp does not match transmitted bits / source pixels",
            recorded=recorded_bpp,
            recomputed=recomputed,
            tx_bits=tx_bits,
            source_pixels=source_pixels,
        )
    else:
        recorder.pass_("accounting.rate_bpp", "recorded bpp matches transmitted bits / source pixels")


def _is_image_rate_run(summary: Mapping[str, Any]) -> bool:
    recipe = summary.get("recipe")
    if not isinstance(recipe, Mapping):
        return False
    metadata = recipe.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    research = metadata.get("research")
    research = research if isinstance(research, Mapping) else {}
    research_task = research.get("task")
    research_task = research_task if isinstance(research_task, Mapping) else {}
    task_id = str(research_task.get("id") or metadata.get("task_id") or "").strip()
    if task_id == "image_reconstruction":
        return True
    for step in recipe.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        operation_id = str(step.get("op") or "")
        if operation_id == "source.image_dataset" or operation_id.startswith("metrics.image_reconstruction"):
            return True
    return False


def _check_benchmark_schema(result: JsonDict, recorder: _CheckRecorder) -> None:
    _require_type(result, "schema_version", int, recorder, "benchmark")
    _require_type(result, "kind", str, recorder, "benchmark")
    _require_type(result, "benchmark", dict, recorder, "benchmark")
    _require_type(result, "status", str, recorder, "benchmark")
    _require_type(result, "recipes", list, recorder, "benchmark")
    _check_exact_version(
        result.get("schema_version"),
        _BENCHMARK_RESULT_SCHEMA_VERSION,
        recorder,
        "schema.benchmark.schema_version_supported",
        "benchmark-result schema version",
    )
    if result.get("kind") != _BENCHMARK_RESULT_KIND:
        recorder.error(
            "schema.benchmark.kind_value",
            "benchmark-result kind is missing or unsupported",
            expected=_BENCHMARK_RESULT_KIND,
            actual=result.get("kind"),
        )
    else:
        recorder.pass_(
            "schema.benchmark.kind_value",
            "benchmark-result kind is supported",
        )
    benchmark = dict(result.get("benchmark") or {})
    if not benchmark.get("id"):
        recorder.error("benchmark.id", "benchmark id is missing")
    if not benchmark.get("version"):
        recorder.error("benchmark.version", "benchmark version is missing")
    if not isinstance(benchmark.get("metrics") or [], list):
        recorder.error("benchmark.metrics", "benchmark metrics must be a list")
    recipes = result.get("recipes")
    if not isinstance(recipes, list):
        return
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, Mapping):
            recorder.error(
                "schema.benchmark.recipe",
                "benchmark recipe entry %d must be an object" % index,
            )
            continue
        if not isinstance(recipe.get("metrics"), Mapping):
            recorder.error(
                "schema.benchmark.recipe_metrics",
                "benchmark recipe %s metrics must be an object"
                % (recipe.get("id") or recipe.get("label") or index),
            )


def _check_benchmark_attempt_ledger(
    store: LocalStore,
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    """Cross-bind a sealed result to its append-only terminal attempt event."""

    attempt_id = result.get("attempt_id")
    descriptor = result.get("attempt_ledger")
    if not isinstance(attempt_id, str) or not attempt_id:
        recorder.error(
            "benchmark.attempt_ledger",
            "benchmark result has no attempt_id",
        )
        return
    if not isinstance(descriptor, Mapping):
        recorder.error(
            "benchmark.attempt_ledger",
            "benchmark result has no attempt-ledger descriptor",
        )
        return
    if (
        descriptor.get("kind") != "noema.benchmark_attempt_ledger"
        or descriptor.get("relative_path") != ".attempt-ledger"
        or not _is_sha256(descriptor.get("start_event_sha256"))
    ):
        recorder.error(
            "benchmark.attempt_ledger",
            "benchmark attempt-ledger descriptor is malformed",
        )
        return

    snapshot = BenchmarkAttemptLedger(store.benchmarks_dir).verify()
    if snapshot.get("status") != "valid":
        recorder.error(
            "benchmark.attempt_ledger",
            "benchmark attempt ledger is invalid: %s"
            % "; ".join(snapshot.get("errors") or []),
        )
        return
    attempts = [
        row
        for row in snapshot.get("attempts") or []
        if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id
    ]
    if len(attempts) != 1:
        recorder.error(
            "benchmark.attempt_ledger",
            "benchmark attempt_id is absent or duplicated in the ledger",
        )
        return
    attempt = attempts[0]
    if attempt.get("start_event_sha256") != descriptor.get("start_event_sha256"):
        recorder.error(
            "benchmark.attempt_ledger.start",
            "result start-event identity differs from the attempt ledger",
        )
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    if (
        attempt.get("benchmark_id") != benchmark.get("id")
        or str(attempt.get("benchmark_version") or "")
        != str(benchmark.get("version") or "")
        or attempt.get("protocol_sha256") != benchmark.get("sha256")
    ):
        recorder.error(
            "benchmark.attempt_ledger.protocol",
            "attempt-ledger protocol identity differs from result.json",
        )

    terminal_status = str(attempt.get("terminal_status") or "")
    expected_terminal = str(
        descriptor.get("expected_terminal_status") or ""
    )
    if (
        not terminal_status
        or not _is_sha256(attempt.get("final_event_sha256"))
        or terminal_status != expected_terminal
    ):
        recorder.error(
            "benchmark.attempt_ledger.terminal",
            "result terminal-attempt expectation differs from the ledger",
            expected=expected_terminal,
            actual=terminal_status,
        )
    result_identity = attempt.get("result")
    result_path = result_dir / "result.json"
    actual_identity = {
        "result_id": result_dir.name,
        "result_json_sha256": file_sha256(result_path),
        "result_json_size_bytes": int(result_path.stat().st_size),
    }
    if not isinstance(result_identity, Mapping) or dict(result_identity) != actual_identity:
        recorder.error(
            "benchmark.attempt_ledger.result_identity",
            "live result.json differs from the immutable attempt-ledger identity",
            expected=(
                dict(result_identity)
                if isinstance(result_identity, Mapping)
                else result_identity
            ),
            actual=actual_identity,
        )
    else:
        recorder.pass_(
            "benchmark.attempt_ledger.result_identity",
            "live result.json matches the terminal attempt-ledger identity",
        )
    expected_outcomes = [
        {
            "recipe_id": str(row.get("id") or ""),
            "status": str(row.get("status") or ""),
        }
        for row in result.get("recipes") or []
        if isinstance(row, Mapping)
    ]
    if attempt.get("recipe_outcomes") != expected_outcomes:
        recorder.error(
            "benchmark.attempt_ledger.outcomes",
            "attempt-ledger recipe outcomes differ from result.json",
        )
    else:
        recorder.pass_(
            "benchmark.attempt_ledger.outcomes",
            "attempt-ledger recipe outcomes match result.json",
        )


def _check_benchmark_status(result: JsonDict, recorder: _CheckRecorder) -> None:
    status = str(result.get("status") or "").lower()
    if status == "completed":
        recorder.pass_("benchmark.status", "benchmark result completed")
    else:
        recorder.error("benchmark.status", "benchmark result status is %s; result is non-comparable" % (status or "missing"))


def _check_benchmark_required_metrics(
    result: JsonDict,
    recorder: _CheckRecorder,
    *,
    validated_run_evidence: Optional[Mapping[str, Any]] = None,
) -> None:
    benchmark = dict(result.get("benchmark") or {})
    all_definitions: Dict[str, JsonDict] = {}
    for index, metric in enumerate(benchmark.get("metrics") or []):
        if not isinstance(metric, Mapping) or not metric.get("id"):
            continue
        definition = dict(metric)
        raw_roles = definition.get("applicable_roles")
        if raw_roles is not None and (
            not isinstance(raw_roles, list)
            or not raw_roles
            or not all(isinstance(role, str) and role.strip() for role in raw_roles)
            or len({role.strip() for role in raw_roles}) != len(raw_roles)
        ):
            recorder.error(
                "benchmark.metric_applicable_roles",
                "benchmark metric %s has invalid applicable_roles"
                % (definition.get("id") or index),
            )
            continue
        all_definitions[str(definition["id"])] = definition
    if not all_definitions:
        recorder.warning("benchmark.required_metrics", "benchmark declares no required metrics")
        return
    evidence_by_index: Dict[int, Mapping[str, Any]] = {}
    if isinstance(validated_run_evidence, Mapping):
        raw_entries = validated_run_evidence.get("entries")
        if isinstance(raw_entries, list):
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, Mapping):
                    continue
                entry_index = raw_entry.get("entry_index")
                if isinstance(entry_index, int) and not isinstance(entry_index, bool):
                    evidence_by_index[entry_index] = raw_entry
    recipes = result.get("recipes") if isinstance(result.get("recipes"), list) else []
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, Mapping):
            recorder.error("benchmark.recipe", "benchmark recipe entry %d is not an object" % index)
            continue
        label = str(recipe.get("id") or recipe.get("label") or index)
        raw_metrics = recipe.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            recorder.error(
                "benchmark.required_metrics",
                "benchmark recipe %s metrics are missing or not an object" % label,
            )
            continue
        metrics = dict(raw_metrics)
        role = str(recipe.get("role") or "candidate").strip()
        required_definitions = {
            metric_id: definition
            for metric_id, definition in all_definitions.items()
            if definition.get("applicable_roles") is None
            or role in definition.get("applicable_roles")
        }
        missing = [metric for metric in required_definitions if metric not in metrics]
        if missing:
            recorder.error(
                "benchmark.required_metrics",
                "benchmark recipe %s is missing required metrics: %s" % (label, ", ".join(missing)),
                recipe=label,
                missing=missing,
            )
        else:
            recorder.pass_("benchmark.required_metrics", "benchmark recipe %s has required metrics" % label)
        if str(recipe.get("status") or "").lower() not in {
            "completed",
            "rejected_resource_budget",
        }:
            continue
        raw_provenance = recipe.get("metric_provenance")
        if not isinstance(raw_provenance, Mapping):
            recorder.error(
                "benchmark.metric_provenance",
                "benchmark recipe %s is missing its metric producer ledger" % label,
                recipe=label,
            )
            continue
        run_evidence = evidence_by_index.get(index)
        if not isinstance(run_evidence, Mapping):
            recorder.error(
                "benchmark.metric_provenance",
                "benchmark recipe %s has no validated result-local run evidence for producer verification"
                % label,
                recipe=label,
            )
        for metric_id, definition in required_definitions.items():
            if metric_id not in metrics:
                continue
            entry = raw_provenance.get(metric_id)
            if not isinstance(entry, Mapping):
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s has no authoritative producer"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            source_step = str(entry.get("source_step") or "").strip()
            source_operation = str(entry.get("source_operation") or "").strip()
            qualified_metric = str(entry.get("qualified_metric") or "").strip()
            expected_qualified = (
                "steps.%s.%s" % (source_step, metric_id) if source_step else ""
            )
            if entry.get("source_scope") != "step" or not source_step:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s is not bound to an evaluator step"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            declared_step = str(
                definition.get("source_step")
                or definition.get("producer_step")
                or ""
            ).strip()
            declared_operation = str(definition.get("source_operation") or "").strip()
            declared_operations = {
                str(value).strip()
                for value in definition.get("source_operations") or []
                if str(value).strip()
            }
            if declared_step and source_step != declared_step:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s came from %s, not declared step %s"
                    % (label, metric_id, source_step, declared_step),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if declared_operation and source_operation != declared_operation:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s came from operation %s, not %s"
                    % (label, metric_id, source_operation or "missing", declared_operation),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if declared_operations and source_operation not in declared_operations:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s came from operation %s, not the declared set %s"
                    % (
                        label,
                        metric_id,
                        source_operation or "missing",
                        ", ".join(sorted(declared_operations)),
                    ),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if qualified_metric != expected_qualified or qualified_metric not in metrics:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s has a missing or inconsistent qualified value"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if canonical_json_sha256({"value": metrics[qualified_metric]}) != canonical_json_sha256(
                {"value": metrics[metric_id]}
            ):
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s differs from its authoritative step value"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            definition_version = definition.get("definition_version")
            if (
                isinstance(definition_version, bool)
                or not isinstance(definition_version, int)
                or definition_version < 1
            ):
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s has an unversioned protocol definition"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if entry.get("definition_version") != definition_version:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s definition version disagrees with its protocol"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if entry.get("definition_sha256") != canonical_json_sha256(definition):
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s is not bound to its protocol definition"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            if not isinstance(run_evidence, Mapping):
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s cannot verify producer identity without result-local run evidence"
                    % (label, metric_id),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            source_error = _check_metric_producer_against_run_evidence(
                entry,
                run_evidence,
                source_step=source_step,
                source_operation=source_operation,
                metric_id=metric_id,
            )
            if source_error:
                recorder.error(
                    "benchmark.metric_provenance",
                    "benchmark recipe %s metric %s %s"
                    % (label, metric_id, source_error),
                    recipe=label,
                    metric=metric_id,
                )
                continue
            recorder.pass_(
                "benchmark.metric_provenance",
                "benchmark recipe %s metric %s is bound to step %s"
                % (label, metric_id, source_step),
            )


def _check_metric_producer_against_run_evidence(
    entry: Mapping[str, Any],
    run_evidence: Mapping[str, Any],
    *,
    source_step: str,
    source_operation: str,
    metric_id: str,
) -> str:
    """Return an error suffix when a producer ledger differs from frozen run evidence."""

    summary = run_evidence.get("summary")
    if not isinstance(summary, Mapping):
        return "has no validated summary payload"
    raw_steps = summary.get("steps")
    if not isinstance(raw_steps, list):
        return "validated summary has no step list"
    matches = [
        step
        for step in raw_steps
        if isinstance(step, Mapping) and str(step.get("id") or "") == source_step
    ]
    if len(matches) != 1:
        return "source step is absent or duplicated in validated run evidence"
    step = matches[0]
    if str(step.get("op") or "") != source_operation:
        return "source operation disagrees with validated run evidence"
    step_metrics = step.get("metrics")
    if not isinstance(step_metrics, Mapping) or metric_id not in step_metrics:
        return "source metric is absent from the validated evaluator step"
    outputs = step.get("outputs")
    binding = step.get("execution_binding")
    if not isinstance(outputs, Mapping) or not isinstance(binding, Mapping):
        return "validated evaluator step lacks outputs or an execution binding"
    implementation_metadata = binding.get("implementation_metadata")
    if not isinstance(implementation_metadata, Mapping):
        return "validated execution binding lacks implementation metadata"
    implementation_sha256 = str(
        implementation_metadata.get("source_sha256") or ""
    ).strip().lower()
    if len(implementation_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in implementation_sha256
    ):
        return "validated execution binding lacks a content-identified implementation"
    descriptor = run_evidence.get("descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    expected = {
        "source_implementation_sha256": implementation_sha256,
        "source_outputs_sha256": canonical_json_sha256(dict(outputs)),
        "source_execution_binding_sha256": canonical_json_sha256(dict(binding)),
        "source_run_evidence_sha256": descriptor.get("files_sha256"),
    }
    for field, expected_value in expected.items():
        if entry.get(field) != expected_value:
            return "%s disagrees with validated run evidence" % field
    return ""


def _check_benchmark_resource_admission(
    result: JsonDict,
    benchmark_json: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    derived_metric_keys = {
        "admitted": "benchmark.resource_budget.admitted",
        "observed": "benchmark.resource_budget.observed",
        "maximum": "benchmark.resource_budget.maximum",
        "tolerance": "benchmark.resource_budget.tolerance",
        "excess": "benchmark.resource_budget.excess",
    }
    recipes = result.get("recipes") if isinstance(result.get("recipes"), list) else []
    protocol_sha256 = str((result.get("benchmark") or {}).get("sha256") or "")
    checked = 0
    saw_declaration = False
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, Mapping):
            continue
        label = str(recipe.get("id") or recipe.get("label") or index)
        status = str(recipe.get("status") or "").lower()
        admission = recipe.get("resource_admission")
        try:
            declaration = _benchmark_resource_budget_declaration(
                benchmark_json, str(recipe.get("id") or "")
            )
        except ValueError as exc:
            recorder.error(
                "benchmark.resource_admission.protocol",
                "benchmark recipe %s has an invalid frozen resource declaration: %s"
                % (label, exc),
            )
            continue
        saw_declaration = saw_declaration or declaration is not None
        if admission is None:
            if declaration is not None or status == "rejected_resource_budget":
                recorder.error(
                    "benchmark.resource_admission",
                    "benchmark recipe %s is missing admission evidence required by the frozen protocol"
                    % label,
                )
            elif isinstance(recipe.get("metrics"), Mapping) and set(
                derived_metric_keys.values()
            ).intersection(recipe["metrics"]):
                recorder.error(
                    "benchmark.resource_admission",
                    "benchmark recipe %s has admission metrics without a frozen admission record"
                    % label,
                )
            continue
        if declaration is None:
            recorder.error(
                "benchmark.resource_admission.protocol",
                "benchmark recipe %s declares admission evidence but the frozen protocol has no budget"
                % label,
            )
            continue
        checked += 1
        required_fields = {
            "admitted",
            "decision",
            "metric",
            "observed",
            "maximum",
            "tolerance",
            "excess",
            "policy",
            "unit",
            "protocol_sha256",
        }
        conversion_declared = declaration.get("conversion") is not None
        if conversion_declared:
            required_fields.add("unit_contract")
        if not isinstance(admission, Mapping) or set(admission) != required_fields:
            recorder.error(
                "benchmark.resource_admission.schema",
                "benchmark recipe %s resource admission is malformed" % label,
            )
            continue
        admitted = admission.get("admitted")
        numbers = {
            name: _as_number(admission.get(name))
            for name in ("observed", "maximum", "tolerance", "excess")
        }
        if not isinstance(admitted, bool) or any(
            value is None or not math.isfinite(value) or value < 0.0
            for value in numbers.values()
        ):
            recorder.error(
                "benchmark.resource_admission.values",
                "benchmark recipe %s resource admission values are invalid" % label,
            )
            continue
        observed = float(numbers["observed"])
        maximum = float(numbers["maximum"])
        tolerance = float(numbers["tolerance"])
        expected_admitted = observed <= maximum + tolerance
        expected_excess = max(0.0, observed - maximum)
        expected_decision = (
            "admitted" if expected_admitted else "rejected_resource_budget"
        )
        expected_status = "completed" if expected_admitted else "rejected_resource_budget"
        if (
            admitted != expected_admitted
            or admission.get("decision") != expected_decision
            or status != expected_status
            or admission.get("policy") != "reject"
            or not str(admission.get("metric") or "").strip()
            or not str(admission.get("unit") or "").strip()
            or abs(float(numbers["excess"]) - expected_excess) > 1e-12
            or admission.get("metric") != declaration["metric"]
            or admission.get("unit") != declaration["unit"]
            or maximum != declaration["maximum"]
            or tolerance != declaration["tolerance"]
            or admission.get("protocol_sha256") != protocol_sha256
            or (
                benchmark_json is not None
                and not benchmark_protocol_sha256_matches(
                    benchmark_json,
                    protocol_sha256,
                )
            )
        ):
            recorder.error(
                "benchmark.resource_admission.decision",
                "benchmark recipe %s resource admission decision is inconsistent"
                % label,
                expected_status=expected_status,
                actual_status=status,
            )
            continue
        raw_metrics = recipe.get("metrics")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
        measured_metric = str(declaration["metric"])
        if measured_metric not in metrics:
            recorder.error(
                "benchmark.resource_admission.observed",
                "benchmark recipe %s is missing the snapshotted admission metric"
                % label,
                metric=measured_metric,
                observed=observed,
            )
            continue
        measured = _as_number(metrics[measured_metric])
        if measured is None:
            recorder.error(
                "benchmark.resource_admission.observed",
                "benchmark recipe %s snapshotted admission metric is not a finite JSON number"
                % label,
                metric=measured_metric,
                observed=observed,
                actual_type=type(metrics[measured_metric]).__name__,
            )
            continue
        if conversion_declared:
            try:
                conversion = _verification_resource_conversion(
                    declaration.get("conversion"),
                    metric_unit=str(declaration["metric_unit"]),
                    budget_unit=str(declaration["unit"]),
                )
                expected_unit_contract = evaluate_typed_resource_admission(
                    observed=ResourceQuantity(
                        measured,
                        str(declaration["metric_unit"]),
                    ),
                    budget=ResourceQuantity(
                        maximum,
                        str(declaration["unit"]),
                    ),
                    tolerance=ResourceQuantity(
                        tolerance,
                        str(declaration["unit"]),
                    ),
                    aggregation_policy=str(
                        declaration["aggregation_policy"]
                    ),
                    conversion=conversion,
                )
            except (ResourceUnitError, ValueError, KeyError) as exc:
                recorder.error(
                    "benchmark.resource_admission.unit_contract",
                    "benchmark recipe %s typed resource contract cannot be "
                    "recomputed: %s" % (label, exc),
                )
                continue
            transformed = float(
                expected_unit_contract["conversion"]["transformed_value"]
            )
            if (
                admission.get("unit_contract") != expected_unit_contract
                or observed != transformed
            ):
                recorder.error(
                    "benchmark.resource_admission.unit_contract",
                    "benchmark recipe %s typed resource conversion does not "
                    "match the frozen metric, units, transform, and budget"
                    % label,
                    metric=measured_metric,
                    measured=measured,
                    observed=observed,
                    expected_observed=transformed,
                )
                continue
        elif measured != observed:
            recorder.error(
                "benchmark.resource_admission.observed",
                "benchmark recipe %s admission value does not match its snapshotted step metric"
                % label,
                metric=measured_metric,
                observed=observed,
                measured=measured,
            )
            continue
        expected_metrics = {
            derived_metric_keys["admitted"]: 1 if admitted else 0,
            derived_metric_keys["observed"]: observed,
            derived_metric_keys["maximum"]: maximum,
            derived_metric_keys["tolerance"]: tolerance,
            derived_metric_keys["excess"]: expected_excess,
        }
        if any(metrics.get(key) != value for key, value in expected_metrics.items()):
            recorder.error(
                "benchmark.resource_admission.metrics",
                "benchmark recipe %s resource-admission metrics are inconsistent"
                % label,
            )
            continue
        recorder.pass_(
            "benchmark.resource_admission",
            "benchmark recipe %s resource admission verifies" % label,
        )
    if not saw_declaration and not checked:
        recorder.pass_(
            "benchmark.resource_admission",
            "benchmark result declares no measured resource-admission records",
        )


def _benchmark_resource_budget_declaration(
    benchmark_json: Optional[Mapping[str, Any]],
    recipe_id: str,
) -> Optional[JsonDict]:
    """Resolve one frozen resource declaration without trusting result rows."""

    if not isinstance(benchmark_json, Mapping):
        return None
    metadata = benchmark_json.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw: Any = metadata.get("resource_budget")
    legacy: Any = None
    raw_recipes = benchmark_json.get("recipes")
    raw_recipes = raw_recipes if isinstance(raw_recipes, list) else []
    matches = [
        row
        for row in raw_recipes
        if isinstance(row, Mapping) and str(row.get("id") or "") == recipe_id
    ]
    if len(matches) != 1:
        raise ValueError("recipe id is missing or duplicated in benchmark.json")
    params = matches[0].get("params")
    params = params if isinstance(params, Mapping) else {}
    if params.get("resource_budget") is not None:
        raw = params.get("resource_budget")
    legacy = params.get("fixed_channel_use_budget")
    if raw is None and legacy is not None:
        raw = {
            "metric": "steps.wireless_channel.channel.uses_per_pixel",
            "maximum": legacy,
            "tolerance": 1e-9,
            "policy": "reject",
            "unit": "channel_use/source_pixel",
        }
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("resource_budget must be an object")
    metric = str(
        raw.get("metric")
        or "steps.wireless_channel.channel.uses_per_pixel"
    ).strip()
    parts = metric.split(".", 2)
    if len(parts) != 3 or parts[0] != "steps" or not parts[1]:
        raise ValueError("resource metric is not step-scoped")
    try:
        maximum = float(raw.get("maximum"))
        tolerance = float(raw.get("tolerance", 1e-9))
    except (TypeError, ValueError) as exc:
        raise ValueError("resource maximum/tolerance is not numeric") from exc
    if (
        not math.isfinite(maximum)
        or maximum < 0.0
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or str(raw.get("policy") or "reject").lower() != "reject"
    ):
        raise ValueError("resource declaration values are invalid")
    unit = str(raw.get("unit") or "channel_use/source_pixel").strip()
    metric_unit = str(raw.get("metric_unit") or unit).strip()
    aggregation_policy = str(
        raw.get("aggregation_policy") or "scalar_metric_as_emitted"
    ).strip()
    if not unit or not metric_unit or not aggregation_policy:
        raise ValueError(
            "resource unit, metric_unit, and aggregation_policy must be non-empty"
        )
    raw_conversion = raw.get("conversion")
    conversion = _verification_resource_conversion(
        raw_conversion,
        metric_unit=metric_unit,
        budget_unit=unit,
    )
    if raw_conversion is not None and "aggregation_policy" not in raw:
        raise ValueError(
            "converted resource budget must explicitly declare aggregation_policy"
        )
    return {
        "metric": metric,
        "maximum": maximum,
        "tolerance": tolerance,
        "policy": "reject",
        "unit": unit,
        "metric_unit": metric_unit,
        "aggregation_policy": aggregation_policy,
        **(
            {"conversion": conversion.to_evidence()}
            if conversion is not None
            else {}
        ),
    }


def _verification_resource_conversion(
    raw: Any,
    *,
    metric_unit: str,
    budget_unit: str,
) -> Optional[IdealizedNativePayloadUseProxy]:
    if raw is None:
        if metric_unit != budget_unit:
            raise ValueError(
                "resource metric and budget units differ without an explicit conversion"
            )
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("resource conversion must be an object")
    if (
        raw.get("kind")
        != "noema.resource_conversion.idealized_native_payload_use_proxy"
    ):
        raise ValueError("resource conversion kind is unsupported")
    raw_bindings = raw.get("executed_bindings")
    bindings = None
    try:
        if raw_bindings is not None:
            if not isinstance(raw_bindings, Mapping):
                raise ResourceUnitError(
                    "executed_bindings must be a mapping"
                )
            bindings = ExecutedCodedModulationBindings(
                modulation_order=raw_bindings.get("modulation_order"),
                code_rate=raw_bindings.get("code_rate"),
                source_unit=raw_bindings.get("source_unit"),
                output_unit=raw_bindings.get("output_unit"),
                binding_identity=raw_bindings.get("binding_identity"),
            )
        conversion = IdealizedNativePayloadUseProxy(
            source_unit=raw.get("source_unit"),
            output_unit=raw.get("output_unit"),
            modulation_order=raw.get("modulation_order"),
            nominal_code_rate=raw.get("nominal_code_rate"),
            executed_bindings=bindings,
        )
    except ResourceUnitError as exc:
        raise ValueError("resource conversion is invalid: %s" % exc) from exc
    if conversion.source_unit != metric_unit:
        raise ValueError(
            "resource conversion source_unit does not match metric_unit"
        )
    if conversion.output_unit != budget_unit:
        raise ValueError(
            "resource conversion output_unit does not match budget unit"
        )
    return conversion


def _check_benchmark_protocol(
    result: JsonDict,
    benchmark_json: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    benchmark = dict(result.get("benchmark") or {})
    profile_requested = False
    public_schema_valid = False
    if benchmark_json is not None:
        if benchmark.get("id") and benchmark_json.get("id") and benchmark.get("id") != benchmark_json.get("id"):
            recorder.error("benchmark.protocol.id", "result benchmark id does not match benchmark.json")
        if benchmark.get("version") and benchmark_json.get("version") and benchmark.get("version") != benchmark_json.get("version"):
            recorder.error("benchmark.protocol.version", "result benchmark version does not match benchmark.json")
        stored_sha = str(benchmark.get("sha256") or "")
        if stored_sha:
            if not benchmark_protocol_sha256_matches(benchmark_json, stored_sha):
                recorder.error(
                    "benchmark.protocol.sha256",
                    "result benchmark SHA-256 does not match benchmark.json",
                )
            else:
                recorder.pass_(
                    "benchmark.protocol.sha256",
                    "result benchmark SHA-256 matches benchmark.json",
                )
        frozen_metadata = (
            benchmark_json.get("metadata")
            if isinstance(benchmark_json.get("metadata"), Mapping)
            else {}
        )
        canonical_tier = frozen_metadata.get("benchmark_tier")
        legacy_tier = frozen_metadata.get("tier")
        if (
            canonical_tier not in (None, "")
            and legacy_tier not in (None, "")
            and str(canonical_tier).strip().lower()
            != str(legacy_tier).strip().lower()
        ):
            recorder.error(
                "benchmark.protocol.tier",
                "benchmark.json metadata.tier conflicts with metadata.benchmark_tier",
            )
            frozen_tier = ""
        else:
            frozen_tier = str(
                canonical_tier or legacy_tier or "smoke"
            ).strip().lower()
        profile_requested = _resolve_traceability_profile_request(
            frozen_metadata,
            recorder,
            context="benchmark.json metadata",
            check_id="benchmark.protocol.traceability_profile_request",
        )
        result_profile_requested = _resolve_traceability_profile_request(
            benchmark,
            recorder,
            context="result benchmark",
            check_id="benchmark.protocol.result_traceability_profile_request",
        )
        public_schema_valid = True
        if profile_requested:
            public_schema_valid = _check_retained_public_benchmark_schema(
                benchmark_json,
                recorder,
            )
        if frozen_tier not in {"smoke", "canonical", "experimental"}:
            recorder.error(
                "benchmark.protocol.tier",
                "benchmark.json tier is unsupported",
            )
        elif (
            benchmark.get("benchmark_tier") != frozen_tier
            or result_profile_requested is not profile_requested
        ):
            recorder.error(
                "benchmark.protocol.certification_state",
                "result does not preserve the frozen tier/traceability-profile request",
            )
        else:
            recorder.pass_(
                "benchmark.protocol.certification_state",
                "result preserves the frozen tier/traceability-profile request",
            )
        if profile_requested and frozen_tier != "canonical":
            recorder.error(
                "benchmark.protocol.publication_tier",
                "strongest-profile evidence must use the canonical benchmark tier",
            )
        if profile_requested:
            expected_profile = publication_verification_profile_binding()
            frozen_profile = frozen_metadata.get("verification_profile")
            result_metadata = benchmark.get("metadata")
            result_metadata = (
                result_metadata
                if isinstance(result_metadata, Mapping)
                else {}
            )
            result_profile = result_metadata.get("verification_profile")
            if frozen_profile != expected_profile:
                recorder.error(
                    "benchmark.protocol.verification_profile",
                    "strongest-profile benchmark does not bind the exact "
                    "%s verification profile" % expected_profile["id"],
                )
            elif result_profile != frozen_profile:
                recorder.error(
                    "benchmark.protocol.verification_profile",
                    "result does not preserve the frozen traceability "
                    "verification profile",
                )
            else:
                recorder.pass_(
                    "benchmark.protocol.verification_profile",
                    "strongest-profile benchmark and result preserve the "
                    "mandatory verification profile",
                )
        if legacy_tier not in (None, "") and canonical_tier in (None, ""):
            recorder.warning(
                "benchmark.protocol.legacy_tier",
                "benchmark uses legacy metadata.tier; migrate to metadata.benchmark_tier",
            )
        if frozen_tier != "canonical" or not profile_requested:
            recorder.warning(
                "benchmark.protocol.publication_scope",
                "internally valid evidence did not request the strongest "
                "canonical traceability profile",
                benchmark_tier=frozen_tier,
                traceability_profile_requested=profile_requested,
            )
    recipes = result.get("recipes") if isinstance(result.get("recipes"), list) else []
    frozen_recipes = (
        benchmark_json.get("recipes")
        if isinstance(benchmark_json, Mapping)
        and isinstance(benchmark_json.get("recipes"), list)
        else []
    )
    if benchmark_json is not None and len(recipes) != len(frozen_recipes):
        recorder.error(
            "benchmark.protocol.recipes",
            "result recipe count does not match benchmark.json",
        )
    if benchmark_json is not None:
        declared_baselines = [
            str(value).strip()
            for value in benchmark_json.get("baselines") or []
            if str(value).strip()
        ]
        linked_baselines = set()
        for row in frozen_recipes:
            if not isinstance(row, Mapping):
                continue
            recipe_id = str(row.get("id") or "").strip()
            params = row.get("params")
            params = params if isinstance(params, Mapping) else {}
            for candidate in (
                recipe_id,
                params.get("baseline_id"),
                params.get("method_id"),
            ):
                if candidate not in (None, ""):
                    linked_baselines.add(str(candidate).strip())
        unlinked = [
            baseline
            for baseline in declared_baselines
            if baseline not in linked_baselines
        ]
        if unlinked:
            recorder.error(
                "benchmark.protocol.baseline_roster",
                "declared baselines are not linked to recipe or method IDs: %s"
                % ", ".join(unlinked),
            )
        elif declared_baselines:
            recorder.pass_(
                "benchmark.protocol.baseline_roster",
                "declared baseline roster is linked to frozen recipe/method IDs",
            )
    dataset = dict(benchmark.get("dataset") or {})
    task = dict(benchmark.get("task") or {})
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, Mapping):
            continue
        label = str(recipe.get("id") or recipe.get("label") or index)
        if index < len(frozen_recipes) and isinstance(frozen_recipes[index], Mapping):
            frozen = frozen_recipes[index]
            expected_id = str(frozen.get("id") or "")
            expected_role = str(frozen.get("role") or "candidate")
            expected_label = str(frozen.get("label") or recipe.get("recipe_name") or "")
            if (
                str(recipe.get("id") or "") != expected_id
                or str(recipe.get("role") or "") != expected_role
                or str(recipe.get("label") or "") != expected_label
            ):
                recorder.error(
                    "benchmark.protocol.recipe_identity",
                    "benchmark recipe %s identity/role/label does not match benchmark.json"
                    % label,
                )
        status = str(recipe.get("status") or "").lower()
        if status not in {"completed", "rejected_resource_budget"}:
            recorder.error("benchmark.recipe.status", "benchmark recipe %s status is %s" % (label, status or "missing"))
        elif status == "rejected_resource_budget":
            recorder.pass_(
                "benchmark.recipe.status",
                "benchmark recipe %s was explicitly rejected by its resource budget"
                % label,
            )
        research = dict(recipe.get("research") or {})
        research_dataset = dict(research.get("dataset") or {})
        research_task = dict(research.get("task") or {})
        if dataset.get("id") and research_dataset.get("id") and dataset.get("id") != research_dataset.get("id"):
            recorder.error("benchmark.protocol.dataset", "recipe %s dataset does not match benchmark protocol" % label)
        if task.get("id") and research_task.get("id") and task.get("id") != research_task.get("id"):
            recorder.error("benchmark.protocol.task", "recipe %s task does not match benchmark protocol" % label)
        pack_unit = ((benchmark_json or {}).get("metadata") or {}).get(
            "statistical_unit"
        ) if isinstance((benchmark_json or {}).get("metadata"), Mapping) else None
        if pack_unit not in (None, "") and recipe.get("statistical_unit") != pack_unit:
            recorder.error(
                "benchmark.protocol.statistical_unit",
                "recipe %s statistical unit does not match benchmark.json" % label,
            )
    identity_message = "benchmark protocol identifiers checked"
    if benchmark_json is not None and profile_requested:
        if public_schema_valid:
            identity_message += " against the shipped public schema"
    recorder.pass_("benchmark.protocol.identity", identity_message)


def _check_retained_public_benchmark_schema(
    benchmark_json: Mapping[str, Any],
    recorder: _CheckRecorder,
) -> bool:
    """Fail closed when a strongest-profile retained pack is off-schema."""

    relative_schema = Path("schemas") / "benchmark_pack.schema.json"
    candidates = (
        Path(__file__).resolve().parents[3] / relative_schema,
        Path(sys.prefix) / "share" / "noema-lab" / relative_schema,
    )
    schema_path = next((path for path in candidates if path.is_file()), None)
    if schema_path is None:
        recorder.error(
            "benchmark.protocol.public_schema",
            "strongest-profile benchmark cannot locate the shipped public "
            "benchmark-pack schema",
            searched=[str(path) for path in candidates],
        )
        return False
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        recorder.error(
            "benchmark.protocol.public_schema",
            "strongest-profile benchmark cannot load the shipped public "
            "benchmark-pack schema",
            schema_path=str(schema_path),
            error=str(exc),
        )
        return False
    errors = sorted(
        validator.iter_errors(dict(benchmark_json)),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if not errors:
        return True
    first = errors[0]
    location = "/" + "/".join(str(value) for value in first.absolute_path)
    recorder.error(
        "benchmark.protocol.public_schema",
        "strongest-profile benchmark violates the shipped public "
        "benchmark-pack schema at %s: %s" % (location, first.message),
        schema_path=str(schema_path),
        error_count=len(errors),
    )
    return False


def _check_benchmark_common_conditions(
    result: JsonDict,
    benchmark_json: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    source = benchmark_json if isinstance(benchmark_json, Mapping) else {}
    metadata = source.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = (result.get("benchmark") or {}).get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    profile_requested = _resolve_traceability_profile_request(
        metadata,
        recorder,
        context="benchmark metadata",
        check_id="benchmark.common_conditions.traceability_profile_request",
    )
    conditions = metadata.get("common_conditions")
    if not profile_requested and not isinstance(conditions, Mapping):
        return
    entries = result.get("recipes")
    entries = entries if isinstance(entries, list) else []
    errors = validate_common_condition_evidence_set(
        [entry for entry in entries if isinstance(entry, Mapping)],
        publication_ready=profile_requested,
    )
    if errors:
        for message in errors:
            recorder.error("benchmark.common_conditions", message)
    else:
        recorder.pass_(
            "benchmark.common_conditions",
            "paired methods materialized identical source, randomness, power, receiver, and failure-denominator conditions",
        )


def _check_benchmark_training_evidence_snapshot(
    result_dir: Path,
    result: JsonDict,
    benchmark_json: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> None:
    try:
        validated = validate_benchmark_training_evidence_snapshot(
            result_dir,
            result,
            benchmark_source=benchmark_json,
        )
    except BenchmarkEvidenceError as exc:
        recorder.error(
            "benchmark.training_evidence_snapshot",
            "training-evidence snapshot is invalid: %s" % exc,
        )
        return
    if not validated.get("present"):
        frozen_metadata = (
            benchmark_json.get("metadata")
            if isinstance(benchmark_json, Mapping)
            and isinstance(benchmark_json.get("metadata"), Mapping)
            else {}
        )
        lineage_requirement = frozen_metadata.get(
            "require_disjoint_training_lineage"
        )
        if lineage_requirement not in (None, False, "", 0):
            recorder.error(
                "benchmark.training_evidence_snapshot",
                "frozen benchmark requires disjoint training lineage but the result "
                "has no training-evidence snapshot",
            )
            return
        recorder.pass_(
            "benchmark.training_evidence_snapshot",
            "benchmark declares no training-evidence snapshot",
        )
        return
    if benchmark_json is not None:
        try:
            from noema_lab.core.benchmarks import (
                BenchmarkPack,
                validate_trained_artifact_lineage_manifest,
            )

            lineage_pack = BenchmarkPack(
                id=str(benchmark_json.get("id") or "benchmark"),
                version=str(benchmark_json.get("version") or "1"),
                recipes=[],
                dataset=dict(benchmark_json.get("dataset") or {}),
                metadata=dict(benchmark_json.get("metadata") or {}),
            )
            projection_payload = validated.get("projection")
            projection_payload = (
                projection_payload
                if isinstance(projection_payload, Mapping)
                else {}
            )
            for entry in projection_payload.get("entries") or []:
                if not isinstance(entry, Mapping):
                    continue
                evidence = entry.get("evidence")
                evidence = evidence if isinstance(evidence, Mapping) else {}
                manifest_record = evidence.get("trained_artifact_manifest")
                if not isinstance(manifest_record, Mapping):
                    continue
                manifest_path = result_dir / str(manifest_record.get("path") or "")
                validate_trained_artifact_lineage_manifest(
                    lineage_pack,
                    manifest_path,
                )
        except (BenchmarkEvidenceError, OSError, ValueError) as exc:
            recorder.error(
                "benchmark.training_lineage",
                "snapshotted training lineage is publication-invalid: %s" % exc,
            )
            return
        recorder.pass_(
            "benchmark.training_lineage",
            "snapshotted trained-artifact fitting IDs and file hashes are held-out disjoint",
        )
    projection = validated.get("projection")
    projection = projection if isinstance(projection, Mapping) else {}
    recorder.pass_(
        "benchmark.training_evidence_snapshot",
        "training-evidence snapshot verifies (%d entries)"
        % len(projection.get("entries") or []),
    )


def _check_benchmark_backing_runs(
    store: LocalStore,
    result: JsonDict,
    registry: Optional[OperationRegistry],
    recorder: _CheckRecorder,
    *,
    deep: bool,
) -> None:
    recipes = result.get("recipes") if isinstance(result.get("recipes"), list) else []
    for index, entry in enumerate(recipes):
        if not isinstance(entry, Mapping):
            continue
        run_id = str(entry.get("run_id") or "")
        if not run_id:
            recorder.warning("benchmark.run_id", "benchmark recipe %s has no backing run_id" % (entry.get("id") or index))
            continue
        if not deep:
            recorder.pass_(
                "benchmark.backing_run",
                "result-local snapshot is authoritative for backing run %s" % run_id,
            )
            continue
        run_dir = store.runs_dir / run_id
        if not run_dir.is_dir():
            recorder.pass_(
                "benchmark.backing_run",
                "external backing run %s is absent; result-local snapshot remains authoritative"
                % run_id,
            )
            continue
        try:
            run_report = verify_run_bundle(store, run_id, registry=registry)
        except Exception as exc:
            recorder.error("benchmark.backing_run", "could not verify backing run %s: %s" % (run_id, exc))
            continue
        if run_report.get("status") == "invalid":
            recorder.warning(
                "benchmark.backing_run",
                "external backing run %s is invalid; result-local snapshot remains authoritative"
                % run_id,
                run_id=run_id,
            )
        elif run_report.get("status") == "warning":
            recorder.warning("benchmark.backing_run", "backing run %s has verifier warnings" % run_id, run_id=run_id)
        else:
            recorder.pass_("benchmark.backing_run", "backing run %s verifies" % run_id)


def _check_benchmark_run_evidence_snapshots(
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
) -> Optional[JsonDict]:
    try:
        validated = validate_benchmark_run_evidence_snapshots(result_dir, result)
    except BenchmarkRunEvidenceError as exc:
        recorder.error(
            "benchmark.run_evidence_snapshot",
            "result-local run-evidence snapshot is invalid: %s" % exc,
        )
        return None
    entries = validated.get("entries")
    entries = entries if isinstance(entries, list) else []
    recorder.pass_(
        "benchmark.run_evidence_snapshot",
        "result-local run evidence verifies (%d completed runs)" % len(entries),
    )
    return validated


def _benchmark_expected_outputs(
    result: Mapping[str, Any],
    benchmark_json: Optional[Mapping[str, Any]],
) -> Any:
    """Return the frozen output declaration, preferring benchmark.json."""

    if isinstance(benchmark_json, Mapping):
        metadata = benchmark_json.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata.get("expected_outputs")
    benchmark = result.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    metadata = benchmark.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return metadata.get("expected_outputs")


def _check_benchmark_expected_outputs(
    result_dir: Path,
    result: JsonDict,
    benchmark_json: Optional[JsonDict],
    recorder: _CheckRecorder,
) -> JsonDict:
    """Require every frozen output declaration and its plot evidence closure."""

    empty: JsonDict = {
        "all": [],
        "plot_artifacts": [],
        "plot_sidecars": [],
    }
    raw = _benchmark_expected_outputs(result, benchmark_json)
    if raw is None:
        recorder.pass_(
            "benchmark.expected_outputs",
            "benchmark declares no additional outputs; generated plots remain optional",
            declared=False,
        )
        return empty
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(set(raw)) != len(raw)
    ):
        recorder.error(
            "benchmark.expected_outputs.schema",
            "benchmark metadata.expected_outputs must be a non-empty array of "
            "unique relative paths",
        )
        return empty

    if isinstance(benchmark_json, Mapping):
        benchmark = result.get("benchmark")
        benchmark = benchmark if isinstance(benchmark, Mapping) else {}
        result_metadata = benchmark.get("metadata")
        result_metadata = (
            result_metadata if isinstance(result_metadata, Mapping) else {}
        )
        if result_metadata.get("expected_outputs") != raw:
            recorder.error(
                "benchmark.expected_outputs.result_projection",
                "result does not preserve the frozen expected-output declaration",
            )

    declared: List[str] = []
    plot_artifacts: List[str] = []
    plot_images: List[str] = []
    invalid = False
    for relative in raw:
        try:
            candidate = _resolve_artifact_path(
                result_dir,
                {"relative_path": relative},
            )
        except ValueError as exc:
            recorder.error(
                "benchmark.expected_outputs.location",
                "declared output %s is unsafe: %s" % (relative, exc),
                relative_path=relative,
            )
            invalid = True
            continue
        declared.append(relative)
        if not candidate.is_file():
            recorder.error(
                "benchmark.expected_outputs.missing",
                "declared output is missing: %s" % relative,
                relative_path=relative,
            )
            invalid = True
        path = Path(relative)
        if len(path.parts) > 1 and path.parts[0] == "plots":
            suffix = path.suffix.lower()
            if suffix in {".csv", ".pdf", ".png", ".svg"}:
                plot_artifacts.append(relative)
            if suffix in {".pdf", ".png", ".svg"}:
                plot_images.append(relative)

    sidecars = [relative + ".plot.json" for relative in plot_images]
    for relative in sidecars:
        try:
            candidate = _resolve_artifact_path(
                result_dir,
                {"relative_path": relative},
            )
        except ValueError as exc:
            recorder.error(
                "benchmark.expected_outputs.location",
                "required plot sidecar %s is unsafe: %s" % (relative, exc),
                relative_path=relative,
            )
            invalid = True
            continue
        if not candidate.is_file():
            recorder.error(
                "benchmark.expected_outputs.missing",
                "declared plot output is missing its result-bound sidecar: %s"
                % relative,
                relative_path=relative,
            )
            invalid = True

    if not invalid:
        recorder.pass_(
            "benchmark.expected_outputs",
            "%d declared output(s) and %d required plot sidecar(s) exist"
            % (len(declared), len(sidecars)),
            declared_paths=declared,
            required_plot_sidecars=sidecars,
        )
    return {
        "all": declared,
        "plot_artifacts": plot_artifacts,
        "plot_sidecars": sidecars,
    }


def _check_benchmark_report_artifacts(
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    reports = result.get("reports")
    if not isinstance(reports, Mapping):
        recorder.error(
            "benchmark.reports",
            "benchmark result is missing hashed report descriptors",
        )
        return
    expected = {
        "metrics_csv": ("noema.benchmark.metrics_csv", "metrics.csv"),
        "recipes_csv": ("noema.benchmark.recipes_csv", "recipes.csv"),
        "summary_markdown": (
            "noema.benchmark.summary_markdown",
            "summary.md",
        ),
    }
    verified_paths: Dict[str, Path] = {}
    for key, (kind, relative_path) in expected.items():
        descriptor = reports.get(key)
        if not isinstance(descriptor, Mapping):
            recorder.error(
                "benchmark.reports.%s" % key,
                "benchmark report descriptor %s is missing" % key,
            )
            continue
        if (
            descriptor.get("kind") != kind
            or descriptor.get("relative_path") != relative_path
            or not isinstance(descriptor.get("size_bytes"), int)
            or int(descriptor.get("size_bytes") or -1) < 0
        ):
            recorder.error(
                "benchmark.reports.%s" % key,
                "benchmark report descriptor %s is malformed" % key,
            )
            continue
        sha256 = str(descriptor.get("sha256") or "").lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            recorder.error(
                "benchmark.reports.%s" % key,
                "benchmark report descriptor %s has no valid SHA-256" % key,
            )
            continue
        candidate = (result_dir / relative_path).resolve()
        if (
            candidate.parent != result_dir.resolve()
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            recorder.error(
                "benchmark.reports.%s" % key,
                "benchmark report %s is missing or escapes the result" % key,
            )
            continue
        if (
            candidate.stat().st_size != int(descriptor["size_bytes"])
            or file_sha256(candidate) != sha256
        ):
            recorder.error(
                "benchmark.reports.%s" % key,
                "benchmark report %s does not match its result-bound digest" % key,
            )
            continue
        verified_paths[key] = candidate
    if len(verified_paths) != len(expected):
        return
    try:
        from noema_lab.core.benchmarks import reproduce_benchmark_report_artifacts

        with tempfile.TemporaryDirectory(prefix="noema-verify-reports-") as temp_dir:
            reproduced = reproduce_benchmark_report_artifacts(
                Path(temp_dir), result
            )
            for key in expected:
                actual_sha = file_sha256(verified_paths[key])
                expected_sha = file_sha256(reproduced[key])
                if actual_sha != expected_sha:
                    recorder.error(
                        "benchmark.reports.%s" % key,
                        "benchmark report %s differs from the reproduced result "
                        "projection" % key,
                        expected=expected_sha,
                        actual=actual_sha,
                    )
                else:
                    recorder.pass_(
                        "benchmark.reports.%s" % key,
                        "benchmark report %s bytes and result projection verify" % key,
                    )
    except Exception as exc:
        recorder.error(
            "benchmark.reports.reproduction",
            "benchmark reports could not be reproduced: %s" % exc,
        )


def _check_benchmark_plot_artifacts(
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    """Bind each recorded plot to its exact image and plotted-data bytes."""

    raw_plots = result.get("plots")
    if raw_plots is None or raw_plots == []:
        recorder.pass_("benchmark.plots", "benchmark result records no generated plots")
        return
    if not isinstance(raw_plots, list):
        recorder.error("benchmark.plots.schema", "benchmark plots must be an array")
        return

    seen_ids = set()
    seen_paths = set()
    verified = 0
    for index, raw_plot in enumerate(raw_plots):
        if not isinstance(raw_plot, Mapping):
            recorder.error(
                "benchmark.plot.schema",
                "benchmark plot %d is not an object" % index,
            )
            continue
        plot_id = str(raw_plot.get("id") or "").strip()
        if not plot_id or plot_id in seen_ids:
            recorder.error(
                "benchmark.plot.id",
                "benchmark plot %d has a missing or duplicate id" % index,
            )
        else:
            seen_ids.add(plot_id)

        artifacts = raw_plot.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "image",
            "data_csv",
        }:
            recorder.error(
                "benchmark.plot.artifacts",
                "benchmark plot %s must bind image and data_csv artifacts"
                % (plot_id or index),
            )
            continue

        bindings = (
            (
                "image",
                "relative_path",
                "sha256",
                "size_bytes",
            ),
            (
                "data_csv",
                "data_csv_relative_path",
                "data_csv_sha256",
                "data_csv_size_bytes",
            ),
        )
        resolved: Dict[str, Path] = {}
        binding_failed = False
        if "semantic_projection_sha256" in raw_plot:
            declared_semantic_sha = raw_plot.get("semantic_projection_sha256")
            if not _is_sha256(declared_semantic_sha):
                recorder.error(
                    "benchmark.plot.semantic_projection",
                    "benchmark plot %s semantic projection SHA-256 is malformed"
                    % (plot_id or index),
                )
                binding_failed = True
            else:
                try:
                    from noema_lab.core.benchmark_plots import (
                        benchmark_plot_semantic_projection_sha256,
                    )

                    reproduced_semantic_sha = (
                        benchmark_plot_semantic_projection_sha256(
                            result,
                            raw_plot,
                        )
                    )
                except Exception as exc:
                    recorder.error(
                        "benchmark.plot.semantic_projection",
                        "benchmark plot %s semantic projection could not be "
                        "reproduced: %s" % (plot_id or index, exc),
                    )
                    binding_failed = True
                else:
                    if reproduced_semantic_sha != declared_semantic_sha:
                        recorder.error(
                            "benchmark.plot.semantic_projection",
                            "benchmark plot %s semantic projection identity differs "
                            "from the selected ordered result rows"
                            % (plot_id or index),
                            expected=declared_semantic_sha,
                            actual=reproduced_semantic_sha,
                        )
                        binding_failed = True
                    else:
                        recorder.pass_(
                            "benchmark.plot.semantic_projection",
                            "benchmark plot %s semantic projection identity verifies"
                            % plot_id,
                        )
        for role, path_key, sha_key, size_key in bindings:
            descriptor = artifacts.get(role)
            if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "relative_path",
                "sha256",
                "size_bytes",
            }:
                recorder.error(
                    "benchmark.plot.artifact_schema",
                    "benchmark plot %s %s descriptor is invalid"
                    % (plot_id or index, role),
                )
                binding_failed = True
                continue
            relative = str(descriptor.get("relative_path") or "")
            expected_sha = descriptor.get("sha256")
            expected_size = descriptor.get("size_bytes")
            if (
                raw_plot.get(path_key) != relative
                or raw_plot.get(sha_key) != expected_sha
                or raw_plot.get(size_key) != expected_size
            ):
                recorder.error(
                    "benchmark.plot.projection",
                    "benchmark plot %s %s top-level and artifact bindings differ"
                    % (plot_id or index, role),
                )
                binding_failed = True
            if relative in seen_paths:
                recorder.error(
                    "benchmark.plot.path_duplicate",
                    "benchmark plot artifact path is reused: %s" % relative,
                )
                binding_failed = True
            else:
                seen_paths.add(relative)
            if not _is_sha256(expected_sha):
                recorder.error(
                    "benchmark.plot.sha256",
                    "benchmark plot %s %s SHA-256 is missing or malformed"
                    % (plot_id or index, role),
                )
                binding_failed = True
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                recorder.error(
                    "benchmark.plot.size",
                    "benchmark plot %s %s size is missing or malformed"
                    % (plot_id or index, role),
                )
                binding_failed = True
            try:
                path = _resolve_artifact_path(
                    result_dir,
                    {"relative_path": relative},
                )
            except ValueError as exc:
                recorder.error(
                    "benchmark.plot.location",
                    "benchmark plot %s %s: %s" % (plot_id or index, role, exc),
                )
                binding_failed = True
                continue
            if not path.is_file():
                recorder.error(
                    "benchmark.plot.exists",
                    "benchmark plot %s %s artifact is missing"
                    % (plot_id or index, role),
                    path=str(path),
                )
                binding_failed = True
                continue
            resolved[role] = path
            if isinstance(expected_size, int) and not isinstance(expected_size, bool):
                actual_size = path.stat().st_size
                if actual_size != expected_size:
                    recorder.error(
                        "benchmark.plot.size",
                        "benchmark plot %s %s size mismatch"
                        % (plot_id or index, role),
                        expected=expected_size,
                        actual=actual_size,
                    )
                    binding_failed = True
            if _is_sha256(expected_sha):
                actual_sha = file_sha256(path)
                if actual_sha != expected_sha:
                    recorder.error(
                        "benchmark.plot.sha256",
                        "benchmark plot %s %s hash mismatch"
                        % (plot_id or index, role),
                        expected=expected_sha,
                        actual=actual_sha,
                    )
                    binding_failed = True

        data_path = resolved.get("data_csv")
        if data_path is not None:
            try:
                with data_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeError, csv.Error) as exc:
                recorder.error(
                    "benchmark.plot.data_csv",
                    "benchmark plot %s data CSV cannot be parsed: %s"
                    % (plot_id or index, exc),
                )
                binding_failed = True
            else:
                point_count = raw_plot.get("point_count")
                if (
                    isinstance(point_count, bool)
                    or not isinstance(point_count, int)
                    or point_count != len(rows)
                ):
                    recorder.error(
                        "benchmark.plot.point_count",
                        "benchmark plot %s point_count does not match its data CSV"
                        % (plot_id or index),
                        expected=len(rows),
                        actual=point_count,
                    )
                    binding_failed = True
                for metric_key in ("x_metric", "y_metric"):
                    declared = str(raw_plot.get(metric_key) or "")
                    if not declared or any(
                        str(row.get(metric_key) or "") != declared for row in rows
                    ):
                        recorder.error(
                            "benchmark.plot.metric_projection",
                            "benchmark plot %s %s does not match every data row"
                            % (plot_id or index, metric_key),
                        )
                        binding_failed = True
        image_path = resolved.get("image")
        if data_path is not None and image_path is not None and not binding_failed:
            declared_format = str(raw_plot.get("format") or "").strip().lower()
            actual_format = image_path.suffix.lower().lstrip(".")
            if not declared_format or declared_format != actual_format:
                recorder.error(
                    "benchmark.plot.format",
                    "benchmark plot %s format does not match its image path"
                    % (plot_id or index),
                    declared=declared_format,
                    actual=actual_format,
                )
                binding_failed = True
            else:
                try:
                    from noema_lab.core.benchmark_plots import (
                        reproduce_benchmark_plot_artifacts,
                    )

                    with tempfile.TemporaryDirectory(
                        prefix="noema-verify-plot-"
                    ) as temp_dir:
                        temp_root = Path(temp_dir)
                        reproduced_image = temp_root / ("plot." + declared_format)
                        reproduced_data = temp_root / "plot.csv"
                        reproduced_rows = reproduce_benchmark_plot_artifacts(
                            result,
                            raw_plot,
                            image_path=reproduced_image,
                            data_path=reproduced_data,
                        )
                        if len(reproduced_rows) != raw_plot.get("point_count"):
                            raise ValueError(
                                "reproduced row count differs from point_count"
                            )
                        reproduced_data_sha = file_sha256(reproduced_data)
                        if reproduced_data_sha != file_sha256(data_path):
                            recorder.error(
                                "benchmark.plot.data_semantics",
                                "benchmark plot %s data CSV differs from the reproduced "
                                "verified result projection" % (plot_id or index),
                                expected=reproduced_data_sha,
                                actual=file_sha256(data_path),
                            )
                            binding_failed = True
                        reproduced_image_sha = file_sha256(reproduced_image)
                        if reproduced_image_sha != file_sha256(image_path):
                            recorder.error(
                                "benchmark.plot.image_semantics",
                                "benchmark plot %s rendered image differs from the "
                                "reproduced data and render specification"
                                % (plot_id or index),
                                expected=reproduced_image_sha,
                                actual=file_sha256(image_path),
                            )
                            binding_failed = True
                except Exception as exc:
                    recorder.error(
                        "benchmark.plot.reproduction",
                        "benchmark plot %s could not be reproduced: %s"
                        % (plot_id or index, exc),
                    )
                    binding_failed = True
        if not binding_failed and len(resolved) == 2:
            verified += 1
            recorder.pass_(
                "benchmark.plot.artifacts",
                "benchmark plot %s image and data CSV verify" % plot_id,
            )
    if verified == len(raw_plots):
        recorder.pass_(
            "benchmark.plots",
            "%d benchmark plot artifact set(s) verify" % verified,
        )


def _check_benchmark_plot_sidecars(
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
    *,
    declared_plot_outputs: Sequence[str] = (),
    required_plot_sidecars: Sequence[str] = (),
) -> None:
    paths = sorted(result_dir.rglob("*.plot.json"))
    if not paths:
        if declared_plot_outputs or required_plot_sidecars:
            recorder.error(
                "benchmark.declared_plot_outputs",
                "declared plot outputs have no result-bound plot sidecar",
                declared_plot_outputs=list(declared_plot_outputs),
                required_plot_sidecars=list(required_plot_sidecars),
            )
            return
        recorder.pass_(
            "benchmark.plot_sidecars",
            "benchmark result records no post-run plot sidecars",
        )
        return
    result_path = result_dir / "result.json"
    source_identity = {
        "result_id": result_dir.name,
        "result_json_sha256": file_sha256(result_path),
        "result_json_size_bytes": int(result_path.stat().st_size),
    }
    plots: List[JsonDict] = []
    plots_by_sidecar: Dict[str, JsonDict] = {}
    seen_ids = set()
    for path in paths:
        try:
            sidecar_relative = path.relative_to(result_dir).as_posix()
        except ValueError:
            sidecar_relative = str(path)
        if path.is_symlink() or not path.is_file():
            recorder.error(
                "benchmark.plot_sidecar",
                "benchmark plot sidecar is missing or unsafe: %s" % path,
            )
            continue
        payload = _load_json_file(path, recorder, "benchmark plot sidecar")
        if payload is None:
            continue
        declared_sha = payload.get("sha256")
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "noema.benchmark_plot_evidence"
            or not _is_sha256(declared_sha)
            or declared_sha != _canonical_digest_without(payload, "sha256")
        ):
            recorder.error(
                "benchmark.plot_sidecar",
                "benchmark plot sidecar schema/digest is invalid: %s" % path.name,
            )
            continue
        if payload.get("source_result") != source_identity:
            recorder.error(
                "benchmark.plot_sidecar.result_identity",
                "benchmark plot sidecar is not bound to the live sealed result",
            )
            continue
        plot = payload.get("plot")
        if not isinstance(plot, Mapping):
            recorder.error(
                "benchmark.plot_sidecar",
                "benchmark plot sidecar has no plot record",
            )
            continue
        plot_id = str(plot.get("id") or "")
        if not plot_id or plot_id in seen_ids:
            recorder.error(
                "benchmark.plot_sidecar",
                "benchmark plot sidecar has a missing or duplicate plot id",
            )
            continue
        seen_ids.add(plot_id)
        plot_record = dict(plot)
        plots.append(plot_record)
        plots_by_sidecar[sidecar_relative] = plot_record
    if plots:
        projection = dict(result)
        projection["plots"] = plots
        _check_benchmark_plot_artifacts(result_dir, projection, recorder)
    if declared_plot_outputs or required_plot_sidecars:
        declared_set = set(declared_plot_outputs)
        observed_artifacts = {
            str(descriptor.get("relative_path") or "")
            for plot in plots
            for descriptor in (
                (plot.get("artifacts") or {}).values()
                if isinstance(plot.get("artifacts"), Mapping)
                else []
            )
            if isinstance(descriptor, Mapping)
        }
        missing_artifacts = sorted(declared_set - observed_artifacts)
        missing_sidecars = sorted(
            set(required_plot_sidecars) - set(plots_by_sidecar)
        )
        mismatched_sidecars: List[str] = []
        for sidecar_relative in required_plot_sidecars:
            plot = plots_by_sidecar.get(sidecar_relative)
            if plot is None:
                continue
            expected_image = sidecar_relative[: -len(".plot.json")]
            artifacts = plot.get("artifacts")
            artifacts = artifacts if isinstance(artifacts, Mapping) else {}
            image = artifacts.get("image")
            image = image if isinstance(image, Mapping) else {}
            if image.get("relative_path") != expected_image:
                mismatched_sidecars.append(sidecar_relative)
        if missing_artifacts or missing_sidecars or mismatched_sidecars:
            recorder.error(
                "benchmark.declared_plot_outputs",
                "declared plot image/CSV outputs are not closed by their exact "
                "result-bound sidecars",
                missing_artifacts=missing_artifacts,
                missing_sidecars=missing_sidecars,
                mismatched_sidecars=mismatched_sidecars,
            )
        else:
            recorder.pass_(
                "benchmark.declared_plot_outputs",
                "%d declared plot artifact(s) are closed by %d exact sidecar(s)"
                % (len(declared_set), len(required_plot_sidecars)),
                declared_plot_outputs=sorted(declared_set),
                required_plot_sidecars=sorted(required_plot_sidecars),
            )
    if len(plots) == len(paths):
        recorder.pass_(
            "benchmark.plot_sidecars",
            "%d result-bound plot sidecar(s) verify" % len(plots),
        )


def _check_benchmark_resource_guard_sidecar(
    result_dir: Path,
    result: JsonDict,
    recorder: _CheckRecorder,
) -> None:
    path = result_dir / "resource-guard.json"
    if path.is_symlink():
        recorder.error(
            "benchmark.resource_guard_sidecar",
            "benchmark resource-guard sidecar must be a regular file, not a symlink",
        )
        return
    if not path.exists():
        recorder.pass_(
            "benchmark.resource_guard_sidecar",
            "benchmark has no optional post-run resource-guard sidecar",
        )
        return
    payload = _load_json_file(path, recorder, "benchmark resource guard")
    if payload is None:
        return
    result_path = result_dir / "result.json"
    expected_result = {
        "result_id": result_dir.name,
        "result_json_sha256": file_sha256(result_path),
        "result_json_size_bytes": int(result_path.stat().st_size),
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "noema.benchmark_resource_guard_evidence"
        or not isinstance(payload.get("resource_guard"), Mapping)
        or not _is_sha256(payload.get("sha256"))
        or payload.get("sha256")
        != _canonical_digest_without(payload, "sha256")
        or payload.get("result") != expected_result
    ):
        recorder.error(
            "benchmark.resource_guard_sidecar",
            "benchmark resource-guard sidecar is malformed or bound to another result",
        )
        return
    recorder.pass_(
        "benchmark.resource_guard_sidecar",
        "post-run resource-guard evidence is bound to the sealed result",
    )


def _benchmark_flat_metrics(result: JsonDict) -> JsonDict:
    metrics: JsonDict = {}
    for recipe in result.get("recipes") or []:
        if not isinstance(recipe, Mapping):
            continue
        label = str(recipe.get("id") or recipe.get("label") or "recipe")
        raw_metrics = recipe.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            continue
        for key, value in raw_metrics.items():
            metrics["recipes.%s.%s" % (label, key)] = value
    return metrics


def _require_type(
    payload: Mapping[str, Any],
    key: str,
    expected_type: type,
    recorder: _CheckRecorder,
    scope: str,
) -> None:
    value = payload.get(key)
    if isinstance(value, expected_type) and not (expected_type is int and isinstance(value, bool)):
        recorder.pass_("schema.%s.%s" % (scope, key), "%s.%s has expected type" % (scope, key))
    else:
        recorder.error(
            "schema.%s.%s" % (scope, key),
            "%s.%s is missing or has the wrong type" % (scope, key),
            expected=expected_type.__name__,
            actual=type(value).__name__,
        )


def _resolve_artifact_path(run_dir: Path, record: Mapping[str, Any]) -> Path:
    relative = str(record.get("relative_path") or "")
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or any(ord(char) < 32 or ord(char) == 127 for char in relative)
    ):
        raise ValueError("artifact relative_path is not a safe bundle-relative path")
    cursor = run_dir
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("artifact relative_path traverses a symlink")
    candidate = cursor.resolve()
    if not _path_within(candidate, run_dir):
        raise ValueError("artifact relative_path escapes the run bundle")
    return candidate


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _artifact_label(record: Mapping[str, Any], index: int) -> str:
    return "%s.%s" % (record.get("step_id") or "artifact_%d" % index, record.get("output_name") or "output")


def _first_metric(metrics: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in metrics:
            value = _as_number(metrics[key])
            if value is not None:
                return value
        prefixed = [name for name in metrics if str(name).endswith("." + key)]
        for name in prefixed:
            value = _as_number(metrics[name])
            if value is not None:
                return value
    return None


def _source_pixels_from_artifacts(summary: JsonDict) -> Optional[float]:
    for step in summary.get("steps") or []:
        if not isinstance(step, Mapping) or str(step.get("id") or "") != "data":
            continue
        outputs = step.get("outputs") if isinstance(step.get("outputs"), Mapping) else {}
        for output in outputs.values():
            if not isinstance(output, Mapping):
                continue
            metadata = output.get("metadata") if isinstance(output.get("metadata"), Mapping) else {}
            shape = metadata.get("shape")
            if not isinstance(shape, list):
                arrays = metadata.get("arrays") if isinstance(metadata.get("arrays"), Mapping) else {}
                image_info = arrays.get("images") if isinstance(arrays.get("images"), Mapping) else {}
                shape = image_info.get("shape")
            pixels = _pixels_from_shape(shape)
            if pixels is not None:
                return pixels
    return None


def _pixels_from_shape(shape: Any) -> Optional[float]:
    if not isinstance(shape, list) or len(shape) < 2:
        return None
    try:
        dims = [int(item) for item in shape]
    except Exception:
        return None
    if len(dims) >= 4:
        return float(dims[0] * dims[1] * dims[2])
    if len(dims) == 3:
        return float(dims[0] * dims[1]) if dims[-1] in (1, 3, 4) else float(dims[0] * dims[1] * dims[2])
    return float(dims[0] * dims[1])


def _as_number(value: Any) -> Optional[float]:
    if not _is_json_number(value):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expects_nonnegative(name: str) -> bool:
    lowered = _metric_identity_for_plausibility(name)
    if "psnr" in lowered:
        return False
    return any(token in lowered for token in _NONNEGATIVE_TOKENS)


def _expects_unit_interval(name: str) -> bool:
    lowered = _metric_identity_for_plausibility(name)
    return any(token in lowered for token in _UNIT_INTERVAL_TOKENS)


def _metric_identity_for_plausibility(name: str) -> str:
    """Remove benchmark recipe and step qualifiers before classifying a metric."""

    parts = str(name).lower().split(".")
    if len(parts) >= 3 and parts[0] == "recipes":
        parts = parts[2:]
    if len(parts) >= 3 and parts[0] == "steps":
        parts = parts[2:]
    return ".".join(parts)


def _check_key(label: str) -> str:
    return "_".join(str(label).lower().split())


def _clean_details(details: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in details.items() if value is not None}


def _run_report_metadata(summary: Optional[JsonDict], manifest: Optional[JsonDict]) -> JsonDict:
    manifest_recipe = dict((manifest or {}).get("recipe") or {})
    payload: JsonDict = {
        "recipe_name": (summary or {}).get("recipe_name") or (manifest or {}).get("recipe_name"),
        "recipe_sha": manifest_recipe.get("sha256") or (summary or {}).get("recipe_sha256"),
        "created_time": (summary or {}).get("created_at_utc") or (manifest or {}).get("created_at_utc"),
        "completed_time": (summary or {}).get("completed_at_utc") or (manifest or {}).get("completed_at_utc"),
    }
    research = manifest_recipe.get("research") if isinstance(manifest_recipe.get("research"), Mapping) else {}
    benchmark = dict(research.get("benchmark") or {}) if isinstance(research, Mapping) else {}
    if benchmark.get("id"):
        payload["benchmark_id"] = benchmark.get("id")
    if benchmark.get("version"):
        payload["benchmark_version"] = benchmark.get("version")
    return payload
