from __future__ import annotations

import csv
import json
import copy
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.attempt_ledger import BenchmarkAttemptLedger
from noema_lab.core.executor import LocalExecutor
from noema_lab.core.execution_controls import (
    executor_options,
    normalize_execution_controls,
    require_strict_lint,
)
from noema_lab.core.benchmark_evidence import (
    BenchmarkEvidenceError,
    benchmark_training_evidence_bindings,
    snapshot_benchmark_training_evidence,
)
from noema_lab.core.benchmark_run_evidence import (
    audit_benchmark_run_evidence_snapshots,
    copy_file_independent,
    iter_benchmark_run_evidence_snapshots,
    snapshot_benchmark_run_evidence,
    validate_benchmark_run_evidence_snapshot,
)
from noema_lab.core.common_conditions import (
    CommonConditionError,
    bind_common_conditions,
    materialize_common_condition_evidence,
    validate_common_condition_evidence_set,
)
from noema_lab.core.lint import lint_recipe_invariants
from noema_lab.core.matrix import (
    RecipeMatrixError,
    canonicalize_recipe_matrix,
    materialize_recipe_matrix_selection,
    matrix_variant_id,
)
from noema_lab.core.operations import ExecutionCancelled, OperationRegistry
from noema_lab.core.planner import validate_recipe_against_registry
from noema_lab.core.publication_profile import (
    LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD,
    TRACEABILITY_PROFILE_REQUEST_FIELD,
    publication_verification_profile_binding,
    traceability_profile_requested,
)
from noema_lab.core.recipes import RecipeValidationError, load_recipe, recipe_from_dict
from noema_lab.core.reproducibility import (
    canonical_json_sha256,
    git_snapshot,
    utc_now_iso,
)
from noema_lab.core.resource_units import (
    ExecutedCodedModulationBindings,
    IdealizedNativePayloadUseProxy,
    ResourceQuantity,
    ResourceUnitError,
    evaluate_resource_admission as evaluate_typed_resource_admission,
)
from noema_lab.core.research import research_specs_from_recipe
from noema_lab.core.research_catalog import validate_research_specs_against_catalog
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)
from noema_lab.core.trained_artifacts import (
    TrainedArtifactError,
    validate_trained_artifact_publication_readiness,
)

JsonDict = Dict[str, Any]

BENCHMARK_METRIC_DEFINITION_VERSION = 1

# Protocol identities originally included BenchmarkPack.path, which is loader
# provenance rather than protocol content. Preserve the one retained identity
# produced by that legacy behavior while making all new identities independent
# of the checkout directory.
_LEGACY_BENCHMARK_PROTOCOL_IDENTITIES = {
    "c1377589944e786fe6bccafa225dd8226e275a062e6d057ca8bd96dfafcd360c": (
        "eada0783bcc3c532930a3242ab85cf15dbf3359f511a26530a405186f31bb824"
    ),
}


class BenchmarkError(RecipeValidationError):
    pass


def _benchmark_traceability_profile_requested(pack: "BenchmarkPack") -> bool:
    try:
        return traceability_profile_requested(
            pack.metadata or {},
            context="metadata",
        )
    except ValueError as exc:
        raise BenchmarkError("Benchmark %s %s" % (pack.id, exc)) from exc


@dataclass
class BenchmarkRecipe:
    id: str
    path: Path
    label: Optional[str] = None
    role: str = "candidate"
    params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "id": self.id,
            "path": str(self.path),
            "role": self.role,
            "params": dict(self.params),
        }
        if self.label:
            payload["label"] = self.label
        return payload


@dataclass
class BenchmarkPack:
    id: str
    version: str
    recipes: List[BenchmarkRecipe]
    path: Optional[Path] = None
    name: Optional[str] = None
    description: Optional[str] = None
    dataset: JsonDict = field(default_factory=dict)
    task: JsonDict = field(default_factory=dict)
    metrics: List[JsonDict] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    suite: JsonDict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "schema_version": self.schema_version,
            "id": self.id,
            "version": self.version,
            "dataset": dict(self.dataset),
            "task": dict(self.task),
            "metrics": [dict(metric) for metric in self.metrics],
            "baselines": list(self.baselines),
            "recipes": [recipe.to_dict() for recipe in self.recipes],
            "metadata": dict(self.metadata),
        }
        if self.suite:
            payload["suite"] = dict(self.suite)
        if self.name:
            payload["name"] = self.name
        if self.description:
            payload["description"] = self.description
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


def benchmark_protocol_payload(
    pack: BenchmarkPack | Mapping[str, Any],
) -> JsonDict:
    """Return protocol content without loader-local filesystem provenance."""

    payload = pack.to_dict() if isinstance(pack, BenchmarkPack) else dict(pack)
    payload.pop("path", None)
    return payload


def benchmark_protocol_sha256(
    pack: BenchmarkPack | Mapping[str, Any],
) -> str:
    """Identify a benchmark protocol portably, retaining audited old IDs."""

    portable_sha256 = canonical_json_sha256(benchmark_protocol_payload(pack))
    return _LEGACY_BENCHMARK_PROTOCOL_IDENTITIES.get(
        portable_sha256,
        portable_sha256,
    )


def benchmark_protocol_sha256_matches(
    pack: BenchmarkPack | Mapping[str, Any],
    expected_sha256: str,
) -> bool:
    """Accept portable identities and exact legacy snapshots during migration."""

    if benchmark_protocol_sha256(pack) == expected_sha256:
        return True
    raw = pack.to_dict() if isinstance(pack, BenchmarkPack) else dict(pack)
    return canonical_json_sha256(raw) == expected_sha256


@dataclass
class _BenchmarkResumeState:
    source_result_id: str
    source_result_dir: Path
    source_attempt_id: str
    source_result_sha256: str
    reusable_rows: Dict[int, JsonDict]
    validated_evidence: Dict[int, JsonDict]


def load_benchmark_pack(path: Path) -> BenchmarkPack:
    data = _load_mapping(path)
    schema_version = int(data.get("schema_version") or 1)
    if schema_version != 1:
        raise BenchmarkError("Unsupported benchmark schema_version: %s" % schema_version)
    benchmark_id = _required_string(data, "id", "benchmark")
    version = str(data.get("version") or "1")
    raw_recipes = data.get("recipes")
    if not isinstance(raw_recipes, list) or not raw_recipes:
        raise BenchmarkError("Benchmark %s requires a non-empty recipes list" % benchmark_id)
    raw_suite = data.get("suite") or {}
    if not isinstance(raw_suite, Mapping):
        raise BenchmarkError("Benchmark %s suite must be a mapping" % benchmark_id)
    recipes = [_recipe_from_mapping(item, index) for index, item in enumerate(raw_recipes)]
    return BenchmarkPack(
        id=benchmark_id,
        version=version,
        name=_optional_string(data, "name", "benchmark"),
        description=_optional_string(data, "description", "benchmark"),
        dataset=dict(data.get("dataset") or {}),
        task=dict(data.get("task") or {}),
        metrics=_metric_list(data.get("metrics") or []),
        baselines=_string_list(data.get("baselines") or [], "benchmark.baselines"),
        recipes=recipes,
        metadata=dict(data.get("metadata") or {}),
        suite=dict(raw_suite),
        path=path,
        schema_version=schema_version,
    )


def list_benchmark_packs(directory: Path) -> List[BenchmarkPack]:
    if not directory.exists():
        return []
    packs = []
    failures = []
    for path in sorted([*directory.rglob("*.yaml"), *directory.rglob("*.yml"), *directory.rglob("*.json")]):
        try:
            packs.append(load_benchmark_pack(path))
        except Exception as exc:
            failures.append("%s: %s: %s" % (path, type(exc).__name__, exc))
    if failures:
        raise BenchmarkError(
            "Malformed benchmark pack%s discovered:\n%s"
            % (
                "s" if len(failures) != 1 else "",
                "\n".join("- %s" % failure for failure in failures),
            )
        )
    return packs


def validate_benchmark_pack(
    pack: BenchmarkPack,
    registry: OperationRegistry,
    project_root: Path,
    *,
    strict_lint: bool = False,
) -> JsonDict:
    _validate_benchmark_metadata(pack)
    if _benchmark_traceability_profile_requested(pack):
        _validate_public_benchmark_pack_schema(pack, project_root)
    _validate_publication_source_bindings(pack, registry)
    pack_catalog_validation = validate_research_specs_against_catalog(
        {
            "dataset": dict(pack.dataset or {}),
            "task": dict(pack.task or {}),
            "metrics": [dict(metric) for metric in pack.metrics],
        }
    )
    if pack_catalog_validation["errors"]:
        raise BenchmarkError("; ".join(pack_catalog_validation["errors"]))
    recipe_rows = []
    for entry in pack.recipes:
        _benchmark_run_evidence_retained_artifact_paths(entry)
        recipe_path = resolve_benchmark_recipe_path(pack, entry, project_root)
        recipe = _load_benchmark_recipe(pack, entry, project_root)
        validate_recipe_against_registry(recipe, registry)
        lint_report = lint_recipe_invariants(recipe, registry, strict=True)
        if (
            (_benchmark_tier(pack) == "canonical" or strict_lint)
            and lint_report["status"] != "passed"
        ):
            raise BenchmarkError(
                "Benchmark %s requires strict-lint-clean recipes; %s failed: %s"
                % (
                    pack.id,
                    entry.id,
                    "; ".join(issue["message"] for issue in lint_report["issues"] if issue["severity"] == "error"),
                )
            )
        specs = research_specs_from_recipe(recipe)
        recipe_catalog_validation = specs.get("catalog_validation") or {}
        if recipe_catalog_validation.get("errors"):
            raise BenchmarkError(
                "Recipe %s has invalid research specs: %s"
                % (entry.id, "; ".join(recipe_catalog_validation.get("errors") or []))
            )
        _validate_research_compatibility(pack, entry, specs)
        _validate_resource_budget_declaration(pack, entry)
        if not _skip_benchmark_recipe(entry):
            _validate_recipe_training_lineage(
                pack,
                entry,
                recipe,
                recipe_path=recipe_path,
                project_root=project_root,
                registry=registry,
            )
        recipe_rows.append(
            {
                "id": entry.id,
                "label": entry.label or recipe.name,
                "role": entry.role,
                "path": str(recipe_path),
                "recipe_name": recipe.name,
                "research": specs,
                "catalog_validation": recipe_catalog_validation,
                "lint": lint_report,
            }
        )
    payload = {
        "schema_version": 1,
        "id": pack.id,
        "version": pack.version,
        "name": pack.name,
        "recipe_count": len(recipe_rows),
        "recipes": recipe_rows,
        "catalog_validation": pack_catalog_validation,
        "metadata": dict(pack.metadata or {}),
        "suite": dict(pack.suite or {}),
        "benchmark_tier": _benchmark_tier(pack),
        "sha256": benchmark_protocol_sha256(pack),
    }
    return payload


def run_benchmark_pack(
    pack: BenchmarkPack,
    registry: OperationRegistry,
    store: LocalStore,
    project_root: Path,
    event_sink: Optional[Callable[[JsonDict], None]] = None,
    *,
    strict_lint: bool = False,
    backend: Optional[str] = None,
    implementation: Optional[str] = None,
    parallel_workers: int = 1,
    use_plan_cache: bool = True,
    resume_result_id: Optional[str] = None,
    retain_backing_runs: bool = False,
) -> Path:
    if not isinstance(retain_backing_runs, bool):
        raise ValueError("retain_backing_runs must be a boolean")
    execution = normalize_execution_controls(
        {
            "strict_lint": strict_lint,
            "backend": backend,
            "implementation": implementation,
            "parallel_workers": parallel_workers,
            "use_plan_cache": use_plan_cache,
        }
    )
    protocol_sha256 = benchmark_protocol_sha256(pack)
    ledger = BenchmarkAttemptLedger(store.benchmarks_dir)
    access_policy = dict((pack.dataset or {}).get("access_policy") or {})
    publication_test = bool(access_policy.get("publication_test"))
    access_token_id = str(access_policy.get("seal_sha256") or "").strip()
    access_budget = int(access_policy.get("access_budget") or 0)
    git = git_snapshot(project_root)
    invocation: JsonDict = {
        "argv": list(sys.argv),
        "code_commit": git.get("commit"),
        "git_available": bool(git.get("available")),
        "git_dirty": git.get("dirty"),
        "selection_role": str(
            (pack.dataset or {}).get("selection_role") or "unspecified"
        ),
        "test_access": publication_test,
        "dataset_id": (pack.dataset or {}).get("id"),
        "dataset_version": (pack.dataset or {}).get("version"),
        "dataset_split": (pack.dataset or {}).get("split"),
        "execution": dict(execution),
        "storage": {
            "retain_backing_runs": bool(retain_backing_runs),
        },
    }
    if resume_result_id is not None:
        resume_result_id = str(resume_result_id).strip()
        if not resume_result_id or Path(resume_result_id).name != resume_result_id:
            raise BenchmarkError(
                "resume result ID must be one benchmark-result directory name"
            )
        invocation["resume"] = {
            "explicit": True,
            "source_result_id": resume_result_id,
        }
    if publication_test:
        invocation.update(
            {
                "access_token_id": access_token_id,
                "access_budget": access_budget,
                "declared_access_ledger": access_policy.get("access_ledger"),
            }
        )
    started_attempt = ledger.begin(
        benchmark_id=pack.id,
        benchmark_version=pack.version,
        protocol_sha256=protocol_sha256,
        invocation=invocation,
    )
    attempt_id = str(started_attempt["attempt_id"])
    reserved_invocation = dict(
        (started_attempt.get("payload") or {}).get("invocation") or {}
    )
    attempt_finalized = False
    resume_state: Optional[_BenchmarkResumeState] = None
    try:
        if publication_test and (
            reserved_invocation.get("access_granted") is not True
        ):
            raise BenchmarkError(
                "Publication-test access budget is exhausted for sealed population %s"
                % (access_token_id or "<missing-seal>")
            )
        validation = validate_benchmark_pack(
            pack,
            registry,
            project_root,
            strict_lint=bool(execution["strict_lint"]),
        )
        if resume_result_id is not None:
            resume_state = _load_benchmark_resume_state(
                store,
                ledger,
                resume_result_id,
                pack=pack,
                protocol_sha256=str(validation["sha256"]),
                execution=execution,
            )
        benchmark_dir = store.create_benchmark_dir(pack.id)
    except Exception as exc:
        ledger.finalize(attempt_id, status="failed", error=str(exc))
        raise
    benchmark_source = pack.to_dict()
    profile_requested = _benchmark_traceability_profile_requested(pack)
    result_metadata = dict(pack.metadata or {})
    result_metadata.pop(LEGACY_TRACEABILITY_PROFILE_REQUEST_FIELD, None)
    result_metadata[TRACEABILITY_PROFILE_REQUEST_FIELD] = profile_requested
    result: JsonDict = {
        "schema_version": 1,
        "kind": "noema.benchmark_result",
        "benchmark": {
            "id": pack.id,
            "version": pack.version,
            "name": pack.name,
            "sha256": validation["sha256"],
            "benchmark_tier": str(
                validation.get("benchmark_tier") or _benchmark_tier(pack)
            ),
            TRACEABILITY_PROFILE_REQUEST_FIELD: profile_requested,
            "dataset": dict(pack.dataset or {}),
            "task": dict(pack.task or {}),
            "metrics": [dict(metric) for metric in pack.metrics],
            "baselines": list(pack.baselines),
            "metadata": result_metadata,
            "suite": dict(pack.suite or {}),
            "catalog_validation": validation.get("catalog_validation"),
        },
        "status": "running",
        "attempt_id": attempt_id,
        "attempt_ledger": {
            "kind": "noema.benchmark_attempt_ledger",
            "relative_path": ".attempt-ledger",
            "start_event_sha256": started_attempt.get("event_sha256"),
        },
        "created_at_utc": utc_now_iso(),
        "execution": dict(execution),
        "storage": {
            "retain_backing_runs": bool(retain_backing_runs),
            "backing_run_policy": (
                "retain"
                if retain_backing_runs
                else "prune_after_verified_snapshot"
            ),
            "pruned_backing_run_count": 0,
            "backing_run_prune_failures": [],
        },
        "recipes": [],
    }
    if resume_state is not None:
        result["resume"] = {
            "schema_version": 1,
            "kind": "noema.benchmark_resume",
            "explicit": True,
            "source_result_id": resume_state.source_result_id,
            "source_attempt_id": resume_state.source_attempt_id,
            "source_result_json_sha256": resume_state.source_result_sha256,
            "eligible_recipe_count": len(resume_state.reusable_rows),
            "reused_recipe_count": 0,
        }
    try:
        store.write_json(benchmark_dir / "benchmark.json", benchmark_source)
        store.write_json(benchmark_dir / "result.json", result)
        if event_sink is not None:
            event_sink(
                {
                    "kind": "benchmark_created",
                    "message": "Created benchmark result directory",
                    "result_id": benchmark_dir.name,
                    "benchmark_id": pack.id,
                }
            )
        snapshot_benchmark_training_evidence(
            benchmark_dir,
            result,
            benchmark_source,
            project_root,
        )
        bindings = benchmark_training_evidence_bindings(
            benchmark_dir,
            result,
            benchmark_source,
            project_root,
        )
        prepared_recipes = []
        binding_use = {str(binding["snapshot_path"]): 0 for binding in bindings}
        for entry in pack.recipes:
            recipe_path = resolve_benchmark_recipe_path(pack, entry, project_root)
            recipe = _load_benchmark_recipe(pack, entry, project_root)
            if not _skip_benchmark_recipe(entry):
                recipe, used, semantic_recipe_sha256 = _bind_recipe_training_evidence(
                    recipe,
                    recipe_path=recipe_path,
                    pack=pack,
                    project_root=project_root,
                    bindings=bindings,
                )
                for snapshot_path in used:
                    binding_use[snapshot_path] += 1
                _validate_recipe_training_lineage(
                    pack,
                    entry,
                    recipe,
                    recipe_path=recipe_path,
                    project_root=project_root,
                    registry=registry,
                    allowed_manifest_paths={
                        Path(str(binding["snapshot_path"])).resolve()
                        for binding in bindings
                    },
                )
            else:
                semantic_recipe_sha256 = canonical_json_sha256(recipe.to_dict())
            prepared_recipes.append(
                (entry, recipe_path, recipe, semantic_recipe_sha256)
            )
        unused_bindings = [path for path, count in binding_use.items() if count == 0]
        if unused_bindings:
            raise BenchmarkEvidenceError(
                "no executable benchmark recipe consumes snapshotted trained artifact(s): %s"
                % ", ".join(unused_bindings)
            )
        if resume_state is not None:
            _validate_resume_recipe_bindings(resume_state, prepared_recipes)
        store.write_json(benchmark_dir / "result.json", result)

        for entry_index, (entry, recipe_path, recipe, semantic_recipe_sha256) in enumerate(
            prepared_recipes
        ):
            if (
                resume_state is not None
                and entry_index in resume_state.reusable_rows
            ):
                reused_row = _reuse_benchmark_recipe_row(
                    resume_state,
                    benchmark_dir,
                    entry_index=entry_index,
                )
                result["recipes"].append(reused_row)
                result["resume"]["reused_recipe_count"] = int(
                    result["resume"].get("reused_recipe_count") or 0
                ) + 1
                store.write_json(benchmark_dir / "result.json", result)
                validate_benchmark_run_evidence_snapshot(
                    benchmark_dir,
                    reused_row,
                    entry_index=entry_index,
                    include_payloads=False,
                )
                _durably_commit_benchmark_recipe_snapshot(
                    benchmark_dir,
                    reused_row,
                )
                if event_sink is not None:
                    event_sink(
                        {
                            "kind": "benchmark_recipe_reused",
                            "message": "Reused validated benchmark recipe evidence",
                            "result_id": benchmark_dir.name,
                            "source_result_id": resume_state.source_result_id,
                            "entry_id": entry.id,
                            "entry_index": entry_index,
                            "run_id": reused_row.get("run_id"),
                        }
                    )
                continue
            if _skip_benchmark_recipe(entry):
                result["recipes"].append(
                    {
                        "id": entry.id,
                        "label": entry.label or recipe.name,
                        "role": entry.role,
                        "recipe_name": recipe.name,
                        "recipe_path": str(recipe_path),
                        "status": "skipped",
                        "skip_reason": str(entry.params.get("skip_reason") or "placeholder recipe requires a user-provided adapter"),
                        "metrics": {},
                    }
                )
                store.write_json(benchmark_dir / "result.json", result)
                _durably_commit_benchmark_result_checkpoint(benchmark_dir)
                continue
            try:
                if execution["strict_lint"]:
                    require_strict_lint(
                        recipe,
                        registry,
                        context="%s/%s" % (pack.id, entry.id),
                    )
                run_dir = LocalExecutor(registry, store).run(
                    recipe,
                    **executor_options(execution),
                )
                summary = store.get_run(run_dir.name)
                manifest = store.get_manifest(run_dir.name)
                role_metric_definitions = _metric_definitions_for_role(
                    pack.metrics,
                    entry.role,
                )
                _, preliminary_metric_provenance = (
                    _collect_summary_metrics_with_provenance(
                        summary,
                        role_metric_definitions,
                    )
                )
                metric_producer_steps = sorted(
                    {
                        str(producer.get("source_step") or "")
                        for definition in role_metric_definitions
                        for producer in [
                            preliminary_metric_provenance.get(
                                str(definition.get("id") or "")
                            )
                        ]
                        if isinstance(producer, Mapping)
                        and str(producer.get("source_step") or "")
                    }
                )
                run_evidence_snapshot = snapshot_benchmark_run_evidence(
                    benchmark_dir,
                    entry_id=entry.id,
                    entry_index=entry_index,
                    run_dir=run_dir,
                    run_id=run_dir.name,
                    semantic_recipe_sha256=semantic_recipe_sha256,
                    metric_producer_steps=metric_producer_steps,
                    retained_artifact_paths=(
                        _benchmark_run_evidence_retained_artifact_paths(entry)
                    ),
                )
            except Exception as recipe_exc:
                result["recipes"].append(
                    {
                        "id": entry.id,
                        "label": entry.label or recipe.name,
                        "role": entry.role,
                        "recipe_name": recipe.name,
                        "recipe_path": str(recipe_path),
                        "status": "failed",
                        "error": str(recipe_exc),
                        "metrics": {},
                    }
                )
                raise
            metrics, metric_provenance = _collect_summary_metrics_with_provenance(
                summary,
                role_metric_definitions,
                source_run_evidence_sha256=str(
                    run_evidence_snapshot.get("files_sha256") or ""
                ),
            )
            resource_admission = _evaluate_resource_admission(
                pack,
                entry,
                metrics,
                protocol_sha256=str(validation["sha256"]),
            )
            row_status = summary.get("status")
            if resource_admission is not None:
                metrics.update(_resource_admission_metrics(resource_admission))
                if not resource_admission["admitted"]:
                    row_status = "rejected_resource_budget"
            row = {
                    "id": entry.id,
                    "label": entry.label or recipe.name,
                    "role": entry.role,
                    "recipe_name": recipe.name,
                    "recipe_path": str(recipe_path),
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "status": row_status,
                    "manifest": str(run_dir / "manifest.json"),
                    "recipe_sha256": manifest.get("recipe", {}).get("sha256"),
                    "semantic_recipe_sha256": semantic_recipe_sha256,
                    "research": manifest.get("recipe", {}).get("research"),
                    "metrics": metrics,
                    "metric_provenance": metric_provenance,
                    "run_evidence_snapshot": run_evidence_snapshot,
                }
            common_condition_evidence = materialize_common_condition_evidence(
                store.read_json(run_dir / "recipe.json"), summary
            )
            if common_condition_evidence:
                row["common_condition_evidence"] = common_condition_evidence
            recipe_metadata = dict(getattr(recipe, "metadata", {}) or {})
            pairing_id = next(
                (
                    recipe_metadata.get(key)
                    for key in (
                        "pairing_id",
                        "paired_seed",
                        "benchmark_paired_seed",
                    )
                    if recipe_metadata.get(key) not in (None, "")
                ),
                None,
            )
            if pairing_id not in (None, ""):
                row["pairing_id"] = str(pairing_id)
                if isinstance(pairing_id, (int, float)) and not isinstance(pairing_id, bool):
                    row["pairing_seed"] = pairing_id
            aggregation_cell_id = recipe_metadata.get("aggregation_cell_id")
            if aggregation_cell_id not in (None, ""):
                row["aggregation_cell_id"] = str(aggregation_cell_id)
            statistical_unit = recipe_metadata.get("statistical_unit")
            if statistical_unit in (None, ""):
                statistical_unit = (pack.metadata or {}).get("statistical_unit")
            if statistical_unit not in (None, ""):
                row["statistical_unit"] = copy.deepcopy(statistical_unit)
            if resource_admission is not None:
                row["resource_admission"] = resource_admission
            result["recipes"].append(row)
            store.write_json(benchmark_dir / "result.json", result)
            validate_benchmark_run_evidence_snapshot(
                benchmark_dir,
                row,
                entry_index=entry_index,
                include_payloads=False,
            )
            _durably_commit_benchmark_recipe_snapshot(
                benchmark_dir,
                row,
            )
            if not retain_backing_runs:
                try:
                    _prune_benchmark_backing_run(store, run_dir)
                except Exception as prune_exc:
                    result["storage"]["backing_run_prune_failures"].append(
                        {
                            "run_id": run_dir.name,
                            "error": str(prune_exc),
                        }
                    )
                    if event_sink is not None:
                        event_sink(
                            {
                                "kind": "benchmark_backing_run_prune_failed",
                                "message": "Could not prune redundant benchmark backing run",
                                "result_id": benchmark_dir.name,
                                "entry_id": entry.id,
                                "run_id": run_dir.name,
                                "error": str(prune_exc),
                            }
                        )
                else:
                    result["storage"]["pruned_backing_run_count"] = int(
                        result["storage"].get("pruned_backing_run_count") or 0
                    ) + 1
                    if event_sink is not None:
                        event_sink(
                            {
                                "kind": "benchmark_backing_run_pruned",
                                "message": "Pruned redundant benchmark backing run",
                                "result_id": benchmark_dir.name,
                                "entry_id": entry.id,
                                "run_id": run_dir.name,
                            }
                        )
                store.write_json(benchmark_dir / "result.json", result)
                _durably_commit_benchmark_result_checkpoint(benchmark_dir)
        audit_benchmark_run_evidence_snapshots(benchmark_dir, result)
        condition_errors = validate_common_condition_evidence_set(
            [row for row in result.get("recipes") or [] if isinstance(row, Mapping)],
            publication_ready=_benchmark_traceability_profile_requested(pack),
        )
        if condition_errors and _benchmark_traceability_profile_requested(pack):
            raise CommonConditionError("; ".join(condition_errors))
        skipped_rows = [
            row
            for row in result.get("recipes") or []
            if isinstance(row, Mapping)
            and str(row.get("status") or "").lower() == "skipped"
        ]
        result["status"] = "incomplete" if skipped_rows else "completed"
        if skipped_rows:
            result["incomplete_reason"] = (
                "benchmark contains %d skipped/placeholder method(s)"
                % len(skipped_rows)
            )
        result["completed_at_utc"] = utc_now_iso()
        executable_rows = [
            row
            for row in result.get("recipes") or []
            if isinstance(row, Mapping) and row.get("status") != "skipped"
        ]
        terminal_status = (
            "incomplete"
            if skipped_rows
            else "resource_rejected"
            if executable_rows
            and all(row.get("status") == "rejected_resource_budget" for row in executable_rows)
            else "completed"
        )
        result["attempt_ledger"]["expected_terminal_status"] = terminal_status
        write_benchmark_reports(benchmark_dir, result)
        store.write_json(benchmark_dir / "result.json", result)
        ledger.finalize(
            attempt_id,
            status=terminal_status,
            result_dir=benchmark_dir,
            recipe_outcomes=[
                {
                    "recipe_id": str(row.get("id") or ""),
                    "status": str(row.get("status") or ""),
                }
                for row in result.get("recipes") or []
                if isinstance(row, Mapping)
            ],
            details=_benchmark_attempt_details(result),
        )
        attempt_finalized = True
        return benchmark_dir
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["completed_at_utc"] = utc_now_iso()
        failure_status = (
            "cancelled" if isinstance(exc, ExecutionCancelled) else "failed"
        )
        result["attempt_ledger"]["expected_terminal_status"] = failure_status
        write_benchmark_reports(benchmark_dir, result)
        store.write_json(benchmark_dir / "result.json", result)
        if not attempt_finalized:
            ledger.finalize(
                attempt_id,
                status=failure_status,
                result_dir=benchmark_dir,
                recipe_outcomes=[
                    {
                        "recipe_id": str(row.get("id") or ""),
                        "status": str(row.get("status") or ""),
                    }
                    for row in result.get("recipes") or []
                    if isinstance(row, Mapping)
                ],
                error=str(exc),
                details=_benchmark_attempt_details(result),
            )
        raise


def _load_benchmark_resume_state(
    store: LocalStore,
    ledger: BenchmarkAttemptLedger,
    source_result_id: str,
    *,
    pack: BenchmarkPack,
    protocol_sha256: str,
    execution: Mapping[str, Any],
) -> _BenchmarkResumeState:
    """Validate a failed terminal result before any of its work can be reused."""

    source_result_dir = store.get_benchmark_result_dir(source_result_id)
    if (
        source_result_dir.is_symlink()
        or not source_result_dir.is_dir()
        or source_result_dir.resolve().parent != store.benchmarks_dir.resolve()
    ):
        raise BenchmarkError(
            "resume source is not a safe benchmark result directory: %s"
            % source_result_id
        )
    result_path = source_result_dir / "result.json"
    benchmark_path = source_result_dir / "benchmark.json"
    for label, path in (
        ("result.json", result_path),
        ("benchmark.json", benchmark_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise BenchmarkError(
                "resume source %s has no safe %s"
                % (source_result_id, label)
            )
    source_result = store.read_json(result_path)
    if str(source_result.get("status") or "").lower() != "failed":
        raise BenchmarkError(
            "benchmark resume requires a failed terminal result; %s has status %s"
            % (
                source_result_id,
                source_result.get("status") or "<missing>",
            )
        )
    source_benchmark = source_result.get("benchmark")
    if not isinstance(source_benchmark, Mapping):
        raise BenchmarkError(
            "resume source %s has no benchmark identity" % source_result_id
        )
    if (
        source_benchmark.get("id") != pack.id
        or str(source_benchmark.get("version") or "") != str(pack.version)
        or source_benchmark.get("sha256") != protocol_sha256
    ):
        raise BenchmarkError(
            "resume source benchmark pack/protocol identity does not match the requested pack"
        )
    source_pack = store.read_json(benchmark_path)
    if not benchmark_protocol_sha256_matches(source_pack, protocol_sha256):
        raise BenchmarkError(
            "resume source benchmark.json does not match the requested protocol"
        )
    source_execution = source_result.get("execution")
    if not isinstance(source_execution, Mapping) or dict(source_execution) != dict(
        execution
    ):
        raise BenchmarkError(
            "resume source execution settings do not match; use the same strict-lint, "
            "backend, implementation, parallel-worker, and plan-cache options"
        )

    descriptor = source_result.get("attempt_ledger")
    source_attempt_id = source_result.get("attempt_id")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("kind") != "noema.benchmark_attempt_ledger"
        or descriptor.get("relative_path") != ".attempt-ledger"
        or descriptor.get("expected_terminal_status") != "failed"
        or not isinstance(source_attempt_id, str)
        or not source_attempt_id
    ):
        raise BenchmarkError(
            "resume source is not bound to a failed terminal benchmark attempt"
        )
    ledger_snapshot = ledger.verify()
    if ledger_snapshot.get("status") != "valid":
        raise BenchmarkError(
            "cannot resume while the benchmark attempt ledger is invalid: %s"
            % "; ".join(ledger_snapshot.get("errors") or [])
        )
    attempts = [
        row
        for row in ledger_snapshot.get("attempts") or []
        if isinstance(row, Mapping) and row.get("attempt_id") == source_attempt_id
    ]
    if len(attempts) != 1:
        raise BenchmarkError(
            "resume source attempt is absent or duplicated in the attempt ledger"
        )
    source_attempt = attempts[0]
    if (
        source_attempt.get("terminal_status") != "failed"
        or source_attempt.get("start_event_sha256")
        != descriptor.get("start_event_sha256")
        or source_attempt.get("benchmark_id") != pack.id
        or str(source_attempt.get("benchmark_version") or "")
        != str(pack.version)
        or source_attempt.get("protocol_sha256") != protocol_sha256
    ):
        raise BenchmarkError(
            "resume source attempt-ledger identity does not match the failed result"
        )
    attempt_invocation = source_attempt.get("invocation")
    if (
        not isinstance(attempt_invocation, Mapping)
        or dict(attempt_invocation.get("execution") or {}) != dict(execution)
    ):
        raise BenchmarkError(
            "resume source attempt-ledger execution settings do not match"
        )
    source_result_sha256 = file_sha256(result_path)
    expected_identity = {
        "result_id": source_result_id,
        "result_json_sha256": source_result_sha256,
        "result_json_size_bytes": int(result_path.stat().st_size),
    }
    if source_attempt.get("result") != expected_identity:
        raise BenchmarkError(
            "resume source result.json differs from its immutable attempt-ledger identity"
        )

    raw_rows = source_result.get("recipes")
    if not isinstance(raw_rows, list) or len(raw_rows) > len(pack.recipes):
        raise BenchmarkError("resume source recipe progress is malformed")
    reusable_rows: Dict[int, JsonDict] = {}
    seen_failed = False
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise BenchmarkError(
                "resume source recipe row %d is malformed" % index
            )
        row = dict(raw_row)
        if row.get("id") != pack.recipes[index].id:
            raise BenchmarkError(
                "resume source recipe ordering differs at row %d" % index
            )
        status = str(row.get("status") or "").lower()
        if status == "failed":
            if seen_failed or index != len(raw_rows) - 1:
                raise BenchmarkError(
                    "resume source must stop at its first failed recipe row"
                )
            seen_failed = True
            continue
        if status not in {
            "completed",
            "rejected_resource_budget",
            "skipped",
        }:
            raise BenchmarkError(
                "resume source recipe row %d has non-reusable status %s"
                % (index, status or "<missing>")
            )
        if seen_failed:
            raise BenchmarkError(
                "resume source contains recipe work after its failed row"
            )
        if status in {"completed", "rejected_resource_budget"}:
            reusable_rows[index] = copy.deepcopy(row)

    expected_outcomes = [
        {
            "recipe_id": str(row.get("id") or ""),
            "status": str(row.get("status") or ""),
        }
        for row in raw_rows
        if isinstance(row, Mapping)
    ]
    if source_attempt.get("recipe_outcomes") != expected_outcomes:
        raise BenchmarkError(
            "resume source recipe outcomes differ from its immutable attempt ledger"
        )
    try:
        validated_evidence: Dict[int, JsonDict] = {}
        for row in iter_benchmark_run_evidence_snapshots(
            source_result_dir,
            source_result,
            include_payloads=False,
        ):
            entry_index = int(row["entry_index"])
            validated_evidence[entry_index] = {
                "entry_index": entry_index,
                "descriptor": row["descriptor"],
                "snapshot_manifest": row["snapshot_manifest"],
            }
    except Exception as exc:
        raise BenchmarkError(
            "resume source run evidence is invalid: %s" % exc
        ) from exc
    missing_evidence = sorted(set(reusable_rows).difference(validated_evidence))
    if missing_evidence:
        raise BenchmarkError(
            "resume source is missing validated evidence for completed recipe row(s): %s"
            % ", ".join(str(index) for index in missing_evidence)
        )
    return _BenchmarkResumeState(
        source_result_id=source_result_id,
        source_result_dir=source_result_dir,
        source_attempt_id=source_attempt_id,
        source_result_sha256=source_result_sha256,
        reusable_rows=reusable_rows,
        validated_evidence=validated_evidence,
    )


def _prune_benchmark_backing_run(
    store: LocalStore,
    run_dir: Path,
) -> None:
    """Remove one redundant backing run without following an unsafe path."""

    run_id = run_dir.name
    expected = store.runs_dir / run_id
    if (
        not run_id
        or Path(run_id).name != run_id
        or run_dir != expected
        or store.runs_dir.is_symlink()
        or run_dir.is_symlink()
        or not run_dir.is_dir()
        or run_dir.resolve().parent != store.runs_dir.resolve()
    ):
        raise BenchmarkError(
            "refusing to prune unsafe benchmark backing-run path: %s"
            % run_dir
        )
    shutil.rmtree(run_dir)


def _durably_commit_benchmark_recipe_snapshot(
    benchmark_dir: Path,
    row: Mapping[str, Any],
) -> None:
    """Flush the independent evidence and its committed result projection."""

    descriptor = row.get("run_evidence_snapshot")
    if not isinstance(descriptor, Mapping):
        raise BenchmarkError(
            "cannot durably commit a benchmark row without a run-evidence snapshot"
        )
    relative_root = Path(str(descriptor.get("root") or ""))
    if (
        not relative_root.parts
        or relative_root.is_absolute()
        or ".." in relative_root.parts
    ):
        raise BenchmarkError(
            "cannot durably commit an unsafe run-evidence snapshot path"
        )
    snapshot_root = benchmark_dir / relative_root
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise BenchmarkError(
            "cannot durably commit a missing run-evidence snapshot"
        )
    snapshot_files = sorted(
        (
            path
            for path in snapshot_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.as_posix(),
    )
    if not snapshot_files:
        raise BenchmarkError("run-evidence snapshot contains no files")
    for path in [
        *snapshot_files,
        benchmark_dir / "result.json",
    ]:
        if path.is_symlink() or not path.is_file():
            raise BenchmarkError(
                "cannot durably commit missing benchmark evidence: %s" % path
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor_fd = os.open(str(path), flags)
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)
    directories = {
        benchmark_dir,
        snapshot_root,
        snapshot_root.parent,
        *(
            path.parent
            for path in snapshot_files
        ),
    }
    for path in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)


def _durably_commit_benchmark_result_checkpoint(benchmark_dir: Path) -> None:
    """Flush one growing result checkpoint without requiring final reports."""

    result_path = benchmark_dir / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise BenchmarkError(
            "cannot durably commit missing benchmark result checkpoint"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor_fd = os.open(str(result_path), flags)
    try:
        os.fsync(descriptor_fd)
    finally:
        os.close(descriptor_fd)
    descriptor_fd = os.open(str(benchmark_dir), os.O_RDONLY)
    try:
        os.fsync(descriptor_fd)
    finally:
        os.close(descriptor_fd)


def _validate_resume_recipe_bindings(
    resume_state: _BenchmarkResumeState,
    prepared_recipes: Iterable[Any],
) -> None:
    rows = list(prepared_recipes)
    for index, source_row in resume_state.reusable_rows.items():
        if index >= len(rows):
            raise BenchmarkError(
                "resume source contains work outside the requested benchmark pack"
            )
        entry, _recipe_path, recipe, semantic_recipe_sha256 = rows[index]
        if (
            source_row.get("id") != entry.id
            or source_row.get("role") != entry.role
            or source_row.get("recipe_name") != recipe.name
            or source_row.get("semantic_recipe_sha256")
            != semantic_recipe_sha256
        ):
            raise BenchmarkError(
                "resume source completed recipe %s no longer matches its prepared "
                "recipe or bound trained-artifact evidence"
                % entry.id
            )


def _reuse_benchmark_recipe_row(
    resume_state: _BenchmarkResumeState,
    benchmark_dir: Path,
    *,
    entry_index: int,
) -> JsonDict:
    source_row = resume_state.reusable_rows[entry_index]
    evidence = resume_state.validated_evidence[entry_index]
    descriptor = evidence["descriptor"]
    relative_root = Path(str(descriptor["root"]))
    source_root = resume_state.source_result_dir / relative_root
    destination_root = benchmark_dir / relative_root
    if destination_root.exists():
        raise BenchmarkError(
            "resume destination run evidence already exists: %s"
            % relative_root.as_posix()
        )
    staging_parent = Path(
        tempfile.mkdtemp(prefix=".resume-evidence-", dir=str(benchmark_dir))
    )
    staging_root = staging_parent / relative_root.name
    staging_root.mkdir()
    try:
        snapshot_manifest = evidence["snapshot_manifest"]
        records = list(snapshot_manifest.get("files") or [])
        manifest_relative = Path(str(descriptor["manifest"]["path"])).relative_to(
            relative_root
        )
        relative_files = [manifest_relative]
        for record in records:
            relative_files.append(
                Path(str(record["path"])).relative_to(relative_root)
            )
        for relative_file in relative_files:
            source = source_root / relative_file
            destination = staging_root / relative_file
            if source.is_symlink() or not source.is_file():
                raise BenchmarkError(
                    "resume source evidence changed after validation: %s"
                    % source
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_file_independent(source, destination)
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging_root), str(destination_root))
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    reused_row = copy.deepcopy(source_row)
    reused_row["resume_reuse"] = {
        "source_result_id": resume_state.source_result_id,
        "source_attempt_id": resume_state.source_attempt_id,
        "source_result_json_sha256": resume_state.source_result_sha256,
        "source_entry_index": entry_index,
        "evidence_revalidated": True,
    }
    return reused_row


def _benchmark_attempt_details(result: Mapping[str, Any]) -> JsonDict:
    resume = result.get("resume")
    if not isinstance(resume, Mapping):
        return {}
    return {"resume": copy.deepcopy(dict(resume))}


def finalize_resource_exhausted_benchmark(
    store: LocalStore,
    result_id: str,
    error: str,
    resource_guard: Mapping[str, Any],
    failure_kind: str = "resource_exhausted",
) -> None:
    """Turn an abruptly killed benchmark bundle into durable failed evidence."""

    if not result_id:
        return
    path = store.get_benchmark_result_dir(result_id) / "result.json"
    try:
        result = store.read_json(path)
    except (OSError, ValueError, TypeError):
        result = {
            "schema_version": 1,
            "kind": "noema.benchmark_result",
            "recipes": [],
        }
    if str(result.get("status") or "") in {"completed", "failed"}:
        return
    result["status"] = "failed"
    result["error"] = str(error)
    result["completed_at_utc"] = utc_now_iso()
    result["failure"] = {"kind": str(failure_kind)}
    result["resource_guard"] = dict(resource_guard)
    attempt_ledger = result.get("attempt_ledger")
    if isinstance(attempt_ledger, dict):
        attempt_ledger["expected_terminal_status"] = "failed"
    store.write_json(path, result)
    attempt_id = result.get("attempt_id")
    if isinstance(attempt_id, str) and attempt_id:
        ledger = BenchmarkAttemptLedger(store.benchmarks_dir)
        snapshot = ledger.verify()
        attempt = next(
            (
                row
                for row in snapshot.get("attempts") or []
                if isinstance(row, Mapping) and row.get("attempt_id") == attempt_id
            ),
            None,
        )
        if isinstance(attempt, Mapping) and attempt.get("terminal_status") is None:
            ledger.finalize(
                attempt_id,
                status="failed",
                result_dir=path.parent,
                recipe_outcomes=[
                    {
                        "recipe_id": str(row.get("id") or ""),
                        "status": str(row.get("status") or ""),
                    }
                    for row in result.get("recipes") or []
                    if isinstance(row, Mapping)
                ],
                error=str(error),
                details={"failure_kind": str(failure_kind), "resource_guard": dict(resource_guard)},
            )


def write_benchmark_reports(benchmark_dir: Path, result: JsonDict) -> JsonDict:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = benchmark_dir / "metrics.csv"
    recipes_path = benchmark_dir / "recipes.csv"
    summary_path = benchmark_dir / "summary.md"
    _write_metrics_csv(metrics_path, result)
    _write_recipes_csv(recipes_path, result)
    _write_summary_markdown(summary_path, result)
    def report_record(kind: str, path: Path) -> JsonDict:
        return {
            "kind": kind,
            "path": str(path),
            "relative_path": path.relative_to(benchmark_dir).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    reports = {
        "generated_at_utc": utc_now_iso(),
        "metrics_csv": report_record("noema.benchmark.metrics_csv", metrics_path),
        "recipes_csv": report_record("noema.benchmark.recipes_csv", recipes_path),
        "summary_markdown": report_record(
            "noema.benchmark.summary_markdown", summary_path
        ),
    }
    result["reports"] = reports
    return reports


def write_benchmark_resource_guard_evidence(
    store: LocalStore,
    result_id: str,
    resource_guard: Mapping[str, Any],
) -> JsonDict:
    """Write post-run supervisor evidence without rewriting sealed result.json."""

    result_dir = store.get_benchmark_result_dir(result_id)
    result_path = result_dir / "result.json"
    if not result_path.is_file() or result_path.is_symlink():
        raise BenchmarkError(
            "benchmark result %s has no safe result.json" % result_id
        )
    payload: JsonDict = {
        "schema_version": 1,
        "kind": "noema.benchmark_resource_guard_evidence",
        "result": {
            "result_id": result_id,
            "result_json_sha256": file_sha256(result_path),
            "result_json_size_bytes": int(result_path.stat().st_size),
        },
        "resource_guard": dict(resource_guard),
        "recorded_at_utc": utc_now_iso(),
    }
    payload["sha256"] = canonical_json_sha256(payload)
    store.write_json(result_dir / "resource-guard.json", payload)
    return payload


def reproduce_benchmark_report_artifacts(
    output_dir: Path, result: Mapping[str, Any]
) -> Dict[str, Path]:
    """Rebuild the three human-facing reports from result.json semantics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(result))
    paths = {
        "metrics_csv": output_dir / "metrics.csv",
        "recipes_csv": output_dir / "recipes.csv",
        "summary_markdown": output_dir / "summary.md",
    }
    _write_metrics_csv(paths["metrics_csv"], payload)
    _write_recipes_csv(paths["recipes_csv"], payload)
    _write_summary_markdown(paths["summary_markdown"], payload)
    return paths


def resolve_benchmark_recipe_path(pack: BenchmarkPack, entry: BenchmarkRecipe, project_root: Path) -> Path:
    path = entry.path
    if path.is_absolute():
        return path
    candidates = []
    if pack.path is not None:
        candidates.append(pack.path.parent / path)
    candidates.append(project_root / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_benchmark_recipe(pack: BenchmarkPack, entry: BenchmarkRecipe, project_root: Path):
    recipe_path = resolve_benchmark_recipe_path(pack, entry, project_root)
    recipe = load_recipe(recipe_path)
    payload = recipe.to_dict()
    if entry.params:
        _apply_benchmark_recipe_params(payload, entry)
    _bind_benchmark_dataset_split(payload, pack, entry)
    conditions = (pack.metadata or {}).get("common_conditions")
    if isinstance(conditions, Mapping):
        bind_common_conditions(
            payload,
            benchmark_id=pack.id,
            benchmark_version=pack.version,
            entry_id=entry.id,
            conditions=conditions,
            publication_ready=_benchmark_traceability_profile_requested(pack),
        )
    return recipe_from_dict(payload)


def _bind_recipe_training_evidence(
    recipe: Any,
    *,
    recipe_path: Path,
    pack: BenchmarkPack,
    project_root: Path,
    bindings: List[JsonDict],
) -> tuple[Any, set[str], str]:
    if not bindings:
        return recipe, set(), canonical_json_sha256(recipe.to_dict())
    payload = recipe.to_dict()
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    benchmark_method = str(metadata.get("benchmark_method") or "").strip()
    used: set[str] = set()

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key == "artifact_manifest_path" and isinstance(child, str) and child.strip():
                    matching = [
                        binding
                        for binding in bindings
                        if _artifact_path_matches_binding(
                            child,
                            binding,
                            recipe_path=recipe_path,
                            pack=pack,
                            project_root=project_root,
                        )
                    ]
                    if len(matching) > 1:
                        series_matching = [
                            binding
                            for binding in matching
                            if str(binding.get("series") or "") == benchmark_method
                        ]
                        matching = series_matching
                    if len(matching) > 1:
                        raise BenchmarkEvidenceError(
                            "recipe %s artifact_manifest_path matches multiple training-evidence snapshots"
                            % recipe.name
                        )
                    if matching:
                        snapshot_path = str(matching[0]["snapshot_path"])
                        value[key] = snapshot_path
                        used.add(snapshot_path)
                    continue
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    for step in payload.get("steps") or []:
        if isinstance(step, dict):
            rewrite(step.get("params"))
    relevant = [
        binding
        for binding in bindings
        if str(binding.get("series") or "") == benchmark_method
    ]
    if relevant and not used:
        raise BenchmarkEvidenceError(
            "benchmark method %s does not consume its declared trained artifact snapshot"
            % benchmark_method
        )
    bound_recipe = recipe_from_dict(payload)
    return (
        bound_recipe,
        used,
        semantic_benchmark_recipe_sha256(bound_recipe, bindings),
    )


def semantic_benchmark_recipe_sha256(
    recipe: Any,
    bindings: List[JsonDict],
) -> str:
    """Hash benchmark semantics without result-local artifact path identity."""

    payload = copy.deepcopy(recipe.to_dict())
    snapshot_hashes = {
        str(Path(str(binding.get("snapshot_path") or "")).resolve()): str(
            binding.get("sha256") or ""
        ).lower()
        for binding in bindings
        if binding.get("snapshot_path") and binding.get("sha256")
    }

    def canonicalize(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key == "artifact_manifest_path" and isinstance(child, str):
                    digest = snapshot_hashes.get(str(Path(child).resolve()))
                    if digest:
                        value[key] = "artifact-manifest-sha256:%s" % digest
                    continue
                canonicalize(child)
        elif isinstance(value, list):
            for child in value:
                canonicalize(child)

    canonicalize(payload)
    return canonical_json_sha256(payload)


def _artifact_path_matches_binding(
    raw_path: str,
    binding: Mapping[str, Any],
    *,
    recipe_path: Path,
    pack: BenchmarkPack,
    project_root: Path,
) -> bool:
    expected = Path(str(binding.get("source_path") or "")).resolve()
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve() == expected
    bases = [project_root.resolve(), recipe_path.parent.resolve()]
    if pack.path is not None:
        bases.append(pack.path.parent.resolve())
    bases.append(Path.cwd().resolve())
    return any((base / candidate).resolve() == expected for base in bases)


def _skip_benchmark_recipe(entry: BenchmarkRecipe) -> bool:
    params = dict(entry.params or {})
    role = str(entry.role or "").lower()
    return bool(params.get("skip")) or "placeholder" in role


def _benchmark_run_evidence_retained_artifact_paths(
    entry: BenchmarkRecipe,
) -> List[str]:
    """Normalize one pack entry's exact result-local artifact allowlist.

    Run-evidence retention is deliberately an entry-level execution setting:
    heterogeneous benchmark packs can retain transport evidence for protected
    digital methods without requiring the same artifact from semantic methods.
    The declaration remains part of ``BenchmarkRecipe.params`` and therefore
    of the frozen benchmark-pack identity.
    """

    params = entry.params if isinstance(entry.params, Mapping) else {}
    raw_policy = params.get("run_evidence")
    if raw_policy is None:
        return []
    if not isinstance(raw_policy, Mapping):
        raise BenchmarkError(
            "Benchmark recipe %s params.run_evidence must be a mapping"
            % entry.id
        )
    unsupported = sorted(
        str(field)
        for field in raw_policy
        if field != "retained_artifact_paths"
    )
    if unsupported:
        raise BenchmarkError(
            "Benchmark recipe %s params.run_evidence has unsupported field(s): %s"
            % (entry.id, ", ".join(unsupported))
        )
    if "retained_artifact_paths" not in raw_policy:
        raise BenchmarkError(
            "Benchmark recipe %s params.run_evidence must declare "
            "retained_artifact_paths" % entry.id
        )
    raw_paths = raw_policy.get("retained_artifact_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise BenchmarkError(
            "Benchmark recipe %s params.run_evidence.retained_artifact_paths "
            "must be a non-empty list" % entry.id
        )

    paths: List[str] = []
    for index, raw_path in enumerate(raw_paths):
        label = (
            "Benchmark recipe %s "
            "params.run_evidence.retained_artifact_paths[%d]"
            % (entry.id, index)
        )
        if not isinstance(raw_path, str):
            raise BenchmarkError("%s must be a string" % label)
        path = Path(raw_path)
        if (
            not raw_path
            or path.is_absolute()
            or not raw_path.startswith("artifacts/")
            or len(path.parts) < 2
            or ".." in path.parts
            or "\\" in raw_path
            or raw_path != path.as_posix()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in raw_path
            )
        ):
            raise BenchmarkError(
                "%s must be one canonical manifest-relative path below artifacts/"
                % label
            )
        paths.append(raw_path)
    if len(paths) != len(set(paths)):
        raise BenchmarkError(
            "Benchmark recipe %s "
            "params.run_evidence.retained_artifact_paths must be unique"
            % entry.id
        )
    return sorted(paths)


def _apply_benchmark_recipe_params(payload: JsonDict, entry: BenchmarkRecipe) -> None:
    params = dict(entry.params or {})
    matrix_selection = _benchmark_matrix_selection(params, entry.id)
    matrix_step_params: JsonDict = {}
    materialized_matrix_index: Optional[int] = None
    source_metadata = payload.get("metadata")
    has_source_matrix = isinstance(source_metadata, Mapping) and any(
        field in source_metadata for field in ("matrix", "sweeps", "ui_sweeps")
    )
    if matrix_selection is not None and has_source_matrix:
        try:
            source_recipe = recipe_from_dict(copy.deepcopy(payload))
            compiled_matrix = canonicalize_recipe_matrix(source_recipe)
            if compiled_matrix.enabled:
                materialized = materialize_recipe_matrix_selection(
                    source_recipe,
                    matrix_selection,
                )
                matrix_step_params = copy.deepcopy(
                    dict(compiled_matrix.definition.get("step_params") or {})
                )
                materialized_matrix_index = int(
                    (materialized.get("metadata") or {}).get("matrix_index", 0)
                )
                payload.clear()
                payload.update(materialized)
        except (RecipeMatrixError, RecipeValidationError) as exc:
            raise BenchmarkError(
                "Benchmark recipe %s has invalid params.matrix_selection: %s"
                % (entry.id, exc)
            ) from exc
    if params.get("name"):
        payload["name"] = str(params["name"])
    elif params:
        payload["name"] = "%s__%s" % (str(payload.get("name") or "recipe"), entry.id)
    if params.get("description"):
        payload["description"] = str(params["description"])

    metadata = dict(payload.get("metadata") or {})
    metadata.update(copy.deepcopy(dict(params.get("metadata") or {})))
    if matrix_selection is not None:
        metadata["matrix_selection"] = copy.deepcopy(matrix_selection)
        metadata["matrix_variant_id"] = matrix_variant_id(matrix_selection)
        # ``sweep_values`` was the pre-matrix name for the coordinates of one
        # concrete benchmark point.  Accept it at the pack boundary, but do not
        # propagate a second source of truth into the instantiated recipe.
        metadata.pop("sweep_values", None)
        # A benchmark entry identifies one concrete point. Matrix definitions
        # belong to the authored source recipe and must not survive on the
        # instantiated benchmark recipe as a second execution instruction.
        metadata.pop("matrix", None)
        metadata.pop("sweeps", None)
        metadata.pop("ui_sweeps", None)
        if materialized_matrix_index is not None:
            metadata["matrix_index"] = materialized_matrix_index
    if params.get("fixed_channel_use_budget") is not None:
        metadata["fixed_channel_use_budget"] = params["fixed_channel_use_budget"]
    if _skip_benchmark_recipe(entry):
        metadata["benchmark_placeholder"] = True
    payload["metadata"] = metadata

    step_params = dict(params.get("step_params") or {})
    if step_params:
        steps = {str(step.get("id")): step for step in payload.get("steps") or [] if isinstance(step, Mapping)}
        for step_id, overrides in step_params.items():
            if str(step_id) not in steps:
                raise BenchmarkError("Benchmark recipe %s overrides unknown step %s" % (entry.id, step_id))
            if not isinstance(overrides, Mapping):
                raise BenchmarkError("Benchmark recipe %s step_params.%s must be a mapping" % (entry.id, step_id))
            bound_params = matrix_step_params.get(str(step_id))
            if isinstance(bound_params, Mapping):
                for param_name in set(overrides).intersection(bound_params):
                    expected = dict(steps[str(step_id)].get("params") or {}).get(param_name)
                    actual = overrides[param_name]
                    if canonical_json_sha256(expected) != canonical_json_sha256(actual):
                        raise BenchmarkError(
                            "Benchmark recipe %s step_params.%s.%s conflicts with "
                            "params.matrix_selection"
                            % (entry.id, step_id, param_name)
                        )
            steps[str(step_id)].setdefault("params", {})
            steps[str(step_id)]["params"].update(copy.deepcopy(dict(overrides)))


def _benchmark_sample_ids(pack: BenchmarkPack) -> List[str]:
    dataset = dict(pack.dataset or {})
    raw = dataset.get("sample_ids") or dataset.get("image_ids")
    split = dataset.get("split")
    if raw in (None, "") and isinstance(split, Mapping):
        raw = split.get("sample_ids") or split.get("image_ids")
    if raw in (None, ""):
        metadata_split = (pack.metadata or {}).get("dataset_split")
        if isinstance(metadata_split, Mapping):
            raw = metadata_split.get("sample_ids") or metadata_split.get("image_ids")
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        return []
    output = [str(item).strip() for item in values if str(item).strip()]
    normalized = [item.lower() for item in output]
    if len(normalized) != len(set(normalized)):
        raise BenchmarkError(
            "Benchmark %s dataset sample_ids must be unique" % pack.id
        )
    return output


def _bind_benchmark_dataset_split(
    payload: JsonDict,
    pack: BenchmarkPack,
    entry: BenchmarkRecipe,
) -> None:
    """Bind one pack-owned held-out sample set into declared source operations.

    A benchmark entry may change methods and channel parameters, but it may not
    silently select a different evaluation population.  The authored source
    recipe remains unchanged; the materialized benchmark recipe records the
    binding in its semantic hash. Publication packs use explicit
    ``dataset.source_bindings`` records, which replace (rather than merge into)
    each source operation's complete parameter map. This prevents recipe-local
    crop, resize, multiplicity, or alternate-manifest parameters from surviving
    a nominal common-source declaration.
    """

    sample_ids = _benchmark_sample_ids(pack)
    if not sample_ids:
        return
    dataset_id = str((pack.dataset or {}).get("id") or "").strip()
    raw_preprocessing = (pack.dataset or {}).get("preprocessing")
    if raw_preprocessing is not None and not isinstance(raw_preprocessing, Mapping):
        raise BenchmarkError(
            "Benchmark %s dataset.preprocessing must be a mapping" % pack.id
        )
    preprocessing = copy.deepcopy(dict(raw_preprocessing or {}))
    raw_operation_params = preprocessing.get("operation_params")
    if raw_operation_params is None:
        operation_params: JsonDict = {}
    elif isinstance(raw_operation_params, Mapping):
        operation_params = copy.deepcopy(dict(raw_operation_params))
    else:
        raise BenchmarkError(
            "Benchmark %s dataset.preprocessing.operation_params must be a mapping"
            % pack.id
        )
    if bool((pack.metadata or {}).get("require_identical_source_transform")) and not preprocessing:
        raise BenchmarkError(
            "Benchmark %s requires dataset.preprocessing to bind one source transform"
            % pack.id
        )
    raw_bindings = (pack.dataset or {}).get("source_bindings")
    explicit_bindings = raw_bindings is not None
    if raw_bindings is None:
        bindings: List[JsonDict] = [
            {
                "operation": "source.image_dataset",
                "selection_param": "image_ids",
                "selection_format": "comma_separated",
                "dataset_param": "dataset",
                "params": {},
            }
        ]
    elif isinstance(raw_bindings, list) and raw_bindings:
        bindings = []
        for index, raw_binding in enumerate(raw_bindings):
            if not isinstance(raw_binding, Mapping):
                raise BenchmarkError(
                    "Benchmark %s dataset.source_bindings[%d] must be a mapping"
                    % (pack.id, index)
                )
            binding = copy.deepcopy(dict(raw_binding))
            operation = str(binding.get("operation") or "").strip()
            selection_param = str(binding.get("selection_param") or "").strip()
            if not operation or not selection_param:
                raise BenchmarkError(
                    "Benchmark %s dataset.source_bindings[%d] requires operation and selection_param"
                    % (pack.id, index)
                )
            bindings.append(binding)
    else:
        raise BenchmarkError(
            "Benchmark %s dataset.source_bindings must be a non-empty list"
            % pack.id
        )
    profile_requested = _benchmark_traceability_profile_requested(pack)
    if profile_requested and not explicit_bindings:
        raise BenchmarkError(
            "Strongest-profile benchmark %s must declare dataset.source_bindings"
            % pack.id
        )
    bound_steps: List[str] = []
    binding_evidence: List[JsonDict] = []
    for binding_index, binding in enumerate(bindings):
        operation = str(binding.get("operation") or "").strip()
        selection_param = str(binding.get("selection_param") or "image_ids").strip()
        selection_format = str(
            binding.get("selection_format") or "comma_separated"
        ).strip()
        if selection_format == "comma_separated":
            selection_value: Any = ",".join(sample_ids)
        elif selection_format == "list":
            selection_value = list(sample_ids)
        else:
            raise BenchmarkError(
                "Benchmark %s dataset.source_bindings[%d].selection_format must be comma_separated or list"
                % (pack.id, binding_index)
            )
        fixed_params = binding.get("params")
        if fixed_params is None:
            fixed_params = {}
        if not isinstance(fixed_params, Mapping):
            raise BenchmarkError(
                "Benchmark %s dataset.source_bindings[%d].params must be a mapping"
                % (pack.id, binding_index)
            )
        complete_params = copy.deepcopy(dict(fixed_params))
        # The shared preprocessing map remains a convenient pack-level spelling;
        # binding-local values win only when they are identical.
        for key, value in operation_params.items():
            if key in complete_params and canonical_json_sha256(
                complete_params[key]
            ) != canonical_json_sha256(value):
                raise BenchmarkError(
                    "Benchmark %s source binding conflicts with dataset.preprocessing.operation_params.%s"
                    % (pack.id, key)
                )
            complete_params[key] = copy.deepcopy(value)
        complete_params[selection_param] = copy.deepcopy(selection_value)
        dataset_param = str(binding.get("dataset_param") or "dataset").strip()
        if dataset_param and dataset_id:
            if dataset_param in complete_params and str(
                complete_params[dataset_param]
            ).strip() != dataset_id:
                raise BenchmarkError(
                    "Benchmark %s source binding dataset parameter conflicts with dataset.id"
                    % pack.id
                )
            complete_params[dataset_param] = dataset_id
        match_params = binding.get("match_params") or {}
        if not isinstance(match_params, Mapping):
            raise BenchmarkError(
                "Benchmark %s dataset.source_bindings[%d].match_params must be a mapping"
                % (pack.id, binding_index)
            )
        matched_steps: List[str] = []
        for step in payload.get("steps") or []:
            if not isinstance(step, dict) or str(step.get("op") or "") != operation:
                continue
            authored_params = dict(step.get("params") or {})
            if any(
                canonical_json_sha256(authored_params.get(key))
                != canonical_json_sha256(value)
                for key, value in match_params.items()
            ):
                continue
            if not profile_requested:
                merged = authored_params
                merged.update(copy.deepcopy(complete_params))
                step["params"] = merged
            else:
                if not complete_params:
                    raise BenchmarkError(
                        "Strongest-profile benchmark %s source binding %d must close the complete params map"
                        % (pack.id, binding_index)
                    )
                step["params"] = copy.deepcopy(complete_params)
            step_id = str(step.get("id") or "")
            matched_steps.append(step_id)
            bound_steps.append(step_id)
        if not matched_steps and binding.get("required", True) is not False:
            raise BenchmarkError(
                "Benchmark %s source binding %d found no %s step in recipe %s"
                % (pack.id, binding_index, operation, entry.id)
            )
        binding_evidence.append(
            {
                "operation": operation,
                "source_steps": matched_steps,
                "selection_param": selection_param,
                "selection_format": selection_format,
                "complete_params": complete_params,
                "complete_params_sha256": canonical_json_sha256(complete_params),
                "closed": profile_requested,
            }
        )
    if not bound_steps:
        raise BenchmarkError(
            "Benchmark %s declares dataset sample_ids but recipe %s has no matching "
            "declared source-binding step" % (pack.id, entry.id)
        )
    metadata = dict(payload.get("metadata") or {})
    metadata["benchmark_dataset_binding"] = {
        "benchmark_id": pack.id,
        "dataset_id": dataset_id,
        "split": copy.deepcopy((pack.dataset or {}).get("split")),
        "sample_ids": list(sample_ids),
        "source_steps": bound_steps,
        "source_bindings": binding_evidence,
        "preprocessing": preprocessing,
        "preprocessing_sha256": canonical_json_sha256(preprocessing),
    }
    payload["metadata"] = metadata


def _resource_budget_declaration(
    pack: BenchmarkPack,
    entry: BenchmarkRecipe,
) -> Optional[JsonDict]:
    raw: Any = (pack.metadata or {}).get("resource_budget")
    entry_budget = (entry.params or {}).get("resource_budget")
    if entry_budget is not None:
        raw = entry_budget
    legacy = (entry.params or {}).get("fixed_channel_use_budget")
    if raw is None and legacy is not None:
        raw = {
            "metric": "steps.wireless_channel.channel.uses_per_pixel",
            "maximum": legacy,
            "tolerance": 1e-9,
            "policy": "reject",
            "legacy_field": "fixed_channel_use_budget",
        }
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget must be a mapping" % entry.id
        )
    budget = dict(raw)
    metric = str(
        budget.get("metric")
        or "steps.wireless_channel.channel.uses_per_pixel"
    ).strip()
    maximum = budget.get("maximum")
    tolerance = budget.get("tolerance", 1e-9)
    policy = str(budget.get("policy") or "reject").strip().lower()
    try:
        maximum_value = float(maximum)
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget maximum/tolerance must be numeric"
            % entry.id
        ) from exc
    if not metric:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.metric must be non-empty" % entry.id
        )
    metric_parts = metric.split(".", 2)
    if len(metric_parts) != 3 or metric_parts[0] != "steps" or not metric_parts[1]:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.metric must identify one measured "
            "step as steps.<step_id>.<metric>" % entry.id
        )
    if not math.isfinite(maximum_value) or maximum_value < 0.0:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.maximum must be finite and nonnegative"
            % entry.id
        )
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.tolerance must be finite and nonnegative"
            % entry.id
        )
    if policy != "reject":
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.policy must be reject" % entry.id
        )
    unit = str(budget.get("unit") or "channel_use/source_pixel").strip()
    metric_unit = str(budget.get("metric_unit") or unit).strip()
    if not unit or not metric_unit:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget unit and metric_unit must be non-empty"
            % entry.id
        )
    aggregation_policy = str(
        budget.get("aggregation_policy") or "scalar_metric_as_emitted"
    ).strip()
    if not aggregation_policy:
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.aggregation_policy must be non-empty"
            % entry.id
        )
    raw_conversion = budget.get("conversion")
    conversion = _resource_conversion_from_declaration(
        raw_conversion,
        metric_unit=metric_unit,
        budget_unit=unit,
        entry_id=entry.id,
    )
    if raw_conversion is not None and "aggregation_policy" not in budget:
        raise BenchmarkError(
            "Benchmark recipe %s converted resource budget must explicitly "
            "declare aggregation_policy" % entry.id
        )
    if legacy is not None:
        try:
            legacy_value = float(legacy)
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(
                "Benchmark recipe %s deprecated fixed_channel_use_budget must be numeric"
                % entry.id
            ) from exc
        if not math.isclose(
            legacy_value,
            maximum_value,
            rel_tol=0.0,
            abs_tol=max(tolerance_value, 1e-12),
        ):
            raise BenchmarkError(
                "Benchmark recipe %s has conflicting resource_budget.maximum and "
                "deprecated fixed_channel_use_budget" % entry.id
            )
    return {
        "metric": metric,
        "maximum": maximum_value,
        "tolerance": tolerance_value,
        "policy": policy,
        "unit": unit,
        "metric_unit": metric_unit,
        "aggregation_policy": aggregation_policy,
        **(
            {"conversion": conversion.to_evidence()}
            if conversion is not None
            else {}
        ),
        **(
            {"legacy_field": str(budget["legacy_field"])}
            if budget.get("legacy_field")
            else {}
        ),
    }


def _validate_public_benchmark_pack_schema(
    pack: BenchmarkPack,
    project_root: Path,
) -> None:
    """Require strongest-profile packs to match the shipped public schema."""

    schema_path = project_root / "schemas" / "benchmark_pack.schema.json"
    if not schema_path.is_file():
        raise BenchmarkError(
            "Strongest-profile benchmark %s cannot locate public schema %s"
            % (pack.id, schema_path)
        )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        raise BenchmarkError(
            "Strongest-profile benchmark %s cannot load public schema: %s"
            % (pack.id, exc)
        ) from exc
    payload = pack.to_dict()
    payload.pop("path", None)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/" + "/".join(
            str(value) for value in first.absolute_path
        )
        raise BenchmarkError(
            "Strongest-profile benchmark %s violates the public benchmark-pack "
            "schema at %s: %s" % (pack.id, location, first.message)
        )


def _validate_resource_budget_declaration(
    pack: BenchmarkPack,
    entry: BenchmarkRecipe,
) -> None:
    _resource_budget_declaration(pack, entry)


def _resource_conversion_from_declaration(
    raw: Any,
    *,
    metric_unit: str,
    budget_unit: str,
    entry_id: str,
) -> Optional[IdealizedNativePayloadUseProxy]:
    if raw is None:
        if metric_unit != budget_unit:
            raise BenchmarkError(
                "Benchmark recipe %s resource metric unit %r differs from "
                "budget unit %r without an explicit conversion"
                % (entry_id, metric_unit, budget_unit)
            )
        return None
    if not isinstance(raw, Mapping):
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.conversion must be a mapping"
            % entry_id
        )
    if (
        raw.get("kind")
        != "noema.resource_conversion.idealized_native_payload_use_proxy"
    ):
        raise BenchmarkError(
            "Benchmark recipe %s resource_budget.conversion kind is unsupported"
            % entry_id
        )
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
        raise BenchmarkError(
            "Benchmark recipe %s has invalid resource conversion: %s"
            % (entry_id, exc)
        ) from exc
    if conversion.source_unit != metric_unit:
        raise BenchmarkError(
            "Benchmark recipe %s conversion source_unit does not match metric_unit"
            % entry_id
        )
    if conversion.output_unit != budget_unit:
        raise BenchmarkError(
            "Benchmark recipe %s conversion output_unit does not match budget unit"
            % entry_id
        )
    return conversion


def _evaluate_resource_admission(
    pack: BenchmarkPack,
    entry: BenchmarkRecipe,
    metrics: Mapping[str, Any],
    *,
    protocol_sha256: Optional[str] = None,
) -> Optional[JsonDict]:
    budget = _resource_budget_declaration(pack, entry)
    if budget is None:
        return None
    metric = str(budget["metric"])
    if metric not in metrics:
        raise BenchmarkError(
            "Benchmark recipe %s cannot enforce resource budget: metric %s is missing"
            % (entry.id, metric)
        )
    value = metrics[metric]
    if isinstance(value, bool):
        raise BenchmarkError(
            "Benchmark recipe %s resource metric %s is not numeric" % (entry.id, metric)
        )
    try:
        source_observed = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(
            "Benchmark recipe %s resource metric %s is not numeric" % (entry.id, metric)
        ) from exc
    if not math.isfinite(source_observed) or source_observed < 0.0:
        raise BenchmarkError(
            "Benchmark recipe %s resource metric %s must be finite and nonnegative"
            % (entry.id, metric)
        )
    maximum = float(budget["maximum"])
    tolerance = float(budget["tolerance"])
    conversion = _resource_conversion_from_declaration(
        budget.get("conversion"),
        metric_unit=str(budget["metric_unit"]),
        budget_unit=str(budget["unit"]),
        entry_id=entry.id,
    )
    try:
        unit_contract = evaluate_typed_resource_admission(
            observed=ResourceQuantity(
                source_observed,
                str(budget["metric_unit"]),
            ),
            budget=ResourceQuantity(maximum, str(budget["unit"])),
            tolerance=ResourceQuantity(tolerance, str(budget["unit"])),
            aggregation_policy=str(budget["aggregation_policy"]),
            conversion=conversion,
        )
    except ResourceUnitError as exc:
        raise BenchmarkError(
            "Benchmark recipe %s cannot evaluate typed resource budget: %s"
            % (entry.id, exc)
        ) from exc
    observed = float(unit_contract["conversion"]["transformed_value"])
    excess = max(0.0, observed - maximum)
    admitted = bool(unit_contract["admission"]["admitted"])
    return {
        "admitted": admitted,
        "decision": "admitted" if admitted else "rejected_resource_budget",
        "metric": metric,
        "observed": observed,
        "maximum": maximum,
        "tolerance": tolerance,
        "excess": excess,
        "policy": str(budget["policy"]),
        "unit": str(budget["unit"]),
        "protocol_sha256": str(
            protocol_sha256 or benchmark_protocol_sha256(pack)
        ),
        **(
            {"unit_contract": unit_contract}
            if conversion is not None
            else {}
        ),
    }


def _resource_admission_metrics(admission: Mapping[str, Any]) -> JsonDict:
    return {
        "benchmark.resource_budget.admitted": 1 if admission.get("admitted") else 0,
        "benchmark.resource_budget.observed": float(admission["observed"]),
        "benchmark.resource_budget.maximum": float(admission["maximum"]),
        "benchmark.resource_budget.tolerance": float(admission["tolerance"]),
        "benchmark.resource_budget.excess": float(admission["excess"]),
    }


def _benchmark_matrix_selection(params: Mapping[str, Any], entry_id: str) -> Optional[JsonDict]:
    """Resolve canonical benchmark coordinates with one-way legacy compatibility."""

    has_matrix_selection = "matrix_selection" in params
    has_sweep_values = "sweep_values" in params
    if not has_matrix_selection and not has_sweep_values:
        return None

    matrix_selection: Optional[JsonDict] = None
    sweep_values: Optional[JsonDict] = None
    if has_matrix_selection:
        raw_matrix_selection = params.get("matrix_selection")
        if not isinstance(raw_matrix_selection, Mapping):
            raise BenchmarkError(
                "Benchmark recipe %s params.matrix_selection must be a mapping" % entry_id
            )
        matrix_selection = copy.deepcopy(dict(raw_matrix_selection))
    if has_sweep_values:
        raw_sweep_values = params.get("sweep_values")
        if not isinstance(raw_sweep_values, Mapping):
            raise BenchmarkError(
                "Benchmark recipe %s params.sweep_values must be a mapping" % entry_id
            )
        sweep_values = copy.deepcopy(dict(raw_sweep_values))

    if matrix_selection is not None and sweep_values is not None:
        if canonical_json_sha256(matrix_selection) != canonical_json_sha256(sweep_values):
            raise BenchmarkError(
                "Benchmark recipe %s defines conflicting params.matrix_selection and deprecated "
                "params.sweep_values" % entry_id
            )
        return matrix_selection
    return matrix_selection if matrix_selection is not None else sweep_values


def _collect_summary_metrics(summary: JsonDict) -> JsonDict:
    """Collect unambiguous metrics for legacy internal callers.

    Publication benchmark execution uses
    :func:`_collect_summary_metrics_with_provenance` so every unqualified
    metric is tied to exactly one producer.  This wrapper intentionally omits
    ambiguous unqualified names instead of retaining the first producer in
    plan order.
    """

    metrics, _ = _collect_summary_metrics_with_provenance(summary, [])
    return metrics


def _metric_definitions_for_role(
    metric_definitions: Iterable[Mapping[str, Any]],
    role: str,
) -> List[JsonDict]:
    """Select metrics that the protocol declares meaningful for a recipe role."""

    selected: List[JsonDict] = []
    normalized_role = str(role or "candidate").strip()
    for definition in metric_definitions:
        item = dict(definition)
        raw_roles = item.get("applicable_roles")
        if raw_roles is None or normalized_role in raw_roles:
            selected.append(item)
    return selected


def _collect_summary_metrics_with_provenance(
    summary: Mapping[str, Any],
    metric_definitions: Iterable[Mapping[str, Any]],
    *,
    source_run_evidence_sha256: Optional[str] = None,
) -> tuple[JsonDict, JsonDict]:
    """Return benchmark metrics plus an authoritative producer ledger.

    Step-qualified metrics are always retained.  An unqualified metric is
    admitted only when it has one producer or when the benchmark declaration
    names the producer explicitly.  This prevents an earlier model, source,
    or adapter step from laundering a value under the name of the declared
    evaluator metric.
    """

    report_sha256 = str(source_run_evidence_sha256 or "").strip().lower()
    if not report_sha256:
        # Internal callers that are not building a benchmark bundle still get
        # a content identity for the exact summary used for projection.
        report_sha256 = canonical_json_sha256(dict(summary))
    if len(report_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in report_sha256
    ):
        raise BenchmarkError(
            "source_run_evidence_sha256 must be a lowercase SHA-256"
        )

    metrics: JsonDict = {}
    provenance: JsonDict = {}
    run_metrics = summary.get("metrics")
    if run_metrics is None:
        run_metrics = {}
    if not isinstance(run_metrics, Mapping):
        raise BenchmarkError("Run summary metrics must be a mapping")
    for metric_id, value in run_metrics.items():
        key = str(metric_id)
        metrics[key] = value

    producers: Dict[str, List[JsonDict]] = {}
    for index, raw_step in enumerate(summary.get("steps") or []):
        if not isinstance(raw_step, Mapping):
            raise BenchmarkError("Run summary step %d must be a mapping" % index)
        step_id = str(raw_step.get("id") or "").strip()
        operation_id = str(raw_step.get("op") or "").strip()
        if not step_id:
            raise BenchmarkError("Run summary step %d is missing its id" % index)
        raw_metrics = raw_step.get("metrics")
        if raw_metrics is None:
            raw_metrics = {}
        if not isinstance(raw_metrics, Mapping):
            raise BenchmarkError("Run summary step %s metrics must be a mapping" % step_id)
        raw_outputs = raw_step.get("outputs")
        if raw_outputs is None:
            raw_outputs = {}
        if not isinstance(raw_outputs, Mapping):
            raise BenchmarkError("Run summary step %s outputs must be a mapping" % step_id)
        raw_binding = raw_step.get("execution_binding")
        if not isinstance(raw_binding, Mapping):
            raise BenchmarkError(
                "Run summary step %s is missing its execution binding" % step_id
            )
        binding = dict(raw_binding)
        implementation_metadata = binding.get("implementation_metadata")
        implementation_metadata = (
            dict(implementation_metadata)
            if isinstance(implementation_metadata, Mapping)
            else {}
        )
        implementation_sha256 = str(
            implementation_metadata.get("source_sha256") or ""
        ).strip().lower()
        if raw_metrics and (
            len(implementation_sha256) != 64
            or any(char not in "0123456789abcdef" for char in implementation_sha256)
        ):
            raise BenchmarkError(
                "Run summary step %s has metrics but no content-identified implementation"
                % step_id
            )
        outputs_sha256 = canonical_json_sha256(dict(raw_outputs))
        binding_sha256 = canonical_json_sha256(binding)
        for raw_metric_id, value in raw_metrics.items():
            metric_id = str(raw_metric_id)
            qualified = "steps.%s.%s" % (step_id, metric_id)
            metrics[qualified] = value
            producers.setdefault(metric_id, []).append(
                {
                    "metric_id": metric_id,
                    "source_scope": "step",
                    "source_step": step_id,
                    "source_operation": operation_id,
                    "qualified_metric": qualified,
                    "source_implementation_sha256": implementation_sha256,
                    "source_outputs_sha256": outputs_sha256,
                    "source_execution_binding_sha256": binding_sha256,
                    "source_run_evidence_sha256": report_sha256,
                    "value": value,
                }
            )

    definitions: Dict[str, JsonDict] = {}
    for index, raw_definition in enumerate(metric_definitions):
        if not isinstance(raw_definition, Mapping):
            raise BenchmarkError("benchmark.metrics[%d] must be a mapping" % index)
        definition = dict(raw_definition)
        metric_id = str(definition.get("id") or "").strip()
        if not metric_id:
            raise BenchmarkError("benchmark.metrics[%d].id must be non-empty" % index)
        if metric_id in definitions:
            raise BenchmarkError("benchmark metric id is duplicated: %s" % metric_id)
        definition_version = definition.get("definition_version")
        if (
            isinstance(definition_version, bool)
            or not isinstance(definition_version, int)
            or definition_version < 1
        ):
            raise BenchmarkError(
                "benchmark.metrics[%d].definition_version must be a positive integer"
                % index
            )
        definitions[metric_id] = definition

    # Preserve convenient unqualified access for every metric with exactly one
    # step producer, but never choose among collisions by plan order.
    for metric_id, candidates in producers.items():
        if metric_id not in metrics and len(candidates) == 1:
            selected = dict(candidates[0])
            metrics[metric_id] = selected.pop("value")

    for metric_id, definition in definitions.items():
        declared_step = str(
            definition.get("source_step")
            or definition.get("producer_step")
            or ""
        ).strip()
        declared_operation = str(definition.get("source_operation") or "").strip()
        declared_operations = [
            str(value).strip()
            for value in definition.get("source_operations") or []
            if str(value).strip()
        ]
        candidates = list(producers.get(metric_id) or [])
        if declared_step:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("source_step") == declared_step
            ]
        if declared_operation:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("source_operation") == declared_operation
            ]
        if declared_operations:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("source_operation") in declared_operations
            ]
        if len(candidates) != 1:
            if not candidates:
                qualifier = ""
                if declared_step:
                    qualifier += " from step %s" % declared_step
                if declared_operation:
                    qualifier += " using %s" % declared_operation
                if declared_operations:
                    qualifier += " using one of %s" % ", ".join(declared_operations)
                raise BenchmarkError(
                    "Required benchmark metric %s has no authoritative producer%s"
                    % (metric_id, qualifier)
                )
            raise BenchmarkError(
                "Required benchmark metric %s has %d producers; declare source_step "
                "and source_operation explicitly"
                % (metric_id, len(candidates))
            )
        selected = dict(candidates[0])
        value = selected.pop("value")
        metrics[metric_id] = value
        selected["definition_version"] = definition["definition_version"]
        selected["definition_sha256"] = canonical_json_sha256(definition)
        provenance[metric_id] = selected
    return metrics, provenance


def _write_metrics_csv(path: Path, result: JsonDict) -> None:
    benchmark = dict(result.get("benchmark") or {})
    fieldnames = [
        "benchmark_id",
        "benchmark_version",
        "recipe_id",
        "recipe_label",
        "recipe_role",
        "run_id",
        "status",
        "metric",
        "value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for recipe in _recipes(result):
            metrics = dict(recipe.get("metrics") or {})
            for metric in sorted(metrics):
                writer.writerow(
                    {
                        "benchmark_id": benchmark.get("id") or "",
                        "benchmark_version": benchmark.get("version") or "",
                        "recipe_id": recipe.get("id") or "",
                        "recipe_label": recipe.get("label") or recipe.get("recipe_name") or "",
                        "recipe_role": recipe.get("role") or "",
                        "run_id": recipe.get("run_id") or "",
                        "status": recipe.get("status") or "",
                        "metric": metric,
                        "value": _csv_value(metrics[metric]),
                    }
                )


def _write_recipes_csv(path: Path, result: JsonDict) -> None:
    benchmark = dict(result.get("benchmark") or {})
    metric_columns = list(_summary_metric_columns(result))
    fieldnames = [
        "benchmark_id",
        "benchmark_version",
        "recipe_id",
        "recipe_label",
        "recipe_role",
        "recipe_name",
        "run_id",
        "status",
        *metric_columns,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for recipe in _recipes(result):
            metrics = dict(recipe.get("metrics") or {})
            row = {
                "benchmark_id": benchmark.get("id") or "",
                "benchmark_version": benchmark.get("version") or "",
                "recipe_id": recipe.get("id") or "",
                "recipe_label": recipe.get("label") or recipe.get("recipe_name") or "",
                "recipe_role": recipe.get("role") or "",
                "recipe_name": recipe.get("recipe_name") or "",
                "run_id": recipe.get("run_id") or "",
                "status": recipe.get("status") or "",
            }
            for metric in metric_columns:
                row[metric] = _csv_value(metrics.get(metric))
            writer.writerow(row)


def _write_summary_markdown(path: Path, result: JsonDict) -> None:
    benchmark = dict(result.get("benchmark") or {})
    title = benchmark.get("name") or benchmark.get("id") or "Benchmark"
    metric_columns = list(_summary_metric_columns(result))
    headers = ["Recipe", "Role", "Status", "Run", *metric_columns]
    lines = [
        "# %s" % title,
        "",
        "- Benchmark: `%s`" % (benchmark.get("id") or ""),
        "- Version: `%s`" % (benchmark.get("version") or ""),
        "- Status: `%s`" % (result.get("status") or ""),
        "- Created: `%s`" % (result.get("created_at_utc") or ""),
    ]
    dataset = dict(benchmark.get("dataset") or {})
    task = dict(benchmark.get("task") or {})
    if dataset.get("id"):
        lines.append("- Dataset: `%s`" % dataset["id"])
    if task.get("id"):
        lines.append("- Task: `%s`" % task["id"])
    if result.get("completed_at_utc"):
        lines.append("- Completed: `%s`" % result["completed_at_utc"])
    if result.get("error"):
        lines.append("- Error: `%s`" % result["error"])
    plots = _plots(result)
    if plots:
        lines.extend(["", "## Figures", ""])
        for plot in plots:
            label = str(plot.get("plot") or plot.get("id") or "benchmark plot").replace("-", " ").title()
            image_ref = _markdown_cell(plot.get("relative_path") or plot.get("path") or "")
            data_ref = _markdown_cell(plot.get("data_csv_relative_path") or plot.get("data_csv_path") or "")
            if image_ref:
                if Path(str(image_ref)).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif"}:
                    lines.append("![%s](%s)" % (label, image_ref))
                else:
                    lines.append("[%s](%s)" % (label, image_ref))
                if data_ref:
                    lines.append("")
                    lines.append("Plotted data: `%s`" % data_ref)
                    lines.append("")
    lines.extend(["", "| %s |" % " | ".join(headers), "| %s |" % " | ".join(["---"] * len(headers))])
    for recipe in _recipes(result):
        metrics = dict(recipe.get("metrics") or {})
        row = [
            _markdown_cell(recipe.get("label") or recipe.get("recipe_name") or recipe.get("id") or ""),
            _markdown_cell(recipe.get("role") or ""),
            _markdown_cell(recipe.get("status") or ""),
            _markdown_cell(recipe.get("run_id") or ""),
        ]
        for metric in metric_columns:
            row.append(_markdown_cell(_display_value(metrics.get(metric))))
        lines.append("| %s |" % " | ".join(row))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_metric_columns(result: JsonDict) -> Iterable[str]:
    preferred = [
        "quality.psnr_db",
        "quality.mse",
        "quality.mae",
        "semantic.lexical_similarity",
        "text.unigram_bleu_proxy",
        "text.edit_similarity",
        "text.exact_match",
        "task.accuracy",
        "task.exact_match",
        "classification.balanced_accuracy",
        "vqa.single_reference_exact_match",
        "detection.f1_at_iou_0p5",
        "segmentation.miou",
        "caption.unigram_bleu_proxy",
        "caption.lexical_similarity",
        "retrieval.recall_at_1",
        "retrieval.recall_at_5",
        "retrieval.recall_at_10",
        "channel.snr_db",
        "channel.packet_success_rate",
        "channel.outage_rate",
        "channel.transmitted_bit_count",
        "channel.payload_bit_count",
        "channel.channel_use_count",
        "channel.uses_per_pixel",
        "channel.code_rate",
        "channel.bits_per_symbol",
        "rate.payload_bpp",
        "rate.framed_bpp",
        "rate.coded_bpp",
        "rate.padded_bpp",
        "codec.bit_count",
        "codec.bytes",
        "memory.run.peak_rss_bytes",
    ]
    present = set()
    for recipe in _recipes(result):
        present.update(dict(recipe.get("metrics") or {}).keys())
    columns = [metric for metric in preferred if metric in present]
    if not columns:
        columns = sorted(present)[:12]
    return columns


def _recipes(result: JsonDict) -> List[JsonDict]:
    rows = result.get("recipes") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _plots(result: JsonDict) -> List[JsonDict]:
    rows = result.get("plots") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _validate_research_compatibility(pack: BenchmarkPack, entry: BenchmarkRecipe, specs: JsonDict) -> None:
    pack_dataset = str((pack.dataset or {}).get("id") or "")
    recipe_dataset = str((specs.get("dataset") or {}).get("id") or "")
    if pack_dataset and recipe_dataset and pack_dataset != recipe_dataset:
        raise BenchmarkError(
            "Benchmark %s expects dataset %s but recipe %s uses %s"
            % (pack.id, pack_dataset, entry.id, recipe_dataset)
        )
    pack_task = str((pack.task or {}).get("id") or "")
    recipe_task = str((specs.get("task") or {}).get("id") or "")
    if pack_task and recipe_task and pack_task != recipe_task:
        raise BenchmarkError(
            "Benchmark %s expects task %s but recipe %s uses %s"
            % (pack.id, pack_task, entry.id, recipe_task)
        )
    expected_samples = [item.lower() for item in _benchmark_sample_ids(pack)]
    recipe_params = dict((specs.get("dataset") or {}).get("params") or {})
    actual_samples = [
        str(item).strip().lower()
        for item in (recipe_params.get("image_ids") or [])
        if str(item).strip()
    ]
    if expected_samples and actual_samples != expected_samples:
        raise BenchmarkError(
            "Benchmark %s requires the exact held-out sample order %s but recipe %s uses %s"
            % (pack.id, expected_samples, entry.id, actual_samples)
        )


def _validate_recipe_training_lineage(
    pack: BenchmarkPack,
    entry: BenchmarkRecipe,
    recipe: Any,
    *,
    recipe_path: Path,
    project_root: Path,
    registry: Optional[OperationRegistry] = None,
    allowed_manifest_paths: Optional[set[Path]] = None,
) -> None:
    """Reject returned models whose fitting population overlaps benchmark test IDs."""

    test_ids = {item.lower() for item in _benchmark_sample_ids(pack)}
    require_lineage = bool(
        (pack.metadata or {}).get("require_disjoint_training_lineage", False)
    )
    require_publication_readiness = _benchmark_traceability_profile_requested(pack)
    if not test_ids and not require_lineage and not require_publication_readiness:
        return
    manifest_paths: List[Path] = []
    for step in getattr(recipe, "steps", []) or []:
        params = dict(getattr(step, "params", {}) or {})
        runtime = str(params.get("runtime") or "").strip()
        raw_paths: List[str] = []

        def collect_manifest_paths(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if (
                        key == "artifact_manifest_path"
                        and isinstance(child, str)
                        and child.strip()
                    ):
                        raw_paths.append(child)
                    else:
                        collect_manifest_paths(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect_manifest_paths(child)

        collect_manifest_paths(params)
        if registry is not None:
            try:
                operation_description = registry.get(
                    str(getattr(step, "op", "") or "")
                ).describe()
            except Exception as exc:
                if require_lineage:
                    raise BenchmarkEvidenceError(
                        "benchmark recipe %s cannot inspect training lineage for operation %s: %s"
                        % (
                            entry.id,
                            str(getattr(step, "op", "") or ""),
                            exc,
                        )
                    ) from exc
                operation_description = {}
            external = operation_description.get("external_adapter")
            if external is not None and not isinstance(external, Mapping):
                if require_lineage:
                    raise BenchmarkEvidenceError(
                        "benchmark recipe %s operation %s has malformed external-adapter lineage metadata"
                        % (entry.id, str(getattr(step, "op", "") or ""))
                    )
                external = None
            if isinstance(external, Mapping):
                external_training = external.get("training")
                external_training = (
                    external_training
                    if isinstance(external_training, Mapping)
                    else {}
                )
                external_manifest = external_training.get("artifact_manifest_path")
                if isinstance(external_manifest, str) and external_manifest.strip():
                    raw_paths.append(external_manifest)
                elif require_lineage:
                    raise BenchmarkEvidenceError(
                        "benchmark recipe %s uses checkpoint-backed external adapter %s "
                        "without a portable trained artifact manifest; publication benchmarks "
                        "must use learned_artifact evidence"
                        % (entry.id, str(getattr(step, "op", "") or ""))
                    )
        if not raw_paths:
            if require_lineage and runtime == "learned_artifact":
                raise BenchmarkEvidenceError(
                    "benchmark recipe %s uses a learned artifact without artifact_manifest_path"
                    % entry.id
                )
            continue
        for raw_path in raw_paths:
            path = _resolve_lineage_manifest_path(
                str(raw_path),
                recipe_path=recipe_path,
                pack=pack,
                project_root=project_root,
            )
            if allowed_manifest_paths is not None and require_lineage:
                allowed = {candidate.resolve() for candidate in allowed_manifest_paths}
                if path.resolve() not in allowed:
                    raise BenchmarkEvidenceError(
                        "trained artifact %s is not bound to the benchmark training-evidence snapshot"
                        % path
                    )
            if path not in manifest_paths:
                manifest_paths.append(path)
    for manifest_path in manifest_paths:
        validate_trained_artifact_lineage_manifest(
            pack,
            manifest_path,
            require_lineage=require_lineage,
        )
        if require_publication_readiness:
            try:
                validate_trained_artifact_publication_readiness(
                    manifest_path,
                    project_root=project_root,
                    registry=registry,
                )
            except TrainedArtifactError as exc:
                raise BenchmarkEvidenceError(
                    "strongest-profile benchmark %s rejects trained artifact %s: %s"
                    % (pack.id, manifest_path, exc)
                ) from exc


def validate_trained_artifact_lineage_manifest(
    pack: BenchmarkPack,
    manifest_path: Path,
    *,
    require_lineage: Optional[bool] = None,
) -> JsonDict:
    """Validate declared fitting IDs and bytes against a frozen held-out set."""

    artifact = _load_mapping(manifest_path)
    training = dict(artifact.get("training") or {})
    partitions = dict(training.get("data_partitions") or {})
    if require_lineage is None:
        require_lineage = bool(
            (pack.metadata or {}).get("require_disjoint_training_lineage", False)
        )
    if _truthy_lineage_flag(training.get("test_images_used")) or _truthy_lineage_flag(
        partitions.get("test_images_used")
    ):
        raise BenchmarkEvidenceError(
            "trained artifact %s declares test_images_used=true" % manifest_path
        )

    partition_ids: Dict[str, List[str]] = {}
    for key, value in partitions.items():
        if str(key).endswith("_image_ids"):
            partition_ids[str(key)] = _lineage_ids(value)
    train_ids = partition_ids.get("train_image_ids", [])
    validation_ids = partition_ids.get("validation_image_ids", [])
    if set(train_ids).intersection(validation_ids):
        raise BenchmarkEvidenceError(
            "trained artifact %s has overlapping train and validation image IDs"
            % manifest_path
        )
    fitting_ids = {
        sample_id for values in partition_ids.values() for sample_id in values
    }
    test_ids = {item.lower() for item in _benchmark_sample_ids(pack)}
    overlap = sorted(test_ids.intersection(fitting_ids))
    if overlap:
        raise BenchmarkEvidenceError(
            "trained artifact %s overlaps benchmark test samples: %s"
            % (manifest_path, ", ".join(overlap))
        )

    heldout_hashes = _benchmark_sample_hashes(pack)
    heldout_lineage = _benchmark_lineage_population(pack, require_complete=False)
    data_contract_path = _artifact_data_contract_path(artifact, manifest_path)
    contract_ids: set[str] = set()
    contract_hashes: set[str] = set()
    contract_lineage = _empty_lineage_population()
    if data_contract_path is not None:
        contract = _load_mapping(data_contract_path)
        contract_ids, contract_hashes = _fitting_data_contract_population(
            contract, data_contract_path
        )
        contract_lineage = _fitting_data_contract_lineage_population(
            contract,
            data_contract_path,
            require_complete=bool(heldout_lineage["complete"]),
        )
        contract_overlap = sorted(test_ids.intersection(contract_ids))
        hash_overlap = sorted(set(heldout_hashes.values()).intersection(contract_hashes))
        identity_overlap = sorted(
            heldout_lineage["identity_ids"].intersection(
                contract_lineage["identity_ids"]
            )
        )
        group_overlap = sorted(
            heldout_lineage["group_ids"].intersection(
                contract_lineage["group_ids"]
            )
        )
        source_overlap = sorted(
            heldout_lineage["source_sha256"].intersection(
                contract_lineage["source_sha256"]
            )
        )
        post_transform_overlap = sorted(
            heldout_lineage["post_transform_sha256"].intersection(
                contract_lineage["post_transform_sha256"]
            )
        )
        if (
            contract_overlap
            or hash_overlap
            or identity_overlap
            or group_overlap
            or source_overlap
            or post_transform_overlap
        ):
            details = (
                contract_overlap
                + ["sha256:%s" % item for item in hash_overlap]
                + ["lineage:%s" % item for item in identity_overlap]
                + ["group:%s" % item for item in group_overlap]
                + ["source_sha256:%s" % item for item in source_overlap]
                + ["post_transform_sha256:%s" % item for item in post_transform_overlap]
            )
            raise BenchmarkEvidenceError(
                "trained artifact %s data contract overlaps benchmark test content or ancestry: %s"
                % (manifest_path, ", ".join(details))
            )
        if fitting_ids and contract_ids and fitting_ids != contract_ids:
            raise BenchmarkEvidenceError(
                "trained artifact %s data_partitions do not match its hashed data contract"
                % manifest_path
            )
    if require_lineage:
        if not fitting_ids:
            raise BenchmarkEvidenceError(
                "trained artifact %s does not declare fitting sample IDs"
                % manifest_path
            )
        if heldout_hashes and (data_contract_path is None or not contract_hashes):
            raise BenchmarkEvidenceError(
                "trained artifact %s requires a hashed fitting-data contract for "
                "content-level held-out validation" % manifest_path
            )
        if bool(heldout_lineage["complete"]) and not bool(
            contract_lineage["complete"]
        ):
            raise BenchmarkEvidenceError(
                "trained artifact %s requires source/group/transform ancestry in its fitting-data contract"
                % manifest_path
            )
    return {
        "manifest": str(manifest_path),
        "fitting_ids": sorted(fitting_ids),
        "fitting_sha256": sorted(contract_hashes),
        "heldout_ids": sorted(test_ids),
        "heldout_sha256": sorted(heldout_hashes.values()),
        "heldout_lineage": _serializable_lineage_population(heldout_lineage),
        "fitting_lineage": _serializable_lineage_population(contract_lineage),
        "data_contract": str(data_contract_path) if data_contract_path else None,
    }


def _resolve_lineage_manifest_path(
    raw_path: str,
    *,
    recipe_path: Path,
    pack: BenchmarkPack,
    project_root: Path,
) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [
        recipe_path.parent / path,
        project_root / path,
        *(([pack.path.parent / path]) if pack.path is not None else []),
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BenchmarkEvidenceError(
        "trained artifact manifest is missing for lineage validation: %s" % raw_path
    )


def _lineage_ids(value: Any) -> List[str]:
    if isinstance(value, str):
        rows = value.split(",")
    elif isinstance(value, (list, tuple)):
        rows = value
    else:
        return []
    output = [str(item).strip().lower() for item in rows if str(item).strip()]
    if len(output) != len(set(output)):
        raise BenchmarkEvidenceError(
            "trained artifact data partition contains duplicate sample IDs"
        )
    return output


def _truthy_lineage_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
        return True
    return False


def _benchmark_sample_hashes(pack: BenchmarkPack) -> Dict[str, str]:
    dataset_manifest = (pack.dataset or {}).get("manifest")
    dataset_manifest = (
        dataset_manifest if isinstance(dataset_manifest, Mapping) else {}
    )
    files = dataset_manifest.get("files") or dataset_manifest.get("samples")
    files = files if isinstance(files, list) else []
    output: Dict[str, str] = {}
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise BenchmarkEvidenceError(
                "benchmark dataset manifest file %d is invalid" % index
            )
        sample_id = str(row.get("sample_id") or "").strip().lower()
        digest = str(row.get("sha256") or "").strip().lower()
        if not sample_id or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise BenchmarkEvidenceError(
                "benchmark dataset manifest file %d lacks sample_id/SHA-256" % index
            )
        if sample_id in output and output[sample_id] != digest:
            raise BenchmarkEvidenceError(
                "benchmark dataset manifest contains conflicting sample %s" % sample_id
            )
        output[sample_id] = digest
    return output


def _empty_lineage_population() -> Dict[str, Any]:
    return {
        "identity_ids": set(),
        "group_ids": set(),
        "source_sha256": set(),
        "transform_fingerprint_sha256": set(),
        "post_transform_sha256": set(),
        "complete": False,
        "record_count": 0,
    }


def _benchmark_lineage_population(
    pack: BenchmarkPack, *, require_complete: Optional[bool] = None
) -> Dict[str, Any]:
    manifest = (pack.dataset or {}).get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    rows = manifest.get("files") or manifest.get("samples") or []
    if not isinstance(rows, list):
        raise BenchmarkEvidenceError(
            "benchmark dataset manifest files/samples must be a list"
        )
    selected = {item.lower() for item in _benchmark_sample_ids(pack)}
    population = _empty_lineage_population()
    complete = bool(rows)
    seen_samples: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BenchmarkEvidenceError(
                "benchmark dataset manifest lineage record %d is invalid" % index
            )
        sample_id = str(
            row.get("sample_id") or row.get("image_id") or row.get("id") or ""
        ).strip().lower()
        if selected and sample_id not in selected:
            continue
        if sample_id in seen_samples:
            raise BenchmarkEvidenceError(
                "benchmark dataset manifest repeats lineage record for %s"
                % sample_id
            )
        seen_samples.add(sample_id)
        record_complete = _accumulate_lineage_record(
            population,
            row,
            context="benchmark dataset manifest record %d" % index,
            require_complete=False,
        )
        complete = complete and record_complete
    population["complete"] = bool(population["record_count"] and complete)
    if require_complete is None:
        require_complete = _benchmark_traceability_profile_requested(pack)
    if require_complete:
        missing = selected.difference(
            {
                str(row.get("sample_id") or row.get("image_id") or row.get("id") or "")
                .strip()
                .lower()
                for row in rows
                if isinstance(row, Mapping)
            }
        )
        if missing or not population["complete"]:
            raise BenchmarkEvidenceError(
                "strongest-profile benchmark %s requires complete source/group/transform ancestry for every held-out sample"
                % pack.id
            )
    return population


def _fitting_data_contract_lineage_population(
    contract: Mapping[str, Any],
    contract_path: Path,
    *,
    require_complete: bool,
) -> Dict[str, Any]:
    population = _empty_lineage_population()
    complete = True
    raw_splits = contract.get("splits")
    if not isinstance(raw_splits, list):
        return population
    for split_index, split in enumerate(raw_splits):
        if not isinstance(split, Mapping):
            continue
        rows = split.get("files")
        if not isinstance(rows, list):
            continue
        for file_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            record_complete = _accumulate_lineage_record(
                population,
                row,
                context="training data contract %s split %d file %d"
                % (contract_path, split_index, file_index),
                require_complete=require_complete,
            )
            complete = complete and record_complete
    population["complete"] = bool(population["record_count"] and complete)
    return population


def _accumulate_lineage_record(
    population: Dict[str, Any],
    row: Mapping[str, Any],
    *,
    context: str,
    require_complete: bool,
) -> bool:
    sample_id = str(
        row.get("sample_id") or row.get("image_id") or row.get("id") or ""
    ).strip().lower()
    source_id = str(row.get("source_id") or "").strip().lower()
    group_id = str(row.get("group_id") or "").strip().lower()
    ancestry = row.get("ancestry_ids") or row.get("parent_ids")
    ancestry_ids = (
        [str(item).strip().lower() for item in ancestry if str(item).strip()]
        if isinstance(ancestry, list)
        else []
    )
    source_sha = str(row.get("source_sha256") or "").strip().lower()
    transform_sha = str(
        row.get("transform_fingerprint_sha256") or ""
    ).strip().lower()
    post_transform_sha = str(
        row.get("post_transform_sha256") or ""
    ).strip().lower()
    complete = bool(
        sample_id
        and source_id
        and group_id
        and ancestry_ids
        and _is_sha256(source_sha)
        and _is_sha256(transform_sha)
    )
    transform = row.get("transform")
    if isinstance(transform, Mapping) and _is_sha256(transform_sha):
        if canonical_json_sha256(dict(transform)) != transform_sha:
            raise BenchmarkEvidenceError(
                "%s transform_fingerprint_sha256 does not match transform" % context
            )
    if require_complete and not complete:
        raise BenchmarkEvidenceError(
            "%s requires sample_id, source_id, group_id, ancestry_ids, source_sha256, and transform_fingerprint_sha256"
            % context
        )
    population["record_count"] += 1
    population["identity_ids"].update(
        item for item in [sample_id, source_id, *ancestry_ids] if item
    )
    if group_id:
        population["group_ids"].add(group_id)
    if _is_sha256(source_sha):
        population["source_sha256"].add(source_sha)
    if _is_sha256(transform_sha):
        population["transform_fingerprint_sha256"].add(transform_sha)
    if _is_sha256(post_transform_sha):
        population["post_transform_sha256"].add(post_transform_sha)
    return complete


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _serializable_lineage_population(population: Mapping[str, Any]) -> JsonDict:
    return {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in population.items()
    }


def _artifact_data_contract_path(
    artifact: Mapping[str, Any], manifest_path: Path
) -> Optional[Path]:
    source = artifact.get("source")
    source = source if isinstance(source, Mapping) else {}
    reference = source.get("data_contract")
    if not isinstance(reference, Mapping) or not reference.get("path"):
        return None
    relative = Path(str(reference.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkEvidenceError(
            "trained artifact %s data contract path must be manifest-relative"
            % manifest_path
        )
    candidate = (manifest_path.parent / relative).resolve()
    if not candidate.is_file():
        raise BenchmarkEvidenceError(
            "trained artifact %s data contract is missing: %s"
            % (manifest_path, relative)
        )
    expected = str(reference.get("file_sha256") or "").strip().lower()
    actual = file_sha256(candidate)
    if len(expected) != 64 or expected != actual:
        raise BenchmarkEvidenceError(
            "trained artifact %s data contract SHA-256 is invalid" % manifest_path
        )
    return candidate


def _fitting_data_contract_population(
    contract: Mapping[str, Any], contract_path: Path
) -> tuple[set[str], set[str]]:
    raw_splits = contract.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise BenchmarkEvidenceError(
            "training data contract %s requires explicit fitting splits" % contract_path
        )
    sample_ids: set[str] = set()
    hashes: set[str] = set()
    for split_index, split in enumerate(raw_splits):
        if not isinstance(split, Mapping):
            raise BenchmarkEvidenceError(
                "training data contract %s split %d is invalid"
                % (contract_path, split_index)
            )
        files = split.get("files")
        if not isinstance(files, list):
            raise BenchmarkEvidenceError(
                "training data contract %s split %d has no file inventory"
                % (contract_path, split_index)
            )
        for file_index, row in enumerate(files):
            if not isinstance(row, Mapping):
                raise BenchmarkEvidenceError(
                    "training data contract %s split %d file %d is invalid"
                    % (contract_path, split_index, file_index)
                )
            sample_id = str(
                row.get("image_id") or row.get("sample_id") or row.get("id") or ""
            ).strip().lower()
            digest = str(row.get("sha256") or "").strip().lower()
            if not sample_id or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise BenchmarkEvidenceError(
                    "training data contract %s split %d file %d lacks an ID/SHA-256"
                    % (contract_path, split_index, file_index)
                )
            if sample_id in sample_ids:
                raise BenchmarkEvidenceError(
                    "training data contract %s repeats fitting sample %s"
                    % (contract_path, sample_id)
                )
            sample_ids.add(sample_id)
            hashes.add(digest)
    return sample_ids, hashes


def _benchmark_tier(pack: BenchmarkPack) -> str:
    metadata = dict(pack.metadata or {})
    canonical = metadata.get("benchmark_tier")
    legacy = metadata.get("tier")
    if (
        canonical not in (None, "")
        and legacy not in (None, "")
        and str(canonical).strip().lower() != str(legacy).strip().lower()
    ):
        raise BenchmarkError(
            "Benchmark %s metadata.tier conflicts with metadata.benchmark_tier"
            % pack.id
        )
    tier = str(canonical or legacy or "smoke").strip().lower()
    return tier or "smoke"


def _validate_baseline_roster(pack: BenchmarkPack) -> None:
    declared = [str(value).strip() for value in pack.baselines if str(value).strip()]
    if len(declared) != len(set(declared)):
        raise BenchmarkError(
            "Benchmark %s baselines must be unique" % pack.id
        )
    linked = set()
    for entry in pack.recipes:
        if str(entry.id or "").strip():
            linked.add(str(entry.id).strip())
        params = entry.params if isinstance(entry.params, Mapping) else {}
        for key in ("baseline_id", "method_id"):
            value = params.get(key)
            if value not in (None, ""):
                linked.add(str(value).strip())
    unlinked = [baseline for baseline in declared if baseline not in linked]
    if unlinked:
        raise BenchmarkError(
            "Benchmark %s declares baselines that are not linked to any "
            "recipe id, params.baseline_id, or params.method_id: %s"
            % (pack.id, ", ".join(unlinked))
        )


def _validate_benchmark_metadata(pack: BenchmarkPack) -> None:
    metadata = dict(pack.metadata or {})
    dataset = dict(pack.dataset or {})
    _validate_baseline_roster(pack)
    tier = _benchmark_tier(pack)
    profile_requested = _benchmark_traceability_profile_requested(pack)
    if tier not in {"smoke", "canonical", "experimental"}:
        raise BenchmarkError(
            "Benchmark %s metadata.benchmark_tier must be one of smoke, canonical, experimental"
            % pack.id
        )
    if profile_requested and tier != "canonical":
        raise BenchmarkError(
            "Strongest-profile benchmark %s must use metadata.benchmark_tier=canonical"
            % pack.id
        )
    if (
        profile_requested
        and metadata.get("verification_profile")
        != publication_verification_profile_binding()
    ):
        raise BenchmarkError(
            "Strongest-profile benchmark %s must bind the exact "
            "%s verification profile"
            % (
                pack.id,
                publication_verification_profile_binding()["id"],
            )
        )
    selection_role = str(dataset.get("selection_role") or "").strip()
    access_policy = dataset.get("access_policy")
    if access_policy is not None and not isinstance(access_policy, Mapping):
        raise BenchmarkError(
            "Benchmark %s dataset.access_policy must be a mapping" % pack.id
        )
    access_policy = dict(access_policy or {})
    if metadata.get("adaptive_policy") and selection_role in {
        "publication_test",
        "sealed_test",
    }:
        raise BenchmarkError(
            "Benchmark %s cannot tune an adaptive policy on its publication-test population"
            % pack.id
        )
    if bool(access_policy.get("publication_test")):
        if selection_role not in {"publication_test", "sealed_test"}:
            raise BenchmarkError(
                "Benchmark %s publication-test access requires dataset.selection_role=publication_test"
                % pack.id
            )
        if access_policy.get("state") != "sealed_single_access":
            raise BenchmarkError(
                "Benchmark %s publication-test population must use sealed_single_access"
                % pack.id
            )
        for key in ("access_ledger", "access_budget", "seal_sha256"):
            if access_policy.get(key) in (None, "", []):
                raise BenchmarkError(
                    "Benchmark %s publication-test access_policy.%s is required"
                    % (pack.id, key)
                )
        if access_policy.get("access_budget") != 1:
            raise BenchmarkError(
                "Benchmark %s publication-test access budget must be exactly one"
                % pack.id
            )
    if profile_requested:
        _validate_publication_common_conditions(pack)
        if not bool(access_policy.get("publication_test")):
            raise BenchmarkError(
                "Strongest-profile benchmark %s requires a sealed publication-test population"
                % pack.id
            )
        if not isinstance(dataset.get("preprocessing"), Mapping) or not dataset.get(
            "preprocessing"
        ):
            raise BenchmarkError(
                "Strongest-profile benchmark %s must bind dataset.preprocessing"
                % pack.id
            )
        if not _benchmark_sample_ids(pack):
            raise BenchmarkError(
                "Strongest-profile benchmark %s must bind an ordered held-out sample_ids list"
                % pack.id
            )
        if not isinstance(dataset.get("source_bindings"), list) or not dataset.get(
            "source_bindings"
        ):
            raise BenchmarkError(
                "Strongest-profile benchmark %s must declare dataset.source_bindings"
                % pack.id
            )
        try:
            _benchmark_lineage_population(pack, require_complete=True)
        except BenchmarkEvidenceError as exc:
            raise BenchmarkError(str(exc)) from exc
        if metadata.get("require_identical_source_transform") is not True:
            raise BenchmarkError(
                "Strongest-profile benchmark %s must require one identical source transform"
                % pack.id
            )
        if not isinstance(metadata.get("resource_budget"), Mapping):
            raise BenchmarkError(
                "Strongest-profile benchmark %s must enforce a resource budget"
                % pack.id
            )
        if metadata.get("require_disjoint_training_lineage") is not True:
            raise BenchmarkError(
                "Strongest-profile benchmark %s must require disjoint training lineage"
                % pack.id
            )
        for index, metric in enumerate(pack.metrics):
            version = metric.get("definition_version")
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version < 1
            ):
                raise BenchmarkError(
                    "Strongest-profile benchmark %s metric %d must have a positive definition_version"
                    % (pack.id, index)
                )
            if not str(
                metric.get("source_step") or metric.get("producer_step") or ""
            ).strip():
                raise BenchmarkError(
                    "Strongest-profile benchmark %s metric %s must declare source_step"
                    % (pack.id, metric.get("id") or index)
                )
            if not str(metric.get("source_operation") or "").strip() and not (
                isinstance(metric.get("source_operations"), list)
                and metric.get("source_operations")
            ):
                raise BenchmarkError(
                    "Strongest-profile benchmark %s metric %s must declare source_operation(s)"
                    % (pack.id, metric.get("id") or index)
                )
    if tier != "canonical":
        return
    missing = []
    for key in ("protocol_id", "protocol_version", "dataset_split", "channel", "rate_accounting", "expected_outputs"):
        if not metadata.get(key):
            missing.append("metadata.%s" % key)
    if metadata.get("frozen") is not True:
        missing.append("metadata.frozen=true")
    if missing:
        raise BenchmarkError(
            "Canonical benchmark %s is missing required protocol metadata: %s"
            % (pack.id, ", ".join(missing))
        )


def _validate_publication_common_conditions(pack: BenchmarkPack) -> None:
    """Require explicit fairness declarations for the strongest local profile.

    This is a declaration gate. Individual operation contracts and result evidence
    remain responsible for proving that a run materialized the declared semantics.
    """

    conditions = (pack.metadata or {}).get("common_conditions")
    if not isinstance(conditions, Mapping):
        raise BenchmarkError(
            "Strongest-profile benchmark %s must declare metadata.common_conditions"
            % pack.id
        )
    requirements = {
        "power": ("coordinate", "normalization_scope", "target"),
        "randomness": (
            "pairing_keys",
            "seed_derivation",
            "paired_operation_ids",
        ),
        "receiver": ("channel_state_information", "receiver_processing"),
        "failure": ("outage_definition", "decode_failure_policy", "denominator_policy"),
    }
    for section, fields in requirements.items():
        value = conditions.get(section)
        if not isinstance(value, Mapping):
            raise BenchmarkError(
                "Strongest-profile benchmark %s common_conditions.%s must be a mapping"
                % (pack.id, section)
            )
        for field_name in fields:
            field_value = value.get(field_name)
            if field_name in {"pairing_keys", "paired_operation_ids"}:
                valid = (
                    isinstance(field_value, list)
                    and bool(field_value)
                    and all(
                        isinstance(item, str) and item.strip()
                        for item in field_value
                    )
                    and len(field_value) == len(set(field_value))
                )
            elif field_name == "target":
                valid = (
                    isinstance(field_value, (int, float))
                    and not isinstance(field_value, bool)
                    and math.isfinite(float(field_value))
                    and float(field_value) > 0.0
                )
            else:
                valid = isinstance(field_value, str) and bool(field_value.strip())
            if not valid:
                raise BenchmarkError(
                    "Strongest-profile benchmark %s common_conditions.%s.%s is required"
                    % (pack.id, section, field_name)
                )
    power = dict(conditions.get("power") or {})
    if str(power.get("normalization_scope") or "").strip() != "source_item":
        raise BenchmarkError(
            "Strongest-profile benchmark %s must normalize power at source_item scope"
            % pack.id
        )
    pairing_keys = {
        str(item).strip().lower()
        for item in dict(conditions.get("randomness") or {}).get("pairing_keys") or []
    }
    if not any("source_item" in item for item in pairing_keys) or not any(
        "replicate" in item or "seed" in item for item in pairing_keys
    ):
        raise BenchmarkError(
            "Strongest-profile benchmark %s randomness pairing_keys must include source-item identity and replicate/seed identity"
            % pack.id
        )


def _validate_publication_source_bindings(
    pack: BenchmarkPack, registry: OperationRegistry
) -> None:
    """Prove that a strongest-profile source binding names every operation parameter."""

    if not _benchmark_traceability_profile_requested(pack):
        return
    dataset = dict(pack.dataset or {})
    raw_bindings = dataset.get("source_bindings")
    if not isinstance(raw_bindings, list):
        return
    preprocessing = dataset.get("preprocessing")
    preprocessing = preprocessing if isinstance(preprocessing, Mapping) else {}
    shared_params = preprocessing.get("operation_params")
    shared_params = shared_params if isinstance(shared_params, Mapping) else {}
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            continue
        operation = str(raw.get("operation") or "").strip()
        try:
            description = registry.get(operation).describe()
        except Exception as exc:
            raise BenchmarkError(
                "Strongest-profile benchmark %s cannot resolve source binding operation %s"
                % (pack.id, operation)
            ) from exc
        schema = description.get("params_schema")
        schema = schema if isinstance(schema, Mapping) else {}
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        declared = set(dict(raw.get("params") or {}))
        declared.update(str(key) for key in shared_params)
        declared.add(str(raw.get("selection_param") or ""))
        dataset_param = str(raw.get("dataset_param") or "dataset").strip()
        if dataset_param:
            declared.add(dataset_param)
        declared.discard("")
        expected = {str(key) for key in properties}
        missing = sorted(expected.difference(declared))
        unknown = sorted(declared.difference(expected))
        if missing or unknown:
            details = []
            if missing:
                details.append("missing %s" % ", ".join(missing))
            if unknown:
                details.append("unknown %s" % ", ".join(unknown))
            raise BenchmarkError(
                "Strongest-profile benchmark %s source binding %d does not close the complete %s parameter set: %s"
                % (pack.id, index, operation, "; ".join(details))
            )


def _load_mapping(path: Path) -> JsonDict:
    try:
        data = load_strict_yaml_or_json(path)
    except StructuredInputError as exc:
        raise BenchmarkError(
            "Invalid structured YAML/JSON input %s: %s" % (path, exc)
        ) from exc
    if not isinstance(data, Mapping):
        raise BenchmarkError(
            "Structured YAML/JSON input must contain a mapping at the top level: %s"
            % path
        )
    return dict(data)


def _recipe_from_mapping(value: Any, index: int) -> BenchmarkRecipe:
    data = _require_mapping(value, "benchmark.recipes[%d]" % index)
    path_value = _required_string(data, "path", "benchmark.recipes[%d]" % index)
    recipe_id = str(data.get("id") or Path(path_value).stem)
    return BenchmarkRecipe(
        id=recipe_id,
        path=Path(path_value),
        label=_optional_string(data, "label", "benchmark.recipes[%d]" % index),
        role=str(data.get("role") or "candidate"),
        params=dict(data.get("params") or {}),
    )


def _metric_list(value: Any) -> List[JsonDict]:
    if not isinstance(value, list):
        raise BenchmarkError("benchmark.metrics must be a list")
    metrics = []
    seen_ids = set()
    for index, item in enumerate(value):
        if isinstance(item, str):
            metric = {"id": item}
        elif isinstance(item, Mapping):
            metric = dict(item)
        else:
            raise BenchmarkError("benchmark.metrics[%d] must be a string or mapping" % index)
        metric_id = metric.get("id")
        if not isinstance(metric_id, str) or not metric_id.strip():
            raise BenchmarkError("benchmark.metrics[%d].id must be a non-empty string" % index)
        metric_id = metric_id.strip()
        if metric_id in seen_ids:
            raise BenchmarkError("benchmark metric id is duplicated: %s" % metric_id)
        seen_ids.add(metric_id)
        metric["id"] = metric_id
        definition_version = metric.get(
            "definition_version", BENCHMARK_METRIC_DEFINITION_VERSION
        )
        if (
            isinstance(definition_version, bool)
            or not isinstance(definition_version, int)
            or definition_version < 1
        ):
            raise BenchmarkError(
                "benchmark.metrics[%d].definition_version must be a positive integer"
                % index
            )
        metric["definition_version"] = definition_version
        for field_name in ("source_step", "producer_step", "source_operation"):
            field_value = metric.get(field_name)
            if field_value is not None and (
                not isinstance(field_value, str) or not field_value.strip()
            ):
                raise BenchmarkError(
                    "benchmark.metrics[%d].%s must be a non-empty string"
                    % (index, field_name)
                )
        source_operations = metric.get("source_operations")
        if source_operations is not None:
            if (
                not isinstance(source_operations, list)
                or not source_operations
                or not all(
                    isinstance(value, str) and value.strip()
                    for value in source_operations
                )
                or len(source_operations) != len(set(source_operations))
            ):
                raise BenchmarkError(
                    "benchmark.metrics[%d].source_operations must be a unique, non-empty string list"
                    % index
                )
        applicable_roles = metric.get("applicable_roles")
        if applicable_roles is not None:
            if (
                not isinstance(applicable_roles, list)
                or not applicable_roles
                or not all(
                    isinstance(role, str) and role.strip()
                    for role in applicable_roles
                )
                or len({role.strip() for role in applicable_roles})
                != len(applicable_roles)
            ):
                raise BenchmarkError(
                    "benchmark.metrics[%d].applicable_roles must be a non-empty unique string list"
                    % index
                )
            metric["applicable_roles"] = [
                role.strip() for role in applicable_roles
            ]
        if metric.get("source_operation") and source_operations:
            raise BenchmarkError(
                "benchmark.metrics[%d] cannot define both source_operation and source_operations"
                % index
            )
        if metric.get("source_step") and metric.get("producer_step"):
            if metric["source_step"] != metric["producer_step"]:
                raise BenchmarkError(
                    "benchmark.metrics[%d] defines conflicting source_step and producer_step"
                    % index
                )
        metrics.append(metric)
    return metrics


def _require_mapping(value: Any, label: str) -> JsonDict:
    if not isinstance(value, Mapping):
        raise BenchmarkError("%s must be a mapping" % label)
    return dict(value)


def _required_string(data: JsonDict, key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise BenchmarkError("%s.%s must be a non-empty string" % (label, key))
    return value


def _optional_string(data: JsonDict, key: str, label: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkError("%s.%s must be a string" % (label, key))
    return value


def _string_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list):
        raise BenchmarkError("%s must be a list" % label)
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise BenchmarkError("%s[%d] must be a non-empty string" % (label, index))
        output.append(item)
    return output
