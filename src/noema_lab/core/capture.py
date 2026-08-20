from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from noema_lab.core.executor import LocalExecutor, compile_effective_recipe_for_runner
from noema_lab.core.matrix import matrix_variant_id
from noema_lab.core.operations import OperationError, OperationRegistry
from noema_lab.core.planner import plan_recipe
from noema_lab.core.recipes import Recipe, recipe_from_dict
from noema_lab.core.reproducibility import (
    SEED_MODULUS,
    master_seed_from_recipe,
    recipe_fingerprint,
    seed_namespace_from_recipe,
    seed_policy,
    utc_now_iso,
)
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    decode_strict_json_object,
    decode_strict_yaml_or_json,
)
from noema_lab.core.variants import (
    RecipeVariantPlan,
    plan_recipe_variants,
    prepare_compiled_single_run_recipe,
)

JsonDict = Dict[str, Any]

# Dataset capture is intentionally finite even when authored from the UI.  The
# limits are high enough for the documented 100k-sample workflows while
# preventing a typo from creating billions of records, runs, or one enormous
# in-memory shard.
MAX_CAPTURE_SAMPLES = 1_000_000
MAX_CAPTURE_SHARD_SIZE = 100_000
MAX_CAPTURE_RUNS = 100_000
DEFAULT_CAPTURE_SHARD_SIZE = 4096
CAPTURE_DISK_RESERVE_BYTES = 256 * 1024 * 1024


class DatasetCaptureError(ValueError):
    pass


def validate_dataset_capture_contract(
    recipe: Recipe,
    registry: OperationRegistry,
) -> JsonDict:
    """Validate and normalize capture settings without executing the recipe."""

    config = _dataset_capture_config(recipe)
    steps_by_id = {step.id: step for step in recipe.steps}
    for tap in config["taps"]:
        tap_id = str(tap["id"])
        reference = str(tap["from"])
        step_id, output_name = reference.split(".", 1)
        step = steps_by_id.get(step_id)
        if step is None:
            raise DatasetCaptureError(
                "Dataset Capture tap %s references unknown step %s"
                % (tap_id, step_id)
            )
        try:
            operation = registry.get(step.op)
        except OperationError as exc:
            raise DatasetCaptureError(
                "Dataset Capture tap %s references unavailable operation %s on step %s"
                % (tap_id, step.op, step_id)
            ) from exc
        if output_name not in dict(operation.output_kinds or {}):
            raise DatasetCaptureError(
                "Dataset Capture tap %s references unknown output %s"
                % (tap_id, reference)
            )
    return config


def run_dataset_capture_recipe(
    recipe: Recipe,
    registry: OperationRegistry,
    store: LocalStore,
    out_dir: Path,
    *,
    force: bool = False,
    event_sink: Optional[Callable[[JsonDict], None]] = None,
    progress_sink: Optional[Callable[[JsonDict], None]] = None,
) -> JsonDict:
    config = validate_dataset_capture_contract(recipe, registry)
    recipe = _apply_capture_matrix_mode(recipe, config)
    if progress_sink is not None:
        requested_samples = config.get("samples")
        progress_sink(
            {
                "percent": 0.0,
                "phase": "validating",
                "message": "Validating dataset capture plan",
                "split": config["split"],
                "completed_samples": 0,
                "total_samples": (
                    int(requested_samples) if requested_samples is not None else None
                ),
                "unit": "samples",
            }
        )
    try:
        variant_plan = plan_recipe_variants(recipe, registry)
    except ValueError as exc:
        raise DatasetCaptureError(
            "Dataset Capture recipe variants are invalid: %s" % exc
        ) from exc
    sweep_plan = _sweep_plan(config.get("sweep"))
    if variant_plan.canonical_matrix.enabled and sweep_plan.get("enabled"):
        raise DatasetCaptureError(
            "Dataset Capture cannot combine a recipe metadata.matrix with "
            "dataset_capture.sweep; use metadata.matrix as the single variant definition"
        )

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise DatasetCaptureError("Dataset Capture output directory already exists and is not empty: %s" % out_dir)
    _preflight_dataset_capture_recipes(
        recipe,
        registry,
        config,
        variant_plan,
        sweep_plan,
    )

    if force and out_dir.exists():
        return _run_dataset_capture_staged(
            recipe,
            registry,
            store,
            out_dir,
            config=config,
            variant_plan=variant_plan,
            sweep_plan=sweep_plan,
            event_sink=event_sink,
            progress_sink=progress_sink,
        )
    result = _run_dataset_capture_into(
        recipe,
        registry,
        store,
        out_dir,
        public_out_dir=out_dir,
        config=config,
        variant_plan=variant_plan,
        sweep_plan=sweep_plan,
        event_sink=event_sink,
        progress_sink=progress_sink,
    )
    _report_dataset_capture_completed(
        result,
        event_sink=event_sink,
        progress_sink=progress_sink,
    )
    return result


def _run_dataset_capture_staged(
    recipe: Recipe,
    registry: OperationRegistry,
    store: LocalStore,
    out_dir: Path,
    *,
    config: JsonDict,
    variant_plan: RecipeVariantPlan,
    sweep_plan: JsonDict,
    event_sink: Optional[Callable[[JsonDict], None]],
    progress_sink: Optional[Callable[[JsonDict], None]],
) -> JsonDict:
    """Capture beside an existing dataset and replace it only after success."""

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".%s.recapture-" % out_dir.name,
            dir=str(out_dir.parent),
        )
    )
    try:
        result = _run_dataset_capture_into(
            recipe,
            registry,
            store,
            staging_dir,
            public_out_dir=out_dir,
            config=config,
            variant_plan=variant_plan,
            sweep_plan=sweep_plan,
            event_sink=event_sink,
            progress_sink=progress_sink,
        )
        _replace_capture_directory(staging_dir, out_dir)
        _report_dataset_capture_completed(
            result,
            event_sink=event_sink,
            progress_sink=progress_sink,
        )
        return result
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _replace_capture_directory(staging_dir: Path, out_dir: Path) -> None:
    """Swap a completed staged capture into place, rolling back commit errors."""

    backup_dir = out_dir.with_name(
        ".%s.previous-%s" % (out_dir.name, uuid.uuid4().hex)
    )
    out_dir.replace(backup_dir)
    try:
        staging_dir.replace(out_dir)
    except BaseException:
        try:
            backup_dir.replace(out_dir)
        except BaseException as rollback_exc:
            raise DatasetCaptureError(
                "Dataset Capture replacement failed and the previous dataset could not be "
                "restored automatically; it remains at %s" % backup_dir
            ) from rollback_exc
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _report_dataset_capture_completed(
    result: JsonDict,
    *,
    event_sink: Optional[Callable[[JsonDict], None]],
    progress_sink: Optional[Callable[[JsonDict], None]],
) -> None:
    if progress_sink is not None:
        requested_samples = result.get("requested_samples")
        progress_sink(
            {
                "percent": 100.0,
                "phase": "completed",
                "message": "Dataset capture completed",
                "split": result.get("split"),
                "completed_samples": int(result.get("captured_samples") or 0),
                "total_samples": (
                    int(requested_samples) if requested_samples is not None else None
                ),
                "unit": "samples",
                "run_count": len(result.get("run_ids") or []),
                "shard_count": int(result.get("number_of_shards") or 0),
            }
        )
    if event_sink is not None:
        event_sink(
            {
                "kind": "capture_completed",
                "message": "Completed dataset capture",
                "recipe_name": result.get("recipe"),
                "split": result.get("split"),
                "out_dir": result.get("out_dir"),
                "captured_samples": int(result.get("captured_samples") or 0),
                "shard_count": int(result.get("number_of_shards") or 0),
            }
        )


def _run_dataset_capture_into(
    recipe: Recipe,
    registry: OperationRegistry,
    store: LocalStore,
    out_dir: Path,
    *,
    public_out_dir: Path,
    config: JsonDict,
    variant_plan: RecipeVariantPlan,
    sweep_plan: JsonDict,
    event_sink: Optional[Callable[[JsonDict], None]],
    progress_sink: Optional[Callable[[JsonDict], None]],
) -> JsonDict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    _require_capture_disk_space(shards_dir)

    requested_samples = config.get("samples")
    shard_size = int(config["shard_size"])
    max_runs = int(config["max_runs"])
    executor = LocalExecutor(registry, store)

    last_progress = 0.0
    estimated_samples_per_run = 1

    def emit_event(kind: str, message: str, **fields: Any) -> None:
        if event_sink is not None:
            event_sink({"kind": kind, "message": message, **fields})

    def report_progress(percent: float, phase: str, message: str, **fields: Any) -> None:
        nonlocal last_progress
        last_progress = max(last_progress, min(100.0, float(percent)))
        payload: JsonDict = {
            "percent": round(last_progress, 2),
            "phase": phase,
            "message": message,
            "split": config["split"],
            "completed_samples": int(fields.pop("completed_samples", 0)),
            "total_samples": int(requested_samples) if requested_samples is not None else None,
            "unit": "samples",
            **fields,
        }
        if progress_sink is not None:
            progress_sink(payload)

    def capture_percent(
        completed_samples: int,
        *,
        within_run: float = 0.0,
    ) -> float:
        """Map whole-split sample progress onto the capture portion of the job."""

        if requested_samples is None:
            return 5.0 + (80.0 * max(0.0, min(1.0, within_run)))
        total = max(1, int(requested_samples))
        projected = float(completed_samples) + (
            max(0.0, min(1.0, within_run)) * float(estimated_samples_per_run)
        )
        sample_fraction = min(1.0, projected / float(total))
        return 2.0 + (94.0 * sample_fraction)

    emit_event(
        "capture_started",
        "Started dataset capture",
        recipe_name=recipe.name,
        split=config["split"],
        out_dir=str(public_out_dir),
        requested_samples=requested_samples,
    )
    report_progress(1, "preparing", "Prepared dataset capture output")

    tap_records: List[JsonDict] = []
    tap_schema: Dict[str, JsonDict] = {}
    run_records: List[JsonDict] = []
    shard_records: List[JsonDict] = []
    current_buffers: Dict[str, List[np.ndarray]] = {str(tap["id"]): [] for tap in config["taps"]}
    current_source_run_ids: List[str] = []
    current_count = 0
    captured_samples = 0
    run_index = 0

    while True:
        if requested_samples is not None and captured_samples >= int(requested_samples):
            break
        if requested_samples is None and run_index >= 1:
            break
        if run_index >= max_runs:
            raise DatasetCaptureError(
                "Dataset Capture stopped after max_runs=%d with %d/%s samples captured"
                % (max_runs, captured_samples, requested_samples if requested_samples is not None else "unbounded")
            )

        shard_index = len(shard_records)
        (
            selected_recipe,
            sweep_assignment,
            matrix_selection,
            matrix_index,
            matrix_identity,
            matrix_source,
        ) = _dataset_capture_variant_for_run(
            recipe,
            variant_plan,
            sweep_plan,
            run_index,
        )
        run_recipe = _dataset_capture_run_recipe(
            selected_recipe,
            config,
            run_index,
            shard_index,
            sweep_assignment,
            matrix_selection=matrix_selection,
            matrix_index=matrix_index,
            matrix_identity=matrix_identity,
        )
        step_positions = {step.id: index for index, step in enumerate(run_recipe.steps)}
        step_count = max(1, len(run_recipe.steps))

        def executor_event(event: JsonDict) -> None:
            forwarded = dict(event)
            forwarded["capture_run_index"] = run_index
            if event_sink is not None:
                event_sink(forwarded)
            step_id = str(event.get("step_id") or "")
            if step_id not in step_positions:
                return
            position = int(step_positions[step_id])
            kind = str(event.get("kind") or "")
            fraction = 0.0
            if kind == "step_completed":
                fraction = 1.0
            elif kind == "step_progress":
                try:
                    fraction = max(0.0, min(1.0, float(event.get("percent", 0)) / 100.0))
                except (TypeError, ValueError):
                    fraction = 0.0
            within_run = (position + fraction) / float(step_count)
            report_progress(
                capture_percent(captured_samples, within_run=within_run),
                "executing",
                str(event.get("message") or "Running capture recipe"),
                completed_samples=captured_samples,
                run_index=run_index,
                step_id=step_id,
                step_index=position,
                step_count=step_count,
            )

        report_progress(
            capture_percent(captured_samples),
            "executing",
            "Running capture recipe",
            completed_samples=captured_samples,
            run_index=run_index,
            step_count=step_count,
        )
        _require_capture_disk_space(shards_dir)
        _require_capture_disk_space(
            store.workspace
            if store.workspace.exists()
            else store.workspace.parent
        )
        run_dir = executor.run(
            run_recipe,
            event_sink=executor_event,
            runner="dataset_capture",
        )
        summary = decode_strict_yaml_or_json(
            (run_dir / "summary.json").read_text(encoding="utf-8"),
            input_format="json",
        )
        if summary.get("status") != "completed":
            raise DatasetCaptureError("Dataset Capture recipe did not complete: %s" % summary.get("status"))

        record_batch, run_tap_records, run_tap_schema, run_sample_count = _capture_records_from_run(summary, config)
        if run_sample_count <= 0:
            raise DatasetCaptureError("Dataset Capture run %s produced no dataset capture records" % summary.get("run_id"))
        estimated_samples_per_run = run_sample_count
        if not tap_records:
            tap_records = [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "artifact_path",
                        "artifact_sha256",
                        "metadata",
                        "shape",
                        "record_count",
                    }
                }
                | {"record_count": 0, "source_artifacts": []}
                for record in run_tap_records
            ]
            tap_schema = run_tap_schema
        else:
            _validate_tap_schema(tap_schema, run_tap_schema, summary.get("run_id"))

        run_record = {
            "run_index": run_index,
            "run_id": summary.get("run_id"),
            "run_dir": str(run_dir),
            "recipe_name": run_recipe.name,
            "seed": master_seed_from_recipe(run_recipe),
            "captured_samples": 0,
            "available_samples": run_sample_count,
            "sweep": sweep_assignment,
            "matrix_selection": matrix_selection,
            "matrix_index": matrix_index,
            "matrix_variant_id": matrix_identity,
            "matrix_source": matrix_source,
            "channel_distribution": _channel_distribution(run_recipe, summary),
        }
        run_records.append(run_record)

        remaining = run_sample_count
        offset = 0
        if requested_samples is not None:
            remaining = min(remaining, int(requested_samples) - captured_samples)
        while remaining > 0:
            space = shard_size - current_count
            take = min(space, remaining)
            for tap_id, records in record_batch.items():
                current_buffers[tap_id].append(records[offset : offset + take])
            source_run_id = str(summary.get("run_id") or "")
            if source_run_id and source_run_id not in current_source_run_ids:
                current_source_run_ids.append(source_run_id)
            current_count += take
            captured_samples += take
            offset += take
            remaining -= take
            if current_count == shard_size:
                shard_record = _write_capture_shard(
                    shards_dir,
                    len(shard_records),
                    current_buffers,
                    recipe,
                    config,
                    captured_samples - current_count,
                    current_count,
                    current_source_run_ids,
                )
                shard_records.append(shard_record)
                emit_event(
                    "capture_shard_written",
                    "Wrote capture shard %d" % shard_record["index"],
                    run_index=run_index,
                    shard=shard_record,
                    captured_samples=captured_samples,
                )
                report_progress(
                    capture_percent(captured_samples),
                    "writing",
                    "Wrote capture shard %d" % shard_record["index"],
                    completed_samples=captured_samples,
                    run_index=run_index,
                    shard_index=shard_record["index"],
                )
                current_buffers = {str(tap["id"]): [] for tap in config["taps"]}
                current_source_run_ids = []
                current_count = 0

        run_record["captured_samples"] = min(run_sample_count, offset)
        run_taps_by_id = {
            str(record.get("id") or ""): record for record in run_tap_records
        }
        for tap_record in tap_records:
            tap_id = str(tap_record.get("id") or "")
            run_tap = run_taps_by_id[tap_id]
            tap_record["record_count"] = int(tap_record.get("record_count") or 0) + int(
                run_record["captured_samples"]
            )
            tap_record["source_artifacts"].append(
                {
                    "run_id": summary.get("run_id"),
                    "run_index": run_index,
                    "artifact_path": run_tap.get("artifact_path"),
                    "artifact_sha256": run_tap.get("artifact_sha256"),
                    "artifact_kind": run_tap.get("artifact_kind"),
                    "array_name": run_tap.get("array_name"),
                    "dtype": run_tap.get("dtype"),
                    "shape": run_tap.get("shape"),
                    "available_record_count": run_tap.get("record_count"),
                    "captured_record_count": run_record["captured_samples"],
                    "metadata": run_tap.get("metadata"),
                }
            )
        emit_event(
            "capture_run_completed",
            "Captured %d samples from run %d" % (run_record["captured_samples"], run_index),
            run_index=run_index,
            run_id=summary.get("run_id"),
            captured_samples=captured_samples,
            available_samples=run_sample_count,
        )
        if requested_samples is not None:
            progress_message = "Captured %d of %d samples" % (
                captured_samples,
                int(requested_samples),
            )
        else:
            progress_message = "Captured %d samples" % captured_samples
        report_progress(
            capture_percent(captured_samples),
            "capturing",
            progress_message,
            completed_samples=captured_samples,
            run_index=run_index,
            run_count=run_index + 1,
            shard_count=len(shard_records),
        )
        run_index += 1

    if current_count:
        shard_record = _write_capture_shard(
            shards_dir,
            len(shard_records),
            current_buffers,
            recipe,
            config,
            captured_samples - current_count,
            current_count,
            current_source_run_ids,
        )
        shard_records.append(shard_record)
        emit_event(
            "capture_shard_written",
            "Wrote capture shard %d" % shard_record["index"],
            run_index=max(0, run_index - 1),
            shard=shard_record,
            captured_samples=captured_samples,
        )
        report_progress(
            capture_percent(captured_samples),
            "writing",
            "Wrote capture shard %d" % shard_record["index"],
            completed_samples=captured_samples,
            run_index=max(0, run_index - 1),
            shard_index=shard_record["index"],
        )

    if captured_samples <= 0:
        raise DatasetCaptureError("Dataset Capture produced no samples")

    report_progress(
        99,
        "finalizing",
        "Finalizing dataset metadata",
        completed_samples=captured_samples,
        run_count=run_index,
        shard_count=len(shard_records),
    )

    channel_distribution = _capture_channel_distribution(run_records)
    sweep_distribution = _sweep_distribution(run_records, sweep_plan)
    matrix_distribution = _matrix_distribution(run_records, variant_plan, sweep_plan)
    schema = {
        "schema_version": 1,
        "kind": "noema.capture_dataset",
        "recipe": recipe.name,
        "recipe_sha256": recipe_fingerprint(recipe),
        "created_at_utc": utc_now_iso(),
        "split": config["split"],
        "requested_samples": requested_samples,
        "captured_samples": captured_samples,
        "number_of_shards": len(shard_records),
        "shard_format": "npz",
        "shard_size": shard_size,
        "shards": shard_records,
        "seed_policy": _capture_seed_policy(recipe, config, run_records),
        "master_seed": master_seed_from_recipe(recipe),
        "channel_distribution": channel_distribution,
        "matrix_distribution": matrix_distribution,
        "sweep_distribution": sweep_distribution,
        "taps": tap_records,
        "tap_schemas": tap_schema,
        "runs": run_records,
    }
    tap_manifest = {
        "schema_version": 1,
        "taps": tap_records,
        "tap_schemas": tap_schema,
        "source_run_ids": [record["run_id"] for record in run_records],
        "source_run_dirs": [record["run_dir"] for record in run_records],
        "runs": run_records,
    }
    split = {
        "schema_version": 1,
        "split": config["split"],
        "requested_samples": requested_samples,
        "captured_samples": captured_samples,
        "shards": [record["path"] for record in shard_records],
    }

    _write_json(out_dir / "schema.json", schema)
    _write_json(out_dir / "tap_manifest.json", tap_manifest)
    _write_json(out_dir / "recipe.json", recipe.to_dict())
    _write_json(out_dir / "channel_distribution.json", channel_distribution)
    _write_json(out_dir / "matrix_distribution.json", matrix_distribution)
    _write_json(out_dir / "split.json", split)

    result = {
        "status": "captured",
        "recipe": recipe.name,
        "run_id": run_records[0]["run_id"] if run_records else None,
        "run_ids": [record["run_id"] for record in run_records],
        "matrix_variant_ids": [
            record["matrix_variant_id"]
            for record in run_records
            if record.get("matrix_variant_id")
        ],
        "out_dir": str(public_out_dir),
        "split": config["split"],
        "requested_samples": requested_samples,
        "captured_samples": captured_samples,
        "tap_count": len(tap_records),
        "number_of_shards": len(shard_records),
        "shards": [record["path"] for record in shard_records],
        "files": [
            *[record["path"] for record in shard_records],
            "schema.json",
            "tap_manifest.json",
            "recipe.json",
            "channel_distribution.json",
            "matrix_distribution.json",
            "split.json",
        ],
    }
    return result


def _dataset_capture_config(recipe: Recipe) -> JsonDict:
    dataset_capture = dict(recipe.dataset_capture or {})
    taps = dataset_capture.get("taps")
    if not isinstance(taps, list) or not taps:
        raise DatasetCaptureError("Recipe dataset_capture.taps must contain at least one tap")
    normalized_taps: List[JsonDict] = []
    seen = set()
    for index, item in enumerate(taps):
        if not isinstance(item, Mapping):
            raise DatasetCaptureError("dataset_capture.taps[%d] must be a mapping" % index)
        tap_id = str(item.get("id") or "").strip()
        reference = str(item.get("from") or "").strip()
        if not tap_id:
            raise DatasetCaptureError("dataset_capture.taps[%d] requires id" % index)
        if tap_id == "metadata_json":
            raise DatasetCaptureError("Tap id is reserved: metadata_json")
        if tap_id in seen:
            raise DatasetCaptureError("Duplicate dataset capture tap id: %s" % tap_id)
        if len(reference.split(".")) != 2:
            raise DatasetCaptureError("Dataset Capture tap %s must use from: <step_id>.<output_name>" % tap_id)
        seen.add(tap_id)
        normalized_taps.append({"id": tap_id, "from": reference})
    samples = dataset_capture.get("samples")
    normalized_samples = (
        _bounded_capture_integer(
            samples,
            "dataset_capture.samples",
            MAX_CAPTURE_SAMPLES,
        )
        if samples is not None
        else None
    )
    raw_shard_size = dataset_capture.get("shard_size")
    shard_size = (
        _bounded_capture_integer(
            raw_shard_size,
            "dataset_capture.shard_size",
            MAX_CAPTURE_SHARD_SIZE,
        )
        if raw_shard_size is not None
        else (
            min(normalized_samples, DEFAULT_CAPTURE_SHARD_SIZE)
            if normalized_samples is not None
            else DEFAULT_CAPTURE_SHARD_SIZE
        )
    )
    raw_max_runs = dataset_capture.get("max_runs")
    max_runs = (
        _bounded_capture_integer(
            raw_max_runs,
            "dataset_capture.max_runs",
            MAX_CAPTURE_RUNS,
        )
        if raw_max_runs is not None
        else (normalized_samples if normalized_samples is not None else 1)
    )
    seed_mode = str(dataset_capture.get("seed_mode") or "increment_run_seed").strip().lower()
    if seed_mode not in {"increment_run_seed", "fixed_seed", "recipe_seed_plus_shard"}:
        raise DatasetCaptureError(
            "dataset_capture.seed_mode must be one of increment_run_seed, fixed_seed, recipe_seed_plus_shard"
        )
    matrix_mode = str(
        dataset_capture.get("matrix_mode") or "inherit"
    ).strip().lower()
    if matrix_mode not in {"inherit", "exclude"}:
        raise DatasetCaptureError(
            "dataset_capture.matrix_mode must be one of inherit or exclude"
        )
    return {
        "split": str(dataset_capture.get("split") or "train"),
        "samples": normalized_samples,
        "shard_size": shard_size,
        "max_runs": max_runs,
        "seed_mode": seed_mode,
        "matrix_mode": matrix_mode,
        "sweep": dataset_capture.get("sweep"),
        "taps": normalized_taps,
    }


def _bounded_capture_integer(value: Any, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetCaptureError("%s must be a positive integer" % field)
    normalized = int(value)
    if normalized <= 0:
        raise DatasetCaptureError("%s must be a positive integer" % field)
    if normalized > maximum:
        raise DatasetCaptureError("%s must be at most %d" % (field, maximum))
    return normalized


def _apply_capture_matrix_mode(
    recipe: Recipe,
    config: Mapping[str, Any],
) -> Recipe:
    if str(config.get("matrix_mode") or "inherit") != "exclude":
        return recipe
    payload = recipe.to_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("matrix", None)
    metadata.pop("sweeps", None)
    metadata.pop("ui_sweeps", None)
    payload["metadata"] = metadata
    return recipe_from_dict(payload)


def _preflight_dataset_capture_recipes(
    recipe: Recipe,
    registry: OperationRegistry,
    config: Mapping[str, Any],
    variant_plan: RecipeVariantPlan,
    sweep_plan: Mapping[str, Any],
) -> None:
    """Runner-plan every distinct declared capture variant without executing it."""

    concrete_count = (
        len(variant_plan.variants)
        or len(sweep_plan.get("assignments") or [])
        or 1
    )
    planned_identities = set()
    for concrete_index in range(concrete_count):
        (
            selected_recipe,
            sweep_assignment,
            matrix_selection,
            matrix_index,
            matrix_identity,
            matrix_source,
        ) = _dataset_capture_variant_for_run(
            recipe,
            variant_plan,
            sweep_plan,
            concrete_index,
        )
        identity = matrix_identity or "base"
        if identity in planned_identities:
            continue
        planned_identities.add(identity)
        run_recipe = _dataset_capture_run_recipe(
            selected_recipe,
            config,
            concrete_index,
            concrete_index,
            sweep_assignment,
            matrix_selection=matrix_selection,
            matrix_index=matrix_index,
            matrix_identity=matrix_identity,
        )
        if matrix_source == "dataset_capture.sweep":
            label = "dataset_capture.sweep variant %d (%s)" % (
                int(matrix_index),
                identity,
            )
        elif matrix_source:
            label = "matrix variant %d (%s)" % (int(matrix_index), identity)
        else:
            label = "base recipe"
        try:
            effective_recipe = compile_effective_recipe_for_runner(
                run_recipe,
                registry,
                runner="dataset_capture",
            )
            effective_recipe = prepare_compiled_single_run_recipe(effective_recipe)
            plan_recipe(
                effective_recipe,
                registry,
                runner="dataset_capture",
            )
        except (OperationError, ValueError) as exc:
            raise DatasetCaptureError(
                "Dataset Capture preflight failed for %s: %s" % (label, exc)
            ) from exc


def _dataset_capture_variant_for_run(
    recipe: Recipe,
    variant_plan: RecipeVariantPlan,
    sweep_plan: Mapping[str, Any],
    run_index: int,
) -> Tuple[
    Recipe,
    JsonDict,
    Optional[JsonDict],
    Optional[int],
    Optional[str],
    Optional[str],
]:
    sweep_assignment = _sweep_assignment(sweep_plan, run_index)
    selected_recipe = recipe
    matrix_selection: Optional[JsonDict] = None
    matrix_index: Optional[int] = None
    matrix_identity: Optional[str] = None
    matrix_source: Optional[str] = None
    if variant_plan.variants:
        variant = variant_plan.variants[run_index % len(variant_plan.variants)]
        selected_recipe = variant.recipe
        sweep_assignment = {}
        matrix_selection = dict(variant.matrix_selection)
        matrix_index = variant.matrix_index
        matrix_identity = variant.matrix_variant_id
        matrix_source = "metadata.%s" % variant_plan.canonical_matrix.source
    elif sweep_plan.get("enabled"):
        matrix_selection = dict(sweep_assignment)
        matrix_index = run_index % len(sweep_plan["assignments"])
        matrix_identity = matrix_variant_id(matrix_selection)
        matrix_source = "dataset_capture.sweep"
    return (
        selected_recipe,
        sweep_assignment,
        matrix_selection,
        matrix_index,
        matrix_identity,
        matrix_source,
    )


def _dataset_capture_run_recipe(
    recipe: Recipe,
    config: Mapping[str, Any],
    run_index: int,
    shard_index: int,
    sweep_assignment: Mapping[str, Any],
    *,
    matrix_selection: Optional[Mapping[str, Any]] = None,
    matrix_index: Optional[int] = None,
    matrix_identity: Optional[str] = None,
) -> Recipe:
    payload = recipe.to_dict()
    payload["name"] = "%s_dataset_capture_%04d" % (recipe.name, run_index)
    metadata = dict(payload.get("metadata") or {})
    metadata["seed_namespace"] = _capture_seed_namespace(recipe, config)
    base_seed = master_seed_from_recipe(recipe)
    seed_mode = str(config.get("seed_mode") or "increment_run_seed")
    if seed_mode == "increment_run_seed":
        metadata["seed"] = _seed_value(base_seed, run_index)
    elif seed_mode == "recipe_seed_plus_shard":
        metadata["seed"] = _seed_value(base_seed, shard_index)
    elif seed_mode == "fixed_seed" and base_seed is not None:
        metadata["seed"] = int(base_seed)
    if matrix_selection is not None:
        selection = dict(matrix_selection)
        if matrix_index is None:
            raise DatasetCaptureError(
                "Dataset Capture matrix provenance requires matrix_index"
            )
        metadata["matrix_selection"] = selection
        metadata["matrix_index"] = int(matrix_index)
        metadata["matrix_variant_id"] = matrix_identity or matrix_variant_id(selection)
    payload["metadata"] = metadata
    for key, value in sweep_assignment.items():
        _apply_sweep_value(payload, key, value)
    dataset_capture = dict(payload.get("dataset_capture") or {})
    dataset_capture.pop("sweep", None)
    payload["dataset_capture"] = dataset_capture
    return recipe_from_dict(payload)


def _seed_value(base_seed: Optional[int], offset: int) -> int:
    base = 0 if base_seed is None else int(base_seed)
    value = (base + int(offset)) % SEED_MODULUS
    return int(value or SEED_MODULUS)


def _capture_seed_namespace(recipe: Recipe, config: Mapping[str, Any]) -> str:
    """Keep internal run names out of RNG derivation and isolate splits."""

    base = seed_namespace_from_recipe(recipe)
    split = str(config.get("split") or "train").strip() or "train"
    return "%s|dataset_capture_split=%s" % (base, split)


def _capture_records_from_run(summary: Mapping[str, Any], config: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], List[JsonDict], Dict[str, JsonDict], int]:
    outputs = _summary_outputs(summary)
    record_arrays: Dict[str, np.ndarray] = {}
    tap_records: List[JsonDict] = []
    tap_schema: Dict[str, JsonDict] = {}
    record_counts: Dict[str, int] = {}
    for tap in config["taps"]:
        tap_id = str(tap["id"])
        if tap_id == "metadata_json":
            raise DatasetCaptureError("Tap id is reserved: metadata_json")
        reference = str(tap["from"])
        artifact_payload = _artifact_for_reference(outputs, reference)
        array, array_name, embedded_metadata = _load_npz_tap(artifact_payload, reference)
        records, interpretation = _array_as_capture_records(array, embedded_metadata, tap_id, reference)
        record_arrays[tap_id] = records
        record_counts[tap_id] = int(records.shape[0])
        source_step, source_output = reference.split(".", 1)
        tap_records.append(
            {
                "id": tap_id,
                "from": reference,
                "source_step": source_step,
                "source_output": source_output,
                "artifact_kind": artifact_payload.get("kind"),
                "artifact_path": artifact_payload.get("path"),
                "artifact_sha256": artifact_payload.get("sha256"),
                "array_name": array_name,
                "dtype": str(array.dtype),
                "shape": [int(item) for item in array.shape],
                "record_shape": [int(item) for item in records.shape[1:]],
                "record_count": int(records.shape[0]),
                "record_interpretation": interpretation,
                "metadata": embedded_metadata,
            }
        )
        tap_schema[tap_id] = {
            "id": tap_id,
            "from": reference,
            "dtype": str(records.dtype),
            "record_shape": [int(item) for item in records.shape[1:]],
            "stored_shape_template": ["N", *[int(item) for item in records.shape[1:]]],
            "record_interpretation": interpretation,
        }
    sample_count = _matching_record_count(record_counts)
    return record_arrays, tap_records, tap_schema, sample_count


def _array_as_capture_records(
    array: np.ndarray,
    metadata: Mapping[str, Any],
    tap_id: str,
    reference: str,
) -> Tuple[np.ndarray, str]:
    value = np.asarray(array)
    declared_count = metadata.get("capture_record_count")
    declared_shape = metadata.get("capture_record_shape")
    if declared_count is not None or declared_shape is not None:
        if (
            not isinstance(declared_count, int)
            or isinstance(declared_count, bool)
            or declared_count <= 0
        ):
            raise DatasetCaptureError(
                "Dataset Capture tap %s has invalid capture_record_count" % tap_id
            )
        if (
            not isinstance(declared_shape, (list, tuple))
            or not declared_shape
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item <= 0
                for item in declared_shape
            )
        ):
            raise DatasetCaptureError(
                "Dataset Capture tap %s has invalid capture_record_shape" % tap_id
            )
        expected_shape = (
            int(declared_count),
            *[int(item) for item in declared_shape],
        )
        if int(np.prod(expected_shape, dtype=np.int64)) != int(value.size):
            raise DatasetCaptureError(
                "Dataset Capture tap %s declares record shape %s but stores %d elements"
                % (tap_id, list(expected_shape), int(value.size))
            )
        return value.reshape(expected_shape), "explicit_record_shape"
    explicit_axis = _explicit_record_axis(metadata)
    if explicit_axis is not None:
        if explicit_axis != 0:
            raise DatasetCaptureError("Dataset Capture tap %s only supports record_axis=0 for now" % tap_id)
        if value.ndim == 0:
            raise DatasetCaptureError("Dataset Capture tap %s declares record_axis=0 but is scalar" % tap_id)
        return value, "explicit_axis_0"
    if value.ndim >= 2:
        return value, "leading_axis"
    if value.ndim == 1:
        return value.reshape((1, int(value.shape[0]))), "single_record_1d"
    return value.reshape((1,)), "single_record_scalar"


def _explicit_record_axis(metadata: Mapping[str, Any]) -> Optional[int]:
    for key in ("capture_record_axis", "record_axis", "sample_axis"):
        if key in metadata and metadata[key] is not None:
            return int(metadata[key])
    return None


def _matching_record_count(record_counts: Mapping[str, int]) -> int:
    if not record_counts:
        raise DatasetCaptureError("Dataset Capture produced no tap arrays")
    values = set(int(value) for value in record_counts.values())
    if len(values) != 1:
        details = ", ".join("%s=%d" % (key, value) for key, value in sorted(record_counts.items()))
        raise DatasetCaptureError("Dataset Capture taps have incompatible record counts: %s" % details)
    return int(next(iter(values)))


def _validate_tap_schema(expected: Mapping[str, JsonDict], actual: Mapping[str, JsonDict], run_id: Any) -> None:
    for tap_id, expected_schema in expected.items():
        actual_schema = actual.get(tap_id)
        if actual_schema is None:
            raise DatasetCaptureError("Dataset Capture run %s did not produce tap %s" % (run_id, tap_id))
        for key in ("dtype", "record_shape"):
            if actual_schema.get(key) != expected_schema.get(key):
                raise DatasetCaptureError(
                    "Dataset Capture tap %s changed %s in run %s: expected %s, got %s"
                    % (tap_id, key, run_id, expected_schema.get(key), actual_schema.get(key))
                )


def _write_capture_shard(
    shards_dir: Path,
    shard_index: int,
    buffers: Mapping[str, List[np.ndarray]],
    recipe: Recipe,
    config: Mapping[str, Any],
    sample_start: int,
    sample_count: int,
    source_run_ids: Sequence[str],
) -> JsonDict:
    arrays = {}
    for tap_id, chunks in buffers.items():
        if not chunks:
            raise DatasetCaptureError("Dataset Capture shard %d has no records for tap %s" % (shard_index, tap_id))
        arrays[tap_id] = np.concatenate(chunks, axis=0)
    if any(int(array.shape[0]) != int(sample_count) for array in arrays.values()):
        counts = ", ".join("%s=%d" % (key, int(value.shape[0])) for key, value in sorted(arrays.items()))
        raise DatasetCaptureError("Dataset Capture shard %d has inconsistent tap counts: %s" % (shard_index, counts))
    estimated_bytes = sum(int(array.nbytes) for array in arrays.values())
    _require_capture_disk_space(
        shards_dir,
        additional_bytes=estimated_bytes,
        context="shard %d" % shard_index,
    )
    metadata = {
        "schema_version": 1,
        "kind": "noema.capture_shard",
        "recipe": recipe.name,
        "split": config["split"],
        "shard_index": int(shard_index),
        "sample_start": int(sample_start),
        "captured_samples": int(sample_count),
        "tap_ids": list(arrays.keys()),
        "source_run_ids": list(dict.fromkeys(str(item) for item in source_run_ids if str(item))),
        "created_at_utc": utc_now_iso(),
    }
    path = shards_dir / ("shard_%04d.npz" % shard_index)
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    return {
        "path": "shards/%s" % path.name,
        "index": int(shard_index),
        "sample_start": int(sample_start),
        "captured_samples": int(sample_count),
        "tap_ids": list(arrays.keys()),
        "shapes": {key: [int(item) for item in value.shape] for key, value in arrays.items()},
        "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "source_run_ids": list(
            dict.fromkeys(str(item) for item in source_run_ids if str(item))
        ),
    }


def _require_capture_disk_space(
    path: Path,
    *,
    additional_bytes: int = 0,
    context: str = "execution",
) -> None:
    try:
        free_bytes = int(shutil.disk_usage(path).free)
    except OSError as exc:
        raise DatasetCaptureError(
            "Could not determine free disk space for Dataset Capture: %s" % exc
        ) from exc
    required_bytes = max(0, int(additional_bytes)) + CAPTURE_DISK_RESERVE_BYTES
    if free_bytes < required_bytes:
        raise DatasetCaptureError(
            "Dataset Capture %s needs approximately %d bytes plus a %d-byte "
            "disk reserve, but only %d bytes are free"
            % (
                context,
                max(0, int(additional_bytes)),
                CAPTURE_DISK_RESERVE_BYTES,
                free_bytes,
            )
        )


def _sweep_plan(raw: Any) -> JsonDict:
    if raw in (None, {}, []):
        return {"enabled": False, "mode": "none", "assignments": []}
    if isinstance(raw, list):
        assignments = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise DatasetCaptureError("dataset_capture.sweep[%d] must be a mapping" % index)
            assignments.append({str(key): value for key, value in item.items()})
        if not assignments:
            raise DatasetCaptureError("dataset_capture.sweep must not be empty")
        return {"enabled": True, "mode": "round_robin_list", "assignments": assignments}
    if isinstance(raw, Mapping):
        dimensions = {str(key): _sweep_values(value) for key, value in raw.items()}
        for key, values in dimensions.items():
            if not values:
                raise DatasetCaptureError("dataset_capture.sweep.%s has no values" % key)
        keys = sorted(dimensions)
        assignments = [dict(zip(keys, values)) for values in product(*(dimensions[key] for key in keys))]
        return {
            "enabled": bool(assignments),
            "mode": "round_robin_grid",
            "dimensions": dimensions,
            "assignments": assignments,
        }
    raise DatasetCaptureError("dataset_capture.sweep must be a mapping or a list of mappings")


def _sweep_values(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            if ":" in text:
                raise DatasetCaptureError(
                    "dataset_capture.sweep values must use either a:b:c range syntax or a,b,c list syntax; got %s"
                    % text
                )
            parts = [part.strip() for part in text.split(",")]
            if any(not part for part in parts):
                raise DatasetCaptureError("dataset_capture.sweep lists must not contain empty values: %s" % text)
            try:
                return [_clean_number(float(part)) for part in parts]
            except ValueError as exc:
                raise DatasetCaptureError("dataset_capture.sweep numeric lists must use a,b,c syntax; got %s" % text) from exc
        if ":" in text:
            return _parse_range_values(text)
        return [text]
    return [value]


def _parse_range_values(text: str) -> List[float]:
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) == 2:
        start, stop = float(parts[0]), float(parts[1])
        step = 1.0
    elif len(parts) == 3:
        start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
    else:
        raise DatasetCaptureError("dataset_capture.sweep ranges must use a:c or a:b:c syntax; got %s" % text)
    if step == 0:
        raise DatasetCaptureError("dataset_capture.sweep range step must not be zero: %s" % text)
    values = []
    current = start
    if step > 0:
        while current <= stop + 1e-12:
            values.append(_clean_number(current))
            current += step
    else:
        while current >= stop - 1e-12:
            values.append(_clean_number(current))
            current += step
    return values


def _clean_number(value: float) -> float | int:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-12:
        return int(rounded)
    return float(value)


def _sweep_assignment(plan: Mapping[str, Any], run_index: int) -> JsonDict:
    assignments = list(plan.get("assignments") or [])
    if not assignments:
        return {}
    return dict(assignments[int(run_index) % len(assignments)])


def _apply_sweep_value(recipe_payload: JsonDict, key: str, value: Any) -> None:
    step_id, param_name = _resolve_sweep_target(recipe_payload, str(key))
    for step in recipe_payload.get("steps") or []:
        if step.get("id") == step_id:
            step.setdefault("params", {})[param_name] = value
            return
    raise DatasetCaptureError("dataset_capture.sweep target step was not found: %s" % key)


def _resolve_sweep_target(recipe_payload: Mapping[str, Any], key: str) -> Tuple[str, str]:
    parts = key.split(".")
    if len(parts) < 2:
        raise DatasetCaptureError("dataset_capture.sweep key must use <step_id>.<param>, got %s" % key)
    step_id = parts[0]
    param_name = ".".join(parts[1:])
    steps = list(recipe_payload.get("steps") or [])
    if any(step.get("id") == step_id for step in steps if isinstance(step, Mapping)):
        return step_id, param_name
    if step_id == "channel":
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            op = str(step.get("op") or "")
            sid = str(step.get("id") or "")
            params = dict(step.get("params") or {})
            if op.startswith("wireless.") or sid == "wireless_channel" or param_name in params:
                return sid, param_name
    raise DatasetCaptureError("dataset_capture.sweep key references unknown step: %s" % key)


def _capture_seed_policy(recipe: Recipe, config: Mapping[str, Any], run_records: Sequence[Mapping[str, Any]]) -> JsonDict:
    payload = seed_policy(recipe)
    payload["capture_seed_mode"] = config.get("seed_mode")
    payload["capture_seed_namespace"] = _capture_seed_namespace(recipe, config)
    payload["run_seeds"] = [record.get("seed") for record in run_records]
    return payload


def _capture_channel_distribution(run_records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema_version": 1,
        "runs": [
            {
                "run_index": record.get("run_index"),
                "run_id": record.get("run_id"),
                "captured_samples": record.get("captured_samples"),
                "sweep": record.get("sweep"),
                "matrix_selection": record.get("matrix_selection"),
                "matrix_index": record.get("matrix_index"),
                "matrix_variant_id": record.get("matrix_variant_id"),
                "channel_distribution": record.get("channel_distribution"),
            }
            for record in run_records
        ],
    }


def _matrix_distribution(
    run_records: Sequence[Mapping[str, Any]],
    variant_plan: RecipeVariantPlan,
    sweep_plan: Mapping[str, Any],
) -> JsonDict:
    variants: List[JsonDict] = []
    if variant_plan.variants:
        variants = [
            {
                "matrix_variant_id": variant.matrix_variant_id,
                "matrix_index": variant.matrix_index,
                "matrix_selection": dict(variant.matrix_selection),
            }
            for variant in variant_plan.variants
        ]
        source = "metadata.%s" % variant_plan.canonical_matrix.source
    else:
        for index, assignment in enumerate(sweep_plan.get("assignments") or []):
            selection = dict(assignment)
            variants.append(
                {
                    "matrix_variant_id": matrix_variant_id(selection),
                    "matrix_index": index,
                    "matrix_selection": selection,
                }
            )
        source = "dataset_capture.sweep" if variants else "none"

    run_counts: Dict[str, int] = {}
    sample_counts: Dict[str, int] = {}
    for record in run_records:
        identity = record.get("matrix_variant_id")
        if not identity:
            continue
        key = str(identity)
        run_counts[key] = run_counts.get(key, 0) + 1
        sample_counts[key] = sample_counts.get(key, 0) + int(
            record.get("captured_samples") or 0
        )
    return {
        "schema_version": 1,
        "enabled": bool(variants),
        "source": source,
        "canonical_matrix": variant_plan.canonical_matrix.to_dict(),
        "variants": variants,
        "run_counts": run_counts,
        "sample_counts": sample_counts,
    }


def _sweep_distribution(run_records: Sequence[Mapping[str, Any]], sweep_plan: Mapping[str, Any]) -> JsonDict:
    counts: Dict[str, int] = {}
    sample_counts: Dict[str, int] = {}
    for record in run_records:
        assignment = dict(record.get("sweep") or {})
        key = json.dumps(
            assignment,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        counts[key] = counts.get(key, 0) + 1
        sample_counts[key] = sample_counts.get(key, 0) + int(record.get("captured_samples") or 0)
    return {
        "schema_version": 1,
        "enabled": bool(sweep_plan.get("enabled")),
        "mode": sweep_plan.get("mode"),
        "dimensions": sweep_plan.get("dimensions") or {},
        "assignments": sweep_plan.get("assignments") or [],
        "run_counts": counts,
        "sample_counts": sample_counts,
    }


def _summary_outputs(summary: Mapping[str, Any]) -> Dict[Tuple[str, str], JsonDict]:
    outputs: Dict[Tuple[str, str], JsonDict] = {}
    for step in summary.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id") or "")
        for output_name, payload in dict(step.get("outputs") or {}).items():
            if isinstance(payload, Mapping):
                outputs[(step_id, str(output_name))] = dict(payload)
    return outputs


def _artifact_for_reference(outputs: Mapping[Tuple[str, str], JsonDict], reference: str) -> JsonDict:
    step_id, output_name = reference.split(".", 1)
    try:
        return dict(outputs[(step_id, output_name)])
    except KeyError as exc:
        raise DatasetCaptureError("Dataset Capture tap reference was not produced by the run: %s" % reference) from exc


def _load_npz_tap(artifact_payload: Mapping[str, Any], reference: str) -> Tuple[np.ndarray, str, JsonDict]:
    path = Path(str(artifact_payload.get("path") or ""))
    if not path.is_file():
        raise DatasetCaptureError("Dataset Capture tap artifact is missing for %s: %s" % (reference, path))
    if path.suffix != ".npz":
        raise DatasetCaptureError("Dataset Capture MVP supports .npz array taps only; %s uses %s" % (reference, path.suffix or "no suffix"))
    preferred = str((artifact_payload.get("metadata") or {}).get("array") or "")
    with np.load(str(path), allow_pickle=False) as payload:
        names = [name for name in payload.files if name != "metadata_json"]
        if not names:
            raise DatasetCaptureError("Dataset Capture tap %s has no arrays in %s" % (reference, path))
        array_name = preferred if preferred in names else (names[0] if len(names) == 1 else "")
        if not array_name:
            raise DatasetCaptureError("Dataset Capture tap %s has multiple arrays; artifact metadata must name one" % reference)
        array = np.asarray(payload[array_name])
        metadata: JsonDict = {}
        if "metadata_json" in payload.files:
            try:
                embedded = decode_strict_json_object(
                    str(payload["metadata_json"].item()),
                    label="Dataset Capture tap %s metadata_json" % reference,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetCaptureError(
                    "Dataset Capture tap %s contains malformed metadata_json in %s"
                    % (reference, path)
                ) from exc
            metadata.update(embedded)
    return array, array_name, metadata


def _common_leading_dimension(arrays: Mapping[str, np.ndarray]) -> Optional[int]:
    lengths = [int(value.shape[0]) for value in arrays.values() if value.ndim > 0]
    if not lengths:
        return None
    first = lengths[0]
    return first if all(value == first for value in lengths) else None


def _channel_distribution(recipe: Recipe, summary: Mapping[str, Any]) -> JsonDict:
    channel_steps: List[JsonDict] = []
    step_summaries = {str(step.get("id")): step for step in summary.get("steps") or [] if isinstance(step, Mapping)}
    for step in recipe.steps:
        if not (step.op.startswith("wireless.") or step.op.startswith("modulation.") or step.op.startswith("demodulation.") or "channel" in step.id):
            continue
        step_summary = dict(step_summaries.get(step.id) or {})
        channel_steps.append(
            {
                "id": step.id,
                "op": step.op,
                "params": dict(step.params),
                "metrics": dict(step_summary.get("metrics") or {}),
                "metadata": dict(step_summary.get("metadata") or {}),
            }
        )
    return {"schema_version": 1, "steps": channel_steps}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
