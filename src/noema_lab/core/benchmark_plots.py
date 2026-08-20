from __future__ import annotations

import csv
import inspect
import json
import math
import platform
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from noema_lab.core.artifacts import file_sha256
from noema_lab.core.benchmarks import BenchmarkError
from noema_lab.core.reproducibility import canonical_json_sha256, utc_now_iso
from noema_lab.core.storage import LocalStore
from noema_lab.core.structured_input import (
    StructuredInputError,
    load_strict_yaml_or_json,
)

JsonDict = Dict[str, Any]

PLOT_TYPES = {"graceful-degradation", "packet-success", "channel-uses"}
PLOT_IMAGE_SUFFIXES = {".pdf", ".png", ".svg"}
_BENCHMARK_PLOT_RENDERER_FUNCTIONS = (
    "_write_plot_data_csv",
    "_draw_plot",
    "_style_axis",
    "_plot_labels",
    "_metric_label",
)
# Schema-v2 plot records identified this whole module. This audited map ties
# that historical module digest to the renderer-source projection it contained.
# A later drawing change alters the live projection and invalidates the mapping.
_LEGACY_BENCHMARK_PLOT_RENDERER_PROJECTIONS = {
    "1c1fcd2ad112fcd0388a0bb98196067de77e1139ffc1df9d33f99747c4f67126": (
        "bc5e09dcc33abf5ccb2ad911f6351b6edecf54639cc14e7cf469c64c6245708e"
    ),
}

SNR_METRICS = [
    "channel.snr_db",
    "steps.wireless_channel.channel.snr_db",
    "steps.wireless_channel.snr_db",
    "wireless.snr_db",
    "snr_db",
]
CHANNEL_USE_METRICS = [
    "channel.uses_per_pixel",
    "channel.channel_uses_per_pixel",
    "steps.wireless_channel.channel.uses_per_pixel",
    "steps.wireless_channel.channel.channel_uses_per_pixel",
    "channel.channel_use_count",
    "steps.wireless_channel.channel.channel_use_count",
]
SCORE_METRICS = [
    "quality.psnr_db",
    "quality.ms_ssim",
    "task.score",
    "task.accuracy",
    "task.exact_match",
    "semantic.lexical_similarity",
    "text.unigram_bleu_proxy",
    "text.edit_similarity",
    "retrieval.recall_at_1",
    "vqa.single_reference_exact_match",
    "detection.f1_at_iou_0p5",
    "segmentation.miou",
    "caption.unigram_bleu_proxy",
    "caption.clip_text_image_score",
]
PACKET_SUCCESS_METRICS = [
    "channel.packet_success_rate",
    "steps.wireless_channel.channel.packet_success_rate",
    "channel.packet_success",
    "steps.wireless_channel.channel.packet_success",
]
OUTAGE_METRICS = [
    "channel.outage_rate",
    "steps.wireless_channel.channel.outage_rate",
    "channel.outage",
    "steps.wireless_channel.channel.outage",
]


def plot_benchmark_result(
    store: LocalStore,
    result_id: str,
    plot_type: str,
    out: Path,
    y_metric: Optional[str] = None,
    x_metric: Optional[str] = None,
    group_by: str = "method",
    method_order: Optional[str | Sequence[str]] = None,
    style: str = "noema",
    style_config: Optional[Path] = None,
    outage_markers: bool = True,
    packet_success_panel: bool = False,
) -> JsonDict:
    if plot_type not in PLOT_TYPES:
        raise BenchmarkError("Unsupported benchmark plot `%s`; choose one of %s" % (plot_type, ", ".join(sorted(PLOT_TYPES))))
    result_dir = store.get_benchmark_result_dir(result_id)
    result = store.get_benchmark_result(result_id)
    _require_complete_plot_source(result, result_id)
    image_path = _resolve_output_path(result_dir, out)
    _ensure_plot_output_available(result, result_dir, image_path, plot_type)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = image_path.with_suffix(".csv")
    canonical_method_order = _method_order_list(method_order)

    style_options = _plot_style(style, style_config)
    rows = _plot_rows(result, plot_type, x_metric=x_metric, y_metric=y_metric, group_by=group_by)
    rows = _aggregate_plot_rows(rows, method_order=canonical_method_order)
    if not rows:
        raise BenchmarkError("No plottable rows found for `%s` in benchmark result %s" % (plot_type, result_id))
    _write_plot_data_csv(data_path, rows)
    _draw_plot(
        image_path,
        rows,
        plot_type,
        style_options=style_options,
        outage_markers=outage_markers,
        packet_success_panel=packet_success_panel,
    )

    plot_record = {
        "kind": "noema.benchmark.plot",
        "id": _plot_id(plot_type, image_path),
        "plot": plot_type,
        "path": str(image_path),
        "relative_path": _relative_to_or_absolute(image_path, result_dir),
        "sha256": file_sha256(image_path),
        "size_bytes": image_path.stat().st_size,
        "data_csv_path": str(data_path),
        "data_csv_relative_path": _relative_to_or_absolute(data_path, result_dir),
        "data_csv_sha256": file_sha256(data_path),
        "data_csv_size_bytes": data_path.stat().st_size,
        "x_metric": rows[0]["x_metric"],
        "y_metric": rows[0]["y_metric"],
        "requested_x_metric": x_metric,
        "requested_y_metric": y_metric,
        "group_by": group_by,
        "method_order": canonical_method_order,
        "format": image_path.suffix.lower().lstrip(".") or "png",
        "point_count": len(rows),
        "style": style,
        "style_options": style_options,
        "renderer": _benchmark_plot_renderer_identity(),
        "selection_status": "exploratory_post_hoc",
        "selection_protocol": None,
        "outage_markers": bool(outage_markers),
        "packet_success_panel": bool(packet_success_panel),
        "generated_at_utc": utc_now_iso(),
    }
    plot_record["semantic_projection_sha256"] = canonical_json_sha256(
        _benchmark_plot_semantic_projection(plot_record, rows)
    )
    plot_record["artifacts"] = {
        "image": {
            "relative_path": plot_record["relative_path"],
            "sha256": plot_record["sha256"],
            "size_bytes": plot_record["size_bytes"],
        },
        "data_csv": {
            "relative_path": plot_record["data_csv_relative_path"],
            "sha256": plot_record["data_csv_sha256"],
            "size_bytes": plot_record["data_csv_size_bytes"],
        },
    }
    result_path = result_dir / "result.json"
    sidecar = {
        "schema_version": 1,
        "kind": "noema.benchmark_plot_evidence",
        "source_result": {
            "result_id": result_id,
            "result_json_sha256": file_sha256(result_path),
            "result_json_size_bytes": int(result_path.stat().st_size),
        },
        "plot": plot_record,
    }
    sidecar["sha256"] = canonical_json_sha256(sidecar)
    record_path = image_path.with_suffix(image_path.suffix + ".plot.json")
    store.write_json(record_path, sidecar)
    return {
        "result_id": result_id,
        "plot": plot_record,
        "plot_evidence": {
            "path": str(record_path),
            "sha256": file_sha256(record_path),
        },
        "reports": dict(result.get("reports") or {}),
    }


def _require_complete_plot_source(
    result: Mapping[str, Any],
    result_id: str,
) -> None:
    status = str(result.get("status") or "").lower()
    if status != "completed":
        raise BenchmarkError(
            "Cannot plot benchmark result %s because its status is %s"
            % (result_id, status or "missing")
        )
    incomplete_rows = [
        str(row.get("id") or row.get("label") or index)
        for index, row in enumerate(result.get("recipes") or [])
        if not isinstance(row, Mapping)
        or str(row.get("status") or "").lower()
        not in {"completed", "rejected_resource_budget"}
    ]
    if incomplete_rows:
        raise BenchmarkError(
            "Cannot plot an incomplete comparison; non-comparable methods: %s"
            % ", ".join(incomplete_rows)
        )


def load_benchmark_plot_records(result_dir: Path) -> List[JsonDict]:
    """Load valid post-run plot records without mutating sealed result.json."""

    result_dir = result_dir.resolve()
    result_path = result_dir / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise BenchmarkError("benchmark result has no safe result.json")
    expected_source = {
        "result_id": result_dir.name,
        "result_json_sha256": file_sha256(result_path),
        "result_json_size_bytes": int(result_path.stat().st_size),
    }
    records: List[JsonDict] = []
    seen = set()
    for record_path in sorted(result_dir.rglob("*.plot.json")):
        if record_path.is_symlink() or not record_path.is_file():
            raise BenchmarkError("benchmark plot evidence path is unsafe")
        try:
            payload = load_strict_yaml_or_json(record_path)
        except (OSError, StructuredInputError) as exc:
            raise BenchmarkError(
                "benchmark plot evidence is unreadable: %s" % record_path
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("kind") != "noema.benchmark_plot_evidence"
            or payload.get("source_result") != expected_source
            or payload.get("sha256")
            != canonical_json_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "sha256"
                }
            )
            or not isinstance(payload.get("plot"), Mapping)
        ):
            raise BenchmarkError(
                "benchmark plot evidence is invalid or bound to another result: %s"
                % record_path
            )
        plot = dict(payload["plot"])
        plot_id = str(plot.get("id") or "")
        if not plot_id or plot_id in seen:
            raise BenchmarkError(
                "benchmark plot evidence has a missing or duplicate id"
            )
        seen.add(plot_id)
        records.append(plot)
    return records


def benchmark_plot_semantic_projection(
    result: Mapping[str, Any],
    plot_record: Mapping[str, Any],
) -> JsonDict:
    """Return the renderer-independent scientific projection for a plot."""

    rows = _reproduce_benchmark_plot_rows(result, plot_record)
    return _benchmark_plot_semantic_projection(plot_record, rows)


def benchmark_plot_semantic_projection_sha256(
    result: Mapping[str, Any],
    plot_record: Mapping[str, Any],
) -> str:
    """Identify selected ordered plot rows independently of rendered bytes."""

    return canonical_json_sha256(
        benchmark_plot_semantic_projection(result, plot_record)
    )


def reproduce_benchmark_plot_artifacts(
    result: Mapping[str, Any],
    plot_record: Mapping[str, Any],
    *,
    image_path: Path,
    data_path: Path,
) -> List[JsonDict]:
    """Rebuild exact plot inputs and rendered bytes from a stored result record."""

    renderer = plot_record.get("renderer")
    current_renderer = _benchmark_plot_renderer_identity()
    if not isinstance(renderer, Mapping) or not _benchmark_plot_renderer_matches(
        renderer,
        current_renderer,
    ):
        raise BenchmarkError(
            "benchmark plot renderer implementation/environment identity differs "
            "from the recorded renderer"
        )
    style_options = plot_record.get("style_options")
    if not isinstance(style_options, Mapping):
        raise BenchmarkError("benchmark plot style_options must be an object")
    rows = _reproduce_benchmark_plot_rows(result, plot_record)
    if "semantic_projection_sha256" in plot_record:
        declared_semantic_sha = plot_record.get("semantic_projection_sha256")
        reproduced_semantic_sha = canonical_json_sha256(
            _benchmark_plot_semantic_projection(plot_record, rows)
        )
        if declared_semantic_sha != reproduced_semantic_sha:
            raise BenchmarkError(
                "benchmark plot semantic projection identity differs from the "
                "reproduced selected rows and selection contract"
            )
    plot_type = str(plot_record.get("plot") or "")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _write_plot_data_csv(data_path, rows)
    _draw_plot(
        image_path,
        rows,
        plot_type,
        style_options=dict(style_options),
        outage_markers=bool(plot_record.get("outage_markers")),
        packet_success_panel=bool(plot_record.get("packet_success_panel")),
    )
    return rows


def _reproduce_benchmark_plot_rows(
    result: Mapping[str, Any],
    plot_record: Mapping[str, Any],
) -> List[JsonDict]:
    if (
        plot_record.get("selection_status") != "exploratory_post_hoc"
        or plot_record.get("selection_protocol") is not None
    ):
        raise BenchmarkError(
            "benchmark plot selection must be explicitly labeled exploratory "
            "unless a frozen plot-protocol binding is implemented"
        )
    plot_type = str(plot_record.get("plot") or "")
    if plot_type not in PLOT_TYPES:
        raise BenchmarkError("benchmark plot type is missing or unsupported")
    group_by = str(plot_record.get("group_by") or "method")
    requested_x = plot_record.get("requested_x_metric")
    requested_y = plot_record.get("requested_y_metric")
    if requested_x is not None and not isinstance(requested_x, str):
        raise BenchmarkError("benchmark plot requested_x_metric must be a string or null")
    if requested_y is not None and not isinstance(requested_y, str):
        raise BenchmarkError("benchmark plot requested_y_metric must be a string or null")
    raw_method_order = plot_record.get("method_order")
    if not isinstance(raw_method_order, list):
        raise BenchmarkError("benchmark plot method_order must be an array of labels")
    canonical_method_order = _method_order_list(raw_method_order)
    if canonical_method_order != raw_method_order:
        raise BenchmarkError(
            "benchmark plot method_order must contain canonical, unique labels"
        )
    rows = _plot_rows(
        dict(result),
        plot_type,
        x_metric=requested_x,
        y_metric=requested_y,
        group_by=group_by,
    )
    rows = _aggregate_plot_rows(rows, method_order=canonical_method_order)
    if not rows:
        raise BenchmarkError("benchmark plot cannot be reproduced without data rows")
    declared_x = str(plot_record.get("x_metric") or "")
    declared_y = str(plot_record.get("y_metric") or "")
    if any(
        str(row.get("x_metric") or "") != declared_x
        or str(row.get("y_metric") or "") != declared_y
        for row in rows
    ):
        raise BenchmarkError(
            "benchmark plot metric projection differs from the reproduced result rows"
        )
    return rows


def _benchmark_plot_semantic_projection(
    plot_record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Canonical scientific content, excluding renderer and file representation."""

    return {
        "schema_version": 1,
        "kind": "noema.benchmark_plot.semantic_projection",
        "plot": str(plot_record.get("plot") or ""),
        "selection": {
            "status": plot_record.get("selection_status"),
            "protocol": plot_record.get("selection_protocol"),
        },
        "axes": {
            "requested_x_metric": plot_record.get("requested_x_metric"),
            "requested_y_metric": plot_record.get("requested_y_metric"),
            "x_metric": str(plot_record.get("x_metric") or ""),
            "y_metric": str(plot_record.get("y_metric") or ""),
        },
        "grouping": {
            "group_by": str(plot_record.get("group_by") or "method"),
            "method_order": list(plot_record.get("method_order") or []),
        },
        "display_selection": {
            "outage_markers": bool(plot_record.get("outage_markers")),
            "packet_success_panel": bool(
                plot_record.get("packet_success_panel")
            ),
        },
        "rows": [dict(row) for row in rows],
    }


def _plot_rows(
    result: JsonDict,
    plot_type: str,
    x_metric: Optional[str] = None,
    y_metric: Optional[str] = None,
    group_by: str = "method",
) -> List[JsonDict]:
    benchmark = dict(result.get("benchmark") or {})
    recipes = [
        recipe
        for recipe in _recipes(result)
        if str(recipe.get("status") or "").strip().lower() == "completed"
        and _resource_admitted(recipe)
    ]
    if not recipes:
        return []
    metrics_rows = [dict(recipe.get("metrics") or {}) for recipe in recipes]
    if plot_type in {"graceful-degradation", "packet-success"}:
        x_key = _common_numeric_metric(
            metrics_rows,
            [x_metric] if x_metric else SNR_METRICS,
            "x-axis",
        )
    else:
        x_key = _common_numeric_metric(
            metrics_rows,
            [x_metric] if x_metric else CHANNEL_USE_METRICS,
            "x-axis",
        )
    packet_y_invert = False
    if plot_type == "packet-success":
        if y_metric:
            y_key = _common_numeric_metric(metrics_rows, [y_metric], "y-axis")
        else:
            y_key = _optional_common_numeric_metric(
                metrics_rows, PACKET_SUCCESS_METRICS
            )
            if y_key is None:
                outage_key = _optional_common_numeric_metric(
                    metrics_rows, OUTAGE_METRICS
                )
                if outage_key is None:
                    raise BenchmarkError(
                        "No single packet-success or outage metric is present in every "
                        "completed, resource-admitted recipe"
                    )
                y_key = outage_key
                packet_y_invert = True
    else:
        y_key = _common_numeric_metric(
            metrics_rows,
            [y_metric] if y_metric else SCORE_METRICS,
            "y-axis",
        )
    rows: List[JsonDict] = []
    for recipe, metrics in zip(recipes, metrics_rows):
        x_value = _as_number(metrics.get(x_key))
        y_value = _as_number(metrics.get(y_key))
        if x_value is None or y_value is None:
            raise BenchmarkError("Common plot metric selection became inconsistent")
        plotted_y_key = "1 - %s" % y_key if packet_y_invert else y_key
        plotted_y_value = 1.0 - y_value if packet_y_invert else y_value
        series = _series_label(recipe, metrics, group_by)
        packet_key, packet_success = _packet_success_metric(metrics, None)
        outage_key, outage_rate = _outage_metric(metrics)
        rows.append(
            {
                "benchmark_id": benchmark.get("id") or "",
                "benchmark_version": benchmark.get("version") or "",
                "plot": plot_type,
                "series": series,
                "recipe_id": recipe.get("id") or "",
                "recipe_label": recipe.get("label") or recipe.get("recipe_name") or recipe.get("id") or "",
                "recipe_role": recipe.get("role") or "",
                "run_id": recipe.get("run_id") or "",
                "status": recipe.get("status") or "",
                "pairing_id": _pairing_id(recipe, metrics),
                "aggregation_cell_id": _aggregation_cell_id(recipe, metrics),
                "statistical_unit": _statistical_unit(recipe),
                "x_metric": x_key,
                "x_value": float(x_value),
                "y_metric": plotted_y_key,
                "y_value": float(plotted_y_value),
                "packet_success_metric": packet_key or "",
                "packet_success_rate": packet_success,
                "outage_metric": outage_key or "",
                "outage_rate": outage_rate,
            }
        )
    rows.sort(key=lambda item: (str(item["series"]), float(item["x_value"]), str(item["recipe_id"])))
    return rows


def _aggregate_plot_rows(rows: List[JsonDict], method_order: Optional[str | Sequence[str]] = None) -> List[JsonDict]:
    grouped: Dict[Tuple[str, str, float, str], List[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["series"]), str(row["x_metric"]), float(row["x_value"]), str(row["y_metric"]))].append(row)

    output: List[JsonDict] = []
    for (series, x_metric, x_value, y_metric), group_rows in grouped.items():
        aggregation = _validate_aggregation_group(
            group_rows,
            series=series,
            x_metric=x_metric,
            x_value=x_value,
            y_metric=y_metric,
        )
        y_values = [float(row["y_value"]) for row in group_rows]
        packet_values = [float(row["packet_success_rate"]) for row in group_rows if row.get("packet_success_rate") is not None]
        outage_values = [float(row["outage_rate"]) for row in group_rows if row.get("outage_rate") is not None]
        y_mean, y_std, y_ci95 = _mean_std_ci95(y_values)
        packet_mean, packet_std, _ = _mean_std_ci95(packet_values) if packet_values else (None, None, None)
        outage_mean = _mean(outage_values) if outage_values else None
        output.append(
            {
                "benchmark_id": group_rows[0].get("benchmark_id") or "",
                "benchmark_version": group_rows[0].get("benchmark_version") or "",
                "plot": group_rows[0].get("plot") or "",
                "series": series,
                "recipe_id": "|".join(str(row.get("recipe_id") or "") for row in group_rows),
                "recipe_label": group_rows[0].get("recipe_label") or "",
                "recipe_role": group_rows[0].get("recipe_role") or "",
                "run_id": "|".join(str(row.get("run_id") or "") for row in group_rows),
                "status": _combined_status(group_rows),
                "x_metric": x_metric,
                "x_value": float(x_value),
                "y_metric": y_metric,
                "y_value": y_mean,
                "y_std": y_std,
                "y_ci95": y_ci95,
                "y_ci95_method": _ci95_method(len(y_values)),
                "sample_count": len(group_rows),
                "aggregation_cell_id": aggregation["aggregation_cell_id"],
                "aggregation_status": aggregation["status"],
                "statistical_unit": aggregation["statistical_unit"],
                "sampling_unit_ids": "|".join(aggregation["sampling_unit_ids"]),
                "ci_scope": (
                    "between-statistical-unit"
                    if len(group_rows) >= 2
                    else "not_estimable_single_observation"
                ),
                "packet_success_metric": group_rows[0].get("packet_success_metric") or "",
                "packet_success_rate": packet_mean,
                "packet_success_std": packet_std,
                "outage_metric": group_rows[0].get("outage_metric") or "",
                "outage_rate": outage_mean,
                "outage_marker": 1 if outage_mean is not None and outage_mean > 0.0 else 0,
                "source_recipe_ids": "|".join(str(row.get("recipe_id") or "") for row in group_rows),
                "source_run_ids": "|".join(str(row.get("run_id") or "") for row in group_rows),
            }
        )
    _attach_paired_contrasts(output, rows)
    order = _method_order_map(method_order)
    output.sort(key=lambda item: (_series_order(str(item["series"]), order), float(item["x_value"]), str(item["series"])))
    return output


def _packet_success_metric(metrics: JsonDict, y_metric: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if y_metric:
        return _first_numeric(metrics, [y_metric])
    key, value = _first_numeric(metrics, PACKET_SUCCESS_METRICS)
    if key is not None:
        return key, value
    key, value = _first_numeric(metrics, OUTAGE_METRICS)
    if key is None or value is None:
        return None, None
    return "1 - %s" % key, 1.0 - float(value)


def _outage_metric(metrics: JsonDict) -> Tuple[Optional[str], Optional[float]]:
    key, value = _first_numeric(metrics, OUTAGE_METRICS)
    if key is not None:
        return key, value
    key, value = _first_numeric(metrics, PACKET_SUCCESS_METRICS)
    if key is None or value is None:
        return None, None
    return "1 - %s" % key, max(0.0, min(1.0, 1.0 - float(value)))


def _write_plot_data_csv(path: Path, rows: Iterable[JsonDict]) -> None:
    fieldnames = [
        "benchmark_id",
        "benchmark_version",
        "plot",
        "series",
        "recipe_id",
        "recipe_label",
        "recipe_role",
        "run_id",
        "status",
        "x_metric",
        "x_value",
        "y_metric",
        "y_value",
        "y_std",
        "y_ci95",
        "y_ci95_method",
        "sample_count",
        "aggregation_cell_id",
        "aggregation_status",
        "statistical_unit",
        "sampling_unit_ids",
        "ci_scope",
        "paired_reference_series",
        "paired_difference_mean",
        "paired_difference_std",
        "paired_difference_ci95",
        "paired_difference_ci95_method",
        "paired_sample_count",
        "paired_ids",
        "paired_contrast_status",
        "packet_success_metric",
        "packet_success_rate",
        "packet_success_std",
        "outage_metric",
        "outage_rate",
        "outage_marker",
        "source_recipe_ids",
        "source_run_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _draw_plot(
    path: Path,
    rows: List[JsonDict],
    plot_type: str,
    style_options: JsonDict,
    outage_markers: bool,
    packet_success_panel: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise BenchmarkError("Install matplotlib to export benchmark plots") from exc

    title, x_label, y_label = _plot_labels(plot_type, rows)
    has_packet_panel = bool(packet_success_panel and plot_type == "graceful-degradation" and any(row.get("packet_success_rate") is not None for row in rows))
    width = float(style_options.get("figure_width", 7.2))
    height = float(style_options.get("figure_height", 4.4))
    if has_packet_panel:
        height = max(height, 5.8)
    dpi = int(style_options.get("dpi", 220))
    if has_packet_panel:
        fig, axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi, sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08})
        ax = axes[0]
        packet_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        packet_ax = None
    fig.patch.set_facecolor("white")
    palette = list(style_options.get("palette") or ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be185d", "#4b5563"])
    grouped: Dict[str, List[JsonDict]] = {}
    for row in rows:
        grouped.setdefault(str(row["series"] or row["recipe_label"] or row["recipe_id"]), []).append(row)
    for index, (series, points) in enumerate(grouped.items()):
        points = sorted(points, key=lambda item: float(item["x_value"]))
        x_values = [float(item["x_value"]) for item in points]
        y_values = [float(item["y_value"]) for item in points]
        y_err = [float(item.get("y_ci95") or 0.0) for item in points]
        color = palette[index % len(palette)]
        line_style = "-" if len(points) > 1 else "None"
        if any(value > 0.0 for value in y_err):
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_err,
                marker="o",
                markersize=float(style_options.get("marker_size", 5.2)),
                linewidth=float(style_options.get("line_width", 2.1)),
                linestyle=line_style,
                color=color,
                capsize=float(style_options.get("errorbar_capsize", 3.0)),
                label=series,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=float(style_options.get("marker_size", 5.2)),
                linewidth=float(style_options.get("line_width", 2.1)),
                linestyle=line_style,
                color=color,
                label=series,
            )
        if outage_markers:
            outage_points = [item for item in points if float(item.get("outage_rate") or 0.0) > 0.0]
            if outage_points:
                ax.scatter(
                    [float(item["x_value"]) for item in outage_points],
                    [float(item["y_value"]) for item in outage_points],
                    marker="x",
                    s=float(style_options.get("outage_marker_size", 58.0)),
                    linewidths=float(style_options.get("outage_marker_width", 1.7)),
                    color=str(style_options.get("outage_color", "#111827")),
                    zorder=5,
                )
        if packet_ax is not None:
            packet_points = [item for item in points if item.get("packet_success_rate") is not None]
            if packet_points:
                packet_ax.plot(
                    [float(item["x_value"]) for item in packet_points],
                    [float(item["packet_success_rate"]) for item in packet_points],
                    marker="s",
                    markersize=float(style_options.get("packet_marker_size", 4.3)),
                    linewidth=float(style_options.get("packet_line_width", 1.6)),
                    linestyle=line_style,
                    color=color,
                )
    ax.set_title(title, fontsize=float(style_options.get("title_size", 13)), weight="bold", pad=10)
    if packet_ax is None:
        ax.set_xlabel(x_label, fontsize=float(style_options.get("label_size", 11)))
    ax.set_ylabel(y_label, fontsize=float(style_options.get("label_size", 11)))
    _style_axis(ax, style_options)
    if plot_type == "packet-success":
        ax.set_ylim(-0.03, 1.03)
    if packet_ax is not None:
        packet_ax.set_xlabel(x_label, fontsize=float(style_options.get("label_size", 11)))
        packet_ax.set_ylabel("Packet success", fontsize=float(style_options.get("small_label_size", 9.5)))
        packet_ax.set_ylim(-0.05, 1.05)
        _style_axis(packet_ax, style_options)
    ax.legend(loc=str(style_options.get("legend_loc", "best")), fontsize=float(style_options.get("legend_size", 8.5)), frameon=True, framealpha=0.92)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    metadata: Optional[JsonDict]
    suffix = path.suffix.lower()
    if suffix == ".png":
        metadata = {"Software": "Noema benchmark plot renderer v1"}
    elif suffix == ".svg":
        metadata = {
            "Creator": "Noema benchmark plot renderer v1",
            "Date": None,
        }
    elif suffix == ".pdf":
        metadata = {
            "Creator": "Noema benchmark plot renderer v1",
            "CreationDate": None,
            "ModDate": None,
        }
    else:
        metadata = None
    previous_hashsalt = matplotlib.rcParams.get("svg.hashsalt")
    matplotlib.rcParams["svg.hashsalt"] = "noema-benchmark-plot-v1"
    try:
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
    finally:
        matplotlib.rcParams["svg.hashsalt"] = previous_hashsalt
        plt.close(fig)


def _benchmark_plot_renderer_identity() -> JsonDict:
    """Identify the implementation and runtime that must reproduce plot bytes."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import font_manager, ft2font
    except Exception as exc:
        raise BenchmarkError("Install matplotlib to identify the plot renderer") from exc
    font_path = Path(font_manager.findfont(font_manager.FontProperties())).resolve()
    rc_projection = {
        key: list(value) if isinstance(value, (list, tuple)) else value
        for key, value in {
            "font.family": matplotlib.rcParams.get("font.family"),
            "font.sans-serif": matplotlib.rcParams.get("font.sans-serif"),
            "font.size": matplotlib.rcParams.get("font.size"),
            "axes.titlesize": matplotlib.rcParams.get("axes.titlesize"),
            "axes.labelsize": matplotlib.rcParams.get("axes.labelsize"),
            "lines.antialiased": matplotlib.rcParams.get("lines.antialiased"),
            "path.simplify": matplotlib.rcParams.get("path.simplify"),
        }.items()
    }
    return {
        "id": "noema.matplotlib_benchmark_plot",
        "schema_version": 3,
        "implementation_sha256": _benchmark_plot_renderer_source_sha256(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "matplotlib_version": str(matplotlib.__version__),
        "backend": str(matplotlib.get_backend()).lower(),
        "freetype_version": str(getattr(ft2font, "__freetype_version__", "")),
        "default_font": {
            "filename": font_path.name,
            "sha256": file_sha256(font_path),
        },
        "rcparams_sha256": canonical_json_sha256(rc_projection),
    }


def _benchmark_plot_renderer_source_sha256() -> str:
    """Identify only code that serializes plot CSV and image bytes."""

    projection = {
        "schema_version": 1,
        "functions": [
            {
                "name": name,
                "source": inspect.getsource(globals()[name]),
            }
            for name in _BENCHMARK_PLOT_RENDERER_FUNCTIONS
        ],
    }
    return canonical_json_sha256(projection)


def _benchmark_plot_renderer_matches(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Accept exact current identity or an audited equivalent legacy renderer."""

    if dict(recorded) == dict(current):
        return True
    if recorded.get("schema_version") != 2 or current.get("schema_version") != 3:
        return False
    recorded_sha = str(recorded.get("implementation_sha256") or "")
    expected_projection = _LEGACY_BENCHMARK_PLOT_RENDERER_PROJECTIONS.get(
        recorded_sha
    )
    if expected_projection != current.get("implementation_sha256"):
        return False
    # Python itself does not serialize these plots. Permit the supported
    # interpreter range for this audited schema-v2 migration only; Matplotlib,
    # FreeType, font, rcParams, renderer source, and final artifact bytes remain
    # independently checked.
    excluded = {"schema_version", "implementation_sha256", "python_version"}
    recorded_representation = {
        key: value for key, value in recorded.items() if key not in excluded
    }
    current_representation = {
        key: value for key, value in current.items() if key not in excluded
    }
    return recorded_representation == current_representation


def _style_axis(ax: Any, style_options: JsonDict) -> None:
    ax.grid(True, which="major", color=str(style_options.get("grid_color", "#d1d5db")), linewidth=float(style_options.get("grid_width", 0.8)), alpha=float(style_options.get("grid_alpha", 0.8)))
    ax.grid(True, which="minor", color=str(style_options.get("minor_grid_color", "#e5e7eb")), linewidth=float(style_options.get("minor_grid_width", 0.5)), alpha=float(style_options.get("minor_grid_alpha", 0.5)))
    ax.tick_params(axis="both", labelsize=float(style_options.get("tick_size", 9)))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_labels(plot_type: str, rows: List[JsonDict]) -> Tuple[str, str, str]:
    if plot_type == "packet-success":
        return "Packet Success vs SNR", _metric_label(rows[0]["x_metric"]), _metric_label(rows[0]["y_metric"])
    if plot_type == "channel-uses":
        return "Task Score vs Channel Use", _metric_label(rows[0]["x_metric"]), _metric_label(rows[0]["y_metric"])
    return "Graceful Degradation", _metric_label(rows[0]["x_metric"]), _metric_label(rows[0]["y_metric"])


def _metric_label(metric: str) -> str:
    labels = {
        "channel.snr_db": "SNR (dB)",
        "quality.psnr_db": "PSNR (dB)",
        "quality.ms_ssim": "MS-SSIM",
        "task.accuracy": "Task accuracy",
        "task.score": "Task score",
        "semantic.lexical_similarity": "Token F1 lexical similarity",
        "text.unigram_bleu_proxy": "Unigram BLEU proxy",
        "retrieval.recall_at_1": "Recall@1",
        "vqa.single_reference_exact_match": "VQA single-reference exact match",
        "detection.f1_at_iou_0p5": "Detection F1 at IoU 0.5",
        "channel.uses_per_pixel": "Channel uses / pixel",
        "channel.channel_use_count": "Channel uses / sample",
        "channel.packet_success_rate": "Packet success rate",
        "channel.outage_rate": "Outage rate",
    }
    return labels.get(metric, metric.replace("steps.wireless_channel.", "").replace("_", " "))


def _first_numeric(metrics: JsonDict, keys: Iterable[Optional[str]]) -> Tuple[Optional[str], Optional[float]]:
    for key in keys:
        if not key:
            continue
        if key not in metrics:
            continue
        value = _as_number(metrics.get(key))
        if value is not None:
            return str(key), value
    return None, None


def _optional_common_numeric_metric(
    metrics_rows: Sequence[Mapping[str, Any]],
    keys: Iterable[Optional[str]],
) -> Optional[str]:
    for key in keys:
        if not key:
            continue
        if all(_as_number(metrics.get(str(key))) is not None for metrics in metrics_rows):
            return str(key)
    return None


def _common_numeric_metric(
    metrics_rows: Sequence[Mapping[str, Any]],
    keys: Iterable[Optional[str]],
    axis: str,
) -> str:
    candidate_keys = [str(key) for key in keys if key]
    selected = _optional_common_numeric_metric(metrics_rows, candidate_keys)
    if selected is None:
        raise BenchmarkError(
            "No single %s metric from [%s] is present and numeric in every completed, "
            "resource-admitted recipe"
            % (axis, ", ".join(candidate_keys))
        )
    return selected


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _series_label(recipe: JsonDict, metrics: JsonDict, group_by: str) -> str:
    key = str(group_by or "method")
    if key in {"method", "label"}:
        return str(recipe.get("label") or recipe.get("recipe_name") or recipe.get("id") or "method")
    if key in {"recipe", "id", "recipe_id"}:
        return str(recipe.get("id") or recipe.get("recipe_name") or "recipe")
    if key == "role":
        return str(recipe.get("role") or "role")
    if key.startswith("metric:"):
        metric_key = key.split(":", 1)[1]
        return str(metrics.get(metric_key, "missing:%s" % metric_key))
    if key in recipe:
        return str(recipe.get(key) or key)
    if key in metrics:
        return str(metrics.get(key))
    return str(recipe.get("label") or recipe.get("recipe_name") or recipe.get("id") or "method")


def _plot_style(style: str, style_config: Optional[Path]) -> JsonDict:
    base: JsonDict = {
        "figure_width": 7.2,
        "figure_height": 4.4,
        "dpi": 220,
        "line_width": 2.1,
        "marker_size": 5.2,
        "palette": ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be185d", "#4b5563"],
    }
    if style == "paper":
        base.update({"figure_width": 6.4, "figure_height": 4.0, "dpi": 300, "line_width": 1.9, "marker_size": 4.8, "legend_size": 8.0})
    elif style == "compact":
        base.update({"figure_width": 5.2, "figure_height": 3.4, "dpi": 220, "legend_size": 7.5, "label_size": 9.5, "tick_size": 8.0})
    if style_config is not None:
        config = _load_style_config(style_config)
        base.update(config)
    return base


def _load_style_config(path: Path) -> JsonDict:
    try:
        value = load_strict_yaml_or_json(path)
    except (OSError, StructuredInputError) as exc:
        raise BenchmarkError("Plot style config is invalid: %s" % exc) from exc
    if not isinstance(value, Mapping):
        raise BenchmarkError("Plot style config must contain a mapping")
    return dict(value)


def _method_order_list(
    method_order: Optional[str | Sequence[str]],
) -> List[str]:
    if method_order is None:
        return []
    if isinstance(method_order, str):
        items = [item.strip() for item in method_order.split(",") if item.strip()]
    else:
        if any(not isinstance(item, str) for item in method_order):
            raise BenchmarkError("benchmark plot method_order labels must be strings")
        items = [item.strip() for item in method_order]
        if any(not item for item in items):
            raise BenchmarkError(
                "benchmark plot method_order labels cannot be empty"
            )
    if len(set(items)) != len(items):
        raise BenchmarkError(
            "benchmark plot method_order labels must be unique"
        )
    return items


def _method_order_map(method_order: Optional[str | Sequence[str]]) -> Dict[str, int]:
    items = _method_order_list(method_order)
    return {item: index for index, item in enumerate(items)}


def _series_order(series: str, order: Dict[str, int]) -> Tuple[int, str]:
    if series in order:
        return order[series], series
    return len(order) + 1, series


def _resource_admitted(recipe: Mapping[str, Any]) -> bool:
    status = str(recipe.get("status") or "").strip().lower()
    if status in {"rejected_resource_budget", "inadmissible", "resource_budget_rejected"}:
        return False
    admission = recipe.get("resource_admission")
    if isinstance(admission, Mapping) and admission.get("admitted") is False:
        return False
    metrics = recipe.get("metrics")
    if isinstance(metrics, Mapping) and "benchmark.resource_budget.admitted" in metrics:
        value = metrics.get("benchmark.resource_budget.admitted")
        if value is False or value == 0 or str(value).strip().lower() in {"false", "rejected"}:
            return False
    return True


def _pairing_id(recipe: Mapping[str, Any], metrics: Mapping[str, Any]) -> Optional[str]:
    for key in ("pairing_id", "pairing_seed", "benchmark_paired_seed", "paired_seed"):
        value = recipe.get(key)
        if value is not None and str(value) != "":
            return str(value)
    for key in ("benchmark.pairing_id", "benchmark.paired_seed", "paired_seed"):
        value = metrics.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _aggregation_cell_id(
    recipe: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Optional[str]:
    for key in ("aggregation_cell_id", "benchmark_aggregation_cell_id"):
        value = recipe.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in (
        "benchmark.aggregation_cell_id",
        "benchmark_aggregation_cell_id",
    ):
        value = metrics.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _statistical_unit(recipe: Mapping[str, Any]) -> str:
    value = recipe.get("statistical_unit")
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        raise BenchmarkError("statistical_unit must be a JSON-compatible value")


def _validate_aggregation_group(
    rows: List[JsonDict],
    *,
    series: str,
    x_metric: str,
    x_value: float,
    y_metric: str,
) -> JsonDict:
    """Reject pseudo-replication and undeclared cross-condition pooling."""

    if len(rows) == 1:
        pairing_id = rows[0].get("pairing_id")
        return {
            "aggregation_cell_id": str(rows[0].get("aggregation_cell_id") or ""),
            "status": "single_observation",
            "statistical_unit": str(rows[0].get("statistical_unit") or ""),
            "sampling_unit_ids": [str(pairing_id)] if pairing_id not in (None, "") else [],
        }

    context = "%s at %s=%s for %s" % (series, x_metric, x_value, y_metric)
    cell_ids = {
        str(row.get("aggregation_cell_id") or "").strip() for row in rows
    }
    if "" in cell_ids or len(cell_ids) != 1:
        raise BenchmarkError(
            "Cannot aggregate %s: repeated observations require one shared, explicit "
            "aggregation_cell_id" % context
        )
    statistical_units = {
        str(row.get("statistical_unit") or "").strip() for row in rows
    }
    if "" in statistical_units or len(statistical_units) != 1:
        raise BenchmarkError(
            "Cannot aggregate %s: repeated observations require one shared, explicit "
            "statistical_unit declaration" % context
        )
    pairing_ids = [str(row.get("pairing_id") or "").strip() for row in rows]
    if any(not value for value in pairing_ids) or len(set(pairing_ids)) != len(pairing_ids):
        raise BenchmarkError(
            "Cannot aggregate %s: every observation requires a unique pairing_id "
            "identifying its statistical unit" % context
        )
    run_ids = [str(row.get("run_id") or "").strip() for row in rows]
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise BenchmarkError(
            "Cannot aggregate %s: every observation must come from a distinct run_id"
            % context
        )
    roles = {str(row.get("recipe_role") or "").strip().lower() for row in rows}
    if len(roles) != 1:
        raise BenchmarkError(
            "Cannot aggregate %s: observations have heterogeneous recipe roles" % context
        )
    return {
        "aggregation_cell_id": next(iter(cell_ids)),
        "status": "declared_unique_statistical_units",
        "statistical_unit": next(iter(statistical_units)),
        "sampling_unit_ids": sorted(pairing_ids),
    }


def _attach_paired_contrasts(
    aggregated: List[JsonDict], raw_rows: List[JsonDict]
) -> None:
    for row in aggregated:
        row.update(
            {
                "paired_reference_series": None,
                "paired_difference_mean": None,
                "paired_difference_std": None,
                "paired_difference_ci95": None,
                "paired_difference_ci95_method": "not_available",
                "paired_sample_count": 0,
                "paired_ids": "",
                "paired_contrast_status": "no_declared_reference",
            }
        )
    cells: Dict[Tuple[str, float, str], List[JsonDict]] = defaultdict(list)
    for row in raw_rows:
        cells[
            (str(row["x_metric"]), float(row["x_value"]), str(row["y_metric"]))
        ].append(row)
    aggregate_index = {
        (
            str(row["series"]),
            str(row["x_metric"]),
            float(row["x_value"]),
            str(row["y_metric"]),
        ): row
        for row in aggregated
    }
    reference_roles = {"baseline", "reference", "control"}
    for (x_metric, x_value, y_metric), cell_rows in cells.items():
        reference_series = {
            str(row["series"])
            for row in cell_rows
            if str(row.get("recipe_role") or "").strip().lower() in reference_roles
        }
        if len(reference_series) != 1:
            status = "ambiguous_reference" if reference_series else "no_declared_reference"
            for row in cell_rows:
                target = aggregate_index.get(
                    (str(row["series"]), x_metric, x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = status
            continue
        reference_name = next(iter(reference_series))
        by_series: Dict[str, List[JsonDict]] = defaultdict(list)
        for row in cell_rows:
            by_series[str(row["series"])].append(row)
        reference_output = aggregate_index.get(
            (reference_name, x_metric, x_value, y_metric)
        )
        if len(by_series) == 1:
            if reference_output is not None:
                reference_output["paired_reference_series"] = reference_name
                reference_output["paired_contrast_status"] = "reference"
            continue
        comparison_cells = {
            (
                str(row.get("aggregation_cell_id") or "").strip(),
                str(row.get("statistical_unit") or "").strip(),
            )
            for row in cell_rows
        }
        if any(not cell_id or not unit for cell_id, unit in comparison_cells):
            for row in cell_rows:
                target = aggregate_index.get(
                    (str(row["series"]), x_metric, x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = "missing_comparison_cell"
            continue
        if len(comparison_cells) != 1:
            for row in cell_rows:
                target = aggregate_index.get(
                    (str(row["series"]), x_metric, x_value, y_metric)
                )
                if target is not None:
                    target["paired_contrast_status"] = "comparison_cell_mismatch"
            continue
        reference_values = _unique_pair_values(by_series[reference_name])
        if reference_output is not None:
            reference_output["paired_reference_series"] = reference_name
            reference_output["paired_contrast_status"] = "reference"
        if reference_values is None or not reference_values:
            continue
        for series, series_rows in by_series.items():
            if series == reference_name:
                continue
            target = aggregate_index.get((series, x_metric, x_value, y_metric))
            if target is None:
                continue
            target["paired_reference_series"] = reference_name
            candidate_values = _unique_pair_values(series_rows)
            if candidate_values is None or not candidate_values:
                target["paired_contrast_status"] = "missing_or_duplicate_pairing_ids"
                continue
            if set(candidate_values) != set(reference_values):
                target["paired_contrast_status"] = "pairing_set_mismatch"
                continue
            pairing_ids = sorted(reference_values)
            differences = [
                candidate_values[pairing_id] - reference_values[pairing_id]
                for pairing_id in pairing_ids
            ]
            mean, std, ci95 = _mean_std_ci95(differences)
            target.update(
                {
                    "paired_difference_mean": mean,
                    "paired_difference_std": std,
                    "paired_difference_ci95": ci95,
                    "paired_difference_ci95_method": _ci95_method(len(differences)),
                    "paired_sample_count": len(differences),
                    "paired_ids": "|".join(pairing_ids),
                    "paired_contrast_status": "computed",
                }
            )


def _unique_pair_values(rows: List[JsonDict]) -> Optional[Dict[str, float]]:
    values: Dict[str, float] = {}
    for row in rows:
        pairing_id = row.get("pairing_id")
        if pairing_id is None or str(pairing_id) == "":
            return None
        key = str(pairing_id)
        if key in values:
            return None
        values[key] = float(row["y_value"])
    return values


def _ci95_method(sample_count: int) -> str:
    return (
        "student_t_95_two_sided_tabulated_conservative_approximation"
        if sample_count >= 2
        else "not_estimable_n_lt_2"
    )


def _mean(values: List[float]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def _mean_std_ci95(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    mean = _mean(values)
    if len(values) < 2:
        return mean, None, None
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    ci95 = _student_t_critical_975(len(values) - 1) * std / math.sqrt(float(len(values)))
    return mean, std, ci95


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
        31: 2.040,
        32: 2.037,
        33: 2.035,
        34: 2.032,
        35: 2.030,
        36: 2.028,
        37: 2.026,
        38: 2.024,
        39: 2.023,
        40: 2.021,
    }
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom <= 60:
        return 2.021
    if degrees_of_freedom <= 120:
        return 2.000
    if degrees_of_freedom <= 1000:
        return 1.980
    return 1.960


def _combined_status(rows: List[JsonDict]) -> str:
    statuses = sorted({str(row.get("status") or "") for row in rows if row.get("status")})
    return statuses[0] if len(statuses) == 1 else "mixed"


def _recipes(result: JsonDict) -> List[JsonDict]:
    rows = result.get("recipes") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _resolve_output_path(result_dir: Path, out: Path) -> Path:
    if out.is_absolute():
        raise BenchmarkError("benchmark plot output must be bundle-relative")
    if not out.name or out.name in {".", ".."} or ".." in out.parts:
        raise BenchmarkError(
            "benchmark plot output must be a named path inside the result bundle"
        )
    if out.suffix.lower() not in PLOT_IMAGE_SUFFIXES:
        raise BenchmarkError(
            "benchmark plot output suffix must be one of %s"
            % ", ".join(sorted(PLOT_IMAGE_SUFFIXES))
        )
    root = result_dir.resolve()
    unresolved = result_dir / out
    cursor = result_dir
    for part in out.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkError("benchmark plot output cannot traverse a symbolic link")
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BenchmarkError(
            "benchmark plot output must remain inside the result bundle"
        ) from exc
    return candidate


def _ensure_plot_output_available(
    result: Mapping[str, Any],
    result_dir: Path,
    image_path: Path,
    plot_type: str,
) -> None:
    image_relative = image_path.relative_to(result_dir.resolve()).as_posix()
    data_relative = image_path.with_suffix(".csv").relative_to(
        result_dir.resolve()
    ).as_posix()
    plot_id = _plot_id(plot_type, image_path)
    requested_paths = {image_relative, data_relative}
    recorded_plots = [
        *list(result.get("plots") or []),
        *load_benchmark_plot_records(result_dir),
    ]
    for raw in recorded_plots:
        if not isinstance(raw, Mapping) or str(raw.get("id") or "") == plot_id:
            continue
        existing_paths = {
            str(raw.get("relative_path") or ""),
            str(raw.get("data_csv_relative_path") or ""),
        }
        if requested_paths & existing_paths:
            raise BenchmarkError(
                "benchmark plot output is already bound to another recorded plot"
            )


def _relative_to_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _plot_id(plot_type: str, image_path: Path) -> str:
    return "%s:%s" % (plot_type, image_path.name)
