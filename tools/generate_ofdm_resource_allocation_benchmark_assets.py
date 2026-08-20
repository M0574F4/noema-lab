from __future__ import annotations

"""Project the completed paired OFDM allocation benchmark into docs assets."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ID = (
    "20260727T002419Z_"
    "resource_allocation.learned_allocator_post_training_v2"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = (
    ROOT / "docs" / "demo" / "data" / "ofdm_resource_allocation"
)
DEFAULT_CHART_JS = (
    ROOT / "docs" / "_static" / "noema-ofdm-resource-allocation-chart-data.js"
)

EXPECTED_BENCHMARK_ID = (
    "resource_allocation.learned_allocator_post_training_v2"
)
EXPECTED_VERSION = "2.0.0"
EXPECTED_METHODS = ("equal_power", "learned_allocator", "water_filling")
EXPECTED_BUDGETS = (0.5, 1.0, 2.0)
EXPECTED_SEEDS = (71001, 72001, 73001)
T95_DF2 = 4.302652729911275
REPRESENTATIVE_BUDGET = 1.0
REPRESENTATIVE_SEED = 72001

SPECTRAL_EFFICIENCY = (
    "resource.theoretical_shannon_spectral_efficiency_bps_hz"
)
RELATIVE_GAP = "resource.water_filling_relative_optimality_gap"
KKT_RESIDUAL = "resource.water_filling_kkt_normalized_residual"
POWER_ERROR = "resource.power_constraint.max_abs_error"
NEGATIVE_POWER = "resource.power_constraint.max_negative_violation"

METHOD_STYLES = {
    "equal_power": {
        "label": "Equal power",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "learned_allocator": {
        "label": "Learned allocator",
        "color": "#16a34a",
        "dash": [],
        "marker": "circle",
    },
    "water_filling": {
        "label": "Water filling (Shannon oracle)",
        "color": "#dc2626",
        "dash": [3, 3],
        "marker": "diamond",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>equal_power|learned_allocator|water_filling)"
    r"_p(?P<budget>[0-9]+(?:p[0-9]+)?)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate documentation data from the completed paired OFDM "
            "resource-allocation benchmark."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chart-js", type=Path, default=DEFAULT_CHART_JS)
    args = parser.parse_args()
    generated = generate(
        args.result_dir.expanduser().resolve(),
        args.data_dir.expanduser().resolve(),
        args.chart_js.expanduser().resolve(),
    )
    for label, path in generated.items():
        print("%s: %s" % (label, path))
    return 0


def generate(
    result_dir: Path,
    data_dir: Path,
    chart_js: Path,
) -> dict[str, Path]:
    result_path = result_dir / "result.json"
    metrics_path = result_dir / "metrics.csv"
    recipes_path = result_dir / "recipes.csv"
    training_manifest_path = result_dir / "training_evidence" / "manifest.json"
    result = _load_json(result_path)
    _validate_result(result)
    records = _project_records(result)
    summaries = _summary_rows(records)
    charts, representative = _chart_specs(result_dir, result, records)

    data_dir.mkdir(parents=True, exist_ok=True)
    chart_js.parent.mkdir(parents=True, exist_ok=True)
    projection_path = data_dir / "paired_benchmark_projection.csv"
    summary_path = data_dir / "paired_summary_table.csv"
    chart_data_path = data_dir / "paired_chart_data.json"
    manifest_path = data_dir / "paired_snapshot_manifest.json"
    _write_csv(projection_path, records)
    _write_csv(summary_path, summaries)
    _write_json(chart_data_path, charts)
    _write_chart_js(chart_js, charts)

    training_projection = _load_json(
        result_dir / "training_evidence" / "projection.json"
    )
    learned_evidence = training_projection["entries"][0]
    manifest = {
        "kind": "noema.docs_demo_snapshot",
        "schema_version": 1,
        "slug": "learned-ofdm-resource-allocation",
        "source": {
            "result_id": result_dir.name,
            "benchmark_id": EXPECTED_BENCHMARK_ID,
            "benchmark_version": EXPECTED_VERSION,
            "status": "completed",
            "benchmark_tier": "experimental",
            "created_at_utc": result["created_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
            "files": {
                "result.json": _file_evidence(result_path),
                "metrics.csv": _file_evidence(metrics_path),
                "recipes.csv": _file_evidence(recipes_path),
                "training_evidence/manifest.json": _file_evidence(
                    training_manifest_path
                ),
            },
        },
        "statistical_design": {
            "aggregation_cell": "average transmit-power budget",
            "statistical_unit": "paired held-out payload/channel/noise seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_budget": len(EXPECTED_SEEDS),
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out seeds"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three paired held-out seeds support an experimental tutorial "
                "result, not a paper-grade population claim."
            ),
        },
        "projection": {
            "rows": len(records),
            "paired_benchmark_projection.csv": _file_evidence(projection_path),
            "paired_summary_table.csv": _file_evidence(summary_path),
            "paired_chart_data.json": _file_evidence(chart_data_path),
            "chart_javascript": {
                "path": str(Path("docs") / "_static" / chart_js.name),
                **_file_evidence(chart_js),
            },
        },
        "representative_preview": representative,
        "training_evidence": {
            "series": learned_evidence["series"],
            "component": learned_evidence["artifact_references"][0],
            "contract": learned_evidence["artifact_references"][1],
            "trained_artifact_manifest": learned_evidence["evidence"][
                "trained_artifact_manifest"
            ],
            "training_history": learned_evidence["evidence"][
                "training_history"
            ],
            "evaluation_metrics": learned_evidence["evidence"][
                "evaluation_metrics"
            ],
        },
        "runs": [
            {
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "role": recipe["role"],
                "pairing_id": recipe["pairing_id"],
                "pairing_seed": recipe["pairing_seed"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe[
                    "semantic_recipe_sha256"
                ],
            }
            for recipe in result["recipes"]
        ],
    }
    _write_json(manifest_path, manifest)
    return {
        "projection": projection_path,
        "summary": summary_path,
        "chart_data": chart_data_path,
        "chart_javascript": chart_js,
        "manifest": manifest_path,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    benchmark = result.get("benchmark") or {}
    if result.get("status") != "completed":
        raise ValueError("benchmark result is not completed")
    if benchmark.get("id") != EXPECTED_BENCHMARK_ID:
        raise ValueError("unexpected benchmark id")
    if benchmark.get("version") != EXPECTED_VERSION:
        raise ValueError("unexpected benchmark version")
    demo = (benchmark.get("metadata") or {}).get("demo") or {}
    if demo.get("slug") != "learned-ofdm-resource-allocation":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    expected_count = (
        len(EXPECTED_METHODS) * len(EXPECTED_BUDGETS) * len(EXPECTED_SEEDS)
    )
    if len(recipes) != expected_count:
        raise ValueError(
            "benchmark result does not contain the expected %d runs"
            % expected_count
        )
    if any(recipe.get("status") != "completed" for recipe in recipes):
        raise ValueError("benchmark contains an incomplete recipe")


def _project_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for recipe in result["recipes"]:
        match = RECIPE_ID.fullmatch(str(recipe["id"]))
        if match is None:
            raise ValueError("unexpected recipe id: %s" % recipe["id"])
        method = match.group("method")
        budget = float(match.group("budget").replace("p", "."))
        seed = int(match.group("seed"))
        metrics = recipe["metrics"]
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "average_power_budget": _format_number(budget),
                "paired_seed": seed,
                "spectral_efficiency_bps_hz": _format_number(
                    metrics[SPECTRAL_EFFICIENCY]
                ),
                "water_filling_relative_gap": _format_number(
                    metrics[RELATIVE_GAP]
                ),
                "water_filling_kkt_residual": _format_number(
                    metrics[KKT_RESIDUAL]
                ),
                "max_power_constraint_error": _format_number(
                    metrics[POWER_ERROR]
                ),
                "max_negative_power_violation": _format_number(
                    metrics[NEGATIVE_POWER]
                ),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe[
                    "semantic_recipe_sha256"
                ],
            }
        )
    expected_cells = {
        (method, budget, seed)
        for method in EXPECTED_METHODS
        for budget in EXPECTED_BUDGETS
        for seed in EXPECTED_SEEDS
    }
    actual_cells = {
        (
            str(row["method_id"]),
            float(row["average_power_budget"]),
            int(row["paired_seed"]),
        )
        for row in records
    }
    if actual_cells != expected_cells:
        raise ValueError("benchmark cells do not match the frozen protocol")
    return sorted(
        records,
        key=lambda row: (
            float(row["average_power_budget"]),
            int(row["paired_seed"]),
            EXPECTED_METHODS.index(str(row["method_id"])),
        ),
    )


def _summary_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = _group(records)
    rows = []
    for budget in EXPECTED_BUDGETS:
        means = {
            method: statistics.mean(
                float(row["spectral_efficiency_bps_hz"])
                for row in cells[(method, budget)]
            )
            for method in EXPECTED_METHODS
        }
        learned_achievement = statistics.mean(
            100.0
            * _value(cells, "learned_allocator", budget, seed)
            / _value(cells, "water_filling", budget, seed)
            for seed in EXPECTED_SEEDS
        )
        learned_rows = cells[("learned_allocator", budget)]
        rows.append(
            {
                "Average power budget": _format_number(budget),
                "Equal power (bit/s/Hz)": _format_number(
                    means["equal_power"]
                ),
                "Learned (bit/s/Hz)": _format_number(
                    means["learned_allocator"]
                ),
                "Water filling (bit/s/Hz)": _format_number(
                    means["water_filling"]
                ),
                "Learned / oracle (%)": _format_number(learned_achievement),
                "Learned mean relative gap": _format_number(
                    statistics.mean(
                        float(row["water_filling_relative_gap"])
                        for row in learned_rows
                    )
                ),
                "Learned max power error": _format_number(
                    max(
                        float(row["max_power_constraint_error"])
                        for row in learned_rows
                    )
                ),
            }
        )
    return rows


def _chart_specs(
    result_dir: Path,
    result: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    charts = {
        "ofdm-allocation-spectral-efficiency": _spectral_efficiency_chart(
            records
        ),
        "ofdm-allocation-shannon-optimum-achievement": _achievement_chart(
            records
        ),
    }
    representative_recipe_ids = {
        method: "%s_p1_seed%d" % (method, REPRESENTATIVE_SEED)
        for method in EXPECTED_METHODS
    }
    previews: dict[str, Mapping[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    recipe_by_id = {str(recipe["id"]): recipe for recipe in result["recipes"]}
    for method, recipe_id in representative_recipe_ids.items():
        evidence_dir = next(
            path
            for path in (result_dir / "run_evidence").iterdir()
            if path.name.endswith(recipe_id)
        )
        report_path = (
            evidence_dir / "artifacts" / "allocation_evaluation" / "report.json"
        )
        report = _load_json(report_path)
        previews[method] = report["metadata"]["resource_allocation_preview"]
        report_paths[method] = report_path
    snapshot_index = 0
    channel_gain = list(
        previews["water_filling"]["snapshots"][snapshot_index]["channel_gain"]
    )
    subcarriers = list(range(len(channel_gain)))
    power_series = []
    for method in EXPECTED_METHODS:
        style = METHOD_STYLES[method]
        powers = previews[method]["snapshots"][snapshot_index][
            "allocated_power"
        ]
        power_series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "yAxis": "right",
                "values": [
                    [index, value]
                    for index, value in zip(subcarriers, powers)
                ],
            }
        )
    charts["ofdm-allocation-representative-power"] = {
        "type": "line",
        "title": "Representative channel and power allocations",
        "description": (
            "One paired held-out TDL-A channel state at average power 1.0 "
            "and seed 72001. Channel power gain is evaluation context; all "
            "three policies share exactly the same state."
        ),
        "xLabel": "Subcarrier index",
        "yLabel": "Channel power gain",
        "rightYLabel": "Allocated power",
        "allowLog": True,
        "yIncludeZero": True,
        "rightYScale": "linear",
        "rightYIncludeZero": True,
        "series": [
            {
                "id": "channel_gain",
                "label": "Channel power gain",
                "color": "#d97706",
                "dash": [2, 3],
                "marker": "none",
                "values": [
                    [index, value]
                    for index, value in zip(subcarriers, channel_gain)
                ],
            },
            *power_series,
        ],
    }
    representative = {
        "average_power_budget": REPRESENTATIVE_BUDGET,
        "paired_seed": REPRESENTATIVE_SEED,
        "snapshot_index": snapshot_index,
        "subcarrier_count": len(subcarriers),
        "recipes": {
            method: {
                "recipe_id": recipe_id,
                "run_id": recipe_by_id[recipe_id]["run_id"],
                "report": _file_evidence(report_paths[method]),
            }
            for method, recipe_id in representative_recipe_ids.items()
        },
    }
    return charts, representative


def _spectral_efficiency_chart(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in EXPECTED_METHODS:
        values = []
        ranges = []
        for budget in EXPECTED_BUDGETS:
            samples = [
                float(row["spectral_efficiency_bps_hz"])
                for row in cells[(method, budget)]
            ]
            mean = statistics.mean(samples)
            lower, upper = _confidence_interval(samples)
            values.append([budget, mean])
            ranges.append([budget, lower, upper])
        style = METHOD_STYLES[method]
        series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": values,
                "range": ranges,
            }
        )
    return {
        "type": "line",
        "title": "Shannon spectral efficiency vs power budget",
        "description": (
            "Higher is better. Points are means over three paired held-out "
            "payload/channel/noise seeds; bands are Student-t 95% confidence intervals."
        ),
        "xLabel": "Average power budget per subcarrier",
        "yLabel": "Spectral efficiency (bit/s/Hz)",
        "allowLog": True,
        "yIncludeZero": False,
        "series": series,
    }


def _achievement_chart(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in EXPECTED_METHODS:
        values = []
        ranges = []
        for budget in EXPECTED_BUDGETS:
            samples = [
                100.0
                * _value(cells, method, budget, seed)
                / _value(cells, "water_filling", budget, seed)
                for seed in EXPECTED_SEEDS
            ]
            mean = statistics.mean(samples)
            lower, upper = _confidence_interval(samples)
            values.append([budget, mean])
            ranges.append([budget, lower, upper])
        style = METHOD_STYLES[method]
        series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": values,
                "range": ranges,
            }
        )
    return {
        "type": "line",
        "title": "Shannon optimum achievement",
        "description": (
            "Each policy's spectral efficiency divided by exact water "
            "filling on the same paired channel state. Water filling is 100%."
        ),
        "xLabel": "Average power budget per subcarrier",
        "yLabel": "Water-filling optimum achieved (%)",
        "allowLog": False,
        "yIncludeZero": False,
        "series": series,
    }


def _group(
    records: list[Mapping[str, Any]],
) -> dict[tuple[str, float], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in records:
        grouped[
            (str(row["method_id"]), float(row["average_power_budget"]))
        ].append(row)
    return grouped


def _value(
    cells: Mapping[tuple[str, float], list[Mapping[str, Any]]],
    method: str,
    budget: float,
    seed: int,
) -> float:
    row = next(
        item
        for item in cells[(method, budget)]
        if int(item["paired_seed"]) == seed
    )
    return float(row["spectral_efficiency_bps_hz"])


def _confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    margin = T95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV projection")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_chart_js(path: Path, charts: Mapping[str, Any]) -> None:
    payload = json.dumps(charts, indent=2, sort_keys=True)
    path.write_text(
        """(function () {
  "use strict";
  const charts = %s;
  window.NOEMA_DEMO_CHARTS = Object.freeze({
    ...(window.NOEMA_DEMO_CHARTS || {}),
    ...charts,
  });
})();
"""
        % payload,
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_evidence(path: Path) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _format_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite benchmark value")
    return format(number, ".12g")


if __name__ == "__main__":
    raise SystemExit(main())
