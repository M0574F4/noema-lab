from __future__ import annotations

"""Project one completed delayed-CSI benchmark into compact documentation assets."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ID = (
    "20260726T192946Z_"
    "resource_allocation.delayed_csi_finite_blocklength_post_training_v2"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "reliability_aware_delayed_csi_ofdm_allocation"
)
DEFAULT_CHART_JS = (
    ROOT / "docs" / "_static" / "noema-delayed-csi-demo-chart-data.js"
)

EXPECTED_BENCHMARK_ID = (
    "resource_allocation.delayed_csi_finite_blocklength_post_training_v2"
)
EXPECTED_VERSION = "2.0.0"
EXPECTED_METHODS = (
    "equal_power",
    "observed_csi_water_filling",
    "robust_csi_water_filling",
    "learned_allocator",
)
EXPECTED_BUDGETS = (0.4, 0.6, 0.8, 1.0, 1.4)
EXPECTED_SEEDS = (95101, 95201, 95301)
REPRESENTATIVE_BUDGET = 0.8
REPRESENTATIVE_SEED = 95101
T95_DF2 = 4.302652729911275

GOODPUT = "resource.finite_blocklength.expected_goodput_bps_hz"
BLER = "resource.finite_blocklength.predicted_bler"
P05_GOODPUT = "resource.finite_blocklength.p05_goodput_bps_hz"
GAIN_CORRELATION = "resource.csi.observed_actual_gain_correlation"
POWER_ERROR = "resource.power_constraint.max_abs_error"
NEGATIVE_POWER = "resource.power_constraint.max_negative_violation"
POWER_BUDGET = "resource.average_transmit_power_budget"

METHOD_STYLES = {
    "equal_power": {
        "label": "Equal power",
        "color": "#2563eb",
        "dash": [],
        "marker": "square",
    },
    "observed_csi_water_filling": {
        "label": "Water filling on delayed/noisy CSI",
        "color": "#d97706",
        "dash": [8, 4],
        "marker": "diamond",
    },
    "robust_csi_water_filling": {
        "label": "Uncertainty-shrunk water filling",
        "color": "#9333ea",
        "dash": [4, 3],
        "marker": "triangle",
    },
    "learned_allocator": {
        "label": "Learned reliability-aware allocator",
        "color": "#dc2626",
        "dash": [],
        "marker": "cross",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>.+)_p(?P<budget>[0-9]+(?:p[0-9]+)?)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact, reproducible documentation assets from one "
            "completed delayed-CSI benchmark result."
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
    projection_path = data_dir / "benchmark_projection.csv"
    summary_path = data_dir / "summary_table.csv"
    chart_data_path = data_dir / "chart_data.json"
    manifest_path = data_dir / "snapshot_manifest.json"

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
        "slug": "reliability-aware-delayed-csi-ofdm-allocation",
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
            "statistical_unit": "paired held-out TDL trajectory seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_budget": len(EXPECTED_SEEDS),
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out trajectory seeds"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three paired seeds support an experimental demonstration, "
                "not a paper-grade population claim."
            ),
        },
        "projection": {
            "rows": len(records),
            "benchmark_projection.csv": _file_evidence(projection_path),
            "summary_table.csv": _file_evidence(summary_path),
            "chart_data.json": _file_evidence(chart_data_path),
            "chart_javascript": {
                "path": str(chart_js.relative_to(ROOT)),
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
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe["semantic_recipe_sha256"],
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
    if demo.get("slug") != "reliability-aware-delayed-csi-ofdm-allocation":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    if len(recipes) != (
        len(EXPECTED_METHODS) * len(EXPECTED_BUDGETS) * len(EXPECTED_SEEDS)
    ):
        raise ValueError("benchmark result does not contain the expected 60 runs")
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
                "expected_goodput_bps_hz": _format_number(metrics[GOODPUT]),
                "predicted_bler": _format_number(metrics[BLER]),
                "p05_goodput_bps_hz": _format_number(metrics[P05_GOODPUT]),
                "delayed_current_gain_correlation": _format_number(
                    metrics[GAIN_CORRELATION]
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
    method_index = {method: index for index, method in enumerate(EXPECTED_METHODS)}
    records.sort(
        key=lambda row: (
            float(row["average_power_budget"]),
            int(row["paired_seed"]),
            method_index[row["method_id"]],
        )
    )
    if {
        float(record["average_power_budget"]) for record in records
    } != set(EXPECTED_BUDGETS):
        raise ValueError("unexpected power-budget coordinates")
    if {int(record["paired_seed"]) for record in records} != set(
        EXPECTED_SEEDS
    ):
        raise ValueError("unexpected paired-seed coordinates")
    return records


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (float(record["average_power_budget"]), record["method_id"])
        ].append(record)

    output = []
    for budget in EXPECTED_BUDGETS:
        baseline_means = {
            method: statistics.mean(
                float(row["expected_goodput_bps_hz"])
                for row in grouped[(budget, method)]
            )
            for method in EXPECTED_METHODS[:-1]
        }
        strongest = max(baseline_means, key=baseline_means.__getitem__)
        learned_by_seed = {
            int(row["paired_seed"]): float(row["expected_goodput_bps_hz"])
            for row in grouped[(budget, "learned_allocator")]
        }
        baseline_by_seed = {
            int(row["paired_seed"]): float(row["expected_goodput_bps_hz"])
            for row in grouped[(budget, strongest)]
        }
        paired_deltas = [
            learned_by_seed[seed] - baseline_by_seed[seed]
            for seed in EXPECTED_SEEDS
        ]
        delta_mean, delta_low, delta_high = _ci95(paired_deltas)
        learned_goodput = list(learned_by_seed.values())
        learned_mean, learned_low, learned_high = _ci95(learned_goodput)
        learned_bler = [
            float(row["predicted_bler"])
            for row in grouped[(budget, "learned_allocator")]
        ]
        learned_bler_mean, learned_bler_low, learned_bler_high = _ci95(
            learned_bler,
            lower=0.0,
            upper=1.0,
        )
        baseline_mean = baseline_means[strongest]
        power_error = max(
            float(row["max_power_constraint_error"])
            for method in EXPECTED_METHODS
            for row in grouped[(budget, method)]
        )
        output.append(
            {
                "Average power budget": _format_number(budget),
                "Learned goodput mean (bit/s/Hz)": _format_number(
                    learned_mean
                ),
                "Learned goodput 95% CI": _interval(
                    learned_low, learned_high
                ),
                "Strongest baseline": METHOD_STYLES[strongest]["label"],
                "Baseline goodput mean (bit/s/Hz)": _format_number(
                    baseline_mean
                ),
                "Paired goodput gain (bit/s/Hz)": _format_number(delta_mean),
                "Paired gain 95% CI": _interval(delta_low, delta_high),
                "Relative gain (%)": _format_number(
                    100.0 * delta_mean / baseline_mean
                ),
                "Learned predicted BLER mean": _format_number(
                    learned_bler_mean
                ),
                "Learned BLER 95% CI": _interval(
                    learned_bler_low, learned_bler_high
                ),
                "Maximum power error": _format_number(power_error),
            }
        )
    return output


def _chart_specs(
    result_dir: Path,
    result: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_method_budget: dict[tuple[str, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for record in records:
        by_method_budget[
            (record["method_id"], float(record["average_power_budget"]))
        ].append(record)

    goodput_series = []
    bler_series = []
    for method in EXPECTED_METHODS:
        style = METHOD_STYLES[method]
        goodput_values = []
        goodput_ranges = []
        bler_values = []
        bler_ranges = []
        for budget in EXPECTED_BUDGETS:
            rows = by_method_budget[(method, budget)]
            mean, low, high = _ci95(
                [float(row["expected_goodput_bps_hz"]) for row in rows],
                lower=0.0,
            )
            goodput_values.append([budget, mean])
            goodput_ranges.append([budget, low, high])
            mean, low, high = _ci95(
                [float(row["predicted_bler"]) for row in rows],
                lower=0.0,
                upper=1.0,
            )
            bler_values.append([budget, mean])
            bler_ranges.append([budget, low, high])
        goodput_series.append(
            {
                "id": method,
                **style,
                "values": goodput_values,
                "range": goodput_ranges,
            }
        )
        bler_series.append(
            {
                "id": method,
                **style,
                "values": bler_values,
                "range": bler_ranges,
            }
        )

    representative_rows = {
        record["method_id"]: record
        for record in records
        if float(record["average_power_budget"]) == REPRESENTATIVE_BUDGET
        and int(record["paired_seed"]) == REPRESENTATIVE_SEED
    }
    previews = {}
    evidence = {}
    for method in EXPECTED_METHODS:
        recipe_id = representative_rows[method]["recipe_id"]
        recipe = next(
            item for item in result["recipes"] if item["id"] == recipe_id
        )
        root = recipe["run_evidence_snapshot"]["root"]
        summary_path = result_dir / root / "summary.json"
        summary = _load_json(summary_path)
        step = next(
            item
            for item in summary["steps"]
            if item.get("id") == "allocation_evaluation"
        )
        preview = step["metadata"]["resource_allocation_preview"]
        snapshot = next(
            item for item in preview["snapshots"] if int(item["index"]) == 0
        )
        previews[method] = snapshot
        evidence[method] = {
            "recipe_id": recipe_id,
            "run_id": recipe["run_id"],
            "snapshot_index": 0,
            "summary": {
                "path": str((Path(root) / "summary.json")),
                **_file_evidence(summary_path),
            },
        }

    reference = previews["equal_power"]
    for method, preview in previews.items():
        if preview["channel_gain"] != reference["channel_gain"]:
            raise ValueError(
                "representative current channel is not paired for %s" % method
            )
        if (
            preview["observed_channel_gain"]
            != reference["observed_channel_gain"]
        ):
            raise ValueError(
                "representative delayed CSI is not paired for %s" % method
            )

    indices = list(range(0, len(reference["channel_gain"]), 2))
    channel_series = [
        {
            "id": "current_channel",
            "label": "Perfect current CSI (evaluation only)",
            "color": "#16a34a",
            "dash": [],
            "marker": "circle",
            "values": [
                [index, reference["channel_gain"][index]] for index in indices
            ],
        },
        {
            "id": "delayed_csi",
            "label": "Delayed/noisy transmitter CSI (5 symbols old)",
            "color": "#6b7280",
            "dash": [8, 4],
            "marker": "square",
            "values": [
                [index, reference["observed_channel_gain"][index]]
                for index in indices
            ],
        },
    ]
    allocation_series = [
        {
            "id": method,
            **METHOD_STYLES[method],
            "yAxis": "right",
            "values": [
                [index, previews[method]["allocated_power"][index]]
                for index in indices
            ],
        }
        for method in EXPECTED_METHODS
    ]
    charts = {
        "delayed-csi-goodput": {
            "type": "line",
            "title": "Expected short-packet goodput vs power budget",
            "description": (
                "Mean over three paired held-out TDL trajectory seeds; shaded "
                "bands are two-sided Student-t 95% confidence intervals."
            ),
            "xLabel": "Average power budget per subcarrier",
            "yLabel": "Expected goodput (bit/s/Hz)",
            "yScale": "linear",
            "allowLog": True,
            "series": goodput_series,
        },
        "delayed-csi-predicted-bler": {
            "type": "line",
            "title": "Predicted short-packet BLER vs power budget",
            "description": (
                "Finite-blocklength normal-approximation BLER, paired across "
                "the same three held-out TDL trajectory seeds."
            ),
            "xLabel": "Average power budget per subcarrier",
            "yLabel": "Predicted block error rate",
            "yScale": "log",
            "allowLog": True,
            "series": bler_series,
        },
        "delayed-csi-representative-state": {
            "type": "line",
            "title": "CSI available at transmission time and resulting allocations",
            "description": (
                "For each subcarrier k, compare the delayed observation "
                "|Ĥ[t−5,k]|² with perfect evaluation-only |H[t,k]|² and "
                "the powers selected from the delayed history. One paired "
                "held-out state at P=0.8 and seed 95101 is shown."
            ),
            "xLabel": "Subcarrier index",
            "yLabel": "Channel power gain",
            "rightYLabel": "Allocated power",
            "yScale": "linear",
            "allowLog": True,
            "rightYScale": "linear",
            "rightYIncludeZero": True,
            "series": channel_series + allocation_series,
        },
    }
    representative = {
        "average_power_budget": REPRESENTATIVE_BUDGET,
        "paired_seed": REPRESENTATIVE_SEED,
        "snapshot_index": 0,
        "subcarrier_count": len(reference["channel_gain"]),
        "display_stride": 2,
        "feedback_delay_ofdm_symbols": 5,
        "evidence": evidence,
    }
    return charts, representative


def _ci95(
    values: Iterable[float],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, float, float]:
    samples = [float(value) for value in values]
    if len(samples) != len(EXPECTED_SEEDS):
        raise ValueError("documentation snapshot expects exactly three samples")
    mean = statistics.mean(samples)
    half_width = T95_DF2 * statistics.stdev(samples) / math.sqrt(len(samples))
    low = mean - half_width
    high = mean + half_width
    if lower is not None:
        low = max(lower, low)
    if upper is not None:
        high = min(upper, high)
    return mean, low, high


def _file_evidence(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_chart_js(path: Path, charts: Mapping[str, Any]) -> None:
    serialized = json.dumps(charts, indent=2, sort_keys=True)
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
        % serialized,
        encoding="utf-8",
    )


def _format_number(value: float) -> str:
    return "%.10g" % float(value)


def _interval(low: float, high: float) -> str:
    return "[%s, %s]" % (_format_number(low), _format_number(high))


if __name__ == "__main__":
    raise SystemExit(main())
