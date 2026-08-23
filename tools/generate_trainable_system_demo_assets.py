"""Project six completed trainable-system benchmarks into documentation assets."""

from __future__ import annotations

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
T95_DF2 = 4.302652729911275
EXPECTED_SNRS = (0.0, 15.0)
EXPECTED_SEEDS = (81001, 82001, 83001)

CONFIGS: dict[str, dict[str, Any]] = {
    "range-localization": {
        "result_id": "20260823T124143Z_localization_sensing.learned_range_localizer_v1",
        "benchmark_id": "localization_sensing.learned_range_localizer_v1",
        "version": "1.0.0",
        "data_slug": "range_localization",
        "chart_js": "noema-range-localization-chart-data.js",
        "chart_id": "range-localization-rmse",
        "title": "Range-localization error",
        "description": "Mean RMSE over three paired held-out target, range-noise, and geometry seeds. Bands are two-sided Student-t 95% intervals.",
        "primary_metric": "localization.rmse_m",
        "metrics": (
            ("localization.rmse_m", "RMSE (m)"),
            ("localization.mae_m", "MAE (m)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": ("localization.rmse_m",),
        "y_label": "Position RMSE (m)",
        "y_domain": None,
    },
    "aoa-estimation": {
        "result_id": "20260823T133322Z_localization_sensing.learned_aoa_estimator_v1",
        "benchmark_id": "localization_sensing.learned_aoa_estimator_v1",
        "version": "1.0.0",
        "data_slug": "aoa_estimation",
        "chart_js": "noema-aoa-estimation-chart-data.js",
        "chart_id": "aoa-estimation-rmse",
        "title": "Narrowband AoA error",
        "description": "Mean angle RMSE over three paired held-out source-angle, snapshot, and AWGN seeds. Bands are two-sided Student-t 95% intervals.",
        "primary_metric": "aoa.rmse_deg",
        "metrics": (
            ("aoa.rmse_deg", "RMSE (degree)"),
            ("aoa.mae_deg", "MAE (degree)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": ("aoa.rmse_deg",),
        "y_label": "Angle RMSE (degree)",
        "y_domain": None,
    },
    "beam-selection": {
        "result_id": "20260823T134317Z_beamforming_precoding.learned_beam_selection_v1",
        "benchmark_id": "beamforming_precoding.learned_beam_selection_v1",
        "version": "1.0.0",
        "data_slug": "miso_beam_selection",
        "chart_js": "noema-miso-beam-selection-chart-data.js",
        "chart_id": "miso-beam-selection-rate",
        "title": "MISO beam-selection rate",
        "description": "Mean spectral efficiency over three paired held-out clustered-ULA channel seeds. The learned and fixed DFT codebooks each contain eight beams; perfect-CSIT MRT is an upper bound.",
        "primary_metric": "beamforming.spectral_efficiency_bps_hz",
        "metrics": (
            ("beamforming.spectral_efficiency_bps_hz", "Spectral efficiency (bit/s/Hz)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": ("beamforming.spectral_efficiency_bps_hz",),
        "y_label": "Spectral efficiency (bit/s/Hz)",
        "y_domain": None,
    },
    "isac-allocation": {
        "result_id": "20260823T045328Z_resource_allocation.learned_isac_joint_allocation_v1",
        "benchmark_id": "resource_allocation.learned_isac_joint_allocation_v1",
        "version": "1.0.0",
        "data_slug": "isac_joint_allocation",
        "chart_js": "noema-isac-joint-allocation-chart-data.js",
        "chart_id": "isac-joint-allocation-utility",
        "title": "Joint communication-sensing utility",
        "description": "Mean scalarized utility over three paired held-out communication-channel, sensing-channel, and noise seeds. The number is specific to the contract's fixed sensing weight.",
        "primary_metric": "isac.scalarized_utility",
        "metrics": (
            ("isac.scalarized_utility", "Scalarized utility"),
            ("isac.communication_rate_bps_hz", "Communication rate (bit/s/Hz)"),
            ("isac.sensing_snr_db", "Sensing SNR (dB)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": (
            "isac.scalarized_utility",
            "isac.communication_rate_bps_hz",
            "isac.sensing_snr_db",
        ),
        "y_label": "Scalarized utility",
        "y_domain": None,
    },
    "near-field": {
        "result_id": "20260823T133443Z_localization_sensing.learned_near_field_focusing_v1",
        "benchmark_id": "localization_sensing.learned_near_field_focusing_v1",
        "version": "1.0.0",
        "data_slug": "near_field_xl_mimo",
        "chart_js": "noema-near-field-xl-mimo-chart-data.js",
        "chart_id": "near-field-focusing-gain",
        "title": "Near-field focusing gain",
        "description": "Mean normalized focusing gain over three paired held-out target-state and noise seeds. True-position focusing is a simulation-only oracle.",
        "primary_metric": "near_field.normalized_focusing_gain",
        "metrics": (
            ("near_field.normalized_focusing_gain", "Normalized focusing gain"),
            ("near_field.range_rmse_m", "Range RMSE (m)"),
            ("near_field.angle_rmse_deg", "Angle RMSE (degree)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": (
            "near_field.normalized_focusing_gain",
            "near_field.range_rmse_m",
            "near_field.angle_rmse_deg",
        ),
        "y_label": "Normalized focusing gain",
        "y_domain": (0.0, 1.05),
    },
    "leo-ntn": {
        "result_id": "20260823T045259Z_beamforming_precoding.learned_leo_ntn_tracking_v1",
        "benchmark_id": "beamforming_precoding.learned_leo_ntn_tracking_v1",
        "version": "1.0.0",
        "data_slug": "leo_ntn_tracking",
        "chart_js": "noema-leo-ntn-tracking-chart-data.js",
        "chart_id": "leo-ntn-handover-accuracy",
        "title": "LEO-NTN next-beam handover",
        "description": "Mean one-second-ahead beam accuracy over three paired held-out track and measurement-noise seeds. True future state is a simulation-only oracle.",
        "primary_metric": "ntn.beam_handover_accuracy",
        "metrics": (
            ("ntn.beam_handover_accuracy", "Beam accuracy"),
            ("ntn.beam_outage_rate", "Beam outage rate"),
            ("ntn.doppler_mae_hz", "Doppler MAE (Hz)"),
            ("ntn.pointing_mae_deg", "Pointing MAE (degree)"),
            ("task.score", "Task score"),
        ),
        "summary_metrics": (
            "ntn.beam_handover_accuracy",
            "ntn.doppler_mae_hz",
        ),
        "y_label": "Next-beam accuracy",
        "y_domain": (0.0, 1.05),
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>.+)_snr(?P<snr>m?[0-9]+(?:p[0-9]+)?)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate documentation assets from trainable-system benchmarks."
    )
    parser.add_argument("demo", choices=(*CONFIGS, "all"))
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--chart-js", type=Path)
    args = parser.parse_args()
    if args.demo == "all" and any((args.result_dir, args.data_dir, args.chart_js)):
        parser.error("path overrides require one named demo")
    names = CONFIGS if args.demo == "all" else (args.demo,)
    for name in names:
        config = CONFIGS[name]
        generated = generate(
            name,
            (args.result_dir or ROOT / ".noema" / "benchmarks" / config["result_id"]).resolve(),
            (args.data_dir or ROOT / "docs" / "demo" / "data" / config["data_slug"]).resolve(),
            (args.chart_js or ROOT / "docs" / "_static" / config["chart_js"]).resolve(),
        )
        for label, path in generated.items():
            print(f"{name} {label}: {path}")
    return 0


def generate(
    name: str,
    result_dir: Path,
    data_dir: Path,
    chart_js: Path,
) -> dict[str, Path]:
    config = CONFIGS[name]
    result_path = result_dir / "result.json"
    metrics_path = result_dir / "metrics.csv"
    recipes_path = result_dir / "recipes.csv"
    evidence_path = result_dir / "training_evidence" / "manifest.json"
    evidence_projection_path = result_dir / "training_evidence" / "projection.json"
    result = _load_json(result_path)
    methods = _validate_result(result, config)
    records = _project_records(result, config, methods)
    summaries = _summary_rows(records, config, methods)
    chart = _chart_spec(records, config, methods)

    data_dir.mkdir(parents=True, exist_ok=True)
    chart_js.parent.mkdir(parents=True, exist_ok=True)
    projection_path = data_dir / "benchmark_projection.csv"
    summary_path = data_dir / "summary_table.csv"
    chart_data_path = data_dir / "chart_data.json"
    manifest_path = data_dir / "snapshot_manifest.json"
    _write_csv(projection_path, records)
    _write_csv(summary_path, summaries)
    _write_json(chart_data_path, {config["chart_id"]: chart})
    _write_chart_js(chart_js, {config["chart_id"]: chart})

    training_projection = _load_json(evidence_projection_path)
    manifest = {
        "kind": "noema.docs_demo_snapshot",
        "schema_version": 1,
        "slug": config["data_slug"],
        "source": {
            "result_id": result_dir.name,
            "benchmark_id": config["benchmark_id"],
            "benchmark_version": config["version"],
            "status": "completed",
            "benchmark_tier": result["benchmark"]["benchmark_tier"],
            "created_at_utc": result["created_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
            "files": {
                "result.json": _file_evidence(result_path),
                "metrics.csv": _file_evidence(metrics_path),
                "recipes.csv": _file_evidence(recipes_path),
                "training_evidence/manifest.json": _file_evidence(evidence_path),
                "training_evidence/projection.json": _file_evidence(evidence_projection_path),
            },
        },
        "statistical_design": {
            "aggregation_cell": "channel SNR",
            "statistical_unit": "paired held-out scenario and noise seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_snr": len(EXPECTED_SEEDS),
            "interval": "two-sided Student-t 95% confidence interval over paired held-out seeds",
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": "Three paired held-out seeds support an experimental tutorial result, not a paper-grade population claim.",
        },
        "projection": {
            "rows": len(records),
            "benchmark_projection.csv": _file_evidence(projection_path),
            "summary_table.csv": _file_evidence(summary_path),
            "chart_data.json": _file_evidence(chart_data_path),
            "chart_javascript": {
                "path": str(Path("docs") / "_static" / config["chart_js"]),
                **_file_evidence(chart_js),
            },
        },
        "training_evidence": training_projection["entries"],
        "runs": [
            {
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "role": recipe["role"],
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


def _validate_result(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, str]]:
    benchmark = result.get("benchmark") or {}
    if result.get("status") != "completed":
        raise ValueError("benchmark result is not completed")
    if benchmark.get("id") != config["benchmark_id"]:
        raise ValueError("unexpected benchmark id")
    if benchmark.get("version") != config["version"]:
        raise ValueError("unexpected benchmark version")
    demo = (benchmark.get("metadata") or {}).get("demo") or {}
    raw_methods = demo.get("series") or []
    methods = [
        {"id": str(item["id"]), "label": str(item["label"]), "role": str(item["role"])}
        for item in raw_methods
    ]
    if not methods:
        raise ValueError("benchmark has no declared demo series")
    recipes = result.get("recipes") or []
    expected_count = len(methods) * len(EXPECTED_SNRS) * len(EXPECTED_SEEDS)
    if len(recipes) != expected_count:
        raise ValueError(f"benchmark contains {len(recipes)} runs; expected {expected_count}")
    if any(recipe.get("status") != "completed" for recipe in recipes):
        raise ValueError("benchmark contains an incomplete recipe")
    return methods


def _project_records(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    methods: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_id = {method["id"]: method for method in methods}
    records: list[dict[str, Any]] = []
    for recipe in result["recipes"]:
        match = RECIPE_ID.fullmatch(str(recipe["id"]))
        if match is None:
            raise ValueError(f"unexpected recipe id: {recipe['id']}")
        method_id = match.group("method")
        if method_id not in by_id:
            raise ValueError(f"undeclared method in recipe id: {method_id}")
        token = match.group("snr")
        snr = -float(token[1:].replace("p", ".")) if token.startswith("m") else float(token.replace("p", "."))
        seed = int(match.group("seed"))
        metrics = recipe["metrics"]
        row: dict[str, Any] = {
            "method_id": method_id,
            "method_label": by_id[method_id]["label"],
            "role": by_id[method_id]["role"],
            "snr_db": _number(snr),
            "paired_seed": seed,
        }
        for metric, _ in config["metrics"]:
            row[metric] = _number(metrics[metric])
        row.update(
            {
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe["semantic_recipe_sha256"],
            }
        )
        records.append(row)
    expected = {
        (method["id"], snr, seed)
        for method in methods
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (str(row["method_id"]), float(row["snr_db"]), int(row["paired_seed"]))
        for row in records
    }
    if actual != expected:
        raise ValueError("benchmark cells do not match the frozen paired protocol")
    order = {method["id"]: index for index, method in enumerate(methods)}
    return sorted(
        records,
        key=lambda row: (float(row["snr_db"]), order[str(row["method_id"])], int(row["paired_seed"])),
    )


def _summary_rows(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    methods: list[dict[str, str]],
) -> list[dict[str, Any]]:
    cells = _group(records)
    labels = dict(config["metrics"])
    primary = str(config["primary_metric"])
    rows: list[dict[str, Any]] = []
    for snr in EXPECTED_SNRS:
        for method in methods:
            samples = cells[(method["id"], snr)]
            primary_values = [float(row[primary]) for row in samples]
            low, high = _confidence_interval(primary_values)
            row: dict[str, Any] = {
                "SNR (dB)": _number(snr),
                "Method": method["label"],
                "Role": method["role"],
            }
            for metric in config["summary_metrics"]:
                row[f"Mean {labels[metric]}"] = _number(
                    statistics.mean(float(sample[metric]) for sample in samples)
                )
            row["Primary 95% CI low"] = _number(max(0.0, low))
            row["Primary 95% CI high"] = _number(high)
            rows.append(row)
    return rows


def _chart_spec(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    methods: list[dict[str, str]],
) -> dict[str, Any]:
    cells = _group(records)
    primary = str(config["primary_metric"])
    baselines = 0
    series = []
    for method in methods:
        if method["role"] == "candidate":
            style = {"color": "#16a34a", "dash": [], "marker": "circle"}
        elif method["role"] == "oracle":
            style = {"color": "#d97706", "dash": [3, 3], "marker": "diamond"}
        else:
            baseline_styles = (
                {"color": "#2563eb", "dash": [7, 4], "marker": "square"},
                {"color": "#7c3aed", "dash": [10, 3], "marker": "triangle"},
            )
            style = baseline_styles[baselines % len(baseline_styles)]
            baselines += 1
        values = []
        ranges = []
        for snr in EXPECTED_SNRS:
            samples = [float(row[primary]) for row in cells[(method["id"], snr)]]
            mean = statistics.mean(samples)
            low, high = _confidence_interval(samples)
            if config["y_domain"] is not None:
                low = max(float(config["y_domain"][0]), low)
                high = min(float(config["y_domain"][1]), high)
            else:
                low = max(0.0, low)
            values.append([snr, mean])
            ranges.append([snr, low, high])
        series.append(
            {
                "id": method["id"],
                "label": method["label"],
                **style,
                "values": values,
                "range": ranges,
            }
        )
    chart: dict[str, Any] = {
        "type": "line",
        "title": config["title"],
        "description": config["description"],
        "accessibleSummary": "Each point is the mean of three paired held-out runs. Shaded bands show Student-t 95% confidence intervals; exact values and run identifiers are downloadable below.",
        "xLabel": "SNR (dB)",
        "yLabel": config["y_label"],
        "allowLog": False,
        "yIncludeZero": True,
        "series": series,
    }
    if config["y_domain"] is not None:
        chart["yDomain"] = list(config["y_domain"])
    return chart


def _group(
    records: list[Mapping[str, Any]],
) -> dict[tuple[str, float], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["method_id"]), float(row["snr_db"]))].append(row)
    return grouped


def _confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    margin = T95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV projection")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_chart_js(path: Path, charts: Mapping[str, Any]) -> None:
    payload = json.dumps(charts, indent=2, sort_keys=True)
    path.write_text(
        """(function () {
  \"use strict\";
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


def _number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite benchmark value")
    return format(number, ".12g")


if __name__ == "__main__":
    raise SystemExit(main())
