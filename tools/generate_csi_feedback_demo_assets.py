"""Project a completed CSI-feedback benchmark into compact docs assets."""

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
DEFAULT_RESULT_ID = (
    "20260727T125921Z_channel_estimation.learned_csi_feedback_post_training_v1"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "csi_feedback"
DEFAULT_CHART_JS = ROOT / "docs" / "_static" / "noema-csi-feedback-chart-data.js"

EXPECTED_BENCHMARK_ID = "channel_estimation.learned_csi_feedback_post_training_v1"
EXPECTED_VERSION = "1.0.0"
EXPECTED_METHODS = (
    "truncated_angular_delay",
    "matched_klt",
    "learned_codec",
    "perfect_csit",
)
EXPECTED_SNRS = (0.0, 5.0, 10.0, 15.0, 20.0)
EXPECTED_SEEDS = (91001, 92001, 93001)
T95_DF2 = 4.302652729911275
REPRESENTATIVE_SNR = 10.0
REPRESENTATIVE_SEED = 92001

RATE = "csi_feedback.achieved_spectral_efficiency_bps_hz"
RETENTION = "csi_feedback.spectral_efficiency_retention"
NMSE_DB = "csi_feedback.nmse_db"
COSINE = "csi_feedback.phase_invariant_cosine"
BITS = "csi_feedback.feedback_bits_per_sample"

METHOD_STYLES = {
    "truncated_angular_delay": {
        "label": "Truncated angular-delay",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "matched_klt": {
        "label": "Matched KLT/PCA",
        "color": "#9333ea",
        "dash": [3, 3],
        "marker": "diamond",
    },
    "learned_codec": {
        "label": "Learned CSI codec",
        "color": "#16a34a",
        "dash": [],
        "marker": "circle",
    },
    "perfect_csit": {
        "label": "Perfect CSIT",
        "color": "#d97706",
        "dash": [10, 3],
        "marker": "triangle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>truncated_angular_delay|matched_klt|learned_codec|perfect_csit)"
    r"_snr(?P<snr>m?[0-9]+)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs assets from a completed CSI-feedback benchmark."
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
        "slug": "learned-csi-compression-feedback",
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
            "aggregation_cell": "downlink SNR",
            "statistical_unit": "paired held-out channel seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_snr": len(EXPECTED_SEEDS),
            "channel_realizations_per_run": 512,
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out channel seeds"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three paired channel seeds support an experimental tutorial "
                "result, not a paper-grade population claim."
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
            "artifact_references": learned_evidence["artifact_references"],
            "trained_artifact_manifest": learned_evidence["evidence"][
                "trained_artifact_manifest"
            ],
            "training_history": learned_evidence["evidence"]["training_history"],
            "evaluation_metrics": learned_evidence["evidence"]["evaluation_metrics"],
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
    if demo.get("slug") != "learned-csi-compression-feedback":
        raise ValueError("unexpected demo slug")
    expected_count = len(EXPECTED_METHODS) * len(EXPECTED_SNRS) * len(EXPECTED_SEEDS)
    recipes = result.get("recipes") or []
    if len(recipes) != expected_count:
        raise ValueError("benchmark does not contain the expected 60 runs")
    if any(recipe.get("status") != "completed" for recipe in recipes):
        raise ValueError("benchmark contains an incomplete recipe")


def _project_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for recipe in result["recipes"]:
        match = RECIPE_ID.fullmatch(str(recipe["id"]))
        if match is None:
            raise ValueError("unexpected recipe id: %s" % recipe["id"])
        method = match.group("method")
        token = match.group("snr")
        snr = -float(token[1:]) if token.startswith("m") else float(token)
        seed = int(match.group("seed"))
        metrics = recipe["metrics"]
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _number(snr),
                "paired_seed": seed,
                "spectral_efficiency_bps_hz": _number(metrics[RATE]),
                "rate_retention": _number(metrics[RETENTION]),
                "nmse_db": _number(metrics[NMSE_DB]),
                "phase_invariant_cosine": _number(metrics[COSINE]),
                "feedback_bits_per_sample": (
                    _number(metrics[BITS]) if BITS in metrics else ""
                ),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe["semantic_recipe_sha256"],
            }
        )
    expected = {
        (method, snr, seed)
        for method in EXPECTED_METHODS
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (str(row["method_id"]), float(row["snr_db"]), int(row["paired_seed"]))
        for row in records
    }
    if actual != expected:
        raise ValueError("benchmark cells do not match the frozen protocol")
    return sorted(
        records,
        key=lambda row: (
            float(row["snr_db"]),
            int(row["paired_seed"]),
            EXPECTED_METHODS.index(str(row["method_id"])),
        ),
    )


def _summary_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = _group(records)
    rows = []
    for snr in EXPECTED_SNRS:
        means = {
            method: {
                "rate": statistics.mean(
                    float(row["spectral_efficiency_bps_hz"])
                    for row in cells[(method, snr)]
                ),
                "retention": statistics.mean(
                    float(row["rate_retention"]) for row in cells[(method, snr)]
                ),
                "nmse": statistics.mean(
                    float(row["nmse_db"]) for row in cells[(method, snr)]
                ),
            }
            for method in EXPECTED_METHODS
        }
        rows.append(
            {
                "SNR (dB)": _number(snr),
                "Truncated rate (bit/s/Hz)": _number(
                    means["truncated_angular_delay"]["rate"]
                ),
                "KLT rate (bit/s/Hz)": _number(means["matched_klt"]["rate"]),
                "Learned rate (bit/s/Hz)": _number(means["learned_codec"]["rate"]),
                "Perfect-CSIT rate (bit/s/Hz)": _number(means["perfect_csit"]["rate"]),
                "Learned rate retention": _number(means["learned_codec"]["retention"]),
                "Learned NMSE (dB)": _number(means["learned_codec"]["nmse"]),
                "KLT NMSE (dB)": _number(means["matched_klt"]["nmse"]),
            }
        )
    return rows


def _chart_specs(
    result_dir: Path,
    result: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    charts = {
        "csi-feedback-rate": _metric_chart(
            records,
            metric="spectral_efficiency_bps_hz",
            methods=EXPECTED_METHODS,
            title="Downlink spectral efficiency after CSI feedback",
            description=(
                "All 128-bit methods and the non-deployable perfect-CSIT upper "
                "bound use the same paired held-out channel realizations."
            ),
            y_label="Spectral efficiency (bit/s/Hz)",
            allow_log=True,
        ),
        "csi-feedback-retention": _metric_chart(
            records,
            metric="rate_retention",
            methods=EXPECTED_METHODS,
            title="Perfect-CSIT downlink rate retained",
            description=(
                "Achieved MRT rate divided by perfect-CSIT MRT rate on the "
                "same channel realization."
            ),
            y_label="Rate retention",
            allow_log=False,
        ),
        "csi-feedback-nmse": _metric_chart(
            records,
            metric="nmse_db",
            methods=EXPECTED_METHODS[:-1],
            title="CSI reconstruction NMSE",
            description=(
                "Lower is better. Perfect CSIT is omitted because its numerical "
                "error floor would compress the 128-bit comparison."
            ),
            y_label="NMSE (dB)",
            allow_log=False,
        ),
    }
    response, error, evidence = _representative_charts(result_dir, result)
    charts["csi-feedback-response-preview"] = response
    charts["csi-feedback-error-preview"] = error
    return charts, evidence


def _metric_chart(
    records: list[Mapping[str, Any]],
    *,
    metric: str,
    methods: tuple[str, ...],
    title: str,
    description: str,
    y_label: str,
    allow_log: bool,
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in methods:
        values = []
        ranges = []
        for snr in EXPECTED_SNRS:
            samples = [float(row[metric]) for row in cells[(method, snr)]]
            mean = statistics.mean(samples)
            low, high = _confidence_interval(samples)
            values.append([snr, mean])
            ranges.append([snr, low, high])
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
        "title": title,
        "description": description,
        "xLabel": "Downlink SNR (dB)",
        "yLabel": y_label,
        "allowLog": bool(allow_log),
        "yIncludeZero": False,
        "series": series,
    }


def _representative_charts(
    result_dir: Path,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    method_previews = {}
    report_evidence = {}
    for method in EXPECTED_METHODS[:-1]:
        recipe_id = "%s_snr10_seed92001" % method
        recipe = next(item for item in result["recipes"] if item["id"] == recipe_id)
        evidence_dir = next(
            path
            for path in (result_dir / "run_evidence").iterdir()
            if path.name.endswith(recipe_id)
        )
        report_path = evidence_dir / "artifacts" / "evaluation" / "report.json"
        preview = _load_json(report_path)["metadata"]["csi_feedback_preview"]
        method_previews[method] = preview
        report_evidence[method] = {
            "recipe_id": recipe_id,
            "run_id": recipe["run_id"],
            "report": _file_evidence(report_path),
        }
    subcarriers = list(range(32))
    learned = method_previews["learned_codec"]
    response_series = [
        {
            "id": "true_channel",
            "label": "True channel magnitude",
            "color": "#d97706",
            "dash": [9, 3],
            "marker": "triangle",
            "values": [
                [index, value]
                for index, value in zip(subcarriers, learned["true_magnitude"][0])
            ],
        }
    ]
    error_series = []
    for method in EXPECTED_METHODS[:-1]:
        style = METHOD_STYLES[method]
        preview = method_previews[method]
        response_series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": [
                    [index, value]
                    for index, value in zip(
                        subcarriers, preview["reconstructed_magnitude"][0]
                    )
                ],
            }
        )
        error_series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": [
                    [index, value]
                    for index, value in zip(subcarriers, preview["absolute_error"][0])
                ],
            }
        )
    common = {
        "type": "line",
        "xLabel": "Subcarrier index",
        "allowLog": True,
        "yIncludeZero": True,
    }
    response = {
        **common,
        "title": "Representative reconstructed CSI magnitude",
        "description": (
            "Transmit antenna 0, sample 0, 10 dB, paired seed 92001. "
            "Every deployable method uses the same 128-bit budget."
        ),
        "yLabel": "Channel magnitude",
        "series": response_series,
    }
    error = {
        **common,
        "title": "Representative complex reconstruction error",
        "description": (
            "Absolute complex error for the same channel sample. Lower is better."
        ),
        "yLabel": "Absolute complex error",
        "series": error_series,
    }
    evidence = {
        "snr_db": REPRESENTATIVE_SNR,
        "paired_seed": REPRESENTATIVE_SEED,
        "sample_index": int(learned["sample_index"]),
        "tx_antenna_index": 0,
        "reports": report_evidence,
    }
    return response, error, evidence


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


def _number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite benchmark value")
    return format(number, ".12g")


if __name__ == "__main__":
    raise SystemExit(main())
