"""Project a completed modulation-recognition benchmark into compact docs assets."""

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
    "20260727T150344Z_neural_receiver_ai_phy."
    "learned_modulation_recognition_blind_carrier_post_training_v1"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "modulation_recognition"
DEFAULT_CHART_JS = (
    ROOT / "docs" / "_static" / "noema-modulation-recognition-chart-data.js"
)

EXPECTED_BENCHMARK_ID = (
    "neural_receiver_ai_phy."
    "learned_modulation_recognition_blind_carrier_post_training_v1"
)
EXPECTED_VERSION = "1.0.0"
EXPECTED_METHODS = (
    "blind_cumulant",
    "learned_classifier",
    "oracle_likelihood",
)
EXPECTED_SNRS = (-2.0, 2.0, 6.0, 10.0, 14.0, 18.0)
EXPECTED_SEEDS = (71001, 72001, 73001)
T95_DF2 = 4.302652729911275

ACCURACY = "modulation_recognition.accuracy"
BALANCED_ACCURACY = "modulation_recognition.balanced_accuracy"
MACRO_F1 = "modulation_recognition.macro_f1"
FRAME_COUNT = "modulation_recognition.frame_count"

METHOD_STYLES = {
    "blind_cumulant": {
        "label": "Blind differential cumulant",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "learned_classifier": {
        "label": "Learned blind-carrier classifier",
        "color": "#16a34a",
        "dash": [],
        "marker": "circle",
    },
    "oracle_likelihood": {
        "label": "Oracle-synchronized likelihood",
        "color": "#d97706",
        "dash": [10, 3],
        "marker": "triangle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>blind_cumulant|learned_classifier|oracle_likelihood)"
    r"_snr(?P<snr>m?[0-9]+)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate documentation assets from a completed AMC benchmark."
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
    evidence_path = result_dir / "training_evidence" / "manifest.json"
    result = _load_json(result_path)
    _validate_result(result)
    records = _project_records(result)
    summaries = _summary_rows(records)
    charts = {
        "modulation-recognition-accuracy": _metric_chart(
            records,
            metric="accuracy",
            title="Blind-carrier modulation recognition",
            description=(
                "Mean accuracy over three paired held-out symbol, carrier, and "
                "noise seeds. Bands are two-sided Student-t 95% intervals."
            ),
            y_label="Accuracy",
        ),
        "modulation-recognition-macro-f1": _metric_chart(
            records,
            metric="macro_f1",
            title="Class-balanced recognition quality",
            description=(
                "Macro F1 gives BPSK, QPSK, and 16-QAM equal weight at each SNR."
            ),
            y_label="Macro F1",
        ),
    }

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

    manifest = {
        "kind": "noema.docs_demo_snapshot",
        "schema_version": 1,
        "slug": "learned-automatic-modulation-recognition",
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
                "training_evidence/manifest.json": _file_evidence(evidence_path),
            },
        },
        "statistical_design": {
            "aggregation_cell": "channel SNR",
            "statistical_unit": "paired held-out symbol/carrier/noise seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_snr": len(EXPECTED_SEEDS),
            "frames_per_run": 1536,
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out seeds"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three paired seeds support an experimental tutorial result, "
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
        "runs": [
            {
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
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
    if demo.get("slug") != "learned-automatic-modulation-recognition":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    expected_count = len(EXPECTED_METHODS) * len(EXPECTED_SNRS) * len(EXPECTED_SEEDS)
    if len(recipes) != expected_count:
        raise ValueError("benchmark does not contain the expected 54 runs")
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
        metrics = recipe["metrics"]
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _number(snr),
                "paired_seed": int(match.group("seed")),
                "accuracy": _number(metrics[ACCURACY]),
                "balanced_accuracy": _number(metrics[BALANCED_ACCURACY]),
                "macro_f1": _number(metrics[MACRO_F1]),
                "frame_count": int(metrics[FRAME_COUNT]),
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
                metric: statistics.mean(
                    float(row[metric]) for row in cells[(method, snr)]
                )
                for metric in ("accuracy", "macro_f1")
            }
            for method in EXPECTED_METHODS
        }
        rows.append(
            {
                "SNR (dB)": _number(snr),
                "Blind accuracy": _number(means["blind_cumulant"]["accuracy"]),
                "Learned accuracy": _number(
                    means["learned_classifier"]["accuracy"]
                ),
                "Oracle accuracy": _number(
                    means["oracle_likelihood"]["accuracy"]
                ),
                "Blind macro F1": _number(means["blind_cumulant"]["macro_f1"]),
                "Learned macro F1": _number(
                    means["learned_classifier"]["macro_f1"]
                ),
                "Oracle macro F1": _number(
                    means["oracle_likelihood"]["macro_f1"]
                ),
            }
        )
    return rows


def _metric_chart(
    records: list[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    description: str,
    y_label: str,
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in EXPECTED_METHODS:
        values = []
        ranges = []
        for snr in EXPECTED_SNRS:
            samples = [float(row[metric]) for row in cells[(method, snr)]]
            mean = statistics.mean(samples)
            low, high = _confidence_interval(samples)
            values.append([snr, mean])
            ranges.append([snr, max(0.0, low), min(1.0, high)])
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
        "xLabel": "SNR (dB)",
        "yLabel": y_label,
        "allowLog": False,
        "yIncludeZero": False,
        "series": series,
    }


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
