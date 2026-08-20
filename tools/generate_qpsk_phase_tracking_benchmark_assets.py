from __future__ import annotations

"""Project the completed QPSK carrier-tracking benchmark into docs assets."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ID = (
    "20260727T005814Z_"
    "neural_receiver_ai_phy.learned_qpsk_phase_tracking_v2"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = (
    ROOT / "docs" / "demo" / "data" / "qpsk_phase_tracking"
)
DEFAULT_CHART_JS = (
    ROOT / "docs" / "_static" / "noema-qpsk-phase-tracking-chart-data.js"
)
DEFAULT_RUNS_DIR = ROOT / ".noema" / "runs"

EXPECTED_BENCHMARK_ID = (
    "neural_receiver_ai_phy.learned_qpsk_phase_tracking_v2"
)
EXPECTED_VERSION = "2.0.0"
EXPECTED_METHODS = (
    "uncompensated_qpsk",
    "pilot_interpolation",
    "pilot_smoothing",
    "decision_directed_pll",
    "learned_receiver",
    "oracle_phase",
)
EXPECTED_SNRS = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
EXPECTED_SEEDS = (91001, 92001, 93001)
T95_DF2 = 4.302652729911275
BER = "channel.coded.ber"
BLER = "channel.coded.bler"
BIT_ERRORS = "channel.coded.error_count"
BIT_COUNT = "channel.coded.compare_bit_count"
BLOCK_ERRORS = "channel.coded.block_error_count"
BLOCK_COUNT = "channel.coded.block_count"

METHOD_STYLES = {
    "uncompensated_qpsk": {
        "label": "Uncompensated QPSK",
        "color": "#64748b",
        "dash": [8, 5],
        "marker": "square",
    },
    "pilot_interpolation": {
        "label": "Pilot interpolation",
        "color": "#0284c7",
        "dash": [5, 3],
        "marker": "diamond",
    },
    "pilot_smoothing": {
        "label": "Pilot smoothing",
        "color": "#2563eb",
        "dash": [],
        "marker": "circle",
    },
    "decision_directed_pll": {
        "label": "Decision-directed PLL",
        "color": "#7c3aed",
        "dash": [3, 3],
        "marker": "triangle",
    },
    "learned_receiver": {
        "label": "Learned temporal receiver",
        "color": "#dc2626",
        "dash": [],
        "marker": "circle",
    },
    "oracle_phase": {
        "label": "Oracle phase correction",
        "color": "#059669",
        "dash": [7, 3],
        "marker": "triangle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>uncompensated_qpsk|pilot_interpolation|pilot_smoothing|"
    r"decision_directed_pll|learned_receiver|oracle_phase)"
    r"_snr(?P<snr>m?[0-9]+)_seed(?P<seed>[0-9]+)$"
)

REPRESENTATIVE_RUNS = {
    "pilot_smoothing": (
        "20260723T222136Z_qpsk_pilot_phase_tracking_2048_snapshot__"
        "channel.snr_db_6__receiver.mode_pilot_smoothing"
    ),
    "decision_directed_pll": (
        "20260723T222137Z_qpsk_pilot_phase_tracking_2048_snapshot__"
        "channel.snr_db_6__receiver.mode_decision_directed_pll"
    ),
    "oracle_phase": (
        "20260723T222138Z_qpsk_pilot_phase_tracking_2048_snapshot__"
        "channel.snr_db_6__receiver.mode_oracle"
    ),
    "learned_receiver": (
        "20260723T222140Z_qpsk_pilot_phase_tracking_2048_snapshot__"
        "channel.snr_db_6__receiver.mode_learned_artifact"
    ),
}
TRACE_METHODS = (
    "pilot_smoothing",
    "decision_directed_pll",
    "learned_receiver",
    "oracle_phase",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate documentation data from the completed QPSK "
            "carrier-tracking benchmark."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chart-js", type=Path, default=DEFAULT_CHART_JS)
    parser.add_argument(
        "--import-representative-runs",
        type=Path,
        metavar="RUNS_DIR",
        help=(
            "refresh the compact representative phase trace from the named "
            "local run store before generating the documentation assets"
        ),
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if args.import_representative_runs is not None:
        _import_representative_trace(
            args.import_representative_runs.expanduser().resolve(),
            data_dir,
        )
    generated = generate(
        args.result_dir.expanduser().resolve(),
        data_dir,
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
    trace_path = data_dir / "representative_phase_trace.csv"
    trace_manifest_path = data_dir / "representative_phase_manifest.json"
    trace = _load_trace(trace_path, trace_manifest_path)
    charts = _chart_specs(records, trace)

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
        "slug": "learned-qpsk-phase-tracking-receiver",
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
            "aggregation_cell": "channel SNR",
            "statistical_unit": "paired held-out payload/noise/impairment seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_snr": len(EXPECTED_SEEDS),
            "compared_bits_per_run": 262144,
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out seed units"
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
                "path": str(chart_js.relative_to(ROOT)),
                **_file_evidence(chart_js),
            },
        },
        "representative_phase_trace": {
            "scope": (
                "qualitative 6 dB packet trace from the preserved paired UI "
                "snapshot; statistical BER claims use the 126-run benchmark"
            ),
            "trace": _file_evidence(trace_path),
            "manifest": _file_evidence(trace_manifest_path),
        },
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
        "representative_trace": trace_path,
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
    if demo.get("slug") != "learned-qpsk-phase-tracking-receiver":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    expected = len(EXPECTED_METHODS) * len(EXPECTED_SNRS) * len(
        EXPECTED_SEEDS
    )
    if len(recipes) != expected:
        raise ValueError(
            "benchmark result does not contain the expected %d runs" % expected
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
        token = match.group("snr")
        snr = -float(token[1:]) if token.startswith("m") else float(token)
        seed = int(match.group("seed"))
        metrics = recipe["metrics"]
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _format_number(snr),
                "paired_seed": seed,
                "ber": _format_number(metrics[BER]),
                "bit_errors": int(metrics[BIT_ERRORS]),
                "compared_bits": int(metrics[BIT_COUNT]),
                "bler": _format_number(metrics[BLER]),
                "block_errors": int(metrics[BLOCK_ERRORS]),
                "compared_blocks": int(metrics[BLOCK_COUNT]),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe[
                    "semantic_recipe_sha256"
                ],
            }
        )
    expected_cells = {
        (method, snr, seed)
        for method in EXPECTED_METHODS
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    actual_cells = {
        (
            str(row["method_id"]),
            float(row["snr_db"]),
            int(row["paired_seed"]),
        )
        for row in records
    }
    if actual_cells != expected_cells:
        raise ValueError("benchmark cells do not match the frozen protocol")
    if {int(row["compared_bits"]) for row in records} != {262144}:
        raise ValueError("benchmark does not use the frozen BER denominator")
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
            method: statistics.mean(
                float(row["ber"]) for row in cells[(method, snr)]
            )
            for method in EXPECTED_METHODS
        }
        practical = min(
            (
                "pilot_interpolation",
                "pilot_smoothing",
                "decision_directed_pll",
            ),
            key=lambda method: means[method],
        )
        rows.append(
            {
                "SNR (dB)": _format_number(snr),
                "Uncompensated BER": _format_number(
                    means["uncompensated_qpsk"]
                ),
                "Pilot interpolation BER": _format_number(
                    means["pilot_interpolation"]
                ),
                "Pilot smoothing BER": _format_number(
                    means["pilot_smoothing"]
                ),
                "Decision-directed PLL BER": _format_number(
                    means["decision_directed_pll"]
                ),
                "Learned BER": _format_number(means["learned_receiver"]),
                "Oracle BER": _format_number(means["oracle_phase"]),
                "Best deployable classical tracker": METHOD_STYLES[practical][
                    "label"
                ],
                "Learned BER reduction vs best classical (%)": _format_number(
                    100.0
                    * (means[practical] - means["learned_receiver"])
                    / means[practical]
                ),
            }
        )
    return rows


def _chart_specs(
    records: list[Mapping[str, Any]],
    trace: list[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in EXPECTED_METHODS:
        values = []
        ranges = []
        for snr in EXPECTED_SNRS:
            samples = [
                float(row["ber"]) for row in cells[(method, snr)]
            ]
            mean = statistics.mean(samples)
            lower, upper = _confidence_interval(samples)
            values.append([snr, mean])
            ranges.append([snr, max(lower, 1e-8), max(upper, 1e-8)])
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
    charts = {
        "qpsk-phase-tracking-ber": {
            "type": "line",
            "title": "Data-bit BER vs SNR",
            "description": (
                "Means over three paired held-out payload/noise/impairment "
                "seeds; bands are Student-t 95% confidence intervals."
            ),
            "xLabel": "Channel SNR (dB)",
            "yLabel": "Data-bit error rate",
            "allowLog": True,
            "yScale": "log",
            "yIncludeZero": False,
            "series": series,
        },
        "qpsk-phase-tracking-phase-estimates": {
            "type": "line",
            "title": "Representative carrier-phase tracking at 6 dB",
            "description": (
                "Unwrapped phase estimates for one paired packet. The "
                "simulated truth is shown in orange; the oracle overlaps it."
            ),
            "xLabel": "Frame symbol index",
            "yLabel": "Unwrapped carrier phase (rad)",
            "allowLog": False,
            "yIncludeZero": False,
            "series": _trace_series(trace, error=False),
        },
        "qpsk-phase-tracking-phase-error": {
            "type": "line",
            "title": "Phase-estimation error for the same packet",
            "description": (
                "Circular estimate-minus-truth error. Values nearer zero "
                "mean more accurate phase correction."
            ),
            "xLabel": "Frame symbol index",
            "yLabel": "Circular phase error (rad)",
            "allowLog": False,
            "yIncludeZero": True,
            "series": _trace_series(trace, error=True),
        },
    }
    return charts


def _trace_series(
    trace: list[Mapping[str, Any]],
    *,
    error: bool,
) -> list[dict[str, Any]]:
    series = []
    if not error:
        series.append(
            {
                "id": "simulated_truth",
                "label": "Simulated carrier phase",
                "color": "#d97706",
                "dash": [8, 4],
                "marker": "square",
                "values": [
                    [
                        int(row["frame_symbol_index"]),
                        float(row["true_phase_rad"]),
                    ]
                    for row in trace
                ],
            }
        )
    for method in TRACE_METHODS:
        style = METHOD_STYLES[method]
        field = (
            "%s_error_rad" % method
            if error
            else "%s_phase_rad" % method
        )
        series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": [
                    [int(row["frame_symbol_index"]), float(row[field])]
                    for row in trace
                ],
            }
        )
    return series


def _import_representative_trace(runs_dir: Path, data_dir: Path) -> None:
    traces: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    files: dict[str, Any] = {}
    for method, run_id in REPRESENTATIVE_RUNS.items():
        run_dir = runs_dir / run_id
        diagnostics_path = (
            run_dir / "artifacts" / "demodulator" / "diagnostics.npz"
        )
        truth_path = (
            run_dir
            / "artifacts"
            / "carrier_impairment"
            / "phase_truth.npz"
        )
        with np.load(diagnostics_path, allow_pickle=False) as payload:
            traces[method] = np.asarray(
                payload["phase_estimate_rad"], dtype=np.float64
            )[0]
        with np.load(truth_path, allow_pickle=False) as payload:
            truths[method] = np.asarray(
                payload["phase_rad"], dtype=np.float64
            )[0]
        files[method] = {
            "run_id": run_id,
            "diagnostics": _file_evidence(diagnostics_path),
            "phase_truth": _file_evidence(truth_path),
        }
    reference_truth = truths["learned_receiver"]
    if any(
        not np.array_equal(reference_truth, truth)
        for truth in truths.values()
    ):
        raise ValueError("representative methods do not share phase truth")
    if any(trace.shape != reference_truth.shape for trace in traces.values()):
        raise ValueError("representative phase arrays have inconsistent shapes")

    indices = np.unique(
        np.linspace(0, reference_truth.size - 1, 160, dtype=np.int64)
    )
    truth_unwrapped = np.unwrap(reference_truth)
    estimates = {method: np.unwrap(value) for method, value in traces.items()}
    errors = {
        method: np.angle(np.exp(1j * (value - reference_truth)))
        for method, value in traces.items()
    }
    rows = []
    for index in indices:
        row: dict[str, Any] = {
            "frame_symbol_index": int(index),
            "true_phase_rad": _format_number(truth_unwrapped[index]),
        }
        for method in TRACE_METHODS:
            row["%s_phase_rad" % method] = _format_number(
                estimates[method][index]
            )
            row["%s_error_rad" % method] = _format_number(
                errors[method][index]
            )
        rows.append(row)

    data_dir.mkdir(parents=True, exist_ok=True)
    trace_path = data_dir / "representative_phase_trace.csv"
    manifest_path = data_dir / "representative_phase_manifest.json"
    _write_csv(trace_path, rows)
    manifest = {
        "kind": "noema.docs_representative_phase_trace",
        "schema_version": 1,
        "snr_db": 6,
        "packet_index": 0,
        "source_symbol_count": int(reference_truth.size),
        "projected_point_count": len(rows),
        "source_runs": files,
        "trace": _file_evidence(trace_path),
        "limitation": (
            "Qualitative paired packet from the preserved UI snapshot; "
            "statistical claims use the completed 126-run benchmark."
        ),
    }
    _write_json(manifest_path, manifest)


def _load_trace(path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    if manifest.get("trace", {}).get("sha256") != _sha256(path):
        raise ValueError("representative phase manifest does not bind its CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(manifest["projected_point_count"]):
        raise ValueError("representative phase trace has the wrong row count")
    required = {"frame_symbol_index", "true_phase_rad"}
    required.update(
        "%s_%s_rad" % (method, suffix)
        for method in TRACE_METHODS
        for suffix in ("phase", "error")
    )
    if not rows or set(rows[0]) != required:
        raise ValueError("representative phase trace has the wrong schema")
    return rows


def _group(
    records: list[Mapping[str, Any]],
) -> dict[tuple[str, float], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(
        list
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite benchmark value")
    return format(number, ".12g")


if __name__ == "__main__":
    raise SystemExit(main())
