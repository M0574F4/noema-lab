from __future__ import annotations

"""Project the completed MIMO-OFDM estimator benchmark into docs assets."""

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
    "20260726T235356Z_"
    "mimo_ofdm.learned_channel_estimation_post_training_v2"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = (
    ROOT
    / "docs"
    / "demo"
    / "data"
    / "mimo_ofdm_channel_estimation"
)
DEFAULT_CHART_JS = (
    ROOT / "docs" / "_static" / "noema-mimo-channel-estimation-chart-data.js"
)

EXPECTED_BENCHMARK_ID = "mimo_ofdm.learned_channel_estimation_post_training_v2"
EXPECTED_VERSION = "2.0.0"
EXPECTED_METHODS = (
    "least_squares",
    "fixed_prior_lmmse",
    "learned_estimator",
)
EXPECTED_SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
EXPECTED_SEEDS = (91001, 92001, 93001)
EXPECTED_PROFILE_BY_SEED = {
    91001: "A",
    92001: "C",
    93001: "E",
}
T95_DF2 = 4.302652729911275
REPRESENTATIVE_SNR = 10.0
REPRESENTATIVE_SEED = 92001
REPRESENTATIVE_PROFILE = "C"

NMSE_DB = "mimo.channel_estimation.nmse_db"
ZF_RATE = "mimo.channel_estimation.zf_spectral_efficiency_bps_hz"
PERFECT_ZF_RATE = (
    "mimo.channel_estimation.perfect_csi_zf_spectral_efficiency_bps_hz"
)
ZF_RETENTION = "mimo.channel_estimation.zf_rate_retention"
CORRELATION = "channel_estimation.complex_correlation"

METHOD_STYLES = {
    "least_squares": {
        "label": "Least squares",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "fixed_prior_lmmse": {
        "label": "Fixed-prior LMMSE",
        "color": "#d97706",
        "dash": [3, 3],
        "marker": "diamond",
    },
    "learned_estimator": {
        "label": "Learned dual-domain estimator",
        "color": "#dc2626",
        "dash": [],
        "marker": "circle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>least_squares|fixed_prior_lmmse|learned_estimator)"
    r"_tdl(?P<profile>[ace])_snr(?P<snr>m?[0-9]+)_seed(?P<seed>[0-9]+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact documentation data from the completed "
            "MIMO-OFDM channel-estimation benchmark."
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
        "slug": "learned-mimo-ofdm-channel-estimation",
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
            "statistical_unit": "paired held-out TDL profile/channel seed",
            "paired_seeds": list(EXPECTED_SEEDS),
            "paired_profiles": [
                EXPECTED_PROFILE_BY_SEED[seed] for seed in EXPECTED_SEEDS
            ],
            "sample_count_per_method_and_snr": len(EXPECTED_SEEDS),
            "interval": (
                "two-sided Student-t 95% confidence interval over paired "
                "held-out profile/channel units"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three heterogeneous paired profile/channel units support an "
                "experimental tutorial result, not a paper-grade population claim."
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
    if demo.get("slug") != "learned-mimo-ofdm-channel-estimation":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    expected_count = len(EXPECTED_METHODS) * len(EXPECTED_SNRS) * len(
        EXPECTED_SEEDS
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
        snr_token = match.group("snr")
        snr = -float(snr_token[1:]) if snr_token.startswith("m") else float(
            snr_token
        )
        seed = int(match.group("seed"))
        profile = match.group("profile").upper()
        if EXPECTED_PROFILE_BY_SEED.get(seed) != profile:
            raise ValueError(
                "recipe profile %s does not match the frozen profile/seed design"
                % profile
            )
        metrics = recipe["metrics"]
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _format_number(snr),
                "tdl_profile": profile,
                "paired_seed": seed,
                "nmse_db": _format_number(metrics[NMSE_DB]),
                "zf_spectral_efficiency_bps_hz": _format_number(
                    metrics[ZF_RATE]
                ),
                "perfect_csi_zf_spectral_efficiency_bps_hz": _format_number(
                    metrics[PERFECT_ZF_RATE]
                ),
                "zf_rate_retention": _format_number(metrics[ZF_RETENTION]),
                "complex_correlation": _format_number(metrics[CORRELATION]),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe[
                    "semantic_recipe_sha256"
                ],
            }
        )
    expected_cells = {
        (method, snr, EXPECTED_PROFILE_BY_SEED[seed], seed)
        for method in EXPECTED_METHODS
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    actual_cells = {
        (
            row["method_id"],
            float(row["snr_db"]),
            str(row["tdl_profile"]),
            int(row["paired_seed"]),
        )
        for row in records
    }
    if actual_cells != expected_cells:
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
                "nmse": statistics.mean(
                    float(row["nmse_db"])
                    for row in cells[(method, snr)]
                ),
                "zf": statistics.mean(
                    float(row["zf_spectral_efficiency_bps_hz"])
                    for row in cells[(method, snr)]
                ),
                "retention": statistics.mean(
                    float(row["zf_rate_retention"])
                    for row in cells[(method, snr)]
                ),
            }
            for method in EXPECTED_METHODS
        }
        strongest = min(
            ("least_squares", "fixed_prior_lmmse"),
            key=lambda method: means[method]["nmse"],
        )
        rows.append(
            {
                "SNR (dB)": _format_number(snr),
                "LS NMSE (dB)": _format_number(
                    means["least_squares"]["nmse"]
                ),
                "Fixed-prior LMMSE NMSE (dB)": _format_number(
                    means["fixed_prior_lmmse"]["nmse"]
                ),
                "Learned NMSE (dB)": _format_number(
                    means["learned_estimator"]["nmse"]
                ),
                "Learned − LS NMSE (dB)": _format_number(
                    means["learned_estimator"]["nmse"]
                    - means["least_squares"]["nmse"]
                ),
                "Strongest NMSE baseline": METHOD_STYLES[strongest]["label"],
                "Learned − strongest baseline (dB)": _format_number(
                    means["learned_estimator"]["nmse"]
                    - means[strongest]["nmse"]
                ),
                "Learned post-ZF rate (bit/s/Hz)": _format_number(
                    means["learned_estimator"]["zf"]
                ),
                "Learned / perfect-CSI ZF rate": _format_number(
                    means["learned_estimator"]["retention"]
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
        "mimo-channel-estimation-nmse": _metric_chart(
            records,
            metric="nmse_db",
            title="Channel-estimation NMSE vs SNR",
            description=(
                "Lower is better. Points are means over paired TDL-A/C/E "
                "profile/channel units; bands are Student-t 95% confidence intervals."
            ),
            y_label="NMSE (dB)",
            allow_log=False,
        ),
        "mimo-channel-estimation-profile-gain": _profile_gain_chart(records),
        "mimo-channel-estimation-zf-rate": _metric_chart(
            records,
            metric="zf_spectral_efficiency_bps_hz",
            title="Post-ZF spectral efficiency vs SNR",
            description=(
                "System-level rate obtained when the ZF receiver is designed "
                "from each channel estimate."
            ),
            y_label="Spectral efficiency (bit/s/Hz)",
            allow_log=True,
        ),
        "mimo-channel-estimation-zf-retention": _metric_chart(
            records,
            metric="zf_rate_retention",
            title="Estimated-CSI ZF rate relative to perfect-CSI ZF",
            description=(
                "Estimated-CSI post-ZF rate divided by exact-channel ZF on "
                "the same paired channel. Values above one are possible because "
                "pure ZF is not a capacity or noise-robust oracle."
            ),
            y_label="Estimated / exact-channel ZF rate",
            allow_log=False,
        ),
    }
    recipe_id = "learned_estimator_tdlc_snr10_seed92001"
    recipe = next(
        item for item in result["recipes"] if item["id"] == recipe_id
    )
    evidence_dir = next(
        path
        for path in (result_dir / "run_evidence").iterdir()
        if path.name.endswith(recipe_id)
    )
    report_path = evidence_dir / "artifacts" / "evaluation" / "report.json"
    preview = _load_json(report_path)["metadata"]["channel_estimation_preview"]
    link = 0
    subcarriers = list(range(len(preview["true_magnitude"][link])))
    charts["mimo-channel-estimation-response-preview"] = {
        "type": "line",
        "title": "Representative learned frequency response",
        "description": (
            "TDL-C RX0–TX0 at 10 dB for paired seed 92001. Error uses the "
            "same magnitude axis as the true and learned channel."
        ),
        "xLabel": "Subcarrier index",
        "yLabel": "Magnitude",
        "allowLog": True,
        "yIncludeZero": True,
        "series": [
            {
                "id": "true_channel",
                "label": "True channel magnitude",
                "color": "#0284c7",
                "dash": [7, 3],
                "marker": "square",
                "values": [
                    [index, value]
                    for index, value in zip(
                        subcarriers,
                        preview["true_magnitude"][link],
                    )
                ],
            },
            {
                "id": "learned_channel",
                "label": "Learned estimate magnitude",
                "color": "#dc2626",
                "dash": [],
                "marker": "circle",
                "values": [
                    [index, value]
                    for index, value in zip(
                        subcarriers,
                        preview["estimated_magnitude"][link],
                    )
                ],
            },
            {
                "id": "absolute_error",
                "label": "Complex absolute error",
                "color": "#9333ea",
                "dash": [3, 3],
                "marker": "diamond",
                "values": [
                    [index, value]
                    for index, value in zip(
                        subcarriers,
                        preview["absolute_error"][link],
                    )
                ],
            },
        ],
    }
    representative = {
        "recipe_id": recipe_id,
        "run_id": recipe["run_id"],
        "snr_db": REPRESENTATIVE_SNR,
        "paired_seed": REPRESENTATIVE_SEED,
        "tdl_profile": REPRESENTATIVE_PROFILE,
        "link": preview["link_labels"][link],
        "subcarrier_count": len(subcarriers),
        "estimator_label": preview["estimator_label"],
        "report": _file_evidence(report_path),
    }
    return charts, representative


def _metric_chart(
    records: list[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    description: str,
    y_label: str,
    allow_log: bool,
) -> dict[str, Any]:
    cells = _group(records)
    series = []
    for method in EXPECTED_METHODS:
        values = []
        ranges = []
        for snr in EXPECTED_SNRS:
            samples = [
                float(row[metric]) for row in cells[(method, snr)]
            ]
            mean = statistics.mean(samples)
            lower, upper = _confidence_interval(samples)
            values.append([snr, mean])
            ranges.append([snr, lower, upper])
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
        "xLabel": "Channel SNR (dB)",
        "yLabel": y_label,
        "allowLog": bool(allow_log),
        "yIncludeZero": False,
        "series": series,
    }


def _profile_gain_chart(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_cell = {
        (
            str(row["method_id"]),
            float(row["snr_db"]),
            str(row["tdl_profile"]),
        ): row
        for row in records
    }
    styles = {
        "A": {"color": "#0f766e", "marker": "square", "dash": []},
        "C": {"color": "#7c3aed", "marker": "diamond", "dash": [5, 3]},
        "E": {"color": "#be123c", "marker": "circle", "dash": [2, 3]},
    }
    series = []
    for profile in ("A", "C", "E"):
        values = []
        for snr in EXPECTED_SNRS:
            learned = float(
                by_cell[("learned_estimator", snr, profile)]["nmse_db"]
            )
            strongest = min(
                float(by_cell[("least_squares", snr, profile)]["nmse_db"]),
                float(
                    by_cell[("fixed_prior_lmmse", snr, profile)]["nmse_db"]
                ),
            )
            values.append([snr, learned - strongest])
        style = styles[profile]
        series.append(
            {
                "id": "tdl_%s" % profile.lower(),
                "label": "TDL-%s" % profile,
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "values": values,
            }
        )
    return {
        "type": "line",
        "title": "Learned NMSE advantage by TDL profile",
        "description": (
            "Learned NMSE minus the better of LS and fixed-prior LMMSE in "
            "each paired profile/SNR cell. Negative values favor the learned estimator."
        ),
        "xLabel": "Channel SNR (dB)",
        "yLabel": "Learned − strongest baseline NMSE (dB)",
        "allowLog": False,
        "yIncludeZero": True,
        "series": series,
    }


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
