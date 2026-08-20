"""Project the completed slow-Rayleigh DeepJSCC benchmark into docs assets."""

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
    "20260728T002720Z_semantic_comm.deepjscc_slow_rayleigh_post_training_v1"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "deepjscc_slow_rayleigh"
DEFAULT_CHART_JS = (
    ROOT
    / "docs"
    / "_static"
    / "noema-deepjscc-slow-rayleigh-chart-data-v2.js"
)

EXPECTED_BENCHMARK_ID = (
    "semantic_comm.deepjscc_slow_rayleigh_post_training_v1"
)
EXPECTED_VERSION = "1.0.0"
EXPECTED_SNRS = (0.0, 5.0, 10.0, 15.0, 20.0)
EXPECTED_CHANNELS = (8, 16, 32)
EXPECTED_KAPPAS = (0.125, 0.25, 0.5)
EXPECTED_SEEDS = (81001, 82001, 83001)
RATE_SNR = 10.0
T95_DF2 = 4.302652729911275

METHOD_STYLES = {
    "digital": {
        "label": "JPEG + ideal separation",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "learned": {
        "label": "Blind DeepJSCC",
        "color": "#16a34a",
        "dash": [],
        "marker": "circle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>digital|learned)_"
    r"(?P<snr>m?[0-9]+(?:p[0-9]+)?)_"
    r"c(?P<channels>[0-9]+)_"
    r"seed(?P<seed>[0-9]+)_"
    r"(?P<sweep>snr_sweep|rate_sweep)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs assets from the slow-Rayleigh DeepJSCC result."
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
    training_path = result_dir / "training_evidence" / "manifest.json"
    result = _load_json(result_path)
    _validate_result(result)
    records = _project_records(result)
    pairing = _pairing_rows(records)
    snr_summary = _snr_summary_rows(records)
    rate_summary = _rate_summary_rows(records)
    charts = _build_charts(records)

    data_dir.mkdir(parents=True, exist_ok=True)
    chart_js.parent.mkdir(parents=True, exist_ok=True)
    projection_path = data_dir / "benchmark_projection.csv"
    pairing_path = data_dir / "pairing_audit.csv"
    snr_summary_path = data_dir / "snr_summary.csv"
    rate_summary_path = data_dir / "rate_summary.csv"
    chart_data_path = data_dir / "chart_data.json"
    manifest_path = data_dir / "snapshot_manifest.json"
    _write_csv(projection_path, records)
    _write_csv(pairing_path, pairing)
    _write_csv(snr_summary_path, snr_summary)
    _write_csv(rate_summary_path, rate_summary)
    _write_json(chart_data_path, charts)
    _write_chart_js(chart_js, charts)
    manifest = {
        "kind": "noema.docs_demo_snapshot",
        "schema_version": 1,
        "slug": "deepjscc-slow-rayleigh",
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
                    training_path
                ),
            },
        },
        "statistical_design": {
            "held_out_images": ["kodim21", "kodim22", "kodim23", "kodim24"],
            "held_out_channel_seeds": list(EXPECTED_SEEDS),
            "samples_per_method_and_cell": len(EXPECTED_SEEDS),
            "snr_sweep_bandwidth_ratio": 0.5,
            "rate_slice_average_snr_db": RATE_SNR,
            "rate_slice_bandwidth_ratios": list(EXPECTED_KAPPAS),
            "pairing": (
                "Within each coordinate and seed, digital and learned methods "
                "use the same four per-image slow-Rayleigh gains."
            ),
            "interval": (
                "two-sided Student-t 95% confidence interval over three "
                "held-out channel seeds"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three channel seeds support an experimental tutorial result, "
                "not a paper-grade population claim."
            ),
        },
        "projection": {
            "rows": len(records),
            "benchmark_projection.csv": _file_evidence(projection_path),
            "pairing_audit.csv": _file_evidence(pairing_path),
            "snr_summary.csv": _file_evidence(snr_summary_path),
            "rate_summary.csv": _file_evidence(rate_summary_path),
            "chart_data.json": _file_evidence(chart_data_path),
            "chart_javascript": {
                "path": str(chart_js.relative_to(ROOT)),
                **_file_evidence(chart_js),
            },
        },
        "publication": {
            "publication_ready": False,
            "benchmark_tier": "experimental",
            "dataset_rights_status": "authoritative Kodak terms not archived",
            "excluded_public_assets": ["representative_reconstructions"],
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
        "pairing_audit": pairing_path,
        "snr_summary": snr_summary_path,
        "rate_summary": rate_summary_path,
        "chart_data": chart_data_path,
        "chart_javascript": chart_js,
        "manifest": manifest_path,
    }


def _validate_result(result: Mapping[str, Any]) -> None:
    benchmark = result.get("benchmark") or {}
    demo = (benchmark.get("metadata") or {}).get("demo") or {}
    if result.get("status") != "completed":
        raise ValueError("benchmark result is not completed")
    if benchmark.get("id") != EXPECTED_BENCHMARK_ID:
        raise ValueError("unexpected benchmark id")
    if benchmark.get("version") != EXPECTED_VERSION:
        raise ValueError("unexpected benchmark version")
    if demo.get("slug") != "deepjscc-slow-rayleigh":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    if len(recipes) != 48:
        raise ValueError("benchmark must contain exactly 48 runs")
    if any(recipe.get("status") != "completed" for recipe in recipes):
        raise ValueError("benchmark contains an incomplete recipe")


def _project_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for recipe in result["recipes"]:
        match = RECIPE_ID.fullmatch(str(recipe["id"]))
        if match is None:
            raise ValueError("unexpected recipe id: %s" % recipe["id"])
        method = match.group("method")
        snr = _decode_number(match.group("snr"))
        channels = int(match.group("channels"))
        seed = int(match.group("seed"))
        sweep = match.group("sweep")
        kappa = float(channels) / 64.0
        metrics = recipe["metrics"]
        expected = {
            "quality.psnr_db",
            "quality.ms_ssim",
            "quality.mse",
            "channel.snr_db",
            "channel.uses_per_pixel",
            "channel.gain_power.minimum",
            "channel.gain_power.maximum",
        }
        missing = sorted(expected.difference(metrics))
        if missing:
            raise ValueError(
                "%s omits metric(s): %s"
                % (recipe["id"], ", ".join(missing))
            )
        observed_kappa = float(metrics["channel.uses_per_pixel"])
        if abs(observed_kappa - kappa) > 1e-9:
            raise ValueError("%s has the wrong bandwidth ratio" % recipe["id"])
        records.append(
            {
                "sweep": sweep,
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _number(snr),
                "symbol_channels": channels,
                "uses_per_pixel": _number(observed_kappa),
                "channel_seed": seed,
                "psnr_db": _number(metrics["quality.psnr_db"]),
                "ms_ssim": _number(metrics["quality.ms_ssim"]),
                "mse": _number(metrics["quality.mse"]),
                "outage_rate": (
                    _number(metrics["channel.outage_rate"])
                    if method == "digital"
                    else ""
                ),
                "gain_power_min": _number(
                    metrics["channel.gain_power.minimum"]
                ),
                "gain_power_max": _number(
                    metrics["channel.gain_power.maximum"]
                ),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe["semantic_recipe_sha256"],
            }
        )
    expected_cells = {
        ("snr_sweep", method, snr, 32, seed)
        for method in METHOD_STYLES
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    } | {
        ("rate_sweep", method, RATE_SNR, channels, seed)
        for method in METHOD_STYLES
        for channels in EXPECTED_CHANNELS
        for seed in EXPECTED_SEEDS
    }
    actual_cells = {
        (
            str(row["sweep"]),
            str(row["method_id"]),
            float(row["snr_db"]),
            int(row["symbol_channels"]),
            int(row["channel_seed"]),
        )
        for row in records
    }
    if actual_cells != expected_cells:
        raise ValueError("benchmark cells do not match the frozen protocol")
    return sorted(
        records,
        key=lambda row: (
            str(row["sweep"]),
            float(row["snr_db"]),
            int(row["symbol_channels"]),
            int(row["channel_seed"]),
            str(row["method_id"]),
        ),
    )


def _pairing_rows(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _group(records)
    rows = []
    for key in sorted(grouped):
        cell = grouped[key]
        digital = _single(cell, "digital")
        learned = _single(cell, "learned")
        min_delta = abs(
            float(digital["gain_power_min"])
            - float(learned["gain_power_min"])
        )
        max_delta = abs(
            float(digital["gain_power_max"])
            - float(learned["gain_power_max"])
        )
        if max(min_delta, max_delta) > 1e-5:
            raise ValueError("paired fading gains do not match: %r" % (key,))
        rows.append(
            {
                "Sweep": key[0],
                "SNR (dB)": _number(key[1]),
                "κ": _number(key[2]),
                "Seed": key[3],
                "Minimum gain-power delta": _number(min_delta),
                "Maximum gain-power delta": _number(max_delta),
                "Matched": "yes",
            }
        )
    return rows


def _snr_summary_rows(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for snr in EXPECTED_SNRS:
        digital = _cell_samples(
            records, "snr_sweep", "digital", snr=snr, kappa=0.5
        )
        learned = _cell_samples(
            records, "snr_sweep", "learned", snr=snr, kappa=0.5
        )
        rows.append(
            {
                "Average SNR (dB)": _number(snr),
                "JPEG PSNR (dB)": _mean(digital, "psnr_db"),
                "DeepJSCC PSNR (dB)": _mean(learned, "psnr_db"),
                "DeepJSCC PSNR advantage (dB)": _number(
                    statistics.mean(
                        float(item["psnr_db"]) for item in learned
                    )
                    - statistics.mean(
                        float(item["psnr_db"]) for item in digital
                    )
                ),
                "JPEG MS-SSIM": _mean(digital, "ms_ssim"),
                "DeepJSCC MS-SSIM": _mean(learned, "ms_ssim"),
                "JPEG outage rate": _mean(digital, "outage_rate"),
            }
        )
    return rows


def _rate_summary_rows(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for kappa in EXPECTED_KAPPAS:
        digital = _cell_samples(
            records, "rate_sweep", "digital", snr=RATE_SNR, kappa=kappa
        )
        learned = _cell_samples(
            records, "rate_sweep", "learned", snr=RATE_SNR, kappa=kappa
        )
        rows.append(
            {
                "κ (complex uses/pixel)": _number(kappa),
                "JPEG PSNR (dB)": _mean(digital, "psnr_db"),
                "DeepJSCC PSNR (dB)": _mean(learned, "psnr_db"),
                "JPEG MS-SSIM": _mean(digital, "ms_ssim"),
                "DeepJSCC MS-SSIM": _mean(learned, "ms_ssim"),
                "JPEG outage rate": _mean(digital, "outage_rate"),
            }
        )
    return rows


def _build_charts(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "deepjscc-slow-psnr-snr": _method_chart(
            records,
            sweep="snr_sweep",
            metric="psnr_db",
            x_values=EXPECTED_SNRS,
            x_key="snr_db",
            fixed_snr=None,
            fixed_kappa=0.5,
            title="Reconstruction quality under slow fading",
            description=(
                "Both methods use κ=0.5 and paired per-image Rayleigh gains. "
                "Bands are two-sided Student-t 95% intervals over three "
                "held-out channel seeds."
            ),
            x_label="Average SNR (dB)",
            y_label="PSNR (dB)",
        ),
        "deepjscc-slow-ms-ssim-snr": _method_chart(
            records,
            sweep="snr_sweep",
            metric="ms_ssim",
            x_values=EXPECTED_SNRS,
            x_key="snr_db",
            fixed_snr=None,
            fixed_kappa=0.5,
            title="Perceptual quality under slow fading",
            description=(
                "MS-SSIM at κ=0.5 using the same held-out images and paired "
                "fades as the PSNR comparison."
            ),
            x_label="Average SNR (dB)",
            y_label="MS-SSIM",
            upper_bound=1.0,
        ),
        "deepjscc-slow-rate-psnr": _method_chart(
            records,
            sweep="rate_sweep",
            metric="psnr_db",
            x_values=EXPECTED_KAPPAS,
            x_key="uses_per_pixel",
            fixed_snr=RATE_SNR,
            fixed_kappa=None,
            title="Rate–distortion slice at 10 dB",
            description=(
                "The three exported rates come from one nested checkpoint. "
                "Each κ cell averages three independently seeded fading "
                "replicates; bands expose the resulting outage variance."
            ),
            x_label="κ (complex channel uses/source pixel)",
            y_label="PSNR (dB)",
        ),
    }


def _method_chart(
    records: list[Mapping[str, Any]],
    *,
    sweep: str,
    metric: str,
    x_values: tuple[float, ...],
    x_key: str,
    fixed_snr: float | None,
    fixed_kappa: float | None,
    title: str,
    description: str,
    x_label: str,
    y_label: str,
    upper_bound: float | None = None,
) -> dict[str, Any]:
    series = []
    for method, style in METHOD_STYLES.items():
        values = []
        ranges = []
        for x in x_values:
            snr = x if x_key == "snr_db" else fixed_snr
            kappa = x if x_key == "uses_per_pixel" else fixed_kappa
            samples = _cell_samples(
                records, sweep, method, snr=snr, kappa=kappa
            )
            numbers = [float(row[metric]) for row in samples]
            mean = statistics.mean(numbers)
            low, high = _confidence_interval(numbers)
            values.append([x, mean])
            ranges.append(
                [
                    x,
                    max(0.0, low),
                    min(upper_bound, high)
                    if upper_bound is not None
                    else high,
                ]
            )
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
        "xLabel": x_label,
        "yLabel": y_label,
        "allowLog": False,
        "yIncludeZero": False,
        "series": series,
    }


def _cell_samples(
    records: list[Mapping[str, Any]],
    sweep: str,
    method: str,
    *,
    snr: float | None,
    kappa: float | None,
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in records
        if row["sweep"] == sweep
        and row["method_id"] == method
        and (snr is None or float(row["snr_db"]) == float(snr))
        and (
            kappa is None
            or float(row["uses_per_pixel"]) == float(kappa)
        )
    ]
    if len(rows) != len(EXPECTED_SEEDS):
        raise ValueError(
            "expected three samples for %s/%s/%s/%s"
            % (sweep, method, snr, kappa)
        )
    return rows


def _group(
    records: list[Mapping[str, Any]],
) -> dict[tuple[str, float, float, int], list[Mapping[str, Any]]]:
    grouped: dict[
        tuple[str, float, float, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in records:
        grouped[
            (
                str(row["sweep"]),
                float(row["snr_db"]),
                float(row["uses_per_pixel"]),
                int(row["channel_seed"]),
            )
        ].append(row)
    return grouped


def _single(
    rows: list[Mapping[str, Any]], method: str
) -> Mapping[str, Any]:
    selected = [row for row in rows if row["method_id"] == method]
    if len(selected) != 1:
        raise ValueError("paired cell does not contain exactly one %s run" % method)
    return selected[0]


def _mean(rows: list[Mapping[str, Any]], metric: str) -> str:
    return _number(statistics.mean(float(row[metric]) for row in rows))


def _confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    margin = T95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def _decode_number(token: str) -> float:
    negative = token.startswith("m")
    body = token[1:] if negative else token
    value = float(body.replace("p", "."))
    return -value if negative else value


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
