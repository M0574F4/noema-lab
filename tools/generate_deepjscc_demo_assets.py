"""Project a completed digital-versus-DeepJSCC benchmark into docs assets."""

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
    "20260727T235925Z_semantic_comm.digital_vs_deepjscc_post_training_v3"
)
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / DEFAULT_RESULT_ID
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "digital_vs_deepjscc"
DEFAULT_CHART_JS = ROOT / "docs" / "_static" / "noema-deepjscc-chart-data.js"

EXPECTED_BENCHMARK_ID = "semantic_comm.digital_vs_deepjscc_post_training_v3"
EXPECTED_VERSION = "3.0.0"
EXPECTED_METHODS = ("jpeg_capacity", "learned_deepjscc")
EXPECTED_SNRS = (
    -6.0,
    -4.0,
    -2.0,
    0.0,
    4.0,
    8.0,
    12.0,
    16.0,
)
EXPECTED_SEEDS = (71001, 72001, 73001)
RESOURCE_LIMIT = 0.5
T95_DF2 = 4.302652729911275

PSNR = "quality.psnr_db"
MSE = "quality.mse"
MS_SSIM = "quality.ms_ssim"
USES_PER_PIXEL = "channel.uses_per_pixel"
CAPACITY_BPP = "channel.awgn.capacity_bpp"
NATIVE_CODEC_BPP = "rate.native_codec_bpp"
JPEG_QUALITY_MEAN = "codec.jpeg.selected_quality_mean"
JPEG_QUALITY_MIN = "codec.jpeg.selected_quality_min"
JPEG_QUALITY_MAX = "codec.jpeg.selected_quality_max"
RESOURCE_OBSERVED = "benchmark.resource_budget.observed"
RESOURCE_MAXIMUM = "benchmark.resource_budget.maximum"
RESOURCE_ADMITTED = "benchmark.resource_budget.admitted"

METHOD_STYLES = {
    "jpeg_capacity": {
        "label": "Capacity-matched JPEG",
        "color": "#2563eb",
        "dash": [7, 4],
        "marker": "square",
    },
    "learned_deepjscc": {
        "label": "Learned DeepJSCC",
        "color": "#16a34a",
        "dash": [],
        "marker": "circle",
    },
}

RECIPE_ID = re.compile(
    r"^(?P<method>jpeg_capacity|learned_deepjscc)"
    r"_snr(?P<snr>m?[0-9]+)(?:_seed(?P<seed>[0-9]+))?$"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs assets from a completed DeepJSCC benchmark."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chart-js", type=Path, default=DEFAULT_CHART_JS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the checked-in snapshot without reading the local result bundle.",
    )
    args = parser.parse_args()
    if args.check:
        manifest_path = args.data_dir.expanduser().resolve() / "snapshot_manifest.json"
        verify_snapshot_assets(manifest_path)
        print("DeepJSCC documentation snapshot is valid: %s" % manifest_path)
        return 0
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
    quality_rows = _quality_display_rows(summaries)
    operating_rows = _jpeg_operating_point_rows(summaries)
    resource_rows = _resource_rows(records)
    charts = {
        "deepjscc-psnr": _method_chart(
            records,
            metric="psnr_db",
            title="Reconstruction quality at a fixed bandwidth ratio",
            description=(
                "Capacity-matched JPEG is deterministic. DeepJSCC uses one frozen "
                "model; whiskers show a two-sided 95% Student's t interval across "
                "three channel-noise seeds for the same four crops."
            ),
            accessible_summary=(
                "DeepJSCC has higher mean PSNR at minus 6 and minus 4 dB. "
                "Capacity-matched JPEG is higher from minus 2 through 16 dB."
            ),
            y_label="PSNR (dB)",
            include_zero=False,
        ),
        "deepjscc-ms-ssim": _method_chart(
            records,
            metric="ms_ssim",
            title="Perceptual reconstruction quality under AWGN",
            description=(
                "Capacity-matched JPEG is deterministic. DeepJSCC uses one frozen "
                "model; whiskers show a two-sided 95% Student's t interval across "
                "three channel-noise seeds for the same four crops."
            ),
            accessible_summary=(
                "MS-SSIM follows the PSNR ordering: DeepJSCC leads at minus 6 "
                "and minus 4 dB, then capacity-matched JPEG leads."
            ),
            y_label="MS-SSIM",
            include_zero=False,
            upper_bound=1.0,
        ),
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    chart_js.parent.mkdir(parents=True, exist_ok=True)
    projection_path = data_dir / "benchmark_projection.csv"
    summary_path = data_dir / "summary_table.csv"
    quality_path = data_dir / "quality_summary_table.csv"
    operating_path = data_dir / "jpeg_operating_points_table.csv"
    resource_path = data_dir / "resource_audit.csv"
    chart_data_path = data_dir / "chart_data.json"
    manifest_path = data_dir / "snapshot_manifest.json"
    _write_csv(projection_path, records)
    _write_csv(summary_path, summaries)
    _write_csv(quality_path, quality_rows)
    _write_csv(operating_path, operating_rows)
    _write_csv(resource_path, resource_rows)
    _write_json(chart_data_path, charts)
    _write_chart_js(chart_js, charts)

    manifest = {
        "kind": "noema.docs_demo_snapshot",
        "schema_version": 1,
        "slug": "digital-versus-deepjscc",
        "source": {
            "result_id": result_dir.name,
            "benchmark_id": EXPECTED_BENCHMARK_ID,
            "benchmark_version": EXPECTED_VERSION,
            "status": "completed",
            "benchmark_tier": "experimental",
            "availability": (
                "The raw local result bundle is not distributed with the documentation; "
                "hashes are retained for identity checks."
            ),
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
            "statistical_unit": (
                "deterministic capacity-matched JPEG evaluation or DeepJSCC "
                "channel-noise seed, conditional on one frozen model and four crops"
            ),
            "learned_channel_seeds": list(EXPECTED_SEEDS),
            "sample_count_per_method_and_snr": {
                "jpeg_capacity": 1,
                "learned_deepjscc": len(EXPECTED_SEEDS),
            },
            "held_out_images": ["kodim21", "kodim22", "kodim23", "kodim24"],
            "resource_limit_complex_channel_uses_per_pixel": RESOURCE_LIMIT,
            "interval": (
                "two-sided 95% Student's t interval over learned channel-noise "
                "seeds; deterministic reference has no interval"
            ),
            "degrees_of_freedom": 2,
            "critical_value": T95_DF2,
            "warning": (
                "Three learned channel-noise seeds support an experimental tutorial result, "
                "not a paper-grade population claim."
            ),
        },
        "projection": {
            "rows": len(records),
            "benchmark_projection.csv": _file_evidence(projection_path),
            "summary_table.csv": _file_evidence(summary_path),
            "quality_summary_table.csv": _file_evidence(quality_path),
            "jpeg_operating_points_table.csv": _file_evidence(operating_path),
            "resource_audit.csv": _file_evidence(resource_path),
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
        "summary": summary_path,
        "quality_summary": quality_path,
        "jpeg_operating_points": operating_path,
        "resource_audit": resource_path,
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
    if demo.get("slug") != "digital-versus-deepjscc":
        raise ValueError("unexpected demo slug")
    recipes = result.get("recipes") or []
    expected_count = len(EXPECTED_SNRS) * (1 + len(EXPECTED_SEEDS))
    if len(recipes) != expected_count:
        raise ValueError(
            "benchmark does not contain the expected %d runs" % expected_count
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
        raw_seed = match.group("seed")
        if method == "jpeg_capacity" and raw_seed is not None:
            raise ValueError("deterministic JPEG reference must not declare a seed")
        if method == "learned_deepjscc" and raw_seed is None:
            raise ValueError("learned DeepJSCC recipe must declare a seed")
        metrics = recipe["metrics"]
        admitted = float(metrics[RESOURCE_ADMITTED])
        maximum = float(metrics[RESOURCE_MAXIMUM])
        observed = float(metrics[RESOURCE_OBSERVED])
        if admitted != 1.0 or observed > maximum + 1e-9:
            raise ValueError("resource guard rejected recipe %s" % recipe["id"])
        ms_ssim = metrics.get(MS_SSIM)
        if ms_ssim is None:
            raise ValueError("recipe omits MS-SSIM evidence: %s" % recipe["id"])
        records.append(
            {
                "method_id": method,
                "method_label": METHOD_STYLES[method]["label"],
                "role": recipe["role"],
                "snr_db": _number(snr),
                "channel_seed": (
                    "" if raw_seed is None else int(raw_seed)
                ),
                "psnr_db": _number(metrics[PSNR]),
                "mse": _number(metrics[MSE]),
                "ms_ssim": _number(ms_ssim),
                "uses_per_pixel": _number(metrics[USES_PER_PIXEL]),
                "resource_guard_observed": _number(observed),
                "resource_guard_maximum": _number(maximum),
                "resource_guard_admitted": int(admitted),
                "capacity_bpp": (
                    _number(metrics[CAPACITY_BPP])
                    if method == "jpeg_capacity"
                    else ""
                ),
                "native_codec_bpp": (
                    _number(metrics[NATIVE_CODEC_BPP])
                    if method == "jpeg_capacity"
                    else ""
                ),
                "jpeg_quality_mean": (
                    _number(metrics[JPEG_QUALITY_MEAN])
                    if method == "jpeg_capacity"
                    else ""
                ),
                "jpeg_quality_min": (
                    _number(metrics[JPEG_QUALITY_MIN])
                    if method == "jpeg_capacity"
                    else ""
                ),
                "jpeg_quality_max": (
                    _number(metrics[JPEG_QUALITY_MAX])
                    if method == "jpeg_capacity"
                    else ""
                ),
                "recipe_id": recipe["id"],
                "run_id": recipe["run_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "semantic_recipe_sha256": recipe["semantic_recipe_sha256"],
            }
        )
    expected = {
        ("jpeg_capacity", snr, "")
        for snr in EXPECTED_SNRS
    } | {
        ("learned_deepjscc", snr, seed)
        for snr in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (
            str(row["method_id"]),
            float(row["snr_db"]),
            row["channel_seed"],
        )
        for row in records
    }
    if actual != expected:
        raise ValueError("benchmark cells do not match the frozen protocol")
    return sorted(
        records,
        key=lambda row: (
            float(row["snr_db"]),
            (
                -1
                if row["channel_seed"] == ""
                else int(row["channel_seed"])
            ),
            EXPECTED_METHODS.index(str(row["method_id"])),
        ),
    )


def _summary_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = _group(records)
    rows = []
    for snr in EXPECTED_SNRS:
        capacity = cells[("jpeg_capacity", snr)]
        learned = cells[("learned_deepjscc", snr)]
        baseline = capacity[0]
        rows.append(
            {
                "SNR (dB)": _number(snr),
                "Capacity-matched JPEG PSNR (dB)": _mean(capacity, "psnr_db"),
                "DeepJSCC PSNR (dB)": _mean(learned, "psnr_db"),
                "Capacity-matched JPEG MS-SSIM": _mean(capacity, "ms_ssim"),
                "DeepJSCC MS-SSIM": _mean(learned, "ms_ssim"),
                "JPEG quality mean": baseline["jpeg_quality_mean"],
                "JPEG quality range": "%s–%s"
                % (
                    baseline["jpeg_quality_min"],
                    baseline["jpeg_quality_max"],
                ),
                "JPEG native bpp": baseline["native_codec_bpp"],
                "Ideal capacity bpp": baseline["capacity_bpp"],
            }
        )
    return rows


def _quality_display_rows(
    summaries: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "SNR (dB)": row["SNR (dB)"],
            "JPEG PSNR (dB)": _fixed(row["Capacity-matched JPEG PSNR (dB)"], 2),
            "DeepJSCC PSNR (dB)": _fixed(row["DeepJSCC PSNR (dB)"], 2),
            "JPEG MS-SSIM": _fixed(row["Capacity-matched JPEG MS-SSIM"], 3),
            "DeepJSCC MS-SSIM": _fixed(row["DeepJSCC MS-SSIM"], 3),
        }
        for row in summaries
    ]


def _jpeg_operating_point_rows(
    summaries: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "SNR (dB)": row["SNR (dB)"],
            "JPEG quality mean": _fixed(row["JPEG quality mean"], 2),
            "JPEG quality range": row["JPEG quality range"],
            "JPEG native bpp": _fixed(row["JPEG native bpp"], 3),
            "Ideal capacity bpp": _fixed(row["Ideal capacity bpp"], 3),
        }
        for row in summaries
    ]


def _resource_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in EXPECTED_METHODS:
        method_rows = [row for row in records if row["method_id"] == method]
        rows.append(
            {
                "Method": METHOD_STYLES[method]["label"],
                "Budgeted channel uses/source pixel": _number(
                    statistics.mean(
                        float(row["uses_per_pixel"]) for row in method_rows
                    )
                ),
                "Maximum per-image channel uses/source pixel": _number(
                    max(
                        float(row["resource_guard_observed"])
                        for row in method_rows
                    )
                ),
                "Budget limit": _number(RESOURCE_LIMIT),
                "Admitted": "yes",
            }
        )
    return rows


def _mean(rows: list[Mapping[str, Any]], metric: str) -> str:
    return _number(statistics.mean(float(row[metric]) for row in rows))


def _method_chart(
    records: list[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    description: str,
    accessible_summary: str,
    y_label: str,
    include_zero: bool,
    upper_bound: float | None = None,
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
            ranges.append(
                [
                    snr,
                    max(0.0, low),
                    min(upper_bound, high) if upper_bound is not None else high,
                ]
            )
        style = METHOD_STYLES[method]
        series.append(
            {
                "id": method,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "marker": style["marker"],
                "sampleCount": len(samples),
                "values": values,
                "range": ranges,
            }
        )
    return {
        "type": "line",
        "title": title,
        "description": description,
        "accessibleSummary": accessible_summary,
        "xLabel": "SNR (dB)",
        "yLabel": y_label,
        "allowLog": False,
        "yIncludeZero": include_zero,
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
    if len(values) < 2:
        return mean, mean
    margin = T95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def verify_snapshot_assets(manifest_path: Path) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    root = ROOT.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("snapshot manifest must remain inside the repository") from exc
    manifest = _load_json(manifest_path)
    if manifest.get("kind") != "noema.docs_demo_snapshot":
        raise ValueError("unexpected documentation snapshot kind")
    projection = manifest.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("snapshot projection must be an object")
    for name, evidence in projection.items():
        if not isinstance(evidence, Mapping) or "sha256" not in evidence:
            continue
        if name == "chart_javascript":
            relative = Path(str(evidence.get("path") or ""))
            path = (root / relative).resolve()
        else:
            path = (manifest_path.parent / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("snapshot asset escapes the repository: %s" % path) from exc
        if not path.is_file():
            raise ValueError("snapshot asset is missing: %s" % path)
        actual = _file_evidence(path)
        if actual != {
            "sha256": str(evidence["sha256"]),
            "size_bytes": int(evidence["size_bytes"]),
        }:
            raise ValueError("snapshot asset digest or size mismatch: %s" % path)
    publication = manifest.get("publication") or {}
    if publication.get("publication_ready") is not False:
        raise ValueError("experimental tutorial snapshot must fail closed for publication")
    if "representative_reconstructions" in projection:
        raise ValueError("rights-restricted reconstructions must not be public snapshot assets")


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


def _fixed(value: Any, digits: int) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite benchmark value")
    return f"{number:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
