from __future__ import annotations

"""Preserve and render the completed QPSK I/Q-calibration benchmark result."""

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noema_lab.core.structured_input import decode_strict_json_object


ROOT = Path(__file__).resolve().parents[1]
RESULT_ID = (
    "20260726T155403Z_"
    "neural_receiver_ai_phy.learned_qpsk_iq_calibration_v1"
)
BENCHMARK_ID = "neural_receiver_ai_phy.learned_qpsk_iq_calibration_v1"
DEFAULT_RESULT_DIR = ROOT / ".noema" / "benchmarks" / RESULT_ID
DEFAULT_DATA_DIR = ROOT / "docs" / "demo" / "data" / "qpsk_iq_calibration"
DEFAULT_ASSET_DIR = ROOT / "docs" / "demo" / "assets" / "qpsk_iq_calibration"
DEFAULT_SOURCE = DEFAULT_DATA_DIR / "benchmark_projection.csv"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "benchmark_manifest.json"

METHOD_ORDER = (
    "uncompensated_qpsk",
    "calibrated_iq_oracle",
    "learned_receiver",
)
METHOD_LABELS = {
    "uncompensated_qpsk": "Uncompensated QPSK",
    "calibrated_iq_oracle": "Calibrated I/Q oracle",
    "learned_receiver": "Learned I/Q receiver",
}
METHOD_STYLES = {
    "uncompensated_qpsk": {
        "color": "#6b7280",
        "marker": "s",
        "linestyle": "--",
    },
    "calibrated_iq_oracle": {
        "color": "#059669",
        "marker": "^",
        "linestyle": "-",
    },
    "learned_receiver": {
        "color": "#dc2626",
        "marker": "o",
        "linestyle": "-",
    },
}
EXPECTED_SNRS = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
EXPECTED_SEEDS = (71001, 72001, 73001)
BITS_PER_RUN = 1_048_576

SOURCE_FIELDS = (
    "result_id",
    "recipe_id",
    "recipe_label",
    "role",
    "run_id",
    "method",
    "paired_seed",
    "snr_db",
    "bit_errors",
    "compared_bits",
    "ber",
    "block_errors",
    "compared_blocks",
    "bler",
    "recipe_sha256",
    "semantic_recipe_sha256",
)
SUMMARY_FIELDS = (
    "SNR (dB)",
    "Uncompensated mean BER",
    "Calibrated-oracle mean BER",
    "Learned mean BER",
    "Learned min BER",
    "Learned max BER",
    "Learned reduction vs uncompensated",
    "Learned relative gap to oracle",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the QPSK I/Q-calibration tutorial result assets."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--import-result",
        type=Path,
        metavar="RESULT_DIR",
        help="project the completed benchmark result into the committed snapshot",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when committed derived assets differ from fresh generation",
    )
    args = parser.parse_args(argv)
    if args.check and args.import_result is not None:
        parser.error("--check and --import-result cannot be combined")

    if args.import_result is not None:
        import_result(args.import_result, args.source, args.manifest)

    records = load_records(args.source)
    validate_manifest(args.manifest, args.source, records)
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = generate(
                records,
                root / "data",
                root / "assets",
                manifest_path=args.manifest,
            )
            expected = output_paths(args.data_dir, args.asset_dir)
            stale = [
                name
                for name, path in generated.items()
                if not expected[name].is_file()
                or expected[name].read_bytes() != path.read_bytes()
            ]
        if stale:
            parser.error(
                "derived assets are stale: %s" % ", ".join(sorted(stale))
            )
        return 0

    generate(
        records,
        args.data_dir,
        args.asset_dir,
        manifest_path=args.manifest,
    )
    return 0


def import_result(result_dir: Path, source: Path, manifest_path: Path) -> None:
    result_dir = result_dir.resolve()
    result_path = result_dir / "result.json"
    benchmark_path = result_dir / "benchmark.json"
    metrics_path = result_dir / "metrics.csv"
    result = _load_json(result_path, "benchmark result")
    if result_dir.name != RESULT_ID:
        raise ValueError("unexpected result id: %s" % result_dir.name)
    if result.get("status") != "completed":
        raise ValueError("benchmark result is not completed")
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("id") != BENCHMARK_ID:
        raise ValueError("unexpected benchmark identity")
    raw_recipes = result.get("recipes")
    if not isinstance(raw_recipes, list):
        raise ValueError("benchmark result recipes must be a list")

    records: list[dict[str, Any]] = []
    for raw in raw_recipes:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark recipe result must be an object")
        recipe_id = str(raw.get("id") or "")
        method = _method_from_recipe_id(recipe_id)
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("%s has no metric object" % recipe_id)
        record = {
            "result_id": RESULT_ID,
            "recipe_id": recipe_id,
            "recipe_label": str(raw.get("label") or ""),
            "role": str(raw.get("role") or ""),
            "run_id": str(raw.get("run_id") or ""),
            "method": method,
            "paired_seed": int(raw.get("pairing_seed")),
            "snr_db": float(metrics["channel.snr_db"]),
            "bit_errors": int(metrics["channel.coded.error_count"]),
            "compared_bits": int(metrics["channel.coded.compare_bit_count"]),
            "ber": float(metrics["channel.coded.ber"]),
            "block_errors": int(metrics["channel.coded.block_error_count"]),
            "compared_blocks": int(metrics["channel.coded.block_count"]),
            "bler": float(metrics["channel.coded.bler"]),
            "recipe_sha256": str(raw.get("recipe_sha256") or ""),
            "semantic_recipe_sha256": str(
                raw.get("semantic_recipe_sha256") or ""
            ),
        }
        records.append(record)
    records.sort(
        key=lambda row: (
            float(row["snr_db"]),
            int(row["paired_seed"]),
            METHOD_ORDER.index(str(row["method"])),
        )
    )
    _validate_records(records)

    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SOURCE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: _number(record[field])
                    if field in {"snr_db", "ber", "bler"}
                    else record[field]
                    for field in SOURCE_FIELDS
                }
            )

    metadata = dict(benchmark.get("metadata") or {})
    manifest = {
        "schema_version": 1,
        "kind": "noema.completed_benchmark_projection",
        "evidence_level": "completed_experimental_benchmark",
        "result": {
            "id": RESULT_ID,
            "benchmark_id": BENCHMARK_ID,
            "benchmark_version": str(benchmark.get("version") or ""),
            "status": "completed",
            "created_at_utc": result.get("created_at_utc"),
            "completed_at_utc": result.get("completed_at_utc"),
            "result_json_sha256": _sha256(result_path),
            "benchmark_json_sha256": _sha256(benchmark_path),
            "metrics_csv_sha256": _sha256(metrics_path),
            "benchmark_semantic_sha256": benchmark.get("sha256"),
        },
        "projection": {
            "path": source.name,
            "sha256": _sha256(source),
            "row_count": len(records),
            "fields": list(SOURCE_FIELDS),
        },
        "scope": {
            "methods": list(METHOD_ORDER),
            "snr_db": list(EXPECTED_SNRS),
            "paired_seeds": list(EXPECTED_SEEDS),
            "bits_per_run": BITS_PER_RUN,
            "runs_per_method_and_snr": len(EXPECTED_SEEDS),
            "trained_artifact_runtime_identity_sha256": metadata.get(
                "trained_artifact_runtime_identity_sha256"
            ),
        },
        "verification": {
            "status": "warning",
            "integrity_errors": [],
            "warnings": [
                (
                    "The evidence is internally valid, but this experimental "
                    "suite is not yet designated as a publication-ready "
                    "canonical benchmark."
                )
            ],
        },
        "limitations": [
            (
                "The committed CSV is a deterministic projection of the local "
                "completed result, not a replacement for its full run-evidence "
                "snapshot."
            ),
            (
                "Three paired held-out seeds support a demonstration comparison "
                "but are not a universal publication-strength sample size."
            ),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_records(source: Path) -> list[dict[str, Any]]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ValueError(
                "unexpected benchmark projection columns: %r"
                % (reader.fieldnames,)
            )
        rows = list(reader)
    records = [
        {
            **row,
            "paired_seed": int(row["paired_seed"]),
            "snr_db": float(row["snr_db"]),
            "bit_errors": int(row["bit_errors"]),
            "compared_bits": int(row["compared_bits"]),
            "ber": float(row["ber"]),
            "block_errors": int(row["block_errors"]),
            "compared_blocks": int(row["compared_blocks"]),
            "bler": float(row["bler"]),
        }
        for row in rows
    ]
    _validate_records(records)
    return records


def validate_manifest(
    manifest_path: Path,
    source: Path,
    records: list[dict[str, Any]],
) -> None:
    manifest = _load_json(manifest_path, "benchmark projection manifest")
    projection = manifest.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("manifest projection must be an object")
    if projection.get("sha256") != _sha256(source):
        raise ValueError("manifest does not bind the benchmark projection")
    if projection.get("row_count") != len(records):
        raise ValueError("manifest projection row count is stale")
    result = manifest.get("result")
    if not isinstance(result, Mapping) or result.get("id") != RESULT_ID:
        raise ValueError("manifest result identity is wrong")
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("manifest scope must be an object")
    if scope.get("bits_per_run") != BITS_PER_RUN:
        raise ValueError("manifest bits-per-run contract is wrong")


def generate(
    records: list[dict[str, Any]],
    data_dir: Path,
    asset_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Path]:
    paths = output_paths(data_dir, asset_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(manifest_path, "benchmark projection manifest")
    _write_ber_table(paths["ber_table"], records)
    _write_ber_plot(paths["ber_plot"], records, manifest)
    return paths


def output_paths(data_dir: Path, asset_dir: Path) -> dict[str, Path]:
    return {
        "ber_table": data_dir / "ber_table.csv",
        "ber_plot": asset_dir / "ber_vs_snr.svg",
    }


def _write_ber_table(path: Path, records: list[dict[str, Any]]) -> None:
    grouped = _grouped_ber(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for snr_db in EXPECTED_SNRS:
            methods = grouped[snr_db]
            uncompensated = statistics.mean(methods["uncompensated_qpsk"])
            oracle = statistics.mean(methods["calibrated_iq_oracle"])
            learned_values = methods["learned_receiver"]
            learned = statistics.mean(learned_values)
            writer.writerow(
                {
                    "SNR (dB)": _number(snr_db),
                    "Uncompensated mean BER": "%.8g" % uncompensated,
                    "Calibrated-oracle mean BER": "%.8g" % oracle,
                    "Learned mean BER": "%.8g" % learned,
                    "Learned min BER": "%.8g" % min(learned_values),
                    "Learned max BER": "%.8g" % max(learned_values),
                    "Learned reduction vs uncompensated": "%.2f%%"
                    % (100.0 * (uncompensated - learned) / uncompensated),
                    "Learned relative gap to oracle": "%.2f%%"
                    % (100.0 * (learned - oracle) / oracle),
                }
            )


def _write_ber_plot(
    path: Path,
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    grouped = _grouped_ber(records)
    matplotlib.rcParams["svg.hashsalt"] = (
        "noema-qpsk-iq-calibration-completed-benchmark-v1"
    )
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.edgecolor": "#6b7280",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")
        for method in METHOD_ORDER:
            means = [
                statistics.mean(grouped[snr_db][method])
                for snr_db in EXPECTED_SNRS
            ]
            lows = [min(grouped[snr_db][method]) for snr_db in EXPECTED_SNRS]
            highs = [max(grouped[snr_db][method]) for snr_db in EXPECTED_SNRS]
            style = METHOD_STYLES[method]
            axis.fill_between(
                EXPECTED_SNRS,
                lows,
                highs,
                color=style["color"],
                alpha=0.11,
                linewidth=0,
            )
            axis.plot(
                EXPECTED_SNRS,
                means,
                label=METHOD_LABELS[method],
                linewidth=2.0,
                markersize=6.0,
                markeredgewidth=1.2,
                **style,
            )
        axis.set_yscale("log")
        axis.set_xticks(EXPECTED_SNRS)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Pre-decoder bit error rate")
        axis.grid(True, which="both", color="#d1d5db", linewidth=0.7, alpha=0.72)
        axis.set_axisbelow(True)
        axis.margins(x=0.04)
        axis.legend(frameon=False, ncols=1, loc="lower left")
        result_id = str(dict(manifest.get("result") or {}).get("id") or RESULT_ID)
        figure.savefig(
            path,
            format="svg",
            metadata={
                "Date": None,
                "Creator": (
                    "Noema QPSK I/Q-calibration completed-benchmark asset "
                    "generator; result %s" % result_id
                ),
            },
            facecolor="white",
        )
        plt.close(figure)


def _grouped_ber(
    records: Iterable[Mapping[str, Any]],
) -> dict[float, dict[str, list[float]]]:
    grouped = {
        snr_db: {method: [] for method in METHOD_ORDER}
        for snr_db in EXPECTED_SNRS
    }
    for record in records:
        grouped[float(record["snr_db"])][str(record["method"])].append(
            float(record["ber"])
        )
    return grouped


def _validate_records(records: list[Mapping[str, Any]]) -> None:
    expected_count = len(METHOD_ORDER) * len(EXPECTED_SNRS) * len(EXPECTED_SEEDS)
    if len(records) != expected_count:
        raise ValueError("expected %d benchmark records" % expected_count)
    coordinates = {
        (
            str(record["method"]),
            float(record["snr_db"]),
            int(record["paired_seed"]),
        )
        for record in records
    }
    expected = {
        (method, snr_db, seed)
        for method in METHOD_ORDER
        for snr_db in EXPECTED_SNRS
        for seed in EXPECTED_SEEDS
    }
    if coordinates != expected:
        raise ValueError("benchmark projection coordinates are incomplete")
    if len({str(record["run_id"]) for record in records}) != expected_count:
        raise ValueError("benchmark projection run IDs are not unique")
    for record in records:
        if record["result_id"] != RESULT_ID:
            raise ValueError("benchmark projection mixes result IDs")
        if int(record["compared_bits"]) != BITS_PER_RUN:
            raise ValueError("benchmark projection has the wrong BER denominator")
        expected_ber = int(record["bit_errors"]) / BITS_PER_RUN
        if abs(float(record["ber"]) - expected_ber) > 1e-15:
            raise ValueError("benchmark projection BER does not match its counts")
        for field in ("recipe_sha256", "semantic_recipe_sha256"):
            digest = str(record[field])
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("benchmark projection has an invalid %s" % field)


def _method_from_recipe_id(recipe_id: str) -> str:
    for method in METHOD_ORDER:
        if recipe_id.startswith(method + "_snr"):
            return method
    raise ValueError("cannot infer method from recipe id %s" % recipe_id)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("%s does not exist: %s" % (label, path))
    return decode_strict_json_object(
        path.read_text(encoding="utf-8"),
        label=label,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric result")
    if isinstance(value, int):
        return str(value)
    number = float(value)
    return "%.17g" % number


if __name__ == "__main__":
    raise SystemExit(main())
